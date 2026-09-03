#!/usr/bin/env bash
# Experiment: do the net's *learned priors* beat the scripted ones?
#
# The training loop plays with `--evaluator nn-value` — NN leaf values, but
# scripted priors (botbowl-mcts/src/dynamics.rs:258). So the policy head is
# trained every generation and never used to play, which is why val_policy
# keeps improving to epoch 9 with no strength consequence. `--evaluator nn`
# replaces priors *and* values with the net. This pits the same weights
# against themselves with only the prior source differing.
#
# Scheduling: the production loop must not slow down, and it has exactly one
# CPU-free window per generation — the ~28 min train phase, after generate
# has finished and before eval starts. The loop tears down its nn_server
# before training so the trainer gets the whole GPU, so this runs on tract
# (CPU, no GPU contention with the trainer) using the cores generation has
# just released.
#
# That budget only buys ~8 games at the production 1000 iters, so the screen
# runs at EXP_ITERS=200. Priors matter most when there is little search to
# correct them, so a null result here is strong evidence against NN priors
# at 1000, while a positive result needs confirming at full strength. Batches
# accumulate across generations under distinct seed bases.
#
#   nohup scripts/exp_priors.sh > /dev/null 2>&1 &
#   touch runs/exp-priors/STOP     # exits after the current batch

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4

RUN_DIR="${RUN_DIR:-$REPO/runs/loop14x7}"
OUT_DIR="${OUT_DIR:-$REPO/runs/exp-priors}"
UI="${CARGO_TARGET_DIR:-$REPO/target/14x7}/release/botbowl-ui"
STATUS="$RUN_DIR/status.md"
EXP_ITERS="${EXP_ITERS:-200}"
EXP_GAMES="${EXP_GAMES:-24}"          # sized to finish inside a ~28 min window
EXP_PARALLEL="${EXP_PARALLEL:-6}"     # trainer keeps 1-2 cores for its dataloader
SEED_BASE="${SEED_BASE:-77000000}"    # disjoint from the loop's 1e7.. corpus seeds
MAX_BATCHES="${MAX_BATCHES:-8}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/exp.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# Wait until the loop's status file shows `genNN train:` as its newest phase
# line, i.e. the window has just opened. Returns the generation tag.
wait_for_train_window() {
    local last=""
    while true; do
        [ -e "$OUT_DIR/STOP" ] && return 1
        last=$(grep -oE 'gen[0-9]+ (train|train done|eval|generate|prepare)' "$STATUS" 2>/dev/null | tail -1)
        case "$last" in
            *" train") echo "${last%% *}"; return 0 ;;
        esac
        sleep 20
    done
}

log "experiment start: nn (learned priors) vs nn-value (scripted priors), same weights, ${EXP_ITERS} iters, ${EXP_GAMES} games/batch"

B=0
while [ "$B" -lt "$MAX_BATCHES" ]; do
    GG=$(wait_for_train_window) || { log "STOP file — exiting"; break; }
    # The champion is whatever the loop promoted most recently; read it fresh
    # so later batches test the current net rather than a stale one.
    MODEL=$(cat "$RUN_DIR/champion.txt")
    [ -f "$MODEL" ] || { log "champion $MODEL missing — skipping batch"; sleep 300; continue; }
    SEED=$((SEED_BASE + B * 10000))
    OUT="$OUT_DIR/batch$(printf '%02d' "$B").json"
    if [ -e "$OUT" ]; then B=$((B + 1)); continue; fi

    log "batch $B: $GG window, $(basename "$MODEL"), seed $SEED, ${EXP_GAMES} games x${EXP_PARALLEL} on tract"
    START=$SECONDS
    # Candidate = learned priors, opponent = scripted priors, identical weights.
    # No --nn-server on purpose: the trainer owns the GPU during this window.
    if "$UI" eval --evaluator nn --model "$MODEL" \
            --mcts-iters "$EXP_ITERS" \
            --skip-lectures --skip-fixed-rungs \
            --vs-evaluator nn-value --vs-model "$MODEL" \
            --vs-games "$EXP_GAMES" --seed "$SEED" \
            --parallel-games "$EXP_PARALLEL" \
            --out "$OUT" > "$OUT_DIR/batch$(printf '%02d' "$B").log" 2>&1; then
        log "batch $B done ($(((SECONDS - START) / 60)) min): $("$REPO/train/.venv/bin/python" "$REPO/scripts/eval_summary.py" "$OUT" 2>/dev/null || echo 'see json')"
    else
        log "batch $B FAILED — see batch$(printf '%02d' "$B").log"
    fi
    B=$((B + 1))

    # Do not start another batch inside the same window; wait for the loop to
    # leave the train phase before looking for the next one.
    while grep -oE 'gen[0-9]+ (train|train done|eval|generate|prepare)' "$STATUS" 2>/dev/null \
            | tail -1 | grep -q ' train$'; do
        [ -e "$OUT_DIR/STOP" ] && break
        sleep 20
    done
done
log "experiment finished after $B batches"
