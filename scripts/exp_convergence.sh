#!/usr/bin/env bash
# Plan 028 Stage 0 + B1 — does the search converge, and does normalised Q fix it?
#
# Stage 0 (~1h): the repo owner's hygiene test. Take states from the same
# generator the corpus uses, run the search repeatedly at increasing budgets,
# and measure the total-variation distance between independent runs at each
# stage. If TV does not shrink as the budget grows, the search is not
# converging and everything downstream inherits that.
#
# Plan 025 ran this and found TV *rising* — 0.193 at 100 iterations to 0.383 at
# 16000, peak visit share 0.449 -> 0.742, top-1 agreement peaking at ~500 then
# falling. But that measurement is marked provisional in its own header: it ran
# on the retired gen01 champion under the pre-e107f06 buggy search and its raw
# data was deleted 2026-09-01. Since then plan 023 fixed the side-bias root
# cause, the loop switched to learned priors, and plan 027 found strength does
# improve 250->1000 where 025's label metrics said it should not. So one of
# 025's two headline findings has already been overturned by re-measurement;
# this re-checks the other.
#
# S0a is production (raw c=10). S0b is normalised Q on identical states and
# seeds — plan 026 measured it beating raw at a single budget (peak share 0.444
# vs 0.515, TV 0.168 vs 0.221, agreement 0.69 vs 0.60, paired +0.082 p=0.023)
# but nobody has looked at the *curve*. A real fix turns the TV line from
# rising to falling.
#
# B1 (~3h) then asks the only question that ultimately decides anything: does
# normalised Q win *games* against production? Run regardless of Stage 0's
# outcome — this week established repeatedly that label metrics and strength
# disagree, so the noise curve cannot stand in for a head-to-head.
#
#   nohup scripts/exp_convergence.sh > /dev/null 2>&1 &
#   touch runs/exp-conv/STOP

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4

RUN_DIR="$REPO/runs/loop14x7"
OUT="$REPO/runs/exp-conv"
UI="${CARGO_TARGET_DIR:-$REPO/target/14x7}/release/botbowl-ui"
PY="$REPO/train/.venv/bin/python"
STATES="${STATES:-50}"
REPEATS="${REPEATS:-3}"
BUDGETS="${BUDGETS:-100,200,500,1000,2000,4000,8000,16000}"
SEED="${SEED:-90000000}"
GAMES="${GAMES:-120}"
PARALLEL="${PARALLEL:-6}"

mkdir -p "$OUT"
LOG="$OUT/exp.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# Wait for the production loop to release the box (STOP is armed, so it exits
# after gen07's gate and before gen08 generate).
while pgrep -x -f "bash scripts/train_loop.sh" > /dev/null 2>&1 \
   || pgrep -f "botbowl-ui (eval|dataset)" > /dev/null 2>&1; do
    [ -e "$OUT/STOP" ] && { log "STOP before start"; exit 0; }
    sleep 60
done
log "box free"

MODEL=$(cat "$RUN_DIR/champion.txt")
[ -f "$MODEL" ] || { log "FATAL: champion $MODEL missing"; exit 1; }
log "champion $(basename "$MODEL"), $STATES states x $REPEATS repeats, budgets $BUDGETS"

# ---- Stage 0: the convergence curve ----------------------------------------
# name | puct-mode | puct-c
for ARM in "s0a-raw-c10|raw|10" "s0b-norm-c1|normalised|1"; do
    [ -e "$OUT/STOP" ] && { log "STOP — exiting"; exit 0; }
    NAME=$(echo "$ARM" | cut -d'|' -f1)
    MODE=$(echo "$ARM" | cut -d'|' -f2)
    C=$(echo "$ARM"    | cut -d'|' -f3)
    F="$OUT/$NAME.jsonl"
    [ -e "$F" ] && { log "$NAME done already"; continue; }
    log "$NAME: --puct-mode $MODE --puct-c $C, evaluator nn"
    START=$SECONDS
    if "$UI" convergence --states "$STATES" --repeats "$REPEATS" --budgets "$BUDGETS" \
            --seed "$SEED" --evaluator nn --model "$MODEL" \
            --puct-mode "$MODE" --puct-c "$C" \
            --out "$F" > "$OUT/$NAME.log" 2>&1; then
        log "$NAME done ($(((SECONDS - START) / 60)) min)"
        "$PY" "$REPO/scripts/convergence_summary.py" "$F" >> "$LOG" 2>&1 || log "(summary failed; raw jsonl kept)"
    else
        log "$NAME FAILED — see $NAME.log"
        tail -n 3 "$OUT/$NAME.log" >> "$LOG" 2>/dev/null
    fi
done

# ---- B1: does normalised Q win games? --------------------------------------
# The noise curve cannot answer this. Same net both sides, only the selection
# rule differs; --puct-c unset on the opponent would inherit the candidate's,
# so both are named explicitly.
B1="$OUT/b1-norm-vs-raw.json"
if [ ! -e "$B1" ] && [ ! -e "$OUT/STOP" ]; then
    log "b1: candidate norm c=1 vs opponent raw c=10, $GAMES games x$PARALLEL, 1000 iters"
    START=$SECONDS
    if "$UI" eval --evaluator nn --model "$MODEL" \
            --mcts-iters 1000 --puct-mode normalised --puct-c 1 \
            --vs-evaluator nn --vs-model "$MODEL" \
            --vs-puct-mode raw --vs-puct-c 10 \
            --vs-games "$GAMES" --seed 94000000 \
            --skip-lectures --skip-fixed-rungs \
            --parallel-games "$PARALLEL" \
            --per-game-out "$OUT/b1.games.jsonl" \
            --out "$B1" > "$OUT/b1.log" 2>&1; then
        log "b1 done ($(((SECONDS - START) / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$B1" 2>/dev/null || echo 'see json')"
        "$PY" "$REPO/scripts/paired_summary.py" "$OUT/b1.games.jsonl" >> "$LOG" 2>&1 || true
    else
        log "b1 FAILED — see b1.log"
        tail -n 3 "$OUT/b1.log" >> "$LOG" 2>/dev/null
    fi
fi
log "finished — loop left stopped on purpose; the next move depends on these results"
