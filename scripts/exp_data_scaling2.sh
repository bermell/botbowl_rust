#!/usr/bin/env bash
# Plan 029 stages 3-4 — the two follow-ups the main result needs.
#
# Stage 3, THE WARM-STARTED PAIR. Stages 1-2 trained from scratch and found
# corpus size worth +0.05-0.06 per doubling (D3 0.600, D7 0.662 vs D1). But
# production warm-starts from the champion, and gen03 already encodes
# gen00-gen02 — so it may have already extracted most of what a wider window
# would add. From-scratch does not transfer one-for-one, and this is the arm
# that licenses changing WINDOW_GENS:
#     W1  gen07        }  both --init models/bbnet_14x7_gen03.pt --lr 2e-4
#     W3  gen05-gen07  }  i.e. exactly what the loop does today
# If W3 > W1 the widening pays on top of a champion. If not, the from-scratch
# curve is real but production-irrelevant, and that is worth knowing before
# spending disk and wall clock on a wider window forever.
#
# Stage 4, THE M1 CONTROL. D7 confounds volume with diversity by construction:
# 7x the samples, but also seven generating networks across two regimes instead
# of one. M1 holds sample count at D1's level and varies only the number of
# sources — `head -n 86` of each of the 6 train shards of each of the 7
# generations = 3,612 games against D1's 3,600, drawn from seven distributions
# instead of one. A jsonl line is one game, so this subsamples at game
# granularity with no leakage.
#     M1 vs D1  = diversity at fixed volume
#     D7 vs M1  = volume at fixed diversity   (D7 vs D1 is already 0.662)
# The readings diverge sharply. Pure volume => generate more games. Mostly
# diversity => 4800 games from one net are worth much less than 4800 spread
# across several, which argues for mixing generations rather than just making
# more.
#
#   nohup scripts/exp_data_scaling2.sh > /dev/null 2>&1 &
#   touch runs/exp-data/STOP2

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"

RUN_DIR="$REPO/runs/loop14x7"
OUT="$REPO/runs/exp-data"
UI="$REPO/target/14x7/release/botbowl-ui"
PREPARE="$REPO/target/14x7/release/prepare"
PY="$REPO/train/.venv/bin/python"
SOCK=/tmp/bbnn-data2.sock
HOLD="$RUN_DIR/gen07/prepared_val/dims_16x9"
CHAMP_PT="$REPO/models/bbnet_14x7_gen03.pt"
STEPS="${STEPS:-110000}"
EVAL_EVERY="${EVAL_EVERY:-2500}"
GAMES="${GAMES:-120}"
PARALLEL="${PARALLEL:-6}"
SEED="${SEED:-20260906}"
M1_GAMES_PER_SHARD="${M1_GAMES_PER_SHARD:-86}"

LOG="$OUT/exp2.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
die() { log "FATAL: $*"; exit 1; }
stopped() { [ -e "$OUT/STOP2" ]; }
NN_ARGS=""
NN_PID=""

log "=== plan 029 stages 3-4 start ==="
[ -d "$HOLD" ]      || die "holdout missing"
[ -f "$CHAMP_PT" ]  || die "champion weights missing: $CHAMP_PT"

# --- shared helpers (separate `local` decls: bash expands them all before
# assigning, so `local a="$1" b="$a"` would read the OLD $a — under set -u an
# unbound-variable error that vanishes inside a command substitution) --------
train_arm() {
    local name="$1"
    local dims="$2"
    local init="$3"
    local lr="$4"
    local pt="$OUT/$name.pt"
    local onnx="$OUT/$name.onnx"
    [ -f "$onnx" ] && { log "$name already trained"; return 0; }
    log "$name: $dims, init $(basename "$init"), lr $lr, $STEPS steps"
    local t0=$SECONDS
    if ! "$PY" -m bbnn.train --data "$dims" --val-data "$HOLD" \
            --init "$init" --lr "$lr" --seed "$SEED" \
            --max-steps "$STEPS" --eval-every "$EVAL_EVERY" --select-on combined \
            --device auto --out "$pt" --onnx "$onnx" \
            > "$OUT/$name.train.log" 2>&1; then
        log "$name FAILED — see $name.train.log"; return 1
    fi
    log "$name trained ($(((SECONDS - t0) / 60)) min): $(grep 'restored best-val' "$OUT/$name.train.log" | tail -1)"
    log "$name baseline: $(grep 'warm-start baseline' "$OUT/$name.train.log" | tail -1)"
}

start_sidecar() {
    [ -n "$NN_PID" ] && return 0
    rm -f "$SOCK"
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda \
        --model "$OUT/d1.onnx" --max-models 6 --stats-every 600 >> "$OUT/nn_server2.log" 2>&1 &
    NN_PID=$!
    local i=0
    while [ ! -S "$SOCK" ]; do
        i=$((i + 1))
        if [ "$i" -gt 120 ] || ! kill -0 "$NN_PID" 2>/dev/null; then
            log "WARN: sidecar did not start — running on tract"; NN_PID=""; return 0
        fi
        sleep 1
    done
    NN_ARGS="--nn-server $SOCK"
    log "sidecar up (pid $NN_PID)"
}
cleanup() { [ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM

play() {
    local cand="$1"
    local opp="$2"
    local seed="$3"
    local tag="$4"
    local rep="$OUT/$tag.json"
    [ -e "$rep" ] && { log "$tag already played"; return 0; }
    log "$tag: $cand vs $opp, $GAMES games x$PARALLEL"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    if ! "$UI" eval --evaluator nn --model "$OUT/$cand.onnx" \
            --mcts-iters 1000 \
            --vs-evaluator nn --vs-model "$OUT/$opp.onnx" \
            --vs-games "$GAMES" --seed "$seed" \
            --skip-lectures --skip-fixed-rungs \
            --parallel-games "$PARALLEL" $NN_ARGS \
            --per-game-out "$OUT/$tag.games.jsonl" \
            --out "$rep" > "$OUT/$tag.log" 2>&1; then
        log "$tag FAILED — see $tag.log"; return 1
    fi
    log "$tag done ($(((SECONDS - t0) / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$rep" 2>/dev/null || echo see json)"
    "$PY" "$REPO/scripts/paired_summary.py" "$OUT/$tag.games.jsonl" >> "$LOG" 2>&1 || true
}

# ---- stage 3: warm-started pair --------------------------------------------
stopped && { log "STOP2 before stage 3"; exit 0; }
train_arm w1 "$OUT/prep_d1/dims_16x9"              "$CHAMP_PT" 2e-4 || die "W1 failed"
train_arm w3 "$RUN_DIR/gen07/prepared_train/dims_16x9" "$CHAMP_PT" 2e-4 || die "W3 failed"
# Both warm-started from the same file, so their pre-training baselines must
# match exactly — the same free assertion stages 1-2 used.
WB=$(grep -h 'warm-start baseline' "$OUT"/w1.train.log "$OUT"/w3.train.log | grep -oE 'val_value [0-9.]+' | sort -u)
[ "$(echo "$WB" | wc -l)" -eq 1 ] || die "W arms did not share an init: $(echo $WB)"
log "warm-pair init assertion passed — both start at $WB"
start_sidecar
play w3 w1 98000000 "s3-w3-vs-w1" || true
log "=== stage 3 complete ==="

# ---- stage 4: M1, diversity at fixed volume --------------------------------
stopped && { log "STOP2 before stage 4"; exit 0; }
M1SH="$OUT/m1_shards"
if [ ! -d "$OUT/prep_m1" ]; then
    mkdir -p "$M1SH"
    M1_IN=""
    for g in gen01 gen02 gen03 gen04 gen05 gen06 gen07; do
        for k in 0 1 2 3 5 6; do
            f="$M1SH/${g}_shard$k.jsonl"
            [ -s "$f" ] || head -n "$M1_GAMES_PER_SHARD" "$RUN_DIR/$g/shard$k.jsonl" > "$f"
            M1_IN="$M1_IN $f"
        done
    done
    log "M1 shards: $(echo $M1_IN | wc -w) files x $M1_GAMES_PER_SHARD games = $(cat $M1_IN | wc -l) games (D1 has 3600)"
    # shellcheck disable=SC2086
    "$PREPARE" --in $M1_IN --out "$OUT/prep_m1" >> "$LOG" 2>&1 || die "M1 prepare failed"
fi
log "M1 pool: $(python3 -c "import json;print(json.load(open('$OUT/prep_m1/dims_16x9/manifest.json'))['num_samples'],'samples')" 2>/dev/null || echo '?')"
train_arm m1 "$OUT/prep_m1/dims_16x9" "$OUT/arm_init.pt" 1e-3 || die "M1 failed"
MB=$(grep 'warm-start baseline' "$OUT/m1.train.log" | grep -oE 'val_value [0-9.]+')
[ "$MB" = "val_value 0.6410" ] || log "WARN: M1 baseline $MB != the D-arms' val_value 0.6410"
start_sidecar
play m1 d1 99000000 "s4-m1-vs-d1" || true
log "=== stage 4 complete — box free, loop still stopped ==="
