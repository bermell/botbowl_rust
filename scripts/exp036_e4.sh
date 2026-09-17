#!/usr/bin/env bash
# Plan 036 E4 — does the adopted recipe actually make a better player?
#
# Everything before this was val_* on a held-out set, and W3 changed the label
# those numbers are measured against, so they cannot answer this. Per plan
# 032's ground rule the val numbers selected the arm; only games decide it.
#
# ---- deviation from the plan's letter, on purpose -------------------------
# Plan 036 E4 says "600 games vs `scripted`" for each arm. This plays the two
# arms **head to head** over 600 games instead, for two reasons:
#
#   * It is the same question asked directly. Measuring A and B separately
#     against a third opponent and differencing costs 1200 games and carries
#     the variance of two independent estimates; 600 head-to-head games
#     estimate the difference itself.
#   * The two nets share a warm start, a window, a seed and a batch order, so
#     they are highly correlated — exactly the case where a direct comparison
#     is worth most.
#
# The absolute yardstick is not lost: the loop scores every generation against
# `scripted` anyway, so gen03 (old recipe) and gen04 (new recipe) give that
# reading for free on the report card.
#
# ---- what it is powered to see --------------------------------------------
# Points score, null 0.50, per-game variance ~0.20 with this run's ~18% draw
# rate. 600 games is SE 0.018, so 80% power against +0.05. Plan 036's stated
# +0.03 bar needs ~1745 games and is NOT resolvable here — this match tests
# whether the effect is large, and guards against the recipe having made the
# player *worse*, which is the risk the val numbers genuinely cannot see.
# Read a result in [0.47, 0.53] as "no large effect", not as "no effect".
#
#   nohup scripts/exp036_e4.sh > /dev/null 2>&1 &
#   tail -f runs/exp036/e4/e4.log
#   touch runs/exp036/e4/STOP     # clean exit at the next stage
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/14x7}"

MAIN="${MAIN:-/home/mattias/repos/botbowl_rust}"
RUN_DIR="${RUN_DIR:-$MAIN/runs/az14x7v6}"
OUT="${OUT:-$MAIN/runs/exp036/e4}"
MODEL_DIR="${MODEL_DIR:-$MAIN/models/az_v6}"
# The window to train both arms on, and the net to warm-start from — the loop's
# own regime, so `base` should reproduce that generation's net up to the seed.
GEN="${GEN:-gen03}"
INIT="${INIT:-$MODEL_DIR/bbnet_14x7_gen02.pt}"
LR="${LR:-2e-4}"
WINDOW_GENS="${WINDOW_GENS:-3}"
TRAIN_SHARDS="${TRAIN_SHARDS:-0 1 2 3 5 6}"
VAL_SHARDS="${VAL_SHARDS:-4 7}"
POLICY_ARGS="${POLICY_ARGS:---policy-target cq --tau 100}"
EPOCHS="${EPOCHS:-10}"
EVAL_EVERY="${EVAL_EVERY:-2500}"
TRAIN_SEED="${TRAIN_SEED:-20260917}"
# The adopted recipe (train_loop.sh from gen04 on).
NEW_PREPARE="${NEW_PREPARE:---value-blend 0.5}"
NEW_TRAIN="${NEW_TRAIN:---value-weight 0.25 --per-drive-value-weight}"
GAMES="${GAMES:-600}"
PARALLEL="${PARALLEL:-8}"
MCTS_ITERS="${MCTS_ITERS:-1000}"
# Far from the corpus seeds (loop uses 10e6 + gen*1e6 + shard*1e5) and from
# plan 032's 32e6, so no match replays a trained-on game.
SEED="${SEED:-36000000}"
SOCK="${SOCK:-/tmp/bbnn-e4.sock}"

PREPARE="$CARGO_TARGET_DIR/release/prepare"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"
PY="${PY:-$MAIN/train/.venv/bin/python}"

mkdir -p "$OUT"
LOG="$OUT/e4.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }
stopped() { [ -e "$OUT/STOP" ] && { log "STOP present — exiting cleanly"; return 0; }; return 1; }

[ -x "$PREPARE" ] || die "$PREPARE missing"
[ -x "$UI" ] || die "$UI missing"
[ -f "$INIT" ] || die "warm-start net $INIT missing"

gen_num() { echo "${1#gen}" | sed 's/^0*//'; }
window_inputs() {
    local shards="$1" inputs="" g k i n
    n=$(gen_num "$GEN")
    for ((i = n - WINDOW_GENS + 1; i <= n; i++)); do
        [ "$i" -lt 1 ] && continue
        g=$(printf "gen%02d" "$i")
        for k in $shards; do [ -f "$RUN_DIR/$g/shard$k.jsonl" ] && inputs="$inputs $RUN_DIR/$g/shard$k.jsonl"; done
    done
    echo "$inputs"
}

prep() {  # prep NAME [extra prepare args]
    local name="$1"; shift
    local dir="$OUT/prep_$name"
    [ -d "$dir" ] && { log "prep $name exists"; return 0; }
    local tr vl; tr=$(window_inputs "$TRAIN_SHARDS"); vl=$(window_inputs "$VAL_SHARDS")
    [ -n "$tr" ] || die "no shards in the $GEN window"
    log "prep $name: $(echo "$tr" | wc -w) train + $(echo "$vl" | wc -w) val shards, extra: ${*:-none}"
    # shellcheck disable=SC2086
    nice -n 19 "$PREPARE" --in $tr --out "$dir.tmp/train" $POLICY_ARGS "$@" > "$OUT/prep_$name.log" 2>&1 || die "prep $name failed"
    # shellcheck disable=SC2086
    nice -n 19 "$PREPARE" --in $vl --out "$dir.tmp/val" $POLICY_ARGS "$@" >> "$OUT/prep_$name.log" 2>&1 || die "prep $name val failed"
    mv "$dir.tmp" "$dir"
    log "prep $name done: $(grep -h 'prepare done' "$OUT/prep_$name.log" | head -1)"
}

train_arm() {  # train_arm NAME PREP [extra train args]
    local name="$1" prep_name="$2"; shift 2
    local onnx="$OUT/$name.onnx"
    [ -f "$onnx" ] && { log "arm $name already trained"; return 0; }
    local dt dv
    dt=$(ls -d "$OUT/prep_$prep_name/train"/dims_* | head -1)
    dv=$(ls -d "$OUT/prep_$prep_name/val"/dims_* | head -1)
    log "arm $name: prep $prep_name, warm start $(basename "$INIT") lr $LR, extra: ${*:-none}"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    nice -n 15 "$PY" -m bbnn.train --data "$dt" --val-data "$dv" \
        --init "$INIT" --lr "$LR" --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" \
        --select-on combined --seed "$TRAIN_SEED" --device auto \
        --out "$OUT/$name.pt" --onnx "$onnx" "$@" \
        > "$OUT/$name.train.log" 2>&1 || die "arm $name failed — see $name.train.log"
    log "arm $name done ($(((SECONDS - t0) / 60)) min): $(grep 'restored best-val' "$OUT/$name.train.log" | tail -1)"
}

NN_PID=""
start_sidecar() {
    rm -f "$SOCK"
    "$PY" "$MAIN/scripts/nn_server.py" --socket "$SOCK" --device cuda \
        --model "$OUT/new.onnx" --max-models 4 --stats-every 600 >> "$OUT/nn_server.log" 2>&1 &
    NN_PID=$!
    local i=0
    while [ ! -S "$SOCK" ]; do
        i=$((i + 1))
        if [ "$i" -gt 120 ] || ! kill -0 "$NN_PID" 2>/dev/null; then
            log "WARN: sidecar did not start — running on tract (slower, same result)"; NN_PID=""; return 0
        fi
        sleep 1
    done
    log "sidecar up (pid $NN_PID)"
}
stop_sidecar() { [ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; NN_PID=""; rm -f "$SOCK"; }
trap stop_sidecar EXIT
trap 'stop_sidecar; trap - EXIT; exit 143' INT TERM

log "=== plan 036 E4: adopted recipe vs baseline, head to head ==="
log "window $GEN, warm start $(basename "$INIT") at $LR, seed $TRAIN_SEED, commit $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"

stopped && exit 0
prep base
# Unquoted on purpose: NEW_PREPARE holds a flag *and* its value, and quoting
# it hands prepare one argument called "--value-blend 0.5".
# shellcheck disable=SC2086
prep blend $NEW_PREPARE

stopped && exit 0
# shellcheck disable=SC2086
train_arm base base
stopped && exit 0
# shellcheck disable=SC2086
train_arm new blend $NEW_TRAIN

stopped && exit 0
REP="$OUT/e4_match.json"
if [ ! -e "$REP" ]; then
    log "match: new vs base, $GAMES games x$PARALLEL, seed $SEED, $MCTS_ITERS iters"
    start_sidecar
    NN_ARGS=""; [ -n "$NN_PID" ] && NN_ARGS="--nn-server $SOCK"
    t0=$SECONDS
    # shellcheck disable=SC2086
    "$UI" eval --evaluator nn --model "$OUT/new.onnx" --mcts-iters "$MCTS_ITERS" \
        --vs-evaluator nn --vs-model "$OUT/base.onnx" \
        --vs-games "$GAMES" --seed "$SEED" \
        --skip-lectures --skip-fixed-rungs --parallel-games "$PARALLEL" $NN_ARGS \
        --per-game-out "$OUT/e4_match.games.jsonl" --out "$REP" \
        > "$OUT/e4_match.log" 2>&1 || die "match failed — see e4_match.log"
    stop_sidecar
    log "match done ($(((SECONDS - t0) / 60)) min)"
fi

log "RESULT: $("$PY" "$MAIN/scripts/eval_summary.py" "$REP" 2>&1 | tail -1)"
"$PY" "$MAIN/scripts/paired_summary.py" "$OUT/e4_match.games.jsonl" 2>&1 | tee -a "$LOG" || true
log "Read [0.47, 0.53] as 'no large effect' — 600 games is powered for +0.05, not +0.03."
