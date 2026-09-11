#!/usr/bin/env bash
# Plan 030: score already-trained generations against the frozen anchor with
# exactly the settings the gateless loop uses for its vs-anchor rung (same
# seed base, games, parallelism, sidecar), so the anchor curve is continuous
# across the gate -> gateless switch. Writes gen/anchor.json (+ per-game log)
# and one status.md line per generation; anchor_curve.py reads both.
#
#   nohup scripts/anchor_backfill.sh gen08 gen09 > /dev/null 2>&1 &
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/14x7}"
RUN_DIR="${RUN_DIR:-$REPO/runs/loop14x7}"
MODEL_DIR="${MODEL_DIR:-$REPO/models}"
ANCHOR="${ANCHOR:-$MODEL_DIR/bbnet_14x7_gen03.onnx}"
ANCHOR_GAMES="${ANCHOR_GAMES:-40}"
MCTS_ITERS="${MCTS_ITERS:-1000}"
EVAL_PARALLEL_GAMES="${EVAL_PARALLEL_GAMES:-6}"
SOCK="${SOCK:-/tmp/bbnn-anchor-backfill.sock}"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"
PY="$REPO/train/.venv/bin/python"
STATUS="$RUN_DIR/status.md"
status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { status "FATAL anchor backfill: $*"; exit 1; }

[ $# -ge 1 ] || die "usage: anchor_backfill.sh genNN [genNN...]"
[ -x "$UI" ] || die "$UI missing — build first"
[ -f "$ANCHOR" ] && [ -f "${ANCHOR%.onnx}.pt" ] || die "anchor $ANCHOR (+.pt) missing"

NN_PID=""; NN_ARGS=""
rm -f "$SOCK"
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$ANCHOR" \
    --max-models 4 --stats-every 600 >> "$RUN_DIR/nn_server.log" 2>&1 &
NN_PID=$!
i=0
while [ ! -S "$SOCK" ]; do
    i=$((i + 1))
    if [ "$i" -gt 120 ] || ! kill -0 "$NN_PID" 2>/dev/null; then
        status "WARN anchor backfill: sidecar did not start — running on tract"; NN_PID=""; break
    fi
    sleep 1
done
[ -n "$NN_PID" ] && NN_ARGS="--nn-server $SOCK"
cleanup() { [ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"; }
trap cleanup EXIT
trap 'cleanup; trap - EXIT; exit 143' INT TERM

for GG in "$@"; do
    GEN_DIR="$RUN_DIR/$GG"
    MODEL="$MODEL_DIR/bbnet_14x7_$GG.onnx"
    [ -f "$MODEL" ] || die "$MODEL missing"
    [ -e "$GEN_DIR/anchor.json" ] && { status "anchor backfill $GG: already done"; continue; }
    status "anchor backfill $GG: $ANCHOR_GAMES games vs $(basename "$ANCHOR"), seed 0, x$EVAL_PARALLEL_GAMES"
    SECONDS=0
    # shellcheck disable=SC2086
    "$UI" eval --evaluator nn --model "$MODEL" --mcts-iters "$MCTS_ITERS" \
        --vs-evaluator nn --vs-model "$ANCHOR" --vs-games "$ANCHOR_GAMES" --seed 0 \
        --skip-lectures --skip-fixed-rungs --parallel-games "$EVAL_PARALLEL_GAMES" $NN_ARGS \
        --per-game-out "$GEN_DIR/anchor.games.jsonl" --out "$GEN_DIR/anchor.json" \
        > "$GEN_DIR/anchor.log" 2>&1 || die "$GG anchor eval failed — see $GEN_DIR/anchor.log"
    status "anchor backfill $GG done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$GEN_DIR/anchor.json")"
done
