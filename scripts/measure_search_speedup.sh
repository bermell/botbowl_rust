#!/usr/bin/env bash
# Plan-035/036 speedup, measured at the production generation shape.
#
# Replicates train_loop.sh's generate phase exactly (8 shards, --parallel-games 2,
# 1000 iters, --evaluator nn, gen19 net, sidecar, --truncate, random-start) but with
# GAMES games per shard instead of 600, and runs it once per binary arm.
# Seeds are gen20's own (30000000 + K*1e5), so every arm faces identical start states.
#
#   measure_search_speedup.sh <arm-name> <binary> [nn|heuristic]
#
# Build each arm from a worktree at its commit with its own CARGO_TARGET_DIR, and keep
# one scenario per invocation: two of these running at once contend for the 8 cores and
# neither number means anything.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="${SP:-$REPO/runs/exp-speedup}"   # holds bin/<arm> binaries and the per-arm output
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"
export LC_ALL=C

ARM="${1:?arm}"; BIN="${2:?binary}"; EV="${3:-nn}"
GAMES="${GAMES:-40}"
ITERS="${ITERS:-1000}"
MODEL="$REPO/models/bbnet_14x7_gen19.onnx"
PY="$REPO/train/.venv/bin/python"
SOCK=/tmp/bbnn-thr.sock
OUT="$SP/thr/$ARM-$EV"
mkdir -p "$OUT"
LOG="$SP/thr/throughput.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

NNPID=""
if [ "$EV" = nn ]; then
    rm -f "$SOCK"
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda \
        --model "$MODEL" --max-models 2 --stats-every 600 >> "$OUT/nn_server.log" 2>&1 &
    NNPID=$!
    for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
    [ -S "$SOCK" ] || { log "$ARM: sidecar failed to start"; kill $NNPID 2>/dev/null; exit 1; }
    EV_ARGS="--evaluator nn --model $MODEL --nn-server $SOCK --parallel-games 2"
else
    EV_ARGS="--evaluator heuristic"
fi
trap '[ -n "$NNPID" ] && kill $NNPID 2>/dev/null' EXIT

log "$ARM/$EV: 8 shards x $GAMES games, $ITERS iters, bin=$(basename "$BIN")"
T0=$SECONDS
PIDS=""
for K in 0 1 2 3 4 5 6 7; do
    SEED=$((30000000 + K * 100000))
    # shellcheck disable=SC2086
    "$BIN" dataset --mode random-start --games "$GAMES" --seed "$SEED" \
        --mcts-iters "$ITERS" $EV_ARGS --truncate \
        --out "$OUT/shard$K.jsonl" > "$OUT/shard$K.log" 2>&1 &
    PIDS="$PIDS $!"
done
for P in $PIDS; do wait "$P" || log "$ARM: shard pid $P nonzero"; done
WALL=$((SECONDS - T0))
[ -n "$NNPID" ] && { kill $NNPID 2>/dev/null; wait $NNPID 2>/dev/null; }
TOTAL=0
for K in 0 1 2 3 4 5 6 7; do TOTAL=$((TOTAL + $(wc -l < "$OUT/shard$K.jsonl"))); done
SAMPLES=$(cat "$OUT"/shard*.jsonl | "$PY" -c '
import sys,json
n=0
for line in sys.stdin:
    d=json.loads(line)
    n+=len(d.get("samples",[]))
print(n)' 2>/dev/null || echo "?")
log "$ARM/$EV RESULT: $TOTAL games in ${WALL}s = $(echo "scale=3; $WALL/$TOTAL" | bc) s/game, $SAMPLES samples ($(echo "scale=2; $SAMPLES/$TOTAL" | bc) decisions/game)"
