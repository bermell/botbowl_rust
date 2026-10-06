#!/usr/bin/env bash
# Plan 058 diagnostic: is the Python sidecar's per-request front end the generation bottleneck?
# Two sidecars, two dataset processes (each half the games and streams), run at once.
#   scripts/perf_two_sidecars.sh OUT_DIR [NET.onnx] [STREAMS_EACH] [GAMES_EACH]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out}"; NET="${2:-models/az_v7/bbnet_mix16x9g056_gen04_v9.onnx}"; P="${3:-18}"; G="${4:-24}"
mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"
SIZES="--size-centre 144 --size-temperature 0.3 --size-floor 0.2 --size-max-area 144 --size-min-area 70"
NNS=""
for i in 1 2; do
    rm -f /tmp/bbnn-perf2-$i.sock
    "$PY" scripts/nn_server.py --socket /tmp/bbnn-perf2-$i.sock --device cuda --model "$NET" --stats-every 30 \
        --canvas 11x18 > "$OUT/nn_server.$i.log" 2>&1 & NNS="$NNS $!"
done
for i in 1 2; do for _ in $(seq 120); do [ -S /tmp/bbnn-perf2-$i.sock ] && break; sleep 1; done; done
START=$(date +%s.%N); PIDS=""
for i in 1 2; do
    # shellcheck disable=SC2086
    "$UI" dataset --mode random-start --games "$G" --seed $((58000 + 1000 * i)) --bot-config cfgs/gumbel16_f1000_gen.toml \
        --mcts-iters 1000 --evaluator nn --model "$NET" --nn-server /tmp/bbnn-perf2-$i.sock --parallel-games "$P" \
        $SIZES --out "$OUT/games.$i.jsonl" --truncate > "$OUT/dataset.$i.log" 2>&1 & PIDS="$PIDS $!"
done
# shellcheck disable=SC2086
wait $PIDS
END=$(date +%s.%N)
sleep 31
# shellcheck disable=SC2086
kill $NNS; wait 2>/dev/null
awk -v g=$((2 * G)) -v s="$START" -v e="$END" 'BEGIN { m = (e - s) / 60; printf "two sidecars: %d games in %.1f min = %.1f games/min\n", g, m, g / m }' | tee "$OUT/summary.txt"
for i in 1 2; do grep 'samples/s' "$OUT/nn_server.$i.log" | tail -1 | grep -oE "mean_batch=[0-9.]+|[0-9]+ samples/s" | paste -sd' '; done | tee -a "$OUT/summary.txt"
