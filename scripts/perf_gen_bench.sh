#!/usr/bin/env bash
# Throughput benchmark for generation (plan 058): N random-start games with the loop's generation
# preset on one GPU sidecar, at each --parallel-games in a list. Prints games/min, process CPU (%
# of one core; 800% = all 8 cores) and the sidecar's mean batch, GPU per sample and samples/s.
#
#   scripts/perf_gen_bench.sh OUT_DIR [NET.onnx] [PARALLEL_LIST] [GAMES]
#   scripts/perf_gen_bench.sh runs/perf/base models/az_v7/bbnet_mix16x9g056_gen04_v9.onnx "12 24 36" 48
#
# Env: CONFIG (default cfgs/gumbel16_f1000_gen.toml), ITERS (default 1000), SEED (default 58000).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; NET="${2:-models/az_v7/bbnet_mix16x9g056_gen04_v9.onnx}"
LIST="${3:-12 24 36}"; GAMES="${4:-48}"
CONFIG="${CONFIG:-cfgs/gumbel16_f1000_gen.toml}"; ITERS="${ITERS:-1000}"; SEED="${SEED:-58000}"
mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || { echo "build failed" >&2; exit 1; }
SIZES="--size-centre 144 --size-temperature 0.3 --size-floor 0.2 --size-max-area 144 --size-min-area 70"
for P in $LIST; do
    SOCK="/tmp/bbnn-perf-$$.sock"; rm -f "$SOCK"
    "$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$NET" --stats-every 30 \
        --canvas 11x18 > "$OUT/nn_server.p$P.log" 2>&1 & NN=$!
    for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
    START=$(date +%s.%N)
    # shellcheck disable=SC2086
    /usr/bin/time -f "%P %e" -o "$OUT/time.p$P" "$UI" dataset --mode random-start --games "$GAMES" \
        --seed "$SEED" --bot-config "$CONFIG" --mcts-iters "$ITERS" --evaluator nn --model "$NET" \
        --nn-server "$SOCK" --parallel-games "$P" $SIZES --out "$OUT/games.p$P.jsonl" --truncate \
        > "$OUT/dataset.p$P.log" 2>&1
    END=$(date +%s.%N)
    sleep 31   # one more stats line from the sidecar
    kill "$NN"; wait "$NN" 2>/dev/null; rm -f "$SOCK"
    STATS=$(grep 'samples/s' "$OUT/nn_server.p$P.log" | tail -2 | head -1)
    read -r CPU _ < "$OUT/time.p$P"
    awk -v p="$P" -v g="$GAMES" -v s="$START" -v e="$END" -v cpu="$CPU" -v st="$STATS" 'BEGIN {
        m = (e - s) / 60
        match(st, /mean_batch=[0-9.]+/); mb = substr(st, RSTART + 11, RLENGTH - 11)
        match(st, /[0-9]+us GPU\/sample/); gs = substr(st, RSTART, RLENGTH)
        match(st, /[0-9]+ samples\/s/); sps = substr(st, RSTART, RLENGTH)
        printf "parallel %3d: %d games in %.1f min = %.1f games/min; CPU %s; mean_batch %s, %s, %s\n", p, g, m, g / m, cpu, mb, gs, sps
    }' | tee -a "$OUT/summary.txt"
done
