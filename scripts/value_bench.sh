#!/usr/bin/env bash
# Plan 056 §2: score nets' value heads on a frozen Monte Carlo value benchmark (minutes per net).
#
#   scripts/value_bench.sh BENCH.jsonl OUT_DIR [NAME=]NET.onnx [[NAME=]NET.onnx ...]
#   scripts/value_bench.sh runs/value_bench/g05_gen06.jsonl runs/value_bench/g05 \
#       g05=models/az_v7/bbnet_mix16x9g_gen05.onnx armB=runs/plan056/armB.onnx
#
# One GPU sidecar serves every net; each net's rows land in OUT_DIR/NAME.jsonl (kept, so a rerun
# skips it), and OUT_DIR/summary.txt pairs every net against the FIRST (the reference).
# Env: PARALLEL (threads per net, default 8).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BENCH="${1:?benchmark jsonl}"; OUT="${2:?out dir}"; shift 2
[ $# -ge 1 ] || { echo "no nets" >&2; exit 1; }
PARALLEL="${PARALLEL:-8}"
mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"
SOCK="/tmp/bbnn-vbench-$$.sock"
NN_PID=""; trap '[ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"' EXIT INT TERM
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || { echo "build failed — see $OUT/build.log" >&2; exit 1; }
first="${1#*=}"
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$first" --max-models $(( $# + 1 )) \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
[ -S "$SOCK" ] || { echo "nn_server did not come up — see $OUT/nn_server.log" >&2; exit 1; }
ARGS=()
for spec in "$@"; do
    net="${spec#*=}"; name="${spec%%=*}"; [ "$name" = "$spec" ] && name="$(basename "$net" .onnx)"
    ARGS+=("$name=$OUT/$name.jsonl")
    [ -s "$OUT/$name.jsonl" ] && [ -e "$OUT/$name.done" ] && continue
    "$UI" value-bench --bench "$BENCH" --model "$net" --nn-server "$SOCK" --parallel "$PARALLEL" \
        --out "$OUT/$name.jsonl" 2> "$OUT/$name.log" && touch "$OUT/$name.done" \
        || { echo "value-bench failed on $net — see $OUT/$name.log" >&2; exit 1; }
    tail -1 "$OUT/$name.log"
done
"$PY" scripts/value_bench_summary.py "${ARGS[@]}" | tee "$OUT/summary.txt"
