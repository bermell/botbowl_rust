#!/usr/bin/env bash
# Plan 056 §2: score nets' value heads on a frozen Monte Carlo value benchmark (minutes per net).
#
#   scripts/value_bench.sh BENCH.jsonl OUT_DIR [NAME=]NET.onnx [[NAME=]NET.onnx ...]
#   scripts/value_bench.sh runs/value_bench/g05_gen06.jsonl runs/value_bench/g05 \
#       g05=models/az_v7/bbnet_mix16x9g_gen05.onnx armB=runs/plan056/armB.onnx
#
# Each net's rows land in OUT_DIR/NAME.jsonl (kept, so a rerun
# skips it), and OUT_DIR/summary.txt pairs every net against the FIRST (the reference).
# Env: PARALLEL (threads per net, default 8); BOARD_SIZE_W/H, BOARD_PLAYERS: the build capacity (default
# 16x9/6, plan 065 — any capacity the benchmark's boards fit). NN_BACKEND (plan 066): `inproc` (default),
# each net in its own `value-bench --nn-backend inproc` process in turn (needs NET.ts beside each
# NET.onnx and a `--features torch` build, made here), or `sidecar`, one Python sidecar for all.
# A forward that fell back to tract fails the bench (exit 3).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BENCH="${1:?benchmark jsonl}"; OUT="${2:?out dir}"; shift 2
[ $# -ge 1 ] || { echo "no nets" >&2; exit 1; }
PARALLEL="${PARALLEL:-8}"
mkdir -p "$OUT"
# Plan 065: the capacity follows the caller (the loop exports its BUILD_* as BOARD_SIZE_*), 16x9/6 when
# unset. A binary replays any corpus whose boards fit its capacity (`BoardCapacity::holds`).
export BOARD_SIZE_W="${BOARD_SIZE_W:-16}" BOARD_SIZE_H="${BOARD_SIZE_H:-9}" BOARD_PLAYERS="${BOARD_PLAYERS:-6}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/${BOARD_SIZE_W}x${BOARD_SIZE_H}}"
NN_BACKEND="${NN_BACKEND:-inproc}"
CANVAS="$((BOARD_SIZE_H + 2))x$((BOARD_SIZE_W + 2))"   # the build capacity, border included (11x18 at 16x9)
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="${PY:-$REPO/train/.venv/bin/python}"
SOCK="/tmp/bbnn-vbench-$$.sock"
NN_PID=""; trap '[ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"' EXIT INT TERM
FEATURES=(); [ "$NN_BACKEND" = inproc ] && FEATURES=(--features botbowl-ui/torch)
cargo build --release -j 4 -p botbowl-ui "${FEATURES[@]}" >> "$OUT/build.log" 2>&1 || { echo "build failed — see $OUT/build.log" >&2; exit 1; }
first="${1#*=}"
if [ "$NN_BACKEND" = inproc ]; then
    NN_ARGS=(--nn-backend inproc)
else
    "$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$first" --max-models $(( $# + 1 )) \
        --stats-every 300 --canvas "$CANVAS" >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
    for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
    [ -S "$SOCK" ] || { echo "nn_server did not come up — see $OUT/nn_server.log" >&2; exit 1; }
    NN_ARGS=(--nn-server "$SOCK")
fi
ARGS=()
for spec in "$@"; do
    net="${spec#*=}"; name="${spec%%=*}"; [ "$name" = "$spec" ] && name="$(basename "$net" .onnx)"
    ARGS+=("$name=$OUT/$name.jsonl")
    [ -s "$OUT/$name.jsonl" ] && [ -e "$OUT/$name.done" ] && continue
    BLOOD_NN_CANVAS="$CANVAS" "$UI" value-bench --bench "$BENCH" --model "$net" "${NN_ARGS[@]}" --parallel "$PARALLEL" \
        --out "$OUT/$name.jsonl" 2> "$OUT/$name.log" \
        || { echo "value-bench failed on $net — see $OUT/$name.log" >&2; exit 1; }
    grep -q 'NN_SERVER_FALLBACK' "$OUT/$name.log" && { echo "value-bench on $net fell back to tract — see $OUT/$name.log" >&2; exit 3; }
    if [ "$NN_BACKEND" = inproc ] && ! grep -q 'NN_INPROC loaded' "$OUT/$name.log"; then
        echo "value-bench on $net never loaded the net in-process — see $OUT/$name.log" >&2; exit 3
    fi
    touch "$OUT/$name.done"
    tail -1 "$OUT/$name.log"
done
"$PY" scripts/value_bench_summary.py "${ARGS[@]}" | tee "$OUT/summary.txt"
