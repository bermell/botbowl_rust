#!/usr/bin/env bash
# exp075 (plan 064): does averaging the value over the top-bottom mirror (test-time augmentation)
# cut the value head's error against MC truth? `value-bench` plain and `--symmetrise`, on tract
# (CPU), two cores, nice 19; then the summary pairs every arm against the first.
#
#   scripts/exp075_vbench.sh BENCH.jsonl OUT_DIR NET.onnx [NET.onnx ...]
# The bench's corpus paths are relative to the repo root: run from a checkout whose runs/ holds
# them (a worktree can symlink runs/loopmix16x9v9 to the main checkout's).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BENCH="${1:?bench}"; OUT="${2:?out dir}"; shift 2
CPUS="${CPUS:-6,7}"; PAR="${PAR:-2}"
UI="$("$REPO/scripts/probe_build.sh")"
PY="${PY:-$(command -v python3)}"
mkdir -p "$OUT"
unset BLOOD_NN_SERVER
ARGS=()
for net in "$@"; do
    n="$(basename "$net" .onnx)"
    for arm in plain sym; do
        flag=(); [ "$arm" = sym ] && flag=(--symmetrise)
        [ -s "$OUT/$n.$arm.jsonl" ] || taskset -c "$CPUS" nice -n 19 "$UI" value-bench --bench "$BENCH" \
            --model "$net" --parallel "$PAR" --out "$OUT/$n.$arm.jsonl" "${flag[@]}" 2> "$OUT/$n.$arm.log"
        ARGS+=("$n.$arm=$OUT/$n.$arm.jsonl")
    done
done
"$PY" scripts/value_bench_summary.py "${ARGS[@]}" | tee "$OUT/summary.txt"
