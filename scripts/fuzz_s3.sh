#!/usr/bin/env bash
# Plan 069 item 7: fuzz the engine with `botbowl-ui fuzz` at one board capacity.
#
#   scripts/fuzz_s3.sh                       # 20x11/7, the loop's size mix, 30 min on every core
#   CAP=16x9/6 MINUTES=10 scripts/fuzz_s3.sh
#   CAP=26x15/11 BOARDS=full scripts/fuzz_s3.sh --start full
#   MODEL=models/az_v7/bbnet_mix16x9v9_gen58.onnx scripts/fuzz_s3.sh --nn-prob 0.7   # net-played games (tract, CPU)
#
# Builds `botbowl-ui` with the `fuzz` profile (release + debug assertions + overflow checks) at the
# capacity into its own target dir (never one a loop or worker runs from), then plays games until
# MINUTES run out. Results land in OUT (default runs-free: fuzz_out/<cap>-<date>): `failures.jsonl`
# (one line per failing game: class, message, seed, micro-step, config) and `summary.json`
# (counts, coverage by procedure and action type, distinct failures with their first seed).
# Reproduce one: the same binary with `fuzz --only-seed S [--dump-from STEP] --out DIR` (and the
# same --boards / --start / --policy flags the run used). Extra arguments go to `fuzz` as they are.
#
# On a CPU partition: one task per node, THREADS = the node's cores; give each task its own SEED
# range (`SEED=$((SLURM_PROCID * 100000000))`) so no two tasks play the same games.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
CAP="${CAP:-20x11/7}"
W="${CAP%%x*}"; REST="${CAP#*x}"; H="${REST%%/*}"; P="${CAP##*/}"
MINUTES="${MINUTES:-30}"
THREADS="${THREADS:-0}"
SEED="${SEED:-0}"
BOARDS="${BOARDS:-loop}"
OUT="${OUT:-$REPO/fuzz_out/${W}x${H}-$(date +%Y%m%d-%H%M%S)}"
export BOARD_SIZE_W="$W" BOARD_SIZE_H="$H" BOARD_PLAYERS="$P"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/fuzz_${W}x${H}}"
cargo build --profile fuzz -p botbowl-ui
BIN="$CARGO_TARGET_DIR/fuzz/botbowl-ui"
MODEL_ARGS=()
[ -n "${MODEL:-}" ] && MODEL_ARGS=(--model "$MODEL")
mkdir -p "$OUT"
echo "fuzz: capacity ${W}x${H}/${P}, boards $BOARDS, $MINUTES min, seed $SEED, out $OUT" >&2
"$BIN" fuzz --games 0 --minutes "$MINUTES" --threads "$THREADS" --seed "$SEED" --boards "$BOARDS" \
    --out "$OUT" "${MODEL_ARGS[@]}" "$@" 2>&1 | tee "$OUT/fuzz.log"
