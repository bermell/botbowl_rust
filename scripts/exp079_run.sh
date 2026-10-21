#!/usr/bin/env bash
# Plan 068 / exp079: run this checkout's exp079 botbowl-ui build (scripts/exp079_build.sh) at nice 19
# with the loop's in-process settings (canvas 13x22, inproc on cuda). Arguments go to botbowl-ui.
#   scripts/exp079_run.sh drive-agree --model A.onnx --vs-model B.onnx --positions P.json --out O.jsonl
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${CARGO_TARGET_DIR:-$REPO/target/exp079_20x11}"
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7
export BLOOD_NN_CANVAS="${BLOOD_NN_CANVAS:-13x22}"
export BLOOD_NN_BACKEND="${BLOOD_NN_BACKEND:-inproc}"
export BLOOD_NN_DEVICE="${BLOOD_NN_DEVICE:-cuda}"
# Run from the main checkout so cfgs/, models/ and runs/ paths resolve as the loop's do.
cd "${RUN_FROM:-$REPO}"
exec nice -n 19 "$TARGET/release/botbowl-ui" "$@"
