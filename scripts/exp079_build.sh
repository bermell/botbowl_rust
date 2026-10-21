#!/usr/bin/env bash
# Plan 068 / exp079: build botbowl-ui (torch) at the loop's 20x11/7 capacity into this checkout's own
# target dir. Never the main checkout's target/20x11, target/16x9 or target/release.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/exp079_20x11}"
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7
cd "$REPO"
exec nice -n 19 cargo build --release -j "${BUILD_JOBS:-4}" -p botbowl-ui --features botbowl-ui/torch "$@"
