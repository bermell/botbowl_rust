#!/usr/bin/env bash
# Plan 068: the tests of the crates the divergence gate touches, at the loop's 20x11/7 capacity, in
# this checkout's own target dir (release profile: the search tests are slow in debug).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/exp079_20x11}"
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7
cd "$REPO"
exec nice -n 19 cargo test --release -j "${BUILD_JOBS:-4}" "$@"
