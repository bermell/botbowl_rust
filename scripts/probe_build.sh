#!/usr/bin/env bash
# Plan 064: build the probe binary at the loop's 16x9/6 capacity, niced and on two jobs so it never
# competes with a live loop on the same box. Prints the binary path.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Its own target dir by default, never the loop's `target/16x9`: rebuilding there from a main checkout
# would swap the live loop's botbowl-ui underneath it. PROBE_TARGET: another dir (e.g. to build while
# a probe run is using this one).
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="${PROBE_TARGET:-$REPO/target/probe16x9}"
cd "$REPO"
nice -n 19 cargo build --release -j "${JOBS:-2}" -p botbowl-ui >&2
echo "$CARGO_TARGET_DIR/release/botbowl-ui"
