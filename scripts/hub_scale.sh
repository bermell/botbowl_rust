#!/usr/bin/env bash
# Plan 069 item 6: the hub at 100 / 1,000 / 4,000 fake streams, generate and label (scripts/hub_scale.py
# drives it; botbowl-worker's `botbowl-fake-worker` plays the workers: real protocol, canned
# results, no engine or GPU work). Builds the hub and the fake worker into their own target dir at
# 20x11/7 (hub, fake worker and corpus must share a capacity), makes a canned corpus with a short
# heuristic `botbowl-ui dataset` run unless one is given, and writes OUT/summary.{md,json}.
#
#   scripts/hub_scale.sh OUT_DIR [CANNED.jsonl] [-- hub_scale.py args, e.g. --stages generate:1000:60:180]
#
# Env: CARGO_TARGET_DIR (default target/hubscale; never the loop's), BUILD_JOBS (3), PORT (14000;
# stage i listens on PORT+i, nowhere near the loop's 13337), CANNED_GAMES (48). Keeps >= 70 GB free
# (hub_scale.py stops before a stage below that). Nothing here touches runs/ or a running hub.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; shift
CANNED=""
if [ $# -gt 0 ] && [ "$1" != -- ]; then CANNED="$1"; shift; fi
[ "${1:-}" = -- ] && shift
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/hubscale}"
BIN="$CARGO_TARGET_DIR/release"
df -h / | tail -1
cargo build --release -j "${BUILD_JOBS:-3}" -p botbowl-hub --bin botbowl-hub
cargo build --release -j "${BUILD_JOBS:-3}" -p botbowl-worker --bin botbowl-fake-worker
if [ -z "$CANNED" ]; then
    CANNED="$OUT/canned.jsonl"
    if [ ! -s "$CANNED" ]; then
        cargo build --release -j "${BUILD_JOBS:-3}" -p botbowl-ui --bin botbowl-ui
        # Heuristic search at a low budget: real 20x11 trajectories (per-player setup, next-drive
        # pairs) in minutes. Their lines are smaller than an NN corpus's (no completed-Q fields from
        # a net): the summary prints the mean size, and a real shard can be passed instead.
        "$BIN/botbowl-ui" dataset --mode random-start --games "${CANNED_GAMES:-48}" --evaluator heuristic \
            --mcts-iters 200 --next-drive --parallel-games 3 --seed 69000 --out "$CANNED" --truncate
    fi
fi
ulimit -n "$(ulimit -Hn)"
echo "fds $(ulimit -n); canned $CANNED ($(wc -l < "$CANNED") lines, $(du -h "$CANNED" | cut -f1))"
python3 "$REPO/scripts/hub_scale.py" --hub-bin "$BIN/botbowl-hub" --fake-bin "$BIN/botbowl-fake-worker" \
    --canned "$CANNED" --out "$OUT" --port "${PORT:-14000}" "$@"
df -h / | tail -1
