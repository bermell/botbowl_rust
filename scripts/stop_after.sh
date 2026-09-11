#!/usr/bin/env bash
# Place the loop's STOP file once a given line appears in status.md, so the
# loop exits at the *next* phase boundary after that line rather than the next
# boundary after now. Use it to stop after a specific phase, e.g.
#
#   nohup scripts/stop_after.sh "gen09 eval:" > /dev/null 2>&1 &
#
# waits for gen09's eval to start, then touches STOP, so the loop finishes that
# eval, writes the verdict, and exits before gen10 generate. (Touching STOP by
# hand mid-generate fires before *prepare* — that deadlocked plan 032 #12 on
# 2026-09-10, which waits for "gen09 train done".)
set -u
PATTERN="${1:?usage: stop_after.sh PATTERN [RUN_DIR]}"
RUN_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)/runs/loop14x7}"
STATUS="$RUN_DIR/status.md"
until grep -qF -- "$PATTERN" "$STATUS" 2>/dev/null; do sleep 60; done
touch "$RUN_DIR/STOP"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] STOP placed by stop_after.sh on '$PATTERN' — loop exits at the next boundary" >> "$STATUS"
