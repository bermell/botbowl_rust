#!/usr/bin/env bash
# Plan 032 #1b follow-up — does the completed-Q label close the from-scratch
# gap to the champion? D7 (visits) lost to gen03 0.396, d8h 0.425; Q7 (cq,
# tau 100) beat D7 0.613. Q7 vs gen03 on the same seed base says whether a
# from-scratch retrain with the new label is competitive with the incremental
# champion. No training; one 120-game match after the #7b/#9/#5 chain.
#
#   WAIT_PID=<s7b pid> nohup scripts/exp032_s7c_q7_vs_champ.sh > /dev/null 2>&1 &
SOCK="${SOCK:-/tmp/bbnn-exp032-s7c.sock}"
source "$(dirname "$0")/exp032_lib.sh"

Q7="$OUT/q7.onnx"
CHAMP="$MODELS/bbnet_14x7_gen03.onnx"
WAIT_PID="${WAIT_PID:-}"

[ -f "$Q7" ] || die "$Q7 missing"
[ -f "$CHAMP" ] || die "$CHAMP missing"
if [ -n "$WAIT_PID" ]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do stopped && exit 0; sleep 60; done
    sleep 90
fi
stopped && exit 0
log "=== #1b follow-up: Q7 (cq tau 100, from scratch) vs champion gen03 ==="
start_sidecar "$CHAMP"
play "s7c-q7-vs-gen03" "$Q7" "$CHAMP"
log "=== #7c complete ==="
