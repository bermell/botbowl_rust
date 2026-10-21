#!/usr/bin/env bash
# Plan 068 / exp079: box-time schedule for the validation. Stops the running null arm (pinned pid
# NULL_PID) once it has NULL_LINES drive lines, then plays the gated and the ungated gen45 arms side by
# side at 8 streams each, and stops them at DEADLINE (epoch seconds) if they are not done, ahead of the
# loop's generate phase. Completed lines stay in each arm's eval.games.jsonl; scripts/gate_replay.py
# reads them (a killed arm writes no report.json).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${MAIN_CHECKOUT:-/home/mattias/repos/botbowl_rust}"
V="$REPO/scripts/exp079_validate.sh"
NULL_PID="${NULL_PID:?}"
NULL_LINES="${NULL_LINES:-50}"
DEADLINE="${DEADLINE:?epoch seconds}"
lines() { wc -l < "$1" 2>/dev/null || echo 0; }
while kill -0 "$NULL_PID" 2>/dev/null && [ "$(lines "$R/runs/exp079/val_null/eval.games.jsonl")" -lt "$NULL_LINES" ]; do sleep 20; done
kill "$NULL_PID" 2>/dev/null
echo "null stopped at $(lines "$R/runs/exp079/val_null/eval.games.jsonl") lines $(date +%T)"
sleep 5
export SPRT= PARALLEL=8
ARM=g45gate CAND=gen45 REF=gen48 GATE=report CAP="${CAP_PAIR:-200}" "$V" &
A=$!
sleep 60   # stagger the two net loads
ARM=g45plain CAND=gen45 REF=gen48 GATE= CAP="${CAP_PAIR:-200}" "$V" &
B=$!
while { kill -0 $A 2>/dev/null || kill -0 $B 2>/dev/null; } && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 20; done
# The evals are the arms' children; stop them by their output dirs' own processes.
for arm in g45gate g45plain; do
  for pid in $(ps -eo pid,args | awk -v d="val_$arm/" 'index($0, d) && /release\/botbowl-ui/ {print $1}'); do kill "$pid" 2>/dev/null; done
done
wait
echo "phase2 done $(date +%T): g45gate $(lines "$R/runs/exp079/val_g45gate/eval.games.jsonl") lines, g45plain $(lines "$R/runs/exp079/val_g45plain/eval.games.jsonl") lines"
