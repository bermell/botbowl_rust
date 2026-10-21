#!/usr/bin/env bash
# Plan 068: CPU-only smoke of `botbowl-ui eval --drive-gate` — scripted against scripted and random on
# a drive set. Scripted vs scripted is deterministic, so every pair must come out identical.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${MAIN_CHECKOUT:-/home/mattias/repos/botbowl_rust}"
OUT="${OUT:-/tmp/exp079_gate_smoke}"
rm -f "$OUT.jsonl" "$OUT.json"
BLOOD_NN_BACKEND=sidecar "$REPO/scripts/exp079_run.sh" eval --candidate-bot scripted --rungs scripted,random \
  --skip-lectures --positions "$R/runs/loopmix16x9g/positions/contested_16x9_gen04g.json" --games 12 \
  --drive-gate "${GATE:-report}" --sprt 0.5:0.55 --per-game-out "$OUT.jsonl" --out "$OUT.json" 2>&1 | grep ladder
python3 "$REPO/scripts/eval_summary.py" "$OUT.json" | tail -3
head -c 400 "$OUT.jsonl"; echo
python3 "$REPO/scripts/gate_replay.py" "$OUT.jsonl"
