#!/usr/bin/env bash
# Plan 068 §1 / exp079: how often do two bots play different moves on the drive benchmark?
# For each set and pair: a lockstep run per position (both bots asked at every decision, stopped at
# the first disagreement) at 1000 descents, plus bare-policy and 64-descent lockstep prescreens.
# Shipped preset (chance_all_partial) for the self-agreement floor; its deterministic twin
# (chance_all_partial_det, tie_break = mover) for every pair.
#   COUNT=40 PARALLEL=16 scripts/exp079_measure.sh
#   scripts/drive_agree_summary.py runs/exp079/agree_*.jsonl
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${MAIN_CHECKOUT:-/home/mattias/repos/botbowl_rust}"
OUT="$R/runs/exp079"
RUN="$REPO/scripts/exp079_run.sh"
MD="$R/models/az_v7"
COUNT="${COUNT:-40}"
PAR="${PARALLEL:-16}"
SETS="${SETS:-$R/runs/loopmix16x9v9/positions/contested_20x11_gen43.json $R/runs/loopmix16x9g/positions/contested_16x9_gen04g.json}"
# "tag candidate-net preset": the opponent is always gen48 (the loop's next reference).
PAIRS="${PAIRS:-48ship:gen48:chance_all_partial 48det:gen48:chance_all_partial_det 47det:gen47:chance_all_partial_det 45det:gen45:chance_all_partial_det}"
mkdir -p "$OUT"
for set in $SETS; do
  sname=$(basename "$set" .json)
  for p in $PAIRS; do
    IFS=: read -r tag cand preset <<< "$p"
    out="$OUT/agree_${sname}_${tag}.jsonl"
    [ -s "$out" ] && { echo "skip $out (exists)"; continue; }
    echo "== $sname $tag ($cand vs gen48, $preset) $(date +%T)"
    free -g | awk '/Mem/ {print "   MemAvailable " $7 " GB"}'
    "$RUN" drive-agree --bot-config "$REPO/cfgs/$preset.toml" \
      --model "$MD/bbnet_mix16x9v9_$cand.onnx" --vs-model "$MD/bbnet_mix16x9v9_gen48.onnx" \
      --positions "$set" --count "$COUNT" --parallel "$PAR" --prescreen-iters 0,64 \
      --out "$out.tmp" > "$out.log" 2>&1 && mv "$out.tmp" "$out"
    tail -1 "$out.log"
  done
done
echo "measure done $(date +%T)"
