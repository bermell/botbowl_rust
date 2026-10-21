#!/usr/bin/env bash
# Plan 068 / exp079: where does a net's disagreement with itself come from? Position 0 of
# contested_20x11_gen43, one thread (every forward is a batch of 1, so bit-identical numerics),
# twice with the shipped tie-break, once with tie_break = "mover" (deterministic).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${MAIN_CHECKOUT:-/home/mattias/repos/botbowl_rust}"
OUT="$R/runs/exp079"
RUN="$REPO/scripts/exp079_run.sh"
M="$R/models/az_v7/bbnet_mix16x9v9_gen48.onnx"
P="$R/runs/loopmix16x9v9/positions/contested_20x11_gen43.json"
mkdir -p "$OUT"
cp "$REPO/cfgs/chance_all_partial.toml" "$OUT/cap_mover.toml"
echo 'tie_break = "mover"' >> "$OUT/cap_mover.toml"
for i in 1 2; do
  "$RUN" drive-agree --bot-config "$REPO/cfgs/chance_all_partial.toml" --model "$M" --vs-model "$M" \
    --positions "$P" --count 1 --parallel 1 --out "$OUT/det_hash_$i.jsonl" 2>&1 | tail -1
done
"$RUN" drive-agree --bot-config "$OUT/cap_mover.toml" --model "$M" --vs-model "$M" --positions "$P" \
  --count 1 --parallel 1 --out "$OUT/det_mover.jsonl" 2>&1 | tail -1
