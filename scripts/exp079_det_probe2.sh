#!/usr/bin/env bash
# Plan 068 / exp079: with the Gumbel root ordered by a deterministic tie-break, does gen48 agree with
# itself? Positions 0-2 of contested_20x11_gen43, tie_break = "mover", run twice. PARALLEL=1 (the
# default) makes every forward a batch of 1 (bit-identical numerics); PARALLEL=3 adds the
# in-process batcher's bucket-dependent numerics.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${MAIN_CHECKOUT:-/home/mattias/repos/botbowl_rust}"
OUT="$R/runs/exp079"
RUN="$REPO/scripts/exp079_run.sh"
M="$R/models/az_v7/bbnet_mix16x9v9_gen48.onnx"
P="$R/runs/loopmix16x9v9/positions/contested_20x11_gen43.json"
PAR="${PARALLEL:-1}"
cp "$REPO/cfgs/chance_all_partial.toml" "$OUT/cap_mover.toml"
echo 'tie_break = "mover"' >> "$OUT/cap_mover.toml"
for i in 1 2; do
  "$RUN" drive-agree --bot-config "$OUT/cap_mover.toml" --model "$M" --vs-model "$M" --positions "$P" \
    --count 3 --parallel "$PAR" --out "$OUT/det2_mover_p${PAR}_$i.jsonl" 2>&1 | grep done
done
