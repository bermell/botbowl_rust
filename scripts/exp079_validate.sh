#!/usr/bin/env bash
# Plan 068 §3 / exp079: validate the divergence gate on real drive rungs (local `botbowl-ui eval`, the
# drive rung's own code path), same positions, same --seed, both bots on one preset.
#   ARM=tag CAND=gen45 REF=gen48 GATE=sprt|report|"" PRESET=chance_all_partial_det CAP=200 \
#     SET=.../contested_16x9_gen04g.json scripts/exp079_validate.sh
# Writes runs/exp079/val_<tag>/{report.json,eval.games.jsonl,eval.log}.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${MAIN_CHECKOUT:-/home/mattias/repos/botbowl_rust}"
RUN="$REPO/scripts/exp079_run.sh"
MD="$R/models/az_v7"
ARM="${ARM:?ARM tag}"
CAND="${CAND:-gen45}"
REF="${REF:-gen48}"
GATE="${GATE:-}"
PRESET="${PRESET:-chance_all_partial_det}"
CAP="${CAP:-200}"
SPRT="${SPRT-0.5:0.55}"   # SPRT= (empty): a fixed-size rung, replayed offline by scripts/gate_replay.py
PAR="${PARALLEL:-16}"
SET="${SET:-$R/runs/loopmix16x9g/positions/contested_16x9_gen04g.json}"
OUT="$R/runs/exp079/val_$ARM"
mkdir -p "$OUT"
[ -s "$OUT/report.json" ] && { echo "skip $OUT (done)"; exit 0; }
rm -f "$OUT/eval.games.jsonl"
echo "== val $ARM: $CAND vs $REF, $PRESET, gate '${GATE:-off}', $(basename "$SET"), cap $CAP, $(date +%T)"
"$RUN" eval --evaluator nn --model "$MD/bbnet_mix16x9v9_$CAND.onnx" \
  --vs-evaluator nn --vs-model "$MD/bbnet_mix16x9v9_$REF.onnx" \
  --bot-config "$REPO/cfgs/$PRESET.toml" --mcts-iters 1000 \
  --skip-lectures --skip-fixed-rungs --positions "$SET" --seed 0 \
  ${SPRT:+--sprt "$SPRT"} --vs-games "$CAP" --parallel-games "$PAR" \
  ${GATE:+--drive-gate "$GATE"} \
  --per-game-out "$OUT/eval.games.jsonl" --out "$OUT/report.json" > "$OUT/eval.log" 2>&1
echo "   exit $? $(date +%T)"
grep "ladder" "$OUT/eval.log" | tail -1
python3 "$REPO/scripts/eval_summary.py" "$OUT/report.json" 2>/dev/null | tail -2
