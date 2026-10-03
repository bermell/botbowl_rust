#!/usr/bin/env bash
# exp060b: plan 053 — the Gumbel halving with a floor on its Q normalisation range. exp060's
# convergence probe showed pure min-max following noise in Q (the reference's move, almost always
# the prior's favourite, searched and then dropped in 40-46% of roots). Same probe, floors of 50,
# 200 and 1000 Q points (±1000 = a touchdown), scored against exp057's 16000-descent references.
# Builds into its own target dir so a running exp060 keeps its binaries.
#
#   scripts/exp060b_gumbel_q_floor.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp060/qfloor"; mkdir -p "$OUT/presets"
GEN21="$REPO/models/az_v7/bbnet_mix16x9_gen21.onnx"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9b"
unset BLOOD_MCTS_BUDGET
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; SOCK=/tmp/bbnn-exp060b.sock; PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
NN_PID=""; cleanup() { [ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"; }; trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || die "build"
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$GEN21" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
SECONDS=0
for f in 50 200 1000; do
    printf 'budget_mode = "iterations"\ngumbel_m = 16\ngumbel_q_floor = %s\n' "$f" > "$OUT/presets/gumbel16_f$f.toml"
    for board in 14x7/4 16x9/6; do
        for adv in 0 1; do
            tag="f${f}_${board%%/*}_a$adv"
            [ -s "$OUT/conv_$tag.jsonl" ] && continue
            "$UI" convergence --states 60 --repeats 3 --budgets 250,500,1000,2000 --board "$board" --advance "$adv" \
                --evaluator nn --model "$GEN21" --nn-server "$SOCK" --parallel 8 --bot-config "$OUT/presets/gumbel16_f$f.toml" \
                --seed $(( 91000000 + adv * 100000 )) --out "$OUT/conv_$tag.jsonl" > "$OUT/conv_$tag.log" 2>&1 || die "$tag"
        done
    done
    status "floor $f done ($((SECONDS / 60)) min)"
done
for b in 14x7 16x9; do
    "$PY" "$REPO/scripts/convergence_topk.py" "$REPO"/runs/exp057/conv_${b}_a*.jsonl "$REPO"/runs/exp060/conv_gumbel_${b}_a*.jsonl \
        "$OUT"/conv_f*_${b}_a*.jsonl > "$OUT/summary_$b.txt" 2>&1 || die "summary $b"
done
status "done: summaries in $OUT/summary_{14x7,16x9}.txt"
