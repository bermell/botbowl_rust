#!/usr/bin/env bash
# The origin of runs/loopmix16x9vl0's curve: gen21 vs the gen13 anchor under the loop's own search
# (exact roll model, virtual-loss fix, 500 visits), 200 per board on 14x7 and 16x9. Everything
# measured before 2026-09-30 used the leaking search, so none of it is this curve's scale. Waits for
# the loop's hub and submits alongside generation 1, like the loop's own overlapped benchmarks.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/loopmix16x9vl0/baseline_gen21"; mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9" BLOOD_MCTS_BUDGET=visits
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"
M="$REPO/models/az_v7"
until "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1; do sleep 30; done
"$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$M/bbnet_mix16x9_gen21.onnx" \
    --mcts-iters 500 --games 30 --seed 0 --skip-fixed-rungs --board-sizes 14x7,16x9 --cells-per-player 26 \
    --vs-games 200 --vs-evaluator nn --vs-model "$M/anchor_mix16x9_gen13.onnx" \
    --per-game-out "$OUT/eval.games.jsonl" --out "$OUT/report.json" --wait > "$OUT/eval.log" 2>&1
echo "exit $?" >> "$OUT/eval.log"
