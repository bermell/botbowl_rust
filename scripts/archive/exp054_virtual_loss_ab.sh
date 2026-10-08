#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# The virtual-loss leak (2026-09-30): at one worker virtual loss has no job, and it leaks and buries
# the best root children. Same net (gen21) in both seats, exact roll model, 500 visits, 200 per board
# on 14x7 and 16x9: candidate virtual_loss 0 (cfgs/exact_visits_vl0.toml) vs the shipped 30
# (cfgs/exact_visits.toml).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp054"; mkdir -p "$OUT"
MODELS="$REPO/models/az_v7"; ANCHOR="$MODELS/anchor_mix16x9_gen13.onnx"; LEGACY="$REPO/cfgs/legacy_chance_visits.toml"
MODEL="$MODELS/bbnet_mix16x9_gen21.onnx"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp054.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
for _ in $(seq 60); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 || break; sleep 2; done
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$MODEL" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 14 --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/eval.worker.log" 2>&1 & WORKER_PID=$!
status "eval submitted: gen21 vl0 vs gen21 vl30, 200 per board on 14x7,16x9, exact model"
SECONDS=0
"$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$MODEL" --bot-config "$REPO/cfgs/exact_visits_vl0.toml" \
    --mcts-iters 500 --games 30 --seed 0 --skip-fixed-rungs --board-sizes 14x7,16x9 --cells-per-player 26 \
    --vs-games 200 --vs-evaluator nn --vs-model "$MODEL" --vs-config "$REPO/cfgs/exact_visits.toml" \
    --per-game-out "$OUT/eval.games.jsonl" --out "$OUT/report.json" --wait > "$OUT/eval.log" 2>&1 || die "eval"
status "eval done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report.json" 2>&1 | tr '\n' ' ' | cut -c1-600)"
status "done"
