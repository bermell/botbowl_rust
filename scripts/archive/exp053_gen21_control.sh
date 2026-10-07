#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# Is the ~0.11 "drop" below gen21 real, or was gen21's 0.512 a lucky 200-game draw? gen21 itself vs
# gen13, exactly as exp048/050/052's arms were scored: 200 per board on 14x7 and 16x9, 500 visits,
# legacy roll model both seats, --seed 0 (its first 100 per board replay the loop's benchmark).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp053"; mkdir -p "$OUT"
MODELS="$REPO/models/az_v7"; ANCHOR="$MODELS/anchor_mix16x9_gen13.onnx"; LEGACY="$REPO/cfgs/legacy_chance_visits.toml"
MODEL="$MODELS/bbnet_mix16x9_gen21.onnx"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp053.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD); waiting for exp052"
until grep -qE "^.{22}(done|FATAL)" "$REPO/runs/exp052/status.md"; do sleep 30; done
for _ in $(seq 60); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 || break; sleep 2; done
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$MODEL" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 14 --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/eval.worker.log" 2>&1 & WORKER_PID=$!
status "eval gen21 submitted: 200 vs gen13 on each of 14x7,16x9, legacy both seats"
SECONDS=0
"$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$MODEL" --bot-config "$LEGACY" \
    --mcts-iters 500 --games 30 --seed 0 --skip-fixed-rungs --board-sizes 14x7,16x9 --cells-per-player 26 \
    --vs-games 200 --vs-evaluator nn --vs-model "$ANCHOR" --vs-config "$LEGACY" \
    --per-game-out "$OUT/eval.games.jsonl" --out "$OUT/report.json" --wait > "$OUT/eval.log" 2>&1 || die "eval"
status "eval done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report.json" 2>&1 | tr '\n' ' ' | cut -c1-600)"
status "done"
