#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# exp058: does strength on 16x9 follow exp057's convergence curve? On wide 16x9 roots (about 98
# legal moves), the decision did not move from 125 to 2000 descents and only started to at 4000.
# So: gen21 at 4000 real descents (cfgs/exact_iters.toml) against gen21 at the loop's 500 visits
# (about 255 descents), on drives from cfgs/positions/contested_16x9.json (screened with gen21),
# SPRT 0.5:0.55. 4000-descent trees are large; exp056 OOM-killed the worker at 14 streams of
# 2000-descent trees, so PARALLEL defaults to 4.
#
#   scripts/exp058_16x9_budget4000.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp058"; mkdir -p "$OUT"
GEN21="$REPO/models/az_v7/bbnet_mix16x9_gen21.onnx"
VISITS="$REPO/cfgs/exact_visits.toml"; ITERS="$REPO/cfgs/exact_iters.toml"
PARALLEL="${PARALLEL:-4}"; MEM_FLOOR_MB="${MEM_FLOOR_MB:-4096}"; BUDGET="${BUDGET:-4000}"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp058.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD), gen21 at $BUDGET descents vs 500 visits on 16x9 drives, $PARALLEL streams"
cargo build --release -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$GEN21" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games "$PARALLEL" --mem-floor-mb "$MEM_FLOOR_MB" --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
SECONDS=0
"$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$GEN21" \
    --bot-config "$ITERS" --mcts-iters "$BUDGET" --vs-config "$VISITS" --opponent-iters 500 \
    --games 30 --seed 58000 --skip-fixed-rungs --vs-evaluator nn --vs-model "$GEN21" \
    --positions "$REPO/cfgs/positions/contested_16x9.json" --sprt 0.5:0.55 --vs-games 2000 \
    --per-game-out "$OUT/eval.games.jsonl" --out "$OUT/report.json" --wait > "$OUT/eval.log" 2>&1 || die "eval"
status "done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report.json" 2>&1 | sed -E 's#/home/[^ ]*/models/az_v7/##g' | cut -c1-700)"
status "done"
