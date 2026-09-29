#!/usr/bin/env bash
# Plan 049 companion: how much did the search's roll-model bugs cost? One net, two searches.
#
#   candidate: gen21 with the fixed search       (cfgs/exact_visits.toml)
#   opponent:  gen21 with the pre-fix search      (cfgs/legacy_chance_visits.toml)
#
# 200 games per board on 14x7 and 16x9, 500 visits, seats alternating. No training: the fixes are
# search-only, so this isolates what the wrong model (every armour break a casualty, every pass a
# fumble, harmless fouls, a horizon running through half time) cost at play time.
#
#   nohup systemd-inhibit --what=sleep:idle --who=exp049 --why="search-fix A/B" \
#       scripts/exp049_search_fix_ab.sh > /dev/null 2>&1 &
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp049"
MODEL="$REPO/models/az_v7/bbnet_mix16x9_gen21.onnx"
GAMES="${GAMES:-200}"
BOARDS=14x7,16x9
EVAL_PARALLEL_GAMES="${EVAL_PARALLEL_GAMES:-12}"

export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6
export CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"
WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_PORT=13337
HUB_URL="http://127.0.0.1:$HUB_PORT"
HUB_TOKEN_FILE="$HOME/.config/botbowl/hub.token"
NN_SOCKET=/tmp/bbnn-exp049.sock
PY="$REPO/train/.venv/bin/python"

mkdir -p "$OUT"
STATUS="$OUT/status.md"
status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { status "FATAL: $*"; exit 1; }

HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$NN_SOCKET"; }
trap cleanup EXIT INT TERM

git -C "$REPO" diff --quiet || die "dirty tree: commit before launching"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD), $GAMES per board on $BOARDS, exact vs legacy, both gen21"

"$HUB" serve --bind "0.0.0.0:$HUB_PORT" --token-file "$HUB_TOKEN_FILE" >> "$OUT/hub.log" 2>&1 &
HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 && break; sleep 1; done
"$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 || die "hub did not come up"

rm -f "$NN_SOCKET"
"$PY" "$REPO/scripts/nn_server.py" --socket "$NN_SOCKET" --device cuda --model "$MODEL" \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 &
NN_PID=$!
for _ in $(seq 120); do [ -S "$NN_SOCKET" ] && break; kill -0 "$NN_PID" 2>/dev/null || break; sleep 1; done
[ -S "$NN_SOCKET" ] || die "nn_server did not come up"
"$WORKER" --hub "ws://127.0.0.1:$HUB_PORT/ws" --token-file "$HUB_TOKEN_FILE" --name local \
    --parallel-games "$EVAL_PARALLEL_GAMES" --cache-dir "$OUT/worker-cache" --nn-server "$NN_SOCKET" \
    >> "$OUT/eval.worker.log" 2>&1 &
WORKER_PID=$!

SECONDS=0
"$HUB" job eval --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" \
    --evaluator nn --model "$MODEL" --bot-config "$REPO/cfgs/exact_visits.toml" \
    --mcts-iters 500 --games 30 --seed 0 --skip-fixed-rungs \
    --board-sizes "$BOARDS" --cells-per-player 26 \
    --vs-games "$GAMES" --vs-evaluator nn --vs-model "$MODEL" --vs-config "$REPO/cfgs/legacy_chance_visits.toml" \
    --per-game-out "$OUT/eval.games.jsonl" --out "$OUT/report.json" --wait > "$OUT/eval.log" 2>&1 \
    || die "eval failed — see eval.log"
status "eval done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report.json" 2>&1 | tr '\n' ' ' | cut -c1-900)"
status "done"
