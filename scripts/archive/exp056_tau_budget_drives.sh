#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# exp056: plan 049's tau and search-budget questions, re-asked with plan 051's tools. Drives from
# positions contested for gen21 (cfgs/positions/contested_*.json, screened with gen21, the parent
# in every arm) are judged by SPRT 0.5:0.55, and a drive H1 is confirmed with P1 on full games.
#
#   A. tau, both vs gen21 at the loop's search (exact_visits, 500 visits):
#        tau100  exp055's net, gen01's corpus re-prepared at cq tau 100, fine-tuned from gen21
#        tau20   gen01 itself (the same corpus at tau 20)
#   B. budget, the same net under two settings: gen21 at N real descents (cfgs/exact_iters.toml)
#      vs gen21 at 500 visits (about 255 descents), for N = 500, 1000 and 2000.
#
# Resumable: a finished arm's report.json is kept and skipped.
#
#   scripts/exp056_tau_budget_drives.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp056"; mkdir -p "$OUT"
M="$REPO/models/az_v7"; GEN21="$M/bbnet_mix16x9_gen21.onnx"
SETS="$REPO/cfgs/positions/contested_14x7.json,$REPO/cfgs/positions/contested_16x9.json"
VISITS="$REPO/cfgs/exact_visits.toml"; ITERS="$REPO/cfgs/exact_iters.toml"
SPRT="0.5:0.55"; DRIVE_CAP=2000; GAME_CAP=400
# The 2000-descent arm's trees are large: 14 parallel games OOM-killed the worker (13.7 GB, 15 GB
# box) on 2026-10-02. Fewer streams and a bigger reserve for the worker's memory governor.
PARALLEL="${PARALLEL:-6}"; MEM_FLOOR_MB="${MEM_FLOOR_MB:-4096}"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp056.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
cargo build --release -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"

"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$GEN21" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games "$PARALLEL" --mem-floor-mb "$MEM_FLOOR_MB" --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!

# arm NAME MODEL SEED [job eval args...] — one eval job, backgrounded, pid in J[NAME].
declare -A J
arm() {
    local name="$1" model="$2" seed="$3"; shift 3
    local out="$OUT/$name"
    [ -e "$out/report.json" ] && return 0
    rm -rf "$out"; mkdir -p "$out"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$model" \
        --games 30 --seed "$seed" --skip-fixed-rungs --vs-evaluator nn --vs-model "$GEN21" "$@" \
        --per-game-out "$out/eval.games.jsonl" --out "$out/report.json" --wait > "$out/eval.log" 2>&1 &
    J[$name]=$!
}
collect() {
    local name
    for name in "$@"; do
        [ -n "${J[$name]:-}" ] || continue
        wait "${J[$name]}" || die "$name failed, see $OUT/$name/eval.log"
        status "$name: $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/$name/report.json" 2>&1 | tr '\n' ' ' | sed -E 's#/home/[^ ]*/models/az_v7/##g' | cut -c1-900)"
    done
}
# Any rung of this arm's report decided H1.
any_h1() { "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if any((x.get("sprt") or {}).get("verdict")=="H1" for x in r["ladder"]) else 1)' "$1"; }

DRIVES=(--positions "$SETS" --sprt "$SPRT" --vs-games "$DRIVE_CAP")
SECONDS=0
arm tau100 "$M/exp055_gen01data_tau100.onnx" 56000 --bot-config "$VISITS" --mcts-iters 500 "${DRIVES[@]}"
arm tau20 "$M/bbnet_mix16x9vl0_gen01.onnx" 56000 --bot-config "$VISITS" --mcts-iters 500 "${DRIVES[@]}"
for n in 500 1000 2000; do
    arm "iters$n" "$GEN21" 56000 --bot-config "$ITERS" --mcts-iters "$n" \
        --vs-config "$VISITS" --opponent-iters 500 "${DRIVES[@]}"
done
status "drive arms submitted: tau100, tau20 vs gen21; gen21 at 500/1000/2000 descents vs gen21 at 500 visits"
collect tau100 tau20 iters500 iters1000 iters2000
status "drives done ($((SECONDS / 60)) min)"

# P1 confirmation on full games for every arm with a drive H1: SPRT on paired games, capped at
# 200 per board (the gold block's size), on the same two boards.
SECONDS=0
GAMES=(--board-sizes 14x7,16x9 --cells-per-player 26 --sprt "$SPRT" --vs-games $((GAME_CAP / 2)))
CONFIRM=()
for name in tau100 tau20 iters500 iters1000 iters2000; do
    any_h1 "$OUT/$name/report.json" || continue
    case "$name" in
        tau100) arm p1_tau100 "$M/exp055_gen01data_tau100.onnx" 57000 --bot-config "$VISITS" --mcts-iters 500 "${GAMES[@]}" ;;
        tau20) arm p1_tau20 "$M/bbnet_mix16x9vl0_gen01.onnx" 57000 --bot-config "$VISITS" --mcts-iters 500 "${GAMES[@]}" ;;
        iters*) arm "p1_$name" "$GEN21" 57000 --bot-config "$ITERS" --mcts-iters "${name#iters}" \
            --vs-config "$VISITS" --opponent-iters 500 "${GAMES[@]}" ;;
    esac
    CONFIRM+=("p1_$name")
done
if [ "${#CONFIRM[@]}" -gt 0 ]; then
    status "confirming on full games: ${CONFIRM[*]}"
    collect "${CONFIRM[@]}"
    status "confirmation done ($((SECONDS / 60)) min)"
else
    status "no drive H1, so nothing to confirm"
fi
status "done"
