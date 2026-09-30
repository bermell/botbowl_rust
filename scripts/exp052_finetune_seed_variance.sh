#!/usr/bin/env bash
# Plan 049 / plan 032 #5: how much does one fine-tune vary with nothing but its seed?
# exp048's greedy arm (cq tau=100, warm from gen21, window gen20-22, 3 epochs, same prepared data)
# retrained with --seed 1 and --seed 2, each scored vs gen13 exactly as exp048/exp050 were: 200 per
# board on 14x7 and 16x9, legacy roll model both seats, --seed 0 games. The original (unseeded)
# run read 0.398; tau=20 read 0.464. If the three control draws spread by ~0.07, exp050's gap is
# noise.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
X48="$REPO/runs/exp048"; OUT="$REPO/runs/exp052"; mkdir -p "$OUT"
MODELS="$REPO/models/az_v7"; ANCHOR="$MODELS/anchor_mix16x9_gen13.onnx"; LEGACY="$REPO/cfgs/legacy_chance_visits.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp052.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
for s in 1 2; do
    [ -e "$OUT/.trained_s$s" ] && continue
    SECONDS=0
    "$PY" -m bbnn.train --data "$X48/prepared_greedy" --val-data "$X48/prepared_val" --seed "$s" \
        --epochs 3 --device auto --init "$MODELS/bbnet_mix16x9_gen21.pt" --lr 2e-4 \
        --value-weight 0.25 --per-drive-value-weight --select-on combined --eval-every 2500 \
        --out "$MODELS/exp052_ctl_s$s.pt" --onnx "$MODELS/exp052_ctl_s$s.onnx" > "$OUT/train_s$s.log" 2>&1 || die "train s$s"
    status "train seed $s done ($((SECONDS / 60)) min): $(grep -E '^restored' "$OUT/train_s$s.log")"
    touch "$OUT/.trained_s$s"
done
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$MODELS/exp052_ctl_s1.onnx" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 12 --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/eval.worker.log" 2>&1 & WORKER_PID=$!
declare -A C
for s in 1 2; do
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$MODELS/exp052_ctl_s$s.onnx" --bot-config "$LEGACY" \
        --mcts-iters 500 --games 30 --seed 0 --skip-fixed-rungs --board-sizes 14x7,16x9 --cells-per-player 26 \
        --vs-games 200 --vs-evaluator nn --vs-model "$ANCHOR" --vs-config "$LEGACY" \
        --per-game-out "$OUT/eval_s$s.games.jsonl" --out "$OUT/report_s$s.json" --wait > "$OUT/eval_s$s.log" 2>&1 & C[$s]=$!
done
status "evals submitted"
for s in 1 2; do wait "${C[$s]}" || die "eval s$s"; status "eval seed $s done: $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report_s$s.json" 2>&1 | tr '\n' ' ' | cut -c1-600)"; done
status "done"
