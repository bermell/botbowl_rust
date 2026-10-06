#!/usr/bin/env bash
# Why every loopmix16x9vl0 generation reads ~0.09 below gen21 (0.393 vs 0.485 vs gen13, gen01-04
# pooled). Hypothesis: the fixed search is too small to teach. The virtual-loss fix cut real descents
# per decision from ~330 to ~255 under the same "500 visits" budget, the root fan on 16x9 runs to 130,
# and cq tau 20 leans on Q values from one or two descents per child.
#
#   A. tau: gen01's own corpus re-prepared at tau 100, fine-tuned from gen21 exactly as gen01 was
#      (only tau differs), plus gen21 itself as the control on the same --seed 0 games.
#   B. budget: gen21 at 500 / 1000 / 2000 real descents (cfgs/exact_iters.toml) against gen13 held
#      at the loop's 500 visits (~255 descents). The 500-visit point is A's control, pooled with
#      runs/loopmix16x9vl0/baseline_gen21.
#
# Every rung is 200 per board on 14x7 and 16x9. Resumable: markers skip finished steps.
#
#   scripts/exp055_tau_and_budget.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp055"; mkdir -p "$OUT"
RUN="$REPO/runs/loopmix16x9vl0"; G01="$RUN/gen01"
MODELS="$REPO/models/az_v7"; ANCHOR="$MODELS/anchor_mix16x9_gen13.onnx"; GEN21="$MODELS/bbnet_mix16x9_gen21"
TAU100="$MODELS/exp055_gen01data_tau100"
ITERS_CFG="$REPO/cfgs/exact_iters.toml"; VISITS_CFG="$REPO/cfgs/exact_visits.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9" BLOOD_MCTS_BUDGET=visits
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp055.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
cargo build --release -p botbowl-ui -p botbowl-nn -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"

# ---- A. tau 100 on gen01's corpus -------------------------------------------------------------
# The loop's split (TRAIN_SHARDS / VAL_SHARDS) and its prepare + train flags, tau aside.
if [ ! -e "$OUT/.prepared" ]; then
    T=""; V=""; for k in 0 1 2 3 5 6; do T="$T $G01/shard$k.jsonl"; done; for k in 4 7; do V="$V $G01/shard$k.jsonl"; done
    # shellcheck disable=SC2086
    "$PREPARE" --in $T --out "$OUT/prepared_train" --policy-target cq --tau 100 --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 || die "prepare train"
    # shellcheck disable=SC2086
    "$PREPARE" --in $V --out "$OUT/prepared_val" --policy-target cq --tau 100 --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 || die "prepare val"
    touch "$OUT/.prepared"; status "prepare done (gen01 shards, cq tau 100)"
fi
if [ ! -e "$OUT/.trained" ]; then
    SECONDS=0
    "$PY" -m bbnn.train --data "$OUT/prepared_train" --val-data "$OUT/prepared_val" \
        --epochs 3 --device auto --init "$GEN21.pt" --lr 2e-4 --select-on combined --eval-every 2500 \
        --value-weight 0.25 --per-drive-value-weight \
        --out "$TAU100.pt" --onnx "$TAU100.onnx" > "$OUT/train.log" 2>&1 || die "train"
    status "train done ($((SECONDS / 60)) min): $(grep -E '^restored' "$OUT/train.log")"
    rm -rf "$OUT/prepared_train" "$OUT/prepared_val"
    touch "$OUT/.trained"
fi

"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$GEN21.onnx" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 14 --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/eval.worker.log" 2>&1 & WORKER_PID=$!

# eval NAME MODEL [extra job-eval args...] — 200 per board vs gen13, backgrounded; pid in E[NAME].
declare -A E
eval_job() {
    local name="$1" model="$2"; shift 2
    [ -e "$OUT/.eval_$name" ] && return 0
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$model" \
        --games 30 --seed 0 --skip-fixed-rungs --board-sizes 14x7,16x9 --cells-per-player 26 \
        --vs-games 200 --vs-evaluator nn --vs-model "$ANCHOR" "$@" \
        --per-game-out "$OUT/eval_$name.games.jsonl" --out "$OUT/report_$name.json" --wait > "$OUT/eval_$name.log" 2>&1 &
    E[$name]=$!
}
collect() {
    local name
    for name in "$@"; do
        [ -e "$OUT/.eval_$name" ] && continue
        wait "${E[$name]}" || die "eval $name"
        touch "$OUT/.eval_$name"
        status "eval $name done: $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report_$name.json" 2>&1 | tr '\n' ' ' | cut -c1-700)"
    done
}

SECONDS=0
eval_job tau100 "$TAU100.onnx" --mcts-iters 500
eval_job gen21_visits500 "$GEN21.onnx" --mcts-iters 500
status "A submitted: tau100 and gen21 (control), 500 visits both seats"
collect tau100 gen21_visits500
status "A done ($((SECONDS / 60)) min)"

# ---- B. gen21's own budget, gen13 fixed at 500 visits ------------------------------------------
SECONDS=0
for b in 500 1000 2000; do
    eval_job "gen21_iters$b" "$GEN21.onnx" --mcts-iters "$b" --bot-config "$ITERS_CFG" \
        --opponent-iters 500 --vs-config "$VISITS_CFG"
done
status "B submitted: gen21 at 500/1000/2000 descents vs gen13 at 500 visits"
collect gen21_iters500 gen21_iters1000 gen21_iters2000
status "B done ($((SECONDS / 60)) min)"
status "done"
