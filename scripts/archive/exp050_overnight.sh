#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# Overnight after exp049, two independent questions sharing the box.
#
# (A) Plan 049 finding 1 — does a sharper policy target learn what cq τ=100 cannot?
#     Two arms, each exactly exp048's greedy arm except the target: warm from gen21, lr 2e-4,
#     3 epochs, window gen20 + gen21 + gen22, val = shards 4,7 of gen22 and gen22x under the
#     arm's own target. Arms: cq τ=20, and visit counts. Control: exp048's greedy arm (cq τ=100),
#     0.398 over the same 400 seeds. Eval: 200 per board vs gen13 on 14x7 and 16x9 with the
#     **legacy** roll model on both seats, so the control's numbers stay comparable.
#
# (B) exp049 follow-up — does the fixed roll model lose to the legacy one without a net? The
#     heuristic evaluator (no corpus-trained prior or value), exact vs legacy, 200 games on 14x7.
#     If exact still loses, suspect the fix; if not, gen21's legacy-trained net is the mismatch.
#
#   nohup systemd-inhibit --what=sleep:idle --who=exp050 --why="overnight" \
#       scripts/exp050_overnight.sh > /dev/null 2>&1 &
#   touch runs/exp050/STOP
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/runs/loopmix16x9"
X48="$REPO/runs/exp048"
OUT="$REPO/runs/exp050"
MODELS="$REPO/models/az_v7"
CHAMP="$MODELS/bbnet_mix16x9_gen21"
ANCHOR="$MODELS/anchor_mix16x9_gen13.onnx"
LEGACY="$REPO/cfgs/legacy_chance_visits.toml"
EXACT="$REPO/cfgs/exact_visits.toml"
EVAL_PARALLEL_GAMES="${EVAL_PARALLEL_GAMES:-12}"

export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6
export CARGO_TARGET_DIR="$REPO/target/16x9"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"
WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_PORT=13337
HUB_URL="http://127.0.0.1:$HUB_PORT"
HUB_TOKEN_FILE="$HOME/.config/botbowl/hub.token"
NN_SOCKET=/tmp/bbnn-exp050.sock
PY="$REPO/train/.venv/bin/python"

mkdir -p "$OUT"
STATUS="$OUT/status.md"
status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { status "FATAL: $*"; exit 1; }
check_stop() { [ -e "$OUT/STOP" ] && { status "STOP file present — exiting ($1)"; exit 0; }; return 0; }

HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$NN_SOCKET"; }
trap cleanup EXIT INT TERM

git -C "$REPO" diff --quiet || die "dirty tree: commit before launching"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD); waiting for exp049"
until grep -qE "^.{22}(done|FATAL)" "$REPO/runs/exp049/status.md" 2>/dev/null; do sleep 60; done
status "exp049 finished; starting"
# exp049's trap takes its hub down after it writes "done"; wait for the port to free up.
for _ in $(seq 60); do "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 || break; sleep 2; done

# ---- hub + worker (the sidecar starts once the arms are trained; the heuristic needs none) ----
"$HUB" serve --bind "0.0.0.0:$HUB_PORT" --token-file "$HUB_TOKEN_FILE" >> "$OUT/hub.log" 2>&1 &
HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 && break; sleep 1; done
"$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 || die "hub did not come up"
start_worker() {   # $1 = extra args
    # shellcheck disable=SC2086
    "$WORKER" --hub "ws://127.0.0.1:$HUB_PORT/ws" --token-file "$HUB_TOKEN_FILE" --name local \
        --parallel-games "$EVAL_PARALLEL_GAMES" --cache-dir "$OUT/worker-cache" $1 >> "$OUT/eval.worker.log" 2>&1 &
    WORKER_PID=$!
}
start_worker ""

# ---- (B) heuristic exact vs legacy, on the CPU while (A) trains on the GPU ----
HEUR_CLIENT=""
if [ ! -e "$OUT/.heuristic_done" ]; then
    "$HUB" job eval --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" \
        --evaluator heuristic --bot-config "$EXACT" --mcts-iters 500 --games 30 --seed 0 --skip-fixed-rungs \
        --board-sizes 14x7 --cells-per-player 26 \
        --vs-games 200 --vs-evaluator heuristic --vs-config "$LEGACY" \
        --per-game-out "$OUT/heur.games.jsonl" --out "$OUT/report_heur.json" --wait > "$OUT/eval_heur.log" 2>&1 &
    HEUR_CLIENT=$!
    status "heuristic exact vs legacy submitted: 200 games on 14x7"
fi

# ---- (A) prepare + train the two target arms ----
shards() {   # $1 = kind, rest = gen dirs
    local kind="$1"; shift; local ks d k out=""
    [ "$kind" = train ] && ks="0 1 2 3 5 6" || ks="4 7"
    for d in "$@"; do for k in $ks; do out="$out $d/shard$k.jsonl"; done; done
    echo "$out"
}
arm_target() { case "$1" in tau20) echo "--policy-target cq --tau 20";; visits) echo "--policy-target visits";; esac; }
for arm in tau20 visits; do
    [ -e "$OUT/.trained_$arm" ] && continue
    check_stop "before $arm"
    T="$(arm_target $arm) --value-blend 0.5"
    SECONDS=0
    rm -rf "$OUT/prep_${arm}_train" "$OUT/prep_${arm}_val"
    # shellcheck disable=SC2086,SC2046
    "$PREPARE" --in $(shards train "$SRC/gen20" "$SRC/gen21" "$SRC/gen22") --out "$OUT/prep_${arm}_train" $T >> "$OUT/prepare.log" 2>&1 \
        || die "prepare $arm train failed"
    # shellcheck disable=SC2086,SC2046
    "$PREPARE" --in $(shards val "$SRC/gen22" "$X48/gen22x") --out "$OUT/prep_${arm}_val" $T >> "$OUT/prepare.log" 2>&1 \
        || die "prepare $arm val failed"
    status "train $arm ($T): warm from gen21, lr 2e-4, 3 epochs"
    "$PY" -m bbnn.train --data "$OUT/prep_${arm}_train" --val-data "$OUT/prep_${arm}_val" \
        --epochs 3 --device auto --init "$CHAMP.pt" --lr 2e-4 \
        --value-weight 0.25 --per-drive-value-weight --select-on combined --eval-every 2500 \
        --out "$MODELS/exp050_$arm.pt" --onnx "$MODELS/exp050_$arm.onnx" > "$OUT/train_$arm.log" 2>&1 \
        || die "train $arm failed"
    status "train $arm done ($((SECONDS / 60)) min): $(grep -E '^(restored|policy-only)' "$OUT/train_$arm.log" | tr '\n' ' ')"
    touch "$OUT/.trained_$arm"
    rm -rf "$OUT/prep_${arm}_train"
done

if [ -n "$HEUR_CLIENT" ]; then
    wait "$HEUR_CLIENT" || die "heuristic eval failed — see eval_heur.log"
    status "heuristic exact vs legacy done: $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report_heur.json" 2>&1 | tr '\n' ' ' | cut -c1-600)"
    touch "$OUT/.heuristic_done"
fi

# ---- (A) eval both arms, legacy roll model on both seats ----
check_stop "before eval"
kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null; WORKER_PID=""
rm -f "$NN_SOCKET"
"$PY" "$REPO/scripts/nn_server.py" --socket "$NN_SOCKET" --device cuda --model "$MODELS/exp050_tau20.onnx" \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 &
NN_PID=$!
for _ in $(seq 120); do [ -S "$NN_SOCKET" ] && break; kill -0 "$NN_PID" 2>/dev/null || break; sleep 1; done
[ -S "$NN_SOCKET" ] || die "nn_server did not come up"
start_worker "--nn-server $NN_SOCKET"
declare -A CLIENT
for arm in tau20 visits; do
    [ -e "$OUT/.evaluated_$arm" ] && continue
    "$HUB" job eval --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" \
        --evaluator nn --model "$MODELS/exp050_$arm.onnx" --bot-config "$LEGACY" \
        --mcts-iters 500 --games 30 --seed 0 --skip-fixed-rungs \
        --board-sizes 14x7,16x9 --cells-per-player 26 \
        --vs-games 200 --vs-evaluator nn --vs-model "$ANCHOR" --vs-config "$LEGACY" \
        --per-game-out "$OUT/eval_$arm.games.jsonl" --out "$OUT/report_$arm.json" --wait > "$OUT/eval_$arm.log" 2>&1 &
    CLIENT[$arm]=$!
    status "eval $arm submitted: 200 vs gen13 on each of 14x7,16x9, legacy roll model both seats"
done
SECONDS=0
for arm in "${!CLIENT[@]}"; do
    wait "${CLIENT[$arm]}" || die "eval $arm failed — see eval_$arm.log"
    status "eval $arm done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report_$arm.json" 2>&1 | tr '\n' ' ' | cut -c1-900)"
    touch "$OUT/.evaluated_$arm"
done
status "done"
