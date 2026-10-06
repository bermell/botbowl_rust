#!/usr/bin/env bash
# Plan 047 — is data volume the bottleneck? Train on the whole post-fix corpus of the
# loopmix16x9 plateau (gen12-22, eleven generations from nets of about equal strength)
# instead of the loop's three-generation window, and play the result against the
# frozen gen13 anchor exactly as the loop's benchmark does.
#
#   arm warm:    fine-tune gen21 (the champion) at the loop's lr 2e-4
#   arm scratch: random init at the loop's scratch lr 1e-3
#
# Both arms: train = shards 0-3,5,6 of gen12-22; val = gen22 shards 4,7 (in no training
# pool, and never trained on by gen21 either). gen11 and earlier are excluded: they were
# generated before 1458608 (second-half rerolls) and carry wrong reroll labels.
#
#   nohup systemd-inhibit --what=sleep:idle --who=exp047 --why="plan 047" \
#       scripts/exp047_wide_window.sh > /dev/null 2>&1 &
#   touch runs/exp047/STOP       # exits at the next phase boundary
#
# Each phase leaves a marker, so a relaunch resumes where it stopped.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/runs/loopmix16x9"
OUT="$REPO/runs/exp047"
MODELS="$REPO/models/az_v7"
ANCHOR="$MODELS/anchor_mix16x9_gen13.onnx"
WARM_INIT="$MODELS/bbnet_mix16x9_gen21.pt"
GENS="$(seq 12 22)"
TRAIN_SHARDS="0 1 2 3 5 6"
VAL_GEN=22
VAL_SHARDS="4 7"

# The loop's settings (launch_mix16x9.sh + train_loop.sh defaults), so the only
# difference from a production generation is the training pool.
EPOCHS="${EPOCHS:-3}"
PREPARE_TARGET_ARGS="--policy-target cq --tau 100 --value-blend 0.5"
TRAIN_TARGET_ARGS="--value-weight 0.25 --per-drive-value-weight --select-on combined --eval-every 2500"
EVAL_MCTS_ITERS=500
ANCHOR_GAMES=100
EVAL_BOARD_SIZES=14x7,16x9
export BLOOD_MCTS_BUDGET=visits
EVAL_PARALLEL_GAMES="${EVAL_PARALLEL_GAMES:-10}"   # box otherwise idle; mem_governor is the backstop

export CARGO_TARGET_DIR="$REPO/target/16x9"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"
WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_PORT=13337          # the NAT-forwarded port, so the laptop worker rejoins on its own
HUB_URL="http://127.0.0.1:$HUB_PORT"
HUB_TOKEN_FILE="$HOME/.config/botbowl/hub.token"
NN_SOCKET=/tmp/bbnn-exp047.sock
PY="$REPO/train/.venv/bin/python"

mkdir -p "$OUT"
STATUS="$OUT/status.md"
status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { status "FATAL: $*"; exit 1; }
check_stop() { [ -e "$OUT/STOP" ] && { status "STOP file present — exiting ($1)"; exit 0; }; return 0; }

HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() {
    for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    rm -f "$NN_SOCKET"
}
trap cleanup EXIT INT TERM

status "start: commit $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo -dirty), gens $(echo $GENS | tr ' ' ','), epochs $EPOCHS"

# ---- prepare --------------------------------------------------------------------
if [ ! -e "$OUT/.prepared" ]; then
    check_stop "before prepare"
    TRAIN_IN=""
    for g in $GENS; do
        for k in $TRAIN_SHARDS; do
            f="$SRC/$(printf 'gen%02d' "$g")/shard$k.jsonl"
            [ -s "$f" ] || die "missing $f"
            TRAIN_IN="$TRAIN_IN $f"
        done
    done
    VAL_IN=""
    for k in $VAL_SHARDS; do VAL_IN="$VAL_IN $SRC/gen$VAL_GEN/shard$k.jsonl"; done
    rm -rf "$OUT/prepared_train" "$OUT/prepared_val"
    SECONDS=0
    # shellcheck disable=SC2086
    "$PREPARE" --in $TRAIN_IN --out "$OUT/prepared_train" $PREPARE_TARGET_ARGS > "$OUT/prepare.log" 2>&1 \
        || die "prepare (train) failed — see prepare.log"
    # shellcheck disable=SC2086
    "$PREPARE" --in $VAL_IN --out "$OUT/prepared_val" $PREPARE_TARGET_ARGS >> "$OUT/prepare.log" 2>&1 \
        || die "prepare (val) failed — see prepare.log"
    status "prepare done ($((SECONDS / 60)) min): $(grep -c . <<< "$(echo $TRAIN_IN | tr ' ' '\n')") train shards; $(grep '^prepare done' "$OUT/prepare.log" | head -1 | cut -c1-120)"
    touch "$OUT/.prepared"
fi

# ---- train ----------------------------------------------------------------------
train_arm() {   # $1 = arm name, rest = init args
    local arm="$1"; shift
    local model="$MODELS/exp047_wide_$arm"
    [ -e "$OUT/.trained_$arm" ] && return 0
    check_stop "before train $arm"
    status "train $arm: $* ($EPOCHS epochs)"
    SECONDS=0
    # shellcheck disable=SC2086
    "$PY" -m bbnn.train --data "$OUT/prepared_train" --val-data "$OUT/prepared_val" \
        --epochs "$EPOCHS" --device auto "$@" $TRAIN_TARGET_ARGS \
        --out "$model.pt" --onnx "$model.onnx" > "$OUT/train_$arm.log" 2>&1 \
        || die "train $arm failed — see train_$arm.log"
    status "train $arm done ($((SECONDS / 60)) min): $(grep -E '^(restored|policy-only)' "$OUT/train_$arm.log" | tr '\n' ' ')"
    touch "$OUT/.trained_$arm"
}
train_arm warm --init "$WARM_INIT" --lr 2e-4
train_arm scratch --lr 1e-3
# gen21 on the same held-out set: the warm arm's epoch -1 line is exactly that.
status "gen21 on the gen22 val set: $(grep -m1 'warm-start baseline' "$OUT/train_warm.log")"

# ---- eval -----------------------------------------------------------------------
check_stop "before eval"
if ! "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1; then
    "$HUB" serve --bind "0.0.0.0:$HUB_PORT" --token-file "$HUB_TOKEN_FILE" >> "$OUT/hub.log" 2>&1 &
    HUB_PID=$!
    for _ in $(seq 30); do
        "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 && break
        sleep 1
    done
    "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 || die "hub did not come up"
fi
rm -f "$NN_SOCKET"
"$PY" "$REPO/scripts/nn_server.py" --socket "$NN_SOCKET" --device cuda \
    --model "$MODELS/exp047_wide_warm.onnx" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 &
NN_PID=$!
for _ in $(seq 120); do [ -S "$NN_SOCKET" ] && break; kill -0 "$NN_PID" 2>/dev/null || break; sleep 1; done
[ -S "$NN_SOCKET" ] || die "nn_server did not come up — see nn_server.log"
"$WORKER" --hub "ws://127.0.0.1:$HUB_PORT/ws" --token-file "$HUB_TOKEN_FILE" \
    --name local --parallel-games "$EVAL_PARALLEL_GAMES" --cache-dir "$OUT/worker-cache" \
    --nn-server "$NN_SOCKET" >> "$OUT/eval.worker.log" 2>&1 &
WORKER_PID=$!

# Both jobs are submitted at once so no stream idles between them; the same --seed 0
# as the loop's benchmark, so each arm plays the openings gen21 played.
declare -A CLIENT
for arm in warm scratch; do
    [ -e "$OUT/.evaluated_$arm" ] && continue
    "$HUB" job eval --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" \
        --evaluator nn --model "$MODELS/exp047_wide_$arm.onnx" \
        --mcts-iters "$EVAL_MCTS_ITERS" --games 30 --seed 0 --skip-fixed-rungs \
        --board-sizes "$EVAL_BOARD_SIZES" --cells-per-player 26 \
        --vs-games "$ANCHOR_GAMES" --vs-evaluator nn --vs-model "$ANCHOR" \
        --per-game-out "$OUT/eval_$arm.games.jsonl" \
        --out "$OUT/report_$arm.json" --wait > "$OUT/eval_$arm.log" 2>&1 &
    CLIENT[$arm]=$!
    status "eval $arm submitted: $ANCHOR_GAMES vs $(basename "$ANCHOR") on each of $EVAL_BOARD_SIZES at $EVAL_MCTS_ITERS iters"
done
SECONDS=0
for arm in "${!CLIENT[@]}"; do
    wait "${CLIENT[$arm]}" || die "eval $arm failed — see eval_$arm.log"
    status "eval $arm done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$OUT/report_$arm.json" 2>&1 | tr '\n' ' ' | cut -c1-900)"
    touch "$OUT/.evaluated_$arm"
done
status "done"
