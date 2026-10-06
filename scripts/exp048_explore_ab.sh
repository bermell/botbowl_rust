#!/usr/bin/env bash
# Plan 048 — does self-play exploration make a better corpus? One-generation A/B off gen21.
#
#   greedy arm: runs/loopmix16x9/gen22 — gen21's own greedy generation, already on disk
#   noisy arm:  runs/exp048/gen22x   — gen21 again, same seeds and board draws, with
#               --explore-noise 0.25 --explore-alpha 10 --explore-sample-moves 2
#
# Both arms then take the loop's gen22 train step exactly (warm from gen21, lr 2e-4, 3 epochs,
# window gen20 + gen21 + the arm's corpus) against one common val set (shards 4,7 of both
# corpora), and play 200 games per board vs the gen13 anchor on 14x7 and 16x9.
#
#   nohup systemd-inhibit --what=sleep:idle --who=exp048 --why="plan 048" \
#       scripts/exp048_explore_ab.sh > /dev/null 2>&1 &
#   touch runs/exp048/STOP       # exits at the next phase boundary
#
# Each phase leaves a marker, so a relaunch resumes where it stopped.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/runs/loopmix16x9"
OUT="$REPO/runs/exp048"
GENX="$OUT/gen22x"
MODELS="$REPO/models/az_v7"
CHAMP="$MODELS/bbnet_mix16x9_gen21.onnx"
ANCHOR="$MODELS/anchor_mix16x9_gen13.onnx"
TRAIN_SHARDS="0 1 2 3 5 6"
VAL_SHARDS="4 7"

EXPLORE_ARGS="--explore-noise 0.25 --explore-alpha 10 --explore-sample-moves 2 --explore-temperature 1"
# The loop's gen22 generate call (launch_mix16x9.sh + train_loop.sh), so the only difference
# from runs/loopmix16x9/gen22 is EXPLORE_ARGS. Same seed base -> same random starts and boards.
SEED_BASE=$((10000000 + 22 * 1000000))
GAMES_PER_SHARD=600
MCTS_ITERS=500
SIZE_ARGS="--size-centre $(cat "$SRC/size_centre.txt") --size-temperature 0.3 --size-floor 0.2 \
--size-aspect 1.5-2.8 --size-max-area 144 --size-min-area 70 --cells-per-player 26"
export BLOOD_MCTS_BUDGET=visits
GEN_PARALLEL_GAMES=16

EPOCHS=3
PREPARE_TARGET_ARGS="--policy-target cq --tau 100 --value-blend 0.5"
TRAIN_TARGET_ARGS="--value-weight 0.25 --per-drive-value-weight --select-on combined --eval-every 2500"
EVAL_MCTS_ITERS=500
ANCHOR_GAMES=200
EVAL_BOARD_SIZES=14x7,16x9
EVAL_PARALLEL_GAMES="${EVAL_PARALLEL_GAMES:-12}"   # box otherwise idle; mem_governor is the backstop

export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6
export CARGO_TARGET_DIR="$REPO/target/16x9"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"
WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_PORT=13337
HUB_URL="http://127.0.0.1:$HUB_PORT"
HUB_TOKEN_FILE="$HOME/.config/botbowl/hub.token"
NN_SOCKET=/tmp/bbnn-exp048.sock    # AF_UNIX paths are capped at ~108 bytes
PY="$REPO/train/.venv/bin/python"

mkdir -p "$OUT" "$GENX"
STATUS="$OUT/status.md"
status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
die() { status "FATAL: $*"; exit 1; }
check_stop() { [ -e "$OUT/STOP" ] && { status "STOP file present — exiting ($1)"; exit 0; }; return 0; }

HUB_PID="" NN_PID="" WORKER_PID=""
worker_stop() { [ -n "$WORKER_PID" ] && { kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null; }; WORKER_PID=""; }
nn_stop() { [ -n "$NN_PID" ] && { kill "$NN_PID" 2>/dev/null; wait "$NN_PID" 2>/dev/null; }; NN_PID=""; rm -f "$NN_SOCKET"; }
cleanup() { worker_stop; nn_stop; [ -n "$HUB_PID" ] && { kill "$HUB_PID" 2>/dev/null; wait "$HUB_PID" 2>/dev/null; }; }
trap cleanup EXIT INT TERM

hub_up() {
    "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 && return 0
    "$HUB" serve --bind "0.0.0.0:$HUB_PORT" --token-file "$HUB_TOKEN_FILE" >> "$OUT/hub.log" 2>&1 &
    HUB_PID=$!
    for _ in $(seq 30); do
        "$HUB" status --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" > /dev/null 2>&1 && return 0
        sleep 1
    done
    die "hub did not come up — see hub.log"
}
# $1 = model the sidecar preloads, $2 = parallel games, $3 = worker log
serve_and_work() {
    rm -f "$NN_SOCKET"
    "$PY" "$REPO/scripts/nn_server.py" --socket "$NN_SOCKET" --device cuda \
        --model "$1" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 &
    NN_PID=$!
    for _ in $(seq 120); do [ -S "$NN_SOCKET" ] && break; kill -0 "$NN_PID" 2>/dev/null || break; sleep 1; done
    [ -S "$NN_SOCKET" ] || die "nn_server did not come up — see nn_server.log"
    "$WORKER" --hub "ws://127.0.0.1:$HUB_PORT/ws" --token-file "$HUB_TOKEN_FILE" \
        --name local --parallel-games "$2" --cache-dir "$OUT/worker-cache" \
        --nn-server "$NN_SOCKET" >> "$3" 2>&1 &
    WORKER_PID=$!
}

status "start: commit $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo -dirty), explore: $EXPLORE_ARGS"
hub_up

# ---- 1. generate the noisy arm ----------------------------------------------------------------
if [ ! -e "$GENX/.generated" ]; then
    check_stop "before generate"
    serve_and_work "$CHAMP" "$GEN_PARALLEL_GAMES" "$GENX/generate.worker.log"
    SECONDS=0
    status "generate gen22x: 8x$GAMES_PER_SHARD games from $(basename "$CHAMP"), seed base $SEED_BASE, $EXPLORE_ARGS"
    # shellcheck disable=SC2086
    "$HUB" job generate --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" \
        --mode random-start --games "$GAMES_PER_SHARD" \
        --seed-base "$SEED_BASE" --shard-seed-stride 100000 \
        --mcts-iters "$MCTS_ITERS" --evaluator nn --model "$CHAMP" \
        $SIZE_ARGS $EXPLORE_ARGS \
        --shards "0 1 2 3 4 5 6 7" --heuristic-shards "" \
        --truncate --out-dir "$GENX" --wait > "$GENX/generate.log" 2>&1 \
        || die "generate failed — see gen22x/generate.log"
    worker_stop
    nn_stop
    grep -q NN_SERVER_FALLBACK "$GENX/generate.worker.log" && status "WARN: the worker fell back to tract"
    status "generate done ($((SECONDS / 60)) min): $(cat "$GENX"/shard*.jsonl | wc -l) games"
    for arm_dir in "$SRC/gen22" "$GENX"; do
        status "  $(basename "$arm_dir") corpus: $("$PY" "$REPO/scripts/td_rate.py" "$arm_dir" 2>&1 | head -1 | cut -c1-160)"
    done
    touch "$GENX/.generated"
fi

# ---- 2. prepare -------------------------------------------------------------------------------
shards() {   # $1 = kind (train|val), rest = generation dirs
    local kind="$1"; shift
    local ks d k out=""
    [ "$kind" = train ] && ks="$TRAIN_SHARDS" || ks="$VAL_SHARDS"
    for d in "$@"; do for k in $ks; do
        [ -s "$d/shard$k.jsonl" ] || die "missing $d/shard$k.jsonl"
        out="$out $d/shard$k.jsonl"
    done; done
    echo "$out"
}
prepare_dir() {   # $1 = out dir, rest = shard files
    local out="$1"; shift
    rm -rf "$out"
    # shellcheck disable=SC2068
    "$PREPARE" --in $@ --out "$out" $PREPARE_TARGET_ARGS >> "$OUT/prepare.log" 2>&1 || die "prepare $out failed"
}
if [ ! -e "$OUT/.prepared" ]; then
    check_stop "before prepare"
    SECONDS=0
    prepare_dir "$OUT/prepared_val" $(shards val "$SRC/gen22" "$GENX")
    prepare_dir "$OUT/prepared_greedy" $(shards train "$SRC/gen20" "$SRC/gen21" "$SRC/gen22")
    prepare_dir "$OUT/prepared_noisy" $(shards train "$SRC/gen20" "$SRC/gen21" "$GENX")
    status "prepare done ($((SECONDS / 60)) min): $(grep '^prepare done' "$OUT/prepare.log" | cut -c1-90 | tr '\n' '|')"
    touch "$OUT/.prepared"
fi

# ---- 3. train ---------------------------------------------------------------------------------
train_arm() {   # $1 = arm
    local arm="$1" model="$MODELS/exp048_$1"
    [ -e "$OUT/.trained_$arm" ] && return 0
    check_stop "before train $arm"
    status "train $arm: warm from $(basename "${CHAMP%.onnx}.pt") at lr 2e-4, $EPOCHS epochs"
    SECONDS=0
    # shellcheck disable=SC2086
    "$PY" -m bbnn.train --data "$OUT/prepared_$arm" --val-data "$OUT/prepared_val" \
        --epochs "$EPOCHS" --device auto --init "${CHAMP%.onnx}.pt" --lr 2e-4 $TRAIN_TARGET_ARGS \
        --out "$model.pt" --onnx "$model.onnx" > "$OUT/train_$arm.log" 2>&1 \
        || die "train $arm failed — see train_$arm.log"
    status "train $arm done ($((SECONDS / 60)) min): $(grep -E '^(restored|policy-only)' "$OUT/train_$arm.log" | tr '\n' ' ')"
    touch "$OUT/.trained_$arm"
}
train_arm greedy
train_arm noisy
status "gen21 on the common val set: $(grep -m1 'warm-start baseline' "$OUT/train_greedy.log")"

# ---- 4. eval ----------------------------------------------------------------------------------
check_stop "before eval"
serve_and_work "$MODELS/exp048_noisy.onnx" "$EVAL_PARALLEL_GAMES" "$OUT/eval.worker.log"
declare -A CLIENT
for arm in greedy noisy; do
    [ -e "$OUT/.evaluated_$arm" ] && continue
    "$HUB" job eval --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" \
        --evaluator nn --model "$MODELS/exp048_$arm.onnx" \
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
