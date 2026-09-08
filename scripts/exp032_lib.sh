#!/usr/bin/env bash
# Shared helpers for the plan 032 experiment queue. Source this from a stage
# script; every stage is resumable (a finished arm is detected by its report
# file) and every match writes a per-game log (plan 031 D10 standing rule).
#
#   OUT      where reports/logs go (default runs/exp032)
#   GAMES    games per match (default 120)
#   PARALLEL concurrent games per eval process (default 6)
#   SEED     shared seed base — one base across arms that get differenced
#            against each other (plan 031 D10)

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="$REPO/target/14x7"

RUN_DIR="$REPO/runs/loop14x7"
OUT="${OUT:-$REPO/runs/exp032}"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
PY="$REPO/train/.venv/bin/python"
SOCK="${SOCK:-/tmp/bbnn-exp032.sock}"
GAMES="${GAMES:-120}"
PARALLEL="${PARALLEL:-6}"
SEED="${SEED:-32000000}"
MODELS="$REPO/models"
mkdir -p "$OUT"

LOG="${LOG:-$OUT/exp032.log}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }
stopped() { [ -e "$OUT/STOP" ]; }

NN_ARGS=""
NN_PID=""
start_sidecar() {
    [ -n "$NN_PID" ] && return 0
    local warm="${1:-$MODELS/bbnet_14x7_gen03.onnx}"
    rm -f "$SOCK"
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda \
        --model "$warm" --max-models 6 --stats-every 600 >> "$OUT/nn_server.log" 2>&1 &
    NN_PID=$!
    local i=0
    while [ ! -S "$SOCK" ]; do
        i=$((i + 1))
        if [ "$i" -gt 120 ] || ! kill -0 "$NN_PID" 2>/dev/null; then
            log "WARN: sidecar did not start — running on tract"; NN_PID=""; return 0
        fi
        sleep 1
    done
    NN_ARGS="--nn-server $SOCK"
    log "sidecar up (pid $NN_PID)"
}
stop_sidecar() { [ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; NN_PID=""; NN_ARGS=""; rm -f "$SOCK"; }
# INT/TERM must exit after the cleanup — a bare `trap f TERM` resumes the
# script after f returns, so `kill <pid>` was being swallowed.
trap stop_sidecar EXIT
trap 'stop_sidecar; trap - EXIT; exit 143' INT TERM

# play TAG CAND_ONNX OPP_ONNX [extra eval args...]
# Candidate vs opponent, both `--evaluator nn` at 1000 iterations unless the
# extra args override. Extra args are passed through to `botbowl-ui eval`
# (e.g. --puct-c 30, or env-driven knobs set by the caller).
play() {
    local tag="$1"; local cand="$2"; local opp="$3"; shift 3
    local rep="$OUT/$tag.json"
    [ -e "$rep" ] && { log "$tag already played"; return 0; }
    log "$tag: $(basename "$cand") vs $(basename "$opp"), $GAMES games x$PARALLEL, seed $SEED, extra: $*"
    log "$tag: commit $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty) env: $(env | grep '^BLOOD_' | tr '\n' ' ')"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    if ! "$UI" eval --evaluator nn --model "$cand" \
            --mcts-iters 1000 \
            --vs-evaluator nn --vs-model "$opp" \
            --vs-games "$GAMES" --seed "$SEED" \
            --skip-lectures --skip-fixed-rungs \
            --parallel-games "$PARALLEL" $NN_ARGS \
            --per-game-out "$OUT/$tag.games.jsonl" \
            --out "$rep" "$@" > "$OUT/$tag.log" 2>&1; then
        log "$tag FAILED — see $tag.log"; return 1
    fi
    log "$tag done ($(((SECONDS - t0) / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$rep" 2>/dev/null || echo see json)"
    "$PY" "$REPO/scripts/paired_summary.py" "$OUT/$tag.games.jsonl" >> "$LOG" 2>&1 || true
}

# train_arm NAME DIMS INIT LR [extra train args...]
train_arm() {
    local name="$1"; local dims="$2"; local init="$3"; local lr="$4"; shift 4
    local pt="$OUT/$name.pt"
    local onnx="$OUT/$name.onnx"
    local hold="${HOLD:-$RUN_DIR/gen07/prepared_val/dims_16x9}"
    [ -f "$onnx" ] && { log "$name already trained"; return 0; }
    log "$name: train on $dims, init $(basename "$init"), lr $lr, extra: $*"
    local t0=$SECONDS
    if ! "$PY" -m bbnn.train --data "$dims" --val-data "$hold" \
            --init "$init" --lr "$lr" --seed "${TRAIN_SEED:-20260906}" \
            --eval-every "${EVAL_EVERY:-2500}" --select-on combined \
            --device auto --out "$pt" --onnx "$onnx" "$@" \
            > "$OUT/$name.train.log" 2>&1; then
        log "$name FAILED — see $name.train.log"; return 1
    fi
    log "$name trained ($(((SECONDS - t0) / 60)) min): $(grep 'restored best-val' "$OUT/$name.train.log" | tail -1)"
}
