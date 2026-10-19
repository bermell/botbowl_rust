#!/usr/bin/env bash
# Plan 066 §4 test C: end to end, one search stream, fixed seeds, cuDNN deterministic. The same
# `dataset` drives (random-start, --next-drive, the loop's C1 generation preset) and the same
# `mc-label` run, through the sidecar and through the in-process backend, must write identical
# corpora (only `created_unix_secs` is ignored). The floor is measured too: the sidecar run twice.
#
#   scripts/gatec066.sh OUT_DIR NET.onnx CORPUS.jsonl
#   scripts/gatec066.sh runs/exp077/gatec runs/exp077/models/gen44.onnx runs/loopmix16x9v9/gen44/shard4.jsonl
#
# NET.onnx needs NET.pt and NET.ts beside it. Reproducible search: `botbowl-ui` is built with
# `deterministic_hash` and run under `fixed_getrandom.so` (FIXRAND_PER_CALL=1), as
# `perf_search_bench.sh` does. Every run is one game at a time with one search worker, so every
# batch is a batch of 1.
#
# Env: BOARD (20x11/7), CANVAS (13x22), GAMES (3), SEED (77000), ITERS (1000),
#      CONFIG (cfgs/chance_all_partial_gen.toml), LABEL_TRAJ (2), PLAYOUTS (4).
set -euo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?out dir}"; NET="${2:?net.onnx}"; CORPUS="${3:?corpus.jsonl}"
BOARD="${BOARD:-20x11/7}"; CANVAS="${CANVAS:-13x22}"; GAMES="${GAMES:-3}"; SEED="${SEED:-77000}"
ITERS="${ITERS:-1000}"; CONFIG="${CONFIG:-$REPO/cfgs/chance_all_partial_gen.toml}"
LABEL_TRAJ="${LABEL_TRAJ:-2}"; PLAYOUTS="${PLAYOUTS:-4}"
PY="${PY:-$REPO/train/.venv/bin/python}"
[ -x "$PY" ] || PY="$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')/train/.venv/bin/python"
BW="${BOARD%%x*}"; REST="${BOARD#*x}"; BH="${REST%%/*}"; BP="${REST#*/}"
export BOARD_SIZE_W="$BW" BOARD_SIZE_H="$BH" BOARD_PLAYERS="$BP"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/${BW}x${BH}-torch-dh}"
mkdir -p "$OUT"
(cd "$REPO" && cargo build --release -j 4 -p botbowl-ui --features torch,deterministic_hash) >> "$OUT/build.log" 2>&1
UI="$CARGO_TARGET_DIR/release/botbowl-ui"
SHIM="$CARGO_TARGET_DIR/fixed_getrandom.so"
[ "$SHIM" -nt "$REPO/scripts/fixed_getrandom.c" ] || gcc -O2 -shared -fPIC -o "$SHIM" "$REPO/scripts/fixed_getrandom.c"
export FIXRAND_PER_CALL=1
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/summary.txt"; }
SIZES="--size-centre 220 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8 --size-max-area 220 --size-min-area 70 --cells-per-player 26"
head -n "$LABEL_TRAJ" "$CORPUS" > "$OUT/label_in.jsonl"

SIDE_PID=""
sidecar_up() {  # sidecar_up SOCK
    rm -f "$1"
    "$PY" "$REPO/scripts/nn_server.py" --socket "$1" --device cuda --model "$NET" --canvas "$CANVAS" \
        --deterministic --stats-every 0 >> "$OUT/nn_server.log" 2>&1 &
    SIDE_PID=$!
    for _ in $(seq 120); do [ -S "$1" ] && return 0; sleep 1; done
    echo "sidecar did not come up" >&2; exit 1
}
sidecar_down() { kill "$SIDE_PID" 2>/dev/null; wait "$SIDE_PID" 2>/dev/null || true; SIDE_PID=""; }
trap 'sidecar_down' EXIT

run_both() {  # run_both TAG BACKEND_ARGS...
    local tag="$1"; shift
    log "dataset $tag"
    # shellcheck disable=SC2086
    LD_PRELOAD="$SHIM" "$UI" dataset --mode random-start --next-drive --games "$GAMES" --seed "$SEED" \
        --bot-config "$CONFIG" --mcts-iters "$ITERS" --evaluator nn --model "$NET" --parallel-games 1 \
        $SIZES "$@" --out "$OUT/dataset_$tag.jsonl" --truncate > "$OUT/dataset_$tag.log" 2>&1
    log "mc-label $tag"
    mkdir -p "$OUT/label_$tag"
    LD_PRELOAD="$SHIM" "$UI" mc-label --corpus "$OUT/label_in.jsonl" --model "$NET" --playouts "$PLAYOUTS" \
        --parallel 1 --seed 56045 "$@" --out-dir "$OUT/label_$tag" > "$OUT/label_$tag.log" 2>&1
}

SOCK="/tmp/bbnn-gatec-$$.sock"
for RUN in side_a side_b; do
    sidecar_up "$SOCK"
    run_both "$RUN" --nn-server "$SOCK"
    sidecar_down
done
BLOOD_NN_DETERMINISTIC=1 BLOOD_NN_CANVAS="$CANVAS" run_both inproc --nn-backend inproc --nn-device cuda

for KIND in dataset label; do
    A="$OUT/${KIND}_side_a"; B="$OUT/${KIND}_side_b"; C="$OUT/${KIND}_inproc"
    [ "$KIND" = dataset ] && { A="$A.jsonl"; B="$B.jsonl"; C="$C.jsonl"; }
    log "== $KIND: sidecar vs sidecar (floor)"
    "$PY" "$REPO/scripts/parity066.py" corpus "$A" "$B" | tee -a "$OUT/summary.txt" || true
    log "== $KIND: in-process vs sidecar"
    "$PY" "$REPO/scripts/parity066.py" corpus "$A" "$C" | tee -a "$OUT/summary.txt" || true
done
grep -h "NN_SERVER\|NN_INPROC\|fell_back\|wrote" "$OUT"/dataset_*.log "$OUT"/label_*.log | tee -a "$OUT/summary.txt" || true
log "done"
