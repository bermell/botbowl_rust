#!/usr/bin/env bash
# Plan 066 §4 tests A and B: bit parity of the in-process backend with the sidecar, on one plan of
# encoded corpus states (every board in the corpus, every bucket, policy and value-only batches).
#
#   scripts/parity066.sh OUT_DIR NET.onnx "SHARD.jsonl ..." [MODES]
#   scripts/parity066.sh runs/exp077/parity models/az_v7/bbnet_mix16x9v9_gen44.onnx \
#       "runs/loopmix16x9v9/gen44/shard*.jsonl" "det prod"
#
# NET.onnx needs NET.pt (the sidecar's weights) and NET.ts (`python -m bbnn.export_ts NET.pt`)
# beside it. Per mode (`det` = cuDNN deterministic, benchmark off; `prod` = benchmark on, what the
# loop runs) it writes, each from its own process:
#   side_a, side_b   two sidecar runs (nn_server's own Registry/RunnerPool/GraphRunner)
#   inproc_l0        the in-process graphs on lane 0
#   inproc_l0b       a second in-process run on lane 0
#   inproc_l1        lane 1 (captured after lane 0, as the batchers' second stream is)
# and compares: in-process vs sidecar (the gate), sidecar vs sidecar (the floor B is judged
# against), in-process run to run, and lane 1 vs lane 0. Summaries land in OUT_DIR/summary.txt and
# OUT_DIR/*.json; the big output files are deleted after comparison unless KEEP=1.
#
# Env: BOARD (20x11/7), CANVAS (13x22), MAX_STATES (120000), SEED (66), KEEP (0).
set -euo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?out dir}"; NET="${2:?net.onnx}"; SHARDS="${3:?shards}"; MODES="${4:-det prod}"
BOARD="${BOARD:-20x11/7}"; CANVAS="${CANVAS:-13x22}"; MAX_STATES="${MAX_STATES:-120000}"; SEED="${SEED:-66}"
PY="${PY:-$REPO/train/.venv/bin/python}"
# The main checkout's venv when run from a worktree.
[ -x "$PY" ] || PY="$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')/train/.venv/bin/python"
BW="${BOARD%%x*}"; REST="${BOARD#*x}"; BH="${REST%%/*}"; BP="${REST#*/}"
export BOARD_SIZE_W="$BW" BOARD_SIZE_H="$BH" BOARD_PLAYERS="$BP"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/${BW}x${BH}-torch}"
export BLOOD_NN_CANVAS="$CANVAS"
mkdir -p "$OUT"
TS="${NET%.onnx}.ts"
[ -f "${NET%.onnx}.pt" ] || { echo "no ${NET%.onnx}.pt beside $NET" >&2; exit 1; }
[ -f "$TS" ] || { echo "no $TS — python -m bbnn.export_ts ${NET%.onnx}.pt" >&2; exit 1; }
(cd "$REPO" && cargo build --release -j 4 -p botbowl-nn --features torch --example inproc_parity) >> "$OUT/build.log" 2>&1
BIN="$CARGO_TARGET_DIR/release/examples/inproc_parity"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/summary.txt"; }

if [ ! -f "$OUT/states.bin" ]; then
    # shellcheck disable=SC2086
    "$BIN" dump --out "$OUT/states.bin" --max "$MAX_STATES" $SHARDS 2>&1 | tee -a "$OUT/summary.txt"
fi
[ -f "$OUT/plan.bin" ] || "$PY" "$REPO/scripts/parity066.py" plan --states "$OUT/states.bin" --out "$OUT/plan.bin" --seed "$SEED" | tee -a "$OUT/summary.txt"

cmp() {  # cmp TAG A B
    log "== $1"
    "$PY" "$REPO/scripts/parity066.py" compare "$OUT/$2.bin" "$OUT/$3.bin" --states "$OUT/states.bin" --plan "$OUT/plan.bin" \
        --json "$OUT/$1.json" | tee -a "$OUT/summary.txt"
}
for MODE in $MODES; do
    DET=""; [ "$MODE" = det ] && DET="--deterministic"
    for RUN in a b; do
        log "sidecar $MODE run $RUN"
        "$PY" "$REPO/scripts/parity066.py" sidecar --states "$OUT/states.bin" --plan "$OUT/plan.bin" --out "$OUT/side_${MODE}_$RUN.bin" \
            --model "$NET" --canvas "$CANVAS" --mode "$MODE" > "$OUT/side_${MODE}_$RUN.log" 2>&1
    done
    for RUN in l0 l0b l1; do
        LANE=0; [ "$RUN" = l1 ] && LANE=1
        log "in-process $MODE run $RUN (lane $LANE)"
        # shellcheck disable=SC2086
        "$BIN" run --ts "$TS" --states "$OUT/states.bin" --plan "$OUT/plan.bin" --out "$OUT/inproc_${MODE}_$RUN.bin" \
            --lane "$LANE" $DET --graph "$OUT/inproc_${MODE}_graph.txt" > "$OUT/inproc_${MODE}_$RUN.log" 2>&1
    done
    cmp "${MODE}_inproc_vs_sidecar" "side_${MODE}_a" "inproc_${MODE}_l0"
    cmp "${MODE}_sidecar_vs_sidecar" "side_${MODE}_a" "side_${MODE}_b"
    cmp "${MODE}_inproc_vs_inproc" "inproc_${MODE}_l0" "inproc_${MODE}_l0b"
    cmp "${MODE}_lane1_vs_lane0" "inproc_${MODE}_l0" "inproc_${MODE}_l1"
    cmp "${MODE}_inproc_b_vs_sidecar_b" "side_${MODE}_b" "inproc_${MODE}_l0b"
    [ "${KEEP:-0}" = 1 ] || rm -f "$OUT"/side_"${MODE}"_*.bin "$OUT"/inproc_"${MODE}"_*.bin
done
log "done"
