#!/usr/bin/env bash
# The numerics gate for reduced-precision inference (plan 069 item 4). Not bit parity — that is
# scripts/parity066.sh, and it must still pass at the default precision — but "does fp16/bf16/TF32 change
# what the net says and how it plays":
#
#   hpc/tune/precision_gate.sh NET.onnx PREPARED_DIR OUT_DIR [VALUE_BENCH.jsonl ...]
#   VARIANTS="fp32 default tf32 fp16 bf16 fp16-cl" DRIVES_SET=contested_20x11.json DRIVE_VARIANT=fp16 \
#       hpc/tune/precision_gate.sh $P/models/bbnet_mix16x9v9_gen49.onnx $P/prepared/gen49_s4 $P/runs/precision_gate \
#       $P/runs/value_bench/v9_20x11_val.jsonl
#
# 1. exports the variants beside NET.onnx (`python -m bbnn.export_ts NET.pt --dtype … [--channels-last]`,
#    this checkout's export_ts) — needs NET.pt and NET.ts there;
# 2. hpc/tune/precision_gate.py: value and policy deltas vs fp32 on the held-out PREPARED_DIR, on the
#    GPU's own kernels (OUT/numerics.md; exit 1 if a variant fails its thresholds);
# 3. per value bench and per variant, scripts/value_bench.sh through the in-process backend with
#    BLOOD_NN_PRECISION / BLOOD_NN_CHANNELS_LAST (the real queue, shim and CUDA graphs), paired against
#    fp32 by scripts/value_bench_summary.py (OUT/value_bench_<bench>.txt);
# 4. with DRIVES_SET: paired contested drives, the DRIVE_VARIANT bot against the fp32 bot, same net, one
#    `botbowl-ui eval` process (DRIVE_PAIRS fixed pairs, default 200: equal play scores 0.5 ± ~0.035).
#    The candidate's ONNX is a symlink in OUT/nets/<variant>/ whose sibling .ts is the variant file, so
#    both bots run in one process; the process runs BLOOD_NN_PRECISION=fp32 (true fp32 for the
#    reference; the variant file carries its own dtype). Without DRIVES_SET it prints the command.
# Env: VARIANTS (fp32 default tf32 fp16 bf16 fp16-cl), CAP (20x11/7), CARGO_TARGET_DIR (target/tune-WxH),
#      PY, DRIVES_SET, DRIVE_VARIANT (fp16), DRIVE_PAIRS (200), PRESET (cfgs/chance_all_partial.toml),
#      PARALLEL (16).
set -euo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NET="$(readlink -f "${1:?net.onnx}")"; PREP="${2:?prepared dir}"; OUT="${3:?out dir}"; shift 3
VARIANTS="${VARIANTS:-fp32 default tf32 fp16 bf16 fp16-cl}"
CAP="${CAP:-20x11/7}"; CW="${CAP%%x*}"; CR="${CAP#*x}"; CH="${CR%%/*}"; CP="${CR#*/}"
export BOARD_SIZE_W="$CW" BOARD_SIZE_H="$CH" BOARD_PLAYERS="$CP"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/tune-${CW}x${CH}}"
if [ -z "${PY:-}" ]; then
    for c in "$REPO/train/.venv/bin/python" "$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')/train/.venv/bin/python" \
             /opt/venv/bin/python; do [ -x "$c" ] && { PY="$c"; break; }; done
fi
export PY PYTHONPATH="$REPO/train/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/gate.log"; }

dtype_of() { local d="${1%-cl}"; case "$d" in fp16|bf16) echo "$d" ;; *) echo fp32 ;; esac; }
cl_of() { case "$1" in *-cl) echo 1 ;; *) echo 0 ;; esac; }
file_of() {  # the .ts a variant loads
    local d; d=$(dtype_of "$1")
    if [ "$d" = fp32 ] && [ "$(cl_of "$1")" = 0 ]; then echo "${NET%.onnx}.ts"
    else echo "${NET%.onnx}.${d}$([ "$(cl_of "$1")" = 1 ] && echo -cl).ts"; fi
}

log "1. exports"
for V in $VARIANTS; do
    F=$(file_of "$V")
    [ -s "$F" ] && continue
    ARGS=(--dtype "$(dtype_of "$V")" --out "$F"); [ "$(cl_of "$V")" = 1 ] && ARGS+=(--channels-last)
    "$PY" -m bbnn.export_ts "${NET%.onnx}.pt" "${ARGS[@]}" 2>&1 | tee -a "$OUT/gate.log"
done

log "2. numerics on held-out samples ($PREP)"
RC=0
"$PY" "$REPO/hpc/tune/precision_gate.py" "$NET" "$PREP" --variants "$(echo "$VARIANTS" | tr ' ' ,)" \
    --out "$OUT/numerics.md" | tee -a "$OUT/gate.log" || RC=$?

for BENCH in "$@"; do
    B=$(basename "$BENCH" .jsonl)
    log "3. value bench $B"
    ARGS=()
    for V in $VARIANTS; do
        D=$(dtype_of "$V"); P="$D"; [ "$V" = default ] && P=""; [ "${V%-cl}" = tf32 ] && P=tf32
        # value_bench.sh skips a net whose rows exist, so a rerun resumes.
        BLOOD_NN_PRECISION="$P" BLOOD_NN_CHANNELS_LAST="$(cl_of "$V")" \
            "$REPO/scripts/value_bench.sh" "$BENCH" "$OUT/vb_${B}_$V" "$V=$NET" >> "$OUT/gate.log" 2>&1 \
            || { log "value bench $B $V failed (see $OUT/vb_${B}_$V/$V.log)"; RC=1; continue; }
        ARGS+=("$V=$OUT/vb_${B}_$V/$V.jsonl")
    done
    "$PY" "$REPO/scripts/value_bench_summary.py" "${ARGS[@]}" | tee "$OUT/value_bench_$B.txt" | tee -a "$OUT/gate.log"
done

V="${DRIVE_VARIANT:-fp16}"
mkdir -p "$OUT/nets/$V"
ln -sf "$NET" "$OUT/nets/$V/$(basename "$NET")"
ln -sf "$(file_of "$V")" "$OUT/nets/$V/$(basename "${NET%.onnx}").ts"
CMD=(env BLOOD_NN_PRECISION=fp32 BLOOD_NN_CANVAS="$((CH + 2))x$((CW + 2))" "$CARGO_TARGET_DIR/release/botbowl-ui" eval
     --evaluator nn --model "$OUT/nets/$V/$(basename "$NET")" --vs-evaluator nn --vs-model "$NET" --nn-backend inproc
     --bot-config "$REPO/${PRESET:-cfgs/chance_all_partial.toml}" --mcts-iters 1000 --skip-lectures --skip-fixed-rungs
     --positions "${DRIVES_SET:-CONTESTED_SET.json}" --seed 0 --vs-games "${DRIVE_PAIRS:-200}"
     --parallel-games "${PARALLEL:-16}" --per-game-out "$OUT/drives_$V.games.jsonl" --out "$OUT/drives_$V.json")
if [ -n "${DRIVES_SET:-}" ]; then
    log "4. paired drives: $V bot vs fp32 bot on $(basename "$DRIVES_SET")"
    "${CMD[@]}" > "$OUT/drives_$V.log" 2>&1 || { log "drives failed (see $OUT/drives_$V.log)"; RC=1; }
    grep -q 'NN_SERVER_FALLBACK' "$OUT/drives_$V.log" && { log "drives fell back to tract"; RC=3; }
    "$PY" "$REPO/scripts/eval_summary.py" "$OUT/drives_$V.json" 2>/dev/null | tail -3 | tee -a "$OUT/gate.log" || true
else
    log "4. drives not run (no DRIVES_SET); the command would be:"
    printf '%q ' "${CMD[@]}" | tee -a "$OUT/gate.log"; echo
fi
log "done (exit $RC)"
exit "$RC"
