#!/usr/bin/env bash
# MC-label CPU benchmark (plan 069 §1), the labelling twin of perf_search_bench.sh: instructions
# retired (`perf stat -e instructions:u`) for `botbowl-ui mc-label` over a fixed corpus slice, plus
# a hash of the labels it wrote. Equal hashes on two binaries mean the labels are unchanged.
#
#   scripts/perf_label_bench.sh OUT_DIR TAG CORPUS.jsonl
#
# Env: BIN (a prebuilt botbowl-ui; default: build into target/perf-<CAP>, torch on), CAP (20x11/7),
# MODEL (…/bbnet_mix16x9v9_gen48.onnx, its .ts beside it), PLAYOUTS (8), PARALLEL (1: one stream,
# every forward a batch of 1, so the labels are bit-reproducible with cuDNN deterministic),
# BACKEND (inproc), CORE (7), EVENTS (instructions:u,cycles:u,task-clock), RECORD=1 / PERIOD as in
# perf_search_bench.sh (bucket split in OUT/TAG.buckets, start-up excluded).
# Per-forward numbers need the start-up subtracted: run two slices of different size and difference
# them, or use RECORD=1's buckets.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; TAG="${2:?tag}"; CORPUS="${3:?corpus slice}"
CAP="${CAP:-20x11/7}"; PLAYOUTS="${PLAYOUTS:-8}"; PARALLEL="${PARALLEL:-1}"; BACKEND="${BACKEND:-inproc}"
CORE="${CORE:-7}"; EVENTS="${EVENTS:-instructions:u,cycles:u,task-clock}"
MODEL="${MODEL:-$REPO/models/az_v7/bbnet_mix16x9v9_gen48.onnx}"
mkdir -p "$OUT"
CW="${CAP%%x*}"; CR="${CAP#*x}"; CH="${CR%%/*}"; CP="${CR#*/}"
export BOARD_SIZE_W="$CW" BOARD_SIZE_H="$CH" BOARD_PLAYERS="$CP" CARGO_TARGET_DIR="$REPO/target/perf-${CW}x${CH}"
if [ -z "${BIN:-}" ]; then
    cargo build --release -j "${BUILD_JOBS:-3}" -p botbowl-ui --features deterministic_hash,torch >> "$OUT/build.log" 2>&1 \
        || { echo "build failed, see $OUT/build.log" >&2; exit 1; }
    BIN="$CARGO_TARGET_DIR/release/botbowl-ui"
fi
SHIM="$CARGO_TARGET_DIR/fixed_getrandom.so"
[ "$SHIM" -nt scripts/fixed_getrandom.c ] || gcc -O2 -shared -fPIC -o "$SHIM" scripts/fixed_getrandom.c
export FIXRAND_PER_CALL=1 BLOOD_NN_DETERMINISTIC=1 BLOOD_NN_LANES="${BLOOD_NN_LANES:-1}"
if [ -n "${RECORD:-}" ]; then
    PERF=(perf record -q -e instructions:u -c "${PERIOD:-1000003}" --call-graph lbr -o "$OUT/$TAG.data")
else
    PERF=(perf stat -x, -e "$EVENTS" -o "$OUT/$TAG.stat")
fi
rm -rf "$OUT/$TAG.labels"
LD_PRELOAD="$SHIM" BLOOD_NN_PROFILE=1 taskset -c "$CORE" nice -n 19 "${PERF[@]}" \
    "$BIN" mc-label --corpus "$CORPUS" --model "$MODEL" --nn-backend "$BACKEND" --playouts "$PLAYOUTS" \
    --parallel "$PARALLEL" --out-dir "$OUT/$TAG.labels" > "$OUT/$TAG.out" 2> "$OUT/$TAG.log"
if [ -n "${RECORD:-}" ]; then
    DEMANGLE=cat; command -v rustfilt > /dev/null && DEMANGLE=rustfilt
    perf script -i "$OUT/$TAG.data" -F comm,tid,period,ip,sym,dso 2> /dev/null | "$DEMANGLE" \
        | python3 scripts/perf_buckets.py --label --after "sample_label|play_out" --top 15 > "$OUT/$TAG.buckets"
    INS="sampled:$(sed -n 's/^samples [0-9]*, weight \([0-9]*\).*/\1/p' "$OUT/$TAG.buckets")"
else
    INS=$(grep instructions "$OUT/$TAG.stat" | cut -d, -f1)
fi
LABELS=$(python3 - "$OUT/$TAG.labels" <<'PY'
import glob, hashlib, json, sys
h = hashlib.md5(); n = 0
for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl")):
    for line in open(f):
        for s in json.loads(line).get("samples", []):
            h.update(repr(s.get("outcome_value")).encode()); n += 1
print(f"{h.hexdigest()[:12]} samples={n}")
PY
)
PROF=$(grep -h '^NN_PROFILE' "$OUT/$TAG.log" | tail -1 | cut -d' ' -f2-)
echo "$TAG instructions=$INS labels=$LABELS $PROF" | tee -a "$OUT/results.txt"
