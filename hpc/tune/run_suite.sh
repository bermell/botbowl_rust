#!/usr/bin/env bash
# The inference tuning suite (plan 069 item 4): build the two binaries it drives, then run
# hpc/tune/tune.py over a spec. See hpc/tune/README.md.
#
#   hpc/tune/run_suite.sh SPEC.toml OUT_DIR [tune.py flags: --only micro,gen --dry-run --redo --keep]
#   hpc/tune/run_suite.sh hpc/tune/smoke.toml runs/hpc_tune/smoke            # this box's GTX 1060
#   hpc/tune/run_suite.sh hpc/tune/gh200.toml $P/runs/hpc_tune/pilot1         # one GH200 (inside the image)
#
# Builds `botbowl-ui`, `botbowl-hub` and `botbowl-worker` (torch) and the `inproc_bench` example from THIS checkout at the capacity CAP
# (20x11/7) into CARGO_TARGET_DIR (default target/tune-<W>x<H>: never the loop's target/20x11 or
# target/release), with BUILD_JOBS (3) jobs. Inside the image the checkout builds with the image's
# toolchain and libtorch (its env sets BOTBOWL_TORCH_DIR / BOTBOWL_CUDA_NVCC_INCLUDE), offline from
# the vendored crates when BOTBOWL_CARGO_VENDOR_CONFIG is set. BIN_UI and BIN_BENCH skip the build
# (prebuilt binaries of the same commit; BIN_HUB / BIN_WORKER beside BIN_UI are used for `mix` rows
# with `mode = "worker"`, plan 069 item 4b). PY: the trainer's python (exports the net's fp16/bf16/-cl
# variants with this checkout's bbnn.export_ts; runs tune.py, which needs Python >= 3.11).
set -euo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SPEC="${1:?spec.toml}"; OUT="${2:?out dir}"; shift 2
CAP="${CAP:-20x11/7}"
CW="${CAP%%x*}"; CR="${CAP#*x}"; CH="${CR%%/*}"; CP="${CR#*/}"
export BOARD_SIZE_W="$CW" BOARD_SIZE_H="$CH" BOARD_PLAYERS="$CP"
if [ -z "${PY:-}" ]; then
    for c in "$REPO/train/.venv/bin/python" "$(git -C "$REPO" worktree list 2>/dev/null | head -1 | awk '{print $1}')/train/.venv/bin/python" \
             /opt/venv/bin/python; do
        [ -x "$c" ] && { PY="$c"; break; }
    done
fi
export PY="${PY:?no python found: set PY}"
# This checkout's trainer package first, so the variants come from the export_ts that matches the
# Rust side being measured (a worktree's venv is the main checkout's, editable-installed there).
export PYTHONPATH="$REPO/train/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

if [ -z "${BIN_UI:-}" ] || [ -z "${BIN_BENCH:-}" ]; then
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/tune-${CW}x${CH}}"
    case "$CARGO_TARGET_DIR" in
        */target/release|*/target/20x11|*/target/16x9) echo "refusing to build into a loop/worker target dir: $CARGO_TARGET_DIR" >&2; exit 1 ;;
    esac
    [ -n "${BOTBOWL_CUDA_NVCC_INCLUDE:-}" ] || "$REPO/scripts/fetch_cuda_headers.sh" >> "$OUT/build.log" 2>&1
    C=(--release -j "${BUILD_JOBS:-3}")
    [ -n "${BOTBOWL_CARGO_VENDOR_CONFIG:-}" ] && C+=(--offline --config "$BOTBOWL_CARGO_VENDOR_CONFIG")
    echo "[$(date '+%H:%M:%S')] building into $CARGO_TARGET_DIR (capacity $CAP)"
    (cd "$REPO" && cargo build "${C[@]}" -p botbowl-ui -p botbowl-hub -p botbowl-worker \
            --features botbowl-ui/torch,botbowl-worker/torch \
        && cargo build "${C[@]}" -p botbowl-nn --features torch --example inproc_bench) >> "$OUT/build.log" 2>&1 \
        || { echo "build failed, see $OUT/build.log" >&2; exit 1; }
    export BIN_UI="$CARGO_TARGET_DIR/release/botbowl-ui" BIN_BENCH="$CARGO_TARGET_DIR/release/examples/inproc_bench"
fi
# Plan 069 item 4b: the hub and worker built beside botbowl-ui, unless given.
export BIN_HUB="${BIN_HUB:-$(dirname "$BIN_UI")/botbowl-hub}" BIN_WORKER="${BIN_WORKER:-$(dirname "$BIN_UI")/botbowl-worker}"
exec "$PY" "$REPO/hpc/tune/tune.py" "$SPEC" "$OUT" "$@"
