#!/usr/bin/env bash
# Plan 066 phase 4 on this box: the 20x11 loop's generation, sidecars vs in-process, same net,
# seeds and streams (scripts/perf_gen_steady.sh), then the in-process stream count, then mc-label.
#
#   scripts/perf066.sh OUT_DIR NET.onnx LABEL_CORPUS.jsonl [STEPS]
#   STEPS (default "gen-sidecar gen-inproc streams label"):
#     gen-sidecar  48 streams over 2 sidecars and 2 dataset processes (the loop's GEN_SIDECARS=2)
#     gen-inproc   48 streams, one process, the net in it
#     streams      64 and 96 streams in-process
#     label        mc-label, 48 trajectories x 8 playouts, sidecar vs in-process (scripts/perf_label.sh)
# NET.ts and NET.pt beside NET.onnx. WARM/WINDOW as perf_gen_steady.sh (240/600).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?out dir}"; NET="${2:?net.onnx}"; LCORPUS="${3:?label corpus}"
STEPS="${4:-gen-sidecar gen-inproc streams label}"
export PY="${PY:-$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')/train/.venv/bin/python}"
export BOARD=20x11/7 CANVAS=13x22 CONFIG=cfgs/chance_all_partial_gen.toml
export SIZES="--size-centre 220 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8 --size-max-area 220 --size-min-area 70 --cells-per-player 26"
mkdir -p "$OUT"
for S in $STEPS; do
    echo "[$(date '+%H:%M:%S')] step $S" >> "$OUT/steps.log"
    case "$S" in
        gen-sidecar) SIDECARS=2 BACKEND=sidecar "$REPO/scripts/perf_gen_steady.sh" "$OUT/gen" "$NET" "48" ;;
        gen-inproc) BACKEND=inproc "$REPO/scripts/perf_gen_steady.sh" "$OUT/gen" "$NET" "48" ;;
        streams) BACKEND=inproc "$REPO/scripts/perf_gen_steady.sh" "$OUT/gen" "$NET" "64 96" ;;
        label) TRAJ=48 PLAYOUTS=8 "$REPO/scripts/perf_label.sh" "$OUT/label" "$NET" "$LCORPUS" "sidecar inproc" ;;
    esac
done
echo "[$(date '+%H:%M:%S')] done" >> "$OUT/steps.log"
