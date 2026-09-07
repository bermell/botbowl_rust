#!/usr/bin/env bash
# Plan 032 #1 — does a from-scratch retrain on the whole corpus (D7, plan 029:
# gen01-07 pooled, 110k steps from a fresh init) beat the incremental champion
# gen03 (eight generations of warm-started fine-tunes)?
#
#   nohup scripts/exp032_s1_d7_vs_gen03.sh > /dev/null 2>&1 &
#   touch runs/exp032/STOP
source "$(dirname "$0")/exp032_lib.sh"

log "=== stage 1: D7 vs gen03 ==="
[ -f "$REPO/runs/exp-data/d7.onnx" ] || die "d7.onnx missing"
start_sidecar
play "s1-d7-vs-gen03" "$REPO/runs/exp-data/d7.onnx" "$MODELS/bbnet_14x7_gen03.onnx"
log "=== stage 1 complete ==="
