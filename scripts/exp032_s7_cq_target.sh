#!/usr/bin/env bash
# Plan 032 #7 — completed-Q policy target, the one-variable retrain.
#
# D7 (plan 029: from scratch on gen01-07, arm_init.pt, lr 1e-3, 110k steps,
# visit target) is the control. Q7 is the identical recipe with the pool and
# the held-out set prepared under `--policy-target cq --tau 100`. Same init,
# same seed, same steps; only the policy label differs. Then Q7 vs D7, 120
# games on the shared seed base.
#
# The held-out set is re-prepared under the same target so `--select-on
# combined` picks the checkpoint against the label the net is trained on
# (val_policy is not comparable to D7's; val_value is).
#
# Ordering: prepare + train can overlap the eval stages (GPU has headroom,
# prepare is niced). The match waits for WAIT_PID (the stage-2 runner) so it
# does not share the box's cores with another 6-way eval, and then hands the
# box to NEXT (stage 3) — #7 outranks #3 in the queue.
#
#   nohup scripts/exp032_s7_cq_target.sh > /dev/null 2>&1 &
source "$(dirname "$0")/exp032_lib.sh"

TAU="${TAU:-100}"
INIT="$REPO/runs/exp-data/arm_init.pt"
D7="$REPO/runs/exp-data/d7.onnx"
STEPS="${STEPS:-110000}"
WAIT_PID="${WAIT_PID:-}"
NEXT="${NEXT:-}"
TRAIN_GENS="gen01 gen02 gen03 gen04 gen05 gen06 gen07"
TRAIN_SHARDS="0 1 2 3 5 6"
VAL_SHARDS="4 7"

log "=== #7: completed-Q target, tau=$TAU (control: D7) ==="
[ -f "$INIT" ] || die "$INIT missing"
[ -f "$D7" ] || die "$D7 missing"

prep_cq() {  # prep_cq NAME "gens" "shards"
    local name="$1" gens="$2" shards="$3"
    local dir="$OUT/prep_$name"
    [ -d "$dir" ] && { log "$name pool exists"; return 0; }
    local inputs="" g k
    for g in $gens; do for k in $shards; do inputs="$inputs $RUN_DIR/$g/shard$k.jsonl"; done; done
    log "$name: prepare --policy-target cq --tau $TAU on $(echo $inputs | wc -w) shards"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    nice -n 19 "$PREPARE" --in $inputs --out "$dir.tmp" --policy-target cq --tau "$TAU" \
        > "$OUT/prep_$name.log" 2>&1 || { log "prepare $name FAILED"; return 1; }
    mv "$dir.tmp" "$dir"
    log "$name prepared ($(((SECONDS - t0) / 60)) min): $(tail -1 "$OUT/prep_$name.log")"
}

stopped && exit 0
prep_cq q7_val gen07 "$VAL_SHARDS" || die "val prepare failed"
prep_cq q7 "$TRAIN_GENS" "$TRAIN_SHARDS" || die "train prepare failed"

HOLD="$(ls -d "$OUT"/prep_q7_val/dims_* | head -1)"
train_arm q7 "$(ls -d "$OUT"/prep_q7/dims_* | head -1)" "$INIT" 1e-3 --max-steps "$STEPS" \
    || die "q7 training failed"
log "q7 baseline: $(grep 'warm-start baseline' "$OUT/q7.train.log" | tail -1)"
log "d7 for comparison: $(grep 'restored best-val' "$REPO/runs/exp-data/d7.train.log" | tail -1)"

if [ -n "$WAIT_PID" ]; then
    log "q7 trained; waiting for pid $WAIT_PID before the match"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
stopped && exit 0
start_sidecar "$D7"
play "s7-q7-vs-d7" "$OUT/q7.onnx" "$D7"
log "=== #7 complete ==="
stop_sidecar
[ -n "$NEXT" ] && exec "$NEXT"
