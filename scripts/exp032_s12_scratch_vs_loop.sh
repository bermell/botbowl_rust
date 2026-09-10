#!/usr/bin/env bash
# Plan 032 #12 — is the loop's train step (warm start from the champion at
# 2e-4 on a 3-gen window) weaker than the recipe that produced Q7 (from
# scratch on the whole corpus, lr 1e-3, 110k steps, cq tau 100)?
#
# gen08's fine-tune picked its best val at step 10k of ~50k and val_value
# then drifted up while train loss fell — the warm start is barely moving the
# net. Same data as the loop's gen09 net (gen01..09 vs its gen07..09 window),
# one arm: q9 = Q7 recipe on gen01..09, then q9 vs bbnet_14x7_gen09 head to
# head. If q9 wins clearly (>= 0.60) the loop's train step becomes the
# from-scratch recipe; if it loses or ties the cheap fine-tune stays.
#
# Runs alongside the loop: waits for "gen09 train done" in status.md (so the
# gen09 corpus is complete), trains during gen09's eval phase (GPU mostly
# idle then), and plays the match once the loop has exited at its STOP
# boundary (the STOP file is placed by the operator).
#
#   nohup scripts/exp032_s12_scratch_vs_loop.sh > /dev/null 2>&1 &
SOCK="${SOCK:-/tmp/bbnn-exp032-s12.sock}"
source "$(dirname "$0")/exp032_lib.sh"

TAU=100
INIT="$REPO/runs/exp-data/arm_init.pt"
STEPS="${STEPS:-110000}"
TRAIN_GENS="gen01 gen02 gen03 gen04 gen05 gen06 gen07 gen08 gen09"
TRAIN_SHARDS="0 1 2 3 5 6"
VAL_SHARDS="4 7"
STATUS="$RUN_DIR/status.md"
LOOP_NET="$MODELS/bbnet_14x7_gen09.onnx"

log "=== #12: from-scratch cq (Q7 recipe) on gen01..09 vs the loop's warm-started gen09 ==="
[ -f "$INIT" ] || die "$INIT missing"

wait_for_line() {  # wait_for_line PATTERN
    until grep -q "$1" "$STATUS"; do stopped && exit 0; sleep 60; done
}

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

if [ ! -f "$OUT/q9.onnx" ]; then
    log "waiting for 'gen09 train done' in status.md"
    wait_for_line "gen09 train done"
    for g in $TRAIN_GENS; do for k in $TRAIN_SHARDS $VAL_SHARDS; do
        [ -s "$RUN_DIR/$g/shard$k.jsonl" ] || die "$g/shard$k.jsonl missing"
    done; done
    stopped && exit 0
    prep_cq q9_val gen09 "$VAL_SHARDS" || die "val prepare failed"
    prep_cq q9 "$TRAIN_GENS" "$TRAIN_SHARDS" || die "train prepare failed"
    HOLD="$(ls -d "$OUT"/prep_q9_val/dims_* | head -1)"
    train_arm q9 "$(ls -d "$OUT"/prep_q9/dims_* | head -1)" "$INIT" 1e-3 --max-steps "$STEPS" \
        || die "q9 training failed"
    log "loop's gen09 for comparison: $(grep 'gen09 train done' "$STATUS" | tail -1)"
    rm -rf "$OUT/prep_q9"
    log "q9 pool removed (disk: $(df -h / | awk 'NR==2{print $4}') free)"
fi

log "q9 ready; waiting for the loop to exit (STOP boundary after gen09's verdict)"
while pgrep -f 'bash scripts/train_loop.sh' >/dev/null; do stopped && exit 0; sleep 60; done
sleep 30
[ -f "$LOOP_NET" ] || die "$LOOP_NET missing"
log "loop exited: $(grep 'gen09 eval done' "$STATUS" | tail -1 | cut -c1-300)"
stopped && exit 0
start_sidecar "$LOOP_NET"
play "s12-q9-vs-gen09" "$OUT/q9.onnx" "$LOOP_NET"
log "=== #12 complete ==="
