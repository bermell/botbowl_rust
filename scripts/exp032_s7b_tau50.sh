#!/usr/bin/env bash
# Plan 032 #7 follow-up — completed-Q target at tau=50 (Q7 at tau=100 beat
# D7 0.613, z=+2.8), plus the #9/#5 matches that exp032_s59_seed_capacity.sh
# was holding (that script is killed while it is still in its wait loop, so
# this one owns the post-stage-3 order: s9 (120) -> s7b (120) -> s5 (300)).
#
# q50 = Q7's exact recipe (arm_init.pt, lr 1e-3, 110k steps, same seed) with
# pool + held-out prepared under --tau 50; the wide-fan probe said tau=25-50
# is sharper there but slightly worse than visits at narrow fans, so this is
# the "does the sharper target win on the whole game" arm. Head-to-head vs
# Q7 (not D7) — the question is which target to ship, and a direct match is
# the cheapest discriminator.
#
#   WAIT_PID=<stage 3 pid> nohup scripts/exp032_s7b_tau50.sh > /dev/null 2>&1 &
SOCK="${SOCK:-/tmp/bbnn-exp032-s7b.sock}"
source "$(dirname "$0")/exp032_lib.sh"

TAU="${TAU:-50}"
INIT="$REPO/runs/exp-data/arm_init.pt"
D7="$REPO/runs/exp-data/d7.onnx"
Q7="$OUT/q7.onnx"
STEPS="${STEPS:-110000}"
WAIT_PID="${WAIT_PID:-}"
TRAIN_GENS="gen01 gen02 gen03 gen04 gen05 gen06 gen07"
TRAIN_SHARDS="0 1 2 3 5 6"
VAL_SHARDS="4 7"

log "=== #7b: completed-Q target, tau=$TAU (control: Q7 tau=100); then #9, #5 matches ==="
for f in "$INIT" "$D7" "$Q7" "$OUT/d7w96.onnx" "$OUT/d7s2.onnx"; do [ -f "$f" ] || die "$f missing"; done

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
prep_cq q50_val gen07 "$VAL_SHARDS" || die "val prepare failed"
prep_cq q50 "$TRAIN_GENS" "$TRAIN_SHARDS" || die "train prepare failed"

HOLD="$(ls -d "$OUT"/prep_q50_val/dims_* | head -1)"
train_arm q50 "$(ls -d "$OUT"/prep_q50/dims_* | head -1)" "$INIT" 1e-3 --max-steps "$STEPS" \
    || die "q50 training failed"
log "q7 for comparison: $(grep 'restored best-val' "$OUT/q7.train.log" | tail -1)"
# The training pool is 17-20 GB and regenerable in a minute; the box is tight.
rm -rf "$OUT/prep_q50"
log "q50 pool removed (disk: $(df -h / | awk 'NR==2{print $4}') free)"

if [ -n "$WAIT_PID" ]; then
    log "q50 trained; waiting for pid $WAIT_PID before the matches"
    while kill -0 "$WAIT_PID" 2>/dev/null; do stopped && exit 0; sleep 60; done
    sleep 90
fi
stopped && exit 0
start_sidecar "$D7"
play "s9-d7w96-vs-d7" "$OUT/d7w96.onnx" "$D7"
stopped && exit 0
play "s7b-q50-vs-q7" "$OUT/q50.onnx" "$Q7"
stopped && exit 0
GAMES=300 play "s5-d7s2-vs-d7" "$OUT/d7s2.onnx" "$D7"
log "=== #7b/#9/#5 complete ==="
