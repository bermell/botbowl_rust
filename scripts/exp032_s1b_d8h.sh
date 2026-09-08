#!/usr/bin/env bash
# Plan 032 #1b — does the heuristic bootstrap corpus carry the strength?
#
# Stage 1 found D7 (from scratch on gen01-07) at 0.396 vs the incremental
# champion gen03. gen03's lineage starts from gen00, ten epochs on heuristic-
# MCTS games that D7 never saw. One arm splits "recipe" from "data": d8h is
# D7's exact recipe (arm_init.pt, lr 1e-3, 110k steps, visit target) with the
# gen00 shards added to the pool. Then d8h vs gen03, 120 games on the shared
# seed base — directly comparable to stage 1's D7 vs gen03.
#
# Ordering: prepare + train overlap whatever eval is running (GPU has
# headroom). The match must not share the cores with another 6-way eval, so
# it waits for #7's match report, and stage 3 (exp032_s3_puct_fpu.sh) blocks
# on the `s1b.pending` marker this script holds until its match is done.
# Own socket so the sidecar never collides with the chain's.
#
#   nohup scripts/exp032_s1b_d8h.sh > /dev/null 2>&1 &
SOCK="${SOCK:-/tmp/bbnn-exp032-s1b.sock}"
source "$(dirname "$0")/exp032_lib.sh"

INIT="$REPO/runs/exp-data/arm_init.pt"
CHAMP="$MODELS/bbnet_14x7_gen03.onnx"
STEPS="${STEPS:-110000}"
AFTER_REPORT="${AFTER_REPORT:-$OUT/s7-q7-vs-d7.json}"
TRAIN_GENS="gen00 gen01 gen02 gen03 gen04 gen05 gen06 gen07"
TRAIN_SHARDS="0 1 2 3 5 6"
PENDING="$OUT/s1b.pending"

log "=== #1b: D7 recipe + gen00 heuristic shards (d8h), control: D7 vs gen03 = 0.396 ==="
[ -f "$INIT" ] || die "$INIT missing"
touch "$PENDING"
trap 'rm -f "$PENDING"; stop_sidecar' EXIT INT TERM

prep_pool() {  # prep_pool NAME "gens" "shards"
    local name="$1" gens="$2" shards="$3"
    local dir="$OUT/prep_$name"
    [ -d "$dir" ] && { log "$name pool exists"; return 0; }
    local inputs="" g k
    for g in $gens; do for k in $shards; do inputs="$inputs $RUN_DIR/$g/shard$k.jsonl"; done; done
    log "$name: prepare (visit target) on $(echo $inputs | wc -w) shards"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    nice -n 19 "$PREPARE" --in $inputs --out "$dir.tmp" > "$OUT/prep_$name.log" 2>&1 \
        || { log "prepare $name FAILED"; return 1; }
    mv "$dir.tmp" "$dir"
    log "$name prepared ($(((SECONDS - t0) / 60)) min): $(tail -1 "$OUT/prep_$name.log")"
}

stopped && exit 0
prep_pool d8h "$TRAIN_GENS" "$TRAIN_SHARDS" || die "train prepare failed"

# Same held-out as D7 (the loop's gen07/prepared_val, the train_arm default),
# so d8h's val_* ARE comparable to D7's — the only arm pair where that holds.
train_arm d8h "$(ls -d "$OUT"/prep_d8h/dims_* | head -1)" "$INIT" 1e-3 --max-steps "$STEPS" \
    || die "d8h training failed"
log "d7 for comparison: $(grep 'restored best-val' "$REPO/runs/exp-data/d7.train.log" | tail -1)"

log "d8h trained; waiting for $(basename "$AFTER_REPORT") before the match"
while [ ! -f "$AFTER_REPORT" ]; do stopped && exit 0; sleep 60; done
sleep 90   # let the chain's script stop its sidecar and hand over
stopped && exit 0
start_sidecar "$CHAMP"
play "s1b-d8h-vs-gen03" "$OUT/d8h.onnx" "$CHAMP"
log "=== #1b complete ==="
