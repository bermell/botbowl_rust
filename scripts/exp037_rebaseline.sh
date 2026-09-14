#!/usr/bin/env bash
# Plan 032, post-gen20: re-baseline the anchor on the plan-035/036 binaries, and
# ask the discriminating question at a size that can answer it.
#
# The gen10-20 run was generated and benchmarked at commit 5c7a651, before plan
# 035 (lazy mover tag) and plan 036 (block-dice chance outcomes) changed what the
# search does per iteration. Every anchor point on the curve therefore belongs to
# a binary that no longer exists, and ANCHOR_GAMES=40 resolves only |delta| >~ 0.14
# (plan 032, "Run closed 2026-09-14 at gen20", items 1 and 3).
#
# Two matches, one shared seed base (--seed 0, so pairs use seeds 0..N/2-1 and the
# first 20 pairs are the *same situations* the historical 40-game points used):
#
#   m1  bbnet_14x7_gen19 vs the frozen anchor gen03      -> the re-baselined point
#   m2  scratch          vs the same anchor              -> "can anything trained
#                                                           on this corpus beat 0.61?"
#
# `scratch` is the Q7 recipe (arm_init.pt, lr 1e-3, 110k steps, cq tau 100) on the
# whole accumulated gen10..gen20 corpus, val held out on gen20's shards 4/7. That is
# #12's recipe re-run on the corpus the flat decade actually produced.
#
#   nohup scripts/exp037_rebaseline.sh > /dev/null 2>&1 &
#   touch runs/exp037/STOP    # stops at the next arm boundary
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/runs/exp037}"
SOCK="${SOCK:-/tmp/bbnn-exp037.sock}"
GAMES="${GAMES:-400}"
PARALLEL="${PARALLEL:-8}"
SEED="${SEED:-0}"
source "$(dirname "$0")/exp032_lib.sh"

TAU=100
INIT="$REPO/runs/exp-data/arm_init.pt"
STEPS="${STEPS:-110000}"
TRAIN_GENS="${TRAIN_GENS:-gen10 gen11 gen12 gen13 gen14 gen15 gen16 gen17 gen18 gen19 gen20}"
VAL_GENS="${VAL_GENS:-gen20}"
TRAIN_SHARDS="0 1 2 3 5 6"
VAL_SHARDS="4 7"
ANCHOR="$MODELS/bbnet_14x7_gen03.onnx"
LOOP_NET="$MODELS/bbnet_14x7_gen19.onnx"

log "=== exp037: re-baseline on $(git rev-parse --short HEAD), $GAMES games/match, seed $SEED ==="
[ -f "$INIT" ] || die "$INIT missing"
[ -f "$ANCHOR" ] || die "$ANCHOR missing"
[ -f "$LOOP_NET" ] || die "$LOOP_NET missing"

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

# ---- 1. the from-scratch net on the whole gen10-20 corpus --------------------
if [ ! -f "$OUT/scratch1020.onnx" ]; then
    for g in $TRAIN_GENS; do for k in $TRAIN_SHARDS $VAL_SHARDS; do
        [ -s "$RUN_DIR/$g/shard$k.jsonl" ] || die "$g/shard$k.jsonl missing"
    done; done
    stopped && exit 0
    prep_cq s1020_val "$VAL_GENS" "$VAL_SHARDS" || die "val prepare failed"
    prep_cq s1020 "$TRAIN_GENS" "$TRAIN_SHARDS" || die "train prepare failed"
    HOLD="$(ls -d "$OUT"/prep_s1020_val/dims_* | head -1)"
    export HOLD
    train_arm scratch1020 "$(ls -d "$OUT"/prep_s1020/dims_* | head -1)" "$INIT" 1e-3 \
        --max-steps "$STEPS" || die "scratch1020 training failed"
    rm -rf "$OUT/prep_s1020"
    log "s1020 train pool removed (disk: $(df -h / | awk 'NR==2{print $4}') free)"
fi

# ---- 2. the two matches, shared anchor, shared seed base ---------------------
stopped && exit 0
start_sidecar "$ANCHOR"
play "m1-gen19-vs-gen03" "$LOOP_NET" "$ANCHOR" || die "m1 failed"
stopped && exit 0
play "m2-scratch1020-vs-gen03" "$OUT/scratch1020.onnx" "$ANCHOR" || die "m2 failed"
log "=== exp037 complete ==="
