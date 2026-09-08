#!/usr/bin/env bash
# Plan 032 #5 (training-seed variance) and #9 (capacity at fixed data), both
# on the D7 pool (gen01-07, visit target) with D7 itself as the control —
# the recipe every from-scratch arm in plans 029/032 shares.
#
#   d7s2   = D7's recipe from a second materialised init (seed 20260907 for
#            init, shuffle and augmentation). d7s2 vs d7, 300 games: bounds
#            the seed floor at about ±0.03 (plan 031 D10 sizing).
#   d7w96  = width 96 / blocks 8 (1.39 M params, 2.9x) from the same seed as
#            D7, same steps. d7w96 vs d7, 120 games.
#
# Training runs now on the GPU next to the eval chain (prepare is niced, the
# D7 pool is regenerated in ~1 min if it was pruned). The matches wait for
# WAIT_PID (the stage-3 process) so they never share the cores with another
# eval, and use their own socket.
#
#   WAIT_PID=<stage 3 pid> nohup scripts/exp032_s59_seed_capacity.sh > /dev/null 2>&1 &
SOCK="${SOCK:-/tmp/bbnn-exp032-s59.sock}"
source "$(dirname "$0")/exp032_lib.sh"

D7="$REPO/runs/exp-data/d7.onnx"
POOL="$REPO/runs/exp-data/prep_d7"
HOLD_DIR="$RUN_DIR/gen07/prepared_val/dims_16x9"
STEPS="${STEPS:-110000}"
WAIT_PID="${WAIT_PID:-}"
SEED2="${SEED2:-20260907}"

log "=== #5 seed variance + #9 capacity, control D7 ==="
[ -f "$D7" ] || die "$D7 missing"
[ -d "$HOLD_DIR" ] || die "$HOLD_DIR missing"

if [ ! -d "$POOL" ]; then
    inputs=""
    for g in gen01 gen02 gen03 gen04 gen05 gen06 gen07; do
        for k in 0 1 2 3 5 6; do inputs="$inputs $RUN_DIR/$g/shard$k.jsonl"; done
    done
    log "regenerating the D7 pool ($(echo $inputs | wc -w) shards)"
    # shellcheck disable=SC2086
    nice -n 19 "$PREPARE" --in $inputs --out "$POOL.tmp" > "$OUT/prep_d7.log" 2>&1 \
        || die "prepare d7 pool failed"
    mv "$POOL.tmp" "$POOL"
fi
DIMS="$(ls -d "$POOL"/dims_* | head -1)"

# --- #5: second init, same recipe ------------------------------------------
INIT2="$OUT/arm_init_s2.pt"
if [ ! -f "$INIT2" ]; then
    "$PY" -m bbnn.train --data "$HOLD_DIR" --epochs 0 --seed "$SEED2" --out "$INIT2" --device cpu \
        > "$OUT/arm_init_s2.log" 2>&1 || die "could not materialise init 2"
    log "second init written: $INIT2 (seed $SEED2)"
fi
stopped && exit 0
TRAIN_SEED="$SEED2" train_arm d7s2 "$DIMS" "$INIT2" 1e-3 --max-steps "$STEPS" || die "d7s2 failed"

# --- #9: wider/deeper net, same seed as D7 -------------------------------------
# No --init: a 96x8 net cannot share D7's 64x6 init; --seed pins its own.
stopped && exit 0
if [ ! -f "$OUT/d7w96.onnx" ]; then
    log "d7w96: train 96x8 on $DIMS, seed 20260906, lr 1e-3, --max-steps $STEPS"
    t0=$SECONDS
    "$PY" -m bbnn.train --data "$DIMS" --val-data "$HOLD_DIR" --width 96 --blocks 8 \
        --lr 1e-3 --seed 20260906 --eval-every 2500 --select-on combined --max-steps "$STEPS" \
        --device auto --out "$OUT/d7w96.pt" --onnx "$OUT/d7w96.onnx" > "$OUT/d7w96.train.log" 2>&1 \
        || die "d7w96 failed"
    log "d7w96 trained ($(((SECONDS - t0) / 60)) min): $(grep 'restored best-val' "$OUT/d7w96.train.log" | tail -1)"
fi
log "d7 for comparison: $(grep 'restored best-val' "$REPO/runs/exp-data/d7.train.log" | tail -1)"

if [ -n "$WAIT_PID" ]; then
    log "nets trained; waiting for pid $WAIT_PID before the matches"
    while kill -0 "$WAIT_PID" 2>/dev/null; do stopped && exit 0; sleep 60; done
    sleep 90
fi
stopped && exit 0
start_sidecar "$D7"
play "s9-d7w96-vs-d7" "$OUT/d7w96.onnx" "$D7"
stopped && exit 0
GAMES=300 play "s5-d7s2-vs-d7" "$OUT/d7s2.onnx" "$D7"
log "=== #5/#9 complete ==="
