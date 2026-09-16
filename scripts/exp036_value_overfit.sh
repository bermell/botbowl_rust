#!/usr/bin/env bash
# Plan 036 E1-E3 — the value-head overfit arms, offline on one generation's
# window. No games are played: every arm is a `train.py` run on a fixed
# prepare output, and the numbers reported are
# (restore_step, val_policy@restore, val_value@restore, val_value@end).
#
# Per plan 032's ground rule the val_* numbers *select* the arm for E4, they
# do not decide it. E4 (winner vs baseline, 600 games vs `scripted`) is a
# separate script and is not run here.
#
# ---- which window --------------------------------------------------------
# GEN picks it, and the choice matters more than it looks. Plan 036's symptom
# is about *warm-started fine-tunes*: "every warm-started fine-tune since gen04
# restores its best checkpoint at epoch 0-2 of 10". gen01 of a from-scratch run
# trains from random init at 1e-3 and is a different regime — mechanism 1 (two
# thirds of the window already fitted) is absent there by construction, and only
# mechanism 2 (value-label multiplicity) is in play.
#
# So gen01 is a real but partial read: it measures W1-W5 against label noise
# alone. Re-run on gen02+ (INIT set, lr 2e-4) for the regime the loop actually
# lives in. Both are worth having; the script does whichever it is pointed at
# and stamps which one into the report.
#
# ---- resumability --------------------------------------------------------
# An arm is done when its .train.log holds a `restored best-val` line, and a
# prepare is done when its dir exists. Both are skipped on a re-run, so this is
# safe to kill and relaunch.
#
#   RUN_DIR=$PWD/runs/az14x7v6 GEN=gen02 nohup scripts/exp036_value_overfit.sh &
#   tail -f runs/exp036/exp036.log
#   touch runs/exp036/STOP    # clean exit at the next arm boundary
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/14x7}"

# Run this from a worktree and $REPO/runs is empty — point RUN_DIR at the real
# run. The binaries and the venv still come from wherever this checkout is,
# which is the point of running it from the worktree.
RUN_DIR="${RUN_DIR:-$REPO/runs/az14x7v6}"
OUT="${OUT:-$REPO/runs/exp036}"
GEN="${GEN:-gen01}"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
PY="${PY:-$REPO/train/.venv/bin/python}"
# The loop's own settings, so the baseline arm *is* the loop.
WINDOW_GENS="${WINDOW_GENS:-3}"
TRAIN_SHARDS="${TRAIN_SHARDS:-0 1 2 3 5 6}"
VAL_SHARDS="${VAL_SHARDS:-4 7}"
POLICY_ARGS="${POLICY_ARGS:---policy-target cq --tau 100}"
EPOCHS="${EPOCHS:-10}"
EVAL_EVERY="${EVAL_EVERY:-2500}"
# One seed across every arm: same init, same batch order, same flips, so a
# measured difference is the flag and not the draw (train.py --seed).
TRAIN_SEED="${TRAIN_SEED:-20260916}"
# Warm start. Empty = from-scratch at SCRATCH_LR, which is what gen01 of a
# from-scratch run does. Set both together or neither.
INIT="${INIT:-}"
LR="${LR:-}"
SCRATCH_LR="${SCRATCH_LR:-1e-3}"
WARM_LR="${WARM_LR:-2e-4}"

mkdir -p "$OUT"
LOG="$OUT/exp036.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }
stopped() { [ -e "$OUT/STOP" ] && { log "STOP file present — exiting cleanly"; return 0; }; return 1; }

[ -x "$PREPARE" ] || die "$PREPARE missing — build the 14x7 binaries first"
[ -x "$PY" ] || die "$PY missing"
[ -d "$RUN_DIR/$GEN" ] || die "$RUN_DIR/$GEN missing"

if [ -z "$INIT" ]; then
    LR="${LR:-$SCRATCH_LR}"
    REGIME="from-scratch (random init, lr $LR) — mechanism 2 only, see header"
    INIT_ARGS=""
else
    [ -f "$INIT" ] || die "INIT $INIT missing"
    LR="${LR:-$WARM_LR}"
    REGIME="warm start from $(basename "$INIT") at lr $LR — the loop's own regime"
    INIT_ARGS="--init $INIT"
fi

# The window: the last $WINDOW_GENS generations up to and including $GEN, as
# train_loop.sh builds it. A generation that does not exist is simply absent
# (gen01 of a fresh run has a window of one).
gen_num() { echo "${1#gen}" | sed 's/^0*//'; }
window_inputs() {  # window_inputs "shards..."
    local shards="$1" inputs="" n g k
    n=$(gen_num "$GEN")
    for ((i = n - WINDOW_GENS + 1; i <= n; i++)); do
        [ "$i" -lt 1 ] && continue
        g=$(printf "gen%02d" "$i")
        for k in $shards; do
            [ -f "$RUN_DIR/$g/shard$k.jsonl" ] && inputs="$inputs $RUN_DIR/$g/shard$k.jsonl"
        done
    done
    echo "$inputs"
}

# prep NAME "extra prepare args..."
prep() {
    local name="$1"; shift
    local dir="$OUT/prep_$name"
    [ -d "$dir" ] && { log "prep $name exists"; return 0; }
    local inputs; inputs=$(window_inputs "$TRAIN_SHARDS")
    local vinputs; vinputs=$(window_inputs "$VAL_SHARDS")
    [ -n "$inputs" ] || die "no train shards in the $GEN window"
    log "prep $name: $(echo "$inputs" | wc -w) train + $(echo "$vinputs" | wc -w) val shards, extra: $*"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    nice -n 19 "$PREPARE" --in $inputs --out "$dir.tmp/train" $POLICY_ARGS "$@" \
        > "$OUT/prep_$name.log" 2>&1 || { log "prep $name FAILED"; return 1; }
    # shellcheck disable=SC2086
    nice -n 19 "$PREPARE" --in $vinputs --out "$dir.tmp/val" $POLICY_ARGS "$@" \
        >> "$OUT/prep_$name.log" 2>&1 || { log "prep $name val FAILED"; return 1; }
    mv "$dir.tmp" "$dir"
    log "prep $name done ($(((SECONDS - t0) / 60)) min): $(grep -h 'prepare done' "$OUT/prep_$name.log" | head -1)"
}

# arm NAME PREP_NAME "extra train args..."
arm() {
    local name="$1"; local prep_name="$2"; shift 2
    local tlog="$OUT/$name.train.log"
    grep -q 'restored best-val' "$tlog" 2>/dev/null && { log "arm $name already trained"; return 0; }
    local dtrain dval
    dtrain=$(ls -d "$OUT/prep_$prep_name/train"/dims_* 2>/dev/null | head -1)
    dval=$(ls -d "$OUT/prep_$prep_name/val"/dims_* 2>/dev/null | head -1)
    [ -n "$dtrain" ] && [ -n "$dval" ] || { log "arm $name: prep_$prep_name incomplete"; return 1; }
    log "arm $name: prep $prep_name, extra: ${*:-none}"
    local t0=$SECONDS
    # Niced: a generation run may be using the box, and this is offline work
    # with no deadline. shellcheck disable=SC2086
    if ! nice -n 15 "$PY" -m bbnn.train --data "$dtrain" --val-data "$dval" \
            $INIT_ARGS --lr "$LR" --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" \
            --select-on combined --seed "$TRAIN_SEED" --device auto \
            "$@" > "$tlog" 2>&1; then
        log "arm $name FAILED — see $(basename "$tlog")"; return 1
    fi
    log "arm $name done ($(((SECONDS - t0) / 60)) min): $(grep 'restored best-val' "$tlog" | tail -1)"
    log "arm $name policy optimum: $(grep 'policy-only optimum' "$tlog" | tail -1)"
}

log "=== plan 036 E1-E3 on $RUN_DIR/$GEN ==="
log "regime: $REGIME"
log "commit $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty), seed $TRAIN_SEED, epochs $EPOCHS, eval-every $EVAL_EVERY"

# ---- E1: W1 (value weight) and W2 (weight decay), on the baseline corpus ----
# One prepare serves all four: neither flag touches the data.
stopped && exit 0
prep base || die "baseline prepare failed"

stopped && exit 0; arm baseline base
stopped && exit 0; arm w1_vw025  base --value-weight 0.25
stopped && exit 0; arm w2_wd1e4  base --weight-decay 1e-4
stopped && exit 0; arm w1w2      base --value-weight 0.25 --weight-decay 1e-4

# ---- E2: W3 (blended value target) on top of E1's winner ------------------
# E1's winner is read off the report rather than guessed, so this stage is
# deliberately parameterised: set E1_WINNER_ARGS from the E1 table before
# launching stage 2, or leave it and get W1+W2 (the plan's best guess).
E1_WINNER_ARGS="${E1_WINNER_ARGS:---value-weight 0.25 --weight-decay 1e-4}"
for L in ${BLENDS:-0.3 0.5 0.7}; do
    stopped && exit 0
    prep "blend$L" --value-blend "$L" || die "blend $L prepare failed"
    # shellcheck disable=SC2086
    arm "w3_blend$L" "blend$L" $E1_WINNER_ARGS
done

# ---- E3: + W4 (per-drive value weight), + W5 (dedup) ----------------------
# W4 needs no new prepare — weight.npy is written unconditionally — so it goes
# on E2's best blend. W5 does need one, and its wall-clock saving is the
# prepare + per-epoch time in the log, not a val number.
E2_BEST_BLEND="${E2_BEST_BLEND:-0.5}"
stopped && exit 0
# shellcheck disable=SC2086
arm "w4_perdrive" "blend$E2_BEST_BLEND" $E1_WINNER_ARGS --per-drive-value-weight

stopped && exit 0
prep "blend${E2_BEST_BLEND}_dedup" --value-blend "$E2_BEST_BLEND" --dedup || die "dedup prepare failed"
# shellcheck disable=SC2086
arm "w5_dedup" "blend${E2_BEST_BLEND}_dedup" $E1_WINNER_ARGS --per-drive-value-weight

log "=== all arms done ==="
"$PY" "$REPO/scripts/exp036_report.py" "$OUT" | tee -a "$LOG"
