#!/usr/bin/env bash
# Plan 036 E1-E3, both windows, unattended.
#
# Waits for the from-scratch loop to finish each generation's *generate* phase,
# then runs the arms against that generation's window. Two passes, because they
# answer different halves of the question (see plan 036 "Which window"):
#
#   gen01 — from random init at 1e-3. Mechanism 1 (two thirds of the window
#           already fitted) is absent by construction, so this measures W1-W5
#           against value-label multiplicity alone. A real but partial read.
#   gen02 — warm-started from gen01 at 2e-4. The regime the loop actually lives
#           in, and the one the plan's symptom was observed in.
#
# It runs *alongside* the loop and will slow it: the arms are niced (15) and the
# prepares more so (19), but the GPU is shared with the generation sidecar. That
# is the accepted cost of not waiting a day for the box to be free.
#
# This script lives in the plan-036 worktree and must be run from it — that is
# where the W1-W5 code and its 14x7 binaries are. RUN_DIR and OUT point back at
# the main checkout, so the loop's data is read in place and the reports land
# next to the other experiment dirs.
#
#   nohup scripts/exp036_watch.sh > /dev/null 2>&1 &
#   tail -f <main repo>/runs/exp036/watch.log
#   touch <main repo>/runs/exp036/STOP    # stops at the next arm boundary
set -u

WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="${MAIN:-/home/mattias/repos/botbowl_rust}"
RUN_DIR="${RUN_DIR:-$MAIN/runs/az14x7v6}"
OUT_ROOT="${OUT_ROOT:-$MAIN/runs/exp036}"
MODEL_DIR="${MODEL_DIR:-$MAIN/models/az_v6}"
POLL="${POLL:-60}"

mkdir -p "$OUT_ROOT"
WLOG="$OUT_ROOT/watch.log"
log() { echo "[$(date '+%F %T')] $*" >> "$WLOG"; }

# A STOP anywhere stops everything: the shared file is what the operator
# reaches for, and each pass also honours its own.
stopped() { [ -e "$OUT_ROOT/STOP" ]; }

# Is the loop still alive? Not by process name: it re-execs under
# systemd-inhibit and shows up as plain `bash`, so `pgrep -x train_loop.sh`
# finds nothing — and `pgrep -f` from a monitor script matches the monitor's
# own command line, which is its own class of bug. Use the work instead: every
# phase writes continuously (shard logs per game, loop.log during prepare,
# train and eval), so a run dir that has not been touched in STALE_MIN minutes
# is a dead or finished loop.
STALE_MIN="${STALE_MIN:-60}"
loop_alive() {
    [ -n "$(find "$RUN_DIR" -type f -mmin "-$STALE_MIN" -print -quit 2>/dev/null)" ]
}

# wait_for FILE DESCRIPTION — poll until it exists, the loop goes quiet, or STOP.
wait_for() {
    local f="$1" what="$2" waited=0
    while [ ! -e "$f" ]; do
        stopped && { log "STOP — gave up waiting for $what"; return 1; }
        if ! loop_alive; then
            log "$RUN_DIR untouched for ${STALE_MIN}m and $what never appeared — giving up"
            return 1
        fi
        sleep "$POLL"
        waited=$((waited + POLL))
        # A heartbeat, so a silent log is distinguishable from a dead watcher.
        [ $((waited % 1800)) -eq 0 ] && log "still waiting for $what ($((waited / 60)) min)"
    done
    log "$what present after $((waited / 60)) min"
    return 0
}

# pass GEN OUT_SUBDIR [INIT LR]
pass() {
    local gen="$1" sub="$2" init="${3:-}" lr="${4:-}"
    local out="$OUT_ROOT/$sub"
    if [ -e "$out/.done" ]; then log "$sub already done"; return 0; fi
    log "=== $sub: E1-E3 on $gen's window ${init:+(warm start $(basename "$init") at lr $lr)} ==="
    local t0=$SECONDS
    cd "$WORKTREE"
    RUN_DIR="$RUN_DIR" OUT="$out" GEN="$gen" \
    INIT="$init" LR="$lr" \
    PY="$MAIN/train/.venv/bin/python" \
    PYTHONPATH="$WORKTREE/train/src" \
    CARGO_TARGET_DIR="$WORKTREE/target/14x7" \
        ./scripts/exp036_value_overfit.sh >> "$WLOG" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then log "$sub FAILED (rc $rc) — see $out/exp036.log"; return 1; fi
    touch "$out/.done"
    log "$sub done ($(((SECONDS - t0) / 60)) min)"
    return 0
}

log "=== watcher up: worktree $WORKTREE, run $RUN_DIR, out $OUT_ROOT ==="
log "commit $(git -C "$WORKTREE" rev-parse --short HEAD)$(git -C "$WORKTREE" diff --quiet || echo -dirty)"

# ---- pass 1: gen01, from scratch ----
wait_for "$RUN_DIR/gen01/.generated" "gen01 corpus" || exit 1
stopped || pass gen01 gen01 || log "gen01 pass did not complete"

# ---- pass 2: gen02, warm-started from gen01 ----
# Needs both the corpus and gen01's .pt to warm-start from; the loop writes the
# .pt during gen01's train phase, which is well before gen02's corpus.
wait_for "$RUN_DIR/gen02/.generated" "gen02 corpus" || exit 1
if [ ! -f "$MODEL_DIR/bbnet_14x7_gen01.pt" ]; then
    log "gen01 .pt missing — cannot run the warm-start pass"
    exit 1
fi
stopped || pass gen02 gen02 "$MODEL_DIR/bbnet_14x7_gen01.pt" 2e-4 || log "gen02 pass did not complete"

log "=== watcher done ==="
for sub in gen01 gen02; do
    [ -e "$OUT_ROOT/$sub/.done" ] || continue
    echo "" >> "$WLOG"
    echo "########## $sub ##########" >> "$WLOG"
    "$MAIN/train/.venv/bin/python" "$WORKTREE/scripts/exp036_report.py" "$OUT_ROOT/$sub" >> "$WLOG" 2>&1
done
