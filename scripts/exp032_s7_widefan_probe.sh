#!/usr/bin/env bash
# Plan 032 #7 gate, wide-fan half — convergence probe on mid-turn roots.
#
# The plan-028 dump (runs/exp-conv/s0a-raw-c10.jsonl) only holds turn-start
# activation roots (4-21 legal actions), so audit_q_target.py on it says
# nothing about the >30-fan regime where D2 put the visit-target defect.
# This run advances one production decision from each random start and keeps
# only roots whose pruned fan is >= 30, at budgets 1000 (candidate) and 16000
# (reference) — the two the gate compares.
#
# Single-threaded (one search at a time, 1 worker, tract CPU), so it can run
# beside a 6-way eval stage; expect ~2 h.
#
#   nohup scripts/exp032_s7_widefan_probe.sh > /dev/null 2>&1 &
source "$(dirname "$0")/exp032_lib.sh"

CHAMP="$MODELS/bbnet_14x7_gen03.onnx"
STATES="${STATES:-100}"
REPEATS="${REPEATS:-3}"
BUDGETS="${BUDGETS:-1000,16000}"
SEED="${SEED:-91000000}"
MIN_LEGAL="${MIN_LEGAL:-30}"
TAG="s7-widefan-c10"
F="$OUT/$TAG.jsonl"

log "=== #7 gate: wide-fan convergence probe ($STATES seeds, --advance 1 --min-legal $MIN_LEGAL, budgets $BUDGETS, x$REPEATS) ==="
[ -e "$F.done" ] && { log "$TAG done already"; exit 0; }
t0=$SECONDS
if "$UI" convergence --states "$STATES" --repeats "$REPEATS" --budgets "$BUDGETS" \
        --seed "$SEED" --evaluator nn --model "$CHAMP" --puct-mode raw --puct-c 10 \
        --advance 1 --min-legal "$MIN_LEGAL" --out "$F" > "$OUT/$TAG.log" 2>&1; then
    touch "$F.done"
    log "$TAG done ($(((SECONDS - t0) / 60)) min, $(wc -l < "$F") rows, $(grep -c skipped "$OUT/$TAG.log") seeds skipped)"
    (cd "$REPO/scripts" && "$PY" audit_q_target.py "$F" --budget 1000) > "$OUT/audit_$TAG.txt" 2>&1 \
        && { log "--- audit_q_target $TAG ---"; head -20 "$OUT/audit_$TAG.txt" >> "$LOG"; } \
        || log "audit_q_target FAILED — see $OUT/audit_$TAG.txt"
else
    log "$TAG FAILED — see $TAG.log"; tail -3 "$OUT/$TAG.log" >> "$LOG"
fi
