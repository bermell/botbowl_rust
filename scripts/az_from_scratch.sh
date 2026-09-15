#!/usr/bin/env bash
# Launch train_loop.sh as a from-scratch AlphaZero run (2026-09-15).
#
# Why a fresh run rather than continuing runs/loop14x7: six engine bug fixes
# landed between bdc1ba3 and d39ef7b — ball bounce resolved after knockdowns,
# turn ordering after a touchdown, turnover on knocking yourself down, failed
# catch on a hand-off, and the kickoff setup order — so every corpus generated
# before them was produced under rules the engine no longer implements. The old
# data is kept (runs/loop14x7 and models/ are untouched) but nothing here reads
# it: not as a corpus, not as a warm start, not as the anchor.
#
# And no heuristic bootstrap. The old loop's gen-0 champion was trained on a
# corpus the *scripted* bot generated, which makes the scripted bot the teacher
# of everything downstream. This run starts from random weights instead
# (scripts/make_random_net.py) and lets search be the only source of signal —
# what AlphaZero actually specifies.
#
# What is deliberately NOT reverted to a literal AlphaZero reading:
#   * POLICY_TARGET=cq (not `visits`) — plan 032 #7. `visits` has not converged
#     at 1000 iterations over a wide fan (plan 031 D2), which is engine-
#     independent and still holds.
#   * --mode random-start, drive-bounded — the whole pipeline's value backfill
#     is drive-relative (plan 023), and it is what makes 4800 games/gen affordable.
# Both are noted so the deviation is a choice on the record, not an accident.
#
#   nohup scripts/az_from_scratch.sh > /dev/null 2>&1 &
#   tail -f runs/az14x7/status.md
#   touch runs/az14x7/STOP        # clean exit at the next phase boundary
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Overridable so a smoke test can point the whole thing at a scratch dir
# without editing this file; the defaults are the real run.
export RUN_DIR="${RUN_DIR:-$REPO/runs/az14x7}"
export MODEL_DIR="${MODEL_DIR:-$REPO/models/az}"   # never write next to the old nets
export NN_SOCKET="${NN_SOCKET:-/tmp/bbnn-az.sock}" # distinct from the old loop's socket

SEED_NET="$MODEL_DIR/bbnet_14x7_gen00"

# The gen-0 champion: random weights, so train_loop.sh's heuristic bootstrap
# (which only fires when no champion exists) is skipped entirely.
if [ ! -f "$SEED_NET.onnx" ]; then
    "$REPO/train/.venv/bin/python" "$REPO/scripts/make_random_net.py" \
        --out "$SEED_NET" --seed 0
fi

export INIT_CHAMPION="$SEED_NET.onnx"
# gen01 must train from random init at SCRATCH_LR, not fine-tune the seed at
# WARM_LR. gen02+ warm-start from gen01 as usual.
export NO_WARM_FROM="$SEED_NET.pt"
# The frozen yardstick is the random net: a true zero, on the fixed engine.
# Re-anchor (add a second, overlap three generations, drop this one) once the
# rolling mean passes ~0.75 — a saturated anchor stops discriminating, and
# this one will saturate sooner than gen03 did.
export ANCHOR="$SEED_NET.onnx"
# Back on, because they discriminate again. They were dropped when the trained
# nets saturated them (`random` read 1.000 nine generations running); a net
# starting from random weights will not, so they are the absolute yardstick
# for the early part of this curve. Drop them again once they saturate.
export EVAL_RUNGS="random,scripted"

exec "$REPO/scripts/train_loop.sh" "$@"
