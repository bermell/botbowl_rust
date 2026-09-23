#!/usr/bin/env bash
# Launch the board-size-curriculum run (plan 042 arm B) warm-started from the
# az14x7v6 champion, migrated v6 -> v7.
#
#   nohup scripts/launch_mix16x9.sh > /dev/null 2>&1 &
#   tail -f runs/loopmix16x9/status.md
#   touch runs/loopmix16x9/STOP        # clean stop at the next phase boundary
#
# This is a thin env wrapper around train_loop.sh — every knob below is one of
# that script's, and the reasons live there. Only the choices specific to *this*
# run are commented here.
set -eu
cd "$(dirname "$0")/.."

# Board sizes: plan 042's arm B. Centre starts at 98 (= 14x7, where the
# migrated champion is strong) and size_curriculum.py advances it toward
# 16x9 = 144 as the corpus TD rate clears the gen00 baseline.
export SIZE_MODE=centred
export BUILD_W=16 BUILD_H=9 BUILD_PLAYERS=6
export SIZE_CENTRE=98 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=144

# Keep this run's nets out of models/ root, where the v6-schema nets live. A
# v7 net and a v6 net are not interchangeable and must not share a directory.
export MODEL_DIR="$PWD/models/az_v7"

# The migrated az14x7v6 gen23 champion (0.725 +/- 0.062 vs that run's gen07
# anchor, its best generation). bbnn.migrate verified the v6 -> v7 step
# function-preserving, so generation starts at exactly the strength it had.
# train_loop.sh derives the warm-start .pt as ${champion%.onnx}.pt, which is
# the migrated .pt sitting next to it.
export INIT_CHAMPION="$PWD/models/az_v7/bbnet_14x7_gen23_v7.onnx"

# Frozen benchmark opponent = the starting champion itself, so the curve reads
# "improvement over the v6 champion" and gen01 sits at 0.5 by construction.
# The old run's gen07 anchor was the alternative (it would keep the numbers on
# the old scale) but gen23 already read 0.725 against it, past the ~0.75
# re-anchor threshold in train_loop.sh — it would saturate within a few
# generations. Never retrain or overwrite this file.
export ANCHOR="$PWD/models/az_v7/bbnet_14x7_gen23_v7.onnx"
export ANCHOR_GAMES=40

# Fixed eval set, independent of the training centre — that independence is
# what keeps the per-size ladder honest (plan 042). E0 says whether 12x5 earns
# its place; drop it here if not.
export EVAL_BOARD_SIZES=12x5,14x7,16x9
export EVAL_RUNGS=scripted
export EVAL_GAMES=30

exec scripts/train_loop.sh "$@"
