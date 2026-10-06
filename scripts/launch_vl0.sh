#!/usr/bin/env bash
# The loop after plan 049's fixes, from gen21, into its own run dir so nothing from loopmix16x9 is
# reused or overwritten.
#
#   - the search: exact roll model (the default since a130cb0) and the virtual-loss fix (5e29d89),
#     which alone was worth 0.686 on the same net (exp054)
#   - the target: cq tau 20 (plan 049 #1: +0.063 over tau 100, 3-seed control)
#   - the corpus: self-play exploration (plan 048: +0.065 over greedy)
#   - the benchmark: vs the frozen gen13 anchor, 200 per board on 14x7 and 16x9, so one generation
#     resolves ~0.05. Its origin is gen21 under this same search: scripts/vl0_baseline.sh, submitted
#     to the loop's hub alongside generation 1.
#
#   scripts/launch_vl0.sh                     # resumable: the markers make a relaunch pick up
#   touch runs/loopmix16x9vl0/STOP            # exits at the next phase boundary
set -eu
cd "$(dirname "$0")/.."

export SIZE_MODE=centred
export BUILD_W=16 BUILD_H=9 BUILD_PLAYERS=6
# The size centre starts at the cap loopmix16x9 reached (it advanced to 144 by gen03 and stayed).
export SIZE_CENTRE=144 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=144
export SIZE_MIN_AREA=70
export TIER_OVERRIDE=mix16x9vl0
export MODEL_DIR="$PWD/models/az_v7"
export INIT_CHAMPION="$PWD/models/az_v7/bbnet_mix16x9_gen21.onnx"
export ANCHOR="$PWD/models/az_v7/anchor_mix16x9_gen13.onnx"
export ANCHOR_GAMES=200
export EVAL_BOARD_SIZES=14x7,16x9
export EVAL_RUNGS=
export EVAL_GAMES=30
export MCTS_ITERS=500
export EVAL_MCTS_ITERS=500
export BLOOD_MCTS_BUDGET=visits
export CQ_TAU=20
export EXPLORE_ARGS="--explore-noise 0.25 --explore-alpha 10 --explore-sample-moves 2 --explore-temperature 1"
export GEN_PARALLEL_GAMES=16
export HUB_PORT=13337

exec scripts/train_loop.sh "$@"
