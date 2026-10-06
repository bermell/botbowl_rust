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
# Plan 042 E0 (2026-09-23): boards below area 70 are 2v2 under the density
# rule, and the champion's attack ratio there was 0.95x against a scripted
# mirror — worse than two scripted bots — against 1.60-1.97x everywhere else.
# Skill does not express, so 70 keeps those ~12% of games out of the corpus.
export SIZE_MIN_AREA=70

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
#
# Re-anchored from gen14 (2026-09-28) on a frozen copy of gen13: against gen23, gen13 read 0.710
# (14x7) / 0.735 (16x9) / 0.630 (12x9) at 500 iterations with the visits budget, close enough to
# the ~0.75 saturation point that further gains would have stopped showing. gen10-13 against gen23
# is the last stretch of the old curve (on the 500-iteration scale); gen14+ reads against gen13
# and starts near 0.5 again. The copy has its own name so the curve (keyed on the anchor's file
# name) starts fresh and no generation export can ever overwrite it; it is read-only.
export ANCHOR="$PWD/models/az_v7/anchor_mix16x9_gen13.onnx"
# 100 per board from gen08 (2026-09-27): the curve had sat at 0.51-0.60 pooled since gen03, and
# at 40 per board (pooled SE ~0.039) a 0.03-0.05 step is noise. 100 gives SE ~0.025 pooled.
# Seeds are --seed 0 + game index, so the first 40 are the games gen01-07 played.
export ANCHOR_GAMES=100

# Fixed eval set, independent of the training centre — that independence is
# what keeps the per-size ladder honest (plan 042). E0 answered the question
# the plan left open: 12x5 does not earn its place. At 120 games the champion
# read 0.662 there against 0.958/0.938 at 14x7/16x9, but E1's scripted mirror
# put up 15.17 TD/g on the same board — it is a 2v2 free-for-all that does not
# discriminate between bots, so it would only add noise to the curve.
# 12x9 is in as the held-out probe: outside the aspect band (1.33), so no arm
# ever trains on it, and E0 measured it at 0.950 zero-shot.
export EVAL_BOARD_SIZES=14x7,16x9,12x9
# No fixed rungs: scripted read 0.87-0.98 on every board by gen06 (30 games each), so it no
# longer separates generations and cost ~90 games of eval per generation. The anchor is the curve.
export EVAL_RUNGS=
export EVAL_GAMES=30

# Plan 045 (2026-09-26, net gen06): on 16x9, 500 vs 1000 iterations scored 0.500 +/- 0.041,
# while 250 vs 1000 fell to ~0.40, and 2000/4000 vs 1000 gained nothing significant on
# either board. 500 is the knee, at about half the cost of each game.
export MCTS_ITERS=500
# The anchor benchmark ran at 1000 on both seats through gen09 (and the partial gen10 runs). From
# gen10 it runs at 500 (2026-09-27), because the eval now shares the generation worker (plan 046
# item 0) and at 1000 an eval game's two full-game trees were predicted at 2-4.7 GB each, so the
# memory governor held games back at 10 streams. Plan 045 found 500 even with 1000, but read gen10+
# as a new scale, not a continuation.
export EVAL_MCTS_ITERS=500

# Grow each decision's root to MCTS_ITERS visits instead of adding MCTS_ITERS new descents on top
# of a reused tree (BudgetMode::Visits): -21% descents, -22% forwards and ~-22% peak memory in plan
# 046's measurement. The hub pins it into every job, both eval seats included. Turned on from gen11
# without the strength A/B plan 046 recommended (user's call, 2026-09-27); corpus labels read
# `visits=500`, so these gens can be told apart.
export BLOOD_MCTS_BUDGET=visits

# 16 local streams (up from the SIZE_MAX_AREA-scaled 10): the GPU batches grow with the number of
# waiting requests. mem_governor is the backstop; watch `holding back` in generate.worker.log.
export GEN_PARALLEL_GAMES=16

# The port forwarded through the home NAT, so a worker on another network can
# dial in. The hub binds 0.0.0.0, so this is reachable from the internet with
# only the shared token in front of it — the worker protocol is plain ws://,
# unencrypted. Fine for handing out self-play games; don't put anything else
# on this port.
export HUB_PORT=13337

exec scripts/train_loop.sh "$@"
