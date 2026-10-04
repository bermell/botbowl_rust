#!/usr/bin/env bash
# Plan 054 E8: relaunch the Gumbel loop with the train step that absorbs the search's improvement.
# Called by scripts/exp063_plan054.sh once E2's drives pick a recipe; can be run by hand:
#
#   scripts/launch_plan054.sh <init net .onnx> <cq tau>
#
# Same generation and benchmark as runs/loopmix16x9g (scripts/launch_gumbel.sh): Gumbel m=16 at
# 1000 descents, drives vs the fixed d1k gen04 anchor on its gen04g contested sets, SPRT 0.5:0.55.
# What changes is the train step (plan 054):
#   - cq tau from E2 (CQ_TAU), warm fine-tunes at lr 5e-5 (WARM_LR) instead of 2e-4;
#   - the warm start is a restore candidate, with validation at steps 250/500/1000, every 1000
#     after, and at the last step (--init-candidate --eval-at, EVAL_EVERY=1000);
#   - restore on val_policy alone (SELECT_ON=policy);
#   - FREEZE_BN as in the Gumbel loop; the E1 absorption probe on status.md every generation.
# A fresh run dir, so the window holds only data generated under this recipe's nets.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
INIT="${1:?init net .onnx}"; TAU="${2:?cq tau}"
case "$INIT" in /*) ;; *) INIT="$REPO/$INIT" ;; esac
[ -f "$INIT" ] && [ -f "${INIT%.onnx}.pt" ] || { echo "init net or its .pt missing: $INIT" >&2; exit 1; }
RUN="$REPO/runs/loopmix16x9g054"; mkdir -p "$RUN"   # = runs/loop$TIER, where train_loop.sh puts it
LOG="$RUN/launch.log"; say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
M="$REPO/models/az_v7"; POS="$REPO/runs/loopmix16x9g/positions"
GUMBEL_EVAL="$REPO/cfgs/gumbel16_f1000.toml"; GUMBEL_GEN="$REPO/cfgs/gumbel16_f1000_gen.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
git diff --quiet || { say "FATAL: dirty tree"; exit 1; }
say "start: commit $(git rev-parse --short HEAD), init $(basename "$INIT"), cq tau $TAU"
[ -e "$RUN"/.mirror.done ] || echo "skipped: drives only" > "$RUN"/.mirror.done

export SIZE_MODE=centred BUILD_W=16 BUILD_H=9 BUILD_PLAYERS=6
export SIZE_CENTRE=144 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=144 SIZE_MIN_AREA=70
export TIER_OVERRIDE=mix16x9g054
export MODEL_DIR="$M" INIT_CHAMPION="$INIT"
export ANCHOR="$M/anchor_mix16x9_gen13.onnx" ANCHOR_EVERY=0 P1_GAMES=0 EVAL_BOARD_SIZES=14x7,16x9 EVAL_RUNGS= EVAL_GAMES=30
export MCTS_ITERS=1000 EVAL_MCTS_ITERS=1000 GEN_BOT_CONFIG="$GUMBEL_GEN" EVAL_BOT_CONFIG="$GUMBEL_EVAL" EXPLORE_ARGS=""
export EVAL_VENUE=drives DRIVE_REF="$M/bbnet_mix16x9d1k_gen04.onnx" DRIVE_SPRT=0.5:0.55 DRIVE_CAP=800
export DRIVE_POSITIONS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
export ORIGIN_EVERY=0
export CQ_TAU="$TAU" WARM_LR=5e-5 SELECT_ON=policy EVAL_EVERY=1000 ABSORB_PROBE=on
export TRAIN_EXTRA_ARGS="--freeze-bn --init-candidate --eval-at 250,500"
export GAMES_PER_SHARD=300 GEN_PARALLEL_GAMES=12 WORKER_MEM_FLOOR_MB=1536 HUB_PORT=13337

# A remote worker is admitted on any commit since the last change to the code its games run.
LAST_GAME=$(git log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$(git rev-list --abbrev-commit "$LAST_GAME"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > hub-allowed-commits.toml
say "allowlist: commits since $LAST_GAME; launching train_loop.sh into $RUN"
exec systemd-inhibit --what=sleep:idle --who=train_loop.sh --why="botbowl plan-054 loop" --mode=block scripts/train_loop.sh
