#!/usr/bin/env bash
# Plan 056: restart the Gumbel loop (the user, 2026-10-06) with everything plans 054-056 found:
#   - the mean backup at player nodes (hardcoded since ce4eda1; plan 055);
#   - plan 054's train step: lr 5e-5, the warm start a restore candidate (--init-candidate,
#     --eval-at 250,500, every 1000 after), --freeze-bn, cq tau 100; restore on val_policy +
#     val_value against MC-labelled val shards (SELECT_ON=combined; on val_policy alone gen02 restored
#     its init on a flat policy loss and threw the value head's progress away);
#   - plan 056's value label: Monte Carlo averages, 8 policy-only playouts per train sample under
#     the generator (MC_LABEL_PLAYOUTS=8, VALUE_BLEND=1.0);
#   - init from arm F (exp067: g_gen05 fine-tuned on gen07 with MC labels), the best net we have:
#     its search beat the control's 0.533 head to head and its own policy 0.582 at 1000 descents;
#   - per generation, on status.md: absorption probe, the value benchmark (seconds), and the
#     standing net check (plan 055 §6, in the background).
# Generation and benchmark as runs/loopmix16x9g: Gumbel m=16 at 1000 descents, drives vs the fixed
# d1k gen04 anchor on its gen04g contested sets, SPRT 0.5:0.55, gateless. A fresh run dir, so the
# window holds only data generated under this recipe.
#
#   scripts/launch_plan056.sh
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
M="$REPO/models/az_v7"; INIT="$M/plan056_armF.onnx"
[ -f "$INIT" ] && [ -f "${INIT%.onnx}.pt" ] || { echo "init net or its .pt missing: $INIT" >&2; exit 1; }
RUN="$REPO/runs/loopmix16x9g056"; mkdir -p "$RUN"   # = runs/loop$TIER, where train_loop.sh puts it
LOG="$RUN/launch.log"; say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
POS="$REPO/runs/loopmix16x9g/positions"
GUMBEL_EVAL="$REPO/cfgs/gumbel16_f1000.toml"; GUMBEL_GEN="$REPO/cfgs/gumbel16_f1000_gen.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
git diff --quiet || { say "FATAL: dirty tree"; exit 1; }
say "start: commit $(git rev-parse --short HEAD), init $(basename "$INIT")"
[ -e "$RUN"/.mirror.done ] || echo "skipped: drives only" > "$RUN"/.mirror.done

export SIZE_MODE=centred BUILD_W=16 BUILD_H=9 BUILD_PLAYERS=6
export SIZE_CENTRE=144 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=144 SIZE_MIN_AREA=70
export TIER_OVERRIDE=mix16x9g056
export MODEL_DIR="$M" INIT_CHAMPION="$INIT"
export ANCHOR="$M/anchor_mix16x9_gen13.onnx" ANCHOR_EVERY=0 P1_GAMES=0 EVAL_BOARD_SIZES=14x7,16x9 EVAL_RUNGS= EVAL_GAMES=30
export MCTS_ITERS=1000 EVAL_MCTS_ITERS=1000 GEN_BOT_CONFIG="$GUMBEL_GEN" EVAL_BOT_CONFIG="$GUMBEL_EVAL" EXPLORE_ARGS=""
export EVAL_VENUE=drives DRIVE_REF="$M/bbnet_mix16x9d1k_gen04.onnx" DRIVE_SPRT=0.5:0.55 DRIVE_CAP=800
export DRIVE_POSITIONS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
export ORIGIN_EVERY=0
export CQ_TAU=100 WARM_LR=5e-5 SELECT_ON=combined EVAL_EVERY=1000 ABSORB_PROBE=on
export TRAIN_EXTRA_ARGS="--freeze-bn --init-candidate --eval-at 250,500"
export VALUE_BLEND=1.0 MC_LABEL_PLAYOUTS=8 MC_LABEL_PARALLEL=16
export VALUE_BENCH="$REPO/runs/value_bench/g05_gen06.jsonl" NET_CHECK=on
export GAMES_PER_SHARD=300 GEN_PARALLEL_GAMES=12 WORKER_MEM_FLOOR_MB=1536 HUB_PORT=13337

# A remote worker is admitted on any commit since the last change to the code its games run, plus
# the commits exp066's hub admitted (8cc6ce7 and earlier): the one game-code commit since, 99ae314,
# only adds prepare's TD(lambda) labels, which no worker runs.
LAST_GAME=$(git log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$( (git rev-list --abbrev-commit "$LAST_GAME"^..HEAD; echo 8cc6ce7 2a8fe4e 14d915a e7d5c1e 381c245 e8c0144 d45e391 | tr ' ' '\n') \
        | sort -u | sed 's/.*/"&"/' | paste -sd,)" > hub-allowed-commits.toml
say "allowlist: commits since $LAST_GAME + exp066's; launching train_loop.sh into $RUN"
exec systemd-inhibit --what=sleep:idle --who=train_loop.sh --why="botbowl plan-056 loop" --mode=block scripts/train_loop.sh
