#!/usr/bin/env bash
# Plan 058: the loop on the new master (2026-10-06): 16 newly implemented skills, per-player kickoff
# setup, schema v9, hub protocol v14. Every earlier corpus is unusable there (format and rules), so
# this is a fresh run; the nets carry over through `bbnn.migrate` (v7 -> v9, exact).
#   - init: the g056 run's last generator, gen04 (= gen03's weights), migrated to v9
#     (`models/az_v7/bbnet_mix16x9g056_gen04_v9`): best value bench of that run, net-check slope 0.80;
#   - plan 056's recipe: mean backup, MC-averaged labels (8 policy-only playouts per train and val
#     sample), lr 5e-5, --freeze-bn, restore on val_policy + val_value, no init candidate;
#   - plan 047: --next-drive (the user): a drive that scores is followed through both kickoff setups
#     into the next drive, so per-player setup decisions reach the corpus (mc-label replays them);
#   - plan 058 throughput: 36 local generation streams (12 before; +40% measured, memory-capped),
#     mc-label at 96 threads (sample-level work, GPU-bound), drives vs the anchor every 3 generations;
#     value bench and net check every generation.
# The value benchmark is re-frozen under the new rules from gen01's held-out shards (VALUE_BENCH points
# at it; the loop skips the bench until the file exists).
#
#   scripts/launch_plan058.sh
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
M="$REPO/models/az_v7"; INIT="$M/bbnet_mix16x9g056_gen04_v9.onnx"
[ -f "$INIT" ] && [ -f "${INIT%.onnx}.pt" ] || { echo "init net or its .pt missing: $INIT" >&2; exit 1; }
RUN="$REPO/runs/loopmix16x9v9"; mkdir -p "$RUN"   # = runs/loop$TIER, where train_loop.sh puts it
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
export TIER_OVERRIDE=mix16x9v9
export MODEL_DIR="$M" INIT_CHAMPION="$INIT"
export ANCHOR="$M/anchor_mix16x9_gen13_v9.onnx" ANCHOR_EVERY=0 P1_GAMES=0 EVAL_BOARD_SIZES=14x7,16x9 EVAL_RUNGS= EVAL_GAMES=30
export MCTS_ITERS=1000 EVAL_MCTS_ITERS=1000 GEN_BOT_CONFIG="$GUMBEL_GEN" EVAL_BOT_CONFIG="$GUMBEL_EVAL" EXPLORE_ARGS=""
export EVAL_VENUE=drives DRIVE_REF="$M/bbnet_mix16x9d1k_gen04_v9.onnx" DRIVE_SPRT=0.5:0.55 DRIVE_CAP=800
export DRIVE_POSITIONS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
export ORIGIN_EVERY=0
export CQ_TAU=100 WARM_LR=5e-5 SELECT_ON=combined EVAL_EVERY=1000 ABSORB_PROBE=on
# No --init-candidate (the user, 2026-10-06): at lr 5e-5 the fine-tune moves val by less than noise,
# so the warm start won the restore on ties and gen02/gen04 never moved. Keep a trained checkpoint
# every generation, as AlphaZero does; play (drives, net check) judges whether it helped.
export TRAIN_EXTRA_ARGS="--freeze-bn --eval-at 250,500"
export VALUE_BLEND=1.0 MC_LABEL_PLAYOUTS=8 MC_LABEL_PARALLEL=96 NEXT_DRIVE=1 DRIVE_EVAL_EVERY=3
export VALUE_BENCH="$REPO/runs/value_bench/v9_gen01_val.jsonl" NET_CHECK=on
export GAMES_PER_SHARD=300 GEN_PARALLEL_GAMES=36 WORKER_MEM_FLOOR_MB=1536 HUB_PORT=13337

# A remote worker is admitted on any commit since the last change to the code its games run. The
# laptop must rebuild on this master (protocol v14, new rules).
LAST_GAME=$(git log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$(git rev-list --abbrev-commit "$LAST_GAME"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > hub-allowed-commits.toml
say "allowlist: commits since $LAST_GAME; launching train_loop.sh into $RUN"
exec systemd-inhibit --what=sleep:idle --who=train_loop.sh --why="botbowl plan-058 loop" --mode=block scripts/train_loop.sh
