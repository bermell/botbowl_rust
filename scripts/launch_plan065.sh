#!/usr/bin/env bash
# Plan 065: the v9 loop moves from boards up to 16x9/6 to a 20x11/7 capacity (the user, 2026-10-18:
# "bigger boards are the direction"). Same run, same recipe as scripts/launch_plan058.sh; what changes:
#   - capacity 20x11/7 (BUILD_*, BOARD_SIZE_*, target/20x11; the hub refuses workers of another
#     capacity *and* another env board, so a helper box builds and runs with BOARD_SIZE_W=20
#     BOARD_SIZE_H=11 BOARD_PLAYERS=7 too);
#   - the size mix centred on 20x11 (area 220, T 0.3, floor 0.2, areas 70..220, aspect 1.5-2.8,
#     26 cells a player): 20x11/7 11%, every 7-player board 51%, 16x9/6 4.7%, area > 144 72%
#     (plan 065 §3; 7 players turn the kickoff table on);
#   - the nets (in-process since plan 066; the sidecars before) pad onto a 13x22 canvas (the
#     capacity, border included; NN_CANVAS);
#   - drives on the 20x11 contested set next to the 16x9 one, once the 20x11 set exists (§4);
#   - the net check samples 16x9 and 20x11 decisions; a second value benchmark (VALUE_BENCH_2) on
#     20x11 states once it is frozen (§4);
#   - 150 games a shard instead of 400: a 20x11-mix game costs ~2.7x the GPU (§5).
# Continuing runs/loopmix16x9v9 is deliberate (§6): from this commit a stored GameState loads at any
# capacity its board fits, so the 3-generation window mixes the 16x9-capacity corpora with the new
# ones while it turns over, and the drive curve continues on the same 16x9 set and reference.
# The tier tag (model names bbnet_mix16x9v9_genNN) stays; it names the lineage, not the capacity.
#
# Switch at a generation boundary (plan 065 §7), never by merging under a running loop:
#   nohup scripts/stop_after.sh "genNN train:" runs/loopmix16x9v9 > /dev/null 2>&1 &   # placed during train,
#   (the loop exits at "before genNN eval"; wait for "STOP file present — exiting cleanly"), rm the STOP,
#   merge, commit, then:
#   nohup scripts/launch_plan065.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
M="$REPO/models/az_v7"; INIT="$M/bbnet_mix16x9g056_gen04_v9.onnx"   # only used if the run had no champion
RUN="$REPO/runs/loopmix16x9v9"; mkdir -p "$RUN"   # = runs/loop$TIER, where train_loop.sh puts it
LOG="$RUN/launch.log"; say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
[ -s "$RUN/champion.txt" ] || [ -f "$INIT" ] || { echo "no champion in $RUN and no init net $INIT" >&2; exit 1; }
POS="$REPO/runs/loopmix16x9g/positions"      # the 14x7/16x9 sets (plan 051)
POS20="$RUN/positions"                       # the 20x11 set (plan 065 §4)
GUMBEL_EVAL="$REPO/cfgs/chance_all_partial.toml"; GUMBEL_GEN="$REPO/cfgs/chance_all_partial_gen.toml"
export NET_CHECK_CONFIG="$GUMBEL_EVAL"
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 CARGO_TARGET_DIR="$REPO/target/20x11"
unset BLOOD_MCTS_BUDGET
source "$REPO/scripts/lib/git.sh"
require_clean_tree 2>>"$LOG" || { say "FATAL: dirty tree"; exit 1; }
# Before touching the run: train_loop.sh reuses any hub that already answers on the port. One left
# over from the 16x9 run (built at 16x9/6) would refuse every 20x11 worker, the local one included,
# and the generate phase would wait forever; and a 16x9 loop still running would read the new
# size_centre.txt below. Never launch over either.
if (exec 3<>"/dev/tcp/127.0.0.1/13337") 2>/dev/null; then
    say "FATAL: something still listens on port 13337 (the 16x9 loop's hub?); stop it first"
    echo "something still listens on port 13337; stop the old loop and hub first (plan 065 §7)" >&2
    exit 1
fi
say "start (plan 065, capacity 20x11/7): commit $(git rev-parse --short HEAD), champion $(cat "$RUN/champion.txt" 2>/dev/null || basename "$INIT")"
[ -e "$RUN"/.mirror.done ] || echo "skipped: drives only" > "$RUN"/.mirror.done

export SIZE_MODE=centred BUILD_W=20 BUILD_H=11 BUILD_PLAYERS=7 NN_CANVAS=13x22
export SIZE_CENTRE=220 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=220 SIZE_MIN_AREA=70
export SIZE_ASPECT=1.5-2.8 SIZE_CELLS_PER_PLAYER=26
# size_centre.txt holds the centre the curriculum last wrote (144, capped at the old max area); the
# loop only seeds it when empty. The centre never moves down and is capped at SIZE_MAX_AREA, so
# writing the new cap on every launch is idempotent.
echo "$SIZE_CENTRE" > "$RUN/size_centre.txt"
export TIER_OVERRIDE=mix16x9v9
export MODEL_DIR="$M" INIT_CHAMPION="$INIT"
export ANCHOR="$M/anchor_mix16x9_gen13_v9.onnx" ANCHOR_EVERY=0 P1_GAMES=0 EVAL_BOARD_SIZES=16x9,20x11 EVAL_RUNGS= EVAL_GAMES=30
export MCTS_ITERS=1000 EVAL_MCTS_ITERS=1000 GEN_BOT_CONFIG="$GUMBEL_GEN" EVAL_BOT_CONFIG="$GUMBEL_EVAL" EXPLORE_ARGS=""
# Measurement reset at gen48 (the user, 2026-10-21; relaunched 2026-10-23): drives against gen48 on
# contested sets screened under gen48 (20x11 at 1000 descents, 16x9 at 250), value benches re-frozen
# under gen48's MC labels, and the divergence gate's report (plan 068; free, `sprt` stays off). The
# gen04 curve ends at gen51; MEASURE=gen04 restores it.
export EVAL_VENUE=drives DRIVE_SPRT=0.5:0.55 DRIVE_CAP=800
if [ "${MEASURE:-gen48}" = gen48 ]; then
    export DRIVE_REF="$M/bbnet_mix16x9v9_gen48.onnx" DRIVE_GATE=report
else
    export DRIVE_REF="$M/bbnet_mix16x9d1k_gen04_v9.onnx"
fi
# 16x9 (the curve so far) and, once screened, 20x11. The loop refuses to start on a missing set, so
# the 20x11 one joins at the first launch after it exists (plan 065 §4 builds it).
if [ "${MEASURE:-gen48}" = gen48 ]; then
    SET20="$POS20/contested_20x11_gen48.json"
    export DRIVE_POSITIONS="$POS20/contested_16x9_gen48.json,$SET20"
else
SET20=$(ls -t "$POS20"/contested_20x11_*.json 2>/dev/null | head -1)
export DRIVE_POSITIONS="$POS/contested_16x9_gen04g.json${SET20:+,$SET20}"
fi
[ -n "$SET20" ] || say "no 20x11 contested set in $POS20 yet: drives on 16x9 only (plan 065 §4)"
export ORIGIN_EVERY=0
export CQ_TAU=100 WARM_LR=5e-5 SELECT_ON=combined EVAL_EVERY=1000 ABSORB_PROBE=on
export TRAIN_EXTRA_ARGS="--freeze-bn --eval-at 250,500"
export VALUE_BLEND=1.0 MC_LABEL_PLAYOUTS=8 MC_LABEL_PARALLEL=96 NEXT_DRIVE=1 DRIVE_EVAL_EVERY=3
# 2026-10-19: 96 label streams on 20x11 boards held 9.9 GB in the worker and swapped the 16 GB box
# (memory stalls ~1/3 of the time, labels at ~90 samples/min); 48 streams for the label worker.
export LABEL_PARALLEL_GAMES=48
export VALUE_WEIGHT=0.36
export MAX_GENS=60
export LABEL_VIA_HUB=1
# The v9 benchmark (14x7/16x9 states from gen01) stays the series; a binary of this capacity replays
# its 16x9-capacity corpus. VALUE_BENCH_2 is the 20x11 benchmark, frozen from the first 20x11
# generation's held-out shards (plan 065 §4); scored once the file exists.
export VALUE_BENCH="$REPO/runs/value_bench/v9_gen01_val.jsonl" NET_CHECK=on
export VALUE_BENCH_2="${VALUE_BENCH_2:-$REPO/runs/value_bench/v9_20x11_val.jsonl}"
if [ "${MEASURE:-gen48}" = gen48 ]; then
    export VALUE_BENCH="$REPO/runs/value_bench/v9_16x9_gen49_val.jsonl" VALUE_BENCH_2="$REPO/runs/value_bench/v9_20x11_gen49_val.jsonl"
fi
export NET_CHECK_BOARDS=16x9,20x11
# Plan 065 §5 (gen36 on CPU, 36 games a mix): a game of the 20x11 mix costs 1.90x the forwards of
# one of the 16x9 mix and each forward 1.44x on the 13x22 canvas -> 2.7x the GPU per game (MC labels
# 3.7x: quadratic in drive length). 150 games a shard keep the generate phase at today's length
# (MC labels ~1.4x, samples 0.65x at ~1.5x the cells each); 200 if a generation may be ~35% longer.
export GAMES_PER_SHARD="${GAMES_PER_SHARD:-150}" GEN_PARALLEL_GAMES=48 GEN_SIDECARS=2 WORKER_MEM_FLOOR_MB=1536 HUB_PORT=13337
# Plan 066 phase 5 (the user, 2026-10-19): the nets run in-process — one local worker per phase with
# all 48 generation / 48 label streams in one batch queue (2 lanes), every tool with
# --nn-backend inproc, NET.ts exported beside each NET.onnx by the loop; no Python sidecar. exp077:
# 0 differing bits against the sidecar, +12% GPU samples/s generating, +14% labelling, -1.9 GB RSS.
# NN_BACKEND=sidecar rolls back to the two sidecars of plan 058 (GEN_SIDECARS=2) exactly as before.
export NN_BACKEND="${NN_BACKEND:-inproc}"
export HUB_ALLOW_FROM=""

LAST_GAME=$(git log -1 --format=%h -- $(game_crates))
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$( (git rev-list --abbrev-commit "$LAST_GAME"^..HEAD; for c in ${EXTRA_ALLOW:-}; do echo "$c"; done) \
        | sed 's/.*/"&"/' | paste -sd,)" > hub-allowed-commits.toml
say "allowlist: commits since $LAST_GAME; drives on $(echo "$DRIVE_POSITIONS" | tr ',' '\n' | xargs -n1 basename | paste -sd' '); launching train_loop.sh into $RUN"
exec systemd-inhibit --what=sleep:idle --who=train_loop.sh --why="botbowl plan-065 loop (20x11/7)" --mode=block scripts/train_loop.sh
