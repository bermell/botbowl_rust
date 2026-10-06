#!/usr/bin/env bash
# The Gumbel loop (plan 053 step 3), decided with the user on 2026-10-03:
#   - a fresh run dir (runs/loopmix16x9g) from gen04 of runs/loopmix16x9d1k, so no PUCT corpus
#     ever enters a training window: gen01 trains on gen01's Gumbel games alone;
#   - generation with cfgs/gumbel16_f1000_gen.toml at 1000 descents (Gumbel noise is the
#     exploration; EXPLORE_ARGS off);
#   - each generation benchmarked on contested drives against gen04, the net that generated the
#     first Gumbel data, kept fixed as the anchor (the user, 2026-10-04: a step-by-step 0.55
#     against the previous net is unlikely; against a fixed anchor it should eventually show).
#     Both sides on cfgs/gumbel16_f1000.toml at 1000 descents, positions screened with gen04
#     under Gumbel, SPRT 0.5:0.55. Drives only.
#   - FREEZE_BN=1 adds `--freeze-bn` to every training (exp061 decides).
#
# First screens contested positions for gen04 under the Gumbel search (the same 500 candidates
# per board as plan 051), on a temporary hub; then hands over to train_loop.sh.
#
#   FREEZE_BN=1 scripts/launch_gumbel.sh
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
RUN="$REPO/runs/loopmix16x9g"; POS="$RUN/positions"; mkdir -p "$POS"
LOG="$RUN/launch.log"; say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
M="$REPO/models/az_v7"; GEN04="$M/bbnet_mix16x9d1k_gen04.onnx"
GUMBEL_EVAL="$REPO/cfgs/gumbel16_f1000.toml"; GUMBEL_GEN="$REPO/cfgs/gumbel16_f1000_gen.toml"
CAND="$REPO/runs/plan051_proxy/positions"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-launch-gumbel.sock
PY="$REPO/train/.venv/bin/python"
git diff --quiet || { say "FATAL: dirty tree"; exit 1; }
say "start: commit $(git rev-parse --short HEAD), FREEZE_BN=${FREEZE_BN:-0}"
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && { say "FATAL: a hub already serves $HUB_URL"; exit 1; }
cargo build --release -p botbowl-hub -p botbowl-worker -p botbowl-ui -p botbowl-nn >> "$RUN/build.log" 2>&1 || { say "FATAL: build"; exit 1; }

# A remote worker is admitted on any commit since the last change to the code its games run.
LAST_GAME=$(git log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$(git rev-list --abbrev-commit "$LAST_GAME"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > hub-allowed-commits.toml
say "allowlist: commits since $LAST_GAME (last game-code change)"

# ---- screen contested positions for gen04 under Gumbel ----------------------------------------------
if [ ! -s "$POS/contested_16x9_gen04g.json" ]; then
    HUB_PID="" NN_PID="" WORKER_PID=""
    trap 'for p in $WORKER_PID $NN_PID $HUB_PID; do kill $p 2>/dev/null; done; rm -f "$SOCK"' EXIT
    "$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$RUN" >> "$RUN/screen.hub.log" 2>&1 & HUB_PID=$!
    for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$GEN04" --max-models 4 \
        --stats-every 300 --canvas 11x18 >> "$RUN/screen.nn_server.log" 2>&1 & NN_PID=$!
    for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
    "$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 10 --mem-floor-mb 1536 \
        --cache-dir "$RUN/worker-cache" --nn-server "$SOCK" >> "$RUN/screen.worker.log" 2>&1 & WORKER_PID=$!
    mkdir -p "$POS/screen"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "screen positions with gen04 under Gumbel" \
        --evaluator nn --model "$GEN04" --bot-config "$GUMBEL_EVAL" --mcts-iters 1000 \
        --vs-evaluator nn --vs-model "$GEN04" --vs-config "$GUMBEL_EVAL" --opponent-iters 1000 \
        --games 30 --seed 53000 --skip-fixed-rungs \
        --positions "$CAND/cand_14x7.json,$CAND/cand_16x9.json" --vs-games 2000 \
        --per-game-out "$POS/screen/eval.games.jsonl" --out "$POS/screen/report.json" --wait > "$POS/screen/eval.log" 2>&1 \
        || { say "FATAL: screen job"; exit 1; }
    for b in 14x7 16x9; do
        "$PY" "$REPO/scripts/positions_screen.py" --set "$CAND/cand_$b.json" --games "$POS/screen/eval.games.jsonl" \
            --reference "$GEN04" --band 0.25:0.75 --min-playouts 4 --name "contested_${b}_gen04g" \
            --out "$POS/contested_${b}_gen04g.json" >> "$LOG" 2>&1 || { say "FATAL: screen $b"; exit 1; }
    done
    for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    trap - EXIT; rm -f "$SOCK"; sleep 5
    say "screen done"
fi

# No heuristic mirror match: train_loop.sh's pre-flight plays 100 *full games*, and this phase is
# drives only (the user's rule). Paired matches cancel any seat bias anyway (plan 032). Until
# train_loop.sh defaults it off, mark it done.
mkdir -p "$RUN"; [ -e "$RUN"/.mirror.done ] || echo "skipped: drives only" > "$RUN"/.mirror.done
# ---- the loop ----------------------------------------------------------------------------------------
export SIZE_MODE=centred BUILD_W=16 BUILD_H=9 BUILD_PLAYERS=6
export SIZE_CENTRE=144 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=144 SIZE_MIN_AREA=70
export TIER_OVERRIDE=mix16x9g
export MODEL_DIR="$M" INIT_CHAMPION="$M/bbnet_mix16x9d1k_gen04.onnx"
export ANCHOR="$M/anchor_mix16x9_gen13.onnx" ANCHOR_EVERY=0 P1_GAMES=0 EVAL_BOARD_SIZES=14x7,16x9 EVAL_RUNGS= EVAL_GAMES=30
export MCTS_ITERS=1000 EVAL_MCTS_ITERS=1000 GEN_BOT_CONFIG="$GUMBEL_GEN" EVAL_BOT_CONFIG="$GUMBEL_EVAL" EXPLORE_ARGS=""
export EVAL_VENUE=drives DRIVE_REF="$M/bbnet_mix16x9d1k_gen04.onnx" DRIVE_SPRT=0.5:0.55 DRIVE_CAP=800
export DRIVE_POSITIONS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
export ORIGIN_EVERY=0
export CQ_TAU=100 GAMES_PER_SHARD=300 GEN_PARALLEL_GAMES=12 WORKER_MEM_FLOOR_MB=1536 HUB_PORT=13337
export TRAIN_EXTRA_ARGS="$([ "${FREEZE_BN:-0}" = 1 ] && echo --freeze-bn)"
# Again, right before the loop: its hub runs whatever HEAD is now, and a commit made during the
# screen (2026-10-04: a docs commit) left the earlier file keyed to a stale hub commit, so the
# laptop sat rejected for an hour.
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$(git rev-list --abbrev-commit "$LAST_GAME"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > hub-allowed-commits.toml
say "launching train_loop.sh into $RUN"
exec systemd-inhibit --what=sleep:idle --who=train_loop.sh --why="botbowl Gumbel loop" --mode=block scripts/train_loop.sh
