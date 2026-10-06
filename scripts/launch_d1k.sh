#!/usr/bin/env bash
# The loop after exp056/057: generation at 1000 *real descents* (was 500 visits, about 255
# descents under the fixed search), from gen21, into runs/loopmix16x9d1k.
#
#   - budget: 1000 descents everywhere. On 14x7, exp056 drives (500 and 2000 descents both H1 vs
#     500 visits) and exp057 convergence (regret 27 -> 14 from 250 to 1000, flattening after) agree.
#     On 16x9 the wide roots did not converge below 4000 (exp057), and 4000 is unaffordable here:
#     both sides search at the generation budget and the trees cap the worker at about 4
#     streams, so it costs about 15 h per generation. GEN_SPLIT in train_loop.sh is ready if a
#     per-board split is wanted later; Gumbel root selection is the candidate fix for wide fans.
#   - target: cq tau 100 (the loop default). exp055 (full games) and exp056 (drives) both favoured
#     it over tau 20 at the old budget; per the user, tau gets revisited on corpora at the new budget.
#   - exploration as in the vl0 loop (plan 048).
#   - eval (from gen03, 2026-10-03): plan 051 drives. Each generation plays paired contested
#     drives against gen21 (the sets in cfgs/positions were screened with gen21), SPRT 0.5:0.55,
#     at most 800 drives per set, at 500 *visits*, both seats. Drives only: no full-game
#     confirmation and no anchor match until the user asks for full games.
#   - smaller generations: 2400 games (300 per shard) at four times the old per-game cost.
#
# Waits for exp058 to finish, or stops it at EXP058_DEADLINE (a directional read is enough), then
# writes the commit allowlist so a laptop built at d37a1dc (scripts-only diff) can join, and launches.
#
#   scripts/launch_d1k.sh
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
DEADLINE="${EXP058_DEADLINE:-01:30}"
LOG="$REPO/runs/launch_d1k.log"; say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "start: commit $(git rev-parse --short HEAD); waiting for exp058 until $DEADLINE"
DEADLINE_S=$(date -d "$DEADLINE" +%s); [ "$DEADLINE_S" -lt "$(date +%s)" ] && DEADLINE_S=$(date -d "tomorrow $DEADLINE" +%s)
until tail -1 runs/exp058/status.md 2>/dev/null | grep -qE "\] (done$|FATAL)"; do
    if [ "$(date +%s)" -ge "$DEADLINE_S" ]; then
        say "deadline reached; stopping exp058 for a directional read"
        # The script's trap waits for its foreground child, so stop the job client and the rest too.
        for pat in "exp058_16x9_budget4000.sh" "botbowl-hub job eval" "bbnn-exp058" "botbowl-worker --hub ws://127.0.0.1:13337" "botbowl-hub serve --bind 0.0.0.0:13337"; do
            pgrep -f "$pat" | grep -v "^$$\$" | xargs -r kill 2>/dev/null
            sleep 2
        done
        echo "[$(date '+%F %T')] stopped at the deadline by launch_d1k.sh; read eval.games.jsonl for the interim" >> runs/exp058/status.md
        echo "[$(date '+%F %T')] done" >> runs/exp058/status.md
        break
    fi
    sleep 120
done
sleep 10
say "exp058 over: $(tail -3 runs/exp058/status.md | tr '\n' ' ' | cut -c1-400)"

# The laptop may be built at d37a1dc or any commit since (it gets rebuilt when master moves). None
# of them touches a worker's games (checked below), so admit them all: on 2026-10-03 a laptop
# rebuilt at 03d237c sat rejected for two hours because only d37a1dc was listed.
git diff --quiet d37a1dc HEAD -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts \
    && printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
        "$(git rev-list --abbrev-commit d37a1dc^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > hub-allowed-commits.toml \
    && say "allowlist: every commit since d37a1dc admitted for hub $(git rev-parse --short HEAD)"

# No heuristic mirror match: train_loop.sh's pre-flight plays 100 *full games*, and this phase is
# drives only (the user's rule). Paired matches cancel any seat bias anyway (plan 032). Until
# train_loop.sh defaults it off, mark it done.
mkdir -p runs/loopmix16x9d1k; [ -e runs/loopmix16x9d1k/.mirror.done ] || echo "skipped: drives only" > runs/loopmix16x9d1k/.mirror.done
mkdir -p runs/loopmix16x9d1k
[ -d runs/loopmix16x9d1k/baseline_gen21 ] || cp -r runs/loopmix16x9vl0/baseline_gen21 runs/loopmix16x9d1k/

export SIZE_MODE=centred
export BUILD_W=16 BUILD_H=9 BUILD_PLAYERS=6
export SIZE_CENTRE=144 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=144 SIZE_MIN_AREA=70
export TIER_OVERRIDE=mix16x9d1k
export MODEL_DIR="$REPO/models/az_v7"
export INIT_CHAMPION="$REPO/models/az_v7/bbnet_mix16x9_gen21.onnx"
export ANCHOR="$REPO/models/az_v7/anchor_mix16x9_gen13.onnx"
export ANCHOR_GAMES=200 EVAL_BOARD_SIZES=14x7,16x9 EVAL_RUNGS= EVAL_GAMES=30
export MCTS_ITERS=1000 GEN_BUDGET_MODE=iterations
export EVAL_MCTS_ITERS=500 EVAL_BUDGET_MODE=visits
export CQ_TAU=100
export EVAL_VENUE=drives DRIVE_REF="$INIT_CHAMPION" DRIVE_SPRT=0.5:0.55 DRIVE_CAP=800 P1_GAMES=0 ANCHOR_EVERY=0
export DRIVE_POSITIONS="$REPO/cfgs/positions/contested_14x7.json,$REPO/cfgs/positions/contested_16x9.json"
export EXPLORE_ARGS="--explore-noise 0.25 --explore-alpha 10 --explore-sample-moves 2 --explore-temperature 1"
export GAMES_PER_SHARD=300
export GEN_PARALLEL_GAMES=12 WORKER_MEM_FLOOR_MB=1536
export HUB_PORT=13337
say "launching train_loop.sh into runs/loopmix16x9d1k"
exec systemd-inhibit --what=sleep:idle --who=train_loop.sh --why="botbowl d1k loop" --mode=block scripts/train_loop.sh
