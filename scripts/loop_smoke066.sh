#!/usr/bin/env bash
# Plan 066 phase 5 smoke test: one tiny generation of the 20x11 loop (launch_plan065.sh's settings,
# shrunk) through scripts/train_loop.sh with NN_BACKEND=inproc, in a scratch run dir with its own
# models dir, hub port and STOP file. Every phase that runs a net must show it ran in-process
# (NN_INPROC lines, no NN_SERVER_FALLBACK, no sidecar process); a sampler records peak GPU memory
# (whole card and per process), the box's lowest MemAvailable and the largest process RSS.
#
#   scripts/loop_smoke066.sh SMOKE_DIR        # e.g. runs/exp077/loop_smoke
#
# Reads (never writes) the live run's nets, position sets and value benchmarks; symlinks the nets
# into SMOKE_DIR/models so their .ts exports land there, not in models/.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')"   # the main checkout (runs/, models/, venv)
S="$(mkdir -p "${1:?smoke dir}" && cd "$1" && pwd)"
M="$MAIN/models/az_v7"; mkdir -p "$S/models" "$S/run"
for n in bbnet_mix16x9v9_gen44 bbnet_mix16x9d1k_gen04_v9 anchor_mix16x9_gen13_v9; do
    for x in onnx pt; do ln -sf "$M/$n.$x" "$S/models/$n.$x"; done
done
touch "$S/run/.mirror.done"
[ -s "$S/run/champion.txt" ] || echo "$S/models/bbnet_mix16x9v9_gen44.onnx" > "$S/run/champion.txt"
echo 220 > "$S/run/size_centre.txt"

export PY="$MAIN/train/.venv/bin/python"
export NN_BACKEND=inproc
export SIZE_MODE=centred BUILD_W=20 BUILD_H=11 BUILD_PLAYERS=7 NN_CANVAS=13x22
export SIZE_CENTRE=220 SIZE_TEMPERATURE=0.3 SIZE_FLOOR=0.2 SIZE_MAX_AREA=220 SIZE_MIN_AREA=70
export SIZE_ASPECT=1.5-2.8 SIZE_CELLS_PER_PLAYER=26
export TIER_OVERRIDE=smoke RUN_DIR_OVERRIDE="$S/run" MODEL_DIR="$S/models"
export INIT_CHAMPION="$S/models/bbnet_mix16x9v9_gen44.onnx"
export ANCHOR="$S/models/anchor_mix16x9_gen13_v9.onnx" ANCHOR_EVERY=0 P1_GAMES=0 EVAL_BOARD_SIZES=16x9,20x11 EVAL_GAMES=4
export MCTS_ITERS="${MCTS_ITERS:-200}" EVAL_MCTS_ITERS="${EVAL_MCTS_ITERS:-200}"
export GEN_BOT_CONFIG="$REPO/cfgs/chance_all_partial_gen.toml" EVAL_BOT_CONFIG="$REPO/cfgs/chance_all_partial.toml"
export NET_CHECK_CONFIG="$REPO/cfgs/chance_all_partial.toml"
export EVAL_VENUE=drives DRIVE_REF="$S/models/bbnet_mix16x9d1k_gen04_v9.onnx" DRIVE_SPRT=0.5:0.55 DRIVE_CAP="${DRIVE_CAP:-8}"
export DRIVE_POSITIONS="$MAIN/runs/loopmix16x9g/positions/contested_16x9_gen04g.json,$MAIN/runs/loopmix16x9v9/positions/contested_20x11_gen43.json"
export ORIGIN_EVERY=0
export CQ_TAU=100 WARM_LR=5e-5 SELECT_ON=combined EVAL_EVERY=1000 ABSORB_PROBE=on EPOCHS=1
export TRAIN_EXTRA_ARGS="--freeze-bn"
export VALUE_BLEND=1.0 MC_LABEL_PLAYOUTS=2 MC_LABEL_PARALLEL=48 NEXT_DRIVE=1 DRIVE_EVAL_EVERY=1
export LABEL_PARALLEL_GAMES=48 LABEL_VIA_HUB=1 VALUE_WEIGHT=0.36 MAX_GENS=1 WINDOW_GENS=1
export VALUE_BENCH="$MAIN/runs/value_bench/v9_gen01_val.jsonl" VALUE_BENCH_2="$MAIN/runs/value_bench/v9_20x11_val.jsonl"
export NET_CHECK=on NET_CHECK_BOARDS=16x9,20x11 NET_CHECK_LADDER="64 250" OVERRIDES=4 PLAYOUTS=4
export GAMES_PER_SHARD="${GAMES_PER_SHARD:-2}" GEN_PARALLEL_GAMES=48 GEN_SIDECARS=2 WORKER_MEM_FLOOR_MB=1536 HUB_PORT=13398
export HUB_ALLOW_FROM=""
unset BLOOD_MCTS_BUDGET BLOOD_NN_SERVER

# The sampler: every 2 s, the card's used memory, each compute process's, MemAvailable, and the
# largest RSS among this run's processes; plus any nn_server.py that appears (there must be none).
(
    while :; do
        t=$(date +%s)
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
            | while IFS=', ' read -r p mib; do echo "$p/$(ps -o comm= -p "$p" 2>/dev/null)/$mib"; done | tr '\n' ';')
        avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
        rss=$(ps -eo rss=,comm= | awk '$2 ~ /botbowl|python/ {if ($1 > m) {m = $1; c = $2}} END {printf "%d:%s", m/1024, c}')
        side=$(ps -eo pid=,args= | awk '/scripts\/nn_server\.py/ && !/awk/ {print $1}' | tr '\n' ',')
        echo "$t used=$used avail=$avail maxrss=$rss apps=$apps sidecars=$side"
        sleep 2
    done
) > "$S/sampler.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
SECONDS=0
"$REPO/scripts/train_loop.sh" > "$S/train_loop.out" 2>&1
RC=$?
kill $SAMPLER 2>/dev/null
echo "train_loop.sh exit $RC after $((SECONDS / 60)) min" | tee "$S/result.txt"
awk '{for (i = 2; i <= NF; i++) { split($i, kv, "="); v[kv[1]] = kv[2] }
      split(v["used"], u, " "); if (u[1] > pu) pu = u[1]
      if (mn == "" || v["avail"] < mn) mn = v["avail"]
      split(v["maxrss"], r, ":"); if (r[1] > pr) { pr = r[1]; pc = r[2] }
      if (v["sidecars"] != "") sc++ }
     END { printf "peak GPU used %d MiB; min MemAvailable %d MB; peak process RSS %d MB (%s); samples with a sidecar running: %d\n", pu, mn, pr, pc, sc }' \
    "$S/sampler.log" | tee -a "$S/result.txt"
# Peak GPU memory per process (pid/command/MiB from the per-process column).
grep -o 'apps=[^ ]*' "$S/sampler.log" | sed 's/^apps=//' | tr ';' '\n' | grep . \
    | awk -F/ '{k = $1 "/" $2; if ($3 > p[k]) p[k] = $3} END {for (k in p) print "  " k " peak " p[k] " MiB"}' \
    | sort | tee -a "$S/result.txt"
exit $RC
