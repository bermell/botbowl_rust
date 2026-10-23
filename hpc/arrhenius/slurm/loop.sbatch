#!/bin/bash
#SBATCH --job-name=botbowl-loop
# Plan 069 item 5: the training loop inside one Slurm job ("loop in a job"). Submit it with
# submit.sh (which adds -A, -p, -N, the task/GPU shape, -t, --signal and the output path from the
# env file), never by hand; `submit.sh --dry-run loop` prints this job's plan without Slurm.
#
# One srun step, one task per GH200 superchip (`--ntasks-per-node=4 -c 72`, a GPU per task, the
# centre's own GPU example). Task 0 (node 0, the hub's host) runs scripts/train_loop.sh in the
# image: the hub bound on 0.0.0.0:HUB_PORT, the trainer, the loop's per-phase local worker, the
# net check and value bench, all on its one GPU, exactly as on the training box. Every other task
# runs a long-lived `botbowl-worker` on its GPU, dialling ws://<node 0>:HUB_PORT/ws; the hub hands
# generate, label and drive work to all of them (task.sh).
#
# Resume and chaining: the loop resumes from its markers, so a job is just "run until told".
#   - LOOP_DEADLINE = this job's end: at a phase boundary with less time left than the next phase
#     took last time (x1.25, at least STOP_MARGIN_MIN) the loop exits 75 instead of starting a
#     phase the walltime would cut; a phase longer than a whole job exits 76 and stops the chain;
#   - --signal=B:USR1@SIGNAL_SECS (submit.sh) is the backstop: on USR1 (or TERM: scancel, the
#     walltime) the step gets TERM, the loop's trap stops its hub and worker, the workers stop,
#     and the job records "continue";
#   - CHAIN_LEFT > 0: the job queues its successor first thing (afterany on itself), so the next
#     one ages in the queue while this one runs. A job exits at once if the chain has been
#     stopped: `submit.sh stop` (= touch RUN_DIR/STOP: the loop exits at its next phase boundary),
#     the loop finished (MAX_GENS), or it died (chain.state says why; a human looks before more
#     hours burn).
# train_loop.sh's exit codes read here: 0 = STOP or finished, 75 = deadline, 76 = a phase cannot
# fit in a job, 143 = signalled, anything else = died (FATAL on status.md).
set -u
SLURM_DIR="${BOTBOWL_SLURM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=common.sh
source "$SLURM_DIR/common.sh"
: "${LOOP_ENV:?LOOP_ENV (the run env file, see loop.env.example) must be set}"
# shellcheck disable=SC1090
source "$LOOP_ENV"
slurm_defaults || exit 2
JOB="${SLURM_JOB_ID:-dry}"
JOB_TAG="loop $JOB"
LOGDIR="$STATE_DIR/logs/$JOB"
CHAIN_STATE="$STATE_DIR/chain.state"
mkdir -p "$LOGDIR" "$STATE_DIR/env"
dry || mkdir -p "$RUN_DIR" "$MODEL_DIR"
chain() { dry || echo "[$(date '+%F %T')] job $JOB: $*" >> "$STATE_DIR/chain.log"; jlog "$*"; }

# ---- may this job run at all? -----------------------------------------------------------------
if [ -e "$RUN_DIR/STOP" ]; then
    chain "STOP file present in $RUN_DIR: not starting (submit.sh resume to go on)"
    dry || echo "stop STOP file" > "$CHAIN_STATE"
    exit 0
fi
if grep -qs '^stop' "$CHAIN_STATE"; then
    chain "chain stopped earlier ($(cat "$CHAIN_STATE")): not starting (submit.sh resume to go on)"
    exit 0
fi

# ---- the successor, queued now so it ages while this job runs -----------------------------------
SUCC=""
if [ "$CHAIN_LEFT" -gt 0 ]; then
    ARGS_FILE="$STATE_DIR/sbatch.loop.args"
    if dry; then
        jlog "would queue the successor: sbatch --dependency=afterany:$JOB \$(cat $ARGS_FILE) loop.sbatch, CHAIN_LEFT=$((CHAIN_LEFT - 1))"
    elif [ -s "$ARGS_FILE" ]; then
        # shellcheck disable=SC2046
        if SUCC=$(sbatch --parsable --dependency="afterany:$JOB" \
                --export=ALL,LOOP_ENV="$LOOP_ENV",BOTBOWL_SLURM_DIR="$SLURM_DIR",CHAIN_LEFT=$((CHAIN_LEFT - 1)) \
                $(cat "$ARGS_FILE") "$SLURM_DIR/loop.sbatch"); then
            chain "queued successor $SUCC (afterany:$JOB, $((CHAIN_LEFT - 1)) more after it)"
        else
            SUCC=""
            chain "WARN: could not queue a successor (sbatch refused on a compute node? use submit.sh loop --upfront)"
        fi
    else
        chain "WARN: no $ARGS_FILE (submit with submit.sh): no successor queued"
    fi
fi

# ---- nodes, hub, token, deadline, limits --------------------------------------------------------
mapfile -t HOSTS < <(job_hosts)
NODE0="${HOSTS[0]}"
HUB_HOST="${HUB_HOST:-$NODE0}"
HUB_EXTERNAL="${HUB_EXTERNAL:-0}"
# Thousands of streams are a few hundred sockets plus shard files; the soft limit is often 1024.
ulimit -n "$(ulimit -Hn)" 2> /dev/null
ensure_token
LOOP_DEADLINE=$(job_end_epoch)
STOP_MARGIN_SECS=$((STOP_MARGIN_MIN * 60))
chain "start: ${#HOSTS[@]} node(s) ${HOSTS[*]} x $GPUS_PER_NODE tasks, $BIN_MODE binaries ($BIN), hub ${HUB_HOST}:$HUB_PORT$([ "$HUB_EXTERNAL" = 1 ] && echo " (external)"), ends $(date -d "@$LOOP_DEADLINE" '+%F %T')"
[ "$HUB_EXTERNAL" = 1 ] || dry || printf '%s %s %s\n' "$HUB_HOST" "$HUB_PORT" "$JOB" > "$STATE_DIR/hub.addr"
# The heuristic mirror match (a pre-flight from plan 021) says nothing new on a cluster run and
# costs ~16 core-hours; skipped as launch_plan065.sh skips it, unless SKIP_MIRROR=0.
if [ "${SKIP_MIRROR:-1}" = 1 ] && ! dry && [ ! -e "$RUN_DIR/.mirror.done" ]; then
    echo "skipped: plan 069 cluster run (SKIP_MIRROR)" > "$RUN_DIR/.mirror.done"
fi

ENVF="$STATE_DIR/env/job-$JOB.sh"
write_job_env "$ENVF"
show_env "$ENVF"

# ---- checkout mode: build the checkout's binaries in the image, once, before any task -----------
if [ "$BIN_MODE" = checkout ]; then
    # shellcheck disable=SC2016
    container_cmd "$ENVF" bash -c 'cd "$1" && C="--release --offline -j ${BUILD_JOBS:-32}" &&
        cargo build $C -p botbowl-hub && cargo build $C -p botbowl-nn -p botbowl-worker --features botbowl-worker/torch &&
        cargo build $C -p botbowl-ui --features botbowl-ui/torch' _ "$CHECKOUT"
    if dry; then
        echo "build (on the batch host, before the step):"
        launch build "$LOGDIR/build.log" "${C[@]}"
    elif ! "${C[@]}" >> "$LOGDIR/build.log" 2>&1; then
        chain "FATAL: the checkout build failed — see $LOGDIR/build.log"
        echo "stop build failed (job $JOB)" > "$CHAIN_STATE"
        [ -n "$SUCC" ] && scancel "$SUCC"
        exit 1
    fi
fi

# ---- the step: task 0 the loop, every other task a worker ----------------------------------------
dry && echo "the step: srun $SRUN_TASK_ARGS -N ${#HOSTS[@]} -n $((${#HOSTS[@]} * GPUS_PER_NODE)) --ntasks-per-node=$GPUS_PER_NODE -c $CORES_PER_GPU task.sh"
rm -f "$LOGDIR/loop.rc"
run_tasks loop
if dry; then
    jlog "dry run: nothing started; chain.state, hub.addr and logs would be under $RUN_DIR/slurm"
    exit 0
fi
jlog "step pid $STEP; logs in $LOGDIR"
# Where the GPUs and cores landed, for the first jobs: compare with GPU_NUMA and the hub's
# "joined (... cores ...)" lines.
{ nvidia-smi topo -m; numactl -H; } > "$LOGDIR/topology.txt" 2>&1

SIGNALLED=""
on_signal() {
    [ -n "$SIGNALLED" ] && return
    SIGNALLED="$1"
    chain "got SIG$1: stopping the step (loop and workers)"
    kill -TERM "$STEP" 2> /dev/null   # srun forwards it to every task
}
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
# Until the loop has written its exit code (task 0) or the step is gone.
while kill -0 "$STEP" 2> /dev/null && [ ! -s "$LOGDIR/loop.rc" ]; do
    sleep 10 &
    wait $!
done
# The loop is done (its trap stopped the hub and its own worker): stop the workers.
kill -TERM "$STEP" 2> /dev/null
for _ in $(seq "$GRACE_SECS"); do kill -0 "$STEP" 2> /dev/null || break; sleep 1; done
kill -KILL "$STEP" 2> /dev/null
# The hub is gone with the loop: worker jobs stop dialling it until the next job writes its own.
grep -qs " $JOB\$" "$STATE_DIR/hub.addr" && rm -f "$STATE_DIR/hub.addr"

# ---- what the next job should do ----------------------------------------------------------------
RC=$(cat "$LOGDIR/loop.rc" 2> /dev/null || echo "none")
LAST=$(tail -n 1 "$RUN_DIR/status.md" 2> /dev/null)
case "$RC" in
    75) VERDICT="continue"; WHY="deadline (exited at a phase boundary)" ;;
    76) VERDICT="stop phase longer than a job: $LAST"; WHY="a phase cannot fit in one job" ;;
    0)
        if [ -e "$RUN_DIR/STOP" ]; then VERDICT="stop STOP file"; WHY="STOP file"
        elif grep -q 'loop finished' <<< "$LAST"; then VERDICT="stop finished"; WHY="loop finished"
        else VERDICT="continue"; WHY="exit 0"; fi ;;
    *)
        if [ -n "$SIGNALLED" ]; then VERDICT="continue"; WHY="SIG$SIGNALLED (loop exit $RC)"
        else VERDICT="stop failed: loop exit $RC: $LAST"; WHY="loop died (exit $RC)"; fi ;;
esac
echo "$VERDICT" > "$CHAIN_STATE"
chain "end: $WHY -> $VERDICT"
case "$VERDICT" in
    stop*) [ -n "$SUCC" ] && scancel "$SUCC" && chain "cancelled successor $SUCC" ;;
esac
exit 0
