#!/bin/bash
# Plan 069 item 5: submit and steer the loop on Arrhenius. Everything comes from the run's env file
# (copy loop.env.example, fill in PROJECT, SIF and the loop's knobs).
#
#   submit.sh [--dry-run|--test-only] [--env FILE] loop [--chain N] [--upfront] [--nodes N]
#       the loop job (loop.sbatch) on NODES nodes (--nodes overrides the env file's). --chain N: each job queues its successor
#       (afterany) until N more have run; --upfront: queue all N+1 now instead (for a centre that
#       refuses sbatch from compute nodes). Default --chain 0: one job.
#   submit.sh [--dry-run|--test-only] [--env FILE] workers [--nodes N] [--array SPEC]
#       a workers-only job (workers.sbatch): more GPUs for the running loop's hub, or the GPU side
#       of the persistent-hub variant (HUB_EXTERNAL=1). --array 0-7: eight such jobs.
#   submit.sh [--env FILE] stop      touch RUN_DIR/STOP: the loop exits at its next phase boundary
#                                    and no later job of the chain starts
#   submit.sh [--env FILE] resume    remove STOP and chain.state (then submit loop again)
#   submit.sh [--env FILE] status    the chain's log, the loop's last status lines, the queue
#
# --dry-run runs the job script here with DRY_RUN=1 — no Slurm, no apptainer, nothing started —
# and prints the sbatch line, every task per node with its GPU, pinning, command and log, and the
# env file the container processes source. --test-only hands the real sbatch line to
# `sbatch --test-only` (the centre's syntax and limits check; nothing is queued). LOOP_ENV may also
# come from the environment.
set -u
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "$SLURM_DIR/common.sh"
DRY_RUN="${DRY_RUN:-0}"
TEST_ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --test-only) TEST_ONLY=--test-only; shift ;;
        --env) LOOP_ENV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
        *) break ;;
    esac
done
export DRY_RUN
CMD="${1:-}"; shift || true
: "${LOOP_ENV:?--env FILE or LOOP_ENV: the run env file (see loop.env.example)}"
# shellcheck disable=SC1090
source "$LOOP_ENV"
slurm_defaults || exit 2
PARTITION="${PARTITION:-gpu}"
NODES="${NODES:-1}"
# The centre's default walltime is 2 h and its maximum is not known yet. Every phase of a
# generation must fit in one job (the loop stops the chain with exit 76 when one cannot), and a
# job under ~10 min wastes its startup.
WALLTIME="${WALLTIME:-2:00:00}"
SIGNAL_SECS="${SIGNAL_SECS:-600}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

# sbatch_args NAME NODES: what every job of this run gets: a task per GH200 with its 72 cores and
# one GPU (the centre's example is `-n 8 -c 72 --gpus 8 -p gpu`), on whole nodes.
sbatch_args() {
    local gpus="${GPU_ARGS:---gpus=$(($2 * GPUS_PER_NODE))}"
    # shellcheck disable=SC2086
    echo -A "$PROJECT" -p "$PARTITION" -N "$2" --ntasks-per-node="$GPUS_PER_NODE" -c "$CORES_PER_GPU" $gpus \
        -t "$WALLTIME" --signal=B:USR1@"$SIGNAL_SECS" -o "$STATE_DIR/logs/$1-%j.out" $SBATCH_EXTRA
}
submit() {   # submit SCRIPT ARGS_FILE CHAIN_LEFT [extra sbatch args...]
    local script="$1" args="$2" left="$3"; shift 3
    if [ "$DRY_RUN" = 1 ]; then
        echo "sbatch --parsable --export=ALL,LOOP_ENV=$LOOP_ENV,BOTBOWL_SLURM_DIR=$SLURM_DIR,CHAIN_LEFT=$left $* $(cat "$args") $SLURM_DIR/$script" >&2
        echo "DRYJOB"
        return 0
    fi
    # shellcheck disable=SC2046,SC2086
    sbatch --parsable $TEST_ONLY --export=ALL,LOOP_ENV="$LOOP_ENV",BOTBOWL_SLURM_DIR="$SLURM_DIR",CHAIN_LEFT="$left" \
        "$@" $(cat "$args") "$SLURM_DIR/$script"
}
case "$WALLTIME" in
    *-*|[4-9]:*|[1-9][0-9]:*) ;;
    *) echo "note: WALLTIME=$WALLTIME — every phase of a generation must fit in one job; size GAMES_PER_SHARD so 'generate done (N min)' on status.md stays well under it" >&2 ;;
esac

mkdir -p "$STATE_DIR/logs"
case "$CMD" in
    loop)
        CHAIN=0; UPFRONT=0
        while [ $# -gt 0 ]; do
            case "$1" in
                --chain) CHAIN="$2"; shift 2 ;;
                --upfront) UPFRONT=1; shift ;;
                --nodes) NODES="$2"; shift 2 ;;
                *) echo "unknown loop option $1" >&2; exit 2 ;;
            esac
        done
        if [ -e "$RUN_DIR/STOP" ] || grep -qs '^stop' "$STATE_DIR/chain.state"; then
            echo "the chain is stopped ($([ -e "$RUN_DIR/STOP" ] && echo "STOP file; ")$(cat "$STATE_DIR/chain.state" 2> /dev/null)); run '$0 resume' first" >&2
            [ "$DRY_RUN" = 1 ] || exit 1
        fi
        ARGS="$STATE_DIR/sbatch.loop.args"
        sbatch_args loop "$NODES" > "$ARGS"
        if [ "$UPFRONT" = 1 ]; then
            prev=""
            for _ in $(seq 0 "$CHAIN"); do
                prev=$(submit loop.sbatch "$ARGS" 0 ${prev:+--dependency=afterany:$prev}) || exit 1
                echo "submitted $prev"
            done
        else
            id=$(submit loop.sbatch "$ARGS" "$CHAIN") || exit 1
            echo "submitted $id (it queues up to $CHAIN successor(s) itself)"
        fi
        if [ "$DRY_RUN" = 1 ]; then
            echo "---- the plan of one loop job (DRY_RUN) ----"
            DRY_NODES="${DRY_NODES:-$(seq -f "gpu%03g" 1 "$NODES" | paste -sd" ")}" CHAIN_LEFT="$CHAIN" BOTBOWL_SLURM_DIR="$SLURM_DIR" LOOP_ENV="$LOOP_ENV" \
                bash "$SLURM_DIR/loop.sbatch"
        fi
        ;;
    workers)
        WN="$NODES"; ARRAY=""
        while [ $# -gt 0 ]; do
            case "$1" in
                --nodes) WN="$2"; shift 2 ;;
                --array) ARRAY="$2"; shift 2 ;;
                *) echo "unknown workers option $1" >&2; exit 2 ;;
            esac
        done
        ARGS="$STATE_DIR/sbatch.workers.args"
        sbatch_args workers "$WN" > "$ARGS"
        id=$(submit workers.sbatch "$ARGS" 0 ${ARRAY:+--array=$ARRAY}) || exit 1
        echo "submitted $id"
        if [ "$DRY_RUN" = 1 ]; then
            echo "---- the plan of one workers job (DRY_RUN) ----"
            DRY_NODES="${DRY_NODES:-$(seq -f "gpu%03g" 101 $((100 + WN)) | paste -sd" ")}" BOTBOWL_SLURM_DIR="$SLURM_DIR" LOOP_ENV="$LOOP_ENV" bash "$SLURM_DIR/workers.sbatch"
        fi
        ;;
    stop)
        touch "$RUN_DIR/STOP"
        echo "placed $RUN_DIR/STOP: the loop exits at its next phase boundary; no later job starts"
        ;;
    resume)
        rm -f "$RUN_DIR/STOP" "$STATE_DIR/chain.state"
        echo "removed STOP and chain.state; now: $0 --env $LOOP_ENV loop [--chain N]"
        ;;
    status)
        echo "== $STATE_DIR/chain.log"; tail -n 15 "$STATE_DIR/chain.log" 2> /dev/null
        echo "== chain.state: $(cat "$STATE_DIR/chain.state" 2> /dev/null || echo none)$([ -e "$RUN_DIR/STOP" ] && echo ', STOP file present')"
        echo "== $RUN_DIR/status.md"; tail -n 10 "$RUN_DIR/status.md" 2> /dev/null
        echo "== queue"; squeue -u "$USER" -o '%.12i %.20j %.8T %.10M %.10l %.5D %R' 2> /dev/null
        ;;
    *) sed -n '2,22p' "$0"; exit 2 ;;
esac
