#!/bin/bash
#SBATCH --job-name=botbowl-workers
# Plan 069 item 5: a job of workers only — one task per GH200 superchip, each a long-lived
# `botbowl-worker` on its GPU (task.sh), dialling a hub somewhere else. Two uses:
#   - more GPUs for a running loop job without resizing it: the hub address comes from the loop
#     job's RUN_DIR/slurm/hub.addr, re-read every minute; when the loop job ends (it removes the
#     file) the workers stop, and they restart on the next job's address. Submit as many of these
#     (or one array) as the allocation allows; each ends with its walltime;
#   - the persistent-hub variant: HUB_EXTERNAL=1 and HUB_HOST in the env file name a hub on a PCD
#     virtual machine, and the GPU jobs are pure worker arrays (submit.sh workers --array).
# It exits when the run's chain is stopped (STOP file, or chain.state "stop ..."), and after
# NO_HUB_EXIT_MIN minutes without a hub, so a finished or failed loop does not leave worker jobs
# burning hours.
set -u
SLURM_DIR="${BOTBOWL_SLURM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=common.sh
source "$SLURM_DIR/common.sh"
: "${LOOP_ENV:?LOOP_ENV (the run env file, see loop.env.example) must be set}"
# shellcheck disable=SC1090
source "$LOOP_ENV"
slurm_defaults || exit 2
JOB="${SLURM_JOB_ID:-dry}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
JOB_TAG="workers $JOB"
LOGDIR="$STATE_DIR/logs/workers-$JOB"
mkdir -p "$LOGDIR" "$STATE_DIR/env"
mapfile -t HOSTS < <(job_hosts)
ulimit -n "$(ulimit -Hn)" 2> /dev/null
ensure_token
NO_HUB_EXIT_MIN="${NO_HUB_EXIT_MIN:-30}"

stopped() { [ -e "$RUN_DIR/STOP" ] || grep -qs '^stop' "$STATE_DIR/chain.state"; }
# "host port" of the hub to dial: the env file's (HUB_EXTERNAL=1), else the loop job's hub.addr.
hub_addr() {
    if [ "${HUB_EXTERNAL:-0}" = 1 ]; then echo "${HUB_HOST:?HUB_EXTERNAL=1 needs HUB_HOST} $HUB_PORT"
    elif dry; then echo "<loop-node-0> $HUB_PORT"
    else awk 'NR==1{print $1, $2}' "$STATE_DIR/hub.addr" 2> /dev/null; fi
}

STEP=""
start_step() {
    read -r HUB_HOST HUB_PORT <<< "$1"
    ENVF="$STATE_DIR/env/workers-$JOB.sh"
    write_job_env "$ENVF"
    show_env "$ENVF"
    run_tasks worker
    dry || jlog "workers on ${#HOSTS[@]} node(s) dialling $HUB_HOST:$HUB_PORT (step pid $STEP)"
}
stop_step() {
    [ -n "$STEP" ] || return 0
    kill -TERM "$STEP" 2> /dev/null
    for _ in $(seq "$GRACE_SECS"); do kill -0 "$STEP" 2> /dev/null || break; sleep 1; done
    kill -KILL "$STEP" 2> /dev/null
    STEP=""
}
trap 'stop_step; exit 0' TERM INT USR1

if dry; then
    jlog "dry run: ${#HOSTS[@]} node(s) ${HOSTS[*]}"
    start_step "$(hub_addr)"
    exit 0
fi
stopped && { jlog "the run's chain is stopped (STOP or chain.state): nothing to do"; exit 0; }
CUR=""; NOHUB=0
while :; do
    if stopped; then jlog "the run's chain was stopped: ending"; stop_step; exit 0; fi
    ADDR=$(hub_addr)
    if [ -z "$ADDR" ]; then
        # The loop job ended (it removes hub.addr): stop dialling until the next one is up.
        [ -n "$CUR" ] && { jlog "the hub at $CUR is gone; waiting for the next one"; stop_step; CUR=""; }
        NOHUB=$((NOHUB + 1))
        [ "$NOHUB" -ge "$NO_HUB_EXIT_MIN" ] && { jlog "no hub for $NO_HUB_EXIT_MIN min: ending"; exit 0; }
    else
        NOHUB=0
        if [ "$ADDR" != "$CUR" ]; then
            [ -n "$CUR" ] && jlog "the hub moved: $CUR -> $ADDR"
            stop_step
            start_step "$ADDR"
            CUR="$ADDR"
        fi
    fi
    sleep 60 &
    wait $!
done
