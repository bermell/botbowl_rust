#!/bin/bash
# Plan 069 item 5: one Slurm task of a loop or workers job = one GH200 superchip (its 72 cores and
# its H100, bound by srun). Started by the job's single `srun` (common.sh run_tasks), never by hand.
#
#   task 0 of a loop job (TASK0_ROLE=loop): scripts/train_loop.sh in the image — the hub, the
#     trainer and the loop's per-phase worker on this GPU, as on the training box. Its exit code
#     goes to LOGDIR/loop.rc, which is how the batch script learns the loop has ended.
#   every other task: WORKERS_PER_GPU long-lived `botbowl-worker`s on this GPU, dialling the hub.
#     One that exits is restarted after 30 s (a worker exits by itself only for a bad token or a
#     crash), at most WORKER_RESTARTS times.
# On TERM (the job ending, or the loop done) every process of the task gets TERM, then KILL after
# GRACE_SECS; the hub requeues whatever games the workers held.
#
# Env from the job: LOOP_ENV, JOB_ENV_FILE, LOGDIR, HUB_HOST, TASK0_ROLE, BOTBOWL_SLURM_DIR, and
# Slurm's SLURM_PROCID / SLURM_LOCALID / CUDA_VISIBLE_DEVICES; under DRY_RUN, DRY_HOST names the node.
set -u
SLURM_DIR="${BOTBOWL_SLURM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
# shellcheck source=common.sh
source "$SLURM_DIR/common.sh"
# shellcheck disable=SC1090
source "$LOOP_ENV"
slurm_defaults || exit 2
: "${JOB_ENV_FILE:?}" "${LOGDIR:?}" "${HUB_HOST:?}"
HOST="${DRY_HOST:-$(hostname -s)}"
TASK="${SLURM_PROCID:-0}"
LOCAL="${SLURM_LOCALID:-0}"
JOB_TAG="task $TASK $HOST"
# The GPU srun bound this task to (--gpus-per-task, --gpu-bind); the local index if Slurm set none.
DEV="${CUDA_VISIBLE_DEVICES:-$LOCAL}"
dry && DEV="\$CUDA_VISIBLE_DEVICES(task $TASK)"
read -ra PINW <<< "$(pin_prefix "$LOCAL")"
ROLE=worker
[ "$TASK" = 0 ] && [ "${TASK0_ROLE:-worker}" = loop ] && ROLE=loop
dry && echo "  task $TASK: node ${SLURM_NODEID:-?} $HOST, local $LOCAL, ${CORES_PER_GPU} cores (pin: $PIN${PINW[*]:+, ${PINW[*]}}), GPU $DEV: $ROLE"

# ---- task 0: the loop ---------------------------------------------------------------------------
if [ "$ROLE" = loop ]; then
    container_cmd "$JOB_ENV_FILE" env CUDA_VISIBLE_DEVICES="$DEV" bash "$SRC/scripts/train_loop.sh"
    launch loop "$LOGDIR/loop.out" "${PINW[@]}" "${C[@]}"
    dry && exit 0
    LPID=$LAUNCHED
    jlog "loop started (pid $LPID, process group), log $LOGDIR/loop.out"
    # The loop's own trap (EXIT_ON_SIGNAL=1) stops its hub and worker and exits 143.
    trap 'kill -TERM -- "-$LPID" 2> /dev/null; (sleep "$GRACE_SECS"; kill -KILL -- "-$LPID" 2> /dev/null) &' TERM INT USR1
    RC=0
    while kill -0 "$LPID" 2> /dev/null; do
        wait "$LPID"
        RC=$?
    done
    echo "$RC" > "$LOGDIR/loop.rc"
    jlog "loop exited with $RC"
    exit 0
fi

# ---- every other task: long-lived workers -------------------------------------------------------
declare -A PID=() RESTARTS=() CMDLINE=()
names=()
for k in $(seq 1 "$WORKERS_PER_GPU"); do
    name="$HOST-gpu$LOCAL"; [ "$WORKERS_PER_GPU" -gt 1 ] && name="$name.$k"
    worker_cmd "$name"
    container_cmd "$JOB_ENV_FILE" env CUDA_VISIBLE_DEVICES="$DEV" "${W[@]}"
    names+=("$name")
    CMDLINE[$name]="$(printf '%q ' "${PINW[@]}" "${C[@]}")"
    RESTARTS[$name]=0
done
start_worker() {
    local log="$LOGDIR/worker-$1.log"
    eval "launch $(printf '%q' "$1") $(printf '%q' "$log") ${CMDLINE[$1]}"
    PID[$1]=$LAUNCHED
    dry || jlog "started $1 (pid $LAUNCHED), log $log"
}
if dry; then
    # The first worker in full; the rest differ only in name, cache and log (DRY_VERBOSE=1: all).
    if [ "$TASK" -le 1 ] || [ "${DRY_VERBOSE:-0}" = 1 ]; then
        for n in "${names[@]}"; do start_worker "$n"; done
    else
        for n in "${names[@]}"; do printf '    %-10s (as task 1, its own --name/--cache-dir) >> %s\n' "$n" "$LOGDIR/worker-$n.log"; done
    fi
    exit 0
fi

# Workers read the token at startup; the job wrote it before srun, but a shared home can lag.
for _ in $(seq 60); do [ -s "$HUB_TOKEN_FILE" ] && break; sleep 5; done
[ -s "$HUB_TOKEN_FILE" ] || { jlog "FATAL: no token at $HUB_TOKEN_FILE"; exit 1; }

STOPPING=""
stop_all() {
    STOPPING=1
    local n alive
    for n in "${!PID[@]}"; do kill -TERM -- "-${PID[$n]}" 2> /dev/null; done
    for _ in $(seq "$GRACE_SECS"); do
        alive=""
        for n in "${!PID[@]}"; do kill -0 "${PID[$n]}" 2> /dev/null && alive=1; done
        [ -z "$alive" ] && break
        sleep 1
    done
    for n in "${!PID[@]}"; do kill -KILL -- "-${PID[$n]}" 2> /dev/null; done
    jlog "workers stopped"
    exit 0
}
trap stop_all TERM INT USR1

for n in "${names[@]}"; do start_worker "$n"; done
while [ -z "$STOPPING" ]; do
    sleep 30 &
    wait $!
    for n in "${names[@]}"; do
        kill -0 "${PID[$n]}" 2> /dev/null && continue
        wait "${PID[$n]}" 2> /dev/null
        rc=$?
        if [ "${RESTARTS[$n]}" -ge "$WORKER_RESTARTS" ]; then
            [ "${RESTARTS[$n]}" -eq "$WORKER_RESTARTS" ] && jlog "WARN: $n exited (rc $rc) $WORKER_RESTARTS times; leaving it down"
            RESTARTS[$n]=$((WORKER_RESTARTS + 1))
            continue
        fi
        RESTARTS[$n]=$((RESTARTS[$n] + 1))
        jlog "WARN: $n exited (rc $rc); restart ${RESTARTS[$n]} of $WORKER_RESTARTS"
        start_worker "$n"
    done
done
