# shellcheck shell=bash
# Plan 069 item 5: shared by loop.sbatch, workers.sbatch, task.sh and submit.sh. Sourced, never
# run. Everything a job needs comes from the run's env file (LOOP_ENV, see loop.env.example) plus
# the knobs below; DRY_RUN=1 prints every process the job would start — its node, task, GPU,
# pinning and log — and the env file, instead of starting anything, so a plan can be checked here.
#
# The job shape is the centre's GPU example (`-n 8 -c 72 --gpus 8 -p gpu`): one Slurm task per
# GH200 superchip, each with its 72 Grace cores and its H100. Task 0 of a loop job runs the loop
# (hub, trainer, the loop's per-phase worker); every other task runs one long-lived worker.

# ---- knobs and their defaults ------------------------------------------------------------------
slurm_defaults() {
    : "${PROJECT:?PROJECT (the GPU account, -A naiss<N>-gpu) must be set in the env file}"
    P="${P:-/nobackup/proj/disk/$PROJECT/botbowl}"
    # image: binaries, scripts and cfgs of the image's own commit (/opt/botbowl/{bin,src});
    # checkout: a checkout in project storage, built by the job (inside the image) before it starts.
    BIN_MODE="${BIN_MODE:-image}"
    SIF="${SIF:?SIF (the botbowl .sif) must be set in the env file}"
    CHECKOUT="${CHECKOUT:-$P/botbowl_rust}"
    RUN_NAME="${RUN_NAME:-loop20x11}"
    RUN_DIR="${RUN_DIR:-$P/runs/$RUN_NAME}"
    MODEL_DIR="${MODEL_DIR:-$P/models/$RUN_NAME}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
    CORES_PER_GPU="${CORES_PER_GPU:-72}"            # -c: one Grace CPU per task
    WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
    WORKER_STREAMS="${WORKER_STREAMS:-64}"          # --parallel-games of each long-lived worker
    WORKER_MEM_FLOOR_MB="${WORKER_MEM_FLOOR_MB:-4096}"
    WORKER_EXTRA_ARGS="${WORKER_EXTRA_ARGS:-}"
    WORKER_RESTARTS="${WORKER_RESTARTS:-5}"
    # How a task's processes are pinned. slurm: the task's own binding (srun --cpu-bind=cores with
    # -c 72 gives each task one superchip's cores, and the kernel's first-touch keeps its memory
    # local); numactl: also `numactl --cpunodebind=N --preferred=N` (NUMA_MEM) with N the task's
    # superchip from GPU_NUMA; none: nothing. Check RUN_DIR/slurm/logs/<job>/topology.txt and the
    # hub's "joined (... cores ...)" lines (the worker counts the cores it may run on).
    PIN="${PIN:-slurm}"
    GPU_NUMA="${GPU_NUMA:-0 1 2 3}"                 # NUMA node of local task i's GPU
    NUMA_MEM="${NUMA_MEM:-preferred}"               # numactl memory policy: preferred|bind|none
    HUB_PORT="${HUB_PORT:-7777}"
    HUB_TOKEN_FILE="${HUB_TOKEN_FILE:-$HOME/.config/botbowl/hub.token}"
    # Node-local NVMe (1.8 TB per node, usable during a job): worker model caches, and the loop's
    # prepared data with PREPARED_ROOT=local. NAISS systems export SNIC_TMP; else TMPDIR, else /tmp.
    LOCAL_SCRATCH="${LOCAL_SCRATCH:-${SNIC_TMP:-${TMPDIR:-/tmp}}}"
    APPTAINER="${APPTAINER:-apptainer}"
    APPTAINER_BIND="${APPTAINER_BIND:-$P,$LOCAL_SCRATCH}"
    APPTAINER_CLEANENV="${APPTAINER_CLEANENV:-1}"   # 1: no host env (modules, LD_LIBRARY_PATH) leaks in
    # The one srun step of a job: a task per GPU, Slurm's own CPU and GPU binding, and a worker
    # that dies does not take the step down.
    SRUN_TASK_ARGS="${SRUN_TASK_ARGS:---cpu-bind=cores --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=0}"
    # Margin before the walltime under which no new phase starts. With STOP_PHASE_ESTIMATE (the
    # default) a phase also gets 1.25x what it took last time, so this is the floor, not the rule.
    STOP_MARGIN_MIN="${STOP_MARGIN_MIN:-20}"
    GRACE_SECS="${GRACE_SECS:-120}"                 # TERM -> KILL on shutdown
    CHAIN_LEFT="${CHAIN_LEFT:-0}"                   # successors this job may still submit
    case "$BIN_MODE" in
        image) SRC=/opt/botbowl/src; BIN=/opt/botbowl/bin ;;
        checkout)
            SRC="$CHECKOUT"
            CHECKOUT_TARGET="${CHECKOUT_TARGET:-$P/target/20x11}"
            BIN="$CHECKOUT_TARGET/release" ;;
        *) echo "BIN_MODE must be image or checkout, got '$BIN_MODE'" >&2; return 1 ;;
    esac
    # The job's own files: env files, logs, chain state, the hub's address. A dry run writes its
    # env files to a scratch dir instead (the run dir is on the cluster).
    STATE_DIR="$RUN_DIR/slurm"
    dry && STATE_DIR="${DRY_STATE_DIR:-${TMPDIR:-/tmp}/botbowl-slurm-dry}"
    return 0
}

# ---- output --------------------------------------------------------------------------------------
jlog() { echo "[$(date '+%F %T')] [${JOB_TAG:-job}] $*"; }
dry() { [ "${DRY_RUN:-0}" = 1 ]; }

# The job's hostnames, one per line. DRY_RUN: DRY_NODES names, or gpu001.. for NODES of them.
job_hosts() {
    if dry; then
        if [ -n "${DRY_NODES:-}" ]; then tr ' ,' '\n\n' <<< "$DRY_NODES" | grep .
        else for i in $(seq 1 "${NODES:-1}"); do printf 'gpu%03d\n' "$i"; done; fi
    else
        scontrol show hostnames "$SLURM_JOB_NODELIST"
    fi
}
numa_of() { local a; read -ra a <<< "$GPU_NUMA"; echo "${a[$1]:-$1}"; }

# The command prefix for PIN=numactl (local task $1); empty otherwise.
pin_prefix() {
    [ "$PIN" = numactl ] || return 0
    local n mem=""
    n=$(numa_of "$1")
    case "$NUMA_MEM" in
        preferred) mem="--preferred=$n" ;;
        bind) mem="--membind=$n" ;;
    esac
    echo "numactl --cpunodebind=$n $mem"
}

# ---- the container -------------------------------------------------------------------------------
# write_env FILE VAR...: the variables as `export` lines, sourced inside the container (where
# --cleanenv has dropped the host's environment). Unset variables are left out.
write_env() {
    local f="$1" v; shift
    : > "$f"
    for v in "$@"; do
        [ -n "${!v+x}" ] && printf 'export %s=%q\n' "$v" "${!v}" >> "$f"
    done
    return 0
}
# container_cmd ENVFILE CMD...: sets C to CMD inside the image with the GPU driver (--nv), project
# storage and local scratch bound, and the env file sourced first.
container_cmd() {
    local envfile="$1"; shift
    C=("$APPTAINER" exec --nv)
    [ "$APPTAINER_CLEANENV" = 1 ] && C+=(--cleanenv)
    # shellcheck disable=SC2016
    C+=(--bind "$APPTAINER_BIND" "$SIF" /bin/bash -c 'source "$0" && exec "$@"' "$envfile" "$@")
}
# launch NAME LOGFILE CMD...: start CMD in the background in a session of its own (so its whole
# tree can be signalled as one group) and set LAUNCHED to its pid; under DRY_RUN only print it.
launch() {
    local name="$1" logf="$2"; shift 2
    if dry; then
        printf '    %-10s' "$name"; printf ' %q' "$@"; printf '\n    %-10s  >> %s\n' "" "$logf"
        LAUNCHED=0
        return 0
    fi
    setsid "$@" >> "$logf" 2>&1 < /dev/null &
    LAUNCHED=$!
}

# worker_cmd NAME: sets W to a long-lived worker's command line (inside the container; the task's
# CUDA_VISIBLE_DEVICES, which the env file carries, picks its GPU).
worker_cmd() {
    local name="$1"
    # shellcheck disable=SC2206
    W=("$BIN/botbowl-worker" --hub "ws://$HUB_HOST:$HUB_PORT/ws" --token-file "$HUB_TOKEN_FILE"
        --name "$name" --parallel-games "$WORKER_STREAMS" --nn-backend inproc
        --cache-dir "$LOCAL_SCRATCH/botbowl-cache/${SLURM_JOB_ID:-dry}/$name"
        --mem-floor-mb "$WORKER_MEM_FLOOR_MB" $WORKER_EXTRA_ARGS)
}

# The one token per machine (botbowl-hub/CLAUDE.md): made once if it is missing, in the hub's own
# format (32 alphanumerics, mode 600), before any hub or worker reads it, so no job mints its own.
ensure_token() {
    [ -s "$HUB_TOKEN_FILE" ] && return 0
    if dry; then jlog "would create the hub token $HUB_TOKEN_FILE (missing)"; return 0; fi
    mkdir -p "$(dirname "$HUB_TOKEN_FILE")"
    (umask 077; tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 32 > "$HUB_TOKEN_FILE.tmp" \
        && echo >> "$HUB_TOKEN_FILE.tmp" && mv -n "$HUB_TOKEN_FILE.tmp" "$HUB_TOKEN_FILE")
    rm -f "$HUB_TOKEN_FILE.tmp"
    jlog "created the hub token $HUB_TOKEN_FILE"
}

# Seconds since the epoch at which this job ends: Slurm's EndTime, else now + WALLTIME.
job_end_epoch() {
    local e=""
    if ! dry && [ -n "${SLURM_JOB_ID:-}" ]; then
        e=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null)
        case "$e" in
            [0-9][0-9][0-9][0-9]-*) e=$(date -d "$e" +%s 2>/dev/null) || e="" ;;
            *) e="" ;;   # empty, N/A, Unknown
        esac
    fi
    if [ -z "$e" ]; then
        local w="${WALLTIME:-2:00:00}" d=0 h m s
        case "$w" in *-*) d="${w%%-*}"; w="${w#*-}" ;; esac
        IFS=: read -r h m s <<< "$w"
        e=$(($(date +%s) + d * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0}))
    fi
    echo "$e"
}

# write_job_env FILE: the env file every container process of this job sources: a snapshot of the
# run's env file (LOOP_ENV, kept beside it for provenance), then what the job decided — paths, the
# hub's address, the binaries, the deadline. The loop's knobs live in the env file; this only adds.
write_job_env() {
    local f="$1" snap="${1%.sh}.loop.env" tmp="$1.vars"
    cp "$LOOP_ENV" "$snap"
    RUN_DIR_OVERRIDE="$RUN_DIR"; BOTBOWL_BIN_DIR="$BIN"
    TRAIN_LOOP_NOSLEEP=1; EXIT_ON_SIGNAL=1
    [ "$BIN_MODE" = checkout ] && CARGO_TARGET_DIR="$CHECKOUT_TARGET"
    # PREPARED_ROOT=local: the loop's prepared data on node 0's NVMe (rebuilt by the next job if a
    # deadline falls between prepare and train).
    [ "${PREPARED_ROOT:-}" = local ] && PREPARED_ROOT="$LOCAL_SCRATCH/botbowl-prepared/$RUN_NAME"
    # The long-lived workers must play the hub's env board and stage the loop's canvas; the loop
    # exports both for its own processes, the workers see only this file.
    if [ -n "${BUILD_W:-}" ]; then BOARD_SIZE_W="$BUILD_W"; BOARD_SIZE_H="$BUILD_H"; BOARD_PLAYERS="$BUILD_PLAYERS"; fi
    if [ -n "${NN_CANVAS:-}" ] && [ "$NN_CANVAS" != off ]; then BLOOD_NN_CANVAS="$NN_CANVAS"; fi
    {
        echo "# $(basename "$f"): written by ${JOB_TAG:-a job} at $(date '+%F %T'); sourced inside the container"
        printf 'source %q\n' "$snap"
    } > "$f"
    write_env "$tmp" P PROJECT RUN_DIR RUN_DIR_OVERRIDE MODEL_DIR BOTBOWL_BIN_DIR CARGO_TARGET_DIR \
        HUB_HOST HUB_PORT HUB_EXTERNAL HUB_TOKEN_FILE TRAIN_LOOP_NOSLEEP EXIT_ON_SIGNAL \
        LOOP_DEADLINE STOP_MARGIN_SECS PREPARED_ROOT BOARD_SIZE_W BOARD_SIZE_H BOARD_PLAYERS BLOOD_NN_CANVAS
    cat "$tmp" >> "$f"
    rm -f "$tmp"
}

# Print a written env file under DRY_RUN (the computed part; the snapshot is the env file itself).
show_env() {
    dry || return 0
    echo "env file $1 (sourced by every container process, after the run's env file):"
    sed 's/^/    /' "$1"
}

# run_tasks ROLE_OF_TASK0: the job's one srun step, a task per GPU running task.sh; under DRY_RUN,
# task.sh once per planned task. Sets STEP to srun's pid.
run_tasks() {
    local n=$((${#HOSTS[@]} * GPUS_PER_NODE)) i
    export LOOP_ENV BOTBOWL_SLURM_DIR="$SLURM_DIR" JOB_ENV_FILE="$ENVF" LOGDIR HUB_HOST TASK0_ROLE="$1"
    if dry; then
        for i in $(seq 0 $((n - 1))); do
            DRY_HOST="${HOSTS[$((i / GPUS_PER_NODE))]}" SLURM_PROCID="$i" SLURM_LOCALID=$((i % GPUS_PER_NODE)) \
                SLURM_NODEID=$((i / GPUS_PER_NODE)) bash "$SLURM_DIR/task.sh"
        done
        STEP=0
        return 0
    fi
    # shellcheck disable=SC2086
    srun $SRUN_TASK_ARGS -N "${#HOSTS[@]}" -n "$n" --ntasks-per-node="$GPUS_PER_NODE" -c "$CORES_PER_GPU" \
        bash "$SLURM_DIR/task.sh" >> "$LOGDIR/tasks.out" 2>&1 &
    STEP=$!
}
