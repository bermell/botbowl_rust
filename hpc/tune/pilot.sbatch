#!/bin/bash
# The inference tuning suite as one Slurm job on one GH200 node (plan 069 item 4): build once, then one
# task per superchip, each running every 4th row of the spec on its own GPU and 72 cores.
#
#   export P=/nobackup/proj/disk/<project>/botbowl SIF=$P/images/botbowl-<commit>.sif
#   cd $P/botbowl_rust && sbatch -A <project> hpc/tune/pilot.sbatch                 # gh200.toml, 4 tasks
#   SPEC=hpc/tune/gh200.toml TASKS=1 EXTRA="--only micro,queue" sbatch -A <project> -n 1 --gpus 1 hpc/tune/pilot.sbatch
#
# Partition name, walltime limit and whether a job may take one GPU are first-login questions
# (hpc/arrhenius/README.md); adjust the #SBATCH lines to what `sinfo` says. Rows already in a task's
# results.csv are skipped, so resubmitting the same OUT after a walltime kill resumes.
#SBATCH -J bb-tune
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 72
#SBATCH --gpus 4
#SBATCH -t 01:30:00
#SBATCH -o bb-tune-%j.out
set -euo pipefail
: "${P:?export P=/nobackup/proj/disk/<project>/botbowl}" "${SIF:?export SIF=<the image>}"
SPEC="${SPEC:-hpc/tune/gh200.toml}"
TASKS="${TASKS:-${SLURM_NTASKS:-4}}"
OUT="${OUT:-$P/runs/hpc_tune/pilot-${SLURM_JOB_ID:-manual}}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$P/target/tune-20x11}"
mkdir -p "$OUT"
RUN=(apptainer exec --nv --bind "$P" --bind "${TMPDIR:-/tmp}" "$SIF")

# Build once (the image's toolchain, offline crates), and list the rows: a dry run of the whole spec.
"${RUN[@]}" hpc/tune/run_suite.sh "$SPEC" "$OUT/build" --dry-run | tee "$OUT/rows.txt"
export BIN_UI="$CARGO_TARGET_DIR/release/botbowl-ui" BIN_BENCH="$CARGO_TARGET_DIR/release/examples/inproc_bench"

# One task per superchip: Slurm binds each to its GPU (CUDA_VISIBLE_DEVICES) and its 72 cores, and
# the driver pins inside that set (pin = node / split). Each task writes its own OUT/taskK.
export SPEC OUT TASKS EXTRA="${EXTRA:-}"
srun --ntasks="$TASKS" --cpus-per-task=72 --gpus-per-task=1 --cpu-bind=cores bash -c \
    'exec apptainer exec --nv --bind "$P" --bind "${TMPDIR:-/tmp}" "$SIF" \
        hpc/tune/run_suite.sh "$SPEC" "$OUT/task$SLURM_PROCID" --shard "$SLURM_PROCID/$TASKS" $EXTRA' || true
for t in "$OUT"/task*/results.md; do echo "== $t"; cat "$t"; done
