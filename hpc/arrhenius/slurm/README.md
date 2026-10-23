# The loop in a Slurm job (Arrhenius)

Plan 069 item 5. Login is password + 2FA, so nothing outside the cluster can drive it: the training
loop runs entirely inside Slurm jobs, chained, and resumes from its own markers in each new job. The
scripts here wrap `scripts/train_loop.sh` unchanged in substance; the knobs they need are in it,
all default-off (the loop on the training box behaves exactly as before).

| file | what |
|---|---|
| `submit.sh` | submit, dry-run, stop, resume, status: the only entry point |
| `loop.sbatch` | the loop job: one srun step, task 0 the loop, every other task a worker; chain logic |
| `workers.sbatch` | a workers-only job: more GPUs for a running loop, or the GPU side of the persistent-hub variant |
| `task.sh` | one task = one GH200 superchip: the loop (task 0) or `WORKERS_PER_GPU` long-lived workers |
| `common.sh` | knobs, the container command, the env file, the dry-run printer |
| `loop.env.example` | one run's settings: the job shape and the loop's knobs (the cluster's `launch_plan065.sh`) |

## Shape of a job

The centre's GPU example is `#SBATCH -n 8 -c 72 --gpus 8 -p gpu`: one task per GH200 superchip with
its 72 Grace cores and its H100. A loop job is exactly that on `NODES` whole nodes:

```
node 0  task 0  train_loop.sh (in the image)       hub on 0.0.0.0:HUB_PORT, trainer, the loop's per-phase
                                                   worker, net check, value bench — all on this one GPU
        task 1-3  botbowl-worker x WORKERS_PER_GPU  ws://<node 0>:HUB_PORT/ws, --parallel-games WORKER_STREAMS
node k  task 4k..4k+3  botbowl-worker              same
```

- **One srun step** (`srun --cpu-bind=cores --gpus-per-task=1 --gpu-bind=closest -c 72
  --ntasks-per-node=4`): Slurm binds each task's cores and GPU. `PIN=numactl` adds
  `numactl --cpunodebind/--preferred` on top (`GPU_NUMA` maps local task to NUMA node).
- **Everything runs in the image** (`apptainer exec --nv --cleanenv --bind $P,$LOCAL_SCRATCH`), with a
  per-job env file (`RUN_DIR/slurm/env/job-<id>.sh`) sourced first: a snapshot of the run's env file
  plus what the job decided (hub host, binaries, deadline, paths). `--cleanenv` keeps the host's
  modules and `LD_LIBRARY_PATH` out.
- **Binaries**: `BIN_MODE=image` runs the image's own commit (`/opt/botbowl/{bin,src}`: the loop's
  scripts and `cfgs/` come from the image too, so the run is one commit end to end);
  `BIN_MODE=checkout` builds a checkout in project storage inside the image first (offline, the
  vendored crates; `hpc/arrhenius/README.md`), for a commit newer than the image whose lockfiles
  have not changed. Either way the loop gets `BOTBOWL_BIN_DIR` and never runs cargo itself.
- **Hub token**: the one per machine, `~/.config/botbowl/hub.token` (home is shared), created once
  in the hub's format if it is missing, before any process reads it. No job mints its own.
- **Allowlist**: hub and workers are the same binaries, so the commit check is an exact match and no
  allowlist is needed. `HUB_ALLOWED_COMMITS` (a train_loop knob) passes one to the hub when mixing.
- **Files**: corpora, nets, run dir on Lustre (`$P`, the disk tier); worker model caches on the
  node-local NVMe (`LOCAL_SCRATCH`: `$SNIC_TMP`, else `$TMPDIR`, else `/tmp`);
  `PREPARED_ROOT=local` puts the trainer's prepared data on node 0's NVMe too (or a Lustre flash-tier
  path, if the project has quota there: 15M IOPS random 4K reads, what the loader's mmapped
  `spatial.npy` does). A generation marked prepared whose dirs are gone is prepared again.

## Walltime, resume, the chain

The centre's default walltime is **2 h**; the maximum is not known yet. The loop never relies on a
job outliving a phase:

- `LOOP_DEADLINE` (the job's end, from `squeue -o %e`) and `STOP_PHASE_ESTIMATE`: at each phase
  boundary the loop exits **75** if less time is left than 1.25x what that phase took last time
  (status.md's `genNN <phase> done (N min)` lines; at least `STOP_MARGIN_MIN`). A phase that took
  longer than a whole job exits **76**, which stops the chain: raise the walltime or shrink the
  phase (fewer `GAMES_PER_SHARD`, more nodes) instead of relaunching into the same wall forever.
- `--signal=B:USR1@SIGNAL_SECS` is the backstop. On USR1, TERM (scancel, the walltime) or INT the
  batch script TERMs the step; task 0 TERMs the loop's process group, whose trap
  (`EXIT_ON_SIGNAL=1`) stops its hub and worker and exits 143; the workers stop; the hub requeues
  nothing because it is gone too. What a cut phase loses: **a generate phase restarts from scratch**
  (`--truncate`, the hub keeps no record of finished games across restarts); label shards already
  written are kept (`job label` skips them); an in-flight benchmark is resubmitted; an interrupted
  train retrains.
- **Chain** (`submit.sh loop --chain N`): each job queues its successor first thing
  (`afterany` on itself), so it ages in the queue meanwhile. `--upfront` queues all N+1 at submit
  instead, for a centre that refuses `sbatch` from compute nodes. Each job writes
  `RUN_DIR/slurm/chain.state`: `continue` (deadline, signal) or `stop <why>` (STOP file, loop
  finished, loop died, phase longer than a job, build failed). A job that finds `stop` or a STOP
  file exits at once and cancels nothing else; one that ends with `stop` cancels its queued
  successor. `chain.log` has a line per job start and end.
- **STOP convention**: `submit.sh stop` touches `RUN_DIR/STOP` — the loop exits at its next phase
  boundary (its own convention, unchanged) and no later job starts. `submit.sh resume` removes STOP
  and chain.state.
- The hub's address is in `RUN_DIR/slurm/hub.addr` while a loop job runs (removed at its end), so
  `workers.sbatch` jobs can join: they re-read it every minute and move with it.

Jobs should run at least ~10 min (the centre: startup overhead); a chain of 2 h jobs spends a few
minutes per job on startup, the image's squashfs mount and the workers' first model fetch.

## Persistent-hub variant

The PCD partition is virtual machines (32 nodes of 2x EPYC 9555, CEPH storage, Ethernet), with
container services planned. A hub there would keep the status page, the registry and the web play
app up between jobs, with GPU jobs as pure worker arrays: `HUB_EXTERNAL=1 HUB_HOST=<vm>` in the env
file, `submit.sh loop` (task 0 runs the loop against that hub, which it never starts) plus
`submit.sh workers --array 0-N`. Three things must hold first, all open: GPU compute nodes can reach
the VM; **the VM mounts the project's Lustre** (the hub writes the corpora to the paths the loop
names; a hub on CEPH alone cannot serve generate or label jobs); and the hub is built for the VM's
architecture (x86_64: the training box's build at the same commit and capacity works, `botbowl-hub`
links no libtorch). Until then the job's own hub is the design, and the PCD VM is at most a
read-only mirror of status and registry.

## Check a plan here

```sh
hpc/arrhenius/slurm/submit.sh --dry-run --env my.env loop --nodes 2 --chain 3
hpc/arrhenius/slurm/submit.sh --dry-run --env my.env workers --nodes 1 --array 0-3
```

prints the sbatch line, each task (node, local id, cores, pinning, GPU, role), the exact container
command of the loop and the first worker, the logs, and the env file. On the cluster,
`submit.sh --test-only ...` hands the real line to `sbatch --test-only`.

## First jobs on the cluster

1. `submit.sh --test-only --env E loop`; then a 1-node, 30-minute job with `MAX_GENS` and
   `GAMES_PER_SHARD` small, to see the whole cycle once.
2. In `RUN_DIR/slurm/logs/<job>/`: `topology.txt` (`nvidia-smi topo -m`, `numactl -H`) against
   the task binding; `hub.log` (in RUN_DIR) `joined (64 streams, 72 cores, ...)` per worker — 72
   means the task sees exactly its superchip; each `worker-*.log` starts with
   `nets run in-process on cuda`.
3. A deadline exit: `WALLTIME=0:30:00` with a generate phase that does not fit, to see exit 75/76
   and the successor resume.

## Open questions for the centre

- Maximum walltime (and whether longer jobs wait longer); whether `sbatch` works from compute nodes
  (else `--upfront`).
- Billing per GPU or per node; a 1-node job with `--exclusive` vs `--gpus 4`.
- Whether compute nodes reach a PCD VM, and whether PCD VMs can mount project Lustre.
- The node-local NVMe's path inside a job (`$SNIC_TMP`? `$TMPDIR`?) and whether it is wiped per job.
- Lustre flash-tier quota for training data.
- Whether Slurm sets `CUDA_VISIBLE_DEVICES` per task with `--gpus-per-task` + `--gpu-bind=closest`
  (task.sh passes it through; it falls back to the local id).
- Slurm version (`--gpu-bind`, `--gpus-per-task` need >= 19.05).
