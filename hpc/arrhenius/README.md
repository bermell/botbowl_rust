# botbowl on Arrhenius (NAISS): image, bundles, first login

Plan 069 item 3. The GPU partition is **GH200: aarch64 (Neoverse V2, 72 cores) + one H100 per
superchip, 4 per node**; the CPU partition is x86_64 (AMD Turin). Everything for the GPU nodes runs in
one Apptainer image, `botbowl.sif`, built on an Arm node from this repo and a bundle made on the
training box, so neither the build nor a job needs the internet or a module.

| file | what |
|---|---|
| `botbowl.def` | the image: arm64 `python:3.12-bookworm` (pinned digest), Rust, the trainer venv, the four binaries, the `inproc_parity` harness and the tuning suite's `inproc_bench` built at capacity 20x11/7 with `-C target-cpu=neoverse-v2` and the in-process libtorch backend |
| `make_bundles.sh` | fetches everything the build needs (vendored crates, aarch64 wheels, CUDA header wheels, Rust installer, base image) into `bundle/` (gitignored) |
| `wheelhouse.py` | the wheel part: exactly `train/uv.lock`'s files for cp312 linux aarch64, sha256-checked, plus `requirements.txt` for a no-resolve, no-index install |
| `build_image.sh` | stages HEAD (a shallow clone) into the bundle dir and runs `apptainer build` there |
| `cross_check.sh`, `cross.Dockerfile` | on an x86 box, `cargo check`/`build` for aarch64 incl. the C/C++/asm parts (Debian bookworm's cross gcc 12, the image's compiler and glibc), no qemu |
| `slurm/` | plan 069 item 5: the loop in chained Slurm jobs (a task per GH200, task 0 the loop, the rest workers), workers-only jobs, dry run; its own README |
| `pull_home.sh` | plan 071 R7: nets and run records home in one rsync (one 2FA login), below |
| `retain_corpora.sh` | plan 071 R7: corpus retention on the cluster, dry run by default, below |

## What the image runs on

- **torch 2.13.0+cu126**, the same build as `train/uv.lock` on the training box: PyTorch publishes
  `torch-2.13.0+cu126-cp312-cp312-manylinux_2_28_aarch64.whl` on its cu126 index, and the lock
  already lists it, so the trainer, the shim's libtorch and every kernel match the box's.
- **CUDA 12.6 user space from the wheels** (`nvidia-cuda-runtime-cu12` 12.6.77, cuBLAS 12.6.4,
  **cuDNN 9.10.2.21**, NCCL 2.29.3, …), exactly as on the box. No CUDA toolkit in the image: the shim
  needs only headers, `crt/` and `nv/target`, which come from the nvcc and cccl wheels
  (`BOTBOWL_CUDA_NVCC_INCLUDE=/opt/botbowl/cuda_nvcc/include`).
- **The driver is the host's** (`apptainer exec --nv` binds `libcuda.so` and `nvidia-smi`). The
  cluster's module is CUDA 13.0, so the driver is ≥ 580: a 12.6 user space on a newer driver is the
  ordinary backward-compatible case. **No forward-compatibility package (`cuda-compat`) is needed**;
  that is only for a user space *newer* than the driver. If the driver turns out older than 560
  (12.6's minimum), use a cu124/cu118 build or the compat libs.
- glibc 2.36, gcc 12, Python 3.12.15, Rust as on the box (1.98.0).

## Make the bundle (on the training box, with internet)

```sh
hpc/arrhenius/make_bundles.sh            # → hpc/arrhenius/bundle/, see bundle/MANIFEST
```

At 024c963: 4.1 GB in all: `vendor.tar.gz` 51 MB (455 MB, 24k files unpacked), `wheels/` 2.9 GB
(38 wheels + 2 header wheels; cuDNN 705 MB, cuBLAS 391 MB, torch 331 MB), `rust/` 175 MB, `base/`
994 MB. `SKIP_BASE=1` leaves the base out if the build node can pull from Docker Hub.

Rerun it whenever `Cargo.lock`, `recon_mcts/Cargo.lock` or `train/uv.lock` change (`build_image.sh`
refuses a stale bundle). Copy it to **project storage**, not home (30 GiB / 1 M files; the vendored
crates are 24k files, so they travel as one tarball):

```sh
rsync -a --info=progress2 --exclude /vendor/ hpc/arrhenius/bundle/ <user>@login.hpc.arrhenius.naiss.se:/nobackup/proj/disk/<project>/botbowl/bundle/
```

## Build the image (on the cluster, on an Arm node)

The image is aarch64, so it builds on a **GPU-partition** node (the CPU partition and the login node
may be x86). The checkout must be clean: the image builds the committed HEAD, never edits.

```sh
P=/nobackup/proj/disk/<project>/botbowl
interactive -A <project> -p gpu --gpus 1 -c 16 -t 2:00:00     # flags per `sinfo` / the centre's docs
export APPTAINER_TMPDIR=$P/apptainer/tmp APPTAINER_CACHEDIR=$P/apptainer/cache   # not /tmp, not home
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
cd $P/botbowl_rust && git status          # clean, at the commit you want
hpc/arrhenius/build_image.sh $P/bundle $P/images/botbowl-$(git rev-parse --short HEAD).sif
```

~30-40 min, mostly the release build (`-j` up to 32). If `apptainer build` asks for root, add
`APPTAINER_BUILD_FLAGS=--fakeroot`. Expect ~7-8 GB unpacked (the venv is 5.2 GB, the CUDA
libraries most of it), less as squashfs, and ~12 GB of `APPTAINER_TMPDIR` while it builds. With
Docker Hub reachable and no `bundle/base/`, it pulls the pinned base instead.

Check it on a GPU node:

```sh
apptainer exec --nv $SIF nvidia-smi
apptainer exec --nv $SIF python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
apptainer exec --nv $SIF cat /opt/botbowl/COMMIT
apptainer exec --nv --bind $P $SIF botbowl-worker --help
```

Parity before anything else (plan 069 item 4): `scripts/parity066.sh` builds an example with cargo,
so run it from the checkout in project storage (next section), with a net, its `.pt`/`.ts` and a
shard copied over:
`apptainer exec --nv --bind $P $SIF bash -c "cd $P/botbowl_rust && CARGO_TARGET_DIR=$P/target/20x11-torch scripts/parity066.sh $P/runs/parity $P/models/NET.onnx '$P/runs/SHARD.jsonl'"`.
Then the inference tuning suite: `hpc/tune/README.md` ("On the first pilot node").

## Where things go

| what | where |
|---|---|
| images (`*.sif`), bundle, checkout, Apptainer tmp/cache | `/nobackup/proj/disk/<project>/botbowl/…` |
| run dirs, corpora, models | project storage (`$P/runs/…`, `$P/models/…`): **not backed up**: nets, registry rows and positions are copied back to the training box |
| home | nothing big: `~/.config/botbowl/hub.token`, ssh config |

Bind project storage into the container (`--bind /nobackup`, or `--bind $P`) — Apptainer binds
`$HOME`, the cwd and `/tmp` by default, not project storage. Workers' model cache
(`~/.cache/botbowl/models`) is in home by default: point it at `$P/cache` (worker `--cache-dir`) so nets
and their `.ts` files do not fill home.

## What goes home (`pull_home.sh`, plan 071 §5.2)

Daily, from the training box, one rsync over one login (password + 2FA once):

```sh
hpc/arrhenius/pull_home.sh <user>@login.hpc.arrhenius.naiss.se /nobackup/proj/disk/<project>/botbowl loop20x11 [more runs]
hpc/arrhenius/pull_home.sh --dry-run ...      # rsync -n -v: what would come, nothing written
hpc/arrhenius/pull_home.sh --print-filter ... # the filter list
```

It mirrors part of `$P` into `runs/hpc/` (`HPC_MIRROR`; additive, never deletes) and hard-links the
nets into `models/hpc_<run>/` (`HPC_MODELS`):

- **nets:** `models/<run>/*.{pt,onnx,ts}` (the `.ts` variants too);
- **run records:** `status.md`, the run's `*.txt`/`*.tsv`/`*.json` (champion, `size_curve.tsv`),
  `slurm/chain.{log,state}`, `positions/`; per generation `train.log`, `absorb.txt` and the other
  `*.txt`/`*.json`/`*.tsv` (`td_rate.json`), `verdict`, and in any subdirectory (drives, a drive
  ladder, `net_check/`, `value_bench*/`) `report.json`, `eval.games.jsonl`, `eval.args`,
  `*summary*.txt`, `net_check.txt`; anything named `*tree_stats*`;
- **positions and benches:** `$P/positions/`, `$P/value_bench/`;
- **pilot outputs:** `runs/hpc_tune/**/{results.csv,results.md,meta.json}`.

Never corpora: `shard*.jsonl[.zst]`, `mc/`, `prepared_*/`, worker caches and per-sample bench
outputs stay. For a second rsync in the same session (or to keep the login), give ssh a master
connection through rsync's own `RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h
-o ControlPersist=1h"`, if the centre allows sharing (first-login checklist). `PULL_EXTRA_FILTER`
names a file of extra rsync filter rules, read first. The registry rows are written at home from
the pulled files.

## Corpus retention (`retain_corpora.sh`, plan 071 §5.3)

On the cluster (login node or any job; needs `zstd`), against a run dir. **Dry run unless
`--apply`**; prints every deletion and compression with its size:

```sh
hpc/arrhenius/retain_corpora.sh $P/runs/loop20x11                 # what it would do
hpc/arrhenius/retain_corpora.sh --apply $P/runs/loop20x11         # do it
hpc/arrhenius/retain_corpora.sh --apply --protect "12 13" --prepared-root /scratch/local/botbowl-prepared/loop20x11 $P/runs/loop20x11
```

- **Untouched:** every generation from `C − WINDOW_GENS − 2 + 1` up (`C` the newest `genNN`, in
  progress or not: the window the next training reads plus two of margin; `--window`, else
  `WINDOW_GENS`, else the newest `slurm/env/job-*.loop.env`, else 3; `--keep-extra` the margin), a
  generation not yet `.generated` and `.trained`, and `--protect`'ed ones (an A/B arm's window).
- **Older generations:** `prepared_*`, `probe_val/` and `shadow_probe_val/` deleted (regenerable;
  `.prepared` goes with them, as in the loop's `prune_prepared`); a raw train shard deleted once
  `.mc_labelled` is set and `mc/shardK.jsonl[.zst]` has the same number of lines (else kept, and
  said so); `mc/` train shards and both val shards (raw and `mc/`) compressed to
  `shardK.jsonl.zst` (`zstd --long=27`, level `ZSTD_LEVEL` 10, verified before the original goes);
  a raw train shard with no `mc/` copy is compressed, not deleted. Eval games, reports, logs and
  nets are never touched.
- Idempotent and safe to interrupt: run it after every generation or daily. Read a compressed shard
  with `zstd -q -dc --long=27 shardK.jsonl.zst`; the loop itself never reads a generation this old.

## No modules

Nothing needs `module load`: the image carries its own Rust, Python, torch and CUDA libraries, and
the host only provides the driver through `--nv`. Load no CUDA module in the job script (a
`LD_LIBRARY_PATH` from `GPU/buildenv-nvhpc` leaking into the container could shadow the wheel's
libraries; `apptainer exec --cleanenv` is the belt-and-braces option). The x86 CPU partition can run
today's x86_64 binaries (tract workers) without the image.

## Rebuilding from a checkout inside the image (no new image per commit)

The image's binaries are one commit. For a newer commit, build the checkout with the image's
toolchain, into project storage; the image's env already sets `BOTBOWL_TORCH_DIR`,
`BOTBOWL_CUDA_NVCC_INCLUDE`, `BOARD_SIZE_*`, `PY` and `RUSTFLAGS`. For offline crates, put the vendor
config **above** the checkout, where cargo finds it for every build under that directory without
touching the repo:

```sh
mkdir -p $P/.cargo && tar -xzf $P/bundle/vendor.tar.gz -C $P/bundle
printf '[source.crates-io]\nreplace-with = "vendored-sources"\n[source.vendored-sources]\ndirectory = "%s"\n' \
    $P/bundle/vendor > $P/.cargo/config.toml
apptainer exec --nv --bind $P $SIF bash -c "cd $P/botbowl_rust && CARGO_TARGET_DIR=$P/target/20x11 \
    cargo build --release --offline -p botbowl-ui --features botbowl-ui/torch"
```

The provenance stamp works there too (git is in the image; it reads the checkout's `.git`). The
image's `/opt/venv` is the same lock, so a checkout whose `train/uv.lock` changed needs a new bundle
and image.

## First-login checklist

```sh
storagequota                                   # home and project quotas (files and bytes)
sinfo -s; sinfo -o "%P %l %D %G %c %m"         # partitions, walltime limit, GPUs/cores/memory per node
scontrol show partition gpu | grep -E "MaxTime|DefaultTime|OverSubscribe|TRES"
srun -A <project> -p gpu --gpus 1 -c 18 -t 5 nvidia-smi     # can a job take ONE GPU (a quarter node)? driver version?
srun -A <project> -p gpu --gpus 1 -t 5 bash -c 'curl -sI https://index.crates.io | head -1; curl -sI https://download.pytorch.org | head -1; curl -sI https://registry-1.docker.io/v2/ | head -1'   # internet from a compute node?
apptainer --version                            # ≥ 1.1 for unprivileged builds
uname -m; lscpu | grep -E "Model name|NUMA"    # aarch64, Neoverse-V2, NUMA layout of the superchips
getconf PAGESIZE                               # Grace kernels are often 64K-page: note it (see risks)
projinfo 2>/dev/null || sshare -U -l           # allocation: hours left, expiry, billing per GPU or per node?
scontrol show config | grep -Ei "MaxArraySize|MaxJobCount|SchedulerType|PriorityType"   # job arrays, backfill
apptainer build --fakeroot /tmp/t.sif docker://busybox 2>&1 | tail -1   # may an unprivileged user build here?
srun -A <project> -p gpu --gpus 1 -t 5 bash -c 'getent hosts <persistent-partition-host>; nc -zv <persistent-partition-host> 7777'   # can a compute node reach a hub there?
srun -A <project> -p cpu -n 1 -t 5 hostname    # does the CPU partition run (and draw on the same hours)?
```

On the login node, also check whether SSH connection sharing (`ControlMaster`/`ControlPersist`) is
allowed: with password + 2FA every rsync otherwise asks again.

Note what each says in plan 069 §0b (walltime, one-GPU jobs and billing, CPU-partition billing,
internet, a route to the persistent partition, job arrays, SSH sharing, quotas, allocation expiry). If a
job can take one GPU, the shakedown and the inference suite need no full node.

## Verified here, and what is left for the cluster

Verified on the training box (x86, 2026-10-22, at 024c963 + this change):

- `cross_check.sh`: `cargo check --workspace --all-targets` for aarch64 at 20x11/7 with
  `-C target-cpu=neoverse-v2`, both workspaces, C/asm included (tract-linalg's arm64 kernels,
  mimalloc, zstd); no x86-only code anywhere (no intrinsics, `asm!` or `target_arch` cfgs; the only
  arch match, the worker's target-triple string, has an aarch64 arm).
- Release cross builds of all four binaries: `botbowl-hub` (default features), `botbowl-worker` +
  `prepare` and `botbowl-ui` with `torch`, against the aarch64 torch wheel installed from the
  wheelhouse (`uv pip install --target … --python-platform aarch64-manylinux_2_28`) and the aarch64
  CUDA headers from `fetch_cuda_headers.sh` (offline mode): the shim compiles with aarch64 g++ 12
  and links; the binaries `NEED` libtorch/libtorch_cuda/libc10(_cuda)/libcudart.so.12 with a
  `RUNPATH` to the torch and cudart dirs, as on x86.
- The aarch64 `libtorch_cuda.so` carries SASS for **sm_80 and sm_90** (the H100 is sm_90: native
  kernels, no PTX JIT); it does not link `libcuda.so` (loads without a driver, so `%test` can run at
  build time).
- `cargo check --offline --locked` of both workspaces from the vendored crates with an empty
  `CARGO_HOME` (nothing fetched).
- The base image: g++ 12, git, curl, xz, make, pkg-config, pip, glibc 2.36, Python 3.12.15.

Not verified (needs an Arm node): `apptainer build` itself and `%test`, torch on the H100
(`torch.cuda.get_arch_list()`), the parity scripts (`parity066.sh`, `gatec066.sh`), performance.

## Risks

- **64K pages.** GH200 systems often run 64K-page kernels. Rust, mimalloc and glibc are fine; check
  that torch and the CUDA graphs behave (the parity scripts) before trusting a run.
- **Not bit-identical to x86.** glibc's `exp`/`log`/`pow` take FMA paths on aarch64, and tract's
  arm64 kernels differ from its x86 ones (well inside the 1e-3 canary). Games and searches on Arm
  can differ from the same seed on this box: compare Arm with Arm (exp079-style determinism
  checks included).
- **Apptainer specifics.** Unknown version and build policy (fakeroot or not); `%files` and
  `docker-archive` are standard since Singularity 3. If builds are not allowed on compute nodes at
  all, build on any arm64 machine with Apptainer (or a Mac's Docker + `apptainer build` from a
  docker-archive) and copy the `.sif`.
- **One image per lockfile.** A crate or wheel added on the box needs a new bundle; the build
  refuses a stale one rather than failing halfway.
- **Driver older than expected.** cu126 needs ≥ 560; the cu13.0 module suggests ≥ 580.
