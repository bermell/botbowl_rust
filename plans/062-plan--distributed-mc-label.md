# Plan 062 — MC value labels as a hub job

**Status:** Built 2026-10-10 on a branch; off by default (`LABEL_VIA_HUB=0` keeps the loop's
local `botbowl-ui mc-label` exactly as it was). Hub protocol v19. Output is byte-identical to the
local tool on the same backend (a test). Overhead on CPU, 2 threads: +3%, within noise (§6). A
top-up from n to m playouts is exact (§5, not built). Pipelining with generation is designed, not
built (§7).

## 1. Why

Plan 056's MC value labels won (value RMS −15%, its search 0.533 head to head against the
control), and the loop has run them every generation since (`MC_LABEL_PLAYOUTS=8`). The phase is `botbowl-ui mc-label` on the
training box: 8 policy-only playouts per sample under the generator net, through the GPU sidecar,
~95 min a generation and GPU-bound (plan 058). It is the one game-playing phase the hub's workers
could not share: a laptop on tract that helps generate sits idle while the box labels. Making the
labelling a hub job lets every worker take a share, and is the first step to overlapping it with
generation (§7).

## 2. Design

**The core moved into `botbowl-play` (`mc_label`, `policy`).** Everything that decides a label or
an output byte is there, and both shells call it:

- `replay_all(traj, first)` rebuilds every recorded state from the seed (a deserialised state has
  lost its path buffer), through the seed's first drive for a `--next-drive` record, and fails at
  the first divergence;
- `first_drives(metas)` is the local rule for finding that first drive in a shard (order-free, the
  later line wins on a duplicate);
- `sample_rng(base, traj_seed, drive, k)` / `rng_for` key each sample's playout dice;
- `sample_label` plays `playouts` playouts, playout `i` on the rng's `i`-th `u64`;
- `label_range(nn, traj, first, range, cfg)` replays the whole trajectory and labels a range of it;
- `apply_labels`, `label_tag`, `summary_line` write the labels, the provenance and the log line.

`PolicyBot` and `play_out` moved from `override_audit.rs` to `botbowl_play::policy`; the ui
re-exports them and keeps the tests that pin them. `McLabelArgs` moved to
`botbowl_play::cli_args` so `job label` flattens the same flags (`--in` is now an alias of
`--corpus`). `botbowl-ui mc-label` is a thin shell: files, threads parallel over samples, progress
lines. The old and the new binary write identical bytes (checked with `cmp` on a corpus with
next-drive pairs, an orphan follow-on and a divergent record).

**The hub job.** `botbowl-hub job label --in <shards> --model X.onnx --playouts N --seed S
--out-dir D [--chunk-samples 32] [--batch 1] [--label ..] [--wait]`:

1. The hub reads every shard *before* taking its lock (`label::load_inputs`, one thread per shard):
   the local line rules (`BufRead::lines`, blank lines skipped), the capacity check, each line kept
   zstd'd (a 180 MB shard is a few MB), each follow-on's first drive located, and every trajectory
   cut into **items** of at most `--chunk-samples` samples (even ranges; a trajectory with no
   samples is still one empty item, because whether it replays decides whether it is stamped). A
   shard whose output exists is skipped, as the local tool skips it.
2. Items are handed out like games: a task is `--batch` items of one shard, dedupe on
   `(job, shard, item)`, a vanished or reaped worker's unfinished items go back to the front of the
   queue, three failures of one item fail the job. Label jobs count as critical-path work for the
   eval share cap and as `label_items` / `label_samples` in the rates.
3. An item ships its trajectory's line, and its first drive's for a follow-on record (`LabelItem`,
   zstd). The worker gets the net from its content-addressed model store (the GPU sidecar when it
   has one, tract otherwise), admits the item through the memory governor (policy playouts hold no
   tree), replays the whole trajectory and labels the range. It answers `LabelDone` with the labels
   only (4 bytes a sample), or `Unlabellable(why)`. Every item of one trajectory replays the whole
   trajectory, so they agree on whether it replays — the local per-trajectory verdict.
4. When a shard's last item is in, a thread writes it (`label::ShardWrite`): every line parsed,
   `apply_labels` (or left alone if any item found it unlabellable), re-serialised, `.partial` then
   renamed. The job is done when every shard is on disk. `--wait` prints the local summary line per
   shard, so the loop's "left unlabelled" count reads the same log line. Two differences in that
   line: its paths are absolute, and its "in N s" runs from the job's start to that shard's write
   (shards share the fleet, so there is no per-shard time).

Why sample ranges rather than whole trajectories: a long drive (100+ samples) at 8 playouts is a
minute on a GPU stream and over ten on a laptop's tract. Whole trajectories would leave the tail of
the job waiting on whichever worker drew the last long one. A 32-sample item is ~15 s on a GPU
stream. Replaying the whole trajectory per item costs milliseconds, and shipping it again per item
~30 KB.

**Protocol v19** (its own commit): `Task::Label { id, shard, items: Vec<LabelItem>, cfg:
LabelConfig, model }`, `LabelItem { item, start, end, zstd_json, first_zstd_json }`,
`ToHub::LabelDone { task, item, labels: LabelResult }`, `LabelResult::{Labels(Vec<f32>),
Unlabellable(String)}`. Both enum variants are appended, so the older frames keep their postcard
tags (a test pins `Heartbeat`'s tag). A v18 worker is refused by the version check.

## 3. Equivalence and what can differ

On one backend the hub's file is the local tool's, byte for byte
(`botbowl-ui/tests/mc_label_hub.rs`, tract and the fixture net): two shards, a next-drive pair
whose follow-on comes before its first drive, items of 3 samples (one trajectory's samples cross
tasks and the two workers), an orphan follow-on (its first drive in the other shard) and a record
whose seed no longer reproduces it. Both leave the same trajectories unlabelled, and the job's
counts equal the local summary line's. A label is a pure function of (net, state, dice), and the
dice are keyed on the sample, not on the thread or the machine.

Across backends, labels are not bitwise reproducible: tract and the sidecar's torch forward differ
in the last bits, so a near-tied argmax can pick another move. This is the same as two local runs,
one on tract and one on the GPU. A label job that a laptop shares mixes the two backends inside one
corpus. The labels are the same quantity (the policy's MC value) with the same dice, and the
argmax flips are rare. The corpus records which worker labelled what only in `hub.log` and the
worker logs, as it does for generation.

## 4. Tests

- `botbowl-ui/tests/mc_label_hub.rs`: hub output == local output byte for byte (above). A rerun
  skips written shards and changes nothing. A worker labels one item of a two-item task correctly,
  then vanishes mid-chunk: its other item and its second task are requeued, the real worker does
  every other item, nothing is counted twice, and the files are still the local tool's.
- `botbowl-hub` `label::tests`: reading (blank and CRLF lines, items cover every sample once and run
  trajectory by trajectory, the follow-on finds its first drive, shipped bytes are the line), the
  write of an unlabelled shard is the local re-serialisation, a written shard is skipped, a corpus
  of another capacity is refused. `main.rs`: the loop's command line parses; `--parallel`,
  `--nn-server` and `--playouts 0` are refused.
- `botbowl-hub-proto`: `label_frames_roundtrip` (labels bit for bit, `-0.0` included; appended tags).
- `botbowl-play` `mc_label::tests`: items, first drives, the tag, and the top-up identity (§5).

## 5. Top-up: exact

The labels already carry what a top-up needs. `meta.extra.value_label` is
`mc_policy(playouts=N,model=..,seed=S)`, and every sample of a trajectory has the same N. The
trainer's `prepare` reads only `outcome_value`, so nothing changed there.

A top-up from n to m playouts is **exact**, by two properties:

- `sample_label` draws one `u64` per playout from the sample's rng, in order. Each playout seeds its
  own dice from that draw and draws nothing else from the rng. So playouts 1..n of an m-playout
  labelling *are* the n-playout labelling's playouts.
- Every playout scores −1, 0 or +1, so `round(n · label_n)` is the integer sum exactly: k/n
  rounded to f32 is off by under 1/(2n) for any n below 2^23. `(sum_n + Σ playouts n+1..m) / m`, computed in
  f64 and rounded to f32 as `sample_label` does, is the m-playout label bit for bit.

`topping_up_n_playouts_to_m_gives_the_m_playout_label` pins it: 3 → 7 playouts on four
positions, compared bitwise. The caveats are the usual ones: the same net, the same seed, the same
`max_steps`, and the same backend for bitwise equality.

`--top-up` is **not built**. It would need the worker to skip the first n draws:
`LabelConfig.skip` (a frame field, so protocol v20) plus the prior labels. Better, the hub keeps
the prior label and ships only `skip`, then combines the sums itself. Worth it only if the loop ever
raises `MC_LABEL_PLAYOUTS` mid-run, or wants to re-label an old window more finely.

## 6. Overhead (measured)

**Setup.** Release build at the branch's head, default 28x17/11 board, the fixture net
(`botbowl-nn/tests/fixtures/tiny.onnx`) on tract. The corpus is 30 trajectories, 2008 samples, in
two shards: the first 30 lines of a heuristic `dataset --mode random-start --next-drive` run, one
next-drive pair included. 8 playouts, seed 7, 2 threads either way:

- `mc-label --parallel 2`;
- a test hub on port 13339, one `botbowl-worker --parallel-games 2`, `job label --wait`, default
  `--chunk-samples 32` (80 items).

The script is `target/p062/bench.sh` in the worktree, not committed. The box was running the live
loop (load average 12–17), so the times carry a few percent of noise.

| run | wall time | output |
|---|---|---|
| local `mc-label` | 218 s | — |
| hub job, 1 worker | 225 s (+3%) | both shards byte-identical to local |

The hub's overhead is within the noise:

- reading and compressing the shards at submit;
- shipping the net once;
- the per-item replay and JSON parse on the worker;
- the final write;
- the tail of the last item on two streams.

**On the wire.** An item carries its trajectory's line zstd'd: 616 KB → 14 KB on this board, about
2%. A follow-on record carries its first drive's line too. The answer is 4 bytes a sample. A loop
generation of ~200k samples is ~6k items, ~100–200 MB downstream in all, and a laptop taking a
tenth of it pulls 10–20 MB. The hub holds each shard as zstd lines: a 180 MB shard is a few MB.

## 7. Next: pipelining with generation (not built)

The hub already holds every trajectory the moment a generate worker sends it (`trajectory_done`
has the JSON in hand). A generate job submitted with a label config (`job generate ...
--mc-label-playouts 8`, off by default) could turn each arriving trajectory into label items for
the same net at once. Labelling would then overlap generation, and the mc-label phase would shrink
to the tail of the last few drives. Points to settle:

- **Output order.** The raw shard's line order is arrival order. The labelled shard must keep it.
  The hub writes `mc/shardK.jsonl` when the generate shard and all its items are done, in
  raw-shard order.
- **Next-drive records** arrive together with their first drive (one `TrajectoryDone`), so a
  follow-on's first drive is always at hand.
- **GPU contention.** Generation is GPU-bound on the sidecars (plan 058). Labels would compete for
  the same batches, so the gain on the training box alone is the GPU idle time between phases, not
  2x. The real gain is remote tract workers, which add capacity to whichever phase is running.
- **Markers.** `.mc_labelled` would be touched by the generate step. The resume logic must accept a
  generation whose labels finished with it.

## 8. Turning it on

1. Merge the branch to master and rebuild the loop's binaries at the merge commit (the launcher
   does it). Protocol v19: every helper worker (the laptop) must pull and rebuild at the same
   commit with the same `BOARD_SIZE_*`, or the hub refuses it (the version check, then the commit
   check).
2. Relaunch the loop with `LABEL_VIA_HUB=1`. Optional: `LABEL_PARALLEL_GAMES` (the local worker's
   streams, default `MC_LABEL_PARALLEL`=16), `LABEL_SIDECARS` (default 1), `LABEL_CHUNK_SAMPLES`
   (default 32).
3. Check the first generation: `mc_label.log` has one summary line per shard with the usual
   unlabelled count. `mc_label.worker.log` shows `NN_SERVER connected … (canary ok)` and no
   `NN_SERVER_FALLBACK`. `hub.log` has `[hub] mc-label: … -> …/mc/shardK.jsonl` per shard and the
   rate lines show `+ label N samples/min` per worker. Compare the phase's minutes with the local
   phase's ~95.
