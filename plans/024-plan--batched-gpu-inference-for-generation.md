# Batched GPU inference for self-play generation

**Status: DONE — stages 0–5 built and measured** (0–2 on 2026-08-30, 3–5 on 2026-09-01). Data generation is the
dominant cost of the training loop — `gen01` spent **863 min** generating and **10 min** training — and it was
CPU-bound on single-sample `tract` inference. NN leaf evaluation is now batched and GPU-resident, and
`scripts/train_loop.sh` uses it by default (`NN_SERVER=off` reverts).

**Headline: 3.79× end-to-end**, on *fewer* CPU cores than the baseline. Back-to-back, 8 shards × 4 games:

| arm | wall | cores | fw/s | s/game | µs CPU/forward |
|---|---|---|---|---|---|
| tract (the old loop) | 317.8 s | 5.3 | 1164 | 9.93 | 4543 |
| `--nn-server` | 123.9 s | 2.4 | 2364 | 3.87 | 1017 |
| `--nn-server --parallel-games 4` | **83.8 s** | 4.3 | **4051** | **2.62** | **1065** |

The plan's own thesis held at every stage: **`F` and `N` mattered, the GPU never did.** Stage 2 (`F` ≈ 1.5 ms
in-server) bought only 1.33×; Stage 3 cut `F` to 314 µs and took it to 3.3×; Stage 4 raised the offered
concurrency and took it to 3.8×. `g`, the marginal GPU cost, came in 3× better than assumed and changed
nothing at any point. See §Measured results (stages 0–2) and §Measured results (stages 3–5).

## Why: generation is 12.5× more expensive than it needs to be

From the live run's own status trail (`runs/loop14x7/status.md`), same host, same 8-shard layout, same 4800 games:

| generation | evaluator mix | wall | s/game |
|---|---|---|---|
| `gen00` | 8 × heuristic | **69 min** | 6.9 |
| `gen01` | 5 × nn-value + 3 × heuristic | **863 min** | 86.3 (nn shards) |

The NN corpus costs **12.5×** the heuristic corpus. Meanwhile the GTX 1060 sits completely idle for those 863
minutes — it is only touched by the 10-minute training phase.

### The Amdahl ceiling, and the exact measurement that confirms it

Both arms run 1000 MCTS iters/decision, so per-decision *search* cost should be comparable; the difference is
inference. Drives differ in length (nn 37.2 mean samples vs heuristic 19.6), so normalise per decision:

```
heuristic:  6.9 s/game ÷ 19.6 decisions = 352 ms/decision   (search only)
nn-value : 86.3 s/game ÷ 37.2 decisions = 2320 ms/decision  (search + inference)
inference share = (2320 − 352) / 2320 = 84.8%
Amdahl ceiling  = 1 / 0.152 = 6.6×
forwards/decision = 1968 ms / 2.611 ms = ~754   (⇒ ~28k forwards/game)
```

754 forwards per 1000 iterations is *exactly* the shape the code predicts: under `nn-value` the priors are
scripted, so the only NN call is `score_leaf` → `value_home_i64`, once per newly **materialised** node
(`recon_mcts/src/tree.rs:2135` `materialize_placeholder`, one per descent), minus registry hits and minus chance
nodes (which `score_leaf` returns `None` for, `botbowl-mcts/src/dynamics.rs:698`). The arithmetic closing to
within a few percent is the strongest evidence we have that the ceiling is real — **but it rests on the
assumption that nn and heuristic search cost the same per decision, which is not measured.**

**Stage 0 is the measurement that confirms it** — a forward counter, not a profiler. Run when the loop is idle
(`touch runs/loop14x7/STOP`, wait for the phase boundary):

```sh
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export CARGO_TARGET_DIR="$PWD/target/14x7"
cargo build --release -p botbowl-ui
for EV in "nn-value --model models/bbnet_14x7_gen01.onnx" "heuristic"; do
  BLOOD_NN_PROFILE=1 target/14x7/release/botbowl-ui dataset --mode random-start \
    --games 20 --seed 999000 --mcts-iters 1000 --evaluator $EV \
    --truncate --out /tmp/probe.jsonl 2>&1 | tail -3
done
```

Expected output line (new, from Stage 0): `NN_PROFILE forwards=27931 total_ms=72918 mean_us=2611 share=0.85`.

**Gate: proceed only if `share ≥ 0.70`.** Below that, the rest of this plan buys at most 1.4× and should be
dropped in favour of cheaper generation (fewer iters, more heuristic hedge shards).

> **Measured 2026-08-30 — gate passed at 0.853.** The counter is now in `NnEvaluator` (`FORWARDS` /
> `FORWARD_NANOS`, always on; `BLOOD_NN_PROFILE` only controls printing) and `dataset::run` prints the line per
> game and at exit. Numbers and the cross-check are in §Measured results.

## Two things must change, and they are independent

1. **The runtime must become GPU-capable.** `botbowl-nn/src/eval.rs` is `tract`, pure-Rust CPU, and hardcoded to
   batch 1: `runnable_for` (`eval.rs:64`) builds `f32::fact([1, SPATIAL_CHANNELS, h, w])`, and `forward_raw`
   (`eval.rs:84`) builds a `[1, C, H, W]` tensor per call. tract has no CUDA backend, so this is a replacement,
   not a flag.
2. **Something must supply batches.** MCTS leaf evaluation is inherently sequential: the next leaf is only known
   after the current one is backed up. A GPU runtime with batch 1 is *slower* than tract (4209 µs vs 2611 µs on
   this box) — the whole win is in the batch.

Point 2 is the hard part and the reason this plan is longer than "swap tract for `ort`".

## Where batch parallelism can come from

| source | batch reached | code cost | risk | verdict |
|---|---|---|---|---|
| **A. Cross-process server** (batch across the 8 shard processes) | 8 | new server + client + protocol | server is a new SPOF | **yes — the foundation** |
| **B. More independent streams** (raise shard count, or `--parallel-games` in one process) | 16–64 | ~0 (shell) / ~60 lines | RAM; crash blast radius | **yes — the multiplier** |
| **C. Multi-worker MCTS + virtual loss** (`--mcts-workers > 1`) | 8 × W | none (exists) | changes the search; deadlock class | **no — last resort** |
| **D. Speculative / tree-parallel leaf collection inside one search** | large | deep `recon_mcts` surgery | very high | **no** |

### A — cross-process batching is the right foundation

The loop already runs 8 independent shard processes (`scripts/train_loop.sh:238-249`), each issuing roughly one
forward at a time. A single server process batching across them yields batch ≈ 8 **with no change whatsoever to
MCTS's sequential nature** — the call stays synchronous and blocking from the caller's point of view; the
batching happens across *processes*, below the Rust API.

It also solves a VRAM problem for free: eight in-process CUDA contexts would cost ~300 MB each on a 6 GB card
(2.4 GB of pure overhead, before any activations), plus eight cuDNN workspaces and eight autotune passes. One
server holds one context. The model itself is trivial — ~0.5 M params, ~2 MB — so a single server needs well
under 500 MB even at batch 64.

### B — more streams is where the speedup actually lives

Batch 8 alone is **not enough** (see the throughput table below): with only 8 requests ever in flight, the batch
can never exceed 8 and per-batch latency stays the bottleneck. We need ~16–32 concurrent streams. Two ways:

- **B1 — more shard processes.** `NN_SHARDS`/`HEUR_SHARDS` in `scripts/train_loop.sh:66-67` with
  `GAMES_PER_SHARD` scaled down to keep 4800 games/gen. **Zero Rust code.** Better crash isolation than today
  (a panic loses 1/32 of a generation instead of 1/8 — and plan 021's OOB panic proves panics happen). Costs
  RAM: 32 processes × per-shard RSS, on a 15 GB box that already has an OOM-kill history (plans 020, 021).
  **Needs measurement** (Stage 0b).
- **B2 — `--parallel-games N` inside one process.** A thread pool over `dataset::run`'s `for g in 0..args.games`
  (`botbowl-ui/src/dataset.rs:68`) and `eval::run_ladder_rung`'s `for g in 0..args.games`
  (`botbowl-ui/src/eval.rs:181`). Each thread gets its own `GameState`, its own `MctsBot` pair, and its own seed
  (already derived from `g`), so **the search itself is untouched** — games are embarrassingly parallel. RAM is
  the same as B1 (one tree per in-flight game either way); the win is that it also covers the **eval phase**,
  which is a *single* process running games sequentially and would otherwise get *slower* under a server (N=1
  ⇒ batch 1 ⇒ 4209 µs > 2611 µs).

**Recommendation: B1 for generation (free), B2 for eval (necessary).**

Note a latent bug B1 trips: the seed scheme `SEED_BASE + G*1e6 + K*1e5` (`scripts/train_loop.sh:60,239`)
**collides across generations as soon as K ≥ 10** (shard 10 of gen G = shard 0 of gen G+1). Raising the shard
count requires changing the stride to e.g. `G*1e7 + K*1e5`, and the change must be recorded so future corpora
stay provably disjoint.

### C — multi-worker MCTS: available, but the wrong tool here

`MctsBot::with_workers` and virtual loss already exist (plan 015 Step 5), and W workers would give W concurrent
in-flight leaf evaluations per process at *no RAM cost* (shared tree) — genuinely attractive if RAM binds. But:

- Plan 015 measured **1.24× at 10k iters / 10 workers** and 2.05× at 20k. At our 1000-iter budget the tree is
  small and workers thrash the same PUCT path (plan 015 §"where the contention is", finding 3).
- With a server backend, the blocking NN call moves *inside* held locks: `enumerate_placeholders` holds
  `parent_node.children.write()` across `GD::available_actions` (`recon_mcts/src/tree.rs:2097-2101`), and
  `materialize_placeholder` holds `node.score.write()` across `GD::score_leaf` (`tree.rs:2146`). Lock hold times
  go from microseconds to milliseconds. That is precisely the regime that produced plan 013's fairness deadlock
  on `Node.score`. It is *probably* still safe (both are per-node locks, no lock chaining across the call), but
  "probably" is not what you want at 3 a.m. on hour 14 of a generation.
- Virtual loss changes which leaves get expanded, which changes the corpus distribution, which invalidates
  cross-generation report-card comparisons that plan 022's promotion gate depends on.

So: keep `--mcts-workers 1`. Revisit only if Stage 0b shows RAM cannot support 24+ streams, and if so treat it
as an experiment with its own report card, not a perf change.

## Runtime: Python sidecar, not a Rust GPU crate

| option | sm_61 kernels | build cost | Mac (no GPU) | solves batching? | numerics source |
|---|---|---|---|---|---|
| **Python sidecar (torch, `train/.venv`)** | **proven** — cu126 pin already in `train/pyproject.toml:20-26` | none (venv exists) | server simply not started; tract stays default | no — but it's where the batcher lives | the trainer's own `.pt` |
| `ort` (onnxruntime + CUDA EP) | likely, unverified | ~1–2 GB of CUDA/cuDNN libs, `download-binaries`, LD path | falls back to ORT CPU EP → **a third numerics implementation** | **no** | new |
| `tch` (libtorch) | yes (pick cu126 libtorch) | 2+ GB download, version-matched CUDA, `LD_LIBRARY_PATH` | CPU libtorch, still ~200 MB | **no** | new |
| `candle` | via `cudarc`/nvcc; conv perf on Pascal unproven | moderate | ok | **no** | **BBNet re-implemented in Rust** — breaks `botbowl-nn`'s "encoding/model lives in one place" rule |

**Recommendation: a Python sidecar server speaking a small binary protocol over a Unix domain socket, with a
thin Rust client in `botbowl-nn`.**

The reasoning is that the *batching architecture is the hard part and it is runtime-agnostic*. Once the wire
protocol exists, the server's backend can be swapped (eager torch → TorchScript+CUDA graphs → `ort` → TensorRT)
with **zero Rust changes and zero re-validation of the client**. Meanwhile:

- It reuses the exact torch build already proven to run on sm_61 on this host — the cu126 trap
  (plan 022 §GPU note) is already paid for.
- It consumes the trainer's own `.pt` (`models/bbnet_14x7_genNN.pt`, exported alongside the `.onnx` by
  `train_loop.sh`), so there is **no third implementation of BBNet** and no new numerics surface. `ort`,
  `tch` and `candle` all add one.
- The Mac path is untouched: no server ⇒ `NnEvaluator` stays on tract, no new Cargo dependency, `cargo build`
  and `cargo test --workspace` behave identically. Cross-process IPC is the *only* mechanism on this list where
  the GPU dependency does not enter the Rust build graph at all.

The cost is IPC. Budget: the request payload is `spatial` 37×9×16 f32 = 21,312 B + `global` 15 f32 = 60 B; the
response is 4 B (value only) or 17,280 B (with policy). Over a `SOCK_STREAM` UDS that is ~10 µs of copy plus
~10–40 µs of syscall/scheduling — call it **30–60 µs round-trip overhead**, against the 2611 µs it replaces.
Shared-memory rings would get this to ~5 µs and are not worth the complexity until measurement says otherwise.

## The design

### Two nets at once — where multi-model is, and is not, needed

**Not needed for generation.** The 8 generate shards each use exactly one net: 5 × `nn-value` with the current
champion, 3 × heuristic with no net at all (`scripts/train_loop.sh:238-249`). One net per process, one net for
the whole phase — Stages 1–4a are untouched by anything in this section.

**Needed for eval.** The promotion gate runs two *different* nets inside a *single* process:

```sh
botbowl-ui eval --evaluator nn-value --model <candidate>.onnx \
                --vs-evaluator nn-value --vs-model <champion>.onnx   # train_loop.sh:309-312
```

`eval::run` loads both up front — the candidate's at `botbowl-ui/src/eval.rs:220` and the opponent's `vs_nn` at
`eval.rs:224` — and the `vs:` rung alternates the two bots inside every game (`eval.rs:302-306`, via
`run_ladder_rung` at `eval.rs:169`). That is exactly the phase Stage 4b wants to move onto the server, so the
server must serve two nets at once before Stage 4b can land.

Two ways to do that:

**(A) One server, many models** — a `model_id` in the protocol, a model registry in the server, batch key
`(model_id, h, w)`, per-model canary.
**(B) Two server processes**, one per model, with a second socket path and a second CLI flag
(`--nn-server` / `--vs-nn-server`).

Two things do *not* separate them, and should not be used to argue either way:

- **Batch sizes come out the same.** Samples cannot be batched across different weights, so a single
  multi-model server still forms one batch *per model*. The two bots alternate turns inside a game, so each net
  sees roughly half the in-flight streams under either design: `--parallel-games 32` means ~16 streams per net
  either way. Nobody should expect (A) to produce bigger batches — it does not.
- **VRAM.** The tower is ~0.5 M params (single-digit MB) at ~0.13 GFLOP/forward; two CUDA contexts at ~300 MB
  each is entirely affordable on a 6 GB card. The VRAM argument in §A is about *eight* in-process contexts and
  does not carry over to two server processes.

What genuinely separates them:

| | (A) one server, many models | (B) one server per model |
|---|---|---|
| batch size per net | ~half the streams | ~half the streams (**same**) |
| GPU scheduling | two graphs, one context, one stream — no context switching | the driver **time-slices** between two processes' contexts |
| Stage 5 supervision | one lifecycle to start / canary / health-check / restart / reap | two of everything, plus a new partial-failure state (one up, one down) |
| client config | one `--nn-server`; each `NnEvaluator` names its own model at handshake | a second flag, and a rule for which bot uses which socket |
| protocol / server cost | `model_id` + a registry (no eviction, as built) | none |
| future | serves a ladder against several past champions unchanged | one process per champion |

**Recommendation: (A), one server with a model registry.** Two reasons, in order of weight.

First, **GPU context-switching lands precisely on the term this plan exists to minimise.** The throughput model
below shows the entire outcome hinges on `F`, the fixed per-batch cost — at `F = 4.2 ms` the whole exercise is a
regression at every affordable stream count. Without MPS (not set up here, and its Pascal support would be one
more pinned dependency), two processes submitting work to one GPU are time-sliced by the driver, and every slice
boundary is charged to `F` on a workload whose batches are only ~2 ms long. The magnitude is unmeasured — and
that is the point: (B) puts an unmeasured cost into the one variable we cannot afford to guess at, while (A)
provably has none.

Second, **Stage 5's supervision is where the operational risk lives.** The loop treats a dead shard as a warning
but a dead server as a corpus-wide failure; (B) doubles that surface and adds a partial-failure state, for no
compensating benefit.

(A)'s cost is real but bounded: one protocol field and a registry. (It turned out **not** to need an eviction
policy — see §Failure modes; capacity 4 with a loud refusal beats LRU once graphs are attached to a
`model_id`.) And it is the design that
generalises for free to plan 022's "After the weekend" strength ladder, where one candidate is evaluated against
several past generations in a single run.

### Wire protocol (`botbowl-nn/src/remote.rs`, new; `scripts/nn_server.py`, new)

Length-prefixed little-endian frames on a `SOCK_STREAM` UDS. One request, one response, per connection — each
client thread owns its own connection, so no request IDs and no multiplexing are needed. A connection is
**bound to one model at handshake**, which is what makes the per-model canary below meaningful.

```
handshake (client → server):  magic "BBNN" | u16 version | u16 C | u16 F
                              | u16 path_len | utf8 model_path
handshake (server → client):  u16 status | u16 model_id | u16 h | u16 w
                              | f32 canary_value | f32[A*h*w] canary_policy
request:   u32 len | u16 model_id | u16 flags(want_policy) | u16 h | u16 w
                  | f32[C*h*w] spatial | f32[F] global
response:  u32 len | f32 value | [f32[A*h*w] policy if requested]
```

`model_path` is the same `--model` / `--vs-model` string the Rust side was handed; the server resolves it to its
`.pt` sibling (both are exported side by side by the loop's train phase), loads it on first use, and returns a
stable `model_id`. Two clients naming the same path share one loaded module and one batch queue. `model_id` is
echoed on every request only so the server can key the batch without a per-connection lookup — the client never
chooses it, and the server drops any connection whose request `model_id` disagrees with its handshake.

**The canary handshake is the safety interlock, and it is per-model.** For each model it loads, the server runs
a fixed, deterministic input (the committed `tests/fixtures/parity_9x16_*.npy` tensors) and returns *that
model's* result on *that connection's* handshake. The client runs the same input through tract using the ONNX
path it was given and compares to `< 1e-3`. Because the connection is bound to one model, there is exactly one
right answer to compare against. A mismatch means the server resolved the path to different weights, a different
tier, or broken kernels — and the client **aborts loudly** rather than producing a corpus, or a promotion-gate
verdict, labelled with the wrong network.

This is why model identity belongs in the **handshake** and not only in the request. A single global canary over
a per-request model id would validate one net and then silently vouch for the other — and the promotion gate is
exactly where that failure is undetectable: serve the champion's weights to the candidate's bot and you get a
plausible ~0.50 win rate, the gate says REJECTED, the champion stays, and nothing in the report card looks
wrong. This check is still the most valuable thing in the protocol; it just has to be per-connection to mean
anything.

### Server loop: opportunistic batching, no timeout

```python
while True:
    reqs = [q.get()]              # block for the first
    while len(reqs) < MAX_BATCH:  # drain whatever arrived during the last forward
        try: reqs.append(q.get_nowait())
        except Empty: break
    run_batch(reqs)
```

**No max-wait timer.** A timeout is the classic answer and the wrong one here: with only 2 active shards (end of
a generation, or the eval phase) a 1 ms timer adds 1 ms to every request and makes the server *slower than
tract*. Greedy draining self-regulates instead — batch size grows exactly as fast as offered load, because
requests accumulate during the previous forward. `MAX_BATCH` (default 64) bounds tail latency. A `--max-wait-us`
knob exists but defaults to **0**.

Requests are grouped by **`(model_id, h, w)`** and one queue is drained per iteration. During generation there
is exactly one key; during eval there are two (candidate and champion) and the loop alternates between them —
which is also why each net sees only ~half the streams (§Two nets at once). One tier is active at a time, so
`(h, w)` is effectively constant and the key is really just the model.

### Client-side changes are contained to one file

`NnEvaluator` gains a backend enum; **no call-site signatures change and nothing becomes async**:

```rust
enum Backend {
    Tract { proto: InferenceModel, cache: Mutex<HashMap<(usize, usize), Arc<Runnable>>> },
    Remote { client: RemoteClient, fallback: Box<Backend> },   // fallback is always Tract
}
```

- `forward_raw` (`botbowl-nn/src/eval.rs:84`) dispatches on the backend. **Everything above it is unchanged**:
  `priors` (`eval.rs:108`) still gathers per-action logits and softmaxes in Rust; `value_home_i64`
  (`eval.rs:141`) still clamps, sign-flips via `mover_for`, and rescales ×1000. So the two hot call sites in
  `botbowl-mcts/src/dynamics.rs:327` (`nn.priors(...)`) and `dynamics.rs:731` (`nn.value_home_i64(state)`) do
  not change at all, and neither does the gather logic that must stay in lockstep with
  `train/src/bbnn/model.py:masked_policy_logits`.
- `value_home_i64` sets `want_policy = false` — under `nn-value` (the generator) that cuts the response from
  17 KB to 4 B.
- Construction: `NnEvaluator::from_path_with_server(path, Option<&Path>)`, plus a `--nn-server PATH` flag on
  `dataset` and `eval` (`botbowl-ui/src/cli.rs:125,172`) and a `BLOOD_NN_SERVER` env fallback matching the
  repo's `BLOOD_MCTS_*` convention. **Default is `None` ⇒ tract.** `path` doubles as the tract fallback model
  *and* as the identity sent at handshake, so `eval::run`'s two evaluators (`eval.rs:220`, `eval.rs:224`) each
  open their own connection, name their own model, and get their own canary — over a **single** `--nn-server`
  socket, with no second CLI flag. That is design (A)'s payoff at the call site: `load_nn` (`eval.rs:118`) gains
  one argument and nothing else in `eval.rs` changes.
- Each thread gets its own connection via a `thread_local!` socket, so the client is `Sync` without a lock and
  B2's parallel games each hold an independent stream.

That the API shape survives untouched is the main argument for cross-process batching over every alternative in
the table above.

### Throughput model, and what it predicts

Two-station closed queueing model. Per forward: `D_cpu` = aggregate CPU cost across the whole box, `F` = fixed
per-batch server overhead, `g` = marginal GPU cost per sample. From the measured numbers:

```
D_cpu = 352 ms/decision ÷ 754 forwards ÷ 8 concurrent shards = 58.4 µs
g     = 68 µs                (torch CUDA, batch 32: 2174 µs / 32)
R     = min( N / (D_cpu + F + N·g),  1/D_cpu )      N = concurrent streams ≈ batch
baseline (8 nn shards on tract) = 8 / 3.078 ms = 2600 forwards/s
```

Two hard ceilings drop out, and they agree with the Amdahl number, which is a good sign:

- **CPU ceiling** `1/D_cpu` = 17,100 forwards/s = **6.6×** — identical to the Amdahl ceiling above, as it must be.
- **GPU ceiling** `1/g` = 14,700 forwards/s = **5.7×** (asymptotic; ~13k at realistic batch sizes).

| streams N (≈ batch) | F = 0.3 ms | F = 1.0 ms | F = 2.0 ms | F = 4.2 ms |
|---|---|---|---|---|
| **8** (today's shard count) | 8.9k (**3.4×**) | 5.0k (1.9×) | 3.1k (1.2×) | 1.7k (**0.7× — a regression**) |
| **16** | 11.1k (4.3×) | 7.5k (2.9×) | 5.1k (2.0×) | 3.0k (1.2×) |
| **32** | 12.6k (**4.9×**) | 9.9k (3.8×) | 7.6k (2.9×) | 5.0k (1.9×) |
| **64** | 13.6k (5.2×) | 11.6k (4.5×) | 10.0k (3.8×) | 7.5k (2.9×) |

Read this table as the plan's thesis: **`F` and `N` matter more than the GPU does.** At `F = 4.2 ms` the whole
exercise is a wash or a regression at every shard count we can afford; at `F = 0.3 ms` and 32 streams it is
~5× — near the theoretical ceiling. Hence the stage order: get `F` small, then get `N` large.

> **Measured 2026-08-30: `F = 0.95 ms` in a tight loop, `1.26–1.64 ms` inside the server loop; `g = 22.7 µs`.**
> That is the `F = 1.0`–`2.0 ms` columns, and the 8-shard A/B came in at **1.33×** — squarely between those two
> cells' 1.9× and 1.2×. The model's shape is right and its thesis is confirmed: `g` turned out **3× better**
> than assumed and it changed nothing, because the GPU was never the constraint. Details in §Measured results.

**The `F` measurements we have are mutually inconsistent and must be re-taken.** batch 1 = 4209 µs and batch 32
= 2174 µs cannot both hold for a fixed per-batch cost: 32 × 68 µs already accounts for the entire batch-32
number, implying `F ≈ 0`, while batch 1 implies `F ≈ 4.1 ms`. The likely explanation is that the batch-1 figure
was taken unwarmed (cuDNN autotune) or with pageable-memory H2D plus a per-call sync. Stage 2 re-measures with a
warmed, pinned, TorchScript-traced module across batch ∈ {1,2,4,8,16,32,64} and fits `F` and `g` — that fit is
the single number that determines whether Stage 3 is needed:

```sh
cd train && .venv/bin/python -c "
import torch, time
from bbnn.model import BBNet
m = BBNet().cuda().eval()
m = torch.jit.trace(m, (torch.zeros(1,37,9,16).cuda(), torch.zeros(1,15).cuda()))
torch.backends.cudnn.benchmark = True
for b in (1,2,4,8,16,32,64):
    s = torch.zeros(b,37,9,16, device='cuda'); g = torch.zeros(b,15, device='cuda')
    for _ in range(50): m(s,g)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(200): m(s,g)
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/200*1e6
    print(f'batch {b:3d}: {dt:8.0f} us/batch  {dt/b:6.1f} us/sample')"
```

## Staged implementation order

Each stage is independently verifiable and independently revertible. Stages 0–2 land no behaviour change on the
default (tract) path.

### Stage 0 — confirm the ceiling (half a day, no architecture) — **done, gate passed at 0.853**

`BLOOD_NN_PROFILE=1` counters in `NnEvaluator` (`AtomicU64` forward count + total nanos), printed per game by
`dataset::run` (`botbowl-ui/src/dataset.rs:80`). Run the two 20-game probes above.
**Stage 0b:** sample `ps -o rss=,args= -C botbowl-ui` during a live 8-shard generate phase, to size B1.
**Exit criterion:** inference share ≥ 0.70 and forwards/game within ~2× of the predicted 28k. **If not, stop.**
Test: a unit test asserting the counter increments once per `forward_raw`.

### Stage 1 — protocol, client, fallback (CPU server; no GPU involved) — **done**

Build `botbowl-nn/src/remote.rs`, `scripts/nn_server.py --device cpu`, the per-model canary handshake,
`--nn-server`, and the tract fallback path. **Verifying the plumbing without the GPU as a confound is the whole
point of this stage.** Expected speed: none (slightly slower than tract).

**Ship `model_path` / `model_id` in the frames from day one, but let the server's registry be capacity-1.**
Generation shards use one net each, so single-model is sufficient through Stage 4a; carrying the field from the
start costs a handful of bytes and avoids a protocol version bump — and, more importantly, avoids the temptation
to bolt on a global canary that would have to be redesigned in Stage 4b anyway.
Tests:
- `remote_matches_tract_on_fixture` — env-gated on `BLOOD_NN_SERVER`, `#[ignore]`d otherwise; the parity
  fixture through the client vs through tract, `< 1e-3`.
- `remote_falls_back_to_tract_when_server_absent` — point `--nn-server` at a dead path; assert results are
  bit-identical to tract and that one warning was emitted.
- `canary_mismatch_aborts` — serve a different net than `--model` names; assert the client refuses to start.
- `request_model_id_mismatch_drops_connection` — a request whose `model_id` disagrees with the handshake is
  rejected (guards against a server-side routing bug once the registry grows in Stage 4b).
- Round-trip latency printed by the profile counter; **exit criterion: IPC overhead < 150 µs**.

> **Built, all tests green on a CPU and a CUDA server.** Two deviations: (1) the three protocol tests run
> against an **in-process fake server** so they need no Python and pass on a Mac, with the live-server versions
> env-gated beside them; (2) `canary_mismatch_aborts` is asserted at *construction* (`from_path_with_server`
> returns `Err`) rather than as a process abort, which is testable — a mismatch discovered later, on a new
> connection, still exits the process. **IPC came in at ~190 µs, over the 150 µs criterion**; accepted, because
> it is 7% of a forward against an `F` of 1.5 ms, so the shared-memory ring would be optimising the wrong term.

### Stage 2 — CUDA backend + opportunistic batching (the first real speedup) — **done, 1.33×**

`--device cuda`, TorchScript trace, warm-up over the expected `(h, w)` and batch buckets, pinned host staging
buffers, greedy drain loop, `MAX_BATCH`. Run the existing 8 shards for one generation — **still one model**: the
generate phase never needs two, so multi-model stays out of the stage where GPU risk is being retired.
**Expected: 1.2–3.4× depending on `F`** (row 1 of the table). Report `F` and `g` from the fit above.
**Falsified if:** measured throughput is below the `F` the micro-benchmark predicted — that means IPC or the
server's Python loop, not the GPU, is the cost, and the next move is the shared-memory ring, not Stage 3.
Tests: `batch_invariance` — the same input evaluated alone and padded into a batch of 32 with random neighbours
must agree to `< 1e-5`. This is the test that protects "batching must not make results depend on batch
composition or arrival order", and it should run against a live server in CI-on-the-Linux-host only.

> **Built as described, with two deviations, both deliberate.** Pinned staging buffers were **not** built: the
> per-stage counters show H2D+stack costs 294 µs of a 1676 µs batch while `module()` costs 1248 µs, so pinning
> attacks the wrong term. And the batch-invariance test is an end-to-end one against a live server (24 threads,
> each sample compared to its own batch-1 result) rather than a padded synthetic batch — same property, but it
> also exercises the queue and the arrival-order path.
>
> **Not falsified, but close to the line it warns about.** Measured 1.33× against a fit-implied ~1.5× at
> `mean_batch = 3.4`. The gap is the server's own Python loop, exactly as the falsification clause anticipated —
> but the stage breakdown puts it inside `module()` (dispatch + launch), not in IPC, so the next move is
> **Stage 3 as written**, not the shared-memory ring.

### Stage 3 — drive `F` down — **done: `F` 931 → 314 µs**

CUDA graph capture per `(model_id, h, w, batch-bucket)` with buckets `{1,2,4,8,16,32,64}`, requests padded up
to the next bucket and the padding rows discarded. Graphs eliminate per-op launch and Python dispatch entirely,
which is the suspected content of `F`. Note this makes batch size a *bucket*, which strengthens rather than weakens the
batch-invariance property (a fixed graph per bucket is deterministic).
**Expected: `F` → ~0.2–0.4 ms ⇒ 3.4× at N=8.** Avoid `torch.compile`/Triton — Pascal support is marginal and the
payoff over a traced graph is small for a 6-block tower.

### Stage 4 — raise the stream count (the multiplier) — **done, 4a and 4b**

- **4a (generation, zero Rust):** `NN_SHARDS`/`HEUR_SHARDS` from 8 shards × 600 games to 24–32 shards ×
  150–200 games; fix the seed stride (`G*1e7`); update `TRAIN_SHARDS`/`VAL_SHARDS` to hold out ~4 whole shards
  keeping the nn/heuristic val mix plan 022 specified. Gated on Stage 0b's RSS number: at 200 MB/shard, 32
  shards = 6.4 GB and fits; at 500 MB it does not, and the fallback is 16 shards (4.3× → 2.9× at `F=0.3`).
- **4b (eval: multi-model server + parallel games).** Two changes, and this is the stage that needs them:
  1. **Server registry grows past one model** (design (A) above): `(model_id, h, w)` batch keys, on-demand
     load from `model_path`, `--max-models 4`. **Built, minus the eviction — see §Failure modes.**
  2. `--parallel-games N` over `eval::run_ladder_rung`'s loop (`botbowl-ui/src/eval.rs:181`), with a
     `Mutex<LadderRow>` accumulator, plus the same flag on `dataset` (`dataset.rs:68`) for symmetry.
  Without (2) the eval phase is a **single** stream and the server makes it *slower* than tract; without (1)
  the vs-rung cannot use the server at all, since candidate and champion live in one process.
  **Read the throughput table at `N/2` for this phase**: the two nets alternate turns, so `--parallel-games 32`
  puts eval in the `N=16` row (~4.3× at `F=0.3 ms`), not the `N=32` row. That is a property of having two sets
  of weights, not of the server design — design (B) gives exactly the same halving.
  Tests: `parallel_games_matches_sequential` — same seeds, `--parallel-games 1` vs `4`, assert identical
  per-game results (each game is fully independent given its seed); `two_models_one_socket` — two
  `NnEvaluator`s over one socket, each matching its *own* tract reference on the fixture and each receiving a
  *distinct* canary, so a cross-wired registry fails the test rather than the promotion gate.
- **Expected combined with Stage 3: 4–5× for generation** (`N=32`, `F=0.3 ms` cell), **~4× for eval**
  (`N=16` row, per the halving above). Falsified if per-shard RSS or the OOM killer bites first, or if `D_cpu`
  is larger than 58.4 µs (Stage 0 measures it directly).

### Stage 5 — wire it into the loop, with supervision — **done**

`scripts/train_loop.sh` starts **one** `nn_server.py` before the generate phase, health-checks it, passes
`--nn-server` to the nn shards, restarts it up to N times if it dies, and tears it down before the train phase
so training gets the whole card. The eval phase restarts the same single server and passes `--nn-server` to
`botbowl-ui eval`, which serves both the candidate and the champion from it — one lifecycle for both phases,
which is design (A)'s second payoff. `NN_SERVER=off` disables the whole path; the loop must remain runnable
exactly as today.

## Invariants this must not break

- **Recombination purity.** Pruning (`botbowl-mcts/src/pruning.rs`) and priors (`priors.rs`) stay pure functions
  of `(state, action)` — neither is touched. The subtler question is whether a *batched* evaluation is still
  pure: strictly, cuDNN may pick a different convolution algorithm at a different batch size, so `v(state)`
  could differ by ~1e-6 depending on batch composition. Two things make this safe: (i) the batch-invariance test
  above pins it, and CUDA-graph buckets (Stage 3) make it deterministic per bucket; (ii) structurally, both
  `score_leaf` and `available_actions` run **exactly once per registered node** — a second path to the same
  state is a registry hit (`recon_mcts/src/tree.rs:2011` "Only run `GD::score_leaf` for nodes that don't exist
  in the registry"), so the DAG cannot split on a value that was computed twice. The invariant holds.
- **Dice discipline.** Untouched — no randomness is added anywhere in this plan. The server is a pure function.
- **HashOnly stays forbidden.** Untouched.
- **The CPU path stays the default and stays working.** No new Cargo dependency; `--nn-server` unset ⇒ tract,
  byte-identical to today. `cargo test --workspace` on the Mac must be unaffected — every server-touching test
  is env-gated or `#[ignore]`d.

### The parity contract, and reproducibility

`botbowl-nn/tests/parity.rs` currently asserts tract == PyTorch `< 1e-4` at two board sizes and is not
`#[ignore]`d. **Keep it exactly as is** — tract remains the default runtime and the reference. Add, beside it:

1. `remote == tract` on the same committed fixtures, `< 1e-3` (looser: GPU fp32 reassociates conv reductions),
   env-gated on `BLOOD_NN_SERVER`.
2. **batch invariance** (Stage 2), `< 1e-5`.
3. `value_home_i64` agreement across backends on ~200 sampled states: `|Δ| ≤ 2` on the ±1000 scale and identical
   sign. This is the assertion that actually matters for search behaviour, since the value is cast to `i64`.

> **Status after stage 2:** (1) and (2) are built and green (`tests/remote.rs`); `tests/parity.rs` is untouched
> at 1e-4 as required. **(3) is not built** — the fixture-level agreement is ~1e-5, an order below the `|Δ| ≤ 2`
> that ±1000 scaling would need to break, so it was not the risk worth spending the stage on. It should land
> with Stage 4's distributional acceptance test, where sampled *states* (rather than a fixed tensor) start
> mattering.

**Bit-identical reproduction of old corpora is explicitly not a goal, and is already unattainable**: plan 020
records that MctsBot games are not reproducible from seeds at all, because `recon_mcts`'s std `HashMap`s
randomise tie-break order per process. Different numerics will flip a small number of near-tied PUCT decisions
and change trajectories; that is the same class of variation the loop already tolerates. The contract we *do*
keep is distributional: a `--nn-server` generation must produce a corpus statistically indistinguishable from a
tract one. **Stage 4's acceptance test is exactly that** — one 300-game shard each way, compare TDs/drive
(baseline 0.79), scoreless fraction (0.21), mean samples/drive (37.2), and the −1/0/+1 value-target split
(31/34/35). Any of these moving by more than a couple of points means something is wrong, not fast.

## Failure modes

- **Server dies mid-generation.** Today a dead shard is a warning and its partial JSONL is kept
  (`scripts/train_loop.sh:253`); a dead *server* would kill every nn shard at once and silently halve the
  corpus. Two mitigations, both required: (i) the client falls back to tract on connect/IO error — the shard
  gets 12.5× slower but **finishes**, and logs `NN_SERVER_FALLBACK` once plus a count at exit; (ii) the loop
  supervises and restarts the server. Fallback is the load-bearing one: it preserves the existing failure
  semantics exactly.
- **Client blocks forever on a wedged server.** `SO_RCVTIMEO` of 5 s on every read; timeout ⇒ fallback to tract
  for the rest of the game, then retry the connection on the next game.
- **Batch-timeout starvation.** Designed out by having no timeout (greedy drain). The residual case — a single
  active shard paying batch-1 GPU latency (~4.2 ms, worse than tract's 2.6 ms) — is real at the tail of a
  generation when 31 of 32 shards have finished. Mitigation: the client falls back to tract when the server
  reports a rolling mean batch size < 2 (server pushes the hint in the response header). Cheap; add only if
  Stage 4 measurement shows the tail costs anything.
- **GPU OOM on 6 GB.** Unlikely (model ~2 MB — two or four loaded is still noise — activations at batch 64
  ≈ 50 MB, context ~300 MB), but real if a
  future tier (20x11, 26x15) is combined with a large `MAX_BATCH`. Guard: cap `MAX_BATCH` and set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the loop tears the server down before the train phase so
  the two never contend.
- **Host RAM / OOM killer.** The real constraint (15 GB, with an OOM-kill history in plans 020/021). Gated on
  Stage 0b's RSS measurement; the loop already logs free disk per phase and should log free RAM too.
- **Deadlock.** Only if Stage C (multi-worker MCTS) is ever attempted; blocking ms-scale calls under
  `children.write()` / `score.write()` is the plan-013 fairness-deadlock regime. If tried, build `recon_mcts`
  with the `lockref-guard` feature in debug first.
- **Latency spikes** from cuDNN autotune or Python GC on the first request of a shape: warm every
  `(h, w) × bucket` at startup before accepting connections; `gc.freeze()` after warm-up.
- **Wrong model served, or cross-talk between the two eval nets.** The per-model canary catches a wrong
  resolution at handshake; the `model_id` echo check catches a server-side routing bug at request time. Both
  abort rather than degrade. This is the failure that would otherwise corrupt a promotion-gate verdict
  invisibly, so it gets two independent guards rather than one.
- **Model-registry thrash / eviction. Designed out — `--max-models 4` and nothing is ever evicted.** LRU was
  this plan's suggestion and it was the wrong trade, for a reason only visible after Stage 3: a `model_id` is
  also the key of that model's captured CUDA graphs, and those are owned by the batcher thread. Evicting from a
  connection thread would either leak ~400 MB of VRAM per evicted model or need cross-thread graph teardown —
  real complexity guarding a path that cannot fire when the cap is 4 and the eval phase holds 2. Over the cap
  is a loud refusal, and the fix is to raise the flag. Every load is logged.

## sm_61 is old: what is and is not worth doing

- **fp16: no.** GP106 runs fp16 at **1/64** of fp32 throughput (only GP100 has the 2:1 rate). Half precision on
  this card is a large slowdown, not a speedup, and there are no tensor cores. Revisit only on a newer GPU.
- **TF32/bf16: not available on Pascal at all.**
- **TensorRT: probably not.** It would fuse conv+BN+ReLU and cut `F`, which is the metric that matters — but
  CUDA graphs over a traced module (Stage 3) get most of that benefit for a fraction of the effort, and modern
  TensorRT releases have dropped Pascal support, so we would be pinning an old TRT alongside an already
  carefully-pinned cu126 torch. Revisit only if Stage 3's `F` lands above ~0.5 ms *and* the GPU (not the CPU) is
  measurably the wall.
- **Bigger nets are nearly free.** At batch 32 the card sustains ~1.9 TFLOPS effective on this model. The tower
  is 6 blocks × width 64 (`train/src/bbnn/model.py:44-59`), ~0.13 GFLOP/forward. Once inference is batched, a
  2–3× larger net costs almost nothing in wall clock — a capability lever that this plan unlocks as a side
  effect, and arguably the more interesting one given the repo's current focus.

## What is uncertain, and what must be measured before committing

1. **The 84.8% inference share** assumes nn and heuristic search cost the same per decision. Stage 0 measures it
   directly. *Everything in this plan is downstream of this number.*
2. **`F`, the fixed per-batch cost.** The two existing CUDA measurements are mutually inconsistent (see above).
   Stage 2's fit decides whether Stage 3 is needed and how much Stage 4 can deliver.
3. **Per-shard RSS**, which decides whether N=32 is reachable (Stage 0b).
4. **`D_cpu` = 58.4 µs** is derived, not measured, and it sets the 6.6× hard ceiling. If MCTS search is more
   expensive on nn-generated states than on heuristic ones (longer drives, more legal actions), the real
   ceiling is lower.
5. Whether **UDS IPC really costs < 150 µs** under 32 concurrent clients on 4 cores, or whether the server's
   Python accept/read loop becomes the bottleneck and forces a shared-memory ring.
6. **The cost of GPU context-switching between two processes is unmeasured** — it is the main argument against
   design (B), and it is an argument from risk, not from data. If (A)'s registry ever turns out to be more
   trouble than it is worth, the honest next step is to measure it (two concurrent single-model servers vs one
   two-model server, same total stream count, compare fitted `F`) rather than to re-litigate it on principle.

None of 1–5 requires building anything beyond Stage 0's counter and a standalone benchmark script.

> **Resolved 2026-08-30:** #1 measured at **0.853** (and cross-checked by the subtraction, 0.851). #2 fitted at
> **`F = 0.95 ms` / `g = 22.7 µs`**, 1.5 ms in-server — Stage 3 is required. #3 per-shard RSS **~290 MB**, so 32
> shards is tight on 15 GB and 16–24 is the safe read. #4 `D_cpu` measured at **62 µs** aggregated (derived
> 58.4), so the 6.6–6.8× ceiling stands. #5 IPC is **~190 µs** and the Python loop *does* become the wall, but
> above ~16 streams and after `F`, not before it. #6 (two-process context switching) remains unmeasured, and
> stays unmeasured while the registry is capacity-1.

## Measured results (stages 0–2, 2026-08-30)

Linux host, 14x7 tier (encoded 37×9×16), `models/bbnet_14x7_gen01.{onnx,pt}`, `--mcts-iters 1000`, release
build. **Caveat on every wall clock below: another agent held three `botbowl-ui eval` processes at ~90% CPU
throughout, so 3 of the 4 physical cores were not ours.** The Stage-0 *share* is a ratio measured inside one
process and is robust to that; the Stage-2 arms are reported as a same-conditions A/B, not as absolutes.

### Stage 0 — the ceiling is real. Gate ≥ 0.70, **measured 0.853**

| arm | games | decisions | wall | forwards | in `forward_raw` | share |
|---|---|---|---|---|---|---|
| `nn-value` | 12 | 390 | 769.0 s | 226,923 | 656.2 s | **0.853** |
| `heuristic` | 20 | 523 | 153.7 s | 0 | — | — |

- **Direct** (the counter divided by wall, which the plan could not do before): 656.2/769.0 = **0.853** against
  the predicted 0.848. **Amdahl ceiling 6.8×.**
- **The subtraction now cross-checks it, which is the part that was assumed rather than known.** nn
  1971.7 ms/decision − heuristic 293.9 ms/decision = 0.851. So "nn and heuristic search cost the same per
  decision" (uncertainty #1 — *everything in this plan is downstream of this number*) holds to 0.2 points.
- **582 forwards/decision** (18,910/game) vs ~754 predicted — inside the "within ~2×" exit criterion and the
  right shape (one `score_leaf` per materialised node, minus registry hits and chance nodes).
- **`D_cpu` measured, not derived:** (769.0 − 656.2)/226,923 = **497 µs** of non-inference CPU per forward per
  process = 62 µs over 8 shards, vs the derived 58.4 µs (uncertainty #4 — the 6.6× ceiling survives).
- **Stage 0b:** per-shard RSS **~290 MB**, so 32 shards ≈ 9 GB on a 15 GB box. Tight; 16–24 is the safe read.
- `mean_us = 2892` per tract forward (vs 2611 quoted here) — the contended box.

### Stage 1 — protocol, client, fallback

`botbowl-nn/src/remote.rs` + `scripts/nn_server.py`, debugged against a **CPU** server exactly as intended.
Six tests in `botbowl-nn/tests/remote.rs`; the three that use an in-process fake server run everywhere with no
Python, the three `live_*` ones skip unless `BLOOD_NN_SERVER`/`BLOOD_NN_MODEL` are set, and all six passed
against the real sidecar on **both** `--device cpu` and `--device cuda`:

- `remote_matches_tract_on_fixture`, `canary_mismatch_refuses_to_start`,
  `remote_falls_back_to_tract_when_server_absent` (**bit-identical** to tract, one warning),
  `live_server_matches_tract` (the real net: torch `.pt` vs tract `.onnx` < 1e-3 — the interlock doing its job),
  `live_server_drops_connection_on_model_id_mismatch`, `live_server_is_batch_invariant`.
- **374,815 consecutive remote forwards with `fell_back_to_tract=0`** in the Stage-2 arm.
- **IPC overhead ≈ 190 µs**, above the < 150 µs exit criterion but not the problem: client-observed 2888 µs vs
  server-accounted 2702 µs (queue 1097 + batch 1605) under the 8-shard arm. It is 7% of a forward against an
  `F` of 1.5 ms, so the shared-memory ring stays unbuilt.

### Stage 2 — CUDA, and the `F` that decides everything

`nn_server.py --bench --device cuda` (warmed, TorchScript-frozen, one `synchronize()` per batch):

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| µs/batch | 997 | 1232 | 1088 | 1139 | 1125 | 1384 | 2589 |
| µs/sample | 997 | 616 | 272 | 142 | 70 | 43 | 40 |

**Fit: `F = 953 µs/batch`, `g = 22.7 µs/sample`.**

1. **Both prior `F` numbers were wrong, and the truth is in between.** The 4209 µs batch-1 figure was indeed an
   unwarmed artefact — warmed, batch 1 costs 997 µs. But `F ≈ 0` was equally wrong: `F` is ~1 ms. Meanwhile
   `g = 22.7 µs` is **3× better** than the 68 µs assumed, and it changed nothing — confirming the plan's thesis
   that the GPU was never the constraint.
2. **Inside the server loop `F` is 1.26–1.68 ms**, and the new per-stage counters say where it goes:
   `batch = 1676 µs (stage 294 + fwd 1248 + post 135)`. The **fixed cost lives inside `module()`** — Python/ATen
   dispatch and ~40 kernel launches for a 6-block tower — not in the H2D staging (294 µs, part of which is the
   `np.stack`) and not in the socket writes (135 µs). **That is exactly what Stage 3's CUDA graphs eliminate,
   and it is why pinned staging buffers were left unbuilt: they attack the 294 µs term, not the 1248 µs one.**

**End-to-end A/B, 8 shards × 2 games, back to back under identical contention:**

| arm | forwards | wall | aggregate | mean latency/forward |
|---|---|---|---|---|
| tract (today) | 367,485 | 419.8 s | 875 /s | 4531 µs |
| `--nn-server` CUDA | 374,815 | 321.2 s | **1167 /s** | **2888 µs** |

**1.33× throughput, 1.57× latency** — between the table's `F = 1.0 ms` (1.9×) and `F = 2.0 ms` (1.2×) cells, as
the fit predicts. It also used **~2.4 cores against tract's ~5.4**, which matters on a 4-core box but is not
throughput. Server-side during that arm: `mean_batch = 3.4` — **8 shards do not offer 8 concurrent requests**,
because a shard spends `D_cpu` searching between forwards and, starved of CPU, rather more than that.

**Server ceiling vs stream count** (`nn_server.py --loadgen N`, zero-think-time clients — the search taken out,
to answer "what could the server deliver if Stage 4 raised `N`?"):

| streams | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| forwards/s | 756 | 791 | 1279 | 2170 | 3861 | **6038** |

Still climbing at 32 (mean batch ≈ 25 there). **6038/s is 2.3× this plan's unloaded 2600/s tract baseline and
6.9× the 875/s tract actually managed under the same contention** — so the server is not the wall at N=8; the
offered concurrency is. Note the curve is measured with a *Python* load generator on a busy box, so it is a
floor, not a ceiling. Above ~16 streams the per-request Python work in the connection threads (recv →
`frombuffer` → `queue.put`, then one `sendall` each) starts competing for the GIL, and that — not the GPU — is
the next wall after `F`.

**Batch invariance holds** (`live_server_is_batch_invariant`): 24 threads, different inputs, each compared to
its own batch-1 reference — agreement < 1e-5 on CUDA. Batch composition does not leak into a result, so the
recombination-purity argument survives in practice and not only in principle.

### What this means for stages 3–5

- **Stage 3 is not optional, it is the stage.** `F > 1 ms` was the plan's own trigger, and the stage breakdown
  says the cost is precisely the launch/dispatch that CUDA graphs remove. Do it before anything else.
- **Stage 4a is the other half**, and the loadgen curve is the evidence it will pay: at 8 streams the server is
  idle a third of the time waiting for work. Remember the seed-stride bug (`G*1e7`) blocks it at K ≥ 10.
- **Do not ship `--nn-server` into `train_loop.sh` yet.** 1.33× does not justify a new single point of failure
  in a 14-hour generation; the fallback works, but the operational surface is not worth it until stages 3+4
  land the 3–5×.
- One avoidable cost found on the way: `MctsBot` spawns its workers per decision (`thread::scope`), and
  connections are `thread_local!`, so a shard opens **a connection per decision**. Harmless at ~4/s, but a
  connection pool keyed on the socket rather than the thread would remove it.

## Measured results (stages 3–5, 2026-09-01)

Same host, 14x7 tier, `--mcts-iters 1000`, release build, `--evaluator nn-value`. **Caveat: Chrome held
~1.3–1.7 cores throughout, in every arm.** All comparisons are back-to-back under that same load, so the
ratios hold; the absolutes would be better on a dedicated box. The champion nets were deleted in the plan-023
reset, so these runs use a **random-weights** `bbnet_14x7_bench.{pt,onnx}` — irrelevant to timing (the tower
shape is what costs), and it turned out to be the thing that exposed the vacuous test in Stage 3 below.

Arms are reproducible with `scripts/nn_throughput_probe.sh <arm> <shards> <games/shard> [extra]`.

### Stage 3 — CUDA graphs. `F` 931 → 314 µs

`nn_server.py --bench --device cuda`, warmed, **each iteration ending in the device→host read a served batch
actually pays**:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| eager µs/batch | 873 | 1133 | 1143 | 1265 | 1518 | 1878 | 3346 |
| **graph** µs/batch | **408** | **443** | **482** | **578** | **921** | **1614** | **2998** |

**Fit: eager `F = 931 µs`, `g = 34.9 µs`; graph `F = 314 µs`, `g = 40.9 µs`.**

- **Stage 2's `F = 953 µs` tight-loop figure was right, and its `g = 22.7 µs` was not.** Stage 2's bench
  synchronised once per *sweep*, not per batch, so launches pipelined across iterations and `g` read ~30% low.
  Fixed; the eager column above now agrees with what Stage 2 measured *inside* the server loop.
- Graphs are worth **2.1× at batch 1 and ~1.1× at batch 32**. That is the right shape and the reason they were
  the correct stage: a graph buys launch overhead, and at small batch launch overhead is all there is. Our
  batches *are* small (below).
- 12 graphs cost **396 MB of VRAM** and **0.4 s** to capture. Capture happens on the batcher thread before the
  listener opens.

Server ceiling (`--loadgen`, zero think time), against Stage 2 on the same script:

| streams | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| stage 2 | 756 | 791 | 1279 | 2170 | 3861 | 6038 | — |
| **stage 3** | **1792** | **1976** | **3363** | **5012** | **6816** | **7994** | **8413** |

### Stage 4 — the offered concurrency was the wall, and it still is

With `F` at 314 µs the server drains faster than 8 shards can fill it: during the 8-shard arm
`mean_batch = 1.77`, and **47k of 83k batches were size one**. The sidecar was idle waiting for work. Three
ways to raise `N` were measured rather than argued:

| mechanism | µs CPU/forward as N rises | verdict |
|---|---|---|
| **`--parallel-games` P=3/4/6** | 1073 / 1090 / 1128 | **built.** Flat — parallel games are free per unit of work |
| **more shards 8/16/24** | 999 / 1103 / 1102 | works, free, but 24 shards = 5.1 GB of 7 available |
| **`--mcts-workers` W=1/2/4/8** | **1043 / 1132 / 1204 / 1231** | **rejected.** Monotone tax, and it changes the search |

- **`--mcts-workers` is the one arm that costs something per unit of work** — +18% CPU/forward at W=8. It buys
  wall clock while cores sit idle and loses once they do not, *and* virtual loss changes which leaves get
  expanded. So option C stays rejected, but now on this host's data rather than plan 015's extrapolation.
- **`--parallel-games` beats more shards on RAM, not on speed.** At 24 streams they are within noise
  (8×P=3: 3.21 s/game; 24 shards: 3.07–3.47 s/game) — but one is 8 processes and the other is 24. Per-shard
  RSS is ~238 MB, so sharding alone caps out near 24–32 on this box; `--parallel-games` reaches the same
  concurrency in a third of the processes and keeps going.
- **P = 4 is the peak here** (2.62–2.68 s/game); P = 6 is past it (3.49).

**The CPU is now the binding constraint, and that is the real result.** tract is internally multi-threaded and
was burning **6679 µs of CPU per forward**, saturating 5.3–7.2 of this box's 8 logical (4 physical) cores. The
sidecar cuts that to **~1065 µs, a 4.3× reduction**, which is why the fast arm is also the one that leaves
cores free. The Amdahl ceiling of 6.6–6.8× is a *CPU* ceiling; at 3.79× there is headroom left, and the
loadgen curve says the server is not what is holding it (8413 forwards/s available against 4051 delivered).

### Stage 5 — in the loop

`train_loop.sh` starts one server before generate, waits for the socket (which appears only after the graph
captures, so it is the right readiness signal), passes `--nn-server --parallel-games 4` **to the nn shards
only** — a heuristic shard has no NN call to batch and extra games there would just take cores from the shards
that do — and tears it down before train. A `trap` covers every exit path so no run leaves 400 MB of VRAM
behind. `NN_SERVER=off` restores the old behaviour exactly.

The failure mode that needed attention is not a crash but a *silence*: a shard whose server never came up
falls back to tract, finishes correctly, and is merely 4× slower. The phase now greps `NN_SERVER_FALLBACK`
and says so in `status.md`.

### Three bugs found, all of the silent kind

1. **The registry keyed models on the client's raw path string.** `models/x.onnx` and `/abs/…/models/x.onnx`
   were two different models. At the then-default `--max-models 1` the second spelling was *rejected*, and
   that shard would have run its whole generation on tract behind one warning line; above 1 it loads the same
   weights twice and each copy sees half the batch. Both present only as lost speed. Identity is now the
   resolved `.pt`; `live_server_identifies_a_model_by_its_weights_not_its_path_string` pins it. This is
   exactly the shape of mistake `train_loop.sh` would make, since it passes `$CHAMP` as an absolute path while
   a hand-started server is usually given a relative one.
2. **`live_server_is_batch_invariant` was vacuous.** It tracked `worst = worst.max(v)` from `0.0` — the
   largest *value*, not the largest deviation — so for any net whose values are negative it returned `0` and
   compared `0` against the reference. It had been passing for that reason, not because batching was
   invariant. It now tracks `max |v − reference|` and passes non-vacuously (`mean_batch 13.2, pad 0.11` during
   the storm). A random-weights net is what exposed it.
3. **A pre-existing flaky panic in the generator at the default 28x17 board.** `recon_mcts/src/tree.rs`
   "could not remove dropped node as child's parents", in ~2 runs of 3, reproduced with **no new flag
   involved**:
   `botbowl-ui dataset --mode random-start --games 12 --seed 7700 --mcts-iters 8 --evaluator heuristic`.
   **It did not reproduce at the 14x7 tier** (6/6 clean at 8 iters, 3/3 at 1000), so the training loop is not
   exposed — but it is a real latent bug and it is unowned. `botbowl-ui/tests/parallel_games.rs` skips above
   20 board-width for this reason and records the repro.

### Postscript (2026-09-02): the baseline moved, the ratios did not

The first post-reset `gen00` bootstrap runs at **18.8 s/game per shard**, against plan 022's reference of
**6.9** — 2.7× slower, on the *heuristic* path, which nothing in this plan touches. Measured on a quiet box
with the 8 shards saturating it (769% CPU), so it is not contention.

It is not a regression, and the corpus says so. Against plan 021's health baselines:

| | measured (310 drives) | plan 021 baseline |
|---|---|---|
| TDs/drive | **0.75** | 0.79 |
| scoreless | **0.25** | 0.21 |
| samples/drive | **26.7** | 19.6 |

Outcome quality is on baseline; what changed is that a drive now takes **1.36× more decisions**. The only thing
between the two measurements is the plan-023 work — chiefly the mover-tagging fix (`e107f06`), which corrected
what `available_actions` reports and therefore what the search explores. A correct search doing more work per
drive is the expected shape of that fix, and 1.36× more decisions plus a costlier decision covers the 2.7×.

Two consequences for reading this document:

- **The absolute "before" numbers at the top of this plan (863 min, 86.3 s/game) predate the plan-023 fix and
  are no longer the baseline.** A generation costs more now, in both arms.
- **Every ratio measured on 2026-09-01 is unaffected**, because both arms of every A/B ran the current code
  back to back. The 3.79×, the `F` fit and the CPU-per-forward numbers all stand.

### Stage 4b — eval, done the same day, because it became the bottleneck

Speeding generation up ~4× promoted **eval** to the loop's dominant phase: a single process playing full games
one at a time, on one core of eight, at plan 022's measured **117–690 min** against a generate phase now near
200. So `run_ladder_rung` got the same treatment as `dataset`, and the loop's eval phase now starts the sidecar
too — which is the first phase that actually *needs* the multi-model registry, since candidate and champion
live in one process and share one socket.

Both halves are required together and neither works alone: without `--parallel-games` the phase offers one
stream, and a batching server at one stream is slower than tract.

**Unlike `dataset`, this one asserts exact equivalence — and that assertion is load-bearing.** `play_game`
seeds both bots from the game index, and `ScriptedBot`/`RandomBot` honour a seed, so a scripted-vs-random rung
is a pure function of `(seed, index)`. `--parallel-games 1` and `4` produce byte-identical per-game rows *and*
an identical `LadderRow`. The row is what the promotion gate reads: a lost counter update would silently turn
a REJECTED into a PROMOTED and would be invisible everywhere else. `dataset` cannot have this test because it
always drives `MctsBot` (plan 020: not seed-reproducible).

`EVAL_PARALLEL_GAMES` is a separate knob from `PARALLEL_GAMES` and is set relative to the box differently: eval
has no generate shards competing, but a rung worker holds **two** MCTS trees (candidate + opponent) where a
dataset worker holds one.

### What is left

- **The distributional acceptance test** (one 300-game shard each way, comparing TDs/drive, scoreless
  fraction, samples/drive and the value-target split against plan 021's baselines) has **not** been run —
  there was no trained champion after the plan-023 reset. It should gate the first real generation that uses
  the sidecar, i.e. `gen01`. Note the postscript above: `samples/drive` has moved to ~26.7 on the current
  engine, so compare a sidecar corpus against a **tract corpus generated today**, not against plan 021's 19.6.
- **The connection-per-decision churn** noted after Stage 2 is still there and still not worth fixing:
  amortised over ~580 forwards per decision it is under 10 µs/forward.
- **The heuristic hedge shards are now the generate phase's long pole** (1800 games at 6.45 s/game aggregate
  against the nn shards' 181 min). Inference is no longer what generation waits on. Speeding that up means
  changing the 5 nn / 3 heuristic mix that plan 022 tuned, so it is a corpus-composition decision, not a
  performance one.
- **The mirror match is a 4-hour serial pre-flight** and was skipped at 46/100 on 2026-09-02 to get the box
  onto the bootstrap (`runs/loop14x7/mirror.partial-46of100.log`). It runs `eval`, so it now inherits
  `--parallel-games` and should be re-runnable in well under an hour; `train_loop.sh` does **not** yet pass the
  flag to it.

## Cross-references

- plan 022 — the loop this speeds up: shard layout (`scripts/train_loop.sh`), the seed scheme, the dead-shard
  warning semantics, the cu126/sm_61 trap that makes the Python sidecar the cheap option, and the **net-vs-net
  eval rung** (`--vs-evaluator` / `--vs-model`) that puts two models in one process and forces the multi-model
  design. Its "After the weekend" strength-ladder idea is the reason (A) is preferred over (B).
- plan 021 — the corpus health statistics (0.79 TDs/drive, 21% scoreless, 37.2 samples/drive, 31/34/35 value
  split) that Stage 4's distributional acceptance test compares against.
- plan 020 — "NN generation cost (~2–4× heuristic) if gen-2 wants 10k+ games" under §Next-next; also the
  seed-irreproducibility gotcha that makes bit-identical corpora impossible.
- plan 017 — the architecture, the ONNX export contract (`train/src/bbnn/export.py`, opset 17, dynamic N/H/W)
  and the parity fixture this plan extends rather than replaces.
- plan 015 — multi-worker MCTS speedup numbers (1.24× @ 10k iters) and the contention analysis behind rejecting
  option C.
- plan 013 — the `Node.score` fairness deadlock, the reason blocking calls under held locks are treated as
  dangerous here, and the `lockref-guard` feature.
- plan 016 — lazy expansion, which is why there is exactly one `score_leaf` per descent and therefore why
  "~754 forwards per 1000 iterations" is the expected shape.
- `botbowl-nn/CLAUDE.md` — "encoding lives here, in Rust, once"; the reason the server must consume the
  trainer's own `.pt` rather than a re-implemented model.
