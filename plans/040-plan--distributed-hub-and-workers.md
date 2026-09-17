# Distributed generation and eval: one hub, many workers

**Status:** Designed 2026-09-16. **Phase 0 done 2026-09-16** (`botbowl-play` extracted;
verified behaviour-neutral against the pre-extraction binary at 14x7: identical corpus metadata,
byte-identical search-free eval output, identical report labels/schema). **Phase 1 done
2026-09-16**: `botbowl-hub-proto`, `botbowl-hub` (serve / job eval / status), `botbowl-worker`;
eval jobs end to end; `train_loop.sh`'s eval phase runs through the hub with the local worker
started inside the sidecar's lifetime. Verified with real 14x7 binaries: search-free job equals
`botbowl-ui eval` byte for byte; NN candidate vs NN anchor ships the net by hash and runs; a
schema-incompatible net fails the job in 5 s with the reason and the worker survives (models are
probe-forwarded before use — a panic inside `MctsBot` poisons the tree and aborts the process).
Deviations from the design below: proto depends on `botbowl-play` (not engine-free — no wasm
target exists to protect); `job.json` persistence and `Drain`-on-shutdown deferred to phase 3;
the `mcts-heuristic` rung and `--vs` opponent both use the same `opponent_search` knobs as
before. **Phase 2 done 2026-09-17**: `job generate` — one job writes every shard of a
generation (`--out-dir`, `--shards`, `--heuristic-shards`; shard K seeded at
`seed_base + K*stride`, so the seed sets are exactly the per-shard `botbowl-ui dataset`
ones — round-robin by seed was dropped in favour of that); workers zstd-compress each
trajectory's JSON line and the hub appends the bytes verbatim, so shard files are
`DatasetWriter` format; the corpus label's backup rule is resolved on the hub at submit, not
on the worker. `train_loop.sh`'s bootstrap and per-generation generate phases run through the
hub (`GEN_PARALLEL_GAMES`, default `8*PARALLEL_GAMES`, sizes the local worker). Verified with
real 14x7 binaries against `botbowl-ui dataset` on the same seeds: identical seed sets and
provenance metadata per shard, nn shard shipped by hash, heuristic hedge shard on the
heuristic evaluator. Phases 3+ not started.
Decided: dirty-tree workers are refused (no exception); Windows deferred until a box exists;
`job --wait` progress output still open. Independent of plan 039 (mixed board sizes);
the two compose because board dims are a runtime `GameState` field within one compiled capacity.

## Problem

The loop (`scripts/train_loop.sh`) runs on one 8-core box with a GTX 1060. Per generation
(plan 030 numbers): generate ~256 min, train ~35 min, eval ~287 min. Only train needs the GPU,
and the GPU is idle ~92% of the cycle. Generate and eval are CPU-bound MCTS, bounded by RAM at
400-500 MB per tree (plan 024), and both are embarrassingly parallel per game. Plan 032's
statistics (SE 0.041 at 120 games; detecting 0.55 needs ~628 games) say eval is where extra
games are worth the most.

Other machines are available (Linux x86_64 servers, Apple silicon laptops, possibly Windows),
but they sit on other networks (home, office VPN not assumed) and come and go.

## Decisions

1. **Hub and worker, workers dial out.** The office desktop runs `botbowl-hub` (one forwarded
   TCP port). Workers connect outbound over a websocket, so laptops behind NAT need nothing.
   The desktop also runs a worker against `localhost` and keeps the GPU sidecar
   (`--nn-server`) for itself; remote workers run in-process tract. The hub does not know or
   care which evaluator a worker uses.

2. **Hub replaces the loop's generate and eval phases wholesale.** The hub is a daemon that
   owns a job queue and writes exactly the on-disk layout the loop already consumes
   (`genNN/shard*.jsonl`, `eval.games.jsonl`, `report.json`). `train_loop.sh` swaps the two
   local phases for blocking `botbowl-hub job ... --wait` calls. Prepare and train do not move.
   No single-box fallback path is kept in the script; a hub with one local worker *is* the
   single-box configuration.

3. **Native worker binary. No wasm.** Nothing in engine/mcts/nn compiles to wasm32 today
   (plan 034 phase 4 lists the blockers: `getrandom`, file IO, `Instant`, `thread::scope` in
   `MctsBot`, tract on wasm). A 400 MB tree is near the practical wasm32 ceiling and browsers
   throttle unfocused tabs. Days of work for a throttled laptop core. Revisit only if the
   worker fleet becomes "many strangers' browsers", which it is not.

4. **The hub serves the worker binary.** `GET /worker/<target-triple>` plus `GET /install.sh`
   for a one-liner:
   ```sh
   curl -fsSL http://HUB:PORT/install.sh | sh -s -- --token TOKEN
   ```
   Binaries are cross-built on the desktop (`cargo zigbuild`, see phase 4) into a directory
   the hub serves. Building from the repo remains the fallback for a new platform.

5. **Compatibility = exact git commit + clean tree + board capacity.** Already baked in by
   `botbowl-data/build.rs` (`BOTBOWL_GIT_COMMIT`, `BOTBOWL_GIT_DIRTY`); capacity comes from
   `botbowl-engine`'s build-time env. A worker whose commit differs is told `Update{url}`
   and self-updates: downloads the matching binary for its triple from the hub, verifies the
   BLAKE3 the hub announced, replaces itself, re-execs. If the hub has no binary for that
   triple, the worker exits with the commit to rebuild from. Rationale: a semver we must
   remember to bump is exactly what gets forgotten, and mixed commits inside one corpus
   defeat the per-trajectory stamping the programme relies on. A docs-only commit does
   invalidate workers; self-update makes that a 2 MB download, not a chore. Hub flag
   `--allow-commit-mismatch` for when it is known safe; then the wire is guarded only by
   `PROTOCOL_VERSION` (bumped on any frame change, as `botbowl-nn/src/remote.rs` does).

6. **Auth: shared bearer token in `Hello`.** The data is not sensitive; the only thing worth
   preventing is junk trajectories entering the corpus, and that needs a secret the *worker*
   holds. A server private key with a shipped public key authenticates the hub to workers,
   which protects nobody we care about. Token is generated on first hub start into
   `runs/hub.token` and printed. Rejected `Hello` closes the socket.
   The public-internet exposure does add one real integrity concern: the served binary.
   Phase 5 adds TLS with a self-signed cert whose fingerprint is baked into the worker (that
   is the place the "public key in the binary" idea earns its keep), covering both the token
   and the download. Until then the risk is a MITM on the office uplink, accepted.

7. **Work unit = a small seed batch, results stream per game.** A `Task` carries 4-8 seeds
   (generate) or 4-8 game indices (eval). Each finished game is sent immediately as its own
   frame. A worker that vanishes (no heartbeat for 120 s, or socket closed) has its
   unfinished seeds requeued; the hub dedupes on `(job_id, seed)` so a slow worker that
   reappears cannot double-count. Seeds pin the game setup, not the search
   (recon_mcts HashMap order varies per process), same as today.

8. **Model shipping by content hash.** `model_id = blake3(onnx bytes)`. `Hello` carries the
   ids the worker has cached (`~/.cache/botbowl/models/<id>.onnx`). A task names a
   `model_id`; the hub sends `Model{id, bytes}` (~2 MB) only if the worker lacks it. Eval
   tasks name two ids (candidate, opponent). Identity is bytes, not filename, matching
   `remote.rs`'s "resolved weights file, not path string" rule.

9. **Wire format: websocket binary frames, `postcard` via serde, trajectory payload
   zstd-compressed.** A trajectory is ~575 KB of JSON (full `GameState` per sample); zstd
   level 3 makes it ~30 KB. A 4800-game generation is ~150 MB inbound to the hub, fine for a
   home upload. The `Trajectory` itself stays serde_json inside the frame so the hub writes
   the bytes verbatim to the shard file and `prepare` sees the format it knows.
   `botbowl-data::FORMAT_VERSION` travels in the meta as now.

10. **Worker sizing is the worker's call, hinted by the hub.** `Hello` reports cores and
    RAM. The hub replies with `parallel_games = min(cores, ram_mb / 1024)` (1 GB per stream:
    one tree ~500 MB at 1000 iters, two trees for eval) and a `--parallel-games` override
    on the worker wins. Windows and macOS laptops on battery can pass `--nice`.

11. **Hub state lives in the run directory, not memory.** Jobs are directories under
    `runs/<run>/genNN/`; the queue of outstanding seeds is derived on startup from
    `job.json` minus the seeds already present in the output files. A hub restart mid-job
    loses nothing but in-flight games. No database.

## Crates and binaries

```
botbowl-play/          # NEW lib: play one game, return a Trajectory or an eval line
botbowl-hub-proto/     # NEW: serde message types, PROTOCOL_VERSION, engine-free
botbowl-hub/           # NEW: bin `botbowl-hub` (daemon + `job` CLI), axum 0.8 ws, tokio
botbowl-worker/        # NEW: bin `botbowl-worker`, tokio-tungstenite client, std threads for games
```

`botbowl-play` is the refactor prerequisite: `botbowl-ui/src/dataset.rs` (`self_play_trajectory`,
`random_start_trajectory`, `curriculum_trajectory`) and `eval.rs` (`run_rung_games`, the
per-game JSON line, `LadderRow` accumulation) become library functions taking a plain config
struct and a seed. `botbowl-ui dataset` and `eval` keep working by calling into it, so the
`#[ignore]`d benchmarks and every existing script are unaffected. Bot construction moves with
it (`bot_factory.rs`). `botbowl-nn` remains the only place that knows about tract vs sidecar.

The hub binary is one process with two roles:

- `botbowl-hub serve --run-dir runs/az14x7v6 --bind 0.0.0.0:7777 --dist-dir dist/`
- `botbowl-hub job generate --gen 12 --games 4800 --seed-base ... --mcts-iters 1000 \
     --model models/bbnet_14x7_gen11.onnx --mode random-start --shards 8 --wait`
- `botbowl-hub job eval --gen 12 --candidate ... --rungs scripted --games 120 \
     --vs-model anchor.onnx --vs-games 40 --mcts-iters 1000 --wait`

`job` talks to the daemon over a local HTTP endpoint (`127.0.0.1` only), blocks until the
job's `.done` appears, exits nonzero on failure, prints the same summary lines the loop
logs to `status.md` today. Shard files: the hub writes `shard$K.jsonl` round-robin by seed
so `window_shards`, train/val split by shard index (0-3,5,6 vs 4,7) and `td_rate.py` are
untouched.

## Protocol (`botbowl-hub-proto`)

```rust
pub const PROTOCOL_VERSION: u32 = 1;

// worker -> hub
enum ToHub {
    Hello { protocol: u32, token: String, commit: String, dirty: bool,
            capacity: (u16, u16, u8), triple: String,
            cores: u16, ram_mb: u32, cached_models: Vec<ModelId>, name: String },
    TrajectoryDone { task: TaskId, seed: u64, zstd_json: Vec<u8> },
    EvalGameDone   { task: TaskId, game: u32, line: EvalGameLine },
    TaskFailed     { task: TaskId, seed_or_game: u64, error: String },
    Heartbeat      { games_in_flight: u16 },
}
// hub -> worker
enum ToWorker {
    Welcome { parallel_games: u16 },
    Reject  { reason: RejectReason },              // BadToken | Protocol | Commit{expected} | Capacity | Dirty
    Update  { url: String, blake3: [u8; 32] },
    Model   { id: ModelId, onnx: Vec<u8> },
    Task    (Task),
    Drain,                                         // finish in-flight, no new tasks (hub shutting down)
}
enum Task {
    Generate { id: TaskId, seeds: Vec<u64>, mode: GenMode, mcts_iters: u32,
               model: Option<ModelId>, evaluator: Evaluator, board_sizes: Option<BoardSampler> },
    Eval     { id: TaskId, games: Vec<u32>, rung: Rung, candidate: ModelId,
               opponent: Opponent, mcts_iters: u32, seed: u64 },
}
```

`EvalGameLine` mirrors the hand-written JSON in `eval.rs` (`rung, game, seed, candidate_team,
home_score, away_score, kicking_first_half, finished`); the hub appends it to
`eval.games.jsonl` and folds it into `LadderRow` for `report.json`. `Report` gains
`workers: Vec<{name, games, triple}>` in `extra`; nothing that reads it today breaks.

Compatibility check order in `Hello`: protocol, token, commit (unless `--allow-commit-mismatch`),
dirty (hub refuses dirty workers unless it is dirty itself), capacity. A worker dropped for
`Commit` gets `Update` first if a binary for its triple exists.

## Worker loop

```
connect -> Hello -> Welcome{parallel_games}
spawn parallel_games std threads (games are sync; MctsBot::get_action uses thread::scope)
each thread: pull a seed from the current Task, play via botbowl-play, send result frame
tokio side: one reader (hub frames -> task queue / model cache), one writer (mpsc -> ws)
on disconnect: exponential backoff reconnect (5 s .. 5 min); in-flight games finish and are
  re-sent on reconnect (the hub dedupes)
on Update: download, verify blake3, atomic rename over own exe, exec
```

Model cache on disk keyed by id, so a laptop that reconnects Monday after a weekend of
generations only downloads the new champion.

## Loop integration (`scripts/train_loop.sh`)

- Startup: `botbowl-hub serve` is started once (systemd unit or the loop starts it if not
  running), the local worker likewise with `--nn-server $NN_SOCKET --parallel-games $PARALLEL_GAMES`.
- Generate phase becomes `botbowl-hub job generate ... --wait`; the `nn_server_start/stop`
  bracket stays around it because the *local* worker uses the sidecar and the trainer needs
  the card afterwards. When the job completes the hub simply has no tasks to hand out, so
  remote workers idle until the next job.
- Eval phase becomes `botbowl-hub job eval ... --wait`. The anchor model is shipped by id like
  any other.
- Everything after (`td_rate.py`, `eval_summary.py`, `verdict`, `status.md` lines) is untouched.
- The bootstrap phase (`BOOTSTRAP_GAMES_PER_SHARD`, heuristic evaluator) is just a generate
  job with `evaluator: Heuristic, model: None`.

## Phases

0. **Extract `botbowl-play`.** Move game-playing and bot construction out of `botbowl-ui`
   into a lib; `dataset` and `eval` call it. Behavioural no-op: `cargo test --workspace`,
   the `#[ignore]`d benchmarks, and a 20-game `dataset` diff (same seeds, same set of lines)
   pin it. Also move the eval per-game line into a serde struct.
1. **Proto + hub + worker, eval jobs only.** Smallest surface, biggest win. Hub writes
   `eval.games.jsonl` and `report.json`; loop's eval phase switches over. Integration test:
   hub + 2 workers in-process on 8x3, tract, 4 games, assert the report equals the
   single-process `eval` run on the same seeds modulo line order.
2. **Generate jobs.** Round-robin shard writing, `.generated`, seeds requeue on worker loss
   (test: kill a worker mid-task, assert final seed set is exactly the requested set with no
   duplicates). Loop's generate phase switches over. Bootstrap mode included.
3. **Token auth, heartbeats, requeue timeouts, `Drain`, hub restart recovery** from
   `job.json` + output files. Hub status page `GET /` (plain text: workers, in-flight, job
   progress, games/min per worker) for watching a weekend run from a phone.
4. **Self-update and binary serving.** `/worker/<triple>`, `/install.sh`, `Update` handling,
   blake3 verification, `scripts/dist_build.sh` running `cargo zigbuild --release` for
   `x86_64-unknown-linux-musl`, `aarch64-apple-darwin`, and (when needed)
   `x86_64-pc-windows-gnu`. macOS cross-builds from Linux need the SDK; simplest is to build
   the darwin binary *on* the laptop once per commit and `scp` it into `dist/`, or build
   natively on the desktop if it is a Mac. Note which in the script.
5. **TLS with pinned self-signed cert** (`axum-server` + rustls; worker pins the fingerprint
   baked at build time from `dist/hub.crt`). Optional; do it when the hub port is exposed to
   the internet for more than a weekend at a time.

Phases 0-2 are the deliverable. 3 is needed before the first unattended weekend. 4-5 are
convenience and hardening.

## Cost model and expected gain

Per-game cost is unchanged (~5.9 s aggregate per game on the desktop with the sidecar;
~3-4x that on a tract-only core). A 16-core Linux server with 32 GB adds ~16 tract streams
≈ 4-5 sidecar-equivalent streams; an M-series laptop with 16 GB adds ~8-12 streams overnight.
Rough expectation: generate 256 → ~100 min, eval 287 → ~90 min with one server and one laptop,
and eval game counts can go up to the plan 032 sizes (600+) without lengthening the cycle.
Bandwidth: ~30 KB/game up from workers, 2 MB/champion down. Latency is irrelevant; nothing is
per-leaf across the wire.

## Non-goals

- Distributed *training*: the 1060 is not the bottleneck and `train` is 35 min.
- Per-leaf remote inference (the `nn_server.py` model over the internet): 247 µs/sample locally
  becomes tens of ms; a hard no.
- Browser/wasm workers (decision 3).
- Fairness or scheduling across multiple concurrent runs; one run per hub.

## Open questions

- **Dirty-tree workers during development.** Decision 5 refuses them unless the hub is dirty
  too. For iterating on the worker itself that is right; confirm it does not get in the way
  of "laptop has a local uncommitted fix" (answer should be: commit it).
- **Windows path.** `x86_64-pc-windows-gnu` via zigbuild is untested for tract; if it fails,
  `cargo-xwin` for msvc. Defer until a Windows box exists.
- **Whether `job --wait` should print live progress** to the loop log or just the final line.
  Leaning final line only; the status page is the live view.
