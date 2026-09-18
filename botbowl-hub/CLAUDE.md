# botbowl-hub, botbowl-worker, botbowl-hub-proto

Plan 041: distribute the loop's game-playing phases over machines that dial in. The **hub**
(on the training box) owns a job queue and writes results in the exact layout `train_loop.sh`
consumes; **workers** connect outbound over a websocket, receive small game batches, and stream
one result per game. The **proto** crate is the wire vocabulary shared by both. The game code
itself is `botbowl-play`; nothing here plays a game any other way.

## Run it

```sh
# training box (train_loop.sh does this itself; by hand for remote workers to join early):
botbowl-hub serve --bind 0.0.0.0:7777 --token-file runs/<run>/hub.token
# any machine, same commit, same BOARD_SIZE_* build:
botbowl-worker --hub ws://<office-ip>:7777/ws --token-file hub.token        # sizes itself from cores/RAM
botbowl-worker ... --parallel-games 4 --nn-server /tmp/nn.sock              # the local worker, with the GPU sidecar
# submit (flag-compatible with `botbowl-ui eval`'s ladder half):
botbowl-hub job eval --evaluator nn --model X.onnx --vs-evaluator nn --vs-model anchor.onnx \
    --vs-games 40 --skip-fixed-rungs --per-game-out eval.games.jsonl --out report.json --wait
# generate (flag-compatible with `botbowl-ui dataset`; one job = all shards of a generation):
botbowl-hub job generate --mode random-start --games 600 --mcts-iters 1000 --evaluator nn --model X.onnx \
    --seed-base 22000000 --shard-seed-stride 100000 --shards "0 1 2 3 4 5 6 7" --heuristic-shards "" \
    --truncate --out-dir runs/<run>/gen12 --wait       # writes gen12/shard$K.jsonl, shard K seeded at base + K*stride
botbowl-hub status            # JSON;  curl http://hub:7777/  is the plain-text page
```

## Invariants

- **Compatibility is exact commit + clean tree + board capacity** (`Hello` → `Reject{reason}`).
  `--allow-commit-mismatch` on the hub relaxes only the commit check. Dirty workers are refused
  unless the hub is dirty too — commit, then rebuild. `PROTOCOL_VERSION` in proto is bumped on
  any frame change; `postcard` encoding, so field order matters.
- **Models are bytes, identified by BLAKE3.** `ModelId::of(onnx)`. The hub reads a path once at
  submit and ships bytes only to workers whose `Hello.cached_models` lack the id. Worker cache:
  `~/.cache/botbowl/models/<hex>.onnx`, verified by rehash on startup.
- **A worker probes every net once before any game uses it** (`ModelStore::get`): a
  schema-mismatched ONNX panics inside tract, and inside `MctsBot` that poisons tree locks and
  aborts the process on unwind. The probe turns it into `TaskFailed`; three failures of one game
  fail the job with the message. Keep the probe.
- **Games never run on the tokio runtime.** Worker game threads are `std` threads with
  `GAME_STACK_SIZE`; `MctsBot` spawns its own scoped threads inside them.
- **Results are idempotent.** Hub dedupes on `(job, unit, game)` — a unit is a ladder rung or
  a corpus shard; a vanished worker's in-flight games are requeued at the front; a slow worker
  that reappears cannot double count. The worker's result channel outlives its socket, so
  nothing finished is lost on a reconnect.
- **Liveness is the heartbeat, not the socket.** A worker whose *process* dies closes its
  socket and `ws.rs` requeues immediately; a worker whose *machine* leaves — a slept laptop, a
  dropped VPN — leaves an ESTABLISHED socket the hub cannot tell from a healthy one. Workers
  heartbeat every 30 s and `reap_loop` drops anything silent for `--worker-timeout` (120 s)
  and requeues its games. Without it a job strands on its last few games with every live
  worker idle: gen10 generate sat at 4791/4800 for three hours on 2026-09-18. Worker ids are
  never reused, so a late result from a reaped worker is deduped like any other.
- **Output equals `botbowl-ui eval`'s.** `eval.games.jsonl` lines are `EvalGameLine`
  (serde, field order is the format), `report.json` is `botbowl_play::eval::Report` with
  `lectures: []` — the hub never runs the lecture battery. Rung labels are built by the same
  code as the CLI's, so `eval_summary.py`/`paired_summary.py` keep matching.
- **Output equals `botbowl-ui dataset`'s.** A generate task carries the full `GenerateConfig`
  (serde) and a `seed_base`; game `g` is seed `seed_base + g`, and shard `K` of a job is seeded
  at `--seed-base + K * --shard-seed-stride`, so `gen12/shard3.jsonl` holds exactly the seeds
  the old per-shard process would have. The worker sends each trajectory as its
  `serde_json::to_vec` line, zstd level 3 (~575 KB -> ~30 KB); the hub decompresses and appends
  the bytes verbatim plus `\n`, i.e. `DatasetWriter` format, so `prepare` is untouched.
  `cfg.model` is the *path string as typed* (it is the provenance label); the bytes travel by
  `ModelId`. The label's backup rule is resolved from the hub's `BLOOD_MCTS_BACKUP` at submit,
  never from a worker's environment. Other `BLOOD_MCTS_*` knobs the config leaves `None` are
  still read on the worker — keep helper boxes' environments clean.
- **Control API and workers share one bearer token** (`hub.token`, random on first start).
  No TLS yet (plan 041 phase 5); `http.rs` is a deliberately tiny client that will go with it.

## Tests

`cargo test -p botbowl-hub`: hub + two in-process workers reproduce a single-process eval
exactly (search-free bots so it is exact on any tier); a generate job writes each shard's seed
set exactly once with `dataset`'s provenance labels, truncates or appends as asked; a worker
that vanishes mid-task loses nothing (eval and generate); each incompatibility is rejected with
its reason. For MCTS/NN paths, run real binaries at 14x7 against a current-schema net
(`scripts/make_random_net.py` makes one) — search output is not reproducible across processes,
so compare seed sets, labels and counts, not lines.

## Not yet (plan 041 phases 3-5)

Hub restart recovery (`job.json`) — in-memory job state still dies with the daemon, and there
is no `job cancel`, so an orphaned job has to be cleared by restarting the hub. `Drain` on
shutdown, serving the worker binary + self-update, TLS with a pinned cert.
