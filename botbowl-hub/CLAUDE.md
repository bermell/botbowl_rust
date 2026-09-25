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
# any machine, same commit, same BOARD_SIZE_* build *and* the same BOARD_SIZE_* environment:
botbowl-worker --hub ws://<office-ip>:7777/ws --token-file hub.token        # sizes itself from cores/RAM
botbowl-worker ... --parallel-games 4 --nn-server /tmp/nn.sock              # the local worker, with the GPU sidecar
botbowl-worker ... --mem-floor-mb 2048                                      # raise the headroom reserve (default 1024 MB)
botbowl-worker ... --reconnect-max-secs 30                                  # longest gap between dial attempts (the default)
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

- **Compatibility is exact commit + clean tree + board capacity + active board**
  (`Hello` → `Reject{reason}`). Dirty workers are refused unless the hub is dirty too — commit,
  then rebuild — and *nothing* relaxes that, because a dirty tree has no name for anything to
  assert about. `PROTOCOL_VERSION` in proto is bumped on any frame change; `postcard` encoding, so
  field order matters. **v4** (plan 043) added `SearchConfig.config` — a named `MctsConfig`
  preset — and `GenerateConfig.config_name`; both ride inside the re-exported `botbowl-play`
  types, so no new frame was needed. **v5** added `BuildInfo.env_board` and `RejectReason::Board`.
- **The active board is checked, not just the capacity.** `capacity` is the compile-time ceiling;
  `BoardDims::from_env()` is what a task that names no board of its own actually plays. Two boxes
  built from one commit with different `BOARD_SIZE_*` would otherwise contribute different games
  to one corpus or one rung, silently — plan 042 left that open and the comment claiming the
  capacity check covered it was wrong. It is in `Hello` now.
- **Two ways past a commit mismatch, and they are not interchangeable.**
  `--allow-commit-mismatch` accepts *any* commit; it is for developing the worker itself.
  `--allowed-commits <file>` (default `hub-allowed-commits.toml`, absent = exact match) is the one
  to use while running a programme: an **untracked** TOML that names the hub commit it applies to
  plus the worker commits admitted alongside it.
  ```toml
  hub_commit = "5f818d2"                  # must be the commit the hub runs, or the file is inert
  allow = ["db7fc9c", "5a60d25"]          # docs/web-only commits, checked by hand
  ```
  The design is in `botbowl-hub/src/allowlist.rs`; `hub-allowed-commits.toml.example` is the
  template. Three properties are load-bearing, do not weaken them:
  1. **It self-invalidates.** Keying on `hub_commit` means the file is stale the moment anyone
     commits, and a stale file admits *nobody* — the rule reverts to exact match rather than
     staying permissive. Updating it is a deliberate act after the commit, before the hub starts.
     A malformed or unreadable file is inert the same way (`deny_unknown_fields`, so a typo'd key
     is an error, not a silently empty list).
  2. **It is re-read on every handshake.** Editing it admits workers that are already sitting in
     their reconnect loop, with no hub restart. That pairs with the worker's 30 s retry: commit,
     update the file, and the fleet rejoins on its own.
  3. **It is an assertion, and it is audited.** Provenance comes from the *hub* (`state.rs` stamps
     `git_commit` into `report.json` and every corpus label), so an admitted worker's games claim
     the hub's commit. That is the claim being made — "these commits are the same game" — and the
     only thing that makes it checkable afterwards is the `[hub] ... admitted by ...` line, so
     keep it. Justify an entry with an empty
     `git diff --stat <theirs>..<hub> -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play`.
- **The worker retries; it does not give up.** Backoff starts at 5 s, doubles to
  `--reconnect-max-secs` (30 s), and **resets after any connection that worked** — the old code
  crept to a 5-minute cap and stayed there for the rest of the week. A `Reject` is retried too,
  logged once per distinct reason, because the hub is where a commit mismatch gets resolved and a
  worker that exited cannot benefit from that. `BadToken` is the one fatal rejection: no action on
  the hub makes a wrong secret right.
- **Every search knob resolves on the submitter, never on the worker.** Bot presets
  (`--bot-config` / `--vs-config`, the same flags as `botbowl-ui`) always did; the rest now does
  too, via `SearchConfig::pinned_to_env()`, which the hub calls on both seats of an eval job and
  on a generate job. It fills `config` with a complete `MctsConfig` resolved from the *hub's*
  environment whenever no preset named one. Without it, `config: None` meant
  `MctsConfig::from_env()` **on whichever worker picked the task up** — and `tree_reuse`,
  `virtual_loss`, `tie_break`, `memory_mode`, `horizon` and the PUCT range floor have no per-knob
  field in `SearchConfig` at all, so a stray `BLOOD_MCTS_*` on one helper box changed that box's
  search, appeared nowhere in `report.json`, and was indistinguishable from the arm under test.
  The worker warns at startup about any `BLOOD_MCTS_*` in its environment, all of which are now
  inert. `stats` / `leaf_stats` / `debug_root` are deliberately forced off in the pinned config:
  they are properties of a terminal someone is watching, and `stats` walks the whole DAG after
  every search.
  The preset's *name* still travels separately from its knobs —
  `EvalJobRequest.candidate_config` / `opponent_config` for the report, `GenerateConfig.config_name`
  for corpus provenance — because `SearchConfig` has to stay `Copy`. Generate also keeps its
  per-knob fields `None` on purpose: the corpus label reads them, so a `Some` there would rewrite
  every label.
- **The hub is the place to run a search experiment** (plan 045 and anything like it): candidate
  and opponent are independent `BotSpec`s on the wire, so `--mcts-iters X --opponent-iters 1000
  --board-sizes 14x7/4,16x9/6` is one job, and every arm is described by the submission rather
  than by the boxes that happened to be connected. Two things the rung label still will not show:
  an **iteration** asymmetry (the label only carries puct/horizon/backup/fpu differences, and
  `report.mcts_iters` is the candidate's alone — put the budget in the output file name), and a
  second custom opponent (`--vs-evaluator` gives exactly one per job, so one job per arm).
- **Search telemetry rides on `EvalGameLine`** and folds through `LadderRow::record`, so the hub's
  `report.json` carries the identical `telemetry` block the single-process driver writes. There is
  no distributed `--trace-reuse`: a per-decision trace is a local diagnostic.
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
  never from a worker's environment — as is every other search knob, via `pinned_to_env` (see the
  submitter-resolves-everything invariant above). A helper box's `BLOOD_MCTS_*` reaches nothing.
- **The board travels with the job, not the environment (plan 042, protocol v3).** A generate
  task's `GenerateConfig.board_sizes` draws each game's board from its seed; an eval rung carries
  `RungReq.board` → `Task::Eval.board`, and a multi-size ladder names rungs `opponent@14x7/4`
  (`botbowl_play::eval::rung_name`), one `LadderRow` per (opponent, board), `board_env` listing
  the boards. `job generate` takes `dataset`'s `--board-sizes` / `--size-*` flags, `job eval`
  takes `--board-sizes`. The capacity check still applies — a worker must be *built* large enough
  for every board a job may draw — and since v5 the *active* board is checked too, so a job that
  names no board means one board fleet-wide rather than whatever each box's `BOARD_SIZE_*` says.
- **Control API and workers share one bearer token** (`hub.token`, random on first start).
  No TLS, and none is planned as plan 041 phase 5 designed it (a pinned self-signed cert existed
  mostly to protect the served worker binary of phase 4, which does not exist). Put the hub behind
  WireGuard/Tailscale or an SSH reverse tunnel and let workers dial `ws://127.0.0.1:…`; that is
  mutual auth plus encryption for zero code, and it is what a hub reachable from outside the LAN
  should sit behind. If the port must be public instead, terminate TLS in a reverse proxy — the
  worker then needs a TLS feature on `tokio-tungstenite` to dial `wss://`, and `http.rs` (a
  deliberately tiny `http://`-only client for the *local* control API) can stay as it is.
- **A worker's `--parallel-games` is a ceiling, not a promise (`mem_governor.rs`).** Each game
  thread predicts its next game's tree cost in *cost units* — playable cells scaled by the task's
  iteration budget (`cost_units`), since the ~4.5 MB/cell seed was calibrated at 1000 iters and a
  4000-iteration experiment would otherwise under-predict fourfold until the EWMA caught up — and
  blocks rather than start a game likely
  to push the box under `--mem-floor-mb` (default 1024) of headroom. This exists because the
  board-size curriculum (plan 042) varies tree memory several-fold within one generation while
  the concurrency knobs (`GEN_PARALLEL_GAMES` etc.) stay fixed — a fixed pool size tuned against
  one board size silently overcommits once the curriculum grows past it (2026-09-24: `systemd-oomd`
  killed a training box's entire user session at 54% sustained memory pressure, mid-generation, on
  boards several times 14x7's cell count). The check is per-worker and self-correcting, not a
  static retune.

## Tests

`cargo test -p botbowl-hub`: hub + two in-process workers reproduce a single-process eval
exactly (search-free bots so it is exact on any tier); a generate job writes each shard's seed
set exactly once with `dataset`'s provenance labels, truncates or appends as asked; a worker
that vanishes mid-task loses nothing (eval and generate); each incompatibility is rejected with
its reason; an allowlisted commit connects while an unlisted one and a stale file do not.
For MCTS/NN paths, run real binaries at 14x7 against a current-schema net
(`scripts/make_random_net.py` makes one) — search output is not reproducible across processes,
so compare seed sets, labels and counts, not lines.

## Not yet (plan 041 phases 3-5)

**Phase 3, still open:** hub restart recovery (`job.json`) — in-memory job state dies with the
daemon, and there is no `job cancel`, so an orphaned job has to be cleared by restarting the hub.
`Drain` is half done: the frame exists and the worker implements it fully, but the hub never sends
it (Ctrl-C just drops the server), so a restart always costs the in-flight games. Both are worth
more now that the worker reconnects indefinitely — a hub restart is a routine event, not an
incident.

**Phase 4 (serving the worker binary, `Update`, self-update):** nothing exists. The per-commit
allowlist is the cheap substitute for its main benefit — not having to rebuild every helper box
for a commit that cannot change a game.

**Phase 5 (TLS with a pinned self-signed cert): not planned as designed.** Use a tunnel; see the
bearer-token invariant above for why and for what to do if the port must be public.
