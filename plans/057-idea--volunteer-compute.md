# Plan 057 — Volunteer compute: crowd-sourced generation from the Blood Bowl community

**Status:** Idea, 2026-10-06. Not started; nothing here is live work until funding or a
community pilot makes it so. Prerequisite step 0 (deterministic single-threaded search) is
worth doing on its own and is the only part that touches the current loop.

## Why

Expert-human strength and beyond needs far more generation compute than one training box
plus a few helper machines. The Blood Bowl community is large and interested; if the
motivation problem is solved, an average gamer giving one CPU core is the compute. This plan
is the technical side only: how volunteers connect without a toolchain, how bogus data is kept
out of the corpus, and how everyone stays on the current code. Motivation (credit, leaderboards,
a visible bot to play against) is out of scope except where the stack must support it.

## What already exists (plan 041)

The hub/worker split is the right shape and most of it carries over unchanged:

- Workers dial **outbound** over a websocket, so NAT and home routers are not a problem.
- Reconnect loop with backoff that resets after a good connection; `Reject` is retried, so a
  worker parked on a rejection self-heals when the hub changes.
- One-game tasks, one result per game, trajectories zstd-compressed on the wire. One game is
  the right granularity for a single core.
- Nets are bytes identified by BLAKE3, shipped once, cached at `~/.cache/botbowl/models`,
  rehash-verified on startup. Names travel separately (protocol v14).
- Every search knob resolves **on the hub** (`SearchConfig::pinned_to_env`); a worker's
  environment cannot change the arm under test.
- Inference is CPU ONNX through tract: no GPU story needed on the volunteer side.
- Search telemetry (iterations, timings, reuse counters) rides on every `EvalGameLine` and is
  stamped into each trajectory's `meta.extra`.

What does not carry over: one shared machine token, exact-commit + clean-tree compatibility,
the `BOARD_SIZE_*` build check, the hub writing shards straight into the run directory on the
training box, and trust-by-default for everything a worker returns.

## Design

### 1. Topology

```
volunteers (native binary / browser tab)
        │ wss, per-account token
        ▼
  public coordinator  ── accounts, queue, quarantine, verification, reputation
        │ accepted shards (pull)
        ▼
  training box        ── train_loop.sh unchanged: reads shard$K.jsonl as today
```

- The **coordinator** is `botbowl-hub` grown a layer: accounts, a quarantine store for
  unverified results, the verification queue and a reputation table. It sits behind a reverse
  proxy for TLS. It does not need to be the training box.
- The **training box** pulls accepted shards for a generation. `train_loop.sh` is unchanged;
  only the source of `gen$N/shard$K.jsonl` moves.
- Per-account tokens replace `~/.config/botbowl/hub.token`. A signup page (Discord or GitHub
  login) issues them. Identities are required for reputation anyway.
- Build the worker at the largest board capacity and make the active board a per-task
  parameter. Plan 042 already plays any size within capacity; `Hello`'s `env_board` check
  becomes a release check (section 4). `RejectReason::Board` stays for the capacity ceiling.

### 2. Zero-install client

Two clients, same protocol:

- **Native binary.** Static per-target builds (Windows, macOS, Linux) from CI with
  `cargo-dist`. The actual cost is signing: an Apple developer account for notarization and a
  Windows code-signing certificate, or volunteers see Gatekeeper and SmartScreen warnings that
  kill adoption. Budget both. Reproducible builds so anyone can verify a release matches its
  tag.
- **Browser tab.** `botbowl-web-proto` already targets wasm32; engine, mcts and tract can too.
  A Web Worker running one game at a time is a volunteer node with one click, no install, no
  signing, and updates are a page refresh. Expect roughly 1.5 to 3x slower than native; measure
  before promising anything. Ship this for casual contributors and the binary for the committed.
- **BOINC** is the established alternative and gives updates, credit and replication for free,
  at the price of wrapping the binary in its framework and asking volunteers to install its
  client. Evaluate once; probably not worth the coupling given the browser option.

The worker keeps sizing itself (cores, `--mem-floor-mb`) but defaults to one game thread; the
UI exposes "cores to donate" and nothing else.

### 3. Trust: keeping bogus data out

A fake client cannot be prevented, so the design makes cheating unprofitable and detection
decisive. Layers, cheapest first:

1. **Legality replay on 100% of incoming games.** The coordinator has the engine. Replaying a
   trajectory's actions with its recorded dice is orders of magnitude cheaper than the search
   that produced it, so every game is checked: every action legal in its state, every state
   reached from the previous one, the outcome consistent. Outcome and value labels are
   **recomputed from the replay**, never taken from the worker. MC-averaged labels (plan 056)
   are policy-only playouts and can be crowdsourced as their own task kind or run on the
   training box as now. This kills fabricated trajectories outright.
2. **Deterministic search so spot checks are bit-exact.** Today seeds do not pin games because
   `recon_mcts`'s HashMap iteration order varies per process. Single-threaded volunteers make
   determinism reachable: a seeded hasher, and any remaining process-dependent order removed.
   Then a check is "replay this seed with this net and config, compare the trajectory hash",
   which is decisive rather than plausible, and it also catches the subtle cheat of running
   fewer descents than asked (visit counts differ). Policy targets are the one thing legality
   replay cannot verify; this is what verifies them.
3. **Reputation, not a flat sampling rate.** A new account has a high fraction of its games
   replayed (say half) and all of its results quarantined until the checks pass. The fraction
   decays with clean history toward a floor (a few percent). One failed check bans the account
   and **drops every sample it ever contributed**, so the worker id is stamped into each
   trajectory's provenance alongside the telemetry. Poisoning N games then requires passing
   checks on all N, and one failure costs everything. The expected payoff of cheating is
   negative at any sampling rate above zero.
4. **Crowdsourced replays as redundancy.** A check task is the original game's seed, net and
   config sent to a random second worker from a different account; the coordinator compares
   hashes. Disagreements go to trusted machines (ours) to break the tie, and the loser is the
   one that disagrees with the trusted replay. Collusion needs many accounts that each built
   reputation; rate-limited signup makes that slow. Checks consume roughly the sampled fraction
   of compute again, so a 5% floor costs 5%.
5. **Eval stays on trusted workers** until reputation thresholds exist and have been watched.
   Poisoned eval decides promotion, which is worse than noise in a corpus. Drives (plan 051)
   are cheap enough to keep in-house for now.

Cheap plausibility on top, free because the telemetry already exists: claimed iterations
versus wall time, reuse counters outside the known range, temperature-inconsistent policy
targets (the played action outside the target's support).

Trusted-worker allowlist: our own machines and known helpers skip quarantine, exactly as
today, and are the tie-breakers in (4).

### 4. Updates

- The handshake carries a **release version and build hash** instead of a commit. The hub
  advertises `min_release` and `latest_release`; below minimum is `Reject{Outdated{url}}`.
  The worker's existing retry loop means a rejected worker rejoins by itself once updated.
- **Self-update:** the worker downloads the release named in the reject (or in a
  `ToWorker::Update` nudge), verifies an **ed25519 signature against a key compiled into the
  binary** (so a compromised CDN or GitHub release cannot push code), swaps itself and
  restarts. The `self_update` crate handles the rename dance Windows needs for a running exe.
- **Staged rollout:** `hub-allowed-commits.toml` generalizes to an allowed-releases file with
  the same three properties (self-invalidating, re-read per handshake, audited). Serve
  `latest_release` to a slice of accounts first; widen when their check-failure rate matches
  the fleet.
- Browser client: refresh. The page version-checks against the hub on connect.
- Nets and search configs already resolve on the hub and travel hashed; nothing changes.
- Protocol: `PROTOCOL_VERSION` stays the hard gate. A protocol bump means `min_release` moves
  with it, so the two are released together.

## Steps

0. **Deterministic single-threaded search** in `recon_mcts` + `botbowl-mcts`: seeded hasher,
   audit every iteration-order-dependent choice (tie-breaks, registry probes), and a test that
   two single-threaded searches from one seed produce identical trajectories. Useful on its
   own for reproducing crashes. Do this first.
1. **Legality replay** as a `botbowl-play` function (`verify_trajectory`), used by the hub on
   ingest and by `botbowl-ui` as a corpus check. Outcome and value labels recomputed here.
2. **Accounts and quarantine** in the hub: per-account tokens, provenance stamp, quarantine
   store, reputation table, sampled replay queue, ban-and-drop. Trusted allowlist for current
   helpers.
3. **Release handshake + self-update** in proto and worker; `cargo-dist` CI with signed
   artifacts; allowed-releases file.
4. **Browser worker**: wasm build of engine + mcts + tract behind the same protocol; measure
   the slowdown.
5. **Pilot** with a handful of community members on the native binary before any public
   launch; watch check-failure rates, reconnect behaviour and support load.

## Open questions

- Throughput per volunteer core at the loop's budget (500 to 1000 descents, 16x9): measure, it
  decides whether one-game tasks need batching or splitting.
- Whether the coordinator should be a separate binary from the hub or the hub with a
  `--public` mode. Leaning the latter: one codebase, the local loop keeps working unchanged.
- Credit and motivation: games accepted (post-verification), not games submitted, is the only
  safe number to show on a leaderboard.
- Data volume on the coordinator: a generation of trajectories from thousands of cores is
  large; object storage rather than a directory of `shard$K.jsonl`, with the training box
  assembling shards on pull.
