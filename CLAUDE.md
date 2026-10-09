# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. **Each crate has its own `CLAUDE.md` with architecture detail** — it loads automatically when you work on files in that crate; read it before making non-trivial changes there.

## Repository layout

One git repo containing the botbowl Cargo workspace plus the nested `recon_mcts/` library (folded in via history-preserving subtree merge).

- Botbowl workspace (`Cargo.toml` at repo root) — Blood Bowl 2020 engine + tooling. Member crates sharing one `Cargo.lock` and one `target/` (`botbowl-data` — trajectory schema + JSONL I/O + the git provenance stamp — and `botbowl-nn` are members too; `botbowl-nn` carries its own `CLAUDE.md`):
  - `botbowl-engine/` — pure rules library, procedure-stack state machine. No dependency on the other crates. Board/team size is build-time configurable via env vars (see its CLAUDE.md).
  - `botbowl-curriculum/` — training scenarios (`Lecture` trait, `run_trials`). Depends on `botbowl-engine`.
  - `botbowl-mcts/` — `BloodBowlDynamics` + `MctsBot`, the adapter onto `recon_mcts`. Depends on `botbowl-engine` + `recon_mcts`.
  - `botbowl-play/` — "play one game, return its record": the process-agnostic core under `botbowl-ui dataset`/`eval` (trajectory generation, ladder games, `EvalGameLine`/`LadderRow`/`Report`, bot construction). No files, threads or CLI in it; plan 041's hub/worker reuse it verbatim. Depends on engine, curriculum, mcts, nn, data.
  - `botbowl-hub/`, `botbowl-worker/`, `botbowl-hub-proto/` — distributed generation and eval (plan 041): the hub on the training box queues game batches, workers on any machine dial in over a websocket and stream results back (trajectories zstd-compressed); the hub writes the same files (`shard$K.jsonl`, `eval.games.jsonl`, `report.json`) the local phases wrote. Shared `CLAUDE.md` in `botbowl-hub/`. Depend on `botbowl-play`.
  - `botbowl-ui/` — `ratatui` terminal frontend with `live` / `replay` / `snapshot` / `curriculum` / `placement` subcommands, the headless `dataset` / `eval` shells over `botbowl-play`, the loop stage `mc-label`, and research probes (`convergence`, `positions`, `override-audit`, `value-bench`). Depends on engine, curriculum, mcts, nn, data and play.
  - `botbowl-web/{proto,server,client}/` — human-vs-bot or bot-vs-bot play in a browser (MCTS always on a net), full games or random-start drives, with custom teams, a decision log and the search behind every bot move shown next to the board (plan 034). Also served by the hub at `/play/`. `proto` is engine-free and compiles to wasm32; `server` owns the `GameState` and the bots; `client` is a Leptos CSR app built with `trunk`. Has its own `CLAUDE.md`.
- `recon_mcts/` — generic **re**combining, **con**current MCTS library (safe std-only Rust). A **nested, separate Cargo workspace**, deliberately in the botbowl workspace's `exclude` list — don't merge it in (its `tests/nim/` member compiles with `--features test_internals` by default). Has its own `CLAUDE.md`. No dependency on the botbowl crates.

## Registry: data, nets, experiments, matches

`registry/` is the record of what exists and how it was made (the user, 2026-10-08):
- `registry/DATA.md` — every corpus and frozen benchmark set: commit, generator net, settings, seeds, size, caveats;
- `registry/NETS.md` — every trained net: parent, data window, recipe, commit, and **every single-net benchmark number** (value bench, absorption, net check, val);
- `registry/EXPERIMENTS.md` — question, reproduction (commit, script, settings), result, conclusion;
- `registry/MATCHES.md` — every head-to-head result between two bots (drives, policy-only, SPRT).

**Update them as part of the work, not afterwards:** a new corpus, net, experiment or match result gets
its row when it is produced, with the commit and every setting needed to reproduce it. Plans keep the
narrative; the registry keeps the facts.

**Browse it at `http://<hub>/registry/`** (`botbowl-hub serve`, `--registry-dir`, default
`<repo>/registry`): every `.md` file in the directory, rendered with its tables, read from disk on each
request, so an edit shows on the next reload. The markdown files are the source — edit them directly.
Keep to GitHub-flavoured markdown: one `# ` title, an opening paragraph (the index shows it), `## `/`### `
sections (they get ids and a contents list), pipe tables with a header row, and file-relative links
(`[NETS.md](NETS.md)`, `NETS.md#exp069`). A new file in `registry/` shows up in the index by itself.


- `plans/001-grand-plan.md` — strategic roadmap (AlphaZero-style MCTS via curriculum learning → scripted baseline → heuristic/rollout/NN-guided MCTS → self-play). Read it before proposing architecture changes that span the engine and `recon_mcts`.
- `plans/NNN-idea--*.md` / `plans/NNN-plan--*.md` — designs not yet started or in-flight. `plans/completed/` — closed-out plans with **Status:** headers; historical context, not live work.
- **Experiment log:** `plans/032-plan--ranked-experiment-queue.md` (ranked longer experiments and open questions) — new results go there, not into completed plans. `plans/031-plan--audit-diagnostics.md` (cheap diagnostics) is finished (all ten run 2026-09-07); read it for history only.
- **Search instrumentation + bot presets:** `plans/043-plan--search-instrumentation-and-bot-configs.md` — tree-reuse and recombination counters that reach `report.json`, a trajectory's provenance and the web debug drawer, plus `cfgs/*.toml` bot presets (`--bot-config` / `--vs-config`) so the same net can play itself under two configurations.
- **State-hash discrimination:** `plans/044-plan--state-hash-discrimination.md` — `GameState::hash` now walks the procedure stack and `GameInfo` whole, taking colliding states from 36.8% to 0% and wasted state comparisons from 4.07 to 0.12 per registry probe.
- **Per-player kickoff setup:** `plans/047-plan--per-player-setup.md` — `Setup` asks one `PlacePlayer`/`BenchPlayer` decision per player (reserves staged in the own endzone, fixed queue, the mask guarantees legality); formations are planners (`auto_setup`); NN schema v8 (`bbnn.migrate` permutes a v7 policy head; v9 appends `UseSkill`/`DontUseSkill` for optional skills); MCTS `setup`/`opponent_setup`/`setup_formation` preset knobs, an in-tree opponent model, scripted one-hot teacher samples; `--next-drive` follows a scored drive into the setup it causes and writes it as a second record.
- **Board-size curriculum:** `plans/042-plan--board-size-curriculum.md` — mixed-size generation (`--board-sizes` / `--size-centre …` on `dataset` and `job generate`, per-board eval rungs via `eval --board-sizes`), schema v7, the trainer's multi-dims loader and `train_loop.sh`'s `SIZE_MODE`. Experiments E0–E5 there are the next thing to run.
- **Fast bot ranking:** `plans/051-plan--fast-bot-ranking.md` — SPRT with pentanomial pair scoring and a score margin on the eval ladder, a validation harness of net pairs with gold results, and a paired contested-drive rung kind; the plan adopts proxies only after the harness passes them against full-game ground truth.
  **Current phase: drives are the metric, alone — this rule overrides the plan's full-game validation step.** Judge experiments and the loop on paired contested drives (SPRT). Do not add, run or propose full-game confirmation, P1 checks or anchor matches; the user says when full games are needed.
- **Gumbel root search:** `plans/053-plan--gumbel-root-search.md` — sequential halving over the top-m root moves (`MctsConfig.gumbel_m`, `cfgs/gumbel16_iters.toml`), PUCT below the root; the fix candidate for 16x9's wide fans, measured on drives.
- **Search must beat policy:** `plans/055-plan--search-must-beat-policy.md` — DONE 2026-10-06. Under the mean backup at player nodes (hardcoded, ce4eda1), the search beats its bare policy and gains with budget: 0.525 / 0.542 / 0.590 at 250 / 1000 / 4000 descents. `scripts/net_check.sh` is the standing per-net check.
- **Value labels:** `plans/056-plan--value-labels.md` — **MC-averaged value labels win** (`botbowl-ui mc-label`, 8 policy-only playouts per sample): value RMS −15% on the frozen MC benchmark (`scripts/value_bench.sh`), and the search beats the control's 0.533 head to head and its own policy 0.582 at 1000 descents. TD(λ) only removes bias. Plan 055's budget gate holds under the mean backup (0.525 / 0.542 / 0.590 vs policy at 250 / 1000 / 4000); `scripts/net_check.sh` is the standing per-net check.
- **Loop throughput:** `plans/058-plan--loop-throughput.md` — after the search CPU cuts and the 4x smaller `GameState`, generation is GPU-bound: 48 local streams over two sidecars (`GEN_SIDECARS=2`, `scripts/perf_gen_steady.sh`), 400 games per shard, ~2 h per generate phase; mc-label is GPU-bound too; drives every 3 generations. The live loop is `runs/loopmix16x9v9` (`scripts/launch_plan058.sh`, new master: 16 more skills, per-player setup with `--next-drive`, schema v9; init = g056 gen04 migrated with `bbnn.migrate`).
- **Policy targets, policy-only drives:** `plans/059-plan--policy-targets-policy-only.md` — exp069 re-tested cq τ and Gumbel σ under the mean backup on one generation step, judged by the absorption probe, the value bench and policy-only paired drives (`cfgs/policy_only.toml`, tract, minutes per 600 pairs). τ=50 keeps the value gain and leads; τ=30 and Gumbel σ cost the value head. τ=50 ran gen09-11 and did not speed up the policy (gen11 vs gen08 policy-only 0.496 vs τ=100's 0.507); **back to τ=100 from gen12** (the user). One-step policy-only drives are blunt (argmax moves in ~0.2% of decisions): compare across generations.
- **Tree statistics:** `plans/060-plan--tree-statistics.md` — every search records leaf depth (plies, own decisions, chance share), where each descent ended relative to the horizon (own turn / opponent's turn / horizon / score / half end), the main line and `opp_turn_follows`: a `tree` block on each corpus sample, means in the `telemetry` block, and `scripts/tree_stats.py` for the tables.
- **Current focus: the training loop's throughput and strength.** Significant speedups to generation, MC labelling and evaluation are wanted (the user, 2026-10-06): measure before and after, and propose them. Bot capability (priors, leaf score, pruning, new skills) continues alongside.

## Working alongside the live loop

- **The training loop and `runs/` live on the training box** (Linux, its own checkout). A macOS checkout has no `runs/`; never create one or touch `runs/`, `data/`, `models/` unless asked.
- **A long-lived worker runs from a checkout's build dir.** Rebuilding into the directory it runs from (`target/release`, or the loop's `target/16x9`) swaps its binary underneath it. The loop and any long-lived worker should run from a dedicated `CARGO_TARGET_DIR` (the launchers use `target/${W}x${H}`), not `target/release`; assume a worker may be running from the main checkout's `target/release` and don't build there.
- **Agents and parallel sessions work in a git worktree** (`.claude/worktrees/…`, own `target/`), or at least with their own `CARGO_TARGET_DIR`. Never switch branches in a checkout a loop or worker runs from, and don't `cd` from a worktree into the main checkout to build.
- `Blocking waiting for file lock on build directory` means another cargo process shares the target dir — wait or use your own `CARGO_TARGET_DIR`; it is not a hang.

## Commands

Botbowl workspace (from repo root or any member crate — shared `target/` and `Cargo.lock` either way):

```sh
cargo test --workspace                # all tests, fast (bot trial benchmarks are #[ignore]d)
cargo test --workspace -- --ignored   # bot benchmark suite only (slow, ~2 min)
cargo test -p botbowl-engine <name>   # one crate / single test by substring
cargo run -p botbowl-ui -- live      # also: snapshot --seed 0 --step 0 | curriculum "Score TD" --difficulty easy --bot mcts | replay <file>
```

Web play (one binary plays any board up to the compiled capacity — build `--release`, a debug MCTS
search reads as a hung UI):

```sh
cd botbowl-web/client && trunk build --release      # needs: cargo install trunk; rustup target add wasm32-unknown-unknown
cargo run --release -p botbowl-web-server   # → http://127.0.0.1:8080; sprites: --assets-dir, web.toml, or a sibling ../botbowl checkout (auto-detected)
```

Both commands work from any directory — the server's `--dist-dir`/`--models-dir` defaults are
resolved from its own crate path, not the cwd. `botbowl-hub serve` also serves the same app at
`http://<hub>:7777/play/` (with `/` an index and `/status` the status page) once the client is
built; teams saved from the editor land in `~/.config/botbowl/teams/`. Both read `~/.config/botbowl/web.toml` (sprites, model dirs; created on first run) and also offer the nets in a worker's `~/.cache/botbowl/models`, by the names the hub sends (since hub protocol v14).

Bot presets and search telemetry (plan 043; every flag is optional — unset is exactly the old behaviour):

```sh
botbowl-ui eval --bot-config cfgs/gumbel16_f1000.toml --vs-config cfgs/baseline.toml ...   # same net, two configurations
botbowl-ui dataset --bot-config cfgs/baseline.toml ...                                 # name stamped into the corpus
botbowl-ui eval --trace-reuse /tmp/reuse.jsonl ...                                     # opt-in per-decision trace
BLOOD_MCTS_STATS=1 ...                                                                 # MCTS_TELEMETRY line on stderr
```

`report.json` and `eval.games.jsonl` always carry a `telemetry` block (tree-reuse by procedure,
recombination hits vs wasted state comparisons); `dataset` stamps the same numbers into each
trajectory's `meta.extra`. See `cfgs/README.md`.

Mixed board sizes (plan 042; every flag is optional — unset means the env board, exactly as before):

```sh
botbowl-ui dataset --mode random-start --board-sizes 12x5,14x7:3,16x9 ...        # weighted list, playable WxH[/T][:w]
botbowl-ui dataset --mode random-start --size-centre 98 --size-temperature 0.3 --size-floor 0.2 ...   # centred grid
botbowl-ui eval --board-sizes 12x5,14x7,16x9 ...                                  # every rung once per board, named opponent@14x7/4
SIZE_MODE=centred scripts/train_loop.sh    # builds at 16x9/6 capacity, schedules the centre from the corpus TD rate
```

Distributed generate/eval (plan 041; `train_loop.sh` starts the hub and the local worker itself):

```sh
cargo run --release -p botbowl-hub -- serve --bind 0.0.0.0:7777        # token: ~/.config/botbowl/hub.token, one per machine
cargo run --release -p botbowl-worker -- --hub ws://<hub-ip>:7777/ws   # on each helper box: same commit + BOARD_SIZE_* build, hub's token in ~/.config/botbowl/hub.token
cargo run --release -p botbowl-hub -- job eval ... --wait      # same flags as `botbowl-ui eval`'s ladder
cargo run --release -p botbowl-hub -- job generate --out-dir runs/<run>/gen12 --shards "0 1 2 3 4 5 6 7" ... --wait   # `botbowl-ui dataset` flags
```

recon_mcts (cd into `recon_mcts/` first): `cargo test`, and `cargo fmt` is **required** after edits (enforced by `.cursor/rules`). Demo bins live in `tests/nim/` (see its CLAUDE.md).

## Git workflow

- **Commit directly to `master`.** No feature branch or PR needed for ordinary work — this is a solo repo and the history is linear.
- **Always commit before starting a training/generation run.** The generator stamps the current commit hash into every corpus it writes (and into `runs/*/status.md`), so launching from a dirty tree produces `<hash>-dirty` stamps that cannot be resolved back to the code that produced the data. Commit first, then launch.

## Cross-cutting invariants

These can be violated from any crate, so they live here; the detail behind each is in the owning crate's CLAUDE.md.

- **Dice discipline (engine):** all randomness goes through `state.dice_mode: DiceMode` (`RollDice` / `FixedDice` / `RegisterRolls` / `DicePolicy`) — never call `state.rng` directly inside a procedure. Tests get `FixedDice` by default from `GameStateBuilder::build()`; production/MCTS/lectures must `set_dice_mode` explicitly.
- **Recombination purity (mcts):** pruning rules (`botbowl-mcts/src/pruning.rs`) and priors (`priors.rs`) must be pure functions of `(state, action)`. Impurity silently splits the DAG and breaks recombination.
- **`GameState`'s `Hash` and `PartialEq` must agree, field for field (engine).** Hash a field iff `PartialEq` compares it. Hashing *less* is only slow — the extra candidates get rejected by `PartialEq` — but hashing *more* is a correctness bug: two equal states would land in different registry buckets and the MCTS DAG would silently split. A new procedure must derive `Hash` next to its `Eq` (`AnyProc` derives both), and a new `GameState` field must be added to both or neither. `botbowl-mcts/tests/hash_quality.rs` gates it from both directions.
- **HashOnly is forbidden:** `recon_mcts`'s `HashOnly` memory mode corrupts Blood Bowl search (hash collisions merge distinct states). It's been removed from `MctsBot`'s `MemoryMode`; never reintroduce it.
- **TDD-first (engine):** new rules get a failing test first. "If code can be removed without breaking tests, it should be."
