# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. **Each crate has its own `CLAUDE.md` with architecture detail** — it loads automatically when you work on files in that crate; read it before making non-trivial changes there.

## Repository layout

One git repo containing the botbowl Cargo workspace plus the nested `recon_mcts/` library (folded in via history-preserving subtree merge).

- Botbowl workspace (`Cargo.toml` at repo root) — Blood Bowl 2020 engine + tooling. Member crates sharing one `Cargo.lock` and one `target/` (`botbowl-data` and `botbowl-nn` are members too; both carry their own `CLAUDE.md`):
  - `botbowl-engine/` — pure rules library, procedure-stack state machine. No dependency on the other crates. Board/team size is build-time configurable via env vars (see its CLAUDE.md).
  - `botbowl-curriculum/` — training scenarios (`Lecture` trait, `run_trials`). Depends on `botbowl-engine`.
  - `botbowl-mcts/` — `BloodBowlDynamics` + `MctsBot`, the adapter onto `recon_mcts`. Depends on `botbowl-engine` + `recon_mcts`.
  - `botbowl-play/` — "play one game, return its record": the process-agnostic core under `botbowl-ui dataset`/`eval` (trajectory generation, ladder games, `EvalGameLine`/`LadderRow`/`Report`, bot construction). No files, threads or CLI in it; plan 041's hub/worker reuse it verbatim. Depends on engine, curriculum, mcts, nn, data.
  - `botbowl-hub/`, `botbowl-worker/`, `botbowl-hub-proto/` — distributed generation and eval (plan 041): the hub on the training box queues game batches, workers on any machine dial in over a websocket and stream results back (trajectories zstd-compressed); the hub writes the same files (`shard$K.jsonl`, `eval.games.jsonl`, `report.json`) the local phases wrote. Shared `CLAUDE.md` in `botbowl-hub/`. Depend on `botbowl-play`.
  - `botbowl-ui/` — `ratatui` terminal frontend with `live` / `replay` / `snapshot` / `curriculum` subcommands, plus the headless `dataset` / `eval` shells over `botbowl-play`. Depends on the other four.
  - `botbowl-web/{proto,server,client}/` — human-vs-bot play in a browser, with the bot's search shown next to the board (plan 034). `proto` is engine-free and compiles to wasm32; `server` owns the `GameState` and the bots; `client` is a Leptos CSR app built with `trunk`. Has its own `CLAUDE.md`.
- `recon_mcts/` — generic **re**combining, **con**current MCTS library (safe std-only Rust). A **nested, separate Cargo workspace**, deliberately in the botbowl workspace's `exclude` list — don't merge it in (its `tests/nim/` member compiles with `--features test_internals` by default). Has its own `CLAUDE.md`. No dependency on the botbowl crates.

## Plans

- `plans/001-grand-plan.md` — strategic roadmap (AlphaZero-style MCTS via curriculum learning → scripted baseline → heuristic/rollout/NN-guided MCTS → self-play). Read it before proposing architecture changes that span the engine and `recon_mcts`.
- `plans/NNN-idea--*.md` / `plans/NNN-plan--*.md` — designs not yet started or in-flight. `plans/completed/` — closed-out plans with **Status:** headers; historical context, not live work.
- **Live experimental programme:** `plans/031-plan--audit-diagnostics.md` (cheap diagnostics, run first) and `plans/032-plan--ranked-experiment-queue.md` (ranked longer experiments and open questions). New results go there, not into completed plans.
- **Search instrumentation + bot presets:** `plans/043-plan--search-instrumentation-and-bot-configs.md` — tree-reuse and recombination counters that reach `report.json`, a trajectory's provenance and the web debug drawer, plus `cfgs/*.toml` bot presets (`--bot-config` / `--vs-config`) so the same net can play itself under two configurations.
- **State-hash discrimination:** `plans/044-plan--state-hash-discrimination.md` — `GameState::hash` now walks the procedure stack and `GameInfo` whole, taking colliding states from 36.8% to 0% and wasted state comparisons from 4.07 to 0.12 per registry probe.
- **Board-size curriculum:** `plans/042-plan--board-size-curriculum.md` — mixed-size generation (`--board-sizes` / `--size-centre …` on `dataset` and `job generate`, per-board eval rungs via `eval --board-sizes`), schema v7, the trainer's multi-dims loader and `train_loop.sh`'s `SIZE_MODE`. Experiments E0–E5 there are the next thing to run.
- **Current focus: bot capability** (priors, leaf-score, pruning, scripted heuristics, new lectures). Performance work is deprioritized — don't propose perf tuning, profiling reruns, or speed micro-benchmarks unless explicitly asked.

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
cargo run --release -p botbowl-web-server -- \
    --assets-dir /Users/mattias/repos/blood/botbowl/botbowl/web/static/img   # → http://127.0.0.1:8080
```

Both commands work from any directory — the server's `--dist-dir`/`--models-dir` defaults are
resolved from its own crate path, not the cwd.

Bot presets and search telemetry (plan 043; every flag is optional — unset is exactly the old behaviour):

```sh
botbowl-ui eval --bot-config cfgs/aggressive.toml --vs-config cfgs/baseline.toml ...   # same net, two configurations
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
cargo run --release -p botbowl-hub -- serve --bind 0.0.0.0:7777 --token-file runs/<run>/hub.token
cargo run --release -p botbowl-worker -- --hub ws://<hub-ip>:7777/ws --token-file hub.token   # on each helper box, same commit + BOARD_SIZE_* build
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
