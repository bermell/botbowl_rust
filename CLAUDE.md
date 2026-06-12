# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Everything now lives under a single repo, `botbowl_rust/` (its own `.git`). It contains the botbowl Cargo workspace plus the `recon_mcts/` library, which was folded in (history-preserving subtree merge) from what used to be a separate sibling repo.

- `botbowl_rust/` — a Blood Bowl 2020 game engine (Rust rewrite of [njustesen/botbowl](https://github.com/njustesen/botbowl)), plus a terminal UI, curriculum harness, and MCTS adapter. A **single Cargo workspace** (`botbowl_rust/Cargo.toml`) with four member crates sharing one `Cargo.lock` and one `target/`:
  - `botbowl-engine/` — pure rules library; no dependency on the other crates.
  - `botbowl-curriculum/` — training scenarios (`Lecture` trait, `LectureSession`, `run_trials`). Depends on `botbowl-engine`.
  - `botbowl-mcts/` — `BloodBowlDynamics` (impls `recon_mcts::GameDynamics`) and `MctsBot`. Depends on `botbowl-engine` + `recon_mcts` (path dep to the nested `botbowl_rust/recon_mcts/` crate).
  - `botbowl-ui/` — `ratatui`/`crossterm` terminal frontend with `live` / `replay` / `snapshot` / `curriculum` subcommands. Depends on the other three.
- `botbowl_rust/recon_mcts/` — a generic **re**combining, **con**current Monte Carlo Tree Search library in safe std-only Rust. A **nested, separate Cargo workspace** (library at the root + demo/integration tests in `tests/nim/`, a 2048 implementation despite the directory name) — it is in the botbowl workspace's `exclude` list so its `test_internals`-by-default tests stay isolated. Has its own `CLAUDE.md`. No dependency on the botbowl crates.
- `botbowl_rust/plans/001-grand-plan.md` — strategic roadmap: apply AlphaZero-style MCTS (using `recon_mcts`) to Blood Bowl through a curriculum-learning suite, scripted-agent baseline, then heuristic / rollout / NN-guided MCTS, and finally self-play. Read 001 before proposing architecture changes that span the engine and `recon_mcts`.
  - `plans/NNN-idea--*.md` / `plans/NNN-plan--*.md` — incremental designs not yet started or in-flight.
  - `plans/completed/` — closed-out plans, learnings, and resolved issues. Files end in `--completed.md`. Each carries a **Status:** header pointing at the code that landed (or noting why work was deferred). Read these for historical context, not as live work.
  - **Current focus: bot capability** (priors, leaf-score, pruning, scripted heuristics, new lectures). Performance work is deprioritized — don't propose perf tuning, profiling reruns, or speed micro-benchmarks unless explicitly asked.

`cd` into the project you're working on before running cargo; commands below assume you're inside that project's root.

## Commands

### botbowl_rust (workspace — run from `botbowl_rust/` or any member crate)

```sh
# from botbowl_rust/ (workspace root)
cargo build --workspace
cargo test --workspace                # all tests in every member (fast — bot trial benchmarks are #[ignore]d)
cargo test --workspace -- --ignored   # run only the bot benchmark suite (slow, ~2 min)
cargo test -p botbowl-engine          # one crate only
cargo test -p botbowl-engine <name>   # single test by substring
cargo run -p botbowl-ui -- live
cargo run -p botbowl-ui -- snapshot --seed 0 --step 0
cargo run -p botbowl-ui -- curriculum "Score TD" --difficulty easy --bot mcts
cargo run -p botbowl-ui -- replay path/to/recording.json
cargo tarpaulin -p botbowl-engine --out Html   # coverage report (engine)
```

`cargo` run from inside a member crate still uses the workspace's shared `target/` and `Cargo.lock` at `botbowl_rust/`.

### recon_mcts (run inside `botbowl_rust/recon_mcts/`)

```sh
cargo test                            # runs lib tests + tests/nim/ workspace member
cargo fmt                             # required after edits — enforced by .cursor/rules
cargo run --bin visualize_2048 -p recon_mcts-test_nim
cargo run --bin benchmark_2048 -p recon_mcts-test_nim --release
cargo run --bin compare_2048   -p recon_mcts-test_nim --release
```

Note: `recon_mcts/tests/nim/` is a **separate workspace member** so it can be compiled with `--features test_internals` by default — that feature exposes otherwise-private functions to the tests. Don't move the tests back into the root crate.

## Architecture

### botbowl-engine — procedure-stack game state machine

The engine drives Blood Bowl as a stack of **procedures** rather than a single monolithic state machine. Each rules subsystem (CoinToss, Half, KickOff, Block, Movement, Casualty, Ball, …) is an enum variant of `AnyProc` implementing the `Procedure` trait (`core/model.rs:567`):

```rust
pub trait Procedure: std::fmt::Debug {
    fn step(&mut self, game_state: &mut GameState, input: ProcInput) -> ProcState;
}
```

`GameState.proc_stack: Vec<AnyProc>` is the heart of the engine. Stepping the game pops the top procedure, calls `step`, and the returned `ProcState` tells the runner to:

- `Done` / `DoneNew(proc)` — pop, optionally push successor procedures
- `NotDone` — keep procedure on stack
- `NeedRoll(RequestedRoll)` — engine resolves a die roll (how it resolves depends on `state.dice_mode`) and re-enters with the result as `ProcInput`
- `NeedAction(AvailableActions)` — engine waits for the controlling bot/UI to supply an `Action`

Key invariants:

- All dice resolution is dispatched on `state.dice_mode: DiceMode` — one explicit, mutually-exclusive variant per caller intent. Switch with `state.set_dice_mode(mode)`. Variants:
  - `RollDice` — production play / search rollouts. RNG only (via `state.rng`, a seedable `ChaCha8Rng`).
  - `FixedDice(queue)` — tests and `GameStateBuilder` setup. FIFO queue of pinned values; pop on demand, panic on empty. Test ergonomics: `state.fix_d6(5)`, `state.fix_coin(Coin::Heads)`, etc. — these all panic if the mode isn't `FixedDice`.
  - `RegisterRolls` — MCTS bot. `step()` is forbidden; the engine pauses on `NeedRoll`, stashing the request in `state.pending_roll`. Caller observes, calls `state.step_with_roll(result)` to resume.
  - `DicePolicy(policy)` — lectures and scripted scenarios. The policy is *total*: it must resolve every requested roll. Built-in policies delegate to RNG internally for roll types they don't override (e.g. `SucceedAtOrEasier` pins pickup/dodge outcomes; scatter/bounce stay stochastic).
- Default after `GameStateBuilder::build()` is `FixedDice(empty)` — tests work without ceremony. Production paths (`BotGameRunner`), MCTS (`MctsBot::get_action`), and lectures explicitly call `set_dice_mode` after construction.
- Never bypass the mode by calling `state.rng` directly inside a procedure; the mode is the single point of control for reproducibility.
- `AnyProc` (the variant enum holding all procedures) is generated by the `any_proc!` macro in `core/procedures/any_proc.rs` — adding a new procedure means one line in that macro invocation, not three identical 35-arm matches.
- `GameStateBuilder` has multiple `new_at_*` constructors (`new_start_of_game`, `new_at_setup`, `new_at_kickoff`) that fast-forward through earlier procedures so tests can jump straight to the situation under test.
- Actions split into `SimpleAT` (no position) and `PosAT` (positional). `AvailableActions` carries the legal set plus a `FullPitch<Option<Arc<Node>>>` of precomputed pathfinding nodes — pathing is computed once per turn and reused for legality + movement resolution. (`Arc`, not `Rc`, so `GameState: Send` for the multi-threaded MCTS workers.)
- The engine is developed **TDD-first** (per `botbowl_rust/README.md`): "If code can be removed without breaking tests, it should be." Exception is error-handling for weird states. When adding rules, add the failing test first.

`botbowl-ui` is a `ratatui`/`crossterm` terminal renderer driving a `BotGameRunner`. Bots live in `botbowl-engine/src/bots.rs` (`Bot` trait + `RandomBot`) and `botbowl-engine/src/scripted_bot.rs` (scripted heuristic bot); the MCTS bot is `botbowl_mcts::MctsBot` and is selected via the `--bot mcts` CLI flag.

### botbowl-curriculum — training-scenario harness

`Lecture` trait describes a scenario: `setup(rng) -> GameState`, `evaluate(state, ctx) -> {Success, Failure, InProgress}`, plus metadata like `agent_team()`. `LectureSession` drives one trial one `micro_step` at a time (useful for live ratatui rendering); `run_trials` is the headless batch driver. Lectures are registered in `lib.rs::make_lecture` and surfaced through the UI's `curriculum` subcommand.

### botbowl-mcts — recon_mcts adapter

`BloodBowlDynamics` implements `recon_mcts::GameDynamics<State=GameState, Action=BbAction, Player=BbPlayer, Score=BbScore>`. Key shape:

- Three "players": `Home`, `Away`, `Chance`. Chance nodes appear whenever `state.pending_roll.is_some()` and their children are `BbAction::Chance { outcome, prob_bits }`. In production the chance path is collapsed inside `score_leaf` / `apply_action` so chance children rarely enter the tree (plan 010 Track A.alt).
- `available_actions` filters via `pruning::should_prune` (must be a pure function of `(state, action)` — recombination depends on it). `block_dice::scripted_pick` collapses block-die fan-out to a single scripted choice when the engine offers one, and `scripted::scripted_player_pick` collapses coin toss / kick-receive inside `apply_action`'s quiescent loop. `available_actions` is also **horizon-bounded** (plan 014): once the state has moved past the root's `HorizonAnchor` (turn boundary, score change, game over), it returns `None` and MCTS treats the state as terminal.
- Selection uses PUCT with `prior_for(state, action)` priors (`priors.rs`, ~5 multipliers — see plan 004); `PUCT_C = 10.0` is tuned against the leaf-score magnitudes in `score.rs` and they are coupled — changing one without the other silently degrades search.
- Scores stay **Home-centric end-to-end** (plan 006). Home nodes maximise in both `select_node` (PUCT) and `backprop_scores`; Away nodes mirror via `home_perspective`. Visits sum across children on both Player and Chance branches.
- `score_leaf` "fast-forwards" (`optimistic_leaf_score`) through up to `ff_depth` mid-procedure micro-steps to reach a decision/terminal state, picking the `Pass`/`Advance` outcome each time, so a chance leaf is scored with its optimistic outcome instead of the pre-roll position.
- `recon_mcts` materialises children **lazily** (plan 016): each new child is a cheap placeholder until descent picks it.
- `MctsBot::get_action` clones the root state, sets `DiceMode::RegisterRolls`, force-disables logging and clears the log Vec (otherwise each `apply_action` clone re-copies the whole log), splits `iterations_per_move` across `std::thread::scope` workers (plan 008), and picks the most-visited root child. The tree is **cached and reused** across consecutive `get_action` calls when the horizon anchor matches (plan 015 Step 1). Workers also apply a transient virtual-loss penalty on descent (plan 015 Step 5) to diverge under concurrency.
- **Memory mode** (plan 013): `MctsBot.memory_mode` is always `MemoryMode::StoreState` in production. **GOTCHA:** `recon_mcts`'s `HashOnly` marker is *broken* for Blood Bowl and must never be used — a `GameState` is large enough that hash collisions are inevitable, and `HashOnly` merges any two states that collide into one DAG node, producing illegal actions mid-search, corrupted backprop, and drop-time panics. The `HashOnly` variant has been removed from `MemoryMode`; only `StoreState` (default, structural O(1) equality) and `GetState` (safe replay-based diagnostic) remain. Never reach for `recon_mcts::HashOnly` when wiring a Blood Bowl tree.
- **Env knobs for A/B and debugging** (read once per `get_action`): `BLOOD_MCTS_MEMORY={get|store}` (`hash` panics — see above), `BLOOD_MCTS_WORKERS=N`, `BLOOD_MCTS_HORIZON=off`, `BLOOD_MCTS_TREE_REUSE=off`, `BLOOD_MCTS_VIRTUAL_LOSS=N`, `BLOOD_MCTS_STATS=1`.

### recon_mcts — DAG-shaped concurrent MCTS

The core abstraction is the `GameDynamics` trait (see crate-level doc-comment in `src/lib.rs`). Implementors define `Player`, `State`, `Action`, `Score`, plus `available_actions`, `apply_action`, `select_node`, `score_leaf`, `backprop_scores`. The library handles the tree.

Distinctive design points to keep in mind when touching this crate:

- **Recombining**: states reachable by multiple action sequences share a single node. The tree is therefore a DAG, not a tree — nodes have multiple parents, and backprop fans out to all of them. Don't introduce data structures that assume single-parent.
- **Topologically aware backprop**: a node only propagates upward once it has received updates from all children below it on the current path. This matters when extending `backprop_scores` — preserve the wait-for-all-children semantics.
- **Concurrent**: multiple worker threads grow the same tree; idle threads steal work from the thread expanding a leaf to avoid hot-path log-jams. Anything new must remain thread-safe under this scheme.
- **Feature flags**: `stable` (default), `nightly`, `two_player`, `test_internals`. Public API surface differs by feature. Tests run with `test_internals` to reach private helpers; do not paper over visibility by widening `pub` in `src/` — gate it on the feature instead.
- **No external runtime deps**: the core library is intentionally std-only safe Rust. Rand/rayon usage belongs in `tests/nim/`, not the root crate.

Module map: `tree.rs` (DAG + worker coordination), `game_dynamics.rs` (trait), `lockref.rs` / `map_maybe.rs` / `ref_iter.rs` / `unique_heap.rs` (supporting primitives).

## Conventions

- After edits in `recon_mcts/`, run `cargo fmt` before committing (per `.cursor/rules/about.mdc`).
- New rules in `botbowl-engine` get a test first; if a test passes without the new code, the code isn't needed.
- `recon_mcts/` is a nested **separate Cargo workspace** (excluded from the botbowl workspace), but now shares the one `botbowl_rust` git repo — a change touching both the botbowl crates and `recon_mcts` is a single commit, not two repos.
- The four member crates of `botbowl_rust/` share one `Cargo.lock` at the workspace root — upgrading a shared dep updates every member at once.
- Pruning rules (`botbowl-mcts/src/pruning.rs`) and priors (`botbowl-mcts/src/priors.rs`) **must be pure functions of `(state, action)`**. Two paths to the same logical state that return different action subsets will silently split the DAG and break recombination.
