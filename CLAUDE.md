# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. **Each crate has its own `CLAUDE.md` with architecture detail** — it loads automatically when you work on files in that crate; read it before making non-trivial changes there.

## Repository layout

One git repo containing the botbowl Cargo workspace plus the nested `recon_mcts/` library (folded in via history-preserving subtree merge).

- Botbowl workspace (`Cargo.toml` at repo root) — Blood Bowl 2020 engine + tooling. Four member crates sharing one `Cargo.lock` and one `target/`:
  - `botbowl-engine/` — pure rules library, procedure-stack state machine. No dependency on the other crates. Board/team size is build-time configurable via env vars (see its CLAUDE.md).
  - `botbowl-curriculum/` — training scenarios (`Lecture` trait, `run_trials`). Depends on `botbowl-engine`.
  - `botbowl-mcts/` — `BloodBowlDynamics` + `MctsBot`, the adapter onto `recon_mcts`. Depends on `botbowl-engine` + `recon_mcts`.
  - `botbowl-ui/` — `ratatui` terminal frontend with `live` / `replay` / `snapshot` / `curriculum` subcommands. Depends on the other three.
- `recon_mcts/` — generic **re**combining, **con**current MCTS library (safe std-only Rust). A **nested, separate Cargo workspace**, deliberately in the botbowl workspace's `exclude` list — don't merge it in (its `tests/nim/` member compiles with `--features test_internals` by default). Has its own `CLAUDE.md`. No dependency on the botbowl crates.

## Plans

- `plans/001-grand-plan.md` — strategic roadmap (AlphaZero-style MCTS via curriculum learning → scripted baseline → heuristic/rollout/NN-guided MCTS → self-play). Read it before proposing architecture changes that span the engine and `recon_mcts`.
- `plans/NNN-idea--*.md` / `plans/NNN-plan--*.md` — designs not yet started or in-flight. `plans/completed/` — closed-out plans with **Status:** headers; historical context, not live work.
- **Current focus: bot capability** (priors, leaf-score, pruning, scripted heuristics, new lectures). Performance work is deprioritized — don't propose perf tuning, profiling reruns, or speed micro-benchmarks unless explicitly asked.

## Commands

Botbowl workspace (from repo root or any member crate — shared `target/` and `Cargo.lock` either way):

```sh
cargo test --workspace                # all tests, fast (bot trial benchmarks are #[ignore]d)
cargo test --workspace -- --ignored   # bot benchmark suite only (slow, ~2 min)
cargo test -p botbowl-engine <name>   # one crate / single test by substring
cargo run -p botbowl-ui -- live      # also: snapshot --seed 0 --step 0 | curriculum "Score TD" --difficulty easy --bot mcts | replay <file>
```

recon_mcts (cd into `recon_mcts/` first): `cargo test`, and `cargo fmt` is **required** after edits (enforced by `.cursor/rules`). Demo bins live in `tests/nim/` (see its CLAUDE.md).

## Git workflow

- **Commit directly to `master`.** No feature branch or PR needed for ordinary work — this is a solo repo and the history is linear.
- **Always commit before starting a training/generation run.** The generator stamps the current commit hash into every corpus it writes (and into `runs/*/status.md`), so launching from a dirty tree produces `<hash>-dirty` stamps that cannot be resolved back to the code that produced the data. Commit first, then launch.

## Cross-cutting invariants

These can be violated from any crate, so they live here; the detail behind each is in the owning crate's CLAUDE.md.

- **Dice discipline (engine):** all randomness goes through `state.dice_mode: DiceMode` (`RollDice` / `FixedDice` / `RegisterRolls` / `DicePolicy`) — never call `state.rng` directly inside a procedure. Tests get `FixedDice` by default from `GameStateBuilder::build()`; production/MCTS/lectures must `set_dice_mode` explicitly.
- **Recombination purity (mcts):** pruning rules (`botbowl-mcts/src/pruning.rs`) and priors (`priors.rs`) must be pure functions of `(state, action)`. Impurity silently splits the DAG and breaks recombination.
- **HashOnly is forbidden:** `recon_mcts`'s `HashOnly` memory mode corrupts Blood Bowl search (hash collisions merge distinct states). It's been removed from `MctsBot`'s `MemoryMode`; never reintroduce it.
- **TDD-first (engine):** new rules get a failing test first. "If code can be removed without breaking tests, it should be."
