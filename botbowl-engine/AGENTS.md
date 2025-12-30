# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` exposes the crate API and ties modules together.
- `src/bots.rs` contains bot-related logic and entry points.
- `src/core/` houses the game engine: models, game state, tables, dice, pathing, and procedures (`src/core/procedures/`).
- No dedicated test directory is present yet; tests should live next to modules (e.g., `src/core/model.rs`).

## Project Goals & Roadmap
- Goal: implement Blood Bowl 2020 rules in a fast Rust engine aimed at strong AI play.
- TODO examples: kickoff table completion, useful setup logic, and pathfinding that supports leaping over prone players.
- Future tooling ideas: game recording viewer, MCTS example bot, benchmarking scenarios, and Python FFI.

## Build, Test, and Development Commands
- `cargo build` compiles the crate in debug mode.
- `cargo run` runs a binary if one is configured (none found in this crate yet).
- `cargo test` runs unit tests embedded in modules under `src/`.
- `cargo check` performs a fast type-check without producing binaries.

## Coding Style & Naming Conventions
- Indentation: 4 spaces, Rust standard formatting.
- Naming: `snake_case` for functions/modules, `CamelCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants.
- Prefer `mod.rs` or `mod` files consistent with existing layout in `src/core/`.
- Use `cargo fmt` to apply default rustfmt if installed; no custom config found.

## Testing Guidelines
- Framework: Rust’s built-in test harness (`#[test]` in module files).
- Co-locate tests with code (e.g., `src/core/pathing.rs`) or add a `tests/` folder if integration tests are needed.
- Run all tests with `cargo test` before opening a PR.
- Development is TDD-first; avoid keeping code that is not covered by tests unless it handles unusual error paths.
- Coverage: install `cargo-tarpaulin` and run `cargo tarpaulin --out Html`, then review `git blame` for uncovered code.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and lower-case (e.g., “toggle auto stepping”).
- Keep commits focused; avoid mixing refactors and behavior changes.
- PRs should include: a brief summary, testing performed (`cargo test`/`cargo check`), and links to relevant issues if any.

## Agent-Specific Instructions
- This repository uses Codex skills; follow the instructions in `AGENTS.md` at the repo root when updating or extending automation guidance.
