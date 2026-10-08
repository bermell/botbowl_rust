# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this crate.

## What this is

`recon_mcts` — a generic **re**combining, **con**current Monte Carlo Tree Search library in safe std-only Rust. It is an internal Cargo workspace with the library at the root and the demo/integration tests living in `tests/nim/` (a 2048 implementation, despite the directory name).

This crate lives inside the `botbowl_rust/` repository as a **nested, separate workspace** (it is in the parent workspace's `exclude` list). It is consumed by `botbowl-mcts` via a path dependency (`botbowl-mcts/Cargo.toml` → `path = "../recon_mcts/"`). It has no dependency on the botbowl crates and can be developed in isolation — `cd` into this directory before running cargo.

## Commands

```sh
cargo test                            # runs lib tests + tests/nim/ workspace member
cargo fmt                             # required after edits — enforced by .cursor/rules
cargo run --bin visualize_2048 -p recon_mcts-test_nim
cargo run --bin benchmark_2048 -p recon_mcts-test_nim --release
cargo run --bin compare_2048   -p recon_mcts-test_nim --release
```

Note: `tests/nim/` is a **separate workspace member** so it can be compiled with `--features test_internals` by default — that feature exposes otherwise-private functions to the tests. Don't move the tests back into the root crate. This separation is also why recon_mcts must stay excluded from the parent `botbowl_rust` workspace: merging it in would unify features (forcing `test_internals` everywhere) and pull the nim tests into `cargo test --workspace`.

## Architecture — DAG-shaped concurrent MCTS

The core abstraction is the `GameDynamics` trait (see crate-level doc-comment in `src/lib.rs`). Implementors define `Player`, `State`, `Action`, `Score`, plus `available_actions`, `player_for_child`, `apply_action`, `select_node`, `score_leaf`, `backprop_scores`. The library handles the tree.

**`available_actions` returns bare actions; the mover comes from `player_for_child` (plan 035).** A node's tag names the mover of that node itself, and it is derived once, at materialisation, from `(parent_player, action, child_state)` — not at enumeration time. The old contract had `ActionIter::Item = (Player, Action)`, which forced an implementation to pre-compute each candidate's resulting state just to name its mover: in botbowl that was one full engine advance per candidate at every expansion, discarded immediately, exactly undoing lazy expansion. `Node.player` is therefore a `OnceLock<P>` — `Node::player()` panics (in release too) if anything reads it before materialisation, and `NodeInfo.player` is `Option<P>` because the inspection API can legitimately see an un-descended placeholder. `player_for_child` must be **pure**, like `apply_action`: node identity hashes `(player, state)`, so an impure tag splits the DAG. `tests/player_tag.rs` pins both halves.

Distinctive design points to keep in mind when touching this crate:

- **Recombining**: states reachable by multiple action sequences share a single node. The tree is therefore a DAG, not a tree — nodes have multiple parents, and backprop fans out to all of them. Don't introduce data structures that assume single-parent.
- **DAG enforcement**: the graph must be acyclic (a `GameDynamics` whose states can recur creates a real cycle via recombination). `step_into` carries a per-descent visited-set; on a revisit it dumps the descent path to a temp file (rendered via the `GameDynamics::fmt_state`/`fmt_action` diagnostic hooks) and **panics** — crash-not-hang, `tests/cycle_guard.rs`. The fix always belongs in the `GameDynamics` impl (encode repetition into the state), never in relaxing the guard.
- **States are read, not replayed (2026-10-07)**: under `StoreState` a registered node holds its state, and `descend` reads it instead of calling `apply_action` along each edge; only a placeholder (or a node whose memory mode dropped its state) is derived. Node equality compares stored states in place, and short-circuits on pointer identity: node teardown removes every node from its children's `parents` sets and from the registry, and a cloning comparison made that cost two state clones per removal. Both were the largest CPU costs in botbowl's search (plan 058 §7). Anything that passes states down a descent must keep `PartialEq`-equal states interchangeable.
- **Topologically aware backprop**: a node only propagates upward once it has received updates from all children below it on the current path. This matters when extending `backprop_scores` — preserve the wait-for-all-children semantics.
- **Solved-subtree pruning**: every `Node` carries a monotone `solved` flag — terminal (`Children::None`) nodes are solved, and a node whose children are all solved is solved (propagated by the `backprop_scores` walk). Solved children are hidden from `GD::select_node`; a solved root makes `step()` a no-op, and callers should poll `Tree::is_solved()` to end their step loop (see `tests/solved.rs`). Selection never sees an empty candidate set: if everything is solved the full set is offered as a fallback.
- **Concurrent**: multiple worker threads grow the same tree; idle threads steal work from the thread expanding a leaf to avoid hot-path log-jams. Anything new must remain thread-safe under this scheme.
- **Feature flags**: `stable` (default), `nightly`, `two_player`, `test_internals`, `lockref-guard`. Public API surface differs by feature. Tests run with `test_internals` to reach private helpers; do not paper over visibility by widening `pub` in `src/` — gate it on the feature instead. `lockref-guard` is an opt-in debug-build guard against the chained-`lockref::Ref` deadlock pattern (see botbowl plan 013); downstream GD impls opt in via their dependency declaration.
- **No external runtime deps**: the core library is intentionally std-only safe Rust. Rand/rayon usage belongs in `tests/nim/`, not the root crate.

Module map: `tree.rs` (DAG + worker coordination), `game_dynamics.rs` (trait), `lockref.rs` / `map_maybe.rs` / `ref_iter.rs` / `unique_heap.rs` (supporting primitives).

## Conventions

- Run `cargo fmt` before committing (per `.cursor/rules/about.mdc`).

## `GameDynamics::release_descent` (2026-09-30)

Called by `Tree::step_into` after every descent, once per edge on the descent's path, root first,
with the edge's parent player and its child's current score. It is where a per-descent adjustment
made in `select_node` (virtual loss) gets taken back. Default no-op, so nim and any `DynGD` user are
unchanged. It exists because relying on a backprop to *replace* the child's score leaks: a descent
whose backprop is cut off (`backprop_scores` returning `None`) never replaces the scores above the
cut. `step_into` wraps `descend`, which is the old loop, so every exit path is covered.

## `GameDynamics::observe_descent` (2026-10-08, botbowl plan 060)

Called once per descent, from inside `descend` just before it returns (after the backprop), with
the edges the descent took — `(player of the node the edge leaves, action)`, root first, twin
swaps skipped exactly as the `release_descent` loop skips them — the state it stopped at, and a
`DescentEnd`: `Expanded` (a fresh leaf was materialised and enumerated), `Terminal` (a node with no
children) or `Solved` (every child solved). Observational only, default no-op. Botbowl's tree
statistics (leaf depth, where lines end relative to the horizon) are built on it. The inspection
API gained `Node::get_children` (actions + `Arc`s, no state clones), `Node::mover` and
`Node::with_state` for cheap read-only walks. `tests/observe_descent.rs`.
