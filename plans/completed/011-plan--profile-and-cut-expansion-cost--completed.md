# Plan 011 — Profile and cut state-derivation cost (completed / superseded)

**Status:** Closed out. The original framing ("expansion cost dominates") was refuted by the Step 1
profile; the real bottleneck turned out to be DAG-depth pathology (plan 012) and missing horizon
bound (plan 014), and the real wins came from lazy expansion (plan 016) and tree reuse + virtual
loss (plan 015). The original cheap-wins listed below (bitmask skills, Arc `AvailableActions`, Arc
outer FullPitch) did **not** land — performance work is deprioritized for now in favour of bot
capability.

The stale baseline-numbers file (`plans/011-baseline-results.md`) was deleted along with this
plan's archival; if a future session wants to re-baseline, re-run `expand_bench` (see
`PROFILING.md`) and capture a fresh snapshot.

## What the profile actually said (kept for reference)

Captured 2026-05-24 on macOS arm64 against `expand_bench_for_samply`:

- **Pathing is not the bottleneck.** `PathFinder::player_paths` inclusive was ~0.08%. The
  hypothesis that "path-tree build dominates engine `micro_step` cost" was wrong; the L1 (lazy
  paths) lever was dropped from the plan.
- **Clone is the bottleneck.** `<GameState as Clone>::clone` inclusive was ~55%, dominated by
  `recon_mcts::Node::get_state` cloning the root at every `tree.step()`. This is what tree reuse
  (plan 015 Step 1) and `StoreState` (plan 013) target.
- **The DAG saturated very fast** in the (pre-014) regime — 200 000 iters touched only 3 unique
  nodes on `score_td_easy`. Plan 014's horizon bound and plan 016's lazy expansion both reshape
  this; the descent-vs-expansion ratio has not been re-measured since.

## What's still on the table if we ever return to perf

The list below is from the pre-deprioritization era; each item is independent of the others, none
have landed, and all of them deserve a fresh profile before being picked up.

- **Bitmask `used_skills` + `PlayerStats.skills`.** Today they are `HashSet<Skill>` per fielded
  player (`botbowl-engine/src/core/model.rs`). 22 small `HashMap` allocs per `GameState::clone`.
- **`Arc<AvailableActions>` instead of `Box`.** `Box<AvailableActions>` is on a hot drop path; the
  struct holds two `FullPitch`-shaped options that get deep-copied on every clone.
- **Arc the outer `FullPitch` on `AvailableActions.paths`.** The inner `Option<Arc<Node>>` is
  already shared; the outer 28×17 array still deep-copies.
- **`Tree::make_branch` self-time mystery.** A samply attribution oddity (the function ran ~20
  times in 200k iters but showed 43% self-time). Worth re-running with `-C inline-threshold=0`
  before trusting samply's number.

## Pitfalls (still valid if you do come back to this)

- **Recombination invariant.** Anything that defers state derivation must preserve
  "same (parent, action) → same child state hash" — see `CLAUDE.md` and `pruning.rs` /
  `priors.rs` purity comments.
- **Profile after every change.** The whole premise of this plan was "measure"; eyeballing
  perf changes is what produced the wrong cheap-win ordering the first time round.

## Out of scope (unchanged)

- Heuristic priors beyond the five in `priors.rs` — those are bot-capability work (idea 003),
  not perf.
- Engine API redesign.
