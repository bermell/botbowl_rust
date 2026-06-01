# Plan 011 — Profile and cut state-derivation cost (post-baseline rewrite)

**Status:** Step 1 (profile) is **done**. Baseline numbers, samply breakdown and counter-wrapper call counts live in
`plans/011-baseline-results.md`. The findings substantially **revise this plan** — the headline hypotheses that drove
the original Step 2/3/4/5 ordering were refuted. This document is the post-baseline re-plan.

**Bench/profile harness:** `botbowl-mcts/tests/expand_bench.rs` (three `#[ignore]`d tests: wall-clock, call-counts,
samply target). Recipe in `PROFILING.md`. Re-run after every change in this plan and append a new dated block to
`plans/011-baseline-results.md`.

## What the profile said

(See `plans/011-baseline-results.md` for the full tables. Numbers captured at commit `4504baf`, macOS arm64, seed
`0xCAFE_1234`.)

- **Pathing is not the bottleneck.** `PathFinder::player_paths` inclusive = **0.08%**. The original "Why this matters"
  framing ("path-tree build dominates engine `micro_step` cost… multiple orders of magnitude more expensive than a state
  clone") was wrong. L1 (lazy paths) is dropped from the plan.
- **Clone is the bottleneck.** `<GameState as Clone>::clone` inclusive = **55.8%**. The dominant caller is
  `recon_mcts::Node::get_state` at the _top of every `tree.step()`_, not the per-child clone at expansion time. In
  HashOnly mode the child states are dropped after scoring, so descent into a node must re-derive state from the root by
  cloning it — and almost every step starts that walk from the root.
- **The DAG saturates fast.** With heavy recombination, 200 000 `tree.step()` iters touched only **3 unique nodes on
  `score_td_easy`** and **20 unique nodes on `full_teams`**. After the first handful of iters, ~99.99% of steps are
  "descend one level, hit a recombined existing node, return". Plan 011's original framing assumed expansion-dominated
  cost; the measured regime is descent-dominated.
- **Tree internals are real cost.** `Tree::make_branch` self-time is 43.7% (suspicious — needs follow-up, see open
  questions). `hashbrown` bucket walks (registry lookups) are 10.5%.

## What's left to do — updated cheap wins and bigger levers

Order is by expected wall-clock impact under the measured regime. Re-profile after each step.

### 1. (Highest impact) Tree reuse across `get_action` calls

Was explicitly _out of scope_ in the original plan 011. The samply data flips that — keeping a single `Tree` alive
across consecutive `MctsBot::get_action` calls and `move_root`-ing it down the played PV kills the per-step root-state
clone (55% of CPU today). This is the single largest lever in front of us.

**Why this is now in scope:** the baseline shows we're not paying "expansion cost on big leaves" (which 011 set out to
attack); we're paying "re-derive root state on every iter". Tree reuse hits that directly. The original plan deferred
this thinking it was orthogonal; the profile says it's _the_ fix.

**Scope:** new plan file (012 or similar) — this isn't a plan-011 cheap-win, it's a structural change to how `MctsBot`
is driven. Open questions for that plan: how to interact with `DiceMode::RegisterRolls`'s pending-roll state across
moves, what to do with the opponent's tree when control switches, and how this composes with
`recon_mcts::Tree::move_root`'s existing semantics. File-pointers for that plan: `botbowl-mcts/src/dynamics.rs:478`
(`MctsBot::get_action`), `recon_mcts/src/tree.rs` (`move_root` implementation, find it via `rg "move_root"`).

### 2. Cheap-win B — bitmask `used_skills` + `PlayerStats.skills`

Still applies. `FieldedPlayer.used_skills: HashSet<Skill>` lives on each of 22 fielded players
(`botbowl-engine/src/core/model.rs:411`). Every `GameState::clone` allocates 22 small `HashMap` tables for these;
`PlayerStats.skills` adds more.

Replace both with `u64` bitmask or `enumset::EnumSet<Skill>`. Removes ~22 heap allocs per clone. Confirmed payoff signal
in profile: `<GameState as Clone>::clone` self body is 5.0% and 1.6% self-time goes to dropping the per-pitch
`FullPitch<SmallVec<…>>` arrays. Bitmask skills don't fix that 1.6%, but they do remove the 22 HashSet allocs that show
up as part of the 55% clone inclusive.

Files: `botbowl-engine/src/core/model.rs` (struct), every site that calls `used_skills.insert/contains/clear/iter` (run
a grep first). Expected wall-clock: 1-3% on `tree.step()` — small but cheap.

### 3. Cheap-win C — Arc the `AvailableActions` box

Still applies. `available_actions: Box<AvailableActions>` (`gamestate.rs:373`) is on a hot drop path —
`drop_in_place Box<AvailableActions>` shows up in the inclusive top 30. The struct holds
`positional: Option<FullPitch<SmallVecPosAT>>` and `paths: Option<FullPitch<Option<Arc<Node>>>>`; both are deep-copied
on clone.

Switch to `Arc<AvailableActions>` with `Arc::make_mut` at the engine write sites. `available_actions` only changes when
the engine recomputes legal moves (start of activation, after pathing-altering events) — most clones become refcount
bumps.

Files: `botbowl-engine/src/core/gamestate.rs` (the field + every mutation site — `aa.insert_paths(...)`,
`state.available_actions = …`, etc.). Composable with Cheap-win B.

### 4. Cheap-win A (revised scope) — Arc the outer `FullPitch` on paths

Originally pitched as "Rc→Arc the path cache". The `Rc→Arc` part is already done (`AvailableActions.paths` is
`Option<FullPitch<Option<Arc<Node>>>>` at `model.rs:579`). What remains is wrapping the _outer_ `FullPitch` in `Arc` so
the 476 inner `Option<Arc<Node>>` cells don't get deep-copied on every clone. With paths at 0.08% inclusive in
pathing-the-call this is **lower priority than originally framed** — but the 7.9% self-time in `Cloned::next_unchecked`
is plausibly the array clone of the paths FullPitch and adjacent `[[Option<…>; 17]; 28]` boards. If samply confirms that
attribution, this win is real.

Confirm before doing the work: re-profile with `#[inline(never)]` selectively on the `FullPitch<T>::clone` impl, or
inspect the inlined frames in samply's UI to attribute the `Cloned::next_unchecked` time.

### 5. Drop plan 011's L1 — lazy paths

Don't do. Pathing is 0.08% inclusive. Engine-side `OnceCell` / `OnceLock` complexity is not justified.

### 6. Defer plan 011's L3 — progressive widening

Don't do _yet_. Useful when expansion-to-descent ratio is high (many leaves, big fan-out per expansion). The measured
regime is descent-dominated (20 expansions in 200 000 iters). Revisit after tree reuse (Step 1 above) — if reuse keeps
more of the tree alive and shifts the ratio back toward expansion, widening becomes attractive.

### 7. Defer plan 011's L2 — placeholder children

Don't do _yet_. Same reasoning as L3.

### 8. Cheap-win D (revised) — drop log Vec

Already a no-op. `MctsBot::get_action` calls `set_logging_state(false) + clear_log()` (`dynamics.rs:499-500`). Confirmed
not on the hot path. No change needed.

## Open questions / follow-ups

These are gaps the current profile doesn't resolve:

- **Why is `Tree::make_branch` 43.7% self-time?** It runs ~20 times in 200 000 iters. Even at 1 ms/call (very generous)
  that's 20 ms of 2.5 s wall-clock = 0.8%. The 43.7% figure suggests samply is attributing inlined callee time to
  `make_branch`. To check: re-profile with `-C inline-threshold=0` or inspect inlined frames in the samply UI. Until
  resolved, treat `make_branch`'s share as untrusted.
- **Where exactly does `Cloned::next_unchecked` (7.9% self) live?** Probably the `[[T; HEIGHT]; WIDTH]::clone` for
  `FullPitch`-like structures (board, paths, positional). If so, that's the lever Cheap-win A targets — confirms its
  scope.
- **Tree-reuse design questions for the new plan:** see Step 1 above.

## Success criteria (revised)

The original "≥4× expansion cost drop, ≥2× total wall-clock, ≥3× allocs/iter" targets were sized against the wrong
bottleneck. Revised criteria:

- Re-profile with `tools/samply_flatten.py` after each Step 1-4 commit. Quote inclusive % deltas in the commit message.
- Per `tree.step()` wall-clock (single-thread `expand_bench_for_samply` / `full_teams`) drops from **~9 µs → ≤4 µs**
  after tree reuse + cheap-wins B + C land.
- `MctsBot::new(1000).get_action(state)` parallel wall-clock (`expand_bench_main` / `full_teams`) drops from **~4.8 ms →
  ≤2 ms** on the same workload.
- `mcts_lifts_random_baseline` rate stays ≥0.65 (current threshold) throughout. If a change moves the rate by more than
  ±5 pp, stop and investigate before continuing.

## Pitfalls (still valid)

- **Recombination invariant.** Anything that defers state derivation must preserve "same (parent, action) → same child
  state hash". The `paths` field is not in `GameState::Hash` (confirmed at `gamestate.rs:418-496`); other lazy / Arc
  changes must keep this property.
- **Lazy-init under threading (plan 008).** `OnceCell` isn't thread-safe; use `OnceLock` or guard via the existing
  `RwLock` on `state`. (Cheap-win C must work under concurrent search.)
- **Tree-reuse safety.** `move_root` already exists in recon_mcts; check its safety contract before threading it through
  `MctsBot::get_action`. The new plan should document what it assumes about the registry, score generations, and
  pending-roll state.
- **Don't ship Cheap-wins B + C in the same commit.** Bench each independently so we can attribute the delta. Both are
  small; both can be reviewed quickly.
- **Profile after every change.** The whole premise of this plan is "measure" — defeats itself if we eyeball.

## Sequencing summary (replaces the original)

1. ✅ **Profile** (Step 1) — done. See `plans/011-baseline-results.md`.
2. **Investigate the `make_branch` 43.7% self-time** — one short profiling session with `-C inline-threshold=0`,
   document the answer in `011-baseline-results.md`. This may reframe items 3-5 below.
3. **Cheap-win B** (bitmask skills) — one small PR.
4. **Cheap-win C** (Arc the `AvailableActions`) — one small PR.
5. **Cheap-win A** (Arc the outer FullPitch on paths) — only if step 2 confirms the `Cloned::next_unchecked`
   attribution.
6. **Tree reuse across `get_action`** — split off to a new plan (likely `plans/012-…`). The biggest lever; structurally
   invasive.
7. Re-profile, then decide whether **L3** (progressive widening) is worth pursuing on the post-reuse regime.

## Out of scope (still)

- Heuristic priors beyond the 5 in `priors.rs` (idea 002/003 territory).
- Engine API redesign — pathing stays where it lives; we're not making it lazy, we're just deprioritising it.

## Files still worth reading first (unchanged from original plan)

- `botbowl-mcts/src/dynamics.rs:153-179` — `apply_action`.
- `botbowl-mcts/src/dynamics.rs:478-549` — `MctsBot::get_action` (tree-reuse target).
- `recon_mcts/src/tree.rs:1432-1497` — `Tree::new`, `step`, `step_into`.
- `recon_mcts/src/tree.rs:1596-1648` — `make_branch` (the 43.7% self-time function).
- `recon_mcts/src/tree.rs:768-783` — `Node::get_state` (the 55.8% inclusive function, root-clone driver).
- `recon_mcts/src/tree.rs:438-455` — `HashOnly::modify_state` (why root-clone happens at all).
- `botbowl-engine/src/core/gamestate.rs:370-409` — `GameState` field layout.
- `botbowl-engine/src/core/gamestate.rs:418-496` — `GameState::Hash` (recombination invariant).
- `botbowl-engine/src/core/model.rs:411` — `FieldedPlayer.used_skills` (Cheap-win B target).
- `botbowl-engine/src/core/model.rs:575-580` — `AvailableActions` layout (Cheap-win C target).
