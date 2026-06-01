# Plan 010 — Fast-forward mid-procedure states & make chance nodes visible

**Priority:** #5 in v4 — _after_ 006, 007, 008, 009 are stable. This is the hard one. It's the v3 work that didn't land;
the goal here is to make it landable.

## Why this matters

`plans/005-learnings--mcts-chance-nodes.md` documents v3's blocker: the engine runs `Move(target)` one square at a time
per `micro_step`. When the move _needs_ a roll (pickup, dodge, GFI), `micro_step` exits mid-procedure with
`pending_roll = None` and `available_actions.team = None`. recon_mcts sees no children → treats the node as
`Children::None` terminal → MCTS can't descend through pickup at all.

The v3 attempt — a FF loop that micro-steps until decision / roll / game-over — exposed two problems:

1. **Per-iter cost rose ~10000×** (1µs → ~10ms). Not the FF loop itself (1.0 step/call average); something compounds in
   tree traversal. Best guess: deeper trees with chance↔player alternation that don't recombine enough.
2. **`recon_mcts::Node::on_drop` panics** at `tree.rs:780:66`, `Option::unwrap() on a None value`. The drop walks
   parents via `Node::get_state`, which re-applies actions via `GD::apply_action` and unwraps the result. Our
   `apply_action` returns `None` for some recombined edge during teardown.

Until this lands, `GetTheBall_*` lectures sit at 0% because the bot can't reason about pickup moves.

## Files to read first

- `botbowl_rust/plans/005-learnings--mcts-chance-nodes.md` lines 89-133 — the section "Things that did NOT work (and
  why)". _Read in full._
- `botbowl_rust/botbowl-mcts/src/dynamics.rs`
  - `apply_action`, lines 132-158 — note the deliberate "no FF in v3" comment.
  - `score_leaf` + `optimistic_leaf_score`, lines 293-344 — current workaround.
- `botbowl_rust/botbowl-engine/src/core/gamestate.rs` — `micro_step` impl. Confirm exit conditions (decision / roll /
  game-over).
- `recon_mcts/src/tree.rs`
  - The `Drop` impl for `Node`, around the line referenced by the panic (search for `on_drop` and `get_state`).
  - `step_into`, lines 1457-1490, especially the `Children::None` branch (~line 1489).
- `recon_mcts/src/lib.rs` — DAG / recombination invariants in the crate doc.

## Questions to investigate

### Track A: reproduce + fix the drop panic

1. **Build a minimal reproducer** with FF re-enabled. Smallest case: `score_td_easy.rs`-shaped test driving an MCTS
   search that includes at least one pickup-or-equivalent. Confirm the panic at `tree.rs:780:66`.
2. **Why does `apply_action` return `None` during `Drop`?** Likely answers:
   - The recombined parent state isn't actually reachable from the recorded action — i.e. recombination has joined two
     states that _look_ equal by hash but aren't actually equivalent for the engine. Hashing bug or `should_prune`
     impurity.
   - The action references a player ID / position that no longer exists in that state.
   - Engine error path that's flaky under specific configurations.
3. **Options at the recon_mcts side:**
   - Patch `Node::Drop` to be iterative and tolerant of `apply_action -> None` (early-return without re-deriving state;
     the node is being torn down anyway). This is a `recon_mcts` change, and the recon_mcts crate is in this same
     workspace tree.
   - Avoid the drop walk entirely by not putting chance children into the tree.
4. **Track A.alt — never let chance children into the tree:**
   - Do the FF _inside_ `score_leaf` only.
   - Return a forward-looking score for the leaf.
   - Tree's child state stays mid-procedure; MCTS won't descend further into that branch. Lose some search depth past
     chance points, but bypass the drop bug entirely.

### Track B: investigate the 10000× slowdown

1. **Profile a 1000-iter `MctsBot::get_action` with FF enabled.** Use `cargo flamegraph` (or `samply`) on a release
   build.
2. **Hypothesis A — clone cost.** Did the v2 `log` fix get reverted? Are there other Vecs being cloned per step? Check
   `apply_action` clones.
3. **Hypothesis B — recombination failing.** Hash-equal states are split because something impure (priors? pruning?
   RNG?) makes the same `(parent, action)` produce different children. Track `tree.rs` hit/miss counter (`reg_info.hits`
   / `misses`, lines 1555 / 1569) — a low hit rate under FF would confirm.
4. **Hypothesis C — chance branching factor explosion.** Add a guard around pickup chance children only and see if the
   slowdown localises.

### Track C: re-enable probability-weighted chance selection

Documented in 005 §"Success-first + probability-weighted chance `select_node`". This was reverted in v3 because chance
nodes weren't reachable. Once FF works:

1. Reinstate the success-first + `prob × (total + 1) - visits` ratio in the chance branch of `select_node`.
2. Verify it doesn't reintroduce the drop panic.

## Proposed approach

Sequenced, not parallel:

1. **Reproduce the drop panic in isolation.** Smallest possible test. Confirm exact line / unwrap site.
2. **Decide track A vs A.alt** based on what the reproducer reveals. If `apply_action -> None` is itself a bug we can
   fix (e.g. an action that references a stale player ID — sanitise the action on store), do that. If it's structural,
   patch recon_mcts's Drop to be tolerant.
3. **Address the slowdown.** Independent of the panic fix; should be done on the same FF reproducer with profiling
   tools. Don't merge FF until per-iter cost is back to <100µs (target: <10µs).
4. **Re-enable the success-first chance selector** as a follow-up commit.
5. **Un-`#[ignore]` `GetTheBall*` tests** and target the thresholds in those files (0.70 easy, 0.40 medium).

## Tests / success criteria

- New regression unit test that triggers the previously-panicking drop path (whatever shape ends up reproducing it).
- `MctsBot::get_action(state)` with 1000 iters completes in `<100ms` for a ScoreTdEasy-shaped state with FF enabled.
  Soft target.
- `GetTheBallEasy` passes at ≥0.70.
- `GetTheBallMedium` passes at ≥0.40.
- `ScoreTd_*` remain ≥0.80 / ≥0.50.

## Pitfalls

- **Determinism.** FF + RNG-driven micro_step would explode the recombination invariant. Every step inside the FF loop
  must consume fixed dice or skip rolls (`pending_roll` exits the loop). Re-audit `fix_for_outcome`'s coverage; plan 009
  must land first.
- **Stack depth.** A FF that recurses into engine procedures may push deep. Iterative FF only —
  `while state.needs_step() { state.micro_step(None)?; }`.
- **Don't combine with concurrency for the same commit.** Plan 008 + this is exactly the v3 combo that compounded. Land
  FF on the single-threaded bot first, then re-test under threads.
- **`recon_mcts` changes count as touching a separate project.** Two repos, two commits — see CLAUDE.md §Conventions.

## Out of scope

- Pathfinding-aware priors (idea 003 §"force pass / handoff / blitz path constraints"). Separate ticket.
- Tree reuse across calls.
- Opponent modelling tuning (plan 006 is the perspective fix; this plan is visibility, not policy).
