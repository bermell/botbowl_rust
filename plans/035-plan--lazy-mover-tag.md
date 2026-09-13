# Plan 035 — Lazy mover tag: delete `peek_mover`

**Status:** Not started. Reviewed 2026-09-13 against `tree.rs` / `dynamics.rs`; corrections folded in
(see "Review notes" at the end for what changed and why).

**One line.** The child's `BbPlayer` tag is never read while the child is still a placeholder, so it
does not need to be known at enumeration time. Move its computation to `materialize_placeholder`,
where the true child state is already in hand, and `peek_mover`'s per-candidate `apply_action`
disappears — restoring what plan 016 was supposed to buy.

## Context

Plan 023 established that `available_actions` must tag each action with the mover of the *resulting*
node (recon_mcts's contract), not the current node's own mover — the old tagging was silently wrong
on exactly the transitions that matter (turnovers, rolls, follow-up decisions) and fed `select_node`'s
`home_perspective` and `backprop_scores`'s `want_max` directly.

The fix was `peek_mover` (`botbowl-mcts/src/dynamics.rs:759`): one full `state.clone()` +
`step_with_roll_or_action` + quiescent-advance loop **per candidate action**, at every expansion.
Expanding one player node therefore costs `B` full engine advances where `B` is the post-pruning fan
(routinely 30–100 mid-turn). The resulting states are then discarded and recomputed by descent.

That directly undoes plan 016. Lazy expansion exists so the engine only runs for children descent
actually picks; `peek_mover` eagerly runs it for all of them, then throws the results away. On a
60-wide fan where 5 children are ever visited that is ~12× wasted engine work plus a duplicated
`apply_action` on the 5 that were.

Plan 023's comment accepted the cost ("bot capability over performance"). It is avoidable outright.

## The observation

Every read of a node's own `player` field in `recon_mcts/src/tree.rs` happens *after* the node has a
state:

| Site | What reads it | State available? |
|---|---|---|
| `tree.rs:2206` | `hash_for(&node.player, node_state)` in `materialize_placeholder` | yes — `node_state` is the parameter |
| `tree.rs:2317` | `score_leaf` in `materialize_placeholder` | yes |
| `tree.rs:966` | `backprop_scores(&self.player, …)` | yes — node is materialised |
| `tree.rs:424`, `443` | `StateMemory::eq` (`self.player == rhs.player && self.get_state() == …`) | yes |
| `tree.rs:2014`, `2069`, `2151` | `parent_node.player` for `select_node` / `score_leaf` / `available_actions` | parent is always materialised |
| `tree.rs:1252` | `get_node_info().player` | **inspection API — can hit a placeholder.** See work item 6 |

And crucially the *children's* tags are never exposed to the parent: `select_node` (`tree.rs:2008`)
and `backprop_scores` (`tree.rs:960`) both receive `scores_and_actions` — `(q, action)` pairs. This
matches botbowl-mcts's own design, where `backprop_scores` routes chance-vs-player by the
`BbAction::Chance` variant, not by a child's player tag.

Two supporting facts:

- The illegal-action path (`step_into`, `tree.rs:1911` `None` arm) removes the placeholder from the
  parent's map without ever reading its player. `peek_mover`'s `None => BbPlayer::Chance` fallback is
  already dead code.
- `make_branch` / `create_scored_child` only fire on `Children::BranchWip`, which plan 016 made
  unreachable. They are kept compiling, not exercised.

So between `enumerate_placeholders` and `materialize_placeholder` the tag is pure dead weight, and the
information needed to compute it correctly is sitting in `materialize_placeholder`'s parameter list.

## Blocking design constraint: the mover is *not* a function of state alone

The obvious trait method — `fn player_for_state(&self, &S) -> P` — is **unimplementable for nim**, the
library's own test game: nim's `State = usize` (the pile count) and the mover alternates independently
of it (`recon_mcts/tests/nim/lib.rs:107,117`). That is also precisely why `hash_for` hashes
`(player, state)` rather than state alone.

The correct signature gives the implementor what it can plausibly need and nothing that costs the
tree bookkeeping:

```rust
/// The mover at the child node reached by taking `action` from a node
/// owned by `parent_player`. Called once, when the child is materialised.
/// Must be a pure function of its arguments (recombination invariant).
fn player_for_child(
    &self,
    parent_player: &Self::Player,
    action: &Self::Action,
    child_state: &Self::State,
) -> Self::Player;
```

- nim: `match parent_player { P1 => P2, P2 => P1 }` — ignores action and state.
- botbowl: `player_for_state(child_state)` — the free function that already exists at
  `dynamics.rs:729`, unchanged. Ignores parent and action.
- a hypothetical "pass the turn" game: reads `action`.

**`parent_state` is deliberately absent.** An earlier draft included it. Nothing needs it, and it is
the one argument `step_into` does *not* have to hand at materialisation time: `path` is
`Vec<(ArcNode, Option<A>)>` — nodes and actions, never states — and `node_state` has already been
reassigned to the child state by the time descent reaches the placeholder. Under
`MemoryMode::GetState` the parent node's stored state is `None` too (cleared by `modify_state`), so
recovering it would mean `get_state()`'s replay-from-root, which is more expensive than the
`peek_mover` being removed. Leaving it out means `path[len-2].0` (parent node → its player) and
`path.last().1` (the action) are sufficient, with no extra state threading.

This strictly generalises both the current tuple contract and the naive state-only version, and keeps
`hash_for(player, state)` meaningful.

`ActionIter::Item` then drops to bare `Self::Action`. Keeping `(P, A)` and treating the `P` as
advisory is not an option: botbowl would still have to produce one, which is the cost being removed.

## Work items

1. **`GameDynamics`** (`game_dynamics.rs:65`, and the mirrored decl at `:213`): add `player_for_child`;
   change `type ActionIter: IntoIterator<Item = Self::Action>`; update `available_actions`' doc to say
   the tree derives the mover itself.
2. **`Node.player`** (`tree.rs:768`): `P` → `OnceLock<P>`. `new_root` (`:698`) and `new_child` (`:725`)
   set it eagerly from their `player` argument; `new_placeholder` (`:758`) drops the `player` parameter
   and leaves it empty. Add `fn player(&self) -> &P` that `.expect("player read before
   materialisation")` — keep the expect in release, it is a null check on a pointer.
3. **`materialize_placeholder`** (`:2185`): take `parent_player: &P` and `action: &A` as new
   parameters. In `step_into` (`:1822`) the only caller is the `NewLeaf` arm on an unregistered node,
   which is reached exclusively from the `Branch` arm's `Some(new_state)` push — so `path.last()` is
   `(placeholder, Some(action))` and `path[len-2].0` is the parent. (The Twin arm pushes
   `(twin, None)`, but a twin is registered and never re-enters materialisation.) Set the tag with
   `get_or_init(|| GD::player_for_child(…))` **before** the `hash_for` call at `:2206` — the hash and
   the registry probe both need it, in the Twin arm as much as the miss arm. `get_or_init` not `set`:
   the miss arm is serialised by `score.write()` + the `registered` re-check, but the **Twin arm never
   registers the placeholder**, so a second worker queued on `score_wlk` re-runs the hash and probe
   and must find the same value already there rather than assert.
4. **`enumerate_placeholders`** (`:2143`): iterate bare actions.
5. **`create_scored_child`** (`:2021`) / `make_branch`: the dead `BranchWip` path still has a real
   player in hand — set the `OnceLock` eagerly there and leave the path otherwise untouched. The
   two "returned the same player/action pair twice" debug warnings (`:606`, `:836`) live here and
   become "same action twice".
6. **Every other place that constructs or reads `Node.player` directly:**
   - `lookup_state` (`:2456`) builds a registry probe `Node` with the field set inline — needs
     `OnceLock::from(player)`. `MctsBot`'s tree reuse (`dynamics.rs:2028`) goes through it.
   - `Node`'s `Debug` impl (`:1546`) — used by `panic_with_cycle_dump` on the `path`, whose last
     entry can be an unmaterialised placeholder. It already `unwrap`s `score` and would panic there
     first (pre-existing), but don't add a second one: print the raw `OnceLock` (its `Debug` renders
     uninit fine) rather than going through `player()`.
7. **`NodeInfo.player`** (`tree.rs:317`): `P` → `Option<P>`, since `get_children_info` (`:1274`)
   returns info for unmaterialised placeholders. Downstream: `BbNodeInfo` →
   `NodeStats.player` (`botbowl-mcts/src/dynamics.rs:2345`) becomes `Option<BbPlayer>`;
   `principal_variation` (`dynamics.rs:2295`) ranks children by `node.stats.player` and needs a
   `None` branch (a placeholder has no children, so any default is unobservable there);
   `botbowl-web/server/src/report.rs:48` `player_to_proto(stats.player)` needs a pending/unspecified
   enum value in `botbowl-web/proto` (engine-free, wasm-built — a proto change, small but real);
   `botbowl-web/client/src/inspector.rs:276` renders it. A placeholder already shows as
   0 visits / no Q in the inspector, so "pending" is the honest rendering, not a regression.
8. **Every `GameDynamics` impl — ten, not two.** All need the new method and the bare `ActionIter`
   item type: `recon_mcts/tests/nim/lib.rs`, `tests/nim/test_mcts_2048.rs`, `tests/navigate.rs`,
   `tests/solved.rs`, `tests/cycle_guard.rs`, `tests/deep_drop.rs`, the doc example in
   `recon_mcts/src/lib.rs`, the `BaseGD` mirror in `game_dynamics.rs:213`, `botbowl-mcts/src/dynamics.rs`,
   and **`botbowl-mcts/tests/expand_bench.rs`'s `CountingDynamics` wrapper** — which must *forward*
   `player_for_child` to its inner dynamics or T5 measures nothing.
9. **botbowl-mcts** (`dynamics.rs`): delete `peek_mover` and its `apply_action`; `available_actions`
   returns `Vec<BbAction>` for both the chance and player branches; implement `player_for_child` as
   `player_for_state(child_state)`.
10. **Docs**: `botbowl-mcts/CLAUDE.md` ("Search shape", first two bullets) and `recon_mcts/CLAUDE.md`
   both describe the `(player, action)` contract; plan 023's rationale stays valid but its
   *mechanism* is superseded — add a pointer there, don't rewrite it. `cargo fmt` in `recon_mcts/`
   is mandatory.

## Why this is a refactor, not a behaviour change

Old tag = `player_for_state(apply_action(parent_state, a))`.
New tag = `player_for_state(child_state)` where `child_state` is what descent computed via
`apply_action(parent_state, a)`.

Identical by construction, **provided `apply_action` is deterministic and pure** — which the
recombination invariant already requires of it. The hash inputs are therefore unchanged too, so node
identity, registry hits, and `deterministic_hash` iteration order are all preserved bit-for-bit. That
is what makes the exact-equality test below the right instrument: the change should be a *no-op on
search output*, and anything less than byte equality is a bug.

The three ways it could still break, and what pins each:

| Risk | Pinned by |
|---|---|
| A placeholder's tag is read before materialisation | **T1** — structural; `OnceLock` panics, it cannot be silently wrong |
| `apply_action` is not pure/deterministic | **T3** (pre-existing bug class if so) |
| Concurrency: Twin-arm re-entry re-computes the tag | **T6**; `get_or_init` makes it idempotent |

## Testing and verification

The point of this design over an analytic "predict the mover" function is that **the failure mode is
structural rather than statistical**: there is no second implementation of the engine's transition
semantics that could drift, and an unset tag panics instead of silently flipping a min into a max.
T1 is therefore the load-bearing test, and it is free.

### T1 — the tag cannot be read early (always on, release included)

`Node::player()` returns `&P` via `OnceLock::get().expect(...)`. This is not a test that has to reach
the bad state; it is a guarantee that the bad state cannot pass unnoticed in any build, in any game,
under any worker count. Add one focused unit test in `recon_mcts` that constructs a placeholder and
asserts `get_node_info().player.is_none()`, so the contract is documented rather than incidental.

### T2 — exact search-output identity over a state corpus (the main gate)

Model it on `botbowl-mcts/tests/mirror_search_exact.rs`, which already solved the reproducibility
problem: `deterministic_hash` (already enabled in `botbowl-mcts/Cargo.toml:29`), `.with_workers(1)`,
`TieBreak::Mover`, and `tier()` pinned to 16×9 so the goldens are not board-size dependent.

New test `botbowl-mcts/tests/lazy_mover_identity.rs`:

- corpus: `common::states(40, seed)` — the same `generate_random_start` harness the mirror tests use;
- axes: budgets {200, 1000} × evaluators {`Heuristic`, `Nn(tiny.onnx)`};
- per (state, budget, evaluator) record the root's chosen action plus the full sorted
  `Vec<ChildStat>` (action, visits, Q) and the root value;
- serialise to `botbowl-mcts/tests/data/lazy_mover_goldens.json`, regenerated with `BLESS=1`.

Procedure: **generate and commit the goldens on master first**, then do the refactor and require the
test to pass unchanged. If it does, the refactor provably did not move the search.

Two caveats on keeping it afterwards:

- **The goldens will churn.** Byte-exact root output over 40 states is a function of every prior,
  pruning rule and heuristic constant — exactly what the repo's current focus is changing. Kept as
  a "general regression net" it needs a `BLESS=1` on nearly every capability commit, which trains
  people to re-bless without looking. Use the full matrix as a one-shot gate for this refactor;
  if a persistent version is kept, scope it to `Heuristic` at one budget and treat a bless as a
  reviewable event.
- **`Nn(tiny.onnx)` byte equality is only reliable on one machine.** The runtime's intra-op
  threading and kernel selection can differ per host, so a committed NN golden may fail elsewhere
  through no fault of the search. Generate the before and after on the same machine in the same
  session; do not be surprised if the NN axis fails on another box.

Budget for a real possibility: a diff that is *only* a `Q = None` ordering difference among never-visited
children would be benign, but do not paper over it. Byte equality or investigate.

### T3 — `apply_action` purity (pins the equivalence argument's one assumption)

`botbowl-mcts/tests/mirror_apply_action.rs` already tests this family. Extend it: for each
`(state, legal action)` over `states(200, seed)`, apply twice from independent clones and assert the
resulting `GameState`s compare equal and yield the same `player_for_state`. Cheap, and it converts the
"provided `apply_action` is pure" clause from an argument into a test.

### T4 — nim keeps working

Nim is the generality check: its mover genuinely is not derivable from its state, so if
`player_for_child` is mis-designed nim is where it shows. `cd recon_mcts && cargo test` must stay green,
including `tests/navigate.rs` which walks the whole DAG through `get_node_info` (`:165-166`) and will
exercise the new `Option<P>`. Extend `navigate.rs` with an assertion that every *materialised* node's
reported player equals what `player_for_child` would give from its parent edge — that is the generic,
game-agnostic statement of the whole plan.

### T5 — the perf claim, measured

`botbowl-mcts/tests/expand_bench.rs` already has a `CountingDynamics` wrapper counting `apply_action`
calls (`:254`, `:279`). Record `apply_action/step` on `score_td_easy@1k` and `full_teams@1k` before and
after; the expected drop is roughly the mean post-pruning branching factor. Wall-clock backstop via
`tests/tree_shape.rs`.

Note this is the one number I have not measured — plan 023 itself says the `peek_mover` cost was
"not benchmarked", so T5's before-reading is the *first* measurement. Under `Evaluator::Nn` the two
forwards per node may still dominate and the end-to-end win could be modest. Take the before-reading
*first*; if `apply_action/step` is not already high, the plan's premise is weaker than argued and
that should be recorded rather than glossed. The premise is plausible: each `peek_mover` runs
`apply_action`'s quiescent loop, and every pass of that loop calls `sole_legal_action` →
`get_all_actions`, so it is more than one engine step per candidate.

**Take the reading from a clean tree.** The working tree currently carries an uncommitted
`roll_outcomes.rs` change (a `BlockDice` arm in `enumerate`) that changes chance-node fan. Commit or
stash it before the before-reading, or before/after are not comparable.

### T6 — multi-worker soak

Exact equality is unavailable with workers > 1 (thread-scheduling nondeterminism, as
`mirror_search_exact.rs` documents). Instead: `cargo test --workspace -- --ignored` (the bot benchmark
suite) plus a self-play batch at the production worker count, looking for panics from T1's `expect`
and for the `on_drop` assertions around the twin-swap path. Per the nondeterminism note, reproduce by
looping batches, not by replaying a seed.

### T7 — curriculum backstop

`get_the_ball_easy`, `get_the_ball_medium`, `score_td_easy`, `score_td_medium` before and after. If T2
held these should be unchanged; they run multi-worker, so treat a small drift as expected noise and a
directional change as a T2 escape worth chasing.

## What this buys — say it plainly

This is a pure performance change, and CLAUDE.md says perf work is deprioritised unless asked. The
justification has to be stated, not implied: `SearchBudget` is in iterations, and T2 proves the
search output at a fixed budget is unchanged, so **bot strength at a fixed iteration count does not
move**. The win is wall-clock only. It is worth doing if (a) self-play generation in `train_loop` is
throughput-bound on search, or (b) the saved time is spent on a higher iteration budget. T5's
before-reading decides whether either is material; if `apply_action/step` is low, park the plan.

## Open questions

- Should `hash_for` keep hashing the player now that it is derived at materialisation? **Yes, keep it**
  — nim proves the mover is genuinely independent of state, and changing it would break T2's byte
  equality for no gain.
- Worth exposing `player_for_child`'s result on `ChildStat` / the web inspector so a reviewer can see
  the tag the search actually used? Cheap, and it would have made the plan-023 bug visible. Defer
  unless work item 6 makes it nearly free.

## Review notes (2026-09-13)

What the review checked and what changed as a result. Every `player` read in `tree.rs` was verified
against the table above — it is accurate, and the illegal-action arm in `step_into` (`:1911`) really
does remove the placeholder without touching its tag. The equivalence argument stands. Corrections:

- **Signature: `parent_state` removed from `player_for_child`.** The original work item 3 claimed
  `path` carries the parent state; it carries `(ArcNode, Option<A>)`. See the design section for why
  recovering it would cost more than the plan saves.
- **Concurrency rationale corrected.** The miss arm is serialised; the re-entry that makes
  `get_or_init` necessary is the Twin arm, which never registers the placeholder.
- **Touched-surface list expanded** from two `GameDynamics` impls to ten, plus `lookup_state`, the
  `Node` `Debug` impl, `principal_variation`, and the proto enum.
- **T2 caveats added**: golden churn under capability work, and NN float determinism across hosts.
- **T5**: noted plan 023 never benchmarked the cost, and that the dirty `roll_outcomes.rs` must be
  committed or stashed before the before-reading.
- **Justification section added** — the change cannot improve strength at a fixed budget by
  construction, so the plan has to name the wall-clock consumer it serves.
