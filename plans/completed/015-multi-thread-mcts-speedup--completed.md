# Plan 015 — Multi-thread MCTS speedup (completed / partial)

**Status:** Closed out as partially landed. Performance work is now deprioritized in favour of
bot capability.

**What landed:**

- **Step 0 — chance-node selector** (`BloodBowlDynamics::select_node`): switched from
  `min_by(visits)` to `argmax(p_i · (N_parent + 1) - N_i)` so empirical visit ratio converges
  to the real probability distribution.
- **Step 1 — tree reuse across `get_action`** (`MctsBot.cached_tree` + `last_anchor`): cached
  tree is walked to the new root via `Tree::lookup_state` + `find_path_to` + `apply_action`.
  Reuse gates on `HorizonAnchor` equality; anchor change ⇒ rebuild. `BLOOD_MCTS_TREE_REUSE=off`
  disables.
- **Step 5 — virtual loss** (`BbScore.virtual_loss`, `MctsBot.virtual_loss`): transient penalty
  applied on descent in `select_node`, subtracted in `puct_value`, reset to 0 on backprop.
  Default magnitude 30; `BLOOD_MCTS_VIRTUAL_LOSS` overrides.

**What did not land** (deferred indefinitely with the broader perf pause):

- Step 2 (`GD::score_leaf` outside `registry.write()`) — turned out to already be the case on
  re-reading; no change needed.
- Step 3 (`update_score` write-lock contention reduction).
- Step 4 (`Tree::select_node` lock-chain audit / debug assertion).

The benchmark numbers below were the pre-Step-1 baseline (single-thread 9.7 s @ 10k iters /
20k iters etc.). They were never re-captured after Steps 0/1/5 landed; treat them as historical.

## Context

With the deadlock fix from plan 013's 2026-06-01 update committed, `parallel_bench` runs cleanly at 10 workers up
through 20 000 iters. But the measured speedup over single-thread is modest:

|  iters | trials | serial (1 worker) | parallel (10 workers) | speedup |
| -----: | -----: | ----------------: | --------------------: | ------: |
| 10 000 |      2 |           9.708 s |               7.826 s |   1.24× |
| 20 000 |      2 |          18.585 s |               9.049 s |   2.05× |

Why this matters: the curriculum experiments (`score_td`, `get_the_ball`, plus the unfixed `get_the_ball_easy` /
`_medium` regressions) are bottlenecked by MCTS wall-clock. Faster multi-thread search means more iters per move at the
same wall-clock budget, which directly raises the experimentation ceiling for downstream work (priors, leaf scoring,
horizon, NN guidance). A 5× speedup turns a 4-minute lecture run into 48 seconds; a 2× speedup turns it into 2 minutes.

The deadlock is gone. The remaining gap to ~5× is **contention** within the parallel search, and **wasted work** at the
inter-`get_action` boundary (the bot rebuilds its tree from scratch every move). The two are independent levers and
both are addressed below.

## Where the contention is — current understanding

Confirmed from the post-fix sample profile and a re-read of `recon_mcts/src/tree.rs`:

1. **`registry.write()` held across `GD::score_leaf`** — `Tree::create_scored_child` (`tree.rs:1601-1654`) acquires
   `self.registry.write()` and only drops it after `Node::connect_child`, `Node::register`, the optional
   `set_min_depth`, **and** `GD::score_leaf` have all run. `BloodBowlDynamics::score_leaf` invokes
   `optimistic_leaf_score`, which walks engine `micro_step`s — measured at tens of µs in plan 011. Ten workers serialise
   on this single global lock for tens of µs per leaf expansion. This alone caps the speedup to roughly _useful work per
   iter / time spent under registry-lock_, which on this codebase is ~2-3×.

2. **Backprop serialisation on the shared upper DAG.** Every leaf expansion calls `Node::backprop_scores`, which walks
   up to all reachable ancestors. Near the root the DAG is shared — every worker's leaf converges on the same ~5-10 top
   nodes. Each ancestor's `Node::update_score` does `score.read` → drop → `score.write` (`tree.rs:870-880`). Rust's
   queue-based fair RwLock serialises those writes; readers (other workers descending through the same ancestors via
   `Tree::select_node`) queue behind the writer. The plan-013 deadlock fix removed the _cycle_; the _contention_
   remains.

3. **All workers descend the same hot PUCT path at low budgets.** At iters=1000 split across 10 workers, each worker
   gets only ~100 expansions. The tree is small enough that PUCT consistently picks the same top-of-tree action, so all
   workers thrash on the same `parent.children.read` / `child.score.read` near the root. Larger budgets help (which is
   why 20k iters → 2.05× vs 10k → 1.24×) because the tree grows enough for workers to diverge into different subtrees.

4. **Per-step useful work is tiny.** Plan 011 measured ~2.37 µs / step single-thread. A few hundred ns of lock-acquire
   overhead per descent step is already a meaningful fraction of useful work, so Amdahl-style serial-fraction goes up
   fast as contention rises.

Items 1 and 2 are structural — they are inherent to the current `recon_mcts` shape and the dynamics layer's
`score_leaf`. Items 3 and 4 are workload-shape effects that improve with budget.

## Plan

Roughly in order of expected impact-per-engineering-cost. Stop once we hit a satisfactory speedup or until profiling
says the bottleneck has moved.

### Step 0 — Chance-node selector (landed)

Independent of throughput, the previous `min_by(visits)` chance-node selector in `BloodBowlDynamics::select_node`
(`dynamics.rs:288-307`) ignored `BbAction::Chance::prob_bits` and over-sampled low-probability outcomes. Replaced with
`argmax_i (p_i · (N_parent + 1) - N_i)` — picks the outcome most under-represented relative to its probability. Empirical
visit ratio now converges to the real probability distribution as N grows. `score_td_easy` still passes at 0.96. This is
a correctness fix that happened to be small enough to land alongside this plan; it is not a throughput change.

### Step 1 — Tree reuse across `get_action` calls

Highest-leverage change for end-to-end lecture wall-clock. Today `MctsBot::get_action` (`dynamics.rs:597`) builds a
fresh `Tree` per call and discards it. With a ~50-move trial at 1000 iters/move, that's 50× of search work thrown
away. Reuse would let later moves benefit from the earlier moves' sub-tree.

**Navigation** — how to find the new state in the old tree:

1. Hash the new `state` (the parameter passed to `get_action`). Look it up in `Tree.registry`. If absent, bail and
   rebuild from scratch — the bot or the engine took a path the search never visited.
2. If present, walk upward via `Node.parents` (each parent edge stores the action that produced this child) until we
   reach the current root. Record the chain of actions in reverse.
3. Replay the chain via `Tree::apply_action` — recon_mcts's existing one-step `move_root` (`tree.rs:785-830`) handles
   per-step parent cleanup and prune-lock acquisition. The DAG can have multiple paths from root to the found node;
   any one of them is fine because `move_root` correctly prunes the unchosen ones.

This avoids needing to know the exact game-engine trace (bot action + opponent's full turn + chance resolutions) —
we just need the new state in the tree, and the parent edges tell us how to get there.

**Horizon coupling — decided by anchor equality.** Plan 014 pins a `HorizonAnchor` per `get_action` call
(`dynamics.rs:86-92`: `{ agent_team, home_turn, away_turn, home_score, away_score }`, derived from the root state).
The anchor only changes at specific boundaries: a team scores, the game ends, or the bot's turn counter advances —
i.e. when the previous turn boundary has been crossed. **Within a single bot turn**, the anchor is identical across
`get_action` calls (the bot makes multiple decisions per turn, but no scoring/turn-counter change happens between
them), so every score in the surviving subtree is still computed under the same anchor and is still valid. **At a
turn boundary**, the anchor changes and the surviving subtree's Qs are stale.

So the recipe on each `get_action`:

1. Compute `new_anchor = HorizonAnchor::capture(state, agent_team)`.
2. Compare to the previous call's anchor (stored on `MctsBot`):
   - **Anchor unchanged** → walk to the new root via the registry-lookup-and-replay scheme above. Keep all scores
     intact. This is the cheap and common case (every move within a single bot turn).
   - **Anchor changed** → same walk, but after re-rooting reset every surviving node's `Node.score` to `None` and
     `score_gen` to 0. Structure (registry, `Node.state` cache, `Node.children` action sets / pruning work) is still
     preserved; only the Q values rebuild.
   - **State not in registry** → bail and rebuild the tree from scratch as today.

The "drop scores on anchor change" branch is the floor of correctness; the "keep scores on anchor match" branch is
the upside. We need both to land tree-reuse safely. `HorizonAnchor` already derives `Copy + Clone + Debug` so adding
`PartialEq + Eq` is a one-line change (or just compare fields directly).

If we later want to keep scores even across an anchor change (the old plan-1b idea — store horizon-offset on each
node and recompute Q on read), it stays available as a follow-up. Not needed for first cut.

**Concrete shape of the change:**

- `MctsBot` gains two persistent fields: `tree: Option<Arc<Tree<...>>>` and `last_anchor: Option<HorizonAnchor>`.
- On each `get_action`:
  1. `new_anchor = HorizonAnchor::capture(state, agent_team)`.
  2. If `tree.is_some()` and `Tree::lookup_state(state, &root_player).is_some()`: walk parents to collect the action
     chain, replay via `Tree::apply_action`. If anchor changed vs `last_anchor`, reset scores in the surviving
     subtree; otherwise keep them.
  3. Otherwise: build a fresh `Tree` as today.
  4. Run the iter loop. Store `tree` + `new_anchor` for the next call.
- Add an env knob (`BLOOD_MCTS_TREE_REUSE=off`) to A/B the reuse path against today's fresh-tree behaviour, so we can
  measure speedup directly and catch any silent regressions in lecture rates.

**Recon_mcts surface needed:**

- Public accessor for `Tree.registry` and `Node.parents` (or a helper like `Tree::find_path_to(state_hash) -> Option<Vec<A>>`).
  Today `registry` is `RwLock<HashSet<WeakNode>>` private to the crate. Cleanest is a `Tree::lookup_state(&S, &P) -> Option<&ArcNode<...>>`
  and a `Node::action_to_parent() -> Option<A>` helper exposed behind `pub(crate)` → `pub`.
- `Tree::apply_action` is already accessible via the `SearchTree` trait.

**Risks / caveats:**

- The new state may exist in the tree but not on a descendant of the current root (e.g. the bot's previous move
  pruned to a different subtree). The walk-up-parents must terminate at the current root specifically; if the walk
  passes the root without hitting it, treat as a miss and rebuild.
- If the new state is the current root itself (unusual but possible when the engine made no externally-visible
  change between calls), no replay is needed.
- Determinism / reproducibility: today every `get_action` starts from a fresh RNG state on the Tree. With reuse,
  the prior search's RNG advances carry over. Acceptable for production but worth flagging in lecture-test seed
  comments.

**Expected speedup:** unclear without measurement, but a plausible ballpark is 1.5-3× on lecture wall-clock, on top
of any multi-thread speedup. The closer to 1×, the more it suggests the horizon shift is invalidating too much of
the reused work and 1b is needed.

### Step 2 — Move `GD::score_leaf` out of the `registry.write()` critical section

Highest-leverage change. In `recon_mcts/src/tree.rs:1601-1654`:

```rust
fn create_scored_child(&self, parent_node, player, action, state) {
    let node = Node::new_child(parent_node, player, state);
    let mut reg_wlk = self.registry.write().unwrap();
    match reg_wlk.get(&ArcNode::downgrade(&node)) {
        Some(existing_node) => { /* connect, set_min_depth, drop reg_wlk */ }
        None => {
            let mut score_wlk = node.score.write().unwrap();
            Node::connect_child(parent_node, action, &node);
            Node::register(&node, Some(&mut reg_wlk));
            drop(reg_wlk);
            // ↓↓↓ score_leaf runs HERE — still expensive, but now outside reg_wlk
            *score_wlk = GD::score_leaf(...);
            drop(score_wlk);
            <Node<...> as StateMemory>::modify_state(&node.state);
        }
    }
}
```

Today `score_leaf` runs inside the `None` arm, after `drop(reg_wlk)` — re-reading the code, it _is_ already outside the
registry lock. **Verify this on the actual line numbers** before committing to it; if true, this step is already done
and Step 1 collapses into Step 2. (Plan-013's sample profile didn't show registry contention as the top hit, which is
consistent with this.) If it _isn't_ done, hoist `score_leaf` after `drop(reg_wlk)` and only hold `score_wlk` during its
execution.

Either way, file a quick perf sample (`samply` or `sample` for ≥5 s at iters=20 000, 10 workers) to confirm where the
top wait actually is post-deadlock-fix. The sample will tell us whether Step 1 has any work left or whether the
bottleneck is squarely Step 2's territory.

### Step 3 — Reduce `update_score` write-lock contention on shared upper-DAG ancestors

Two complementary tactics:

**2a — Skip backprop when score didn't change.** `Node::update_score` already returns a bool indicating whether the
score changed; `backprop_scores` (`tree.rs:935-969`) walks all parents regardless. Skip enqueuing a parent for further
backprop when its score didn't change. Confirm whether this is already the behaviour — the `update_score` return-bool
guards the `n_updates += 1` and the parent push at `tree.rs:957`. If so, Step 2a is already in place and we move to 2b.

**2b — Coalesce the read+write on `Node.score` into a single `compare_exchange`-style update.** Today `update_score`
does:

```rust
let score_cur_rlk = self.score.read().expect("no score");
let score_new = GD::backprop_scores(..., score_cur_rlk.as_ref(), scores_and_actions);
drop(score_cur_rlk);
if let Some(score) = score_new {
    let mut score_wlk = self.score.write().expect("no score");
    // ... write, increment score_gen ...
}
```

This is a read → drop → write sequence on the same lock; under contention the write queues, blocking subsequent reads.
Alternative shape: take `score.write()` once for the whole compute-and-store (paying a slightly longer critical section
but avoiding the queue-flip). Or: replace `RwLock<Option<BbScore>>` with an `AtomicU64`-packed `(score, visits)` and
CAS-update — the score is small (`i64 score` + `u32 visits` + enum kind), and the score-derivation function is pure on
its inputs, so a CAS retry loop is well-defined. This is bigger but removes the lock from the hot path entirely.

For 2b, start with the smaller change (single `write()` instead of read→drop→write) and measure. Only escalate to the
CAS-packed shape if speedup is still <3×.

### Step 4 — Audit `Tree::select_node`'s `child.score.read()` chain

The plan-013 fix removed the chain inside `BloodBowlDynamics::select_node`'s comparator. But `Tree::select_node` itself
(`recon_mcts/src/tree.rs:1559-1577`) hands the dynamics a lazy iterator whose items are
`lockref::Ref<RwLockReadGuard<...>>`. If the dynamics impl is well-behaved, only one Ref is held at a time. Add a
debug-build assertion (or a `#[cfg(debug_assertions)]` Drop check on `lockref::Ref`) that flags any code path which
holds two Refs simultaneously, so future GD impls don't re-introduce the deadlock cycle.

This is a small, defensive change — file it alongside Step 3.

### Step 5 — Workload-shape mitigations (only if 1-4 leave us below 5×)

These are bigger and only worth it if Steps 1-4 plateau:

- **Virtual loss** on selected nodes (apply a transient negative score to the path under exploration so other workers
  diverge to other subtrees). Standard MCTS technique; pairs poorly with recombination unless the virtual-loss decay is
  correctly per-edge rather than per-node.
- **Per-worker subtree shards with periodic merge.** Each worker grows its own tree for K iters, then merge into the
  global registry. Loses recombination during a shard window; reduces lock contention sharply.

Skip these unless the profile says we're past the easy wins.

## Verification

For each step, the gate is:

1. `cargo test --workspace --release` is green (no semantic regression).
2. `parallel_bench` at iters=10 000 and iters=20 000, 10 workers, 2 trials, seed `0xCAFE_1234`. Record serial / parallel
   wall-clock and the speedup. Before this plan: 1.24× at 10k, 2.05× at 20k.
3. Single-thread `parallel_bench` (`with_workers(1)`) shouldn't regress more than ~5 % — no perf borrowed from the
   serial path.
4. `score_td_easy` and `score_td_medium` lecture rates stay ≥ 0.90 (current measured: 0.96 / 0.96).

Stretch goal after Step 1 + 3: **3× at 20 000 iters / 10 workers** on `parallel_bench` AND **≥2× lecture wall-clock**
improvement on `score_td_easy` from tree reuse. Aim goal after Step 5: **5× parallel_bench**.

Tree-reuse-specific gates (Step 1):

- `BLOOD_MCTS_TREE_REUSE=off` reproduces today's behaviour exactly.
- `BLOOD_MCTS_TREE_REUSE=on` (the default after Step 1): `score_td_easy` rate stays ≥ 0.90 (today 0.96).
- Lecture trial wall-clock improves measurably (target ≥ 1.5× over today on `score_td_easy` 50-trial run).

The `get_the_ball_easy` / `_medium` lecture failures are out of scope for this plan — they're pre-existing (plan-013
update confirms 0.08 at HEAD~1 for `easy`) and orthogonal to throughput.

## Critical files

- `botbowl-mcts/src/dynamics.rs` — `MctsBot` struct (`:552`) gets the persistent `Option<Arc<Tree<...>>>`;
  `get_action` (`:597`) gets the lookup-and-replay branch. `BloodBowlDynamics::score_leaf` cost (line 407-413:
  `optimistic_leaf_score` walks engine `micro_step`s) determines how much Step 2 buys us.
- `recon_mcts/src/tree.rs` — Step 1 needs a public state lookup (e.g. `Tree::lookup_state(&S, &P) -> Option<ArcNode>`)
  and a way to read `Node.parents` from outside the crate. Steps 2-3 touch `create_scored_child` (`:1589-1655`),
  `Node::update_score` (`:832-933`), `Node::backprop_scores` (`:935-969`), `Tree::select_node` (`:1552-1587`).
  Edits here cross the project boundary — separate commit in the `recon_mcts` repo per `CLAUDE.md`.
- `botbowl-mcts/tests/parallel_bench.rs` — measurement target. May want a `PARALLEL_WORKERS` env knob alongside the
  existing `PARALLEL_ITERS` / `PARALLEL_TRIALS`. Note: `parallel_bench` re-creates the bot per trial — tree-reuse
  benefits a single trial across moves, not across trials, so the `score_td_easy.rs`-style 50-trial lecture suite
  is the more honest measurement target for Step 1's speedup.

## Out of scope

- Get the Ball lecture failures (plan TBD — orthogonal: heuristic / priors / FF problem, not throughput).
- Plan 012 Step A (`set_min_depth` pre-check) — separate concern, only matters at much higher budgets.
- Changing the public `GameDynamics` API surface.
