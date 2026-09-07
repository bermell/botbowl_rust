# Idea: stop scoring intermediate states — only evaluate decision / terminal nodes

**Status:** implemented 2026-06-14 (minimal scope). Captures a design direction discussed 2026-06-12.

**Outcome:** Step 1 concluded **no `recon_mcts` change was needed** — all chance-node machinery
(`select_node` probability descent, `backprop_scores` expectation, `available_actions` outcome
enumeration, `apply_action` roll resolution) already existed on the botbowl side. The only collapse
was the optimistic fast-forward in `score_leaf`. Fix: `score_leaf` returns `None` for pending-roll
states (chance nodes are *expanded, not scored*; value derives from children's probability-weighted
backprop), and `backprop_scores` now detects chance by child-action variant rather than the
(now-`None`) `score_current.node_kind`. Removed `optimistic_leaf_score` + the dead `ff_depth`
plumbing + the unused `is_pass_fail` helper. **Deferred:** scatter/bounce/throw-in/block still
collapse to a single deterministic `Advance` child (`roll_outcomes::enumerate`) — proper weighted
multi-outcome fan-out (with a branching policy) is the open question below, not done.

## Problem

`score_leaf` (botbowl-mcts `dynamics.rs`) currently assigns a value to _any_ leaf state, including a pending dice roll.
For those it calls `optimistic_leaf_score`, which **fast-forwards through the pending roll assuming success**
(`ChanceOutcome::Pass` / `Advance`) and then scores the resulting board (plan 010 "Track A.alt", which deliberately
collapses chance so chance children stay out of the tree).

This has two distinct costs:

1. **It over-values risky outcomes (a live bug).** In `GetTheBallMedium` the ball is marked by an adjacent opponent, so
   a real pickup auto-fails — but the optimistic FF scores "move onto the ball → pickup" as a _guaranteed_ success (≈
   +500 Home ball-control). The search is lured into grabbing the marked ball every time → turnover → 0% win rate.
   (Diagnosed 2026-06-12; the scripted bot solves the same scenario at 98.6%.)
2. **It is conceptually wrong for the AlphaZero endgame (the real reason to fix it).** When we replace the heuristic
   with a learned value head, we do **not** want to train the network to evaluate intermediate states. A value for "the
   board mid-bounce, before the dice land" or "between two squares of a Move" isn't a well-defined target and isn't a
   state the agent ever _acts_ from. The network should only ever see — and be trained on — states where a player must
   make a decision, or terminal/game-over states.

## Desired invariant

> The value function (heuristic today, NN later) is evaluated **only** at **player-decision nodes** (a player must
> choose an action) and **terminal / game-over nodes**. Chance nodes are never assigned a standalone learned value; the
> search reaches the next decision/terminal frontier _through_ them.

This is the standard shape for AlphaZero-style search on a **stochastic** game: the value net lives at decision/terminal
states; chance is handled by expectation over outcomes, not by a learned scalar on the chance state itself.

## Design

### Chance nodes as first-class tree nodes (recommended end state)

Reverse Track A.alt: let chance nodes live in the tree (their children are the weighted roll outcomes —
`available_actions` already enumerates these for pending-roll states). recon_mcts only ever invokes the value model on
decision/terminal leaves; a chance leaf is **expanded, not scored**, and its value is derived from its children's
backprop (visit/probability weighted).

- **Pros:** principled; chance handled by search (more visits → better expectation, variance-aware); recombination works
  across chance; matches how AlphaZero-for-stochastic is actually done; the NN value head is structurally prevented from
  ever seeing a chance/intermediate state.
- **Cons:** the genuinely intrusive option. Reverses a perf optimization (more nodes per tree) — acceptable now:
  CLAUDE.md marks perf deprioritized, capability is the focus.

Step 1 is to pin down what recon_mcts actually requires:

### Step 1 — recon_mcts investigation (do before committing to the change)

- Does recon*mcts call `score_leaf` on **every** new leaf, including chance nodes? If so, what we return there is only
  an \_initial estimate* that gets corrected once the node's children are expanded — in which case "don't score chance"
  means "return a cheap neutral placeholder and never the NN," not a library change.
- How does `select_node` treat a **chance node's** children? At a chance node we must descend by **probability** (or by
  max-uncertainty), **not** PUCT. Verify the chance-selection path is correct and exercised (production currently
  collapses chance, so it may be under-tested).
- Confirm `backprop_scores` already does probability/visit-weighted aggregation across Chance children (CLAUDE.md says
  visits sum across Chance branches — verify the _score_ aggregation is a proper expectation, not a max/sum).

### Step 2 — botbowl-mcts changes

- Stop calling `optimistic_leaf_score`. `score_leaf` branches purely on node type: terminal/game-over → true drive
  outcome; decision → value model (heuristic → NN); chance/mid-proc → neutral placeholder, never the value model.
- Stop collapsing chance in `apply_action`/`score_leaf` so chance nodes enter the tree.

### Step 3 — validate

- `GetTheBallMedium` MCTS win rate recovers toward the ~0.35 the test expects (regression confirmed pre-existing at 0.0
  on 2026-06-12).
- No `micro_step` legality panics; recombination rate sane (`BLOOD_MCTS_STATS=1`).

## Open questions / risks

- **Chance branching factor.** Bounce/scatter/throw-in fan out to 8 dirs; a chain of them is a wide subtree. Need a
  policy: full expansion vs. top-k outcomes by probability vs. progressive widening. (B makes this a search-budget
  question rather than a hidden truncation in `score_leaf`.)
- **PUCT_C coupling.** Leaf-score magnitudes and `PUCT_C` are coupled (CLAUDE.md). Removing the optimistic inflation
  changes the Q magnitudes the constant was tuned against — re-check together.
- **Cross-cutting with priors.** Even with correct expectation, flat priors (every `Start*` = 1.0) leave the search
  under-guided on the multi-activation block→pickup plan. A block-on-marker / pickup prior (priors.rs) is a separate
  lever (plan 004 follow-up) that compounds with this fix. Also fix the pruning-empty fallback that explodes
  post-activation nodes to 150–253 actions (`available_actions` safety-net, dynamics.rs:272) — it dilutes the budget.
- **Perf.** More nodes per tree. Bounded by the horizon (plan 014). Measure, but don't pre-optimize.
