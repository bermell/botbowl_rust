# Plan 006 — Adversarial backprop in `BloodBowlDynamics`

**Priority:** #1 in v4. The single largest correctness gap in the bot today.

## Why this matters

`backprop_scores` in `botbowl-mcts/src/dynamics.rs` currently does `max` over children for _every_ player node,
regardless of whether the node is a Home or Away turn. From Home's perspective, Away should be _minimising_ — but right
now it co-operates with Home. The doc-comment at `dynamics.rs:274` admits it ("we treat Away the same as Home
(single-player optimisation view)").

Concrete symptom: lectures with an active opposing turn (the `GetTheBall_*` suite) sit at 0% success while `ScoreTd_*`
clears the bar — the latter is single-player-shaped, the former isn't. Plausibly the dominant cause.

## Files to read first

- `botbowl_rust/botbowl-mcts/src/dynamics.rs`
  - `backprop_scores` impl, ~lines 228-291.
  - `select_node`, lines 160-226 — it must agree with the new backprop direction.
  - `BbScore` struct, lines 40-54 — does it need a "perspective" field, or do we rely on `node_kind`?
- `botbowl_rust/botbowl-mcts/src/score.rs` — `leaf_score` is _already_ signed from Home's perspective (positive = Home
  good). So leaves don't need flipping; only the aggregator does.
- `recon_mcts/src/tree.rs` lines 911-1100 (`backprop_scores` call site) — verify what `_player` argument is passed in
  (it's the _current node's_ player). Plan hinges on that.
- `recon_mcts/tests/nim/test_mcts_2048.rs` — the 2048 reference is single-player, so it's not a model here. Check
  `recon_mcts/src/game_dynamics.rs` doc comments for the two-player intent; the `two_player` feature flag may already
  imply a convention we should follow.
- `botbowl_rust/plans/005-learnings--mcts-chance-nodes.md` §"Things deliberately left alone" — confirms this was the
  next logical step.

## Questions to investigate

1. **What does `_player` mean inside `backprop_scores`?** Is it the player whose move _leads into_ this node (the
   parent's perspective) or the player who owns this node (decides at it)? Search recon_mcts for the call site.
2. **Is the score sign-convention Home-centric or current-player-centric?** Once we have an answer: simplest design is
   "scores stay Home-centric everywhere, Home nodes max, Away nodes min". Alternative: "scores are always
   current-player-centric, negate on edge crossings". The first is less error-prone for a 2-player zero-sum game.
3. **What about `Chance` nodes?** They already do a probability-weighted average — that's perspective-agnostic, so they
   need no change.
4. **Does `select_node` need to mirror the flip?** PUCT's Q term is currently `q + c·P·√N/(1+n)`. If Away nodes hold a
   Home-centric Q, Away's selector needs to pick `min Q + exploration` (or equivalently negate Q before adding
   exploration). Otherwise the bot's opponent model plays _for_ Home.
5. **`score_leaf`** at terminal / mid-tree leaves — does it need any per-player adjustment? It calls `leaf_score` which
   is already Home-centric. Confirm no double-flip when combined with the new backprop.
6. **Does `MctsBot::get_action` need to know it's playing as Home or Away?** Today it's hardcoded Home-centric. The bot
   is wired via `bot_factory.rs` and could be either side; either generalise (`MctsBot { my_team: TeamType }`) _or_
   mirror everything against `state.available_actions.team` at the root.

## Proposed approach

Recommended: **scores stay Home-centric end-to-end**, change only:

1. `select_node` — for player nodes:
   - if `parent_node_state.available_actions.team == Some(Home)` → maximise Q.
   - if `Away` → minimise Q (or maximise `-Q`).
   - exploration term unchanged (sign-symmetric).
2. `backprop_scores` — for player nodes, branch on the perspective owned by the node (need to capture that on
   `BbScore.node_kind` — already there as `BbPlayer::Home / Away / Chance`):
   - Home → `max` (current behaviour).
   - Away → `min`.
3. `BbScore.visits` aggregation — orthogonal; tackled in plan 007. For now keep whatever convention plan 007 settles on.
4. `MctsBot::get_action`: at the root, the bot is making a move for _the team whose turn it is in the engine state_. If
   `state.available_actions.team` says Away, the bot is the Away agent — and the root selector must pick the action that
   is _best for Away_, i.e. minimises Home-centric Q. Decide whether to keep the bot Home-centric and only invert at the
   root, or generalise. Keeping Home-centric is one branch; generalising adds a field.

## Tests / success criteria

- Add a unit test for `backprop_scores`: build a small tree of fake `BbScore`s, call backprop, assert Home node returns
  max and Away node returns min.
- Add a unit test for `select_node` mirroring: identical Q values across two children but opposing sign — verify Home
  picks +Q, Away picks -Q.
- Existing `ScoreTd_*` lectures must still pass at their thresholds (0.80 / 0.50). These are effectively single-player
  and shouldn't regress.
- Un-`#[ignore]` `get_the_ball_easy.rs` / `get_the_ball_medium.rs` and run them — expect rates to lift off 0%, even if
  they don't yet clear the thresholds. Record the measured rates in the commit message.

## Pitfalls

- **Double-negation.** If `score_leaf` ever returns from current-player perspective and backprop also flips, the sign
  cancels and nothing changes. Sanity check: a 1-deep tree where Away has one winning child should backprop a negative
  root Q.
- **`parent_player` arg shadowing.** `select_node` takes `_parent_player` — currently unused. It might already give us
  what we need without inspecting `parent_node_state`.
- **Recombination.** A DAG node may have both Home and Away parents (probably rare in Blood Bowl, but possible across
  half/turn boundaries). The node's _own_ `node_kind` is well-defined; use that for backprop direction rather than
  parent.
- **Don't touch chance backprop.** It's correct as-is.

## Out of scope

- Tree reuse across moves (separate concern).
- Concurrency (plan 008).
- Re-tuning `PUCT_C` for the new Q distribution — only if a regression appears.
