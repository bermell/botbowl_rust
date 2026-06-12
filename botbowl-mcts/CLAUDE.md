# CLAUDE.md — botbowl-mcts

Adapter between `botbowl-engine` and the `recon_mcts` search library (path dep on `../recon_mcts/`, which has its own CLAUDE.md). `BloodBowlDynamics` implements `recon_mcts::GameDynamics<State=GameState, Action=BbAction, Player=BbPlayer, Score=BbScore>`; `MctsBot` is the playable bot.

## Search shape

- Three "players": `Home`, `Away`, `Chance`. Chance nodes appear whenever `state.pending_roll.is_some()` and their children are `BbAction::Chance { outcome, prob_bits }`. In production the chance path is collapsed inside `score_leaf` / `apply_action` so chance children rarely enter the tree (plan 010 Track A.alt).
- `available_actions` filters via `pruning::should_prune`. `block_dice::scripted_pick` collapses block-die fan-out to a single scripted choice when the engine offers one, and `scripted::scripted_player_pick` collapses coin toss / kick-receive inside `apply_action`'s quiescent loop.
- `available_actions` is **horizon-bounded** (plan 014): once the state has moved past the root's `HorizonAnchor` (turn boundary, score change, game over), it returns `None` and MCTS treats the state as terminal.
- Selection uses PUCT with `prior_for(state, action)` priors (`priors.rs`, ~5 multipliers — see plan 004). **`PUCT_C = 10.0` (`dynamics.rs`) is tuned against the leaf-score magnitudes in `score.rs` and they are coupled — changing one without the other silently degrades search.**
- Scores stay **Home-centric end-to-end** (plan 006). Home nodes maximise in both `select_node` (PUCT) and `backprop_scores`; Away nodes mirror via `home_perspective`. Visits sum across children on both Player and Chance branches.
- `score_leaf` "fast-forwards" (`optimistic_leaf_score`) through up to `ff_depth` mid-procedure micro-steps to reach a decision/terminal state, picking the `Pass`/`Advance` outcome each time, so a chance leaf is scored with its optimistic outcome instead of the pre-roll position.
- `recon_mcts` materialises children **lazily** (plan 016): each new child is a cheap placeholder until descent picks it.

## MctsBot::get_action

Clones the root state, sets `DiceMode::RegisterRolls`, force-disables logging and clears the log Vec (otherwise each `apply_action` clone re-copies the whole log), splits `iterations_per_move` across `std::thread::scope` workers (plan 008), and picks the most-visited root child. The tree is **cached and reused** across consecutive `get_action` calls when the horizon anchor matches (plan 015 Step 1). Workers apply a transient virtual-loss penalty on descent (plan 015 Step 5) to diverge under concurrency.

## Memory mode — HashOnly is forbidden

`MctsBot.memory_mode` is always `MemoryMode::StoreState` in production. **GOTCHA:** `recon_mcts`'s `HashOnly` marker is *broken* for Blood Bowl — a `GameState` is large enough that hash collisions are inevitable, and `HashOnly` merges any two colliding states into one DAG node, producing illegal actions mid-search, corrupted backprop, and drop-time panics. The variant has been removed from `MemoryMode`; only `StoreState` (default, structural O(1) equality) and `GetState` (safe replay-based diagnostic) remain. Never reach for `recon_mcts::HashOnly` when wiring a Blood Bowl tree (plan 013).

## Env knobs for A/B and debugging

Read once per `get_action`: `BLOOD_MCTS_MEMORY={get|store}` (`hash` panics — see above), `BLOOD_MCTS_WORKERS=N`, `BLOOD_MCTS_HORIZON=off`, `BLOOD_MCTS_TREE_REUSE=off`, `BLOOD_MCTS_VIRTUAL_LOSS=N`, `BLOOD_MCTS_STATS=1`.

## Invariant (also stated at repo root)

Pruning rules (`src/pruning.rs`) and priors (`src/priors.rs`) **must be pure functions of `(state, action)`** — recombination depends on it. Two paths to the same logical state that return different action subsets will silently split the DAG.
