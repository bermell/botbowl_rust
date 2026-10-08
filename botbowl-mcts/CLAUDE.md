# CLAUDE.md — botbowl-mcts

Adapter between `botbowl-engine` and the `recon_mcts` search library (path dep on `../recon_mcts/`, which has its own CLAUDE.md). `BloodBowlDynamics` implements `recon_mcts::GameDynamics<State=GameState, Action=BbAction, Player=BbPlayer, Score=BbScore>`; `MctsBot` is the playable bot.

## Search shape

- Three "players": `Home`, `Away`, `Chance`. Chance nodes appear whenever `state.pending_roll.is_some()` and their children are `BbAction::Chance { result, prob_bits }`, where `result` is the concrete engine `RollResult` that `apply_action` feeds straight into `SomeProcInput::Roll` (`roll_outcomes::enumerate` picks it — there is no intermediate outcome abstraction). **Pass/fail rolls (pickup/dodge/GFI/catch — `D6PassFail`/`Sum2D6PassFail`) are first-class chance nodes in the tree (plan 018):** `available_actions` enumerates them into weighted `RollResult::Pass`+`RollResult::Fail` children, `score_leaf` returns `None` for the chance node (expanded, *not* scored), and its Q is the probability-weighted backprop of its outcomes. **Block rolls are chance nodes too (plan 036):** `roll_outcomes::block_outcomes` enumerates all `6^n` face combinations, classifies each die by its *effect* given both players' skills (Block, Dodge) and the push geometry (a push into the crowd counts as defender-removed), resolves each combination by whoever picks (attacker on `One/Two/Three`, defender on the uphill variants), and emits one child per resolved outcome with an exact probability and a representative dice array that forces that outcome through the engine. The attacker-Block-only roll showing both a knockdown-and-push die and `BothDown` is its own child (`[Pow, BothDown, ..]`): `scripted_pick` declines it, so down-in-place vs down-and-pushed is a real player decision in the tree. The **injury roll** (stunned / KO / casualty), the **pass roll** (a raw D6, one child per face) and both **foul rolls** (armour and injury, with ejection on doubles) are enumerated at their real odds too (2026-09-29; before that a broken armour was always a casualty in-tree, every pass fumbled and fouls were harmless). Faces that reach the same state are merged by `BloodBowlDynamics::merge_coinciding_outcomes`: **`recon_mcts` cannot hold two chance edges from one parent into one recombined child** (dropping the tree panics with "could not remove dropped node as child's parents", or deadlocks), so any new enumeration whose outcomes can coincide must go through it (`roll_outcomes::outcomes_may_coincide`). Only scatter/deviate/throw-in/the kickoff table still collapse to a single deterministic child (`roll_outcomes::scripted_result`). **The scripted throw-in must land in bounds *and* off the square it is thrown from** (`throw_in_outcome`): on a board whose narrow axis is ≤ 5 squares the engine divides the 2D6 by `scatter_divisor() >= 3`, so 2D6 = 2 carries 0 squares, and a 0-square model throw made every edge bounce an endless bounce → out → throw-in chain (no cycle: `bounce_squares` grows each lap), one lap deeper per descent and never valued — 14x5 searches averaged hundreds to thousands of plies (plan 060 §6). `ChanceModel::Legacy` (`BLOOD_MCTS_CHANCE=legacy`, preset `chance_model = "legacy"`) restores the old model for head-to-heads only. (Plan 018 reversed plan 010 Track A.alt's optimistic fast-forward, which over-valued risky rolls — e.g. a marked-ball pickup scored as guaranteed success.)
- **Each node's mover comes from `player_for_child`, not from `available_actions` (plan 035).** recon_mcts tags a node with the mover of *that* node, and plan 023 established that reading it off the *parent* is wrong on exactly the transitions that matter (turnovers, rolls, follow-up decisions like a push square after a `Pow`) — the tag feeds `select_node`'s `home_perspective` and `backprop_scores`'s `want_max` directly, so a wrong one silently minimises a Home decision or maximises an Away one. Plan 023's fix, `peek_mover`, got the right answer by running a full `apply_action` per candidate at every expansion and throwing the state away; plan 035 deletes it, because `recon_mcts` now hands the child state to `player_for_child` at materialisation, where `player_for_state(child_state)` is free. Byte-identical search output (`tests/lazy_mover_identity.rs` gates it), 80% fewer engine advances and 4.2x the throughput on a wide mid-turn fan.
- `available_actions` filters via `pruning::should_prune`. Block-die fan-out, coin toss and kick-receive are all collapsed by `scripted::scripted_player_pick` **inside `apply_action`'s quiescent loop** — `block_dice::scripted_pick` is its first arm, and is *not* called from `available_actions` at all (a comment there points at it, which is easy to misread). The exception it declines is the attacker's Pow-vs-BothDown choice above, which stays a real node.
  - **Consequence, measured (plans/032 item 13, `tests/block_reuse.rs`):** because the quiescent loop walks past the die choice, the search never builds a post-roll `Block` node for it — but the *engine* stops and asks the bot. So every real block-die decision arrives at a root the previous search could not have materialised: `Block` reuses the cached tree in **0.8%** of decisions against 65-100% elsewhere, and rebuilds every time. Worse, at that root the bot searches the choice and picks a **different die from the script 51%** of the time, so the tree values every future block under a policy the bot does not follow. Open question, not a settled design.
- `available_actions` is **horizon-bounded** (plan 014): once the state has moved past the root's `HorizonAnchor` (turn boundary, score change, game over, **half time**), it returns `None` and MCTS treats the state as terminal. Half time needs its own test because both turn counters reset to 0 when the second half starts; before 2026-09-29 a first-half turn-8 search ran on into the second half. `HorizonAnchor::drive_over` (game over, score, half) is also what makes a leaf a known outcome for the NN evaluators: half time scores exactly 0, as the corpus labels it.
- Selection uses PUCT with `prior_for(state, action)` priors (`priors.rs`, ~5 multipliers — see plan 004). **`PUCT_C = 10.0` (`dynamics.rs`) is tuned against the leaf-score magnitudes in `score.rs` and they are coupled — changing one without the other silently degrades search.** Unexplored children get **FPU = the parent's Q** (from the descending player's perspective), not `Q = 0` — `leaf_score`'s ~+520 constant offset (ball control) otherwise buries unexplored children ~500 points below any explored sibling and starves wide move fans of exploration. Plan 032 #3 adds an optional **FPU reduction** (`BLOOD_MCTS_FPU_REDUCTION=k`, `MctsBot::with_fpu_reduction`, `eval --fpu-reduction/--vs-fpu-reduction`): `parent_Q − k·√(visited prior share)`, Leela/KataGo form; `k = 0` (default) is the shipped plain FPU.
- **`PUCT_C` has never been re-tuned for NN priors.** `Evaluator::Nn` priors are softmax×`len` (`botbowl-nn/src/eval.rs`), so the mean matches the scripted `BASE = 1.0` but the spread does not: a confident action among 60 legal ones gets `P ≈ 30` against the scripted maximum of 10, and the tail gets `P ≈ 0` and is never revisited after its FPU visit. Plan 026's `c` sweep used the heuristic evaluator. Open item in plan 032.
- Scores stay **Home-centric end-to-end** (plan 006). Player-node backup is **the visit-weighted mean of child Q** (`backprop_scores`), the AlphaZero backup, computed in integers so it stays mirror-exact. It is hardcoded, not a knob (2026-10-05, plan 055): we tried minimax and it didn't work with a learned leaf. A max over noisy NN values picks the children the net overrates, a turn stacks many same-side max nodes, plan 031 D1 measured +0.10 of search-added optimism, and in exp064 the minimax Gumbel search lost to its own bare policy (0.482) where the mean won (0.526). Home nodes maximise in `select_node` (PUCT); Away nodes mirror via `home_perspective`. Visits sum across children on both Player and Chance branches. `backprop_scores` routes Chance (probability-weighted expectation) vs Player (visit-weighted mean) by the **children's action variant** (`BbAction::Chance`), not `score_current` — so it works even though a chance node is unscored (`score_leaf → None`).
- `score_leaf` is the **value function** (heuristic `leaf_score` now, NN later) and is evaluated **only at player-decision and terminal nodes** (plan 018). Pending-roll (chance) states return `None` — never scored — **unless the state is past the horizon or game-over**: those never get roll-outcome children, so they must carry a score or they become backprop dead ends. (Every in-search touchdown is exactly such a state — TD → next kickoff → pending Deviate with the score change past the horizon; leaving it unscored makes TDs invisible to the search.) Mid-procedure states (no roll, no `team`, not game-over) are `available_actions → None` so `recon_mcts` marks them terminal; they are the other unexpandable exception still given `leaf_score`.
- `recon_mcts` materialises children **lazily** (plan 016): each new child is a cheap placeholder until descent picks it.
- `recon_mcts` marks exhausted subtrees **solved** (terminal, or all children solved): solved children leave the selectable set — which also means terminal children stop accruing visit bumps and virtual loss — and `MctsBot`'s workers *and* wall-clock timer stop as soon as the whole tree is solved (`tests/solved_early_stop.rs` pins the early return). Visit counts are **not** "descents through this node": backprop sets them to the children's sum, so they double-count recombined subtrees and lag cut-off descents (see `MctsBot::get_action` below and `trace_root_descents`).

## MctsBot::get_action

Clones the root state, sets `DiceMode::RegisterRolls`, force-disables logging and clears the log Vec (otherwise each `apply_action` clone re-copies the whole log), splits `iterations_per_move` across `std::thread::scope` workers (plan 008), and picks the root child with the **best aggregated Q from the agent's perspective** (visits as tie-break; unscored children rank last). Most-visited is *not* used: descents ending on already-terminal nodes bump visit counters without adding information, so raw visit counts over-weight whichever path saturated first. The tree is **cached and reused** across consecutive `get_action` calls when the horizon anchor matches (plan 015 Step 1). **Two-bot game loops call `release_stale_tree(&state)` on both bots after every move** (`generate::mcts_vs_mcts_samples`, `eval::play_ladder_game`, via `release_stale_tree_of` for a `dyn Bot`): once the anchor has moved (turn or score), the next search would discard the tree anyway, so it is freed now instead of being carried through the opponent's whole turn. That halves the live trees per game, and search output and reuse telemetry are unchanged (`tests/stale_tree_release.rs`; a freed tree still reports as the `AnchorMiss` it anticipated). Workers apply a transient virtual-loss penalty on descent (plan 015 Step 5) to diverge under concurrency. **Each descent takes its own penalty back when it ends** (`recon_mcts`'s `GameDynamics::release_descent`, called over the descent's whole path after its backprop), and backprop carries the in-flight penalty over to a node's replacement score rather than resetting it. Before 2026-09-30 the penalty was cleared only when a backprop happened to replace the chosen child's score; a descent cut off below it (a chance node still withholding its value) left it behind, it piled up, and it buried the best children — **at one worker too**, so every corpus up to then was searched with it, and gen21 at virtual loss 0 beat itself at 30 by **0.686** over 400 games (plan 049 headline). At one worker virtual loss is now exactly inert (`tests/virtual_loss_inert.rs`; the golden at 30 is byte-identical to the old search at 0). At 8 workers it still spreads descents widely, by design, and interacts with the chance nodes' withheld values: a child pushed into by virtual loss looks fine until its chance subtree completes. **Tree visits are not descents**: backprop sets a node's visits to its children's sum, so a subtree recombined under several parents counts for each; `MctsConfig::trace_root_descents` / `SearchSummary::root_descents` counts real per-child descents at the root.

## Forced decisions are not searched

When exactly one action survives pruning (`pruning::search_actions`, the set a search root expands,
with its all-actions fallback when pruning would leave none), `MctsBot` plays it **without a search**
— `forced_action(state)`, checked first in `MctsBot::unsearched_pick`, so it covers `get_action`,
`get_action_with_record` and `get_action_explore`, under PUCT and the Gumbel root alike. No tree is
built, no descent runs, the net is never asked, the telemetry records no search, and `last_search()`
is `None`. The cached tree is left alone, so the next real decision can still reuse it. Inside the
tree the quiescent walk (`sole_legal_action`) already stepped past such states; only the root
searched them, a whole budget to rubber-stamp the one move.

The record is the scripted shape (`unsearched_sample`): `scripted = true`, the one post-pruning
action as the **only** child with one visit, `root_visits = 1`, no `Q`, `root_value = None`. The
corpus keeps one sample per decision (`override_audit::replay_to` and `mc_label` replay every
`chosen_action`), and `prepare` skips samples with fewer than two children, so a forced decision
never reaches training. `tests/forced_move.rs` pins all of it.

## Budget mode: iterations or visits

`MctsConfig.budget_mode` (`BLOOD_MCTS_BUDGET=visits`, preset key `budget_mode = "visits"`,
`MctsBot::with_budget_mode`) says what `SearchBudget::Iterations(n)` counts. `Iterations` (the
default and every corpus so far) runs `n` new descents per decision *on top of* a reused tree, so
the decisions that inherit the most get searched the most. `Visits` stops once the root has `n`
visits and never runs more than `n` descents, so only a reused tree gets cheaper. **A fresh tree
costs the same under both:** only ~3 in 4 descents raised the root's visit count in the test
position (300 descents → 234 root visits; recombination is the likely cause, not yet traced), so a
fresh tree hits the `n`-descent cap before it reaches `n` visits. `SearchTelemetry.iterations` counts the descents actually run,
so `iterations / searches` in a report is the real per-decision budget. `tests/budget_mode.rs`.

## Gumbel root search (`gumbel.rs`, plan 053)

`MctsConfig.gumbel_m > 0` (`BLOOD_MCTS_GUMBEL_M`, preset `gumbel_m`, e.g. `cfgs/gumbel16_iters.toml`)
replaces the PUCT root with sequential halving:
- **Considered set:** the top `m` root moves by `g + ln prior`. `g` is Gumbel noise at
  `gumbel_scale`; 0 is deterministic play.
- **Schedule:** the budget is split evenly over `ceil(log2 m)` phases. The better half survives
  each phase by `g + logit + σ(q̂)`, until two remain. The best survivor is played.
- **Below the root:** plain PUCT.
- **Mechanics:** `run_gumbel` drives `tree.step()` on one thread and names each descent's root
  move in a `ForcedRoot` that `select_node` takes at the root. A descent's first player-node
  selection is always the root's.
- **Solved moves:** they need no descents and are dropped from the phase. When every survivor is
  solved, the search stops early.
- **Budget:** a Gumbel search always counts descents (`budget_mode` is ignored) and ignores
  `workers`.
- **The training record is unchanged:** every root child's visits, Q and prior.
- **Exploration under Gumbel:** `get_action_explore` ignores plan 048's Dirichlet noise and
  visit-sampled moves. The Gumbel noise (`gumbel_scale`) is the exploration, and the move is the
  best survivor.
- **`gumbel_q_floor`** floors q̂'s normalising range (Q points). 0 is the paper's rule, which
  follows noise on our flat Q (exp060). 1000 is the measured setting (`cfgs/gumbel16_f1000*.toml`).
- **Default:** `gumbel_m = 0` is byte-identical to the shipped search. `tests/gumbel_root.rs` pins
  that, the schedule and reproducibility.

## Self-play exploration (`exploration.rs`, plan 048)

`MctsBot::get_action_explore(state, ExploreStep)` is `get_action_with_record` plus two optional
generation-only knobs; `ExploreStep::default()` is exactly the plain record (`tests/root_noise.rs`).

- **Root Dirichlet noise** (`RootNoiseSpec { epsilon, alpha, seed }`): `run_search` puts an
  `Arc<RootNoise>` holding the root state on `BloodBowlDynamics.root_noise`, and
  `available_actions` mixes `ε·S·Dir(α/n)` into the priors **only when the state equals that
  root** — a per-search constant like the horizon anchor, so recombination stays pure. It takes
  effect only on a **fresh** tree: a reused tree keeps the dynamics it was built with and expanded
  this root long ago as a non-root. `RootNoise::applied` / `ExploreOutcome.noised` say which. On
  gen21 at 500 visits that is ~39% of decisions (the rest reuse).
- **The sample keeps the clean priors.** The cq target reads `ln prior`, so the noisy prior must not
  reach the corpus; the root expansion stores the pre-noise priors and the record uses those.
- **Move sampling** (`SampleSpec { temperature, u }`): play a root child ∝ `visits^(1/T)` instead of
  best-Q; `Sample.chosen_action` is the move actually played.
- The heuristic search is Q-dominated at `PUCT_C = 10`: a clear best move keeps its visits under
  any prior, so noise often leaves a heuristic root's visits unchanged. NN priors (softmax × n)
  have far more spread.

## Evaluator: Heuristic vs NN (plan 017)

`BloodBowlDynamics.evaluator: Evaluator` selects the value/prior source. `Evaluator::Heuristic` (the `#[default]`) is the scripted baseline and reproduces prior behaviour **byte-identically** — `available_actions` calls `prior_for_engine_action` per action and `score_leaf` calls `leaf_score`. `Evaluator::Nn(Arc<NnEvaluator>)` (in `botbowl-nn`) swaps in a frozen ONNX net (tract, pure-Rust CPU): `available_actions` does **one** `nn.priors(state, &filtered)` forward and zips the results into `BbAction::player` (NN priors **replace** scripted priors — there is no principled common scale to blend them), and `score_leaf` calls `nn.value_home_i64(state)` — **except for known-outcome leaves**: when the score changed since the `HorizonAnchor` or the game is over, `score_leaf` returns the exact `anchor.score_delta(state).clamp(-1,1) * 1000` instead of asking the net. The outcome is proven there (and gets frozen into solved subtrees as exact minimax), and post-TD kickoff states are out-of-distribution for a net trained only on decision states — an NN guess would re-open the "TDs invisible to the search" failure. Δ-since-anchor (not absolute `leaf_score`) keeps these leaves on the NN's drive-relative scale. Wire it via `MctsBot::new(..).with_evaluator(Arc<NnEvaluator>)`. It is **one** forward per expanded node, not two: since e9ccabd `score_leaf` calls `value_home_i64_prefetch_policy` and the priors call is served from that same pass (`LAST_FORWARD`). Live gen10 measured 319 forwards per 500-iteration decision (plan 046).

- **Purity invariant holds:** a frozen, deterministic CPU network is a pure function of state, so recombination stays sound (same as the pruning/prior invariant below).
- **Scale bridges (calibration, coupled to `PUCT_C`/`leaf_score`):** the NN value is mover-centric in `[-1,1]`; `value_home_i64` clamps, sign-flips to Home-centric via `perspective::mover_for`, and rescales **×1000** to match `leaf_score`'s TD = ±1000. NN priors are a Rust softmax over gathered per-legal-action logits, rescaled **×`legal.len()`** so the mean prior ≈ 1.0 (the un-normalised `BASE = 1.0` scale). `BloodBowlDynamics` is therefore **not `Copy`** (the `Arc` in `Evaluator::Nn`); it's `Clone`d once per search in `run_search`.
- All encoding lives in `botbowl-nn` (`encode.rs`, shared with the offline prepare step → no train/inference skew). The tract parity test (`botbowl-nn/tests/parity.rs`) pins tract == PyTorch at two board sizes.

## Memory mode — HashOnly is forbidden

`MctsBot.memory_mode` is always `MemoryMode::StoreState` in production. **GOTCHA:** `recon_mcts`'s `HashOnly` marker is *broken* for Blood Bowl — a `GameState` is large enough that hash collisions are inevitable, and `HashOnly` merges any two colliding states into one DAG node, producing illegal actions mid-search, corrupted backprop, and drop-time panics. The variant has been removed from `MemoryMode`; only `StoreState` (default, structural O(1) equality) and `GetState` (safe replay-based diagnostic) remain. Never reach for `recon_mcts::HashOnly` when wiring a Blood Bowl tree (plan 013).

## MctsConfig — one struct, resolved once at construction

Every knob that shapes a search lives in `MctsConfig` (`dynamics.rs`): `workers`, `memory_mode`, `tree_reuse`, `virtual_loss`, `puct`, `tie_break`, `fpu_reduction`, `horizon_turns`, `horizon`, `stats`, `leaf_stats`, `debug_root`, `budget_mode`, `chance_model`, and the plan-047 setup knobs `setup`, `opponent_setup`, `setup_formation`, `setup_budget_scale`, `setup_horizon_turns` (see "Kickoff setup" below). `MctsBot::new` uses `MctsConfig::from_env()`, so every CLI path keeps its `BLOOD_MCTS_*` A/B knobs; `MctsBot::with_budget_and_config(budget, MctsConfig::new())` builds a bot that ignores the environment entirely, which is what lets one process (the web server, plan 034) run several differently-tuned bots.

**Plan 043 makes it a committable preset.** `MctsConfig` is `Copy` + serde, and its serde default is `MctsConfig::new` — **not** `Default`, which is `from_env()`: a named configuration that absorbed a stray `BLOOD_MCTS_*` would not be reproducible, which is the entire point of naming one. `deny_unknown_fields` turns a typo into an error rather than a knob that silently stays put. Enum variants are `snake_case` on the wire so a preset reads in the same vocabulary as the CLI flags; renaming them is safe because they cross the hub under postcard (variant-by-index) and no persisted JSON names them. `cfgs/` holds the presets, `botbowl_play::bots::load_mcts_config` loads one, and `SearchConfig.config` carries it — see `cfgs/README.md`.

**Env vars are read once, at `::new`** — setting one between building a bot and calling it no longer does anything. Before plan 034, `BLOOD_MCTS_HORIZON`, `_WORKERS` and `_MEMORY` were re-read inside *every* `get_action` and therefore **overrode** an explicit `with_workers(...)`; the builders now actually win.

`BLOOD_MCTS_MEMORY={get|store}` (`hash` panics — see above), `BLOOD_MCTS_WORKERS=N`, `BLOOD_MCTS_HORIZON=off`, `BLOOD_MCTS_HORIZON_TURNS=N`, `BLOOD_MCTS_TREE_REUSE=off`, `BLOOD_MCTS_VIRTUAL_LOSS=N`, `BLOOD_MCTS_PUCT_MODE`/`_C`/`_RANGE_FLOOR`, `BLOOD_MCTS_TIE_BREAK`, `BLOOD_MCTS_FPU_REDUCTION=k`, `BLOOD_MCTS_BUDGET=visits`, `BLOOD_MCTS_CHANCE=legacy`, `BLOOD_MCTS_SETUP={search|formation}`, `BLOOD_MCTS_OPPONENT_SETUP={search|formation}`, `BLOOD_MCTS_SETUP_FORMATION={line|spread|wedge|zone|random}`, `BLOOD_MCTS_SETUP_BUDGET_SCALE=f`, `BLOOD_MCTS_SETUP_HORIZON_TURNS=N`, `BLOOD_MCTS_STATS=1`, `BLOOD_MCTS_LEAF_STATS=1`, `BLOOD_MCTS_DEBUG_ROOT=1` (dump top-10 root children by visits/Q after each search — first thing to reach for when the bot plays nonsense; all-zero Q means backprop is broken).

## Kickoff setup (plan 047)

The engine's setup is one `PlacePlayer`/`BenchPlayer` decision per player (`botbowl-engine/CLAUDE.md`). What the bot does with it:

- **Own setup — `MctsConfig::setup`.** `search` (the `auto` resolution for a network evaluator): each placement is an ordinary root. The search runs through the rest of its own placements, the *opponent's* setup (see next point), the kickoff and — for the kicker — the receiver's first turn; the horizon stops it where the agent's own turn counter advances, so the receiver's setup search stops at the start of its first turn and takes the value net's opinion of the post-kickoff position (`setup_horizon_turns = 2` lets it search that turn). `formation` (the `auto` resolution for `Heuristic`/`PureTd`, which cannot value a half-built setup): the placement is answered from a formation plan (`setup_formation`, `random` draws one of the fitting formations per drive from the bot's seeded RNG) **without searching**, and `get_action_with_record` emits a *scripted* sample — every legal action as a child, one visit on the one played, `Sample::scripted = true` so `prepare` keeps it despite `root_visits = 1` and `targets.rs` makes it one-hot. That is the gen-0 teacher shard (`cfgs/setup-teacher.toml`). A setup root with a single action left after pruning (the 3-player boards put everyone on the line) is a forced decision like any other (see "Forced decisions are not searched"): no search, and a one-child record that `prepare` skips.
- **Opponent setup — `MctsConfig::opponent_setup`.** Inside the tree the other team's placements are answered by `Formation::Line` in `apply_action`'s quiescent loop (`BloodBowlDynamics::opponent_setup_pick`, keyed on `agent_team`), so a kicker's search reaches the kickoff through a plausible opposing setup instead of eleven more decision levels whose leaves are half-built setups the value net has never seen. It is a modelling assumption — the opponent will not set up in a line — and `search` turns it off. Constant for a search, so still a pure function of state for recombination.
- **Budget — `setup_budget_scale`.** A setup root runs `budget × scale` (`SearchBudget::scaled`). Early placements are mostly prior-driven; a fraction keeps generation cheap.
- **Horizon.** `HorizonAnchor` now also carries the half: `Half` zeroes both turn counters at the second half, so a late-first-half root used to search through the half-time setups and kickoff until a score. A half change is terminal.


## Search telemetry (`telemetry.rs`, plan 043)

Two things the bot does every decision that used to be invisible. Both are **always on** — plain
`u64`s on `MctsBot`, written on the owning thread in `run_search` before any worker spawns, so a
process running several differently-tuned bots (the web server) gets one tally per bot rather than
a global it cannot attribute.

- **`ReuseOutcome`** classifies the tree-reuse attempt into `Reused | Disabled | NoCache |
  AnchorMiss | MarkerMiss | LookupMiss | NoPath`, recorded per decision with the top proc name and
  the pruned action fan. `AnchorMiss` is the expected one (turn boundary / score); `LookupMiss` and
  `NoPath` are the ones worth chasing. Broken down by procedure in `TreeReuseStats::by_proc`.
- **`RecombinationCounts`** mirrors `recon_mcts::RecombinationStats` (which cannot derive serde —
  that crate is std-only by design). Per search it is a **delta**, read from the tree the search
  actually ran on. **Read the baseline after the reuse attempt resolves**, never before: a lookup
  miss probes the cached tree and then searches a brand-new one, and differencing those saturates
  to zero and drops the decision's whole cost. `tests/tree_reuse_stats.rs` pins it.

`MctsBot::telemetry()` reads the totals; `take_telemetry()` drains them, which is how eval makes a
per-game record mean "this game" without differencing maps. `telemetry_of` / `take_telemetry_of` /
`last_search_of` reach them through a `dyn Bot` via the engine's `Bot::as_any` hook. Every field is
a commutative counter, so `merge` folds games, threads and worker machines in any order.
`BLOOD_MCTS_STATS=1` adds a grep-able `MCTS_TELEMETRY …` line.

**What it measured first time out**, and what came of each (heuristic, 16x9):

- **Recombination cost ~8 state comparisons per registry probe**, 82% rejected, ~85% of them
  between states whose full 64-bit hashes agreed — because `GameState::hash` hashed only
  `proc_stack.len()` and `proc_stack_top()`. **Fixed** in plan 044: colliding states 36.8% -> 0%,
  comparisons per probe 4.07 -> 0.12, search 6-9% faster with byte-identical output. The
  `eq_hash_equal` counter is what separated real hash collisions from hashbrown's 7-bit tag
  brushes, and it is still the way to tell them apart.
- **Tree reuse ~48-54% overall**, but wildly uneven by procedure: `FollowUp`/`DodgeProc`/`GfiProc`
  100%, `Push` 78%, `MoveAction` 65%, **`Block` 0.8%**. The `Block` figure is structural, not bad
  luck — `botbowl-mcts/tests/block_reuse.rs` explains it and plans/032 item 13 carries the open
  question it exposed (the bot overrules its own block-die script half the time). See the
  block-die note under "Search shape".

## Tree statistics (`tree_stats.rs`, plan 060)

Every search records how deep its descents went and where they ended, **per descent** (not per
node). `recon_mcts`'s `GameDynamics::observe_descent` hands each finished descent's edges (with
the player of the node each leaves), its leaf state and a `DescentEnd` to
`BloodBowlDynamics::observe_descent`, which folds them into the bot's `DescentLog` (one mutex per
descent; shared with every tree the bot builds, like `ForcedRoot`, because a reused tree keeps the
dynamics it was built with). After the search `run_search` adds the main line (most-visited child
from the root down, read with `Node::get_children`/`with_score`, no state clones) and
`opp_turn_follows` (the `Half` procedure's turn order: is the next turn start before the horizon
the opponent's, with `TURNS_PER_HALF` per team). The result, `botbowl_data::TreeStats`, goes on the
sample (`Sample.tree`) and into `SearchTelemetry.tree` (`TreeTelemetry`, summed counters; the
`MCTS_TELEMETRY` line prints the means), from there into `report.json`, `eval.games.jsonl` and the
corpus `meta.extra` (`tree_*` keys).

- **Phases** (`leaf_phase`) are relative to the root's `HorizonAnchor`, captured even with the
  horizon off: game over, score, half end, then the turn counters — the mover's counter advanced
  `turn_depth` times is the horizon, the opponent's counter ahead of the mover's is `opp_turn`.
- **Valuation**: `solved` (the descent hit an all-solved node), `horizon`, `terminal` (a known
  outcome, or an in-window dead end), `chance` (a fresh in-window chance node) or `new_leaf`.
- **Observational**: the search plays byte-identical games (`scripts/perf_search_bench.sh` strips
  the `tree` block before hashing) at the same instruction count (27.041e9 → 27.040e9).
- `scripts/tree_stats.py CORPUS...` prints plan 060 §3's tables. `tests/tree_stats.rs` pins the
  totals (one phase and one valuation per descent), a turn-end search whose main line enters the
  opponent's turn, the last turn of a half (no opponent turn), and `opp_turn_follows` against
  random games.

## Reading a finished search (`report.rs`)

`MctsBot` keeps the tree it just searched (it already did, for reuse) and exposes it read-only:
`last_search() -> Option<&SearchSummary>` (root + children + timing + evaluator value),
`principal_variation(depth)`, and `explore(&[BbAction], with_state)` which walks **one level at a
time** into the cached DAG via `recon_mcts`'s `Node::get_children_info` / `Node::get_child` /
`Tree::get_root_node`. Inspection is inert — no descent, no visit bump.

- **Q is Home-centric on the wire and agent-centric in `q_agent`.** One frame for a whole
  read-out: signing each node by its *own* player makes a PV flip sign at every ply, so a root at
  `+0.27` whose best child reads `-0.27` looks like the bot picked the worst move.
- **There is exactly one tree — the most recent search's.** Every `get_action` re-roots or rebuilds
  it, so anything holding a path from an earlier search must notice (plan 034's `search_id`).
- Visits are "descents through this node", cumulative across a reused tree and frozen once a
  subtree is solved: a search-effort measure, not a move-quality one.

## Invariant (also stated at repo root)

Pruning rules (`src/pruning.rs`) and priors (`src/priors.rs`) **must be pure functions of `(state, action)`** — recombination depends on it. Two paths to the same logical state that return different action subsets will silently split the DAG.
