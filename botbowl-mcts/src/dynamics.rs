//! `GameDynamics` implementation that plugs Blood Bowl into recon_mcts,
//! plus the MctsBot adapter that exposes the searcher as a `Bot`.
//!
//! Design mirrors the 2048 reference (`recon_mcts/tests/nim/test_mcts_2048.rs`):
//! - `Player` discriminator carries the kind of node (`Home`/`Away` action
//!   nodes vs. `Chance` outcome nodes).
//! - `Score` is a small struct with atomic visit counter, integer score,
//!   and the node-kind discriminator that `backprop_scores` switches on.
//! - `available_actions` returns engine actions for player nodes, and
//!   `RollResult`-carrying chance actions for chance nodes (when
//!   `state.pending_roll` is `Some`) via `roll_outcomes::enumerate`.
//! - `apply_action` resumes the engine with `SomeProcInput::Action` for
//!   player actions and `SomeProcInput::Roll(result)` for chance
//!   actions, feeding the stored `RollResult` straight in.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use botbowl_data::{ChildStat, Sample};
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, SomeProcInput, TeamType};
use botbowl_nn::eval::NnEvaluator;
use recon_mcts::{GameDynamics, GetState, NodeInfo, SearchTree, SelectNodeState, Status, StoreState, Tree, TreeAlias};

use crate::action::{BbAction, BbPlayer};
use crate::priors::prior_for_engine_action;
use crate::pruning::should_prune;
use crate::roll_outcomes;
use crate::score::leaf_score;
use crate::scripted;

/// PUCT exploration constant. Sized so that the `c · P · √N(parent) /
/// (1 + N(a))` term is comparable to leaf-score magnitudes (game score
/// ±1000, ball control ±50, carrier-distance ±26 — see `score.rs`).
/// 10.0 keeps unexplored high-prior children competitive with explored
/// children of moderate `Q` for at least the first few hundred parent
/// visits — verified empirically against the existing UCT baseline test.
const PUCT_C: f32 = 10.0;

/// Default `c` for [`PuctMode::NormalisedQ`]. In the normalised frame the
/// child-Q spread is 1.0 by construction and priors sum to 1, so `c` is
/// AlphaZero's `c_puct` up to the range convention (`[0,1]` here vs
/// `[-1,1]` there). Seeded from the Raw-equivalence identity
/// `c_norm = PUCT_C * sum(p) / D`, which at a median node
/// (`sum(p)` ~ 22, `D` ~ 200) gives ~1.1 — measure before trusting.
const DEFAULT_PUCT_C_NORMALISED: f32 = 1.0;

/// Lower bound on the per-node Q range used as the normalisation
/// denominator. Two jobs:
///
/// 1. **Divide-by-zero guard** — `BbScore.score` is `i64`, so a real
///    range is either 0 or >= 1; the floor is the only thing that ever
///    applies below 1.
/// 2. **Anti-amplification** — without it, a node whose children differ
///    only by `score.rs`'s carrier-distance tier (±26) would have a few
///    points of positional drift stretched across the full decision
///    range, making meaningless differences look decisive. 50 is one
///    ball-control step (`score::ball_control_value` × 10) — the
///    smallest difference that reflects a change of game situation
///    rather than drift.
const DEFAULT_Q_RANGE_FLOOR: f32 = 50.0;

/// Raw leaf-score points corresponding to one full normalised range,
/// used to express `virtual_loss` in the normalised frame. Chosen so the
/// shipped default (30, [`DEFAULT_VIRTUAL_LOSS`]) costs 0.1 of a node's
/// decision range whatever that range's raw size. `0` still disables
/// exactly. This is the one constant here with no measurement behind it.
const NORM_VL_REFERENCE: f32 = 300.0;

/// Selection-rule variant. `Raw` is the historical formula and the
/// default; `NormalisedQ` maps sibling Q into `[0,1]` and priors onto the
/// simplex before adding the exploration bonus, so `c` means the same
/// thing in a node whose children span 6 points as in one spanning 1058
/// (the measured p10..max spread of the top-two Q gap — plan 025).
///
/// **Selection-only.** `score_leaf`, `backprop_scores`, `ChildStat.q` and
/// `Sample::root_value` all stay in raw Home-centric leaf-score units, so
/// training targets are untouched. Do not normalise backprop: that would
/// break the horizon carve-out and the NN value bridge at once.
///
/// `c` lives inside each variant so "normalised mode still carrying the
/// Raw constant" is unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PuctMode {
    /// Historical behaviour: `Q_raw + c * P * sqrt(N) / (1 + n)`.
    Raw { c: f32 },
    /// `Qhat + c * (P / sum P) * sqrt(N) / (1 + n)` with `Qhat` in `[0,1]`.
    NormalisedQ { c: f32, range_floor: f32 },
}

impl Default for PuctMode {
    fn default() -> Self {
        PuctMode::Raw { c: PUCT_C }
    }
}

impl PuctMode {
    /// Historical selection rule at the shipped constant.
    pub fn raw() -> Self {
        PuctMode::Raw { c: PUCT_C }
    }

    /// Normalised selection at `c`, with the default range floor.
    pub fn normalised(c: f32) -> Self {
        PuctMode::NormalisedQ {
            c,
            range_floor: DEFAULT_Q_RANGE_FLOOR,
        }
    }

    /// Short descriptor for corpus/report provenance. Selection changes
    /// `visits`, which is the raw material for the offline policy target
    /// (`botbowl-data`), so corpora from different modes must never be
    /// silently mixed.
    pub fn label(&self) -> String {
        match self {
            PuctMode::Raw { c } => format!("puct=raw(c={c})"),
            PuctMode::NormalisedQ { c, range_floor } => {
                format!("puct=norm(c={c},floor={range_floor})")
            }
        }
    }
}

/// MCTS workers spawned by `MctsBot::get_action` get an explicit
/// 16 MB stack instead of the OS-default ~2 MB. Sized for headroom
/// against the recursive `Node::get_state` and `Arc<Node>` drop
/// chains in `recon_mcts` (plan 013). 16 MB ≈ 16 000 frames at
/// ~1 KB-per-frame, which dwarfs the post-Step-F max DAG depth
/// (~20) and gives plenty of buffer if the horizon's loosened
/// later or a future scenario reaches deeper.
const WORKER_STACK_SIZE: usize = 16 * 1024 * 1024;

/// `recon_mcts` does not maintain its own per-node visit counter — the
/// `Score` we give it is the *sole* source of truth. `visits` is therefore
/// load-bearing for PUCT (read as `N(parent)` and `N(a)` in `puct_value`).
/// It is incremented during descent (`select_node`'s `fetch_add` on the
/// chosen child) and overwritten on backprop (`backprop_scores` returns a
/// fresh `BbScore` whose `visits` is the sum of children's visits).
#[derive(Debug)]
pub struct BbScore {
    pub visits: AtomicU32,
    pub score: i64,
    pub node_kind: BbPlayer,
    /// Plan 015 Step 5 — transient penalty applied during multi-worker
    /// descent. Bumped in `select_node` when a worker chooses this
    /// child; subtracted from `Q` (after perspective flip) in
    /// `puct_value` so concurrent workers diverge to other subtrees.
    /// Reset automatically: `backprop_scores` returns a freshly
    /// constructed `BbScore` (`virtual_loss = 0`) which replaces the
    /// previous struct in `Node.score`. Default magnitude resolved
    /// from `BLOOD_MCTS_VIRTUAL_LOSS` via `MctsBot::new` (default 30,
    /// `0` disables).
    pub virtual_loss: AtomicI32,
}

impl Clone for BbScore {
    fn clone(&self) -> Self {
        BbScore {
            visits: AtomicU32::new(self.visits.load(Ordering::Relaxed)),
            score: self.score,
            node_kind: self.node_kind,
            virtual_loss: AtomicI32::new(self.virtual_loss.load(Ordering::Relaxed)),
        }
    }
}

/// Captures the root-state's turn/score so the search can treat states
/// that have moved past the horizon as terminal.
///
/// "Past the horizon" = our team has started a new turn (i.e. the bot
/// played, the opponent played, and we are back to the bot's next
/// turn) OR either team has scored OR the game has ended. The
/// scoring heuristic (`score::leaf_score`) already evaluates these
/// states meaningfully, so the search can stop descending and read
/// the value off the leaf.
///
/// Must be a pure function of `(state, anchor)` — the anchor is
/// captured once per `get_action` call and is constant for the
/// lifetime of one search, so recombination stays correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HorizonAnchor {
    pub agent_team: TeamType,
    pub home_turn: u8,
    pub away_turn: u8,
    pub home_score: u8,
    pub away_score: u8,
}

impl HorizonAnchor {
    pub fn capture(state: &GameState, agent_team: TeamType) -> Self {
        Self {
            agent_team,
            home_turn: state.info.home_turn,
            away_turn: state.info.away_turn,
            home_score: state.home.score,
            away_score: state.away.score,
        }
    }

    /// Has either team scored since the anchor was captured?
    pub fn score_changed(&self, state: &GameState) -> bool {
        state.home.score != self.home_score || state.away.score != self.away_score
    }

    /// Home-centric score change since the anchor: `Δhome − Δaway`. The
    /// horizon treats any score change as terminal, so within one search
    /// at most one team can have scored → this is `-1`, `0`, or `+1`.
    pub fn score_delta(&self, state: &GameState) -> i64 {
        (state.home.score as i64 - self.home_score as i64) - (state.away.score as i64 - self.away_score as i64)
    }

    /// Has the state moved past the horizon? True ⇒ treat as terminal.
    pub fn diverged(&self, state: &GameState) -> bool {
        if state.info.game_over {
            return true;
        }
        if state.home.score != self.home_score || state.away.score != self.away_score {
            return true;
        }
        // The agent's turn counter only advances when it's their turn
        // to play *again* — meaning the bot's turn ended, the opponent
        // played, and the bot has been handed the next turn. That's
        // exactly "opponent's end-of-turn" per the design.
        match self.agent_team {
            TeamType::Home => state.info.home_turn > self.home_turn,
            TeamType::Away => state.info.away_turn > self.away_turn,
        }
    }
}

/// The leaf value function + prior source. `Heuristic` is the scripted
/// baseline (`leaf_score` / `prior_for_engine_action`); `Nn` swaps in a
/// frozen ONNX network (plan 017). A frozen deterministic CPU net is a
/// pure function of state, so it preserves the recombination-purity
/// invariant. NN priors **replace** scripted priors (no blending — there
/// is no principled common scale). The default stays `Heuristic`, so all
/// existing behaviour is byte-identical.
#[derive(Debug, Clone, Default)]
pub enum Evaluator {
    #[default]
    Heuristic,
    /// Pure touchdown reward, no shaping: leaves score `-1000`/`0`/`+1000`
    /// by the Home-centric score change since the horizon anchor. Priors
    /// stay scripted. Only viable where a TD is usually reachable within
    /// the search horizon (small boards) — on the full board almost every
    /// leaf is 0 and the search has no gradient. Matches the drive-relative
    /// frame the NN value head is trained on, so gen-0 data carries no
    /// shaping bias for later NN generations to unlearn.
    PureTd,
    Nn(Arc<NnEvaluator>),
    /// Hybrid diagnostic (plan 020): NN **leaf value** with the hand-tuned
    /// **scripted priors**. `Nn` replaces both at once, so a weak arm can't
    /// tell whether the value head or the learned priors are to blame —
    /// this variant isolates the value head.
    NnValue(Arc<NnEvaluator>),
}

/// `horizon` (None by default for backwards compatibility) bounds the
/// search depth — `available_actions` returns None as soon as a state
/// has diverged past the anchor. `MctsBot::get_action` always sets a
/// horizon; only direct callers (benches, tests that drive `Tree`
/// without `MctsBot`) see the unbounded form.
///
/// Not `Copy`: `Evaluator::Nn` carries an `Arc`. It is cloned once per
/// search when `run_search` builds the `gd` (cheap — an `Arc` bump).
#[derive(Debug, Clone)]
pub struct BloodBowlDynamics {
    pub horizon: Option<HorizonAnchor>,
    /// Plan 015 Step 5 — magnitude of the transient `BbScore.virtual_loss`
    /// penalty applied on descent in `select_node`. Default 30, calibrated
    /// against the BB Q-scale (ball control ±50, distance ±26). Set to 0
    /// to disable. Honoured per-search; `MctsBot::new` resolves it from
    /// `BLOOD_MCTS_VIRTUAL_LOSS`.
    pub virtual_loss: i32,
    /// Value/prior source. `Heuristic` (default) reproduces the scripted
    /// bot exactly; `Nn` routes priors + leaf value through the network.
    pub evaluator: Evaluator,
    /// Selection rule + exploration constant. Per-search rather than a
    /// module constant so two bots with different settings can play each
    /// other inside one process — the head-to-head this exists for.
    pub puct: PuctMode,
}

impl Default for BloodBowlDynamics {
    fn default() -> Self {
        Self {
            horizon: None,
            virtual_loss: DEFAULT_VIRTUAL_LOSS,
            evaluator: Evaluator::default(),
            puct: PuctMode::default(),
        }
    }
}

/// Plan 015 Step 5 — default virtual-loss magnitude. Sized against the
/// BB leaf-score scale (`score::leaf_score`): ball control ±50, distance
/// ±26, plus the dominating game-score ±1000. 30 is large enough to push
/// workers off a path with a small Q advantage, small enough not to
/// override a real ~100-point Q lead.
const DEFAULT_VIRTUAL_LOSS: i32 = 30;

/// Returns the unique surviving engine action when, after `pruning`
/// filters out domain-bad options, exactly one legal action remains.
/// Used by `apply_action`'s quiescent loop to walk past trivial
/// single-choice nodes so MCTS never spends a tree node modelling
/// them. Pure on state — same input always yields the same result.
fn sole_legal_action(state: &GameState) -> Option<EngineAction> {
    state.available_actions.team?;
    let mut iter = state.get_all_actions().into_iter().filter(|a| !should_prune(state, a));
    let first = iter.next()?;
    if iter.next().is_some() {
        return None;
    }
    Some(first)
}

/// Inspect a state and decide which "player" owns it from MCTS's
/// perspective. Chance nodes are detected by a pending roll; otherwise
/// the engine's `available_actions.team` tells us whose move it is.
fn player_for_state(state: &GameState) -> BbPlayer {
    if state.pending_roll.is_some() {
        return BbPlayer::Chance;
    }
    match state.available_actions.team {
        Some(TeamType::Home) => BbPlayer::Home,
        Some(TeamType::Away) => BbPlayer::Away,
        // Engine has no decision to expose — treat as a chance-like
        // "advance" node so callers know there's nothing to choose.
        None => BbPlayer::Chance,
    }
}

impl GameDynamics for BloodBowlDynamics {
    type Player = BbPlayer;
    type State = GameState;
    type Action = BbAction;
    type Score = BbScore;
    type ActionIter = Vec<(Self::Player, Self::Action)>;

    // TODO: there are currently a few things left to do here.
    //  - Action filtering: with domain knowledge we can prune some action that we know to be bad
    //  - priors calculated in the available actions and cached on the BbAction, to avoid
    //    recomputing them on every descent.
    //  - Scripted actions: Some decisions are effectively deterministic, like picking a block die.
    //    we should just script those. They should automaticaaly be applied in apply_action()
    //  - scripted chance outcomes: To make the tree less bushy we can just pick outcomes. eg armor
    //    breaks never succeed (if armor breaks it means a bunch of more rolls). maybe ball scatter too.

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        if state.info.game_over {
            return None;
        }
        // Horizon bound (Step F, plan 014): once the state has moved
        // past the root's turn or someone has scored, the leaf-score
        // already gives a meaningful value — no need to keep
        // expanding.
        if let Some(anchor) = self.horizon {
            if anchor.diverged(state) {
                return None;
            }
        }

        // Chance node: enumerate roll outcomes.
        if state.pending_roll.is_some() {
            let outcomes = roll_outcomes::enumerate(state, state.pending_roll.as_ref().unwrap());
            return Some(outcomes.into_iter().map(|a| (BbPlayer::Chance, a)).collect());
        }

        // Player node: copy engine's available actions.
        let team = state.available_actions.team?;
        let mcts_player = match team {
            TeamType::Home => BbPlayer::Home,
            TeamType::Away => BbPlayer::Away,
        };
        // Block-die picks and other scripted player decisions are
        // resolved inside `apply_action`'s quiescent-advance loop
        // (see `scripted::scripted_player_pick`), so MCTS never sees
        // those intermediate states. The only way one could surface
        // here is if the *root* state passed to `MctsBot::get_action`
        // is itself mid-block-die — uncommon in practice; the search
        // would waste one expansion fanning out over die choices,
        // then converge after a single `apply_action` step.

        // Safety net: if the pruning rules narrow the list to *zero*
        // legal actions while the engine still offers something, fall
        // back to the unfiltered set. Pruning is supposed to remove
        // wasteful options, not deadlock the search — better to spend
        // budget evaluating bad moves than to mark the node terminal
        // and corrupt the search. In debug builds we log the fallback
        // once per call site so a real bug doesn't go silent.
        let raw_actions = state.get_all_actions();
        let mut filtered: Vec<EngineAction> = raw_actions
            .iter()
            .copied()
            .filter(|a| !should_prune(state, a))
            .collect();
        if filtered.is_empty() && !raw_actions.is_empty() {
            #[cfg(debug_assertions)]
            eprintln!(
                "pruning emptied the action list at player_action_type={:?}; falling back to {} unfiltered actions",
                state.info.player_action_type,
                raw_actions.len()
            );
            filtered = raw_actions;
        }
        // Priors: the heuristic computes one per action; the NN does a
        // single forward over the whole (already-pruned) legal set and
        // gathers per-action logits. NN priors *replace* scripted priors.
        let priors: Vec<f32> = match &self.evaluator {
            Evaluator::Heuristic | Evaluator::PureTd | Evaluator::NnValue(_) => {
                filtered.iter().map(|a| prior_for_engine_action(state, *a)).collect()
            }
            Evaluator::Nn(nn) => nn.priors(state, &filtered),
        };
        let actions: Vec<(BbPlayer, BbAction)> = filtered
            .into_iter()
            .zip(priors)
            .map(|(a, prior)| (mcts_player, BbAction::player(a, prior)))
            .collect();
        if actions.is_empty() {
            None
        } else {
            Some(actions)
        }
    }

    /// Diagnostics only (recon_mcts cycle dump). A compact situation
    /// line first — cycles so far have been readable from turn/ball/
    /// pending-roll alone — then the full state for the hard cases.
    fn fmt_state(&self, state: &Self::State) -> String {
        format!(
            "turn h{}/a{} score {}–{} ball {:?} pending_roll {:?} proc_top {:?}\nfull: {:?}",
            state.info.home_turn,
            state.info.away_turn,
            state.home.score,
            state.away.score,
            state.ball,
            state.pending_roll,
            state.proc_stack_top(),
            state,
        )
    }

    fn fmt_action(&self, action: &Self::Action) -> String {
        format!("{action:?}")
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        let mut new_state = state;
        let proc_input: SomeProcInput = match action {
            BbAction::Player {
                action: engine_action, ..
            } => SomeProcInput::Action(*engine_action),
            BbAction::Chance { result, .. } => {
                // The result was enumerated from this state's pending
                // roll, so it must still be compatible with it. A Chance
                // action only exists because `available_actions` saw a
                // pending roll here — a missing/incompatible roll is a
                // search bug, not a legitimately disallowed action.
                debug_assert!(
                    new_state
                        .pending_roll
                        .as_ref()
                        .is_some_and(|req| req.is_compatible(*result)),
                    "chance result {:?} incompatible with pending_roll {:?}",
                    result,
                    new_state.pending_roll,
                );
                SomeProcInput::Roll(*result)
            }
        };

        new_state.step_with_roll_or_action(proc_input);
        //TODO: ☝️Some of the action filtering we want to do has to do with pathfinding and can only
        // be done after the action is appiled such as:
        //  - After START_HANDOFF / START_PASS action. If player doesn't have ball check that they can pickup ball
        //    AND reach a teammate to hand off to. If not the action should not be allow. And if
        //    they have ball there should be a player available to pass/handoff to.
        //  - Aftert START_BLITZ action, the player needs to be in range of a standing opponent to
        //    blitz, otherwise the action should not be allowed.
        //  to figure all this out we need to apply the action and then check the pathfinding.
        //  recon_mcts support disallowing the appiled action by returning None from this function

        // Quiescent-advance: keep stepping the engine while the state
        // is at a player decision whose outcome is effectively
        // scripted (block-die picks, coin toss, kick/receive) or
        // where pruning has narrowed the action set to a single
        // legal choice. Each pass is a pure function of state, so
        // recombination invariants stay intact.
        //
        // The 32-step budget matches `optimistic_leaf_score`'s
        // ceiling; if we ever hit it the engine is likely stuck in a
        // loop and a debug_assert will surface it in tests.
        let mut budget: u32 = 32;
        while budget > 0 && !new_state.info.game_over && new_state.pending_roll.is_none() {
            let next = if let Some(scripted) = scripted::scripted_player_pick(&new_state) {
                scripted
            } else if let Some(sole) = sole_legal_action(&new_state) {
                sole
            } else {
                break;
            };
            new_state.step_with_roll_or_action(SomeProcInput::Action(next));
            budget -= 1;
        }
        debug_assert!(budget > 0, "apply_action quiescent loop exhausted budget");

        Some(new_state)
    }

    fn select_node<II, Q, A>(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        parent_node_state: &Self::State,
        _purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<Self::Score>>,
        A: Deref<Target = Self::Action>,
    {
        let parent_visits = parent_score.map(|s| s.visits.load(Ordering::Relaxed)).unwrap_or(1) as f32;

        // IMPORTANT: each (Q, A) yielded by `scores_and_actions` carries a
        // live read-guard on the child's `score` RwLock (recon_mcts wraps
        // the read via `lockref::Ref`). Comparator-based selectors like
        // `max_by` / `min_by` keep the *current best* alive across each
        // following acquire of the next candidate — that is the chain
        // that deadlocks under contention against `update_score`'s queued
        // `score.write()` (Rust's queue-based RwLock is fair, so a queued
        // writer blocks new readers; if any worker holds another node's
        // score-read while waiting for this one, the wait graph cycles).
        //
        // We avoid that by collapsing each `(Q, A)` into a scalar inside
        // the `.map()` closure — the Ref drops at the end of the closure,
        // before the next is acquired — and comparing scalars only. We
        // then re-iterate (cheap: `II: Clone`) to find the chosen child
        // and `fetch_add` its `visits` atomic on the live `Q`.
        // Plan 015 Step 5 — `vl` is the per-descent virtual-loss
        // increment applied to the chosen child. Set to 0 on the chance
        // branch (probability-driven; divergence isn't the goal) and to
        // `self.virtual_loss` on the player branch. Subtracting it from
        // Q in `puct_value` pushes other workers off this path until the
        // next backprop replaces the BbScore (which resets vl to 0).
        let bump_chosen = |chosen: &BbAction, vl: i32| {
            for (q, a) in scores_and_actions.clone().into_iter() {
                if *a.deref() == *chosen {
                    if let Some(s) = q.as_ref() {
                        s.visits.fetch_add(1, Ordering::Relaxed);
                        if vl != 0 {
                            s.virtual_loss.fetch_add(vl, Ordering::Relaxed);
                        }
                    }
                    break;
                }
            }
        };

        // Chance node: unscored outcomes first (highest probability
        // first among them), then the outcome whose visit count is
        // furthest below its expected count under the action's
        // probability distribution — score for outcome i =
        // `p_i · (N_parent + 1) - N_i`, pick the argmax. The deficit
        // rule makes the empirical visit ratio converge to the real
        // probability distribution as N grows. (The previous
        // `min_by(visits)` was probability-blind and would over-sample
        // low-probability outcomes — e.g. on a 5/6 GFI it sampled
        // failures 5× more often than they should be.)
        //
        // The unscored-first tier exists because `backprop_scores`
        // withholds the chance node's expectation until every outcome
        // is scored (see the completeness gate there): sweeping the
        // outcomes as fast as possible is what closes that window.
        // `BbAction::Chance` carries `prob_bits`; `Player` variants
        // never appear here (we're under `pending_roll.is_some()`).
        if parent_node_state.pending_roll.is_some() {
            let total = parent_visits + 1.0;
            let pick = scores_and_actions
                .clone()
                .into_iter()
                .map(|(q, a)| {
                    let unscored = q.as_ref().as_ref().is_none();
                    let v = q
                        .as_ref()
                        .as_ref()
                        .map(|s| s.visits.load(Ordering::Relaxed))
                        .unwrap_or(0) as f32;
                    let action = a.deref().clone();
                    let prob = action.prob_f32().unwrap_or(0.0);
                    let deficit = prob * total - v;
                    ((unscored, deficit), action)
                })
                .max_by(|((ua, da), _), ((ub, db), _)| {
                    ua.cmp(ub).then(da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
                })
                .expect("chance node must have at least one outcome");
            bump_chosen(&pick.1, 0);
            return pick.1;
        }

        // Player node: PUCT with domain priors. Priors are cached on the
        // `BbAction::Player` variant at expansion time (see
        // `available_actions`), so this descent reads them off rather
        // than re-querying `prior_for` per visit. `prior_f32()` returns
        // `None` only for chance actions, which never appear in this
        // branch — `parent_node_state.pending_roll` was checked above.
        //
        // Scores are Home-centric. Home maximises PUCT; Away mirrors
        // by negating Q before adding the exploration bonus.
        let _ = parent_node_state; // no longer needed for priors; kept in case future rules want it.
        let home_perspective = *parent_player == BbPlayer::Home;
        // FPU (first-play urgency): unexplored children are estimated at
        // the parent's own Q, seen from the descending player's side. See
        // `puct_value` for why `Q = 0` is not a usable default here. A
        // parent with no score yet (fresh root) falls back to 0 — every
        // child is unexplored there, so the anchor cancels out anyway.
        let fpu = parent_score
            .map(|s| {
                if home_perspective {
                    s.score as f32
                } else {
                    -(s.score as f32)
                }
            })
            .unwrap_or(0.0);
        let pick = match self.puct {
            // Raw: the historical expression, with `c` substituted for the
            // module constant. Same ops in the same order, so at c == PUCT_C
            // this is bit-identical f32 output.
            PuctMode::Raw { c } => scores_and_actions
                .clone()
                .into_iter()
                .map(|(q, a)| {
                    let action = a.deref().clone();
                    let p = action.prior_f32().unwrap_or(1.0);
                    let v = puct_value(q.as_ref(), parent_visits, p, home_perspective, fpu, c);
                    (v, action)
                })
                .max_by(|(va, _), (vb, _)| va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal))
                .expect("player node must have at least one action"),

            PuctMode::NormalisedQ { c, range_floor } => {
                // PASS 1 — build the frame. Structurally identical to
                // `bump_chosen` above: the `(q, a)` binding drops at the end
                // of every loop body, so at most one `lockref::Ref` is ever
                // alive and the plan-013 wait-graph cycle cannot form. This
                // adds sequential acquire/release cycles, never nesting.
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                let mut prior_sum = 0.0f32;
                for (q, a) in scores_and_actions.clone().into_iter() {
                    prior_sum += a.deref().prior_f32().unwrap_or(1.0);
                    if let Some(s) = q.as_ref() {
                        // Flip FIRST, then min/max. The perspective flip is
                        // order-reversing, so taking min/max on the
                        // Home-centric score and negating afterwards would
                        // silently swap lo and hi at every Away node.
                        let qp = if home_perspective {
                            s.score as f32
                        } else {
                            -(s.score as f32)
                        };
                        lo = lo.min(qp);
                        hi = hi.max(qp);
                    }
                }
                let frame = QFrame::new(lo, hi, prior_sum, fpu, c, range_floor, self.virtual_loss);

                // PASS 2 — the existing scalar-collapsing map, unchanged in shape.
                scores_and_actions
                    .clone()
                    .into_iter()
                    .map(|(q, a)| {
                        let action = a.deref().clone();
                        let p = action.prior_f32().unwrap_or(1.0);
                        let v = puct_value_normalised(q.as_ref(), parent_visits, p, home_perspective, fpu, &frame);
                        (v, action)
                    })
                    .max_by(|(va, _), (vb, _)| va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal))
                    .expect("player node must have at least one action")
            }
        };
        bump_chosen(&pick.1, self.virtual_loss);
        pick.1
    }

    fn backprop_scores<II, Q, A>(
        &self,
        player: &Self::Player,
        _score_current: Option<&Self::Score>,
        child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = Self::Action>,
        Q: Deref<Target = Self::Score>,
    {
        // Chance node: probability-weighted average over visited children.
        //
        // Detect a chance node by its children's action variant rather
        // than `score_current.node_kind`: a chance node is *expanded, not
        // scored* (plan 018, `score_leaf` returns `None` for it), so
        // `score_current` is `None` until this aggregation runs and a
        // `node_kind` check would misroute it to the player branch.
        // recon_mcts hands us only already-scored children, and a chance
        // node's children are all `BbAction::Chance` (`pending_roll.is_some()`
        // ⟺ `available_actions` enumerated roll outcomes), so peeking the
        // first child's action is equivalent to the old check on every
        // node that has scored children — and works when the node is
        // unscored.
        let is_chance = child_scores_and_actions
            .clone()
            .into_iter()
            .next()
            .map(|(_, a)| matches!(*a.deref(), BbAction::Chance { .. }))
            .unwrap_or(false);
        if is_chance {
            let mut total_visits: u32 = 0;
            let mut weighted_sum: f64 = 0.0;
            let mut total_prob: f64 = 0.0;
            for (q, a) in child_scores_and_actions.into_iter() {
                let v = q.visits.load(Ordering::Relaxed);
                if v == 0 {
                    continue;
                }
                let prob = a.prob_f32().unwrap_or(0.0) as f64;
                weighted_sum += prob * q.score as f64;
                total_prob += prob;
                total_visits += v;
            }
            if total_visits == 0 {
                return None;
            }
            // Completeness gate: emit the expectation only once every
            // outcome is scored (enumeration normalises probabilities to
            // sum to 1, so scored-probability mass ≈ 1 ⟺ complete).
            // recon_mcts filters unscored children out of this iterator,
            // so a partial sum is the only completeness signal we get.
            // The previous behaviour — normalising over whatever was
            // scored — reported the expectation *conditioned on the
            // sampled outcomes*: a 1-GFI touchdown whose Fail branch
            // hadn't resolved yet backpropped a risk-free 1000, exactly
            // the plan-018 over-valuation this chance-node design exists
            // to prevent. While incomplete the node stays unscored
            // (`update_score` keeps the previous value on None) and the
            // parent's FPU treats it as unexplored; the select branch
            // above sweeps unscored outcomes first to keep that window
            // short.
            if total_prob < 0.999 {
                return None;
            }
            let avg = weighted_sum / total_prob;
            return Some(BbScore {
                visits: AtomicU32::new(total_visits),
                score: avg as i64,
                node_kind: BbPlayer::Chance,
                virtual_loss: AtomicI32::new(0),
            });
        }

        // Player node. Scores are Home-centric, so Home maximises and
        // Away minimises (plan 006 — adversarial backprop). Visits
        // sum across children so PUCT's √N(parent) reflects total
        // descents (plan 007 — matches the Chance branch above).
        let want_max = *player == BbPlayer::Home;
        let mut best_score: Option<i64> = None;
        let mut total_visits: u32 = 0;
        for (q, _) in child_scores_and_actions.into_iter() {
            total_visits += q.visits.load(Ordering::Relaxed);
            let s = q.score;
            best_score = match best_score {
                None => Some(s),
                Some(b) if (want_max && s > b) || (!want_max && s < b) => Some(s),
                Some(b) => Some(b),
            };
        }
        best_score.map(|score| BbScore {
            visits: AtomicU32::new(total_visits),
            score,
            node_kind: *player,
            virtual_loss: AtomicI32::new(0),
        })
    }

    fn score_leaf(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        // The value function (heuristic now, NN later) is evaluated
        // *only* at player-decision and terminal nodes — never on a
        // transient pre-roll (chance) state (plan 018). A leaf state is
        // one of:
        //   1. roll pending (`pending_roll` set) — a CHANCE node. We do
        //      not score it: returning `None` leaves it *expanded, not
        //      scored*. recon_mcts still expands it (its children are the
        //      weighted roll outcomes, see `available_actions`), and its
        //      value is derived purely from its children's
        //      probability-weighted backprop (`backprop_scores`). This
        //      is what kills the old optimistic over-valuation of risky
        //      rolls (e.g. a marked-ball pickup that auto-fails).
        //   2. terminal (`game_over` set) — bare leaf_score is the true
        //      drive outcome.
        //   3. player-decision (`available_actions.team` set) — the value
        //      function; the next player chooses from here.
        //   4. mid-procedure (none of the above) — `available_actions`
        //      returns `None`, so recon_mcts marks it terminal; it cannot
        //      be expanded and therefore must carry a score. We give it
        //      `leaf_score`. This technically scores an intermediate
        //      state, but such states are unexpandable and rare; the
        //      principled fix is to advance through them in
        //      `apply_action`'s quiescent loop (future work).
        // …with one carve-out: "expanded, not scored" only works when the
        // node actually gets expanded. A pending-roll state that is past
        // the horizon (or game over) is *terminal* to `available_actions`
        // — it will never get roll-outcome children, so leaving it
        // unscored makes it invisible to backprop. This is exactly where
        // every in-search touchdown lands (TD → engine advances to the
        // next kickoff → pauses on the kickoff Deviate roll, and the
        // score change has tripped the horizon), so without this
        // carve-out no TD value can ever reach the root. Mirrors the
        // terminal checks at the top of `available_actions`; `horizon`
        // is constant per search, so this stays a pure function of
        // (state, anchor).
        let past_horizon = self.horizon.is_some_and(|anchor| anchor.diverged(state));
        if state.pending_roll.is_some() && !state.info.game_over && !past_horizon {
            return None; // chance node — expanded, not scored
        }
        let score = match &self.evaluator {
            Evaluator::Heuristic => leaf_score(state),
            // Pure TD reward: the Home-centric score change since the
            // anchor, ±1000 on the leaf_score scale, 0 everywhere else.
            // Anchor-relative (not absolute scoreline) because random-start
            // games begin at arbitrary scores — an absolute delta would put
            // a constant nonzero offset on every leaf. Without an anchor
            // (direct `Tree` callers; `MctsBot` always sets one) fall back
            // to the absolute clamped delta so terminal states still rank.
            Evaluator::PureTd => match &self.horizon {
                Some(anchor) => anchor.score_delta(state).clamp(-1, 1) * 1000,
                None => (state.home.score as i64 - state.away.score as i64).clamp(-1, 1) * 1000,
            },
            // Exact-outcome carve-out: once someone has scored since the
            // anchor (or the game has ended), the drive outcome is *known*
            // — exactly the value the net is trained to predict. Asking
            // the NN here would (a) replace a gold-standard target with an
            // estimate that recon_mcts then freezes into solved subtrees
            // as exact minimax, and (b) query the net out-of-distribution:
            // training samples are decision states, never post-TD kickoff
            // states. Δscore *since the anchor* (not absolute `leaf_score`)
            // keeps the value on the NN's drive-relative scale — absolute
            // score would offset these leaves by the root score against
            // every NN-scored sibling. Without an anchor (direct `Tree`
            // callers; `MctsBot` always sets one) the NN scores everything.
            Evaluator::Nn(nn) | Evaluator::NnValue(nn) => match &self.horizon {
                Some(anchor) if state.info.game_over || anchor.score_changed(state) => {
                    anchor.score_delta(state).clamp(-1, 1) * 1000
                }
                _ => nn.value_home_i64(state),
            },
        };
        Some(BbScore {
            visits: AtomicU32::new(1),
            score,
            node_kind: player_for_state(state),
            virtual_loss: AtomicI32::new(0),
        })
    }
}

/// PUCT(a) = Q(a) + c · P(a) · √N(parent) / (1 + N(a))
///
/// Per-node normalisation frame for [`PuctMode::NormalisedQ`]. Built by one
/// scalar-extracting pass over the children in `select_node`; every field is
/// already in the *descending player's* perspective.
#[derive(Debug, Clone, Copy)]
struct QFrame {
    /// Min over scored siblings; falls back to `fpu` when none is scored.
    lo: f32,
    /// `(hi - lo).max(range_floor)` — never zero, never negative.
    denom: f32,
    /// Sum of priors over every offered child. `1.0` if that sum is zero.
    prior_sum: f32,
    c: f32,
    /// `virtual_loss / NORM_VL_REFERENCE`.
    vl_norm: f32,
}

impl QFrame {
    fn new(lo: f32, hi: f32, prior_sum: f32, fpu: f32, c: f32, range_floor: f32, virtual_loss: i32) -> Self {
        let (lo, denom) = if lo <= hi {
            (lo, (hi - lo).max(range_floor))
        } else {
            // No scored sibling yet — anchor the frame on the parent's Q so
            // every child normalises to 0 and the prior-scaled bonus decides,
            // which is what Raw does here too (a constant cancels under argmax).
            (fpu, range_floor)
        };
        QFrame {
            lo,
            denom,
            prior_sum: if prior_sum > 0.0 { prior_sum } else { 1.0 },
            c,
            vl_norm: virtual_loss as f32 / NORM_VL_REFERENCE,
        }
    }

    #[inline]
    fn norm(&self, q: f32) -> f32 {
        ((q - self.lo) / self.denom).clamp(0.0, 1.0)
    }
}

/// PUCT with the Q term mapped into `[0,1]` against the sibling range and
/// priors mapped onto the simplex, so `c` carries one meaning across states.
///
/// Two deliberate departures from pure affine invariance:
/// - `denom` is floored at `range_floor`, so a node whose children differ only
///   by positional drift is *not* stretched to full scale;
/// - virtual loss is subtracted in normalised units, so it always costs the
///   same fraction of the decision range. It is applied *after* the clamp and
///   is deliberately **not** re-clamped: `vl` accumulates per concurrent
///   descent, and clamping at 0 would hide the 2nd and 3rd worker's penalties.
fn puct_value_normalised(
    score: Option<&BbScore>,
    parent_visits: f32,
    prior: f32,
    home_perspective: bool,
    fpu: f32,
    frame: &QFrame,
) -> f32 {
    let parent_term = parent_visits.max(1.0).sqrt();
    let p = prior / frame.prior_sum;
    match score {
        None => frame.norm(fpu) + frame.c * p * parent_term,
        Some(s) => {
            let v = s.visits.load(Ordering::Relaxed) as f32;
            let q = if home_perspective {
                s.score as f32
            } else {
                -(s.score as f32)
            };
            let vl = s.virtual_loss.load(Ordering::Relaxed) as f32 * frame.vl_norm;
            (frame.norm(q) - vl) + frame.c * p * parent_term / (1.0 + v)
        }
    }
}

/// Unexplored children (`score == None`) have `N(a) = 0` and their `Q`
/// is estimated by `fpu` (first-play urgency): the *parent's* Q from the
/// descending player's perspective. `leaf_score` carries a large,
/// mostly-constant offset (ball control ±500 dominates every in-turn
/// state), so estimating an unexplored child at `Q = 0` buries it ~500
/// points below any explored sibling — the exploration term
/// `c·P·√N(parent)` only closes that gap at `N(parent) > (offset/cP)²`
/// (≈110 visits at prior 5), which starves wide fans of exploration.
/// Anchoring at the parent's Q makes "unexplored" mean "about as good as
/// this position" instead of "worthless", so the prior-scaled bonus
/// ranks unexplored children against explored ones on equal footing.
fn puct_value(
    score: Option<&BbScore>,
    parent_visits: f32,
    prior: f32,
    home_perspective: bool,
    fpu: f32,
    c: f32,
) -> f32 {
    let parent_term = parent_visits.max(1.0).sqrt();
    match score {
        None => fpu + c * prior * parent_term,
        Some(s) => {
            let v = s.visits.load(Ordering::Relaxed) as f32;
            // Plan 015 Step 5 — virtual loss subtracted *after* the
            // perspective flip so it always discounts the descending
            // player's view of the path. The child's `BbScore.score` is
            // Home-centric; the flip yields the current player's Q;
            // subtracting `vl` then pushes other workers off this path
            // regardless of which side is descending. `vl` is reset to
            // 0 when `backprop_scores` replaces the BbScore.
            let vl = s.virtual_loss.load(Ordering::Relaxed) as f32;
            let q_perspective = if home_perspective {
                s.score as f32
            } else {
                -(s.score as f32)
            };
            (q_perspective - vl) + c * prior * parent_term / (1.0 + v)
        }
    }
}

/// `Bot` adapter that drives a fresh MCTS search per call.
/// Node-equality strategy used by the underlying `recon_mcts` tree.
/// See `recon_mcts/src/tree.rs:397/416/438` for the markers.
///
/// **GOTCHA — never use `recon_mcts`'s `HashOnly` marker with Blood Bowl.**
/// `HashOnly` treats two nodes as equal iff their state *hashes* match. A
/// `GameState` is large (full pitch + both rosters + proc stack), so hash
/// collisions are inevitable, and a collision silently *merges two genuinely
/// different states into one DAG node* — producing illegal actions mid-search
/// (`micro_step` legality asserts), corrupted backprop, and re-derivation
/// panics on tree drop. It is not a tuning knob; it is broken for this game.
/// The `HashOnly` variant was therefore removed from `MemoryMode` (it lives on
/// in `recon_mcts` only for deterministic games like the `nim`/2048 demo).
/// If you ever reach for `recon_mcts::HashOnly` here, don't — use `StoreState`.
///
/// The two safe modes:
/// - `GetState`  — equality replays the action sequence from root and
///   compares full `GameState`. No spurious merges; pays an O(depth)
///   recompute on each equality check. Diagnostic-only.
/// - `StoreState` — full state stored on every node; equality is
///   structural and O(1). Highest memory cost — but with the horizon bound
///   (plan 014) capping max-depth at ~20 the footprint is bounded too.
///   **This is the default and the only mode production should use.**
///
/// Selectable at runtime via `with_memory_mode` or the `BLOOD_MCTS_MEMORY`
/// env var (`get` / `store` only). Env var wins when set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    GetState,
    StoreState,
}

impl MemoryMode {
    fn resolve(default: MemoryMode) -> Self {
        match std::env::var("BLOOD_MCTS_MEMORY").ok().as_deref() {
            Some("get") => MemoryMode::GetState,
            Some("store") => MemoryMode::StoreState,
            // `hash` is intentionally unsupported — HashOnly corrupts the DAG
            // for Blood Bowl (see the GOTCHA on `MemoryMode`). Fail loudly
            // rather than silently honour a footgun.
            Some("hash") => {
                panic!("BLOOD_MCTS_MEMORY=hash is unsupported: HashOnly corrupts the DAG for Blood Bowl (large state => hash collisions merge distinct nodes). Use 'store' (default) or 'get'.")
            }
            Some(other) => {
                panic!("BLOOD_MCTS_MEMORY={other:?} (expected one of: get | store)")
            }
            None => default,
        }
    }
}

/// The persistent search tree carried between `get_action` calls. Marker
/// type is part of `Tree`'s generic parameters, so the cache must enumerate
/// each `MemoryMode` variant — they monomorphise to different `Tree` types.
enum CachedTree {
    GetState(Arc<TreeAlias<BloodBowlDynamics, GetState>>),
    StoreState(Arc<TreeAlias<BloodBowlDynamics, StoreState>>),
}

/// How long each `MctsBot::get_action` call runs the tree search.
#[derive(Debug, Clone, Copy)]
pub enum SearchBudget {
    /// Run exactly this many `tree.step()` calls, split across workers.
    Iterations(usize),
    /// Run all workers for this wall-clock duration, then stop.
    Time(Duration),
}

pub struct MctsBot {
    pub budget: SearchBudget,
    /// Number of worker threads driving `tree.step()`. For
    /// `SearchBudget::Iterations` the total step count is split across
    /// them; for `SearchBudget::Time` every worker runs until the
    /// time limit fires. Tests that need deterministic results should
    /// pin this to 1 via [`with_workers`].
    pub n_workers: usize,
    /// Which `recon_mcts` state-memory strategy to use. See
    /// [`MemoryMode`]. Always `StoreState` in production (plan 014 + 013);
    /// `HashOnly` was removed entirely because it corrupts the DAG for
    /// Blood Bowl (see the GOTCHA on [`MemoryMode`]). `BLOOD_MCTS_MEMORY`
    /// can switch to the safe `get` diagnostic at runtime.
    pub memory_mode: MemoryMode,
    /// Carry the search tree across `get_action` calls within a single
    /// trial. Within a bot turn the horizon anchor is stable, so the
    /// surviving subtree's Q values stay valid; we just walk the
    /// registry to the new root and resume search there.
    ///
    /// Default on; `BLOOD_MCTS_TREE_REUSE=off` disables and falls back
    /// to today's fresh-tree-per-decision behaviour.
    reuse_enabled: bool,
    /// The tree from the previous `get_action`, or `None` for the
    /// first call / after an anchor change / after a cache miss.
    cached_tree: Option<CachedTree>,
    /// The horizon captured on the previous `get_action`. Used to gate
    /// reuse — when it changes (turn boundary or score), the cached
    /// tree's Q values reflect the old horizon and must be discarded.
    last_anchor: Option<HorizonAnchor>,
    /// Plan 015 Step 5 — magnitude of the transient virtual-loss
    /// penalty applied on descent. Resolved from `BLOOD_MCTS_VIRTUAL_LOSS`
    /// at `::new` (default 30, `0` disables). Threaded into
    /// `BloodBowlDynamics.virtual_loss` per `get_action`.
    virtual_loss: i32,
    /// Value/prior source threaded into `BloodBowlDynamics` each
    /// `get_action`. Default `Heuristic` → byte-identical to the scripted
    /// baseline; `with_evaluator` swaps in a frozen NN (plan 017).
    evaluator: Evaluator,
    /// Selection rule + exploration constant, threaded into
    /// `BloodBowlDynamics.puct` per `get_action`. Resolved from
    /// `BLOOD_MCTS_PUCT_MODE` / `_C` / `_RANGE_FLOOR` at `::new`.
    puct: PuctMode,
}

impl MctsBot {
    pub fn new(budget: SearchBudget) -> Self {
        let n_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let reuse_enabled = !matches!(
            std::env::var("BLOOD_MCTS_TREE_REUSE").ok().as_deref(),
            Some("off") | Some("0") | Some("false")
        );
        let virtual_loss = match std::env::var("BLOOD_MCTS_VIRTUAL_LOSS").ok() {
            Some(s) => s.parse::<i32>().unwrap_or(DEFAULT_VIRTUAL_LOSS),
            None => DEFAULT_VIRTUAL_LOSS,
        };
        // Resolve the mode *before* `c`, so `BLOOD_MCTS_PUCT_MODE=normalised`
        // alone can never leave the Raw constant (10) sitting in the
        // normalised frame — where it would be ~10x too explorative.
        let env_f32 = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<f32>().ok());
        let puct = match std::env::var("BLOOD_MCTS_PUCT_MODE").ok().as_deref() {
            Some("normalised") | Some("normalized") | Some("norm") => PuctMode::NormalisedQ {
                c: env_f32("BLOOD_MCTS_PUCT_C").unwrap_or(DEFAULT_PUCT_C_NORMALISED),
                range_floor: env_f32("BLOOD_MCTS_PUCT_RANGE_FLOOR").unwrap_or(DEFAULT_Q_RANGE_FLOOR),
            },
            _ => PuctMode::Raw {
                c: env_f32("BLOOD_MCTS_PUCT_C").unwrap_or(PUCT_C),
            },
        };
        Self {
            budget,
            n_workers,
            memory_mode: MemoryMode::StoreState,
            reuse_enabled,
            cached_tree: None,
            last_anchor: None,
            virtual_loss,
            evaluator: Evaluator::default(),
            puct,
        }
    }

    /// Swap the scripted heuristic for a frozen NN evaluator (plan 017).
    /// The `Arc` is shared across all search workers. Leaving this unset
    /// keeps the default heuristic behaviour.
    pub fn with_evaluator(mut self, evaluator: Arc<NnEvaluator>) -> Self {
        self.evaluator = Evaluator::Nn(evaluator);
        self
    }

    /// Hybrid diagnostic (plan 020): NN leaf value, scripted priors.
    pub fn with_nn_value(mut self, evaluator: Arc<NnEvaluator>) -> Self {
        self.evaluator = Evaluator::NnValue(evaluator);
        self
    }

    /// Swap the scripted heuristic for the unshaped pure-TD leaf value
    /// (`Evaluator::PureTd`). Priors stay scripted. Intended for small-board
    /// training-data generation where a touchdown is reachable within the
    /// search horizon from almost every state.
    pub fn with_pure_td(mut self) -> Self {
        self.evaluator = Evaluator::PureTd;
        self
    }

    pub fn with_workers(mut self, n_workers: usize) -> Self {
        self.n_workers = n_workers.max(1);
        self
    }

    pub fn with_memory_mode(mut self, memory_mode: MemoryMode) -> Self {
        self.memory_mode = memory_mode;
        self
    }

    /// Override the env-var default for tree reuse. Primarily for tests
    /// that A/B against the fresh-tree baseline; production callers
    /// should leave it on (the default).
    pub fn with_tree_reuse(mut self, reuse_enabled: bool) -> Self {
        self.reuse_enabled = reuse_enabled;
        self
    }

    /// Override the env-var default for the virtual-loss magnitude
    /// (plan 015 Step 5). Primarily for tests that A/B against
    /// disabled (`0`) or aggressive (`100`) settings.
    pub fn with_virtual_loss(mut self, virtual_loss: i32) -> Self {
        self.virtual_loss = virtual_loss;
        self
    }

    /// Override the selection rule / exploration constant. Prefer this to
    /// the env vars when A/B-ing: `MctsBot::new` reads the environment, so
    /// a stray `BLOOD_MCTS_PUCT_*` would silently move every arm at once.
    pub fn with_puct(mut self, puct: PuctMode) -> Self {
        self.puct = puct;
        self
    }
}

/// Raw search output for one decision, shared by [`MctsBot::get_action`]
/// and [`MctsBot::get_action_with_record`]. Holds the root children with
/// their per-child stats, the root aggregate, and which team was to move.
type BbNodeInfo = NodeInfo<GameState, BbPlayer, BbScore>;

struct SearchResult {
    move_info: Vec<(BbAction, BbNodeInfo)>,
    root_info: BbNodeInfo,
    agent_team: TeamType,
}

impl MctsBot {
    /// Run one search from `state` and return the raw tree output (root
    /// children + root aggregate). Handles tree reuse/caching internally;
    /// callers turn the result into an action ([`MctsBot::get_action`]) or
    /// a training sample ([`MctsBot::get_action_with_record`]).
    fn run_search(&mut self, state: &GameState) -> SearchResult {
        // Clone the state and turn on roll-by-roll stepping for the
        // search. `DiceMode::RegisterRolls` keeps `pending_roll` visible
        // on post-action states, so pickup / dodge / GFI rolls become
        // first-class chance nodes in the tree: `score_leaf` returns
        // `None` for them (expanded, not scored — plan 018) and their Q
        // is the probability-weighted backprop of their roll outcomes,
        // not a misleading pre-roll value. (We tried `DiceMode::RollDice`
        // to elide the chance-node modelling, but the engine's internal
        // roll resolution per `micro_step` ran orders of magnitude slower
        // for the same search budget.)
        //
        // `set_dice_mode` also drops any in-flight `registered_roll`
        // and any fixed dice the caller queued — so test scaffolding
        // can't leak into MCTS rollouts.
        let mut root_state = state.clone();
        root_state.set_dice_mode(botbowl_engine::core::gamestate::DiceMode::RegisterRolls);
        // Disable logging on the search state and drop the existing
        // log Vec. Each `apply_action` clones state, and the log Vec
        // gets pushed to on every `micro_step` *and* copied on every
        // clone — without these two calls MCTS pays an O(log-size) cost
        // per clone that compounds catastrophically on deep searches.
        root_state.set_logging_state(false);
        root_state.clear_log();

        let root_player = player_for_state(&root_state);
        // The horizon is captured from the root state and held
        // constant for this entire `get_action` call, so it remains a
        // pure function of `(state, anchor)` from the dynamics' point
        // of view — recombination invariants stay intact. If the root
        // is somehow not a player turn (chance node at root), fall
        // back to the engine's team_turn marker.
        let agent_team = match state.available_actions.team {
            Some(t) => t,
            None => root_state.info.team_turn,
        };
        // `BLOOD_MCTS_HORIZON=off` disables the horizon for A/B
        // comparison (e.g. against the historical unbounded baseline).
        let horizon_disabled = std::env::var("BLOOD_MCTS_HORIZON").ok().as_deref() == Some("off");
        let gd = BloodBowlDynamics {
            horizon: if horizon_disabled {
                None
            } else {
                Some(HorizonAnchor::capture(&root_state, agent_team))
            },
            virtual_loss: self.virtual_loss,
            evaluator: self.evaluator.clone(),
            puct: self.puct,
        };
        // `BLOOD_MCTS_WORKERS` lets benches that wouldn't otherwise pin
        // workers (e.g. `expand_bench_main`) force single-thread for
        // marker-comparison sweeps, without modifying the test.
        let n_workers = match std::env::var("BLOOD_MCTS_WORKERS").ok().as_deref() {
            Some(s) => s.parse::<usize>().unwrap_or(self.n_workers).max(1),
            None => self.n_workers.max(1),
        };
        let budget = self.budget;

        // `BLOOD_MCTS_STATS=1` dumps registry hit/miss/len and DAG
        // depth distribution after the search finishes but before the
        // tree drops. Used to validate recombination + depth claims
        // (plan 013) without modifying the benchmark tests.
        let dump_stats = std::env::var("BLOOD_MCTS_STATS").ok().as_deref() == Some("1");
        let memory_mode = MemoryMode::resolve(self.memory_mode);

        // `new_anchor` mirrors `gd.horizon` when horizon is enabled. We
        // gate tree reuse on it: if the previous call had the same
        // anchor, the cached tree's Q values were computed under the
        // same horizon and stay valid; otherwise (turn boundary, score)
        // the tree is dropped and rebuilt.
        let new_anchor = if horizon_disabled {
            None
        } else {
            Some(HorizonAnchor::capture(&root_state, agent_team))
        };
        let anchor_matches = self.reuse_enabled && self.cached_tree.is_some() && new_anchor == self.last_anchor;
        // If anchor changed (or reuse disabled), discard the cache up
        // front so we don't hold the prior tree alive past the search.
        if !anchor_matches {
            self.cached_tree = None;
        }

        // Marker type is part of `Tree`'s generic parameters, so the
        // arms below monomorphise to three concrete `Tree<...>` types.
        // The post-construction worker/extract code is identical, so
        // we factor it into a local macro to keep the arms readable
        // (a generic helper would force naming `Node<...>` bounds
        // explicitly — not worth it for three call sites).
        macro_rules! run_with_marker {
            ($marker:expr, $mode_label:expr, $cached_arm:ident) => {{
                // Try to reuse the cached tree. Bail to a fresh tree on:
                //   - cache empty / wrong marker / anchor mismatch
                //   - lookup misses (root state not in registry)
                //   - find_path_to returns None (new root not a descendant
                //     of the current cached-tree root)
                let reused: Option<Arc<TreeAlias<BloodBowlDynamics, $cached_arm>>> = if anchor_matches {
                    match self.cached_tree.take() {
                        Some(CachedTree::$cached_arm(t)) => match t.lookup_state(root_player, root_state.clone()) {
                            Some(node) => match t.find_path_to(&node) {
                                Some(path) => {
                                    for a in &path {
                                        t.apply_action(a);
                                    }
                                    Some(t)
                                }
                                None => None,
                            },
                            None => None,
                        },
                        _ => None,
                    }
                } else {
                    None
                };
                let was_reused = reused.is_some();
                let tree = match reused {
                    Some(t) => t,
                    None => Arc::new(Tree::new(gd, $marker, root_player, root_state)),
                };
                // Plan 008: workers spawn via `std::thread::scope`.
                // Bigger stack than the platform default (2 MB on macOS /
                // Linux pthread): `recon_mcts`'s `Node::get_state` and
                // `Arc<Node>` drop chain both recurse with the DAG depth.
                // With Step F's horizon bound depth caps at ~20, but the
                // headroom is cheap insurance against future regressions
                // where it might creep back up.
                match budget {
                    SearchBudget::Iterations(total) => {
                        let base = total / n_workers;
                        let rem = total % n_workers;
                        std::thread::scope(|s| {
                            for w in 0..n_workers {
                                let iters = base + if w < rem { 1 } else { 0 };
                                if iters == 0 {
                                    continue;
                                }
                                let tree = Arc::clone(&tree);
                                std::thread::Builder::new()
                                    .stack_size(WORKER_STACK_SIZE)
                                    .name(format!("mcts-worker-{w}"))
                                    .spawn_scoped(s, move || {
                                        for _ in 0..iters {
                                            // Solved tree: every aggregate is final,
                                            // further steps are no-ops — stop early.
                                            if tree.is_solved() {
                                                break;
                                            }
                                            tree.step();
                                        }
                                    })
                                    .expect("failed to spawn MCTS worker thread");
                            }
                        });
                    }
                    SearchBudget::Time(limit) => {
                        let stop = AtomicBool::new(false);
                        let stop_ref = &stop;
                        std::thread::scope(|s| {
                            let timer_tree = Arc::clone(&tree);
                            std::thread::Builder::new()
                                .name("mcts-timer".into())
                                .spawn_scoped(s, move || {
                                    // Poll rather than one long sleep: a solved
                                    // tree ends the search immediately instead of
                                    // idling out the rest of the budget (the scope
                                    // joins this thread, so its sleep is part of
                                    // get_action's wall time).
                                    let deadline = std::time::Instant::now() + limit;
                                    let tick = Duration::from_millis(2).min(limit);
                                    while std::time::Instant::now() < deadline && !timer_tree.is_solved() {
                                        std::thread::sleep(tick);
                                    }
                                    stop_ref.store(true, Ordering::Relaxed);
                                })
                                .expect("failed to spawn MCTS timer thread");
                            for w in 0..n_workers {
                                let tree = Arc::clone(&tree);
                                std::thread::Builder::new()
                                    .stack_size(WORKER_STACK_SIZE)
                                    .name(format!("mcts-worker-{w}"))
                                    .spawn_scoped(s, move || {
                                        while !stop_ref.load(Ordering::Relaxed) {
                                            // Solved tree: every aggregate is final,
                                            // further steps are no-ops — stop early
                                            // instead of spinning out the wall-clock
                                            // budget.
                                            if tree.is_solved() {
                                                break;
                                            }
                                            tree.step();
                                        }
                                    })
                                    .expect("failed to spawn MCTS worker thread");
                            }
                        });
                    }
                }
                if dump_stats {
                    let info = tree.get_registry_info();
                    let hits = info.hits.load(Ordering::Relaxed);
                    let misses = info.misses.load(Ordering::Relaxed);
                    let len = info.len.load(Ordering::Relaxed);
                    let denom = (hits + misses).max(1);
                    let budget_label = match budget {
                        SearchBudget::Iterations(n) => format!("{n}"),
                        SearchBudget::Time(limit) => format!("~{} ({limit:?})", hits + misses),
                    };
                    eprintln!(
                        "MCTS_STATS mode={} iters={} workers={} reg_len={} hits={} misses={} reuse={:.4}",
                        $mode_label,
                        budget_label,
                        n_workers,
                        len,
                        hits,
                        misses,
                        hits as f64 / denom as f64,
                    );
                    // `find_children_sorted_with_depth` is itself
                    // recursive (`tree.rs:1058`); on a deep DAG this
                    // can overflow the worker stack just like the
                    // search itself. Run it on the main thread (much
                    // larger default stack) so we still get the depth
                    // number when the worker would have crashed.
                    let sorted =
                        std::thread::scope(|s| s.spawn(|| tree.find_children_sorted_with_depth()).join().unwrap());
                    let max_depth = sorted.iter().map(|(_, d)| *d).max().unwrap_or(0);
                    let n_nodes = sorted.len();
                    // depth histogram (inv-depth: 0 == leaf), bucketed
                    let mut buckets = [0usize; 8];
                    for (_, d) in &sorted {
                        let bucket = match *d {
                            0 => 0,
                            1..=2 => 1,
                            3..=5 => 2,
                            6..=10 => 3,
                            11..=20 => 4,
                            21..=50 => 5,
                            51..=100 => 6,
                            _ => 7,
                        };
                        buckets[bucket] += 1;
                    }
                    eprintln!(
                        "MCTS_STATS mode={} reachable_nodes={} max_depth={} \
                         depth_hist[0,1-2,3-5,6-10,11-20,21-50,51-100,100+]={:?}",
                        $mode_label, n_nodes, max_depth, buckets
                    );
                }
                let move_info = tree.get_next_move_info().unwrap_or_else(|| {
                    panic!(
                        "MCTS tree has no move info at root (reused={was_reused}): root_info={:#?}",
                        tree.get_root_info()
                    )
                });
                // Root aggregate (Q / visits / solved) captured before the
                // tree is handed to the cache — the record needs it and
                // the tree is moved out below.
                let root_info = tree.get_root_info();
                (move_info, root_info, tree)
            }};
        }

        let (move_info, root_info, cache_after) = match memory_mode {
            MemoryMode::GetState => {
                let (mi, ri, t) = run_with_marker!(GetState, "get", GetState);
                (mi, ri, CachedTree::GetState(t))
            }
            MemoryMode::StoreState => {
                let (mi, ri, t) = run_with_marker!(StoreState, "store", StoreState);
                (mi, ri, CachedTree::StoreState(t))
            }
        };
        // Stash the tree for the next `get_action`. Reuse-disabled bots
        // (BLOOD_MCTS_TREE_REUSE=off) still get here; the next call
        // takes the `anchor_matches` branch as false and rebuilds, but
        // the cache field stays populated. That's fine — the alternative
        // (clear cache when reuse_enabled is false) buys nothing and
        // adds branches.
        if self.reuse_enabled {
            self.cached_tree = Some(cache_after);
            self.last_anchor = new_anchor;
        } else {
            self.cached_tree = None;
            self.last_anchor = None;
        }

        SearchResult {
            move_info,
            root_info,
            agent_team,
        }
    }

    /// Pick the root child to play from a completed search. See the
    /// comment inside for why this is aggregated-Q, not most-visited.
    fn pick_best_action(move_info: &[(BbAction, BbNodeInfo)], agent_team: TeamType) -> EngineAction {
        if std::env::var("BLOOD_MCTS_DEBUG_ROOT").ok().as_deref() == Some("1") {
            let mut infos: Vec<_> = move_info
                .iter()
                .map(|(a, info)| {
                    let (v, s) = info
                        .score
                        .as_ref()
                        .map(|s| (s.visits.load(Ordering::Relaxed), s.score))
                        .unwrap_or((0, 0));
                    (v, s, a.clone())
                })
                .collect();
            infos.sort_by_key(|(v, _, _)| std::cmp::Reverse(*v));
            eprintln!("== root children ({}):", infos.len());
            for (v, s, a) in infos.iter().take(10) {
                eprintln!("   visits={v:5} q={s:6} {a:?}");
            }
        }
        // Pick by aggregated Q (from the agent's perspective), with
        // visits as the tie-break; unscored children rank below every
        // scored one. Q is a minimax aggregate (Home max / Away min /
        // chance expectation), so it is the principled root decision
        // value. Most-visited — the classic robust-child rule — is NOT
        // reliable here: descents that end on already-terminal nodes
        // bump visit counters without contributing new information, so
        // raw visit counts over-weight whichever path saturated first
        // (observed picking a q=469 GFI gamble over a q=525 safe move,
        // and picking arbitrarily when no child had been visited).
        let q_sign: i64 = match agent_team {
            TeamType::Home => 1,
            TeamType::Away => -1,
        };
        let best = move_info
            .iter()
            .max_by_key(|(_, info)| {
                info.score
                    .as_ref()
                    .map(|s| (1i64, q_sign * s.score, s.visits.load(Ordering::Relaxed) as i64))
                    .unwrap_or((0, 0, 0))
            })
            .expect("root must offer at least one action");
        match &best.0 {
            BbAction::Player { action, .. } => *action,
            BbAction::Chance { .. } => {
                panic!("root selected a chance action — root must be a player turn");
            }
        }
    }

    /// Like [`Bot::get_action`], but also returns a training [`Sample`]:
    /// the decision node, the chosen action, and the raw per-child search
    /// stats (visits / Q / prior / solved) plus the root aggregate. The
    /// returned action is identical to what `get_action` would play — this
    /// method just additionally harvests the search tree before it drops.
    ///
    /// `outcome_value` on the sample is left `None`; backfill it at the end
    /// of the trajectory (see [`botbowl_data::Trajectory::backfill_outcome_value`]).
    pub fn get_action_with_record(&mut self, state: &GameState) -> (EngineAction, Sample) {
        let result = self.run_search(state);
        let action = Self::pick_best_action(&result.move_info, result.agent_team);

        let children = result
            .move_info
            .iter()
            .filter_map(|(a, info)| {
                // Only player edges are training targets; chance edges are
                // search-internal roll outcomes, not agent decisions.
                let engine_action = match a {
                    BbAction::Player { action, .. } => *action,
                    BbAction::Chance { .. } => return None,
                };
                let (visits, q) = info
                    .score
                    .as_ref()
                    .map(|s| (s.visits.load(Ordering::Relaxed), Some(s.score)))
                    .unwrap_or((0, None));
                Some(ChildStat {
                    action: engine_action,
                    visits,
                    q,
                    prior: a.prior_f32(),
                    solved: info.solved,
                    terminal: matches!(info.n_children, Status::Terminal),
                })
            })
            .collect();

        let (root_visits, root_value) = result
            .root_info
            .score
            .as_ref()
            .map(|s| (s.visits.load(Ordering::Relaxed), Some(s.score)))
            .unwrap_or((0, None));

        let sample = Sample {
            state: state.clone(),
            to_move: result.agent_team.into(),
            chosen_action: action,
            children,
            root_value,
            root_visits,
            root_solved: result.root_info.solved,
            outcome_value: None,
        };
        (action, sample)
    }
}

impl Bot for MctsBot {
    fn get_action(&mut self, state: &GameState) -> EngineAction {
        let result = self.run_search(state);
        Self::pick_best_action(&result.move_info, result.agent_team)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::model::Action as EngineAction;
    use botbowl_engine::core::table::SimpleAT;

    fn child(score: i64, visits: u32) -> BbScore {
        BbScore {
            visits: AtomicU32::new(visits),
            score,
            node_kind: BbPlayer::Home,
            virtual_loss: AtomicI32::new(0),
        }
    }

    // A player node's children carry `BbAction::Player` actions — that
    // variant is what `backprop_scores` keys its player/chance routing
    // on (plan 018). The specific engine action is irrelevant here.
    fn placeholder_action() -> BbAction {
        BbAction::player(EngineAction::Simple(SimpleAT::EndTurn), 1.0)
    }

    #[test]
    fn backprop_player_home_maximises_and_sums_visits() {
        let dynamics = BloodBowlDynamics::default();
        let actions = [placeholder_action(), placeholder_action(), placeholder_action()];
        let children = [child(-5, 2), child(10, 5), child(3, 1)];
        let pairs: Vec<(&BbScore, &BbAction)> = children.iter().zip(actions.iter()).collect();
        let result = dynamics
            .backprop_scores(&BbPlayer::Home, None, pairs)
            .expect("backprop should yield a score");
        assert_eq!(result.score, 10, "Home should pick the max-Q child");
        assert_eq!(
            result.visits.load(Ordering::Relaxed),
            8,
            "visits should sum across children, not max"
        );
        assert_eq!(result.node_kind, BbPlayer::Home);
    }

    #[test]
    fn backprop_player_away_minimises_and_sums_visits() {
        let dynamics = BloodBowlDynamics::default();
        let actions = [placeholder_action(), placeholder_action(), placeholder_action()];
        let children = [child(-5, 2), child(10, 5), child(3, 1)];
        let pairs: Vec<(&BbScore, &BbAction)> = children.iter().zip(actions.iter()).collect();
        let result = dynamics
            .backprop_scores(&BbPlayer::Away, None, pairs)
            .expect("backprop should yield a score");
        assert_eq!(
            result.score, -5,
            "Away should pick the min-Q child (Home-centric scoring)"
        );
        assert_eq!(result.visits.load(Ordering::Relaxed), 8);
        assert_eq!(
            result.node_kind,
            BbPlayer::Away,
            "node_kind should mirror the player owning the node"
        );
    }

    /// `apply_action` must collapse scripted player decisions
    /// (coin toss, kick/receive, block-die picks) into a single
    /// engine advance so the MCTS DAG doesn't carry nodes for
    /// them. Drive the engine from the pre-coin-toss state and
    /// confirm the post-apply state is well past the toss + the
    /// kick/receive choice.
    #[test]
    fn apply_action_walks_through_scripted_coin_toss() {
        use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameStateBuilder};
        use botbowl_engine::core::model::Action as EA;
        use botbowl_engine::core::table::SimpleAT;

        let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
        state.set_dice_mode(DiceMode::RegisterRolls);

        // Sanity: the pre-step state really does offer Heads/Tails as
        // a simple choice — i.e. the engine is at the toss decision.
        let simple_before = state.available_actions.get_simple();
        assert!(
            simple_before.contains(&SimpleAT::Heads) && simple_before.contains(&SimpleAT::Tails),
            "pre-condition: expected the toss decision"
        );

        let dynamics = BloodBowlDynamics::default();
        // The MCTS edge that triggers this apply is the scripted
        // `Heads` pick itself (what the quiescent loop would pick on
        // the very next iteration anyway). After this call, we
        // should be past the toss *and* past the kick/receive
        // sub-decision — both are scripted.
        let bb = BbAction::player(EA::Simple(SimpleAT::Heads), 1.0);
        let out = dynamics
            .apply_action(state, &bb)
            .expect("apply_action should not abort on a fresh coin-toss state");

        let simple_after = out.available_actions.get_simple();
        let still_toss = simple_after.contains(&SimpleAT::Heads) || simple_after.contains(&SimpleAT::Tails);
        let still_kick_receive = simple_after.contains(&SimpleAT::Kick) && simple_after.contains(&SimpleAT::Receive);
        assert!(!still_toss, "post-apply: must have left the toss decision");
        assert!(
            !still_kick_receive,
            "post-apply: kick/receive should have been scripted-through too"
        );
    }

    /// The chance-node expectation must not be emitted while outcomes
    /// are missing: normalising over the sampled subset reports the
    /// value *conditioned on those outcomes* — a 1-GFI touchdown whose
    /// Fail branch hasn't resolved backprops a risk-free 1000.
    #[test]
    fn chance_backprop_waits_for_all_outcomes() {
        use botbowl_engine::core::dices::RollResult;

        let dynamics = BloodBowlDynamics::default();
        let pass = BbAction::chance(RollResult::Pass, 5.0 / 6.0);
        let fail = BbAction::chance(RollResult::Fail, 1.0 / 6.0);

        // Only the Pass outcome scored (5/6 of the mass): no aggregate.
        let pass_score = child(1000, 3);
        let partial: Vec<(&BbScore, &BbAction)> = vec![(&pass_score, &pass)];
        assert!(
            dynamics.backprop_scores(&BbPlayer::Chance, None, partial).is_none(),
            "incomplete outcome set must not emit an expectation"
        );

        // Both outcomes scored: exact probability-weighted expectation.
        let fail_score = child(-200, 1);
        let complete: Vec<(&BbScore, &BbAction)> = vec![(&pass_score, &pass), (&fail_score, &fail)];
        let result = dynamics
            .backprop_scores(&BbPlayer::Chance, None, complete)
            .expect("complete outcome set must emit");
        let expected = (5.0 / 6.0 * 1000.0 + 1.0 / 6.0 * -200.0) as i64;
        assert!(
            (result.score - expected).abs() <= 1,
            "expected ≈{expected} (f32 prob round-trip may truncate by 1), got {}",
            result.score
        );
        assert_eq!(result.visits.load(Ordering::Relaxed), 4);
        assert_eq!(result.node_kind, BbPlayer::Chance);
    }

    /// Chance selection sweeps unscored outcomes before rebalancing
    /// visited ones — that is what closes the completeness gate above.
    /// The visit-deficit rule alone would keep hammering a
    /// high-probability scored outcome at high parent visit counts.
    #[test]
    fn chance_select_prefers_unscored_outcomes() {
        use botbowl_engine::core::dices::{D6Target, RequestedRoll, RollResult};
        use botbowl_engine::core::gamestate::GameStateBuilder;
        use botbowl_engine::core::model::Position;

        let mut state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        state.pending_roll = Some(RequestedRoll::D6PassFail(D6Target::TwoPlus));

        let pass = BbAction::chance(RollResult::Pass, 5.0 / 6.0);
        let fail = BbAction::chance(RollResult::Fail, 1.0 / 6.0);
        let pass_score = Some(child(1000, 1));
        let fail_score: Option<BbScore> = None; // unscored placeholder
        let parent = child(1000, 100); // high parent visits → deficit rule alone would pick Pass

        let children: Vec<(&Option<BbScore>, &BbAction)> = vec![(&pass_score, &pass), (&fail_score, &fail)];
        let dynamics = BloodBowlDynamics::default();
        let picked = dynamics.select_node(
            Some(&parent),
            &BbPlayer::Chance,
            &state,
            SelectNodeState::Explore,
            children,
        );
        assert_eq!(picked, fail, "the unscored outcome must be swept first");
    }

    /// Plan 018 leaves pending-roll (chance) states unscored because
    /// their value comes from expanding their roll-outcome children —
    /// but a pending-roll state that is already past the horizon is
    /// terminal to `available_actions` and never gets those children.
    /// It must be scored or it becomes a backprop dead end. This is
    /// where every in-search touchdown lands (TD → next kickoff →
    /// pending Deviate + score change tripped the horizon), so an
    /// unscored dead end here makes TDs invisible to the search.
    #[test]
    fn score_leaf_scores_pending_roll_state_past_horizon() {
        use botbowl_engine::core::dices::RequestedRoll;
        use botbowl_engine::core::gamestate::GameStateBuilder;
        use botbowl_engine::core::model::{Position, TeamType};

        let mut state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        state.pending_roll = Some(RequestedRoll::D8);
        let anchor = HorizonAnchor::capture(&state, TeamType::Home);
        let dynamics = BloodBowlDynamics {
            horizon: Some(anchor),
            virtual_loss: 0,
            ..Default::default()
        };

        // Within the horizon: a chance node, expanded not scored.
        assert!(
            dynamics.score_leaf(None, &BbPlayer::Chance, &state).is_none(),
            "in-horizon pending-roll states stay unscored (plan 018)"
        );

        // Past the horizon (a touchdown has been scored since the
        // anchor): terminal — must carry a score.
        state.home.score += 1;
        let score = dynamics
            .score_leaf(None, &BbPlayer::Chance, &state)
            .expect("past-horizon pending-roll state must be scored");
        assert!(
            score.score >= 1000,
            "the touchdown must dominate the leaf score, got {}",
            score.score
        );
    }

    #[test]
    fn pure_td_leaf_is_anchor_relative_sign_only() {
        use botbowl_engine::core::dices::RequestedRoll;
        use botbowl_engine::core::gamestate::GameStateBuilder;
        use botbowl_engine::core::model::{Position, TeamType};

        // Non-level starting score (random-start games do this): the
        // anchor absorbs it, so in-horizon leaves score 0, not +1000.
        let mut state = GameStateBuilder::new().add_home_player(Position::new((5, 5))).build();
        state.home.score = 2;
        state.away.score = 1;
        let anchor = HorizonAnchor::capture(&state, TeamType::Home);
        let dynamics = BloodBowlDynamics {
            horizon: Some(anchor),
            virtual_loss: 0,
            evaluator: Evaluator::PureTd,
            ..Default::default()
        };

        let score = dynamics
            .score_leaf(None, &BbPlayer::Home, &state)
            .expect("player-decision leaf must be scored");
        assert_eq!(score.score, 0, "no score change since anchor → 0, regardless of scoreline");

        // In-horizon chance node: still expanded-not-scored (plan 018).
        let mut pending = state.clone();
        pending.pending_roll = Some(RequestedRoll::D8);
        assert!(dynamics.score_leaf(None, &BbPlayer::Chance, &pending).is_none());

        // Home TD since the anchor → +1000; Away TD → -1000.
        let mut home_td = pending.clone();
        home_td.home.score += 1;
        assert_eq!(dynamics.score_leaf(None, &BbPlayer::Chance, &home_td).unwrap().score, 1000);
        let mut away_td = pending;
        away_td.away.score += 2; // clamp: multi-TD delta still ±1000
        assert_eq!(dynamics.score_leaf(None, &BbPlayer::Chance, &away_td).unwrap().score, -1000);
    }

    #[test]
    fn puct_mirrors_for_away_player() {
        let a = BbScore {
            visits: AtomicU32::new(10),
            score: 50,
            node_kind: BbPlayer::Home,
            virtual_loss: AtomicI32::new(0),
        };
        let b = BbScore {
            visits: AtomicU32::new(10),
            score: -50,
            node_kind: BbPlayer::Home,
            virtual_loss: AtomicI32::new(0),
        };
        let parent_visits = 100.0;
        let prior = 0.5;

        let va_home = puct_value(Some(&a), parent_visits, prior, true, 0.0, PUCT_C);
        let vb_home = puct_value(Some(&b), parent_visits, prior, true, 0.0, PUCT_C);
        assert!(
            va_home > vb_home,
            "Home should rank +50 above -50 (va={va_home}, vb={vb_home})"
        );

        let va_away = puct_value(Some(&a), parent_visits, prior, false, 0.0, PUCT_C);
        let vb_away = puct_value(Some(&b), parent_visits, prior, false, 0.0, PUCT_C);
        assert!(
            vb_away > va_away,
            "Away should rank -50 above +50 (va={va_away}, vb={vb_away})"
        );
    }

    /// FPU regression guard: leaf scores carry a ~+520 constant offset
    /// (ball control + carrier distance), so with a Q=0 estimate an
    /// unexplored child could never compete with an explored ~525
    /// sibling until √N(parent) grew past the offset (N > ~110). With
    /// FPU anchored at the parent's Q, an unexplored same-prior sibling
    /// must outrank an explored child whose Q merely matches the parent.
    #[test]
    fn fpu_keeps_unexplored_children_competitive() {
        let explored = BbScore {
            visits: AtomicU32::new(30),
            score: 525,
            node_kind: BbPlayer::Home,
            virtual_loss: AtomicI32::new(0),
        };
        let parent_visits = 100.0;
        let prior = 5.0;
        let fpu = 525.0; // parent Q, Home perspective

        let v_explored = puct_value(Some(&explored), parent_visits, prior, true, fpu, PUCT_C);
        let v_unexplored = puct_value(None, parent_visits, prior, true, fpu, PUCT_C);
        assert!(
            v_unexplored > v_explored,
            "unexplored (={v_unexplored}) must beat an explored parent-level sibling (={v_explored})"
        );

        // But a genuinely better explored child (a found touchdown)
        // still dominates the unexplored pool.
        let td = BbScore {
            visits: AtomicU32::new(30),
            score: 1069,
            node_kind: BbPlayer::Home,
            virtual_loss: AtomicI32::new(0),
        };
        let v_td = puct_value(Some(&td), parent_visits, prior, true, fpu, PUCT_C);
        assert!(
            v_td > v_unexplored,
            "a found TD (={v_td}) must outrank unexplored siblings (={v_unexplored})"
        );

        // This second assert is a CALIBRATION BOUND, not an invariant, and it
        // is stated here so a future `c` change fails informatively instead of
        // as an inscrutable float comparison. With this fixture:
        //     v_td         = 1069 + 50c/31
        //     v_unexplored =  525 + 50c
        // so the ordering holds iff 544 > 48.387c, i.e. c < 11.243.
        // (It holds at all only because the fixture's fpu=525 is *below* the TD
        // child — a stale-low parent Q. With the self-consistent fpu that
        // `backprop_scores` produces, fpu == hi, and unexplored beats every
        // explored sibling for all c > 0. See `raw_fpu_beats_..._for_any_c`.)
        const C_MAX_FOR_THIS_FIXTURE: f32 = 11.243;
        assert!(
            PUCT_C < C_MAX_FOR_THIS_FIXTURE,
            "PUCT_C={PUCT_C} inverts the TD/unexplored ordering for this fixture (bound {C_MAX_FOR_THIS_FIXTURE})"
        );
    }

    /// The c-free half of the fixture above: at equal prior and `fpu >= q`, an
    /// unexplored child outranks an explored sibling for *any* positive `c`,
    /// because the bonus is larger by `(1 - 1/(1+n))` and the Q term already
    /// favours it. Swept, so no future exploration constant can trip it.
    #[test]
    fn raw_fpu_beats_parent_level_sibling_for_any_c() {
        for c in [0.0f32, 0.5, 1.0, 2.0, 10.0, 50.0, 300.0] {
            for (q, n) in [(525i64, 30u32), (400, 1), (525, 200)] {
                let explored = BbScore {
                    visits: AtomicU32::new(n),
                    score: q,
                    node_kind: BbPlayer::Home,
                    virtual_loss: AtomicI32::new(0),
                };
                let fpu = 525.0;
                let ve = puct_value(Some(&explored), 100.0, 5.0, true, fpu, c);
                let vu = puct_value(None, 100.0, 5.0, true, fpu, c);
                assert!(vu >= ve, "c={c} q={q} n={n}: unexplored {vu} < explored {ve}");
            }
        }
    }

    /// The mirror test's normalised twin. This is the regression guard for the
    /// flip-before-min/max ordering: taking min/max on the Home-centric score
    /// and negating afterwards swaps `lo`/`hi` at Away nodes, which silently
    /// inverts every Away selection.
    #[test]
    fn normalised_puct_mirrors_for_away_player() {
        let a = BbScore { visits: AtomicU32::new(10), score: 50, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };
        let b = BbScore { visits: AtomicU32::new(10), score: -50, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };

        // Home frame: lo=-50, hi=+50 after the flip (identity).
        let fh = QFrame::new(-50.0, 50.0, 1.0, 0.0, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
        let va = puct_value_normalised(Some(&a), 100.0, 0.5, true, 0.0, &fh);
        let vb = puct_value_normalised(Some(&b), 100.0, 0.5, true, 0.0, &fh);
        assert!(va > vb, "Home should rank +50 above -50 (va={va}, vb={vb})");

        // Away frame: the flip maps {+50,-50} to {-50,+50}, so lo/hi are the
        // same pair — computed flip-first, as `select_node` does.
        let fa = QFrame::new(-50.0, 50.0, 1.0, 0.0, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
        let va2 = puct_value_normalised(Some(&a), 100.0, 0.5, false, 0.0, &fa);
        let vb2 = puct_value_normalised(Some(&b), 100.0, 0.5, false, 0.0, &fa);
        assert!(vb2 > va2, "Away should rank -50 above +50 (va={va2}, vb={vb2})");
    }

    /// **The point of the whole change.** Rescaling every child's Q by a
    /// positive affine map must not change what the search picks. Raw fails
    /// this (see the negative control below); NormalisedQ must not.
    #[test]
    fn normalised_puct_is_invariant_to_affine_rescaling() {
        // Range comfortably above the floor, so the floor is not what is being
        // tested here (that is `normalised_range_floor_...`).
        let base: [i64; 3] = [0, 400, 1000];
        let priors = [1.0f32, 5.0, 0.2];
        for home in [true, false] {
            for (alpha, beta) in [(1i64, 0i64), (2, 1000), (8, -700)] {
                let mut picks = Vec::new();
                for scale in [false, true] {
                    let scores: Vec<i64> = base
                        .iter()
                        .map(|q| if scale { alpha * q + beta } else { *q })
                        .collect();
                    let flip = |q: i64| if home { q as f32 } else { -(q as f32) };
                    let lo = scores.iter().map(|q| flip(*q)).fold(f32::INFINITY, f32::min);
                    let hi = scores.iter().map(|q| flip(*q)).fold(f32::NEG_INFINITY, f32::max);
                    let frame = QFrame::new(lo, hi, priors.iter().sum(), lo, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
                    let best = scores
                        .iter()
                        .zip(priors.iter())
                        .enumerate()
                        .map(|(i, (q, p))| {
                            let sc = BbScore { visits: AtomicU32::new(7), score: *q, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };
                            (i, puct_value_normalised(Some(&sc), 100.0, *p, home, lo, &frame))
                        })
                        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                        .unwrap()
                        .0;
                    picks.push(best);
                }
                assert_eq!(picks[0], picks[1], "home={home} alpha={alpha} beta={beta}: argmax moved under affine rescaling");
            }
        }
    }

    /// Negative control proving the test above measures something real: the
    /// Raw rule *does* change its mind under a pure rescaling of Q.
    #[test]
    fn raw_puct_is_not_invariant_to_affine_rescaling() {
        // High prior on the *low*-Q child, so growing the Q gap can overtake it.
        // (With the prior on the high-Q child, that child wins at every scale
        // and the control proves nothing — which is how this test first failed.)
        let priors = [20.0f32, 1.0];
        let pick = |alpha: i64| {
            [0i64, 30]
                .iter()
                .zip(priors.iter())
                .enumerate()
                .map(|(i, (q, p))| {
                    let sc = BbScore { visits: AtomicU32::new(1), score: alpha * q, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };
                    (i, puct_value(Some(&sc), 100.0, *p, true, 0.0, PUCT_C))
                })
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .unwrap()
                .0
        };
        assert_ne!(pick(1), pick(100), "Raw was expected to be scale-sensitive — if this fails the control is broken, not the code");
    }

    /// The floor stops a node whose children differ only by positional drift
    /// from having that drift stretched to the full decision range.
    #[test]
    fn normalised_range_floor_prevents_noise_amplification() {
        let pick = |scores: [i64; 3]| {
            let priors = [1.0f32, 1.0, 10.0]; // child 2 is the domain-good one
            let lo = scores.iter().map(|q| *q as f32).fold(f32::INFINITY, f32::min);
            let hi = scores.iter().map(|q| *q as f32).fold(f32::NEG_INFINITY, f32::max);
            let frame = QFrame::new(lo, hi, priors.iter().sum(), lo, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
            scores
                .iter()
                .zip(priors.iter())
                .enumerate()
                .map(|(i, (q, p))| {
                    // 50 visits each: past the first-visit sweep, where the
                    // Q term is what apportions further visits. At 1 visit the
                    // bonus swamps everything in both regimes and the test
                    // measures nothing.
                    let sc = BbScore { visits: AtomicU32::new(50), score: *q, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };
                    (i, puct_value_normalised(Some(&sc), 400.0, *p, true, lo, &frame))
                })
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .unwrap()
                .0
        };
        // Drift-sized spread (below the 50 floor): the prior should still lead.
        assert_eq!(pick([6, 3, 0]), 2, "below the floor, a 6-point spread must not out-vote a 10x prior");
        // Real spread: Q leads despite the weaker prior.
        assert_eq!(pick([600, 300, 0]), 0, "above the floor, a 600-point lead must win");
    }

    /// Virtual loss costs the same fraction of the decision range whatever the
    /// node's raw scale — the dual of the invariance property, and the thing a
    /// raw 30-point penalty gets catastrophically wrong on a 6-point range.
    #[test]
    fn normalised_virtual_loss_is_a_fixed_fraction_of_the_range() {
        for (lo, hi) in [(0.0f32, 6.0f32), (0.0, 1058.0)] {
            let frame = QFrame::new(lo, hi, 1.0, lo, 1.0, DEFAULT_Q_RANGE_FLOOR, DEFAULT_VIRTUAL_LOSS);
            let mk = |vl: i32| BbScore { visits: AtomicU32::new(5), score: hi as i64, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(vl) };
            let v0 = puct_value_normalised(Some(&mk(0)), 100.0, 1.0, true, lo, &frame);
            let v1 = puct_value_normalised(Some(&mk(1)), 100.0, 1.0, true, lo, &frame);
            let v3 = puct_value_normalised(Some(&mk(3)), 100.0, 1.0, true, lo, &frame);
            assert!((v0 - v1 - 0.1).abs() < 1e-5, "range {lo}..{hi}: one VL should cost 0.1, got {}", v0 - v1);
            // Accumulates linearly and is NOT re-clamped, so concurrent
            // descents past the first stay visible.
            assert!((v0 - v3 - 0.3).abs() < 1e-5, "range {lo}..{hi}: three VL should cost 0.3, got {}", v0 - v3);
        }
    }

    /// Degenerate frames must stay finite, and a fresh node (nothing scored)
    /// must pick exactly what Raw picks — a constant cancels under argmax.
    #[test]
    fn normalised_frame_degenerate_cases() {
        let priors = [1.0f32, 10.0, 0.2];
        // No scored sibling: lo=+inf, hi=-inf -> frame anchors on fpu.
        let frame = QFrame::new(f32::INFINITY, f32::NEG_INFINITY, priors.iter().sum(), 525.0, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
        assert!(frame.denom >= DEFAULT_Q_RANGE_FLOOR && frame.denom.is_finite());
        let norm_pick = priors
            .iter()
            .enumerate()
            .map(|(i, p)| (i, puct_value_normalised(None, 100.0, *p, true, 525.0, &frame)))
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0;
        let raw_pick = priors
            .iter()
            .enumerate()
            .map(|(i, p)| (i, puct_value(None, 100.0, *p, true, 525.0, PUCT_C)))
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap()
            .0;
        assert_eq!(norm_pick, raw_pick, "a fresh node must rank identically in both modes");

        // All children equal (range 0) and a single child: finite, no NaN.
        for (lo, hi) in [(42.0f32, 42.0f32), (7.0, 7.0)] {
            let f = QFrame::new(lo, hi, 1.0, lo, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
            let sc = BbScore { visits: AtomicU32::new(3), score: lo as i64, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };
            let v = puct_value_normalised(Some(&sc), 100.0, 1.0, true, lo, &f);
            assert!(v.is_finite(), "range {lo}..{hi} produced {v}");
            assert!((f.norm(lo) - 0.0).abs() < 1e-6);
        }
    }

    /// A stale-high parent Q must not let unexplored children run away with
    /// the node. Clamping is what makes the frame robust to that.
    #[test]
    fn normalised_fpu_outside_child_range_is_clamped() {
        let frame = QFrame::new(0.0, 100.0, 11.0, 10_000.0, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
        assert!((frame.norm(10_000.0) - 1.0).abs() < 1e-6, "fpu above hi must clamp to 1.0");
        assert!((frame.norm(-9_000.0) - 0.0).abs() < 1e-6, "fpu below lo must clamp to 0.0");
        // What clamping actually buys: a stale-high parent Q confers *no* Q
        // advantage over the best explored sibling. Both land on 1.0, so the
        // Q terms cancel and only priors and visits separate them.
        //
        // Unclamped, `norm(10_000)` would be 100.0 against a range of 100 —
        // putting unexplored children a hundred range-units clear and making
        // selection permanently first-play for the rest of the node's life.
        assert!(
            (frame.norm(10_000.0) - frame.norm(100.0)).abs() < 1e-6,
            "a stale-high fpu must not outrank the best explored sibling on Q"
        );
        assert!((10_000.0 - frame.lo) / frame.denom > 50.0, "fixture must be genuinely stale for the clamp to matter");

        // NOTE deliberately not asserted: that an explored child outranks an
        // unexplored one here. With the self-consistent `fpu == hi` that
        // `backprop_scores` produces, an unexplored child ties on Q and wins on
        // the bonus, for all c > 0, in BOTH modes -- the first-visit sweep is
        // breadth-first by construction. That is FPU with no *reduction*
        // (Leela/KataGo subtract `c_fpu * sqrt(sum visited priors)`), which is
        // only expressible once Q is normalised. Successor experiment, not this
        // change -- see `raw_fpu_beats_parent_level_sibling_for_any_c`.
    }

    /// Priors are renormalised in-node, so scaling every prior by a constant
    /// changes nothing. Without this `c` would still absorb the branching
    /// factor, which is the same disease in a different variable.
    #[test]
    fn normalised_priors_are_renormalised_within_the_node() {
        let scores = [0i64, 400, 1000];
        let base = [1.0f32, 5.0, 0.2];
        let pick = |k: f32| {
            let priors: Vec<f32> = base.iter().map(|p| p * k).collect();
            let frame = QFrame::new(0.0, 1000.0, priors.iter().sum(), 0.0, 1.0, DEFAULT_Q_RANGE_FLOOR, 0);
            scores
                .iter()
                .zip(priors.iter())
                .enumerate()
                .map(|(i, (q, p))| {
                    let sc = BbScore { visits: AtomicU32::new(4), score: *q, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };
                    (i, puct_value_normalised(Some(&sc), 100.0, *p, true, 0.0, &frame))
                })
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .unwrap()
                .0
        };
        assert_eq!(pick(1.0), pick(37.0), "scaling every prior must not move the argmax");
    }

    /// Guards the refactor itself: `puct_value(.., PUCT_C)` must reproduce the
    /// historical `PUCT_C * prior * parent_term` expression exactly, bit for
    /// bit. This is the only thing standing between the parameterisation and a
    /// silent change to every result committed so far.
    #[test]
    fn raw_mode_is_bit_identical_to_the_historical_formula() {
        for (score, visits, prior, parent_visits, fpu) in [
            (525i64, 30u32, 5.0f32, 100.0f32, 525.0f32),
            (-1069, 1, 0.2, 1.0, 0.0),
            (0, 7, 1.0, 1000.0, -50.0),
            (1058, 512, 10.0, 16000.0, 1058.0),
        ] {
            let s = BbScore { visits: AtomicU32::new(visits), score, node_kind: BbPlayer::Home, virtual_loss: AtomicI32::new(0) };
            let parent_term = parent_visits.max(1.0).sqrt();
            let expected_some = (score as f32 - 0.0) + PUCT_C * prior * parent_term / (1.0 + visits as f32);
            let expected_none = fpu + PUCT_C * prior * parent_term;
            assert_eq!(
                puct_value(Some(&s), parent_visits, prior, true, fpu, PUCT_C).to_bits(),
                expected_some.to_bits(),
                "explored branch drifted for score={score} visits={visits}"
            );
            assert_eq!(
                puct_value(None, parent_visits, prior, true, fpu, PUCT_C).to_bits(),
                expected_none.to_bits(),
                "unexplored branch drifted for prior={prior}"
            );
        }
    }
}
