//! `GameDynamics` implementation that plugs Blood Bowl into recon_mcts,
//! plus the MctsBot adapter that exposes the searcher as a `Bot`.
//!
//! Design mirrors the 2048 reference (`recon_mcts/tests/nim/test_mcts_2048.rs`):
//! - `Player` discriminator carries the kind of node (`Home`/`Away` action
//!   nodes vs. `Chance` outcome nodes).
//! - `Score` is a small struct with atomic visit counter, integer score,
//!   and the node-kind discriminator that `backprop_scores` switches on.
//! - `available_actions` returns engine actions for player nodes,
//!   `ChanceOutcome` choices for chance nodes (when `state.pending_roll`
//!   is `Some`).
//! - `apply_action` calls `state.micro_step(Some(a))` for player actions,
//!   queues a dice fix and calls `state.micro_step(None)` for chance
//!   actions.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, SomeProcInput, TeamType};
use recon_mcts::{GameDynamics, GetState, SearchTree, SelectNodeState, StoreState, Tree, TreeAlias};

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

/// `horizon` (None by default for backwards compatibility) bounds the
/// search depth — `available_actions` returns None as soon as a state
/// has diverged past the anchor. `MctsBot::get_action` always sets a
/// horizon; only direct callers (benches, tests that drive `Tree`
/// without `MctsBot`) see the unbounded form.
#[derive(Debug, Clone, Copy)]
pub struct BloodBowlDynamics {
    pub horizon: Option<HorizonAnchor>,
    /// Plan 015 Step 5 — magnitude of the transient `BbScore.virtual_loss`
    /// penalty applied on descent in `select_node`. Default 30, calibrated
    /// against the BB Q-scale (ball control ±50, distance ±26). Set to 0
    /// to disable. Honoured per-search; `MctsBot::new` resolves it from
    /// `BLOOD_MCTS_VIRTUAL_LOSS`.
    pub virtual_loss: i32,
}

impl Default for BloodBowlDynamics {
    fn default() -> Self {
        Self {
            horizon: None,
            virtual_loss: DEFAULT_VIRTUAL_LOSS,
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
            let req = state.pending_roll.as_ref().unwrap();
            let outcomes = roll_outcomes::enumerate(req);
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
        let actions: Vec<(BbPlayer, BbAction)> = filtered
            .into_iter()
            .map(|a| {
                let prior = prior_for_engine_action(state, a);
                (mcts_player, BbAction::player(a, prior))
            })
            .collect();
        if actions.is_empty() {
            None
        } else {
            Some(actions)
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        let mut new_state = state;
        let proc_input: SomeProcInput = match action {
            BbAction::Player {
                action: engine_action, ..
            } => SomeProcInput::Action(*engine_action),
            BbAction::Chance { outcome, .. } => {
                let req = new_state.pending_roll.as_ref().cloned()?;
                let result = roll_outcomes::result_for_outcome(&req, *outcome);
                SomeProcInput::Roll(result)
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

        // Chance node: pick the outcome whose visit count is furthest
        // below its expected count under the action's probability
        // distribution. Score for outcome i = `p_i · (N_parent + 1) - N_i`;
        // we pick the argmax. This makes the empirical visit ratio
        // converge to the real probability distribution as N grows.
        // The previous `min_by(visits)` was probability-blind and
        // would over-sample low-probability outcomes (e.g. on a 5/6
        // GFI it sampled failures 5× more often than they should be).
        // `BbAction::Chance` carries `prob_bits`; `Player` variants
        // never appear here (we're under `pending_roll.is_some()`).
        if parent_node_state.pending_roll.is_some() {
            let total = parent_visits + 1.0;
            let pick = scores_and_actions
                .clone()
                .into_iter()
                .map(|(q, a)| {
                    let v = q
                        .as_ref()
                        .as_ref()
                        .map(|s| s.visits.load(Ordering::Relaxed))
                        .unwrap_or(0) as f32;
                    let action = a.deref().clone();
                    let prob = action.prob_f32().unwrap_or(0.0);
                    let deficit = prob * total - v;
                    (deficit, action)
                })
                .max_by(|(da, _), (db, _)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
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
        let pick = scores_and_actions
            .clone()
            .into_iter()
            .map(|(q, a)| {
                let action = a.deref().clone();
                let p = action.prior_f32().unwrap_or(1.0);
                let v = puct_value(q.as_ref(), parent_visits, p, home_perspective);
                (v, action)
            })
            .max_by(|(va, _), (vb, _)| va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal))
            .expect("player node must have at least one action");
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
            // Normalise in case some outcomes haven't been visited yet.
            let avg = if total_prob > 0.0 {
                weighted_sum / total_prob
            } else {
                0.0
            };
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
        if state.pending_roll.is_some() {
            return None; // chance node — expanded, not scored
        }
        Some(BbScore {
            visits: AtomicU32::new(1),
            score: leaf_score(state),
            node_kind: player_for_state(state),
            virtual_loss: AtomicI32::new(0),
        })
    }
}

/// PUCT(a) = Q(a) + c · P(a) · √N(parent) / (1 + N(a))
///
/// Unexplored children (`score == None`) have `Q = 0` and `N(a) = 0`, so
/// their value collapses to `c · P · √N(parent)` — ranked purely by
/// prior. This replaces the pure-UCT `f32::INFINITY` sentinel for
/// unexplored children; high-prior unexplored children are still
/// preferred, but low-prior unexplored children no longer crowd out
/// well-scored explored siblings.
fn puct_value(score: Option<&BbScore>, parent_visits: f32, prior: f32, home_perspective: bool) -> f32 {
    let parent_term = parent_visits.max(1.0).sqrt();
    match score {
        None => PUCT_C * prior * parent_term,
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
            (q_perspective - vl) + PUCT_C * prior * parent_term / (1.0 + v)
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
    /// Run all workers for this many whole seconds, then stop.
    Seconds(u64),
}

pub struct MctsBot {
    pub budget: SearchBudget,
    /// Number of worker threads driving `tree.step()`. For
    /// `SearchBudget::Iterations` the total step count is split across
    /// them; for `SearchBudget::Seconds` every worker runs until the
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
        Self {
            budget,
            n_workers,
            memory_mode: MemoryMode::StoreState,
            reuse_enabled,
            cached_tree: None,
            last_anchor: None,
            virtual_loss,
        }
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
}

impl Bot for MctsBot {
    fn get_action(&mut self, state: &GameState) -> EngineAction {
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
                                            tree.step();
                                        }
                                    })
                                    .expect("failed to spawn MCTS worker thread");
                            }
                        });
                    }
                    SearchBudget::Seconds(secs) => {
                        let stop = AtomicBool::new(false);
                        let stop_ref = &stop;
                        std::thread::scope(|s| {
                            std::thread::Builder::new()
                                .name("mcts-timer".into())
                                .spawn_scoped(s, || {
                                    std::thread::sleep(Duration::from_secs(secs));
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
                        SearchBudget::Seconds(s) => format!("~{} ({}s)", hits + misses, s),
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
                let move_info = tree
                    .get_next_move_info()
                    .expect("MCTS tree has no move info at root");
                (move_info, tree)
            }};
        }

        let (move_info, cache_after) = match memory_mode {
            MemoryMode::GetState => {
                let (mi, t) = run_with_marker!(GetState, "get", GetState);
                (mi, CachedTree::GetState(t))
            }
            MemoryMode::StoreState => {
                let (mi, t) = run_with_marker!(StoreState, "store", StoreState);
                (mi, CachedTree::StoreState(t))
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

        let best = move_info
            .iter()
            .max_by_key(|(_, info)| {
                info.score
                    .as_ref()
                    .map(|s| s.visits.load(Ordering::Relaxed))
                    .unwrap_or(0)
            })
            .expect("root must offer at least one action");
        match &best.0 {
            BbAction::Player { action, .. } => *action,
            BbAction::Chance { .. } => {
                panic!("root selected a chance action — root must be a player turn");
            }
        }
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

        let va_home = puct_value(Some(&a), parent_visits, prior, true);
        let vb_home = puct_value(Some(&b), parent_visits, prior, true);
        assert!(
            va_home > vb_home,
            "Home should rank +50 above -50 (va={va_home}, vb={vb_home})"
        );

        let va_away = puct_value(Some(&a), parent_visits, prior, false);
        let vb_away = puct_value(Some(&b), parent_visits, prior, false);
        assert!(
            vb_away > va_away,
            "Away should rank -50 above +50 (va={va_away}, vb={vb_away})"
        );
    }
}
