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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, TeamType};
use recon_mcts::{GameDynamics, HashOnly, SearchTree, SelectNodeState, Tree};

use crate::action::{BbAction, BbPlayer, ChanceOutcome};
use crate::block_dice;
use crate::priors::prior_for;
use crate::pruning::should_prune;
use crate::roll_outcomes;
use crate::score::leaf_score;

/// PUCT exploration constant. Sized so that the `c · P · √N(parent) /
/// (1 + N(a))` term is comparable to leaf-score magnitudes (game score
/// ±1000, ball control ±50, carrier-distance ±26 — see `score.rs`).
/// 10.0 keeps unexplored high-prior children competitive with explored
/// children of moderate `Q` for at least the first few hundred parent
/// visits — verified empirically against the existing UCT baseline test.
const PUCT_C: f32 = 10.0;

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
}

impl Clone for BbScore {
    fn clone(&self) -> Self {
        BbScore {
            visits: AtomicU32::new(self.visits.load(Ordering::Relaxed)),
            score: self.score,
            node_kind: self.node_kind,
        }
    }
}

/// `ff_depth` is the upper bound on roll-resolution steps inside
/// `optimistic_leaf_score`. 1 reproduces the v3 single-step
/// behaviour; 8 (the default) lets multi-roll move chains (GFI then
/// pickup; pickup then dodge) resolve to a stable decision/terminal
/// state before scoring. Pure leaf-scoring effect — chance children
/// still do not enter the tree.
#[derive(Debug, Clone, Copy)]
pub struct BloodBowlDynamics {
    pub ff_depth: u8,
}

impl Default for BloodBowlDynamics {
    fn default() -> Self {
        Self { ff_depth: 8 }
    }
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

    fn available_actions(
        &self,
        _player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::ActionIter> {
        if state.info.game_over {
            return None;
        }

        // Chance node: enumerate roll outcomes.
        if state.pending_roll.is_some() {
            let req = state.pending_roll.as_ref().unwrap();
            let outcomes = roll_outcomes::enumerate(req);
            return Some(
                outcomes
                    .into_iter()
                    .map(|a| (BbPlayer::Chance, a))
                    .collect(),
            );
        }

        // Player node: copy engine's available actions.
        let team = state.available_actions.team?;
        let mcts_player = match team {
            TeamType::Home => BbPlayer::Home,
            TeamType::Away => BbPlayer::Away,
        };
        // Scripted block-die selection: if the engine is asking the
        // bot to pick which block die to apply, collapse the fan-out
        // to a single scripted choice (`block_dice::scripted_pick`).
        // MCTS never sees the other dice — they'd just burn search
        // budget on a decision the rules resolve deterministically
        // given attacker/defender skills.
        if let Some(scripted) = block_dice::scripted_pick(state) {
            return Some(vec![(mcts_player, BbAction::Player(scripted))]);
        }
        let actions: Vec<(BbPlayer, BbAction)> = state
            .available_actions
            .get_all()
            .into_iter()
            .filter(|a| !should_prune(state, a))
            .map(|a| (mcts_player, BbAction::Player(a)))
            .collect();
        if actions.is_empty() {
            None
        } else {
            Some(actions)
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        let mut new_state = state;
        match action {
            BbAction::Player(engine_action) => {
                if new_state.micro_step(Some(*engine_action)).is_err() {
                    return None;
                }
            }
            BbAction::Chance { outcome, .. } => {
                let Some(req) = new_state.pending_roll.as_ref().cloned() else {
                    return None;
                };
                let result = roll_outcomes::result_for_outcome(&req, *outcome);
                if new_state.step_with_roll(result).is_err() {
                    return None;
                }
            }
        }
        // FF is intentionally NOT enabled here in v3 — see
        // dynamics.rs PR notes. The combination of FF + chance-node
        // modeling produces both deep-tree slowdowns (5-10 ms / iter
        // vs 1 µs / iter at v1) and reconstruction panics during
        // tree drop. Instead, v3 keeps the tree shallow and gives the
        // pickup-move chance state a meaningful Q via the optimistic
        // success-outcome `score_leaf` below.
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
        let parent_visits = parent_score
            .map(|s| s.visits.load(Ordering::Relaxed))
            .unwrap_or(1) as f32;

        // Chance node: temporarily reverted to plain min-visits (v1
        // behaviour) while bisecting the v3 reconstruction panic.
        if parent_node_state.pending_roll.is_some() {
            let pick = scores_and_actions
                .clone()
                .into_iter()
                .min_by(|a, b| {
                    let va =
                        a.0.as_ref()
                            .as_ref()
                            .map(|s| s.visits.load(Ordering::Relaxed))
                            .unwrap_or(0);
                    let vb =
                        b.0.as_ref()
                            .as_ref()
                            .map(|s| s.visits.load(Ordering::Relaxed))
                            .unwrap_or(0);
                    va.cmp(&vb)
                })
                .expect("chance node must have at least one outcome");
            if let Some(s) = pick.0.as_ref() {
                s.visits.fetch_add(1, Ordering::Relaxed);
            }
            return pick.1.deref().clone();
        }

        // Player node: PUCT with domain priors. We compute priors lazily
        // here rather than caching them on `BbScore` — the cost is one
        // `prior_for` call per child per descent, which is cheap (enum
        // match + a few position comparisons) and avoids threading
        // parent-state into `score_leaf`.
        //
        // Scores are Home-centric. Home maximises PUCT; Away mirrors
        // by negating Q before adding the exploration bonus.
        let home_perspective = *parent_player == BbPlayer::Home;
        let pick = scores_and_actions
            .clone()
            .into_iter()
            .max_by(|a, b| {
                let pa = prior_for(parent_node_state, &a.1);
                let pb = prior_for(parent_node_state, &b.1);
                let va = puct_value(a.0.as_ref(), parent_visits, pa, home_perspective);
                let vb = puct_value(b.0.as_ref(), parent_visits, pb, home_perspective);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("player node must have at least one action");
        if let Some(s) = pick.0.as_ref() {
            s.visits.fetch_add(1, Ordering::Relaxed);
        }
        pick.1.deref().clone()
    }

    fn backprop_scores<II, Q, A>(
        &self,
        player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = Self::Action>,
        Q: Deref<Target = Self::Score>,
    {
        // Chance node: probability-weighted average over visited children.
        if score_current
            .map(|s| s.node_kind == BbPlayer::Chance)
            .unwrap_or(false)
        {
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
            });
        }

        // Player node. Scores are Home-centric, so Home maximises and
        // Away minimises. Visits sum across children so PUCT's
        // √N(parent) reflects total descents (matching the Chance
        // branch above).
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
        })
    }

    fn score_leaf(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        // A leaf state may be in one of four shapes:
        //   1. terminal (`game_over` set) — bare leaf_score is correct.
        //   2. player-decision (`available_actions.team` set) — bare
        //      leaf_score; the next player chooses next.
        //   3. roll pending (`pending_roll` set) — pre-roll board
        //      understates the value of e.g. Move-onto-ball because
        //      the ball is still `OnGround` until the pickup resolves.
        //   4. mid-procedure (none of the above) — engine has internal
        //      work to do (Move walks one square per micro_step; the
        //      leaf is between squares with neither a roll nor a
        //      decision yet pending).
        //
        // Shapes 3 and 4 both need FF before scoring. We optimistically
        // resolve them by simulating success outcomes inline (Pass /
        // Advance — same constants `roll_outcomes::fix_for_outcome`
        // queues for chance actions) until we hit shape 1 or 2 or the
        // `ff_depth` cap. Pessimistic outcomes still get discovered
        // when MCTS budget allows the chance child to be expanded
        // (currently no chance children — A.alt path); the initial Q
        // is high enough that MCTS will actually choose pickup /
        // dodge / GFI moves over their "safe" alternatives.
        let needs_ff = !state.info.game_over
            && (state.pending_roll.is_some() || state.available_actions.team.is_none());
        let score = if needs_ff {
            optimistic_leaf_score(state, self.ff_depth).unwrap_or_else(|| leaf_score(state))
        } else {
            leaf_score(state)
        };
        Some(BbScore {
            visits: AtomicU32::new(1),
            score,
            node_kind: player_for_state(state),
        })
    }
}

/// Forward-simulate from a transient leaf state (pending roll or
/// mid-procedure) until the engine reaches a decision or terminal
/// point, then score. Success outcomes (Pass for pass/fail rolls,
/// Advance otherwise — the constants `roll_outcomes::result_for_outcome`
/// returns for chance actions) are passed to `step_with_roll` to
/// resume the engine. Engine processing without a pending roll is
/// driven by `micro_step(None)`; with `DiceMode::RegisterRolls` (set
/// on every MCTS root state) procedures requesting a roll surface it
/// as `pending_roll` rather than consuming RNG, keeping the
/// simulation deterministic under recombination.
///
/// `max_steps` defends against an engine bug spinning forever; 8 is
/// the dynamics default and comfortably exceeds the longest known
/// transient chain in a single Move action (a few squares + pickup
/// or GFI).
///
/// Returns None if the engine errors mid-simulation; caller falls
/// back to the bare leaf score of the original (pre-simulation)
/// state in that case.
fn optimistic_leaf_score(state: &GameState, max_steps: u8) -> Option<i64> {
    let mut sim = state.clone();
    for _ in 0..max_steps {
        if sim.info.game_over {
            break;
        }
        if let Some(req) = sim.pending_roll.as_ref().cloned() {
            let outcome = if crate::action::is_pass_fail(&req) {
                ChanceOutcome::Pass
            } else {
                ChanceOutcome::Advance
            };
            let result = roll_outcomes::result_for_outcome(&req, outcome);
            if sim.step_with_roll(result).is_err() {
                return None;
            }
            continue;
        }
        if sim.available_actions.team.is_some() {
            // Decision point — stop and let the caller score this state.
            break;
        }
        // Mid-procedure: engine has internal work to do (e.g. Move
        // walking one square per micro_step).
        if sim.micro_step(None).is_err() {
            return None;
        }
    }
    Some(leaf_score(&sim))
}

/// PUCT(a) = Q(a) + c · P(a) · √N(parent) / (1 + N(a))
///
/// Unexplored children (`score == None`) have `Q = 0` and `N(a) = 0`, so
/// their value collapses to `c · P · √N(parent)` — ranked purely by
/// prior. This replaces the pure-UCT `f32::INFINITY` sentinel for
/// unexplored children; high-prior unexplored children are still
/// preferred, but low-prior unexplored children no longer crowd out
/// well-scored explored siblings.
fn puct_value(
    score: Option<&BbScore>,
    parent_visits: f32,
    prior: f32,
    home_perspective: bool,
) -> f32 {
    let parent_term = parent_visits.max(1.0).sqrt();
    match score {
        None => PUCT_C * prior * parent_term,
        Some(s) => {
            let v = s.visits.load(Ordering::Relaxed) as f32;
            let q = if home_perspective {
                s.score as f32
            } else {
                -(s.score as f32)
            };
            q + PUCT_C * prior * parent_term / (1.0 + v)
        }
    }
}

/// `Bot` adapter that drives a fresh MCTS search per call.
pub struct MctsBot {
    pub iterations_per_move: usize,
    /// Number of worker threads driving `tree.step()`. The total search
    /// budget (`iterations_per_move`) is split across them; the wall-clock
    /// goal is the win, not extra iterations. Tests that need deterministic
    /// search results should pin this to 1 via [`with_workers`].
    pub n_workers: usize,
    /// Forwarded to [`BloodBowlDynamics::ff_depth`] — see that field for
    /// semantics. Default 8.
    pub ff_depth: u8,
}

impl MctsBot {
    pub fn new(iterations_per_move: usize) -> Self {
        let n_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            iterations_per_move,
            n_workers,
            ff_depth: BloodBowlDynamics::default().ff_depth,
        }
    }

    pub fn with_workers(mut self, n_workers: usize) -> Self {
        self.n_workers = n_workers.max(1);
        self
    }

    pub fn with_ff_depth(mut self, ff_depth: u8) -> Self {
        self.ff_depth = ff_depth;
        self
    }
}

impl Bot for MctsBot {
    fn get_action(&mut self, state: &GameState) -> EngineAction {
        // Clone the state and turn on roll-by-roll stepping for the
        // search. `DiceMode::RegisterRolls` keeps `pending_roll` visible
        // on post-action states, letting `score_leaf` invoke
        // `expected_leaf_score` to give pickup / dodge / GFI chance
        // nodes a probability-weighted Q instead of the misleading
        // pre-roll value. (We tried `DiceMode::RollDice` to elide the
        // chance-node modelling, but the engine's internal roll
        // resolution per `micro_step` ran orders of magnitude slower
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
        let tree = Arc::new(Tree::new(
            BloodBowlDynamics {
                ff_depth: self.ff_depth,
            },
            HashOnly,
            root_player,
            root_state,
        ));

        let n_workers = self.n_workers.max(1);
        let base = self.iterations_per_move / n_workers;
        let rem = self.iterations_per_move % n_workers;

        std::thread::scope(|s| {
            for w in 0..n_workers {
                let iters = base + if w < rem { 1 } else { 0 };
                if iters == 0 {
                    continue;
                }
                let tree = Arc::clone(&tree);
                s.spawn(move || {
                    for _ in 0..iters {
                        tree.step();
                    }
                });
            }
        });

        let move_info = tree
            .get_next_move_info()
            .expect("MCTS tree has no move info at root");
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
            BbAction::Player(a) => *a,
            BbAction::Chance { .. } => {
                panic!("root selected a chance action — root must be a player turn");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ChanceOutcome;

    fn child(score: i64, visits: u32) -> BbScore {
        BbScore {
            visits: AtomicU32::new(visits),
            score,
            node_kind: BbPlayer::Home,
        }
    }

    fn placeholder_action() -> BbAction {
        BbAction::chance(ChanceOutcome::Pass, 1.0)
    }

    #[test]
    fn backprop_player_home_maximises_and_sums_visits() {
        let dynamics = BloodBowlDynamics::default();
        let actions = [
            placeholder_action(),
            placeholder_action(),
            placeholder_action(),
        ];
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
        let actions = [
            placeholder_action(),
            placeholder_action(),
            placeholder_action(),
        ];
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

    #[test]
    fn puct_mirrors_for_away_player() {
        let a = BbScore {
            visits: AtomicU32::new(10),
            score: 50,
            node_kind: BbPlayer::Home,
        };
        let b = BbScore {
            visits: AtomicU32::new(10),
            score: -50,
            node_kind: BbPlayer::Home,
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
