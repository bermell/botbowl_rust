//! Plan 060: the shape of a search — how deep its descents go and into whose turn.
//!
//! Collected **per descent** through `recon_mcts`'s `GameDynamics::observe_descent` hook: each
//! descent hands over its edges (with the player of the node each one leaves), the state it
//! stopped at and why it stopped. [`DescentLog`] folds those into histograms; [`main_line`] walks
//! the finished tree; [`opp_turn_follows`] says whether the opponent has a turn inside the horizon
//! at all. The result is a [`TreeStats`] on the training sample and a running total in
//! [`crate::telemetry::TreeTelemetry`].
//!
//! Observational only: nothing here feeds back into selection, scoring or the move played.

use std::sync::atomic::Ordering;
use std::sync::Mutex;

use botbowl_data::{DepthStats, LeafValueCounts, MainLine, Phase, PhaseCounts, TreeStats};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{other_team, TeamType};
use botbowl_engine::core::procedures::TURNS_PER_HALF;
use recon_mcts::{ArcWrap, DescentEnd, NodeAlias, StateMemory};

use crate::action::{BbAction, BbPlayer};
use crate::dynamics::{BloodBowlDynamics, HorizonAnchor};

/// The search player a team's decisions are tagged with.
pub fn team_player(team: TeamType) -> BbPlayer {
    match team {
        TeamType::Home => BbPlayer::Home,
        TeamType::Away => BbPlayer::Away,
    }
}

/// Where `state` lies relative to the root's `anchor`, and whether the opponent's following turn
/// has begun by then.
///
/// Terminal outcomes first (game over, a score, the half over), then the turn counters: the
/// mover's counter advancing `turn_depth` times is the horizon (`HorizonAnchor::diverged`); short
/// of that, the opponent's counter having advanced further than the mover's is the opponent's turn.
pub fn leaf_phase(anchor: &HorizonAnchor, state: &GameState) -> (Phase, bool) {
    let (own, opp, own0, opp0) = match anchor.agent_team {
        TeamType::Home => (
            state.info.home_turn,
            state.info.away_turn,
            anchor.home_turn,
            anchor.away_turn,
        ),
        TeamType::Away => (
            state.info.away_turn,
            state.info.home_turn,
            anchor.away_turn,
            anchor.home_turn,
        ),
    };
    let same_half = state.info.half == anchor.half;
    let own_adv = if same_half { own.saturating_sub(own0) } else { 0 };
    let opp_adv = if same_half { opp.saturating_sub(opp0) } else { 0 };
    let reached_opp = opp_adv > 0;
    let phase = if state.info.game_over {
        Phase::GameOver
    } else if anchor.score_changed(state) {
        Phase::Score
    } else if !same_half {
        Phase::HalfEnd
    } else if own_adv >= anchor.turn_depth {
        Phase::Horizon
    } else if opp_adv > own_adv {
        Phase::OppTurn
    } else {
        Phase::OwnTurn
    };
    (phase, reached_opp)
}

/// Does the opponent get a turn before the search's horizon (the agent's `turn_depth`-th next turn)
/// and before the half ends? Mirrors the engine's `Half` procedure: a half is over once both turn
/// counters reach [`TURNS_PER_HALF`]; otherwise the next turn goes to the receiving team when the
/// counters are level and to the kicking team when they are not.
///
/// `false` at the last turn of a half for the team that kicked (the receiver has already played
/// all its turns), and for a decision taken inside the opponent's own turn.
pub fn opp_turn_follows(root: &GameState, agent: TeamType, turn_depth: u8) -> bool {
    if root.info.game_over {
        return false;
    }
    let kicking = root.kicking_this_half().unwrap_or(if root.info.half == 2 {
        other_team(root.info.kicking_first_half)
    } else {
        root.info.kicking_first_half
    });
    let (mut home, mut away) = (root.info.home_turn, root.info.away_turn);
    let mut own_turns = 0u8;
    loop {
        if home >= TURNS_PER_HALF && away >= TURNS_PER_HALF {
            return false;
        }
        let next = if home == away { other_team(kicking) } else { kicking };
        if next != agent {
            return true;
        }
        own_turns += 1;
        if own_turns >= turn_depth.max(1) {
            return false;
        }
        match next {
            TeamType::Home => home += 1,
            TeamType::Away => away += 1,
        }
    }
}

/// Which [`LeafValueCounts`] slot a descent's leaf belongs in.
fn valued_slot<'c>(counts: &'c mut LeafValueCounts, end: DescentEnd, phase: Phase, leaf: &GameState) -> &'c mut u32 {
    match (end, phase) {
        (DescentEnd::Solved, _) => &mut counts.solved,
        (_, Phase::Horizon) => &mut counts.horizon,
        (_, Phase::Score | Phase::HalfEnd | Phase::GameOver) => &mut counts.terminal,
        (DescentEnd::Terminal, _) => &mut counts.terminal,
        (DescentEnd::Expanded, _) if leaf.pending_roll.is_some() => &mut counts.chance,
        (DescentEnd::Expanded, _) => &mut counts.new_leaf,
    }
}

/// Exact histogram of a small non-negative integer (a depth in plies).
#[derive(Debug, Default, Clone)]
struct Hist {
    counts: Vec<u32>,
    n: u32,
    sum: u64,
}

impl Hist {
    fn record(&mut self, v: u32) {
        let i = v as usize;
        if i >= self.counts.len() {
            self.counts.resize(i + 1, 0);
        }
        self.counts[i] += 1;
        self.n += 1;
        self.sum += u64::from(v);
    }

    /// Nearest rank: the smallest value whose cumulative count reaches `ceil(q·n)`.
    fn percentile(&self, q: f64) -> u32 {
        let rank = ((q * f64::from(self.n)).ceil() as u32).max(1);
        let mut seen = 0;
        for (v, c) in self.counts.iter().enumerate() {
            seen += c;
            if seen >= rank {
                return v as u32;
            }
        }
        0
    }

    fn stats(&self) -> DepthStats {
        if self.n == 0 {
            return DepthStats::default();
        }
        DepthStats {
            mean: (self.sum as f64 / f64::from(self.n)) as f32,
            p90: self.percentile(0.9),
            max: self.counts.len().saturating_sub(1) as u32,
        }
    }

    fn clear(&mut self) {
        self.counts.clear();
        self.n = 0;
        self.sum = 0;
    }
}

#[derive(Debug, Default)]
struct Acc {
    /// The current search's anchor; `None` until the first [`DescentLog::reset`].
    anchor: Option<HorizonAnchor>,
    descents: u32,
    plies: Hist,
    own: Hist,
    chance_plies: u64,
    ends: PhaseCounts,
    reached_opp: u32,
    valued: LeafValueCounts,
}

/// Per-search descent tally, shared by the bot and every tree it builds (a reused tree keeps the
/// dynamics it was built with, so a log made per search would never see a reused tree's
/// descents — the same reason `RootDescents` and `ForcedRoot` live on the bot).
///
/// One mutex, taken once per descent for a few counter bumps: negligible next to the descent.
#[derive(Debug, Default)]
pub struct DescentLog {
    inner: Mutex<Acc>,
}

impl DescentLog {
    /// Start a new search rooted under `anchor` (its `agent_team` is the root's mover).
    pub fn reset(&self, anchor: HorizonAnchor) {
        let mut acc = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        acc.anchor = Some(anchor);
        acc.descents = 0;
        acc.plies.clear();
        acc.own.clear();
        acc.chance_plies = 0;
        acc.ends = PhaseCounts::default();
        acc.reached_opp = 0;
        acc.valued = LeafValueCounts::default();
    }

    /// Fold one descent in. `edges` are `(player of the node left, action)`, root first.
    pub fn record<'a>(
        &self,
        edges: impl Iterator<Item = (&'a BbPlayer, &'a BbAction)>,
        leaf: &GameState,
        end: DescentEnd,
    ) {
        let mut acc = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let Some(anchor) = acc.anchor else {
            return;
        };
        let agent = team_player(anchor.agent_team);
        let (mut plies, mut own, mut chance) = (0u32, 0u32, 0u32);
        for (player, _) in edges {
            plies += 1;
            if *player == agent {
                own += 1;
            } else if *player == BbPlayer::Chance {
                chance += 1;
            }
        }
        let (phase, reached_opp) = leaf_phase(&anchor, leaf);
        acc.descents += 1;
        acc.plies.record(plies);
        acc.own.record(own);
        acc.chance_plies += u64::from(chance);
        acc.ends.record(phase);
        acc.reached_opp += u32::from(reached_opp);
        *valued_slot(&mut acc.valued, end, phase, leaf) += 1;
    }

    /// The search's statistics so far, completed with what is read off the finished tree.
    pub fn finish(&self, main_line: MainLine, opp_turn_follows: bool, proc: Option<String>) -> TreeStats {
        let acc = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        TreeStats {
            descents: acc.descents,
            plies: acc.plies.stats(),
            own: acc.own.stats(),
            chance_plies_mean: if acc.descents == 0 {
                0.0
            } else {
                (acc.chance_plies as f64 / f64::from(acc.descents)) as f32
            },
            ends: acc.ends,
            reached_opp_turn: acc.reached_opp,
            valued: acc.valued,
            main_line,
            opp_turn_follows,
            proc,
        }
    }
}

/// The search's main line: from `root`, the most-visited child at every node — at a chance node
/// that is the most-visited outcome — until a node none of whose children has a visit.
///
/// Reads visit counts and movers in place (`Node::get_children` / `with_score`), so the walk copies
/// one state at most: none, in fact — the last node's phase is read through `with_state`.
pub fn main_line<M>(root: &ArcWrap<NodeAlias<BloodBowlDynamics, M>>, anchor: &HorizonAnchor) -> MainLine
where
    NodeAlias<BloodBowlDynamics, M>: StateMemory,
{
    let agent = team_player(anchor.agent_team);
    let mut node = ArcWrap::clone(root);
    let mut line = MainLine::default();
    // A DAG has no cycles; the bound only guards against a bug turning this into a hang.
    for _ in 0..10_000 {
        let Some(children) = node.get_children() else {
            break;
        };
        let best = children
            .into_iter()
            .map(|(_, child)| {
                let visits = child.with_score(|s| s.map_or(0, |s| s.visits.load(Ordering::Relaxed)));
                (visits, child)
            })
            .filter(|(visits, _)| *visits > 0)
            .max_by_key(|(visits, _)| *visits);
        let Some((_, child)) = best else {
            break;
        };
        line.plies += 1;
        match node.mover() {
            Some(p) if *p == agent => line.own += 1,
            Some(BbPlayer::Chance) => line.chance += 1,
            _ => {}
        }
        node = child;
    }
    line.phase = node.with_state(|s| s.map(|s| leaf_phase(anchor, s).0));
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::Position;

    /// Home to move at the start of its turn (the builder's receiving team), Away kicked.
    fn home_turn_state() -> GameState {
        GameStateBuilder::new()
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((10, 5)))
            .build()
    }

    #[test]
    fn leaf_phases_follow_the_turn_counters_and_terminal_outcomes() {
        let root = home_turn_state();
        let anchor = HorizonAnchor::capture(&root, TeamType::Home);
        assert_eq!(leaf_phase(&anchor, &root), (Phase::OwnTurn, false));

        let mut opp = root.clone();
        opp.info.away_turn += 1;
        assert_eq!(leaf_phase(&anchor, &opp), (Phase::OppTurn, true));

        let mut next_own = opp.clone();
        next_own.info.home_turn += 1;
        assert_eq!(leaf_phase(&anchor, &next_own), (Phase::Horizon, true));

        let mut td = opp.clone();
        td.away.score += 1;
        assert_eq!(
            leaf_phase(&anchor, &td),
            (Phase::Score, true),
            "a score in the opponent's turn"
        );

        let mut half = root.clone();
        half.info.half += 1;
        half.info.home_turn = 0;
        half.info.away_turn = 0;
        assert_eq!(leaf_phase(&anchor, &half).0, Phase::HalfEnd);

        let mut over = root.clone();
        over.info.game_over = true;
        assert_eq!(leaf_phase(&anchor, &over).0, Phase::GameOver);
    }

    /// A deeper horizon: the mover's second turn is still inside it, and reads as its own turn.
    #[test]
    fn a_deeper_horizon_counts_turns_by_whose_counter_moved_last() {
        let root = home_turn_state();
        let anchor = HorizonAnchor::capture_with_depth(&root, TeamType::Home, 2);
        let mut s = root.clone();
        s.info.away_turn += 1;
        assert_eq!(leaf_phase(&anchor, &s).0, Phase::OppTurn);
        s.info.home_turn += 1;
        assert_eq!(leaf_phase(&anchor, &s).0, Phase::OwnTurn);
        s.info.away_turn += 1;
        assert_eq!(leaf_phase(&anchor, &s).0, Phase::OppTurn);
        s.info.home_turn += 1;
        assert_eq!(leaf_phase(&anchor, &s).0, Phase::Horizon);
    }

    #[test]
    fn the_last_turn_of_a_half_has_no_opponent_turn_after_it() {
        let mut s = home_turn_state();
        // Home received, so Home moves first: at Home's turn 8 Away still has its turn 8 to come.
        s.info.home_turn = TURNS_PER_HALF;
        s.info.away_turn = TURNS_PER_HALF - 1;
        assert!(opp_turn_follows(&s, TeamType::Home, 1));
        // Away (the kicker) on its turn 8: both counters are full, the half is over after it.
        s.info.away_turn = TURNS_PER_HALF;
        assert!(!opp_turn_follows(&s, TeamType::Away, 1));
        // Mid-half: the other side always follows.
        s.info.home_turn = 3;
        s.info.away_turn = 2;
        assert!(opp_turn_follows(&s, TeamType::Home, 1));
        // A decision Home takes inside Away's turn 3: Home's next turn comes first.
        s.info.away_turn = 3;
        assert!(!opp_turn_follows(&s, TeamType::Home, 1));
        assert!(opp_turn_follows(&s, TeamType::Home, 2), "a deeper horizon sees past it");
    }

    #[test]
    fn the_log_counts_plies_own_decisions_chance_and_ends_per_descent() {
        let root = home_turn_state();
        let anchor = HorizonAnchor::capture(&root, TeamType::Home);
        let log = DescentLog::default();
        log.reset(anchor);
        let a = BbAction::Chance {
            result: botbowl_engine::core::dices::RollResult::Pass,
            prob_bits: 0,
        };
        // Home, chance, Home, Away: four plies, two own, one chance, ending in Away's turn.
        let edges = [BbPlayer::Home, BbPlayer::Chance, BbPlayer::Home, BbPlayer::Away];
        let mut leaf = root.clone();
        leaf.info.away_turn += 1;
        log.record(edges.iter().map(|p| (p, &a)), &leaf, DescentEnd::Expanded);
        // The root itself, expanded: zero plies.
        log.record(std::iter::empty(), &root, DescentEnd::Expanded);
        let s = log.finish(MainLine::default(), true, Some("Turn".into()));
        assert_eq!(s.descents, 2);
        assert_eq!((s.plies.mean, s.plies.p90, s.plies.max), (2.0, 4, 4));
        assert_eq!((s.own.mean, s.own.max), (1.0, 2));
        assert_eq!(s.chance_plies_mean, 0.5);
        assert_eq!((s.ends.opp_turn, s.ends.own_turn), (1, 1));
        assert_eq!(s.reached_opp_turn, 1);
        assert_eq!(s.valued.new_leaf, 2);

        // A reset starts the next search from nothing.
        log.reset(anchor);
        assert_eq!(log.finish(MainLine::default(), true, None).descents, 0);
    }

    #[test]
    fn leaves_are_valued_by_how_the_descent_ended_and_where() {
        let mut c = LeafValueCounts::default();
        let s = home_turn_state();
        *valued_slot(&mut c, DescentEnd::Solved, Phase::OwnTurn, &s) += 1;
        *valued_slot(&mut c, DescentEnd::Terminal, Phase::Horizon, &s) += 1;
        *valued_slot(&mut c, DescentEnd::Expanded, Phase::Horizon, &s) += 1;
        *valued_slot(&mut c, DescentEnd::Expanded, Phase::Score, &s) += 1;
        *valued_slot(&mut c, DescentEnd::Terminal, Phase::OwnTurn, &s) += 1;
        *valued_slot(&mut c, DescentEnd::Expanded, Phase::OppTurn, &s) += 1;
        assert_eq!(
            c,
            LeafValueCounts {
                new_leaf: 1,
                chance: 0,
                terminal: 2,
                solved: 1,
                horizon: 2
            }
        );
    }
}
