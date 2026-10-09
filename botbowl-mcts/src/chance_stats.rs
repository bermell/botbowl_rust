//! Plan 061 (d): how often each kind of roll occurs, in search and in play.
//!
//! In search, [`CHANCE_STATS`] counts per [`RollKind`] the chance nodes `available_actions`
//! expanded (and the outcome children it gave them) and the descents `select_node` sent through a
//! chance node, plus how the chance backup went (withheld, emitted complete, emitted on part of
//! the mass). Process-wide relaxed counters, always on, like `LEAF_STATS`; printed as one
//! `MCTS_CHANCE_STATS` line wherever `MCTS_LEAF_STATS` is (`BLOOD_MCTS_STATS=1` /
//! `BLOOD_MCTS_LEAF_STATS=1`), cumulative over the process.
//!
//! In play, `botbowl-ui roll-census` replays a corpus and classifies every roll the engine
//! resolved with the same [`RollKind::of`].

use std::sync::atomic::{AtomicU64, Ordering};

use botbowl_engine::core::dices::RequestedRoll;
use botbowl_engine::core::gamestate::GameState;

/// What a roll is *for*: the request type, split by the procedure asking where one request type
/// serves several rules (a D8 bounces a ball or scatters it in the weather; a deviate is a
/// kickoff's or a wildly inaccurate pass's; a 2D6 is the kickoff table's or the weather's).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RollKind {
    /// Block dice. Enumerated by resolved outcome.
    Block,
    /// Pickup / dodge / GFI / catch / ... (`D6PassFail`). Pass + fail.
    PassFail,
    /// `Sum2D6PassFail` (armour). Pass + fail.
    PassFail2d6,
    /// The injury roll (`D6ThreeOutcomes` / `Sum2D6ThreeOutcomes`). Up to three children.
    Injury,
    /// The pass roll (a raw D6). One child per face, merged.
    PassRoll,
    FoulArmor,
    FoulInjury,
    /// A D8 under `Bounce`: a live ball bounce (enumerated, see `BounceModel`).
    Bounce,
    /// A D8 under `Bounce` with the kickoff flag: a kickoff bounce.
    KickBounce,
    /// Any other D8 (the weather's scatter of a landed kickoff). Scripted.
    D8Other,
    /// A wildly inaccurate pass's deviate. Scripted unless `PassScatterModel::Grouped`.
    PassDeviate,
    /// The kickoff's deviate. Scripted.
    KickoffDeviate,
    /// An inaccurate pass's scatter (three D8). Scripted unless `PassScatterModel::Grouped`.
    PassScatter,
    /// A throw-in. Scripted unless `ThrowInModel::Grouped`.
    ThrowIn,
    /// The kickoff table's 2D6. Scripted.
    KickoffTable,
    /// The weather's 2D6. Scripted.
    Weather,
    /// The coin toss. Scripted.
    Coin,
}

impl RollKind {
    pub const ALL: [RollKind; 17] = [
        RollKind::Block,
        RollKind::PassFail,
        RollKind::PassFail2d6,
        RollKind::Injury,
        RollKind::PassRoll,
        RollKind::FoulArmor,
        RollKind::FoulInjury,
        RollKind::Bounce,
        RollKind::KickBounce,
        RollKind::D8Other,
        RollKind::PassDeviate,
        RollKind::KickoffDeviate,
        RollKind::PassScatter,
        RollKind::ThrowIn,
        RollKind::KickoffTable,
        RollKind::Weather,
        RollKind::Coin,
    ];

    /// Classify the roll `state` is paused on. Reads only the request and the procedure on top
    /// of the stack.
    pub fn of(state: &GameState, req: &RequestedRoll) -> RollKind {
        use botbowl_engine::core::procedures::AnyProc;
        match req {
            RequestedRoll::BlockDice(_) => RollKind::Block,
            RequestedRoll::Coin => RollKind::Coin,
            RequestedRoll::D6 => RollKind::PassRoll,
            RequestedRoll::D6PassFail(_) => RollKind::PassFail,
            RequestedRoll::Sum2D6PassFail(_) => RollKind::PassFail2d6,
            RequestedRoll::D6ThreeOutcomes(..) | RequestedRoll::Sum2D6ThreeOutcomes(..) => RollKind::Injury,
            RequestedRoll::FoulArmor(_) => RollKind::FoulArmor,
            RequestedRoll::FoulInjury(..) => RollKind::FoulInjury,
            RequestedRoll::D8 => match state.proc_stack_peek() {
                Some(AnyProc::Bounce(b)) if b.is_kick() => RollKind::KickBounce,
                Some(AnyProc::Bounce(_)) => RollKind::Bounce,
                _ => RollKind::D8Other,
            },
            RequestedRoll::Deviate => match state.proc_stack_peek() {
                Some(AnyProc::Pass(_)) => RollKind::PassDeviate,
                _ => RollKind::KickoffDeviate,
            },
            RequestedRoll::Scatter => RollKind::PassScatter,
            RequestedRoll::ThrowIn => RollKind::ThrowIn,
            RequestedRoll::Sum2D6 => match state.proc_stack_top() {
                Some("ChangingWeather") => RollKind::Weather,
                _ => RollKind::KickoffTable,
            },
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            RollKind::Block => "block",
            RollKind::PassFail => "pass_fail",
            RollKind::PassFail2d6 => "pass_fail_2d6",
            RollKind::Injury => "injury",
            RollKind::PassRoll => "pass_roll",
            RollKind::FoulArmor => "foul_armor",
            RollKind::FoulInjury => "foul_injury",
            RollKind::Bounce => "bounce",
            RollKind::KickBounce => "kick_bounce",
            RollKind::D8Other => "d8_other",
            RollKind::PassDeviate => "pass_deviate",
            RollKind::KickoffDeviate => "kickoff_deviate",
            RollKind::PassScatter => "pass_scatter",
            RollKind::ThrowIn => "throw_in",
            RollKind::KickoffTable => "kickoff_table",
            RollKind::Weather => "weather",
            RollKind::Coin => "coin",
        }
    }

    fn index(self) -> usize {
        RollKind::ALL.iter().position(|k| *k == self).unwrap()
    }
}

const N_KINDS: usize = RollKind::ALL.len();
#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);

/// See the module doc. Every field is a plain relaxed counter.
pub struct ChanceStats {
    /// Chance nodes expanded (`available_actions` on a pending roll), per kind.
    created: [AtomicU64; N_KINDS],
    /// Outcome children those expansions produced, per kind.
    outcomes: [AtomicU64; N_KINDS],
    /// Descents through a chance node (`select_node` on a pending roll), per kind.
    visited: [AtomicU64; N_KINDS],
    /// Chance backups that returned no value (the mode's coverage not reached yet).
    pub backup_withheld: AtomicU64,
    /// Chance backups emitted with every outcome scored.
    pub backup_complete: AtomicU64,
    /// Chance backups emitted on part of the probability mass (`ChanceBackup` other than
    /// `Complete`).
    pub backup_partial: AtomicU64,
}

impl ChanceStats {
    pub const fn new() -> Self {
        ChanceStats {
            created: [ZERO; N_KINDS],
            outcomes: [ZERO; N_KINDS],
            visited: [ZERO; N_KINDS],
            backup_withheld: AtomicU64::new(0),
            backup_complete: AtomicU64::new(0),
            backup_partial: AtomicU64::new(0),
        }
    }

    pub fn record_created(&self, kind: RollKind, n_outcomes: usize) {
        self.created[kind.index()].fetch_add(1, Ordering::Relaxed);
        self.outcomes[kind.index()].fetch_add(n_outcomes as u64, Ordering::Relaxed);
    }

    pub fn record_visit(&self, kind: RollKind) {
        self.visited[kind.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// `(created, outcomes, visited)` for one kind.
    pub fn get(&self, kind: RollKind) -> (u64, u64, u64) {
        let i = kind.index();
        (
            self.created[i].load(Ordering::Relaxed),
            self.outcomes[i].load(Ordering::Relaxed),
            self.visited[i].load(Ordering::Relaxed),
        )
    }

    /// One `MCTS_CHANCE_STATS` line: `kind=created/outcomes/visited` for every kind seen, then the
    /// backup tally. Cumulative over the process.
    pub fn summary(&self) -> String {
        let mut parts = vec!["MCTS_CHANCE_STATS".to_string()];
        for k in RollKind::ALL {
            let (c, o, v) = self.get(k);
            if c + v > 0 {
                parts.push(format!("{}={c}/{o}/{v}", k.name()));
            }
        }
        let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
        parts.push(format!(
            "backup_withheld={} backup_complete={} backup_partial={}",
            load(&self.backup_withheld),
            load(&self.backup_complete),
            load(&self.backup_partial)
        ));
        parts.join(" ")
    }
}

impl Default for ChanceStats {
    fn default() -> Self {
        Self::new()
    }
}

/// The process-wide tally.
pub static CHANCE_STATS: ChanceStats = ChanceStats::new();

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::dices::{D6Target, RollResult};
    use botbowl_engine::core::gamestate::{DiceMode, GameStateBuilder};
    use botbowl_engine::core::model::{Action, Position, SomeProcInput};
    use botbowl_engine::core::table::{PosAT, SimpleAT};

    #[test]
    fn a_failed_pickup_bounce_is_a_bounce_and_a_plain_d6_is_the_pass_roll() {
        let (start, ball) = (Position::new((4, 5)), Position::new((5, 5)));
        let mut s = GameStateBuilder::new()
            .add_home_player(start)
            .add_ball_pos(ball)
            .build();
        s.set_dice_mode(DiceMode::RegisterRolls);
        s.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartMove, start)));
        s.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Move, ball)));
        let req = s.pending_roll.unwrap();
        assert_eq!(RollKind::of(&s, &req), RollKind::PassFail);
        s.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Fail));
        s.step_with_roll_or_action(SomeProcInput::Action(Action::Simple(SimpleAT::DontUseReroll)));
        assert_eq!(RollKind::of(&s, &RequestedRoll::D8), RollKind::Bounce);
        // Off the bounce, a D8 is the weather's.
        let plain = GameStateBuilder::new().add_home_player(start).build();
        assert_eq!(RollKind::of(&plain, &RequestedRoll::D8), RollKind::D8Other);
        assert_eq!(RollKind::of(&plain, &RequestedRoll::D6), RollKind::PassRoll);
        assert_eq!(
            RollKind::of(&plain, &RequestedRoll::D6PassFail(D6Target::FourPlus)),
            RollKind::PassFail
        );
        assert_eq!(RollKind::of(&plain, &RequestedRoll::Deviate), RollKind::KickoffDeviate);
    }

    #[test]
    fn counters_accumulate_per_kind_and_the_summary_names_them() {
        let stats = ChanceStats::new();
        stats.record_created(RollKind::ThrowIn, 1);
        stats.record_created(RollKind::ThrowIn, 1);
        stats.record_created(RollKind::Bounce, 7);
        stats.record_visit(RollKind::Bounce);
        assert_eq!(stats.get(RollKind::ThrowIn), (2, 2, 0));
        assert_eq!(stats.get(RollKind::Bounce), (1, 7, 1));
        let line = stats.summary();
        assert!(line.starts_with("MCTS_CHANCE_STATS "), "{line}");
        assert!(
            line.contains("throw_in=2/2/0") && line.contains("bounce=1/7/1"),
            "{line}"
        );
        assert!(!line.contains("block="), "unseen kinds stay out: {line}");
    }
}
