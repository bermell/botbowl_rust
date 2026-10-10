//! The bare policy as a bot, and policy-only drive playouts: what `botbowl-ui override-audit`,
//! `value-bench` and the MC value labels ([`crate::mc_label`], on one box or on a hub worker) are
//! built from. Moved here from `botbowl-ui`'s `override_audit.rs` (plan 062) so the worker plays
//! exactly the code the single-box tool plays; `override_audit` re-exports it and keeps the tests
//! that pin it (`PolicyBot` against `cfgs/policy_only.toml`, playouts against the drive benchmark).

use std::sync::Arc;

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::{Action as EngineAction, TeamType};
use botbowl_mcts::pruning::search_actions;
use botbowl_nn::eval::NnEvaluator;

use crate::drives::DriveStart;

/// The legal set a search root offers: the engine's actions minus `should_prune`, or all of them
/// when pruning would leave none, in the engine's sorted order. The search's own definition
/// (`botbowl_mcts::pruning::search_actions`), not a copy of it.
pub fn search_legal(state: &GameState) -> Vec<EngineAction> {
    search_actions(state)
}

/// Index of the first maximum. The Gumbel root sorts its candidates stably by `ln prior`, so on a
/// tie the policy-only preset plays the earliest action too.
fn first_argmax(priors: &[f32]) -> usize {
    let mut best = 0;
    for (i, p) in priors.iter().enumerate() {
        if *p > priors[best] {
            best = i;
        }
    }
    best
}

/// The bare policy, without a search: what `cfgs/policy_only.toml` (`gumbel_m = 1`, no noise)
/// plays, at one forward per decision instead of the preset's root expansion plus its descents.
/// A decision with one legal move costs no forward at all.
pub struct PolicyBot {
    nn: Arc<NnEvaluator>,
}

impl PolicyBot {
    pub fn new(nn: Arc<NnEvaluator>) -> Self {
        PolicyBot { nn }
    }

    /// The legal set, its priors (softmax × len, as the search sees them) and the argmax's index.
    pub fn priors(&self, state: &GameState) -> (Vec<EngineAction>, Vec<f32>, usize) {
        let legal = search_legal(state);
        let priors = if legal.len() > 1 {
            self.nn.priors(state, &legal)
        } else {
            vec![1.0; legal.len()]
        };
        let best = first_argmax(&priors);
        (legal, priors, best)
    }
}

impl Bot for PolicyBot {
    fn get_action(&mut self, state: &GameState) -> EngineAction {
        let (legal, _, best) = self.priors(state);
        legal[best]
    }
}

/// One playout of the rest of a drive.
#[derive(Clone, Debug, PartialEq)]
pub struct Playout {
    /// In `mover`'s frame: +1 it scored, -1 the opponent did, 0 neither (half, game end, cap).
    pub outcome: f32,
    /// The net's value at the first decision after `first`, in `mover`'s frame; the exact outcome
    /// when `first` ended the drive. `None` without a net.
    pub v_after: Option<f32>,
    /// False only when `max_steps` ran out before the drive ended.
    pub finished: bool,
    pub steps: u32,
}

/// The net's value at `state` in `team`'s frame, in TD units.
pub fn value_for(nn: &NnEvaluator, state: &GameState, team: TeamType) -> f32 {
    in_frame(team, nn.value_home_i64(state) as f32 / 1000.0)
}

/// A Home-centric number in `team`'s frame (`0.0 - x`, so a draw never prints as `-0`).
pub fn in_frame(team: TeamType, home_centric: f32) -> f32 {
    match team {
        TeamType::Home => home_centric,
        TeamType::Away => 0.0 - home_centric,
    }
}

/// Play `first` (if any) from `state`, then the rest of the drive with `home` / `away` choosing,
/// under real dice from `dice_seed`. The state is cloned: the same `(state, first, dice_seed)`
/// and deterministic bots always give the same playout, which is what pairs two moves' playouts.
#[allow(clippy::too_many_arguments)]
pub fn play_out(
    state: &GameState,
    first: Option<EngineAction>,
    mover: TeamType,
    home: &mut dyn Bot,
    away: &mut dyn Bot,
    nn: Option<&NnEvaluator>,
    dice_seed: u64,
    max_steps: u32,
) -> (Playout, GameState) {
    let mut st = state.clone();
    st.set_seed(dice_seed);
    st.set_dice_mode(DiceMode::RollDice);
    st.set_logging_state(false);
    let drive = DriveStart::of(&st);
    let mut steps = 0u32;
    if let Some(a) = first {
        st.step(a).expect("engine step failed on the audited move");
        steps += 1;
    }
    // The value after the move: read at the first decision, after the bot has asked for its
    // priors, so on the policy bot it is the same forward (the evaluator's memo).
    let mut v_after = match (first, nn) {
        (Some(_), Some(_)) if drive.over(&st) => Some(drive.outcome_for(&st, mover)),
        _ => None,
    };
    let mut want_v = first.is_some() && nn.is_some() && v_after.is_none();
    while !drive.over(&st) && steps < max_steps {
        let action = match st.available_actions.team {
            Some(TeamType::Home) => home.get_action(&st),
            Some(TeamType::Away) => away.get_action(&st),
            None => break,
        };
        if want_v {
            v_after = Some(value_for(nn.expect("want_v implies a net"), &st, mover));
            want_v = false;
        }
        st.step(action).expect("engine step failed during an audit playout");
        steps += 1;
    }
    let playout = Playout {
        outcome: drive.outcome_for(&st, mover),
        v_after,
        finished: drive.over(&st),
        steps,
    };
    (playout, st)
}
