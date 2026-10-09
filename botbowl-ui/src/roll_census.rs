//! `botbowl-ui roll-census` (plan 061 §2): how often real games roll each kind of die.
//!
//! The search scripts some rolls to one outcome (scatter, deviate, throw-in) and used to drop the
//! ball bouncing onto a player. Whether that matters depends on how often those rolls happen in
//! the games the loop actually plays, which the corpus records only implicitly: a trajectory holds
//! the decisions, not the dice. So this replays each random-start trajectory from its seed under
//! its own dice (`GameState::step_observing_rolls`, which draws the same dice `step` does), checks
//! every recorded state on the way, and classifies each roll the engine resolved with the same
//! [`RollKind::of`] the in-search counters use.
//!
//! For every live ball bounce it also records where the ball could go — out of bounds, an empty
//! square, a standing player (a catch attempt) or a player on the ground (it bounces on) — as
//! probability mass, the share the old bounce model dropped or renormalised away.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

use botbowl_data::Trajectory;
use botbowl_engine::core::dices::RequestedRoll;
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::Direction;
use botbowl_engine::core::procedures::AnyProc;
use botbowl_mcts::chance_stats::RollKind;
use botbowl_play::drives::position_state;

use crate::cli::RollCensusArgs;
use crate::override_audit::bias_of;

#[derive(Default, Debug, serde::Serialize)]
struct Census {
    trajectories: u64,
    skipped: u64,
    diverged: u64,
    decisions: u64,
    rolls: BTreeMap<&'static str, u64>,
    /// Live (non-kickoff) bounces: summed probability mass of each kind of landing.
    bounce_mass: BTreeMap<&'static str, f64>,
    /// Throw-ins: thrown from where the ball went out (first throw of a chain) vs a re-throw after
    /// one landed out of bounds, vs one whose ball already bounced through 2+ squares.
    throw_in: BTreeMap<&'static str, u64>,
}

impl Census {
    fn observe(&mut self, s: &GameState, req: RequestedRoll) {
        let kind = RollKind::of(s, &req);
        *self.rolls.entry(kind.name()).or_default() += 1;
        match kind {
            RollKind::Bounce => {
                let Some(ball) = s.get_ball_position() else { return };
                for dir in Direction::all_directions_as_array() {
                    let to = ball + dir;
                    let class = if s.is_out(to) {
                        "out"
                    } else {
                        match s.get_player_at(to) {
                            None => "empty",
                            Some(p) if p.can_catch() => "standing_player",
                            Some(_) => "player_down",
                        }
                    };
                    *self.bounce_mass.entry(class).or_default() += 1.0 / 8.0;
                }
            }
            RollKind::ThrowIn => {
                let rethrow = match s.proc_stack_peek() {
                    Some(AnyProc::ThrowIn(t)) => s.get_ball_position() != Some(t.origin()),
                    _ => false,
                };
                let class = if rethrow {
                    "rethrow"
                } else if s.bounce_squares.len() > 1 {
                    "late_in_chain"
                } else {
                    "first"
                };
                *self.throw_in.entry(class).or_default() += 1;
            }
            _ => {}
        }
    }

    fn replay(&mut self, traj: &Trajectory) {
        let start = traj
            .meta
            .seed
            .ok_or(())
            .and_then(|seed| bias_of(&traj.meta).map(|b| (b, seed)).map_err(|_| ()));
        let Ok((bias, seed)) = start else {
            self.skipped += 1;
            return;
        };
        let mut state = position_state(&bias, traj.meta.board_dims, seed);
        let mut census = Census::default();
        for sample in &traj.samples {
            if state != sample.state {
                self.diverged += 1;
                return;
            }
            census.decisions += 1;
            if state
                .step_observing_rolls(sample.chosen_action, |s, req| census.observe(s, req))
                .is_err()
            {
                self.diverged += 1;
                return;
            }
        }
        self.trajectories += 1;
        self.decisions += census.decisions;
        for (k, v) in census.rolls {
            *self.rolls.entry(k).or_default() += v;
        }
        for (k, v) in census.bounce_mass {
            *self.bounce_mass.entry(k).or_default() += v;
        }
        for (k, v) in census.throw_in {
            *self.throw_in.entry(k).or_default() += v;
        }
    }

    fn print(&self) {
        let drives = self.trajectories.max(1) as f64;
        let decisions = self.decisions.max(1) as f64;
        println!(
            "roll census: {} drives replayed ({} skipped, {} diverged), {} decisions",
            self.trajectories, self.skipped, self.diverged, self.decisions
        );
        println!("{:<16} {:>9} {:>10} {:>14}", "kind", "rolls", "per drive", "per decision");
        let mut rows: Vec<_> = self.rolls.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1));
        for (k, n) in rows {
            println!(
                "{k:<16} {n:>9} {:>10.3} {:>14.4}",
                *n as f64 / drives,
                *n as f64 / decisions
            );
        }
        let bounce_total: f64 = self.bounce_mass.values().sum();
        if bounce_total > 0.0 {
            let parts: Vec<String> = self
                .bounce_mass
                .iter()
                .map(|(k, v)| format!("{k} {:.3}", v / bounce_total))
                .collect();
            println!("live bounce landing mass: {}", parts.join(", "));
        }
        if !self.throw_in.is_empty() {
            let parts: Vec<String> = self.throw_in.iter().map(|(k, v)| format!("{k} {v}")).collect();
            println!("throw-ins: {}", parts.join(", "));
        }
    }
}

pub fn run(args: RollCensusArgs) -> io::Result<()> {
    let mut census = Census::default();
    'files: for path in &args.corpus {
        let reader = BufReader::new(File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let traj: Trajectory = serde_json::from_str(&line)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {e}")))?;
            census.replay(&traj);
            if args.max_trajectories > 0 && census.trajectories as usize >= args.max_trajectories {
                break 'files;
            }
        }
    }
    census.print();
    if let Some(out) = &args.out {
        std::fs::write(out, serde_json::to_string_pretty(&census)?)?;
    }
    Ok(())
}
