use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::model::TeamType;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::lecture::{Lecture, LectureContext, LectureStatus};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrialStats {
    pub trials: u32,
    pub successes: u32,
    pub failures: u32,
    pub timeouts: u32,
}

impl TrialStats {
    pub fn success_rate(&self) -> f64 {
        if self.trials == 0 {
            0.0
        } else {
            self.successes as f64 / self.trials as f64
        }
    }
}

/// Run `n_trials` of `lecture` with `agent` controlling `lecture.agent_team()`.
/// The opposing team is driven by an internal `RandomBot`. Each trial is
/// seeded deterministically from `seed`.
///
/// A trial ends when `lecture.evaluate` returns `Success` or `Failure`, or
/// when `max_steps_per_trial` `micro_step`s have been taken (counted as a
/// timeout, which is also a failure for the success-rate calculation).
pub fn run_trials(
    lecture: &dyn Lecture,
    agent: &mut dyn Bot,
    n_trials: u32,
    seed: u64,
    max_steps_per_trial: u32,
) -> TrialStats {
    let agent_team = lecture.agent_team();
    let opponent_team = match agent_team {
        TeamType::Home => TeamType::Away,
        TeamType::Away => TeamType::Home,
    };

    let mut stats = TrialStats::default();

    for trial_idx in 0..n_trials {
        let trial_seed = seed.wrapping_add(trial_idx as u64);
        let mut setup_rng = ChaCha8Rng::seed_from_u64(trial_seed);
        let mut state = lecture.setup(&mut setup_rng);
        let context = LectureContext::from_state(&state);

        let mut opponent = RandomBot::new();
        opponent.set_seed(ChaCha8Rng::seed_from_u64(trial_seed ^ 0xA5A5_A5A5_A5A5_A5A5));
        agent.set_seed(ChaCha8Rng::seed_from_u64(trial_seed ^ 0x5A5A_5A5A_5A5A_5A5A));

        let mut outcome = LectureStatus::InProgress;
        for _ in 0..max_steps_per_trial {
            outcome = lecture.evaluate(&state, &context);
            if outcome != LectureStatus::InProgress {
                break;
            }
            let action = match state.available_actions.team {
                Some(t) if t == agent_team => Some(agent.get_action(&state)),
                Some(t) if t == opponent_team => Some(opponent.get_action(&state)),
                Some(_) | None => None,
            };
            state.micro_step(action).unwrap();
        }
        if outcome == LectureStatus::InProgress {
            outcome = lecture.evaluate(&state, &context);
        }

        stats.trials += 1;
        match outcome {
            LectureStatus::Success => stats.successes += 1,
            LectureStatus::Failure => stats.failures += 1,
            LectureStatus::InProgress => stats.timeouts += 1,
        }
    }

    stats
}
