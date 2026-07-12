use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::gamestate::GameState;
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

/// One in-progress trial of a lecture. Lets a caller drive the trial one
/// `micro_step` at a time (e.g. from a ratatui tick) instead of running the
/// whole loop inside `run_trials`.
pub struct LectureSession<'l> {
    lecture: &'l dyn Lecture,
    state: GameState,
    context: LectureContext,
    agent_team: TeamType,
    opponent_team: TeamType,
    opponent: RandomBot,
    status: LectureStatus,
    steps_taken: u32,
    max_steps: u32,
    agent_actions_taken: u32,
    /// Cap on agent `get_action` calls per trial; `None` = unlimited.
    /// Hitting the cap ends the trial as a timeout, same as `max_steps`.
    /// This is the knob that keeps expensive bots (MCTS) from burning a
    /// full search per micro-step on trials that are already lost.
    max_agent_actions: Option<u32>,
}

impl<'l> LectureSession<'l> {
    /// Build a session, mirroring the per-trial setup in `run_trials`: same
    /// setup-RNG seeding, same opponent/agent seed mixing constants.
    pub fn new(lecture: &'l dyn Lecture, seed: u64, max_steps: u32, agent: &mut dyn Bot) -> Self {
        let agent_team = lecture.agent_team();
        let opponent_team = match agent_team {
            TeamType::Home => TeamType::Away,
            TeamType::Away => TeamType::Home,
        };

        let mut setup_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut state = lecture.setup(&mut setup_rng);
        // `GameStateBuilder::build()` defaults to printing the engine log to
        // stdout. That corrupts any caller running a ratatui-style alt-screen,
        // and is just noise in headless `run_trials`. Trial drivers can still
        // read the in-memory log via `state.get_log()`.
        state.set_logging_state(false);
        let context = LectureContext::from_state(&state);

        let mut opponent = RandomBot::new();
        opponent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5_A5A5_A5A5_A5A5));
        agent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0x5A5A_5A5A_5A5A_5A5A));

        let status = lecture.evaluate(&state, &context);

        Self {
            lecture,
            state,
            context,
            agent_team,
            opponent_team,
            opponent,
            status,
            steps_taken: 0,
            max_steps,
            agent_actions_taken: 0,
            max_agent_actions: None,
        }
    }

    /// Cap agent `get_action` calls per trial. See `max_agent_actions`.
    pub fn with_max_agent_actions(mut self, max_agent_actions: Option<u32>) -> Self {
        self.max_agent_actions = max_agent_actions;
        self
    }

    pub fn state(&self) -> &GameState {
        &self.state
    }

    pub fn status(&self) -> LectureStatus {
        self.status
    }

    pub fn steps_taken(&self) -> u32 {
        self.steps_taken
    }

    pub fn agent_actions_taken(&self) -> u32 {
        self.agent_actions_taken
    }

    pub fn is_finished(&self) -> bool {
        self.status != LectureStatus::InProgress
            || self.steps_taken >= self.max_steps
            || self.max_agent_actions.is_some_and(|cap| self.agent_actions_taken >= cap)
    }

    /// Advance one micro-step. No-op once `is_finished()`. Caches the new
    /// `LectureStatus` so callers can `status()` without re-evaluating.
    pub fn step(&mut self, agent: &mut dyn Bot) {
        if self.is_finished() {
            return;
        }
        let action = match self.state.available_actions.team {
            Some(t) if t == self.agent_team => {
                self.agent_actions_taken += 1;
                Some(agent.get_action(&self.state))
            }
            Some(t) if t == self.opponent_team => Some(self.opponent.get_action(&self.state)),
            Some(_) | None => None,
        };
        self.state.step(action.unwrap()).unwrap();
        self.steps_taken += 1;
        self.status = self.lecture.evaluate(&self.state, &self.context);
    }
}

/// Per-run knobs for [`run_trials_cfg`].
#[derive(Debug, Clone, Copy)]
pub struct TrialConfig {
    pub n_trials: u32,
    pub seed: u64,
    /// Max micro-steps per trial (any actor). Exceeding = timeout.
    pub max_steps_per_trial: u32,
    /// Max agent `get_action` calls per trial; `None` = unlimited.
    /// Exceeding = timeout. The knob that bounds wall-clock for
    /// expensive bots — see `LectureSession::max_agent_actions`.
    pub max_agent_actions: Option<u32>,
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
    run_trials_cfg(
        lecture,
        agent,
        TrialConfig {
            n_trials,
            seed,
            max_steps_per_trial,
            max_agent_actions: None,
        },
    )
}

/// [`run_trials`] with the full knob set — see [`TrialConfig`].
pub fn run_trials_cfg(lecture: &dyn Lecture, agent: &mut dyn Bot, cfg: TrialConfig) -> TrialStats {
    let mut stats = TrialStats::default();

    for trial_idx in 0..cfg.n_trials {
        let trial_seed = cfg.seed.wrapping_add(trial_idx as u64);
        let mut session = LectureSession::new(lecture, trial_seed, cfg.max_steps_per_trial, agent)
            .with_max_agent_actions(cfg.max_agent_actions);
        while !session.is_finished() {
            session.step(agent);
        }

        stats.trials += 1;
        match session.status() {
            LectureStatus::Success => stats.successes += 1,
            LectureStatus::Failure => stats.failures += 1,
            LectureStatus::InProgress => stats.timeouts += 1,
        }
    }

    stats
}
