//! Training-data trajectories (grand-plan steps 6–7).
//!
//! Drives the MCTS bot through one self-play game, one random-start drive
//! or one curriculum lecture trial, harvesting a [`botbowl_data::Sample`]
//! at every agent decision (state + raw search distribution + root value),
//! then backfills the drive/game outcome into one [`Trajectory`] stamped
//! with the git commit and board config that produced it.

use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use botbowl_curriculum::{
    available_lectures, generate_random_start, make_lecture, Difficulty, LectureContext, LectureStatus,
    RandomStartConfig,
};
use botbowl_data::{Outcome, Sample, Trajectory, TrajectoryMeta};
use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::TeamType;
use botbowl_mcts::SearchBudget;
use botbowl_nn::eval::NnEvaluator;

use crate::bots::{make_mcts, Evaluator, SearchConfig};

// Mirror the seed-mixing constants in `botbowl-curriculum`'s runner so a
// curriculum dataset trial is reproducible against `run_trials` for the
// same seed.
const OPPONENT_SEED_MIX: u64 = 0xA5A5_A5A5_A5A5_A5A5;
const AGENT_SEED_MIX: u64 = 0x5A5A_5A5A_5A5A_5A5A;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GenMode {
    /// Both teams MCTS, one full game from kickoff.
    #[default]
    SelfPlay,
    /// MCTS agent vs RandomBot on one lecture trial.
    Curriculum,
    /// Both teams MCTS, one drive from a randomized mid-game state (plan 019).
    RandomStart,
}

/// Placement biases for random-start mode (plan 020). Mirrors
/// `RandomStartConfig` plus the alternating second temperature.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomStartBias {
    pub ball_distance: f32,
    pub front_line: f32,
    pub mark_teammate: f32,
    pub mark_opponent: f32,
    pub own_side: f32,
    pub temperature: f32,
    /// Used on odd seeds, so the corpus mixes sharp and flat placements.
    pub temperature2: f32,
    pub carried_prob: f32,
    pub line_fraction: f32,
    pub pocket_fraction: f32,
}

/// The defaults `botbowl-ui dataset` and `botbowl-hub job generate` share.
impl Default for RandomStartBias {
    fn default() -> Self {
        RandomStartBias {
            ball_distance: 1.30,
            front_line: 2.20,
            mark_teammate: 1.5,
            mark_opponent: 1.5,
            own_side: 1.5,
            temperature: 0.60,
            temperature2: 1.5,
            carried_prob: 0.75,
            line_fraction: 0.80,
            pocket_fraction: 0.25,
        }
    }
}

impl RandomStartBias {
    pub fn to_config(&self) -> RandomStartConfig {
        RandomStartConfig {
            ball_distance: self.ball_distance,
            front_line: self.front_line,
            mark_teammate: self.mark_teammate,
            mark_opponent: self.mark_opponent,
            own_side: self.own_side,
            temperature: self.temperature,
            carried_prob: self.carried_prob,
            line_fraction: self.line_fraction,
            pocket_fraction: self.pocket_fraction,
            board_dims: None,
        }
    }
}

/// Everything one trajectory needs besides its seed. Serializable so a
/// hub can ship it to workers verbatim (plan 040).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateConfig {
    pub mode: GenMode,
    pub search: SearchConfig,
    pub evaluator: Evaluator,
    /// Model path, for the provenance label only; the loaded net is passed
    /// separately.
    pub model: Option<String>,
    pub max_steps: u32,
    /// Curriculum mode only.
    pub lecture: Option<String>,
    pub difficulty: Difficulty,
    /// Random-start mode only.
    pub bias: RandomStartBias,
}

/// Play one trajectory for `seed`. `Ok(None)` is reserved for modes that
/// can legitimately produce nothing; `Err` is a configuration error (an
/// unknown lecture) that will recur on every seed, so callers should stop.
pub fn play_trajectory(
    cfg: &GenerateConfig,
    nn: Option<&Arc<NnEvaluator>>,
    seed: u64,
) -> Result<Option<Trajectory>, String> {
    match cfg.mode {
        GenMode::SelfPlay => Ok(Some(self_play_trajectory(cfg, nn, seed))),
        GenMode::RandomStart => Ok(Some(random_start_trajectory(cfg, nn, seed))),
        GenMode::Curriculum => curriculum_trajectory(cfg, nn, seed),
    }
}

/// Provenance label stamped into `TrajectoryMeta.home_bot/away_bot`.
pub fn budget_label(cfg: &GenerateConfig) -> String {
    let eval = match cfg.evaluator {
        Evaluator::Heuristic => "heuristic".to_string(),
        Evaluator::PureTd => "pure-td".to_string(),
        Evaluator::Nn => format!("nn:{}", cfg.model.as_deref().unwrap_or("?")),
        Evaluator::NnValue => format!("nn-value:{}", cfg.model.as_deref().unwrap_or("?")),
    };
    // `make_mcts` takes the backup rule from `BLOOD_MCTS_BACKUP` when the
    // config leaves it unset (both sides share it); stamp it into the
    // corpus provenance when non-default so a plan-032 mean-backup corpus
    // can never be mistaken for a minimax one.
    let backup = match cfg.search.backup.unwrap_or_else(botbowl_mcts::BackupMode::from_env) {
        botbowl_mcts::BackupMode::Minimax => String::new(),
        b => format!(",{}", b.label()),
    };
    let workers = cfg.search.workers;
    match cfg.search.budget {
        SearchBudget::Time(d) => format!("mcts(time={}ms,workers={workers},eval={eval}{backup})", d.as_millis()),
        SearchBudget::Iterations(n) => format!("mcts(iters={n},workers={workers},eval={eval}{backup})"),
    }
}

/// Play `state` with MctsBot on both teams, sampling both teams'
/// decisions, until game over, the step cap, or `stop(state)` — the
/// latter lets random-start mode end the trajectory at the end of the
/// current drive instead of playing the game out.
fn mcts_vs_mcts_samples(
    state: &mut GameState,
    cfg: &GenerateConfig,
    nn: Option<&Arc<NnEvaluator>>,
    seed: u64,
    stop: impl Fn(&GameState) -> bool,
) -> Vec<Sample> {
    let mut home = make_mcts(&cfg.search, cfg.evaluator, nn);
    let mut away = make_mcts(&cfg.search, cfg.evaluator, nn);
    home.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xA));
    away.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB));

    let mut samples: Vec<Sample> = Vec::new();
    let mut steps = 0u32;

    while !state.info.game_over && !stop(state) && steps < cfg.max_steps {
        let action = match state.available_actions.team {
            Some(TeamType::Home) => {
                let (a, s) = home.get_action_with_record(state);
                samples.push(s);
                a
            }
            Some(TeamType::Away) => {
                let (a, s) = away.get_action_with_record(state);
                samples.push(s);
                a
            }
            // Under RollDice, `step` auto-resolves chance internally, so a
            // running game always presents a team to act until game-over.
            None => break,
        };
        state.step(action).expect("engine step failed during self-play");
        steps += 1;
    }
    samples
}

/// One full MctsBot-vs-MctsBot game, from kickoff.
fn self_play_trajectory(cfg: &GenerateConfig, nn: Option<&Arc<NnEvaluator>>, seed: u64) -> Trajectory {
    let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let samples = mcts_vs_mcts_samples(&mut state, cfg, nn, seed, |_| false);

    let label = budget_label(cfg);
    let meta = TrajectoryMeta::new("self-play", board_dims)
        .with_bots(label.clone(), label)
        .with_seed(seed)
        .with_extra("mode", "self-play")
        .with_extra("max_steps", cfg.max_steps.to_string());
    let outcome = Outcome::from_state(&state, None);
    Trajectory::new(meta, samples, outcome)
}

/// One MctsBot-vs-MctsBot **drive** from a randomized mid-game state
/// (plan 019): the trajectory ends when either team scores, the half
/// ends, or the game ends — never plays into the next drive. Everything
/// after the drive resolves would be downstream of self-play (correlated
/// states, bot-chosen kickoff formations) instead of the diverse random
/// placement this mode exists to provide (plan 020).
fn random_start_trajectory(cfg: &GenerateConfig, nn: Option<&Arc<NnEvaluator>>, seed: u64) -> Trajectory {
    let mut rs = cfg.bias.to_config();
    // Alternate the placement temperature per game so the corpus mixes
    // sharp (clustered) and flat (scattered) player distributions.
    if seed % 2 == 1 {
        rs.temperature = cfg.bias.temperature2;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = generate_random_start(&rs, &mut rng);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let (start_half, start_home_turn, start_away_turn) = (state.info.half, state.info.home_turn, state.info.away_turn);
    let (start_home_score, start_away_score) = (state.home.score, state.away.score);
    let start_score = format!("{start_home_score}-{start_away_score}");
    let samples = mcts_vs_mcts_samples(&mut state, cfg, nn, seed, |s| {
        s.home.score != start_home_score || s.away.score != start_away_score || s.info.half != start_half
    });

    let label = budget_label(cfg);
    let bias = &cfg.bias;
    let meta = TrajectoryMeta::new("random-start", board_dims)
        .with_bots(label.clone(), label)
        .with_seed(seed)
        .with_extra("mode", "random-start")
        .with_extra("max_steps", cfg.max_steps.to_string())
        .with_extra("ball_distance", bias.ball_distance.to_string())
        .with_extra("front_line", bias.front_line.to_string())
        .with_extra("mark_teammate", bias.mark_teammate.to_string())
        .with_extra("mark_opponent", bias.mark_opponent.to_string())
        .with_extra("own_side", bias.own_side.to_string())
        .with_extra("temperature", rs.temperature.to_string())
        .with_extra("temperature2", bias.temperature2.to_string())
        .with_extra("drive_bounded", "true".to_string())
        .with_extra("carried_prob", bias.carried_prob.to_string())
        .with_extra("line_fraction", bias.line_fraction.to_string())
        .with_extra("pocket_fraction", bias.pocket_fraction.to_string())
        .with_extra("start_half", start_half.to_string())
        .with_extra("start_home_turn", start_home_turn.to_string())
        .with_extra("start_away_turn", start_away_turn.to_string())
        .with_extra("start_score", start_score);
    let outcome = Outcome::from_state(&state, None);
    Trajectory::new(meta, samples, outcome)
}

/// One curriculum lecture trial: MctsBot agent vs RandomBot opponent.
fn curriculum_trajectory(
    cfg: &GenerateConfig,
    nn: Option<&Arc<NnEvaluator>>,
    seed: u64,
) -> Result<Option<Trajectory>, String> {
    let name = cfg
        .lecture
        .as_deref()
        .ok_or_else(|| "curriculum mode requires --lecture NAME".to_string())?;
    let difficulty = cfg.difficulty;
    let lecture = make_lecture(name, difficulty).ok_or_else(|| {
        let mut msg = format!("unknown lecture: {name:?} ({difficulty:?}).\nAvailable lectures:");
        for (n, d) in available_lectures() {
            msg.push_str(&format!("\n  --lecture {n:?} --difficulty {d:?}"));
        }
        msg
    })?;

    let agent_team = lecture.agent_team();
    let opponent_team = match agent_team {
        TeamType::Home => TeamType::Away,
        TeamType::Away => TeamType::Home,
    };

    let mut setup_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = lecture.setup(&mut setup_rng);
    state.set_logging_state(false);
    let context = LectureContext::from_state(&state);

    let mut opponent = RandomBot::new();
    opponent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ OPPONENT_SEED_MIX));
    let mut agent = make_mcts(&cfg.search, cfg.evaluator, nn);
    agent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ AGENT_SEED_MIX));

    let board_dims = state.board_dims;
    let mut samples: Vec<Sample> = Vec::new();
    let mut status = lecture.evaluate(&state, &context);
    let mut steps = 0u32;

    while status == LectureStatus::InProgress && steps < cfg.max_steps {
        let action = match state.available_actions.team {
            Some(t) if t == agent_team => {
                let (a, s) = agent.get_action_with_record(&state);
                samples.push(s);
                Some(a)
            }
            Some(t) if t == opponent_team => Some(opponent.get_action(&state)),
            Some(_) | None => None,
        };
        let Some(action) = action else { break };
        state.step(action).expect("engine step failed during lecture");
        steps += 1;
        status = lecture.evaluate(&state, &context);
    }

    let meta = TrajectoryMeta::new(lecture.name(), board_dims)
        .with_bots(budget_label(cfg), "random")
        .with_seed(seed)
        .with_extra("mode", "curriculum")
        .with_extra("difficulty", format!("{difficulty:?}"))
        .with_extra("lecture_status", format!("{status:?}"));
    let outcome = Outcome::from_state(&state, Some(format!("{status:?}")));
    Ok(Some(Trajectory::new(meta, samples, outcome)))
}
