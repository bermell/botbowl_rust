//! Training-data trajectories (grand-plan steps 6–7).
//!
//! Drives the MCTS bot through one self-play game, one random-start drive
//! (plus, under `--next-drive`, the drive it sets up — plan 047) or one
//! curriculum lecture trial, harvesting a [`botbowl_data::Sample`] at every
//! agent decision (state + raw search distribution + root value), then
//! backfills the drive/game outcome into one [`Trajectory`] per drive,
//! stamped with the git commit and board config that produced it.

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
use botbowl_mcts::{MctsBot, SearchBudget, SearchTelemetry};
use botbowl_nn::eval::NnEvaluator;

use crate::board_sizes::SizeDist;
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
/// hub can ship it to workers verbatim (plan 041).
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
    /// Plan 042: which board each game is played on, drawn per game from
    /// this distribution by the game's seed. `None` keeps the process's
    /// env board (`BoardDims::from_env()`), exactly as before. Curriculum
    /// mode ignores it — lectures place on the full pitch.
    #[serde(default)]
    pub board_sizes: Option<SizeDist>,
    /// Plan 043: the name of the preset in `search.config`, for provenance only — the knobs
    /// themselves travel in `SearchConfig`. `None` when no preset was named.
    #[serde(default)]
    pub config_name: Option<String>,
    /// Plan 047 (random-start only): when the drive ends in a score, keep playing through the
    /// next drive — both setups, the kickoff and the turns until it too resolves — and write
    /// that drive as its own record. It is how kickoff setups reach the corpus: the setup
    /// samples are labelled by the drive they set up, and the roster they set up with is
    /// whatever the first drive left (injuries, knock-outs, the randomised stats).
    #[serde(default)]
    pub next_drive: bool,
}

impl GenerateConfig {
    /// The board game `seed` plays on, or `None` for the env board.
    pub fn board_for(&self, seed: u64) -> Option<botbowl_engine::core::model::BoardDims> {
        self.board_sizes.as_ref().map(|d| d.sample(seed))
    }
}

/// Play the trajectories one `seed` produces: one, or none for a mode that
/// can legitimately produce nothing, or two when a random-start drive scores
/// and [`GenerateConfig::next_drive`] carries on into the next one. `Err` is
/// a configuration error (an unknown lecture) that will recur on every seed,
/// so callers should stop.
pub fn play_trajectory(
    cfg: &GenerateConfig,
    nn: Option<&Arc<NnEvaluator>>,
    seed: u64,
) -> Result<Vec<Trajectory>, String> {
    match cfg.mode {
        GenMode::SelfPlay => Ok(vec![self_play_trajectory(cfg, nn, seed)]),
        GenMode::RandomStart => Ok(random_start_trajectory(cfg, nn, seed)),
        GenMode::Curriculum => curriculum_trajectory(cfg, nn, seed).map(|t| t.into_iter().collect()),
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
    // A named preset (plan 043) replaces the whole configuration, so the individual knobs above
    // stop describing the bot — the name is what does. Stamp it so a corpus can be traced back to
    // the configuration that generated it.
    let config = match &cfg.config_name {
        Some(name) => format!(",cfg={name}"),
        None => String::new(),
    };
    let workers = cfg.search.workers;
    // Same resolution as `make_mcts`: a pinned config's mode, else this process's environment.
    let visits = cfg
        .search
        .config
        .map_or_else(botbowl_mcts::BudgetMode::from_env, |c| c.budget_mode)
        == botbowl_mcts::BudgetMode::Visits;
    match cfg.search.budget {
        SearchBudget::Time(d) => format!(
            "mcts(time={}ms,workers={workers},eval={eval}{backup}{config})",
            d.as_millis()
        ),
        SearchBudget::Iterations(n) => {
            let unit = if visits { "visits" } else { "iters" };
            format!("mcts({unit}={n},workers={workers},eval={eval}{backup}{config})")
        }
    }
}

/// The two MctsBots of a self-play game, so a game can be played in
/// segments (plan 047: a drive, then the drive after it) by the same bots.
struct BotPair {
    home: MctsBot,
    away: MctsBot,
}

impl BotPair {
    fn new(cfg: &GenerateConfig, nn: Option<&Arc<NnEvaluator>>, seed: u64) -> Self {
        let mut home = make_mcts(&cfg.search, cfg.evaluator, nn);
        let mut away = make_mcts(&cfg.search, cfg.evaluator, nn);
        home.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xA));
        away.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB));
        BotPair { home, away }
    }

    /// Play `state` with MctsBot on both teams, sampling both teams'
    /// decisions, until game over, the step cap, or `stop(state)` — the
    /// latter lets random-start mode end a segment at the end of the
    /// current drive instead of playing the game out. The telemetry is
    /// this segment's alone.
    fn play(
        &mut self,
        state: &mut GameState,
        cfg: &GenerateConfig,
        stop: impl Fn(&GameState) -> bool,
    ) -> (Vec<Sample>, SearchTelemetry) {
        let mut samples: Vec<Sample> = Vec::new();
        let mut steps = 0u32;

        while !state.info.game_over && !stop(state) && steps < cfg.max_steps {
            let action = match state.available_actions.team {
                Some(TeamType::Home) => {
                    let (a, s) = self.home.get_action_with_record(state);
                    samples.push(s);
                    a
                }
                Some(TeamType::Away) => {
                    let (a, s) = self.away.get_action_with_record(state);
                    samples.push(s);
                    a
                }
                // Under RollDice, `step` auto-resolves chance internally, so a
                // running game always presents a team to act until game-over.
                None => break,
            };
            state.step(action).expect("engine step failed during self-play");
            // The idle side's tree from its last turn cannot be reused once the turn moves on;
            // free it now rather than at that side's next search.
            self.home.release_stale_tree(state);
            self.away.release_stale_tree(state);
            steps += 1;
        }
        // Plan 043: both bots' search health, pooled. A self-play trajectory has no "candidate"
        // side to single out, and the two are configured identically, so one number is the
        // honest summary.
        let mut telemetry = self.home.take_telemetry();
        telemetry.merge(&self.away.take_telemetry());
        (samples, telemetry)
    }
}

/// See [`BotPair::play`]; one segment with fresh bots.
fn mcts_vs_mcts_samples(
    state: &mut GameState,
    cfg: &GenerateConfig,
    nn: Option<&Arc<NnEvaluator>>,
    seed: u64,
    stop: impl Fn(&GameState) -> bool,
) -> (Vec<Sample>, SearchTelemetry) {
    BotPair::new(cfg, nn, seed).play(state, cfg, stop)
}

/// Stamp a trajectory's search health into its provenance.
///
/// `TrajectoryMeta.extra` is the documented extension point ("adding one never breaks the
/// schema"), so this needs no `FORMAT_VERSION` bump and no reader change. Flattened to a handful
/// of scalars rather than the whole nested structure — a corpus wants the rates, and the
/// per-procedure breakdown belongs in an eval report where it can be read.
fn with_telemetry(mut meta: TrajectoryMeta, t: &SearchTelemetry) -> TrajectoryMeta {
    if t.searches == 0 {
        return meta;
    }
    let r = &t.reuse.total;
    meta = meta
        .with_extra("searches", t.searches.to_string())
        .with_extra("iterations", t.iterations.to_string())
        .with_extra("reuse_reused", r.reused.to_string())
        .with_extra("reuse_anchor_miss", r.anchor_miss.to_string())
        .with_extra("reuse_lookup_miss", r.lookup_miss.to_string())
        .with_extra("reuse_no_path", r.no_path.to_string())
        .with_extra("recomb_hits", t.recombination.hits.to_string())
        .with_extra("recomb_probes", t.recombination.probes.to_string())
        .with_extra("eq_checks", t.recombination.eq_checks.to_string())
        .with_extra("eq_rejects", t.recombination.eq_rejects.to_string());
    if let Some(p) = t.fan.percentile(0.5) {
        meta = meta.with_extra("fan_p50", p.to_string());
    }
    if let Some(p) = t.fan.percentile(0.9) {
        meta = meta.with_extra("fan_p90", p.to_string());
    }
    meta
}

/// One full MctsBot-vs-MctsBot game, from kickoff.
fn self_play_trajectory(cfg: &GenerateConfig, nn: Option<&Arc<NnEvaluator>>, seed: u64) -> Trajectory {
    let mut builder = GameStateBuilder::new();
    builder.set_state(BuilderState::CoinToss);
    if let Some(dims) = cfg.board_for(seed) {
        builder.with_board_dims(dims);
    }
    let mut state = builder.build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let (samples, telemetry) = mcts_vs_mcts_samples(&mut state, cfg, nn, seed, |_| false);

    let label = budget_label(cfg);
    let mut meta = TrajectoryMeta::new("self-play", board_dims)
        .with_bots(label.clone(), label)
        .with_seed(seed)
        .with_extra("mode", "self-play")
        .with_extra("max_steps", cfg.max_steps.to_string());
    if let Some(d) = &cfg.board_sizes {
        meta = meta.with_extra("size_dist", d.label.clone());
    }
    if let Some(name) = &cfg.config_name {
        meta = meta.with_extra("mcts_config", name.clone());
    }
    let meta = with_telemetry(meta, &telemetry);
    let outcome = Outcome::from_state(&state, None);
    Trajectory::new(meta, samples, outcome)
}

/// One MctsBot-vs-MctsBot **drive** from a randomized mid-game state
/// (plan 019): the record ends when either team scores, the half ends, or
/// the game ends. Everything after the drive resolves would be downstream
/// of self-play (correlated states) instead of the diverse random placement
/// this mode exists to provide (plan 020) — with one exception, plan 047:
/// under [`GenerateConfig::next_drive`] a drive that *scores* is followed
/// through the kickoff that comes next (both per-player setups, the kick,
/// the turns) until that drive resolves too, and that second drive is its
/// own record (`drive = 2`, `start = setup`). Two records, not one, so every
/// per-drive consumer (`prepare`'s per-drive weight, `td_rate.py`, the hub's
/// TD count) keeps meaning what it means, and the setup samples are labelled
/// by the drive they set up, nothing else.
fn random_start_trajectory(cfg: &GenerateConfig, nn: Option<&Arc<NnEvaluator>>, seed: u64) -> Vec<Trajectory> {
    let mut rs = cfg.bias.to_config();
    // Alternate the placement temperature per game so the corpus mixes
    // sharp (clustered) and flat (scattered) player distributions.
    if seed % 2 == 1 {
        rs.temperature = cfg.bias.temperature2;
    }
    rs.board_dims = cfg.board_for(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = generate_random_start(&rs, &mut rng);
    state.set_logging_state(false);

    let board_dims = state.board_dims;
    let label = budget_label(cfg);
    let bias = &cfg.bias;
    let base_meta = |drive: u32, start: &str, s: &GameState| {
        let mut meta = TrajectoryMeta::new("random-start", board_dims)
            .with_bots(label.clone(), label.clone())
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
            .with_extra("drive", drive.to_string())
            .with_extra("start", start)
            .with_extra("start_half", s.info.half.to_string())
            .with_extra("start_home_turn", s.info.home_turn.to_string())
            .with_extra("start_away_turn", s.info.away_turn.to_string())
            .with_extra("start_score", format!("{}-{}", s.home.score, s.away.score));
        if let Some(d) = &cfg.board_sizes {
            meta = meta.with_extra("size_dist", d.label.clone());
        }
        if let Some(name) = &cfg.config_name {
            meta = meta.with_extra("mcts_config", name.clone());
        }
        meta
    };
    let drive_over = |s: &GameState, half: u8, home: u8, away: u8| {
        s.home.score != home || s.away.score != away || s.info.half != half
    };

    let mut bots = BotPair::new(cfg, nn, seed);
    let (half, home, away) = (state.info.half, state.home.score, state.away.score);
    let meta = base_meta(1, "random", &state);
    let (samples, telemetry) = bots.play(&mut state, cfg, |s| drive_over(s, half, home, away));
    let mut out = vec![Trajectory::new(
        with_telemetry(meta, &telemetry),
        samples,
        Outcome::from_state(&state, None),
    )];

    // A score (not a half end, not game over) leaves the engine at the next
    // drive's first setup prompt: the scoring team kicks and is asked to
    // place its first player.
    let scored = (state.home.score, state.away.score) != (home, away);
    if cfg.next_drive && scored && !state.info.game_over && state.info.half == half && state.setup_team().is_some() {
        let (home, away) = (state.home.score, state.away.score);
        let meta = base_meta(2, "setup", &state);
        let (samples, telemetry) = bots.play(&mut state, cfg, |s| drive_over(s, half, home, away));
        out.push(Trajectory::new(
            with_telemetry(meta, &telemetry),
            samples,
            Outcome::from_state(&state, None),
        ));
    }
    out
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

    let mut meta = TrajectoryMeta::new(lecture.name(), board_dims)
        .with_bots(budget_label(cfg), "random")
        .with_seed(seed)
        .with_extra("mode", "curriculum")
        .with_extra("difficulty", format!("{difficulty:?}"))
        .with_extra("lecture_status", format!("{status:?}"));
    if let Some(name) = &cfg.config_name {
        meta = meta.with_extra("mcts_config", name.clone());
    }
    // Only the agent searches here; the opponent is a RandomBot.
    let meta = with_telemetry(meta, &agent.take_telemetry());
    let outcome = Outcome::from_state(&state, Some(format!("{status:?}")));
    Ok(Some(Trajectory::new(meta, samples, outcome)))
}
