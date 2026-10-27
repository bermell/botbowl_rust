//! `botbowl-ui fuzz` (plan 069 item 7): seeded, parallel random-game fuzzing of the engine, with
//! invariant checks after every micro-step.
//!
//! Each game is a pure function of its seed: the seed picks the board (from the loop's size mix,
//! or one board), the start (a full game from the coin toss, or a random-start drive played on to
//! the end of the game, which covers `--next-drive` continuations, half time and every later
//! kickoff), the policy, the dice, the rosters' skills and the jump toggle; then the game is played
//! to its end. A failure is recorded with its seed and micro-step; `--only-seed S` replays one game
//! (and `--dump-from N` writes its states from step N as JSON lines, for a test).
//!
//! Two dice modes. `register` (most games): `DiceMode::RegisterRolls`, every roll resolved by the
//! fuzzer — fair, or *flattened* (pass/fail 50/50, 2D6 sums uniform over 2..12) so rare branches
//! (kick-off table results, fumbles, failed jumps) come up — with the invariants checked at every
//! roll and every decision. `rolldice`: `DiceMode::RollDice` with the state's own seeded rng and
//! `GameState::step`, the production path that generation, `mc-label` and the value benches replay.
//!
//! Checked (the classes in `summary.json`):
//! - `panic`: any panic, inside the engine, the encoder, the search or the checks themselves;
//! - `step_state`: a micro-step returned something but `NeedAction` / `NeedRoll` / `GameOver`;
//! - `mask`: a decision with no listed action, a listed action `is_legal_action` refuses, a hidden
//!   action type listed, no deciding team, or the deciding team disagreeing with `turn_team` /
//!   `setup_team`; `search_actions` empty or not a subset; `action_apply`: a listed (or hidden)
//!   action that panics when applied to a clone;
//! - `nn`: the encoder panics, emits a non-finite value or the wrong shape, or a listed action maps
//!   to a policy cell out of range (33 channels, the board's h x w), or does not round-trip;
//! - `hash`: a clone, a JSON round trip (the corpus path) or a double mirror / y-flip of a `Turn`
//!   decision is unequal to the state or hashes differently;
//! - `players`: roster conservation (pitch + dugout, per team), board/player agreement, players
//!   off the active board, more than `team_size` on the pitch at a `Turn` decision, a dangling
//!   active player;
//! - `ball`: a carrier who is not on the pitch, or not standing at a `Turn` decision; the ball on
//!   the ground out of bounds or under a player at a `Turn` decision; in the air or off the pitch
//!   at a `Turn` decision; `bounce_squares` non-empty while the ball is not in the air;
//! - `counters`: half or turn markers going backwards (Time-out's one step back excepted), a
//!   marker past 8, scores going down, a winner that disagrees with the score;
//! - `termination`: no game over within `--max-steps` micro-steps;
//! - `replay`: the game rebuilt from its seed and inputs diverges (or the start state does).

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufWriter, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

use botbowl_curriculum::{generate_random_start, RandomStartConfig};
use botbowl_engine::bots::Bot;
use botbowl_engine::core::dices::{resolve_with_rng, RequestedRoll, RollResult, RollTarget, Sum2D6};
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{
    Action, BallState, BoardDims, MicroStepState, PlayerStatus, Position, SomeProcInput, TeamType, HEIGHT, TEAM_SIZE,
    WIDTH,
};
use botbowl_engine::core::table::{PosAT, SimpleAT, Skill};
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_mcts::pruning::search_actions;
use botbowl_mcts::{MctsBot, SearchBudget};
use botbowl_nn::actions::{action_cell, action_from_cell, POLICY_CHANNELS};
use botbowl_nn::encode::{encode, GLOBAL_FEATURES, SPATIAL_CHANNELS};
use botbowl_nn::eval::NnEvaluator;
use botbowl_nn::perspective::mover_for;
use botbowl_play::board_sizes::{board_label, CentredSpec, SizeDist};
use botbowl_play::policy::PolicyBot;

use crate::cli::FuzzArgs;

const TURNS_PER_HALF: u8 = 8;

// ---------------------------------------------------------------------------------------------
// Per-game configuration, a pure function of the seed.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Start {
    /// From the coin toss, both setups and every kickoff.
    Full,
    /// A random-start drive (the loop's generator), played on to the end of the game.
    Random,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Policy {
    /// Uniform over the listed actions.
    Uniform,
    /// Uniform over action *types*, then over that type's squares: passes, fouls, hand-offs and
    /// secure-the-ball come up as often as moves.
    Typed,
    /// The scripted bot, with a typed-random move one decision in five.
    Scripted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Dice {
    Fair,
    /// Pass/fail 50/50 and every 2D6 sum equally likely (only outcomes the roll can produce).
    Flat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Mode {
    Register,
    RollDice,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct GameCfg {
    seed: u64,
    #[serde(serialize_with = "ser_board")]
    board: BoardDims,
    start: Start,
    policy: Policy,
    dice: Dice,
    mode: Mode,
    /// Rosters re-drawn with random stats and skills from every `Skill` (the loop's random starts
    /// only hand out `good_skills`), and random team resources.
    wide_skills: bool,
    /// A wide game whose rosters are full of Throw / Kick Team-mate and Right Stuff and whose
    /// players take a hidden (human-only) action three decisions in ten.
    tag_team: bool,
    /// A wide game built to run the rosters out: soft armour, Mighty Blow, Claws and Dirty Player
    /// all over, so setups with too few players and empty benches come up.
    brutal: bool,
    jump_paths: bool,
    replay: bool,
}

fn ser_board<S: serde::Serializer>(d: &BoardDims, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&board_label(*d))
}

struct Knobs {
    boards: SizeDist,
    start: Option<Start>,
    policy: Option<Policy>,
    replay_share: f64,
    rolldice_share: f64,
    wide_share: f64,
    flat_share: f64,
    hidden_prob: f64,
    mcts_prob: f64,
    nn_prob: f64,
    apply_every: u64,
    serde_every: u64,
    max_steps: u64,
}

fn game_cfg(seed: u64, k: &Knobs) -> GameCfg {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5EED_F022);
    let start = k
        .start
        .unwrap_or(if rng.gen_bool(0.5) { Start::Full } else { Start::Random });
    let policy = k.policy.unwrap_or(match rng.gen_range(0..10) {
        0 => Policy::Uniform,
        1..=4 => Policy::Typed,
        _ => Policy::Scripted,
    });
    let wide_skills = rng.gen_bool(k.wide_share);
    GameCfg {
        seed,
        board: k.boards.sample(rng.gen()),
        start,
        policy,
        dice: if rng.gen_bool(k.flat_share) {
            Dice::Flat
        } else {
            Dice::Fair
        },
        mode: if rng.gen_bool(k.rolldice_share) {
            Mode::RollDice
        } else {
            Mode::Register
        },
        wide_skills,
        tag_team: wide_skills && rng.gen_bool(0.2),
        brutal: wide_skills && rng.gen_bool(0.2),
        jump_paths: rng.gen_bool(0.85),
        replay: rng.gen_bool(k.replay_share),
    }
}

// ---------------------------------------------------------------------------------------------
// Building the start state.

/// Every skill, for the wide rosters (unimplemented ones simply do nothing).
fn wide_skill(rng: &mut ChaCha8Rng) -> Skill {
    Skill::ALL[rng.gen_range(0..Skill::ALL.len())]
}

fn widen_stats(stats: &mut botbowl_engine::core::model::PlayerStats, cfg: &GameCfg, rng: &mut ChaCha8Rng) {
    let tag_team = cfg.tag_team;
    if cfg.brutal {
        stats.armour = stats.armour.min(rng.gen_range(3..=7));
        for (skill, p) in [
            (Skill::MightyBlow, 0.5),
            (Skill::Claws, 0.4),
            (Skill::DirtyPlayer, 0.4),
            (Skill::PileDriver, 0.2),
        ] {
            if rng.gen_bool(p) {
                stats.give_skill(skill);
            }
        }
    }
    if tag_team {
        for (skill, p) in [
            (Skill::ThrowTeamMate, 0.3),
            (Skill::KickTeamMate, 0.3),
            (Skill::RightStuff, 0.4),
        ] {
            if rng.gen_bool(p) {
                stats.give_skill(skill);
            }
        }
    }
    if rng.gen_bool(0.5) {
        stats.ma = rng.gen_range(1..=9);
        stats.str_ = rng.gen_range(1..=6);
        stats.agility = rng.gen_range(1..=6);
        stats.passing = rng.gen_range(1..=6);
        stats.armour = if cfg.brutal {
            rng.gen_range(3..=7)
        } else {
            rng.gen_range(6..=11)
        };
    }
    while rng.gen_bool(0.55) && stats.skills.len() < 6 {
        stats.give_skill(wide_skill(rng));
    }
}

fn widen_teams(state: &mut GameState, rng: &mut ChaCha8Rng) {
    for team in [TeamType::Home, TeamType::Away] {
        let t = state.get_mut_team(team);
        t.rerolls_start = rng.gen_range(0..=5);
        t.rerolls = t.rerolls_start;
        t.bribes = rng.gen_range(0..=2);
        t.fan_factor = rng.gen_range(0..=6);
        t.cheerleaders = rng.gen_range(0..=3);
        t.assistant_coaches = rng.gen_range(0..=3);
    }
}

fn build_start(cfg: &GameCfg) -> GameState {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut state = match cfg.start {
        Start::Full => GameStateBuilder::new()
            .with_board_dims(cfg.board)
            .set_state(BuilderState::CoinToss)
            .build(),
        Start::Random => {
            let rs = RandomStartConfig {
                board_dims: Some(cfg.board),
                temperature: if cfg.seed % 2 == 1 { 1.2 } else { 0.6 },
                ..RandomStartConfig::default()
            };
            generate_random_start(&rs, &mut rng)
        }
    };
    state.set_logging_state(false);
    state.set_jump_paths(cfg.jump_paths);
    if cfg.wide_skills {
        widen_teams(&mut state, &mut rng);
        for p in state.get_dugout_mut() {
            widen_stats(&mut p.stats, cfg, &mut rng);
        }
        let ids: Vec<_> = state.get_players_on_pitch().map(|p| p.id).collect();
        for id in ids {
            let mut s = state.get_player_unsafe(id).stats.clone();
            widen_stats(&mut s, cfg, &mut rng);
            state.get_mut_player_unsafe(id).stats = s;
        }
        if state.proc_stack_top() == Some("Turn") {
            // The offerings were made for the old rosters (a Throw Team-mate declaration...).
            let _ = state.reask_turn_decision();
        }
    }
    match cfg.mode {
        Mode::Register => state.set_dice_mode(DiceMode::RegisterRolls),
        Mode::RollDice => {
            state.set_dice_mode(DiceMode::RollDice);
            state.set_seed(cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1CE);
        }
    }
    state
}

// ---------------------------------------------------------------------------------------------
// Dice and policies.

fn flat_sum(rng: &mut ChaCha8Rng) -> Sum2D6 {
    Sum2D6::try_from(rng.gen_range(2u8..=12)).unwrap()
}

fn roll(req: RequestedRoll, dice: Dice, rng: &mut ChaCha8Rng) -> RollResult {
    if dice == Dice::Fair || rng.gen_bool(0.5) {
        return resolve_with_rng(req, rng);
    }
    match req {
        RequestedRoll::D6PassFail(_) => {
            if rng.gen_bool(0.5) {
                RollResult::Pass
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::D6ThreeOutcomes(low, high) => {
            let mut options = vec![RollResult::Pass, RollResult::Fail];
            if low < high {
                options.push(RollResult::MiddleOutcome);
            }
            options[rng.gen_range(0..options.len())]
        }
        RequestedRoll::Sum2D6 => RollResult::Sum2D6(flat_sum(rng)),
        RequestedRoll::Sum2D6PassFail(t) => {
            if t.is_success(flat_sum(rng)) {
                RollResult::Pass
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::Sum2D6ThreeOutcomes(low, high) => {
            let s = flat_sum(rng);
            if high.is_success(s) {
                RollResult::Pass
            } else if low.is_success(s) {
                RollResult::MiddleOutcome
            } else {
                RollResult::Fail
            }
        }
        other => resolve_with_rng(other, rng),
    }
}

fn kind_of(a: &Action) -> (u8, u8) {
    match a {
        Action::Positional(at, _) => (0, *at as u8),
        Action::Simple(at) => (1, *at as u8),
    }
}

fn typed_pick(actions: &[Action], rng: &mut ChaCha8Rng) -> Action {
    let mut kinds: Vec<(u8, u8)> = actions.iter().map(kind_of).collect();
    kinds.sort();
    kinds.dedup();
    let k = kinds[rng.gen_range(0..kinds.len())];
    let members: Vec<&Action> = actions.iter().filter(|a| kind_of(a) == k).collect();
    *members[rng.gen_range(0..members.len())]
}

struct Players {
    rng: ChaCha8Rng,
    scripted: ScriptedBot,
    mcts: Option<MctsBot>,
    nn: Option<PolicyBot>,
    hidden_taken: u64,
}

impl Players {
    fn new(seed: u64, shared: &Shared) -> Self {
        let mut scripted = ScriptedBot::new();
        scripted.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0x5C21));
        let mcts = shared.mcts_iters.filter(|_| shared.knobs.mcts_prob > 0.0).map(|iters| {
            let mut bot =
                MctsBot::with_budget_and_config(SearchBudget::Iterations(iters), shared.mcts_config).with_workers(1);
            bot.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0x3C75));
            bot
        });
        Players {
            rng: ChaCha8Rng::seed_from_u64(seed ^ 0xAC71),
            scripted,
            mcts,
            nn: shared.nn.as_ref().map(|nn| PolicyBot::new(Arc::clone(nn))),
            hidden_taken: 0,
        }
    }

    fn choose(&mut self, state: &GameState, actions: &[Action], cfg: &GameCfg, knobs: &Knobs) -> Action {
        let policy = cfg.policy;
        let hidden_prob = if cfg.tag_team { 0.3 } else { knobs.hidden_prob };
        let r: f64 = self.rng.gen();
        if r < hidden_prob {
            let mut hidden = Vec::new();
            state.available_actions.collect_hidden_actions(&mut hidden);
            if !hidden.is_empty() {
                hidden.sort();
                self.hidden_taken += 1;
                return hidden[self.rng.gen_range(0..hidden.len())];
            }
        }
        let r: f64 = self.rng.gen();
        if r < knobs.mcts_prob {
            if let Some(bot) = self.mcts.as_mut() {
                return bot.get_action(state);
            }
        } else if r < knobs.mcts_prob + knobs.nn_prob {
            if let Some(bot) = self.nn.as_mut() {
                return bot.get_action(state);
            }
        }
        match policy {
            Policy::Uniform => actions[self.rng.gen_range(0..actions.len())],
            Policy::Typed => typed_pick(actions, &mut self.rng),
            Policy::Scripted => {
                if self.rng.gen_bool(0.2) {
                    typed_pick(actions, &mut self.rng)
                } else {
                    self.scripted.get_action(state)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Invariants.

#[derive(Clone, Debug, Serialize)]
struct Failure {
    class: &'static str,
    message: String,
    seed: u64,
    step: u64,
    decision: u64,
    proc_top: String,
    half: u8,
    home_turn: u8,
    away_turn: u8,
    cfg: GameCfg,
}

fn hash_of(state: &GameState) -> u64 {
    let mut h = DefaultHasher::new();
    state.hash(&mut h);
    h.finish()
}

fn team_counts(state: &GameState) -> [usize; 2] {
    let mut n = [0usize; 2];
    for p in state.get_players_on_pitch() {
        n[p.stats.team as usize] += 1;
    }
    for p in state.get_dugout() {
        n[p.stats.team as usize] += 1;
    }
    n
}

fn stack_has(state: &GameState, names: &[&str]) -> bool {
    state.proc_stack_iter().any(|p| names.contains(&p.name()))
}

struct Prev {
    half: u8,
    turns: [u8; 2],
    scores: [u8; 2],
    roster: [usize; 2],
    game_over: bool,
}

impl Prev {
    fn of(state: &GameState) -> Self {
        Prev {
            half: state.info.half,
            turns: [state.info.home_turn, state.info.away_turn],
            scores: [state.home.score, state.away.score],
            roster: team_counts(state),
            game_over: state.info.game_over,
        }
    }
}

type Check = Result<(), (&'static str, String)>;

fn fail(class: &'static str, msg: String) -> Check {
    Err((class, msg))
}

/// The checks that hold at every micro-step.
fn check_always(state: &GameState, prev: &Prev, coarse: bool) -> Check {
    // Roster conservation.
    let roster = team_counts(state);
    if roster != prev.roster {
        return fail(
            "players",
            format!("roster {:?} -> {:?} (pitch + dugout)", prev.roster, roster),
        );
    }
    // Board / player agreement, everyone on the active board.
    let mut fielded = 0usize;
    for p in state.get_players_on_pitch() {
        fielded += 1;
        // A player pushed into the crowd stands on the border square while the push resolves
        // (the follow-up decision, the crowd injury rolls); `check_quiet` wants them gone by the
        // next `Turn` decision.
        let in_frame = p.position.x >= 0
            && p.position.y >= 0
            && p.position.x < state.board_dims.width
            && p.position.y < state.board_dims.height;
        if !in_frame {
            return fail(
                "players",
                format!("player {} fielded off the board at {:?}", p.id, p.position),
            );
        }
        if state.get_player_id_at(p.position) != Some(p.id) {
            return fail(
                "players",
                format!(
                    "player {} at {:?} but the board holds {:?}",
                    p.id,
                    p.position,
                    state.get_player_id_at(p.position)
                ),
            );
        }
    }
    let mut occupied = 0usize;
    for x in 0..WIDTH as i8 {
        for y in 0..HEIGHT as i8 {
            if state.get_player_id_at(Position::new((x, y))).is_some() {
                occupied += 1;
            }
        }
    }
    if occupied != fielded {
        return fail(
            "players",
            format!("{occupied} occupied squares, {fielded} fielded players"),
        );
    }
    // Ball.
    match state.ball {
        BallState::Carried(id) => {
            if state.get_player(id).is_err() {
                return fail("ball", format!("carrier {id} is not on the pitch"));
            }
        }
        BallState::OnGround(pos) if state.is_out(pos) => {
            return fail("ball", format!("ball on the ground out of bounds at {pos:?}"));
        }
        _ => {}
    }
    if !state.bounce_squares.is_empty() && !matches!(state.ball, BallState::InAir(_)) {
        return fail(
            "ball",
            format!(
                "bounce_squares {:?} while the ball is {:?}",
                state.bounce_squares, state.ball
            ),
        );
    }
    // Counters.
    let i = &state.info;
    if i.half < prev.half || i.half > 2 {
        return fail("counters", format!("half {} -> {}", prev.half, i.half));
    }
    let turns = [i.home_turn, i.away_turn];
    if turns.iter().any(|&t| t > TURNS_PER_HALF) {
        return fail("counters", format!("turn markers {turns:?} past {TURNS_PER_HALF}"));
    }
    if i.half == prev.half {
        // S3 Time-out!: both markers one space back, in one micro-step (`KickoffTable`). Under
        // `RollDice` a step is a whole decision, so a marker may also have moved on since.
        let both_back = turns[0] + 1 == prev.turns[0] && turns[1] + 1 == prev.turns[1];
        for k in 0..2 {
            if turns[k] < prev.turns[k] {
                let ok = if coarse {
                    turns[k] + 1 == prev.turns[k]
                } else {
                    both_back
                };
                if !ok {
                    return fail("counters", format!("turn markers {:?} -> {turns:?}", prev.turns));
                }
            }
        }
    }
    let scores = [state.home.score, state.away.score];
    if scores[0] < prev.scores[0] || scores[1] < prev.scores[1] {
        return fail("counters", format!("score {:?} -> {scores:?}", prev.scores));
    }
    if prev.game_over && !i.game_over {
        return fail("counters", "game_over cleared".into());
    }
    if i.game_over {
        let expect = match scores[0].cmp(&scores[1]) {
            std::cmp::Ordering::Greater => Some(TeamType::Home),
            std::cmp::Ordering::Less => Some(TeamType::Away),
            std::cmp::Ordering::Equal => None,
        };
        if i.winner != expect {
            return fail("counters", format!("game over {scores:?}, winner {:?}", i.winner));
        }
    }
    // Hash/eq on a clone.
    let c = state.clone();
    if &c != state {
        return fail("hash", "a clone is unequal to its original".into());
    }
    if hash_of(&c) != hash_of(state) {
        return fail("hash", "a clone hashes differently".into());
    }
    Ok(())
}

/// At a `Turn` decision the position is "quiet": no procedure is mid-way.
fn check_quiet(state: &GameState) -> Check {
    for t in [&state.home, &state.away] {
        if t.rerolls > t.rerolls_start {
            return fail(
                "counters",
                format!(
                    "{} team re-rolls, {} at the start of the half",
                    t.rerolls, t.rerolls_start
                ),
            );
        }
    }
    for p in state.get_players_on_pitch() {
        if state.is_out(p.position) {
            return fail(
                "players",
                format!("player {} out of bounds at {:?} at a Turn decision", p.id, p.position),
            );
        }
    }
    for team in [TeamType::Home, TeamType::Away] {
        let n = state.get_players_on_pitch_in_team(team).count();
        if n > state.board_dims.team_size {
            return fail(
                "players",
                format!(
                    "{team:?} has {n} on the pitch, team size {}",
                    state.board_dims.team_size
                ),
            );
        }
    }
    match state.ball {
        BallState::Carried(id) => {
            let p = state.get_player_unsafe(id);
            if p.status != PlayerStatus::Up {
                return fail("ball", format!("carrier {id} is {:?} at a Turn decision", p.status));
            }
            // S3: a standing carrier in the end zone they attack scores, at once, in either
            // team's turn.
            if p.position.x == state.board_dims.endzone_x(p.stats.team) {
                return fail(
                    "rules",
                    format!("carrier {id} stands in the end zone at {:?}, no touchdown", p.position),
                );
            }
        }
        BallState::OnGround(pos) => {
            if let Some(id) = state.get_player_id_at(pos) {
                return fail(
                    "ball",
                    format!("ball on the ground under player {id} at {pos:?} at a Turn decision"),
                );
            }
        }
        other => return fail("ball", format!("ball {other:?} at a Turn decision")),
    }
    Ok(())
}

fn check_decision(state: &GameState, actions: &[Action]) -> Check {
    if actions.is_empty() {
        return fail("mask", "decision with no listed action".into());
    }
    let Some(team) = state.available_actions.team else {
        return fail("mask", "decision with no deciding team".into());
    };
    let top = state.proc_stack_top().unwrap_or("");
    if top == "Turn" {
        if state.turn_team() != Some(team) {
            return fail(
                "mask",
                format!("Turn decision for {team:?}, turn_team {:?}", state.turn_team()),
            );
        }
    }
    if let Some(setup) = state.setup_team() {
        if top == "Setup" && setup != team {
            return fail("mask", format!("Setup decision for {team:?}, setup_team {setup:?}"));
        }
    }
    if let Some(id) = state.info.active_player {
        if state.get_player(id).is_err() {
            return fail("players", format!("active player {id} is not on the pitch"));
        }
    }
    for a in actions {
        if !state.is_legal_action(a) {
            return fail("mask", format!("listed action {a:?} is not legal"));
        }
        if matches!(
            a,
            Action::Simple(SimpleAT::PassSquare) | Action::Positional(PosAT::Jump | PosAT::ThrowTeamMate, _)
        ) {
            return fail("mask", format!("hidden action {a:?} listed"));
        }
    }
    let searched = search_actions(state);
    if searched.is_empty() {
        return fail("mask", "search_actions is empty".into());
    }
    if let Some(a) = searched.iter().find(|a| !actions.contains(a)) {
        return fail("mask", format!("search_actions offers unlisted {a:?}"));
    }
    if searched != search_actions(&state.clone()) {
        return fail("mask", "search_actions differs on a clone (pruning impure)".into());
    }
    if state.proc_stack_top() == Some("Turn") {
        check_quiet(state)?;
        check_declarations(state, team, actions)?;
    }
    if actions.contains(&Action::Simple(SimpleAT::KickoffAimMiddle)) {
        for t in [TeamType::Home, TeamType::Away] {
            if !state.is_setup_legal(t) {
                return fail("setup", format!("{t:?}'s setup is illegal at the kick"));
            }
        }
    }
    Ok(())
}

/// What a `Turn` decision may offer: one of each special action per team turn (the `GameInfo`
/// flags), Secure the Ball only on a loose ball with no Standing opponent within 2 squares and
/// never in a Charge, and declarations only for the acting team's unused players.
fn check_declarations(state: &GameState, team: TeamType, actions: &[Action]) -> Check {
    let i = &state.info;
    for a in actions {
        let Action::Positional(at, pos) = *a else { continue };
        let flag = match at {
            PosAT::StartBlitz => Some(i.blitz_available),
            PosAT::StartPass => Some(i.pass_available),
            PosAT::StartHandoff => Some(i.handoff_available),
            PosAT::StartFoul => Some(i.foul_available),
            PosAT::StartSecureBall => Some(i.secure_ball_available && !state.in_charge()),
            PosAT::StartMove | PosAT::StartBlock => Some(true),
            _ => None,
        };
        let Some(flag) = flag else { continue };
        if !flag {
            return fail(
                "rules",
                format!("{at:?} offered with its once-per-turn flag spent ({a:?})"),
            );
        }
        match state.get_player_at(pos) {
            Some(p) if p.stats.team == team && !p.used => {}
            other => {
                return fail(
                    "rules",
                    format!("{a:?} declared on {:?}", other.map(|p| (p.id, p.stats.team, p.used))),
                )
            }
        }
        if at == PosAT::StartSecureBall {
            let BallState::OnGround(ball) = state.ball else {
                return fail(
                    "rules",
                    format!("Secure the Ball offered with the ball {:?}", state.ball),
                );
            };
            if let Some(o) = state
                .get_players_on_pitch_in_team(botbowl_engine::core::model::other_team(team))
                .find(|o| o.status == PlayerStatus::Up && o.position.distance_to(&ball) <= 2)
            {
                return fail(
                    "rules",
                    format!("Secure the Ball offered with opponent {} standing within 2", o.id),
                );
            }
        }
    }
    Ok(())
}

fn check_nn(state: &GameState, actions: &[Action]) -> Check {
    let dims = state.board_dims;
    let (h, w) = (dims.height as usize, dims.width as usize);
    let enc = encode(state);
    if enc.h != h || enc.w != w || enc.spatial.len() != SPATIAL_CHANNELS * h * w || enc.global.len() != GLOBAL_FEATURES
    {
        return fail(
            "nn",
            format!(
                "encoded shape {}x{} spatial {} global {}",
                enc.h,
                enc.w,
                enc.spatial.len(),
                enc.global.len()
            ),
        );
    }
    if let Some(i) = enc.spatial.iter().position(|v| !v.is_finite()) {
        return fail("nn", format!("non-finite spatial value at {i}"));
    }
    if let Some(i) = enc.global.iter().position(|v| !v.is_finite()) {
        return fail("nn", format!("non-finite global feature {i}: {}", enc.global[i]));
    }
    let mover = mover_for(state);
    let mut cells = Vec::with_capacity(actions.len());
    for &a in actions {
        let cell = action_cell(a, mover, dims);
        if cell.channel >= POLICY_CHANNELS || cell.y >= h || cell.x >= w {
            return fail(
                "nn",
                format!("{a:?} -> cell {cell:?} out of range ({POLICY_CHANNELS} x {h} x {w})"),
            );
        }
        let back = action_from_cell(cell, mover, dims);
        if back != a {
            return fail("nn", format!("{a:?} -> {cell:?} -> {back:?}"));
        }
        cells.push((cell.channel, cell.y, cell.x, cell.is_simple));
    }
    cells.sort();
    if cells.windows(2).any(|w| w[0] == w[1]) {
        return fail("nn", "two listed actions share a policy cell".into());
    }
    Ok(())
}

fn check_serde(state: &GameState) -> Check {
    let json = serde_json::to_string(state).map_err(|e| ("hash", format!("serialise: {e}")))?;
    let mut back: GameState = serde_json::from_str(&json).map_err(|e| ("hash", format!("deserialise: {e}")))?;
    back.set_jump_paths(state.jump_paths());
    if &back != state {
        return fail("hash", "a JSON round trip is unequal".into());
    }
    if hash_of(&back) != hash_of(state) {
        return fail("hash", "a JSON round trip hashes differently".into());
    }
    if state.proc_stack_top() == Some("Turn") {
        let m = state.mirrored().mirrored();
        if &m != state || hash_of(&m) != hash_of(state) {
            return fail("hash", "mirrored twice differs at a Turn decision".into());
        }
        let y = state.y_flipped().y_flipped();
        if &y != state || hash_of(&y) != hash_of(state) {
            return fail("hash", "y-flipped twice differs at a Turn decision".into());
        }
    }
    // The trainer encodes corpus states, the search live ones: they must see the same tensors.
    let (live, stored) = (encode(state), encode(&back));
    if live.spatial != stored.spatial || live.global != stored.global {
        return fail("nn", "a JSON round trip encodes differently".into());
    }
    if state.proc_stack_top() == Some("Turn") {
        // Mover-centric: the mirrored position (sides swapped) is the same position for the
        // other side to move, so it must encode identically.
        let m = encode(&state.mirrored());
        if m.spatial != live.spatial {
            let plane = live.h * live.w;
            let i = (0..live.spatial.len())
                .find(|&i| m.spatial[i] != live.spatial[i])
                .unwrap();
            return fail(
                "nn",
                format!("the mirror encodes differently: spatial channel {}", i / plane),
            );
        }
        if let Some(g) = (0..live.global.len()).find(|&g| m.global[g] != live.global[g]) {
            return fail("nn", format!("the mirror encodes differently: global feature {g}"));
        }
        // The y-flip is a symmetry: every plane flipped along y, the globals unchanged.
        let y = encode(&state.y_flipped());
        let (h, w) = (live.h, live.w);
        for c in 0..live.spatial.len() / (h * w) {
            for yy in 0..h {
                for x in 0..w {
                    if y.spatial[c * h * w + (h - 1 - yy) * w + x] != live.spatial[c * h * w + yy * w + x] {
                        return fail("nn", format!("the y-flip encodes differently: spatial channel {c}"));
                    }
                }
            }
        }
        if let Some(g) = (0..live.global.len()).find(|&g| y.global[g] != live.global[g]) {
            return fail("nn", format!("the y-flip encodes differently: global feature {g}"));
        }
    }
    Ok(())
}

/// The search's chance children at a roll, under the shipped roll models and the loop's
/// (`cfgs/chance_all_partial*.toml`: bounces with catches, grouped pass scatter and throw-ins):
/// each a result the roll can take, probabilities in [0, 1] summing to 1, and each one applies.
fn check_chance(state: &GameState, req: RequestedRoll) -> Check {
    use botbowl_mcts::roll_outcomes::{enumerate_full, BounceModel, PassScatterModel, RollModel, ThrowInModel};
    let loop_models = RollModel {
        bounce: BounceModel::Catch,
        pass_scatter: PassScatterModel::Grouped,
        throw_in: ThrowInModel::Grouped,
    };
    for (name, model) in [("shipped", RollModel::default()), ("loop", loop_models)] {
        let children = panic::catch_unwind(AssertUnwindSafe(|| {
            enumerate_full(state, &req, botbowl_mcts::dynamics::ChanceModel::Exact, model)
        }))
        .map_err(|_| {
            (
                "chance",
                format!("{name} enumeration of {req:?} panicked: {}", take_panic_msg()),
            )
        })?;
        if children.is_empty() {
            return fail("chance", format!("{name}: no outcome for {req:?}"));
        }
        let mut total = 0.0f64;
        for c in &children {
            let botbowl_mcts::BbAction::Chance { result, .. } = *c else {
                return fail("chance", format!("{name}: a player action among {req:?}'s outcomes"));
            };
            let p = c.prob_f32().unwrap_or(f32::NAN);
            if !(0.0..=1.0 + 1e-6).contains(&p) {
                return fail("chance", format!("{name}: {req:?} -> {result:?} with probability {p}"));
            }
            total += p as f64;
            if !req.is_compatible(result) {
                return fail("chance", format!("{name}: {req:?} -> incompatible {result:?}"));
            }
            let mut c = state.clone();
            let out = panic::catch_unwind(AssertUnwindSafe(|| {
                c.step_with_roll_or_action(SomeProcInput::Roll(result))
            }));
            if out.is_err() {
                return fail(
                    "chance",
                    format!("{name}: applying {req:?} -> {result:?} panicked: {}", take_panic_msg()),
                );
            }
        }
        if (total - 1.0).abs() > 1e-3 {
            return fail(
                "chance",
                format!("{name}: {req:?}'s {} outcomes sum to {total}", children.len()),
            );
        }
    }
    Ok(())
}

/// Apply every listed (and hidden) action to a clone, one micro-step, under `RegisterRolls`.
fn check_apply(state: &GameState, actions: &[Action]) -> Check {
    let mut all = actions.to_vec();
    state.available_actions.collect_hidden_actions(&mut all);
    for a in all {
        let mut c = state.clone();
        c.set_dice_mode(DiceMode::RegisterRolls);
        let out = panic::catch_unwind(AssertUnwindSafe(|| {
            c.step_with_roll_or_action(SomeProcInput::Action(a))
        }));
        match out {
            Err(_) => return fail("action_apply", format!("applying {a:?} panicked: {}", take_panic_msg())),
            Ok(MicroStepState::NeedAction | MicroStepState::NeedRoll | MicroStepState::GameOver) => {}
            Ok(other) => return fail("action_apply", format!("applying {a:?} returned {other:?}")),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Panics.

thread_local! {
    static PANIC_MSG: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

fn take_panic_msg() -> String {
    PANIC_MSG.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

fn install_panic_hook() {
    panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "?".to_string()
        };
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        PANIC_MSG.with(|m| *m.borrow_mut() = format!("{msg} @ {loc}"));
    }));
}

// ---------------------------------------------------------------------------------------------
// One game.

#[derive(Default, Clone, Serialize)]
struct Stats {
    games: u64,
    steps: u64,
    decisions: u64,
    rolls: u64,
    replays: u64,
    applied: u64,
    serde_checks: u64,
    hidden_taken: u64,
    game_overs: u64,
    touchdowns: u64,
    /// Kicks with a side fielding fewer than `team_size` (a run-out roster), and with nobody.
    short_kickoffs: u64,
    empty_kickoffs: u64,
    /// Micro-steps at which each procedure was on top: what the games reached.
    procs: BTreeMap<&'static str, u64>,
    /// Decisions by listed action type.
    action_types: BTreeMap<String, u64>,
    /// Games and steps per board.
    boards: BTreeMap<String, (u64, u64)>,
}

impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.games += o.games;
        self.steps += o.steps;
        self.decisions += o.decisions;
        self.rolls += o.rolls;
        self.replays += o.replays;
        self.applied += o.applied;
        self.serde_checks += o.serde_checks;
        self.hidden_taken += o.hidden_taken;
        self.game_overs += o.game_overs;
        self.touchdowns += o.touchdowns;
        self.short_kickoffs += o.short_kickoffs;
        self.empty_kickoffs += o.empty_kickoffs;
        for (k, v) in &o.procs {
            *self.procs.entry(k).or_default() += v;
        }
        for (k, v) in &o.action_types {
            *self.action_types.entry(k.clone()).or_default() += v;
        }
        for (k, v) in &o.boards {
            let e = self.boards.entry(k.clone()).or_default();
            e.0 += v.0;
            e.1 += v.1;
        }
    }
}

/// What a game recorded for its replay.
struct Record {
    inputs: Vec<SomeProcInput>,
    hashes: Vec<u64>,
    start: GameState,
}

struct Progress {
    step: u64,
    decision: u64,
}

fn failure(cfg: &GameCfg, state: Option<&GameState>, p: &Progress, class: &'static str, message: String) -> Failure {
    Failure {
        class,
        message,
        seed: cfg.seed,
        step: p.step,
        decision: p.decision,
        proc_top: state.and_then(|s| s.proc_stack_top()).unwrap_or("?").to_string(),
        half: state.map_or(0, |s| s.info.half),
        home_turn: state.map_or(0, |s| s.info.home_turn),
        away_turn: state.map_or(0, |s| s.info.away_turn),
        cfg: *cfg,
    }
}

struct Dump {
    from: u64,
    out: BufWriter<File>,
}

fn play_game(cfg: &GameCfg, shared: &Shared, stats: &mut Stats, dump: &mut Option<Dump>) -> Option<Failure> {
    let k = &shared.knobs;
    let mut progress = Progress { step: 0, decision: 0 };
    let mut state = build_start(cfg);
    let start = state.clone();
    let mut players = Players::new(cfg.seed, shared);
    let mut dice_rng = ChaCha8Rng::seed_from_u64(cfg.seed ^ 0xD1CE_D1CE);
    let mut prev = Prev::of(&state);
    let mut last_turn: Option<(u8, TeamType, u8, u8)> = None;
    let mut record = Record {
        inputs: Vec::new(),
        hashes: Vec::new(),
        start,
    };
    let label = board_label(cfg.board);
    let mut local = Stats::default();

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        loop {
            if let Some(d) = dump.as_mut() {
                if progress.step >= d.from {
                    let _ = serde_json::to_writer(&mut d.out, &state);
                    let _ = writeln!(d.out);
                }
            }
            if let Err((class, msg)) = check_always(&state, &prev, cfg.mode == Mode::RollDice) {
                return Some(failure(cfg, Some(&state), &progress, class, msg));
            }
            prev = Prev::of(&state);
            if let Some(top) = state.proc_stack_top() {
                *local.procs.entry(top).or_default() += 1;
            }
            if state.info.game_over {
                local.game_overs += 1;
                break;
            }
            if progress.step >= k.max_steps {
                return Some(failure(
                    cfg,
                    Some(&state),
                    &progress,
                    "termination",
                    format!("no game over in {} micro-steps", k.max_steps),
                ));
            }
            progress.step += 1;
            let input = if let Some(req) = state.pending_roll {
                local.rolls += 1;
                if k.apply_every > 0 && local.rolls % k.apply_every == 0 {
                    if let Err((class, msg)) = check_chance(&state, req) {
                        return Some(failure(cfg, Some(&state), &progress, class, msg));
                    }
                }
                SomeProcInput::Roll(roll(req, cfg.dice, &mut dice_rng))
            } else {
                progress.decision += 1;
                let actions = state.get_all_actions();
                // A Throw / Kick Team-mate, once a human declared it (a hidden action), asks only
                // hidden decisions: bots never get there, so none is listed.
                let human_only = actions.is_empty() && state.proc_stack_top() == Some("ThrowTeamMate");
                if human_only {
                    let mut hidden = Vec::new();
                    state.available_actions.collect_hidden_actions(&mut hidden);
                    if hidden.is_empty() {
                        return Some(failure(
                            cfg,
                            Some(&state),
                            &progress,
                            "mask",
                            "ThrowTeamMate offers nothing".into(),
                        ));
                    }
                    hidden.sort();
                    players.hidden_taken += 1;
                    SomeProcInput::Action(hidden[players.rng.gen_range(0..hidden.len())])
                } else {
                    if let Err((class, msg)) = check_decision(&state, &actions) {
                        return Some(failure(cfg, Some(&state), &progress, class, msg));
                    }
                    // The first `Turn` decision of a team turn: nobody of that team has acted yet.
                    if state.proc_stack_top() == Some("Turn") && state.info.active_player.is_none() {
                        let team = state.turn_team().unwrap();
                        let key = (state.info.half, team, state.info.home_turn, state.info.away_turn);
                        if last_turn != Some(key) {
                            last_turn = Some(key);
                            if let Some(p) = state.get_players_on_pitch_in_team(team).find(|p| p.used) {
                                return Some(failure(
                                    cfg,
                                    Some(&state),
                                    &progress,
                                    "rules",
                                    format!("player {} is already `used` as {team:?}'s turn starts", p.id),
                                ));
                            }
                        }
                    }
                    if let Err((class, msg)) = check_nn(&state, &actions) {
                        return Some(failure(cfg, Some(&state), &progress, class, msg));
                    }
                    if k.serde_every > 0 && progress.decision % k.serde_every == 0 {
                        local.serde_checks += 1;
                        if let Err((class, msg)) = check_serde(&state) {
                            return Some(failure(cfg, Some(&state), &progress, class, msg));
                        }
                    }
                    if k.apply_every > 0 && progress.decision % k.apply_every == 0 {
                        local.applied += 1;
                        if let Err((class, msg)) = check_apply(&state, &actions) {
                            return Some(failure(cfg, Some(&state), &progress, class, msg));
                        }
                    }
                    if actions.contains(&Action::Simple(SimpleAT::KickoffAimMiddle)) {
                        for t in [TeamType::Home, TeamType::Away] {
                            let n = state.get_players_on_pitch_in_team(t).count();
                            local.short_kickoffs += (n < state.board_dims.team_size) as u64;
                            local.empty_kickoffs += (n == 0) as u64;
                        }
                    }
                    for a in &actions {
                        let name = match a {
                            Action::Positional(at, _) => format!("{at:?}"),
                            Action::Simple(at) => format!("{at:?}"),
                        };
                        *local.action_types.entry(name).or_default() += 1;
                    }
                    let a = players.choose(&state, &actions, cfg, k);
                    if !state.is_legal_action(&a) {
                        return Some(failure(
                            cfg,
                            Some(&state),
                            &progress,
                            "mask",
                            format!("a bot chose illegal {a:?}"),
                        ));
                    }
                    SomeProcInput::Action(a)
                }
            };
            let before = state.home.score + state.away.score;
            match cfg.mode {
                Mode::Register => {
                    let out = state.step_with_roll_or_action(input);
                    if !matches!(
                        out,
                        MicroStepState::NeedAction | MicroStepState::NeedRoll | MicroStepState::GameOver
                    ) {
                        return Some(failure(cfg, Some(&state), &progress, "step_state", format!("{out:?}")));
                    }
                }
                Mode::RollDice => {
                    let SomeProcInput::Action(a) = input else {
                        return Some(failure(
                            cfg,
                            Some(&state),
                            &progress,
                            "step_state",
                            "a roll pending under RollDice".into(),
                        ));
                    };
                    if let Err(e) = state.step(a) {
                        return Some(failure(
                            cfg,
                            Some(&state),
                            &progress,
                            "step_state",
                            format!("step: {e:?}"),
                        ));
                    }
                }
            }
            local.touchdowns += (state.home.score + state.away.score - before) as u64;
            if cfg.replay {
                record.inputs.push(input);
                record.hashes.push(hash_of(&state));
            }
        }
        None
    }));
    local.games = 1;
    local.steps = progress.step;
    local.decisions = progress.decision;
    local.hidden_taken = players.hidden_taken;
    local.boards.insert(label, (1, progress.step));
    let out = match result {
        Err(_) => Some(failure(cfg, None, &progress, "panic", take_panic_msg())),
        Ok(Some(f)) => Some(f),
        Ok(None) if cfg.replay => {
            local.replays += 1;
            replay(cfg, &record, &state)
        }
        Ok(None) => None,
    };
    stats.merge(&local);
    out
}

/// Rebuild the game from its seed and replay its inputs: the start state and every state after
/// each input must be the recorded one.
fn replay(cfg: &GameCfg, rec: &Record, end: &GameState) -> Option<Failure> {
    let mut p = Progress { step: 0, decision: 0 };
    let res = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut state = build_start(cfg);
        if state != rec.start || hash_of(&state) != hash_of(&rec.start) {
            return Some(failure(
                cfg,
                Some(&state),
                &p,
                "replay",
                "the start state does not rebuild".into(),
            ));
        }
        for (k, (input, h)) in rec.inputs.iter().zip(&rec.hashes).enumerate() {
            p.step = k as u64 + 1;
            match (cfg.mode, input) {
                (Mode::RollDice, SomeProcInput::Action(a)) => {
                    if let Err(e) = state.step(*a) {
                        return Some(failure(cfg, Some(&state), &p, "replay", format!("step {a:?}: {e:?}")));
                    }
                }
                _ => {
                    state.step_with_roll_or_action(*input);
                }
            }
            if hash_of(&state) != *h {
                return Some(failure(
                    cfg,
                    Some(&state),
                    &p,
                    "replay",
                    format!("diverged after input {k}: {input:?}"),
                ));
            }
        }
        if &state != end {
            return Some(failure(cfg, Some(&state), &p, "replay", "the end state differs".into()));
        }
        None
    }));
    match res {
        Ok(f) => f,
        Err(_) => Some(failure(
            cfg,
            None,
            &p,
            "replay",
            format!("replay panicked: {}", take_panic_msg()),
        )),
    }
}

// ---------------------------------------------------------------------------------------------
// Driver.

struct Shared {
    knobs: Knobs,
    mcts_iters: Option<usize>,
    mcts_config: botbowl_mcts::MctsConfig,
    nn: Option<Arc<NnEvaluator>>,
}

#[derive(Serialize)]
struct Summary<'a> {
    commit: String,
    capacity: String,
    boards_spec: String,
    seconds: f64,
    stats: &'a Stats,
    failures_by_class: BTreeMap<&'static str, u64>,
    /// Distinct (class, message with numbers stripped) and the first seed of each.
    distinct: Vec<(String, String, u64, u64)>,
}

fn boards_from(spec: &str) -> Result<SizeDist, String> {
    let cap = BoardDims::new(WIDTH as i8, HEIGHT as i8, TEAM_SIZE);
    match spec {
        "full" => Ok(SizeDist::single(cap)),
        "loop" => {
            let area = ((cap.width - 2) as f64) * ((cap.height - 2) as f64);
            SizeDist::centred(&CentredSpec {
                centre_area: area,
                temperature: 0.3,
                floor: 0.2,
                aspect_min: 1.5,
                aspect_max: 2.8,
                cells_per_player: 26.0,
                min_area: Some(70.0_f64.min(area)),
                max_area: Some(area),
            })
        }
        list => SizeDist::parse_list(list, 26.0),
    }
}

fn normalise(msg: &str) -> String {
    // Numbers vary per occurrence (ids, squares, seeds); strip them to group like failures.
    let mut out = String::with_capacity(msg.len());
    let mut last_digit = false;
    for c in msg.chars() {
        if c.is_ascii_digit() {
            if !last_digit {
                out.push('#');
            }
            last_digit = true;
        } else {
            out.push(c);
            last_digit = false;
        }
    }
    out
}

pub fn run(args: FuzzArgs) -> io::Result<()> {
    let boards = boards_from(&args.boards).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let parse_start = |s: &str| match s {
        "full" => Ok(Some(Start::Full)),
        "random" => Ok(Some(Start::Random)),
        "mix" => Ok(None),
        o => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("--start {o}: full|random|mix"),
        )),
    };
    let parse_policy = |s: &str| match s {
        "uniform" => Ok(Some(Policy::Uniform)),
        "typed" => Ok(Some(Policy::Typed)),
        "scripted" => Ok(Some(Policy::Scripted)),
        "mix" => Ok(None),
        o => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("--policy {o}: uniform|typed|scripted|mix"),
        )),
    };
    let knobs = Knobs {
        boards,
        start: parse_start(&args.start)?,
        policy: parse_policy(&args.policy)?,
        replay_share: args.replay_share,
        rolldice_share: args.rolldice_share,
        wide_share: args.wide_skills_share,
        flat_share: args.flat_dice_share,
        hidden_prob: args.hidden_prob,
        mcts_prob: args.mcts_prob,
        nn_prob: if args.model.is_some() { args.nn_prob } else { 0.0 },
        apply_every: args.apply_every,
        serde_every: args.serde_every,
        max_steps: args.max_steps,
    };
    let mcts_config = match &args.mcts_config {
        Some(p) => botbowl_play::bots::load_mcts_config(p)?.config,
        None => botbowl_mcts::MctsConfig::new(),
    };
    let nn = match &args.model {
        Some(p) => Some(Arc::new(
            NnEvaluator::from_path(p).map_err(|e| io::Error::other(format!("{p}: {e}")))?,
        )),
        None => None,
    };
    let shared = Arc::new(Shared {
        knobs,
        mcts_iters: Some(args.mcts_iters).filter(|&n| n > 0),
        mcts_config,
        nn,
    });
    fs::create_dir_all(&args.out)?;
    let out_dir = PathBuf::from(&args.out);
    install_panic_hook();

    let started = Instant::now();
    // `--only-seed`: one game, on this thread, optionally dumping its states.
    if let Some(seed) = args.only_seed {
        let cfg = game_cfg(seed, &shared.knobs);
        eprintln!("cfg: {}", serde_json::to_string(&cfg).unwrap());
        let mut dump = match args.dump_from {
            Some(from) => Some(Dump {
                from,
                out: BufWriter::new(File::create(out_dir.join(format!("seed{seed}.states.jsonl")))?),
            }),
            None => None,
        };
        let mut stats = Stats::default();
        let f = play_game(&cfg, &shared, &mut stats, &mut dump);
        if let Some(d) = dump.as_mut() {
            d.out.flush()?;
        }
        match f {
            Some(f) => println!("{}", serde_json::to_string_pretty(&f).unwrap()),
            None => println!("seed {seed}: ok, {} steps, {} decisions", stats.steps, stats.decisions),
        }
        return Ok(());
    }

    let threads = if args.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        args.threads
    };
    let next = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let failures = Arc::new(Mutex::new(BufWriter::new(File::create(
        out_dir.join("failures.jsonl"),
    )?)));
    let total = Arc::new(Mutex::new(Stats::default()));
    let by_class = Arc::new(Mutex::new(BTreeMap::<&'static str, u64>::new()));
    let distinct = Arc::new(Mutex::new(BTreeMap::<(String, String), (u64, u64)>::new()));
    let deadline = (args.minutes > 0.0).then(|| started + Duration::from_secs_f64(args.minutes * 60.0));

    let mut handles = Vec::new();
    for _ in 0..threads {
        let (shared, next, stop, failures, total, by_class, distinct) = (
            Arc::clone(&shared),
            Arc::clone(&next),
            Arc::clone(&stop),
            Arc::clone(&failures),
            Arc::clone(&total),
            Arc::clone(&by_class),
            Arc::clone(&distinct),
        );
        let (games, base) = (args.games, args.seed);
        handles.push(std::thread::spawn(move || {
            let mut stats = Stats::default();
            loop {
                if stop.load(Ordering::Relaxed) || deadline.is_some_and(|d| Instant::now() >= d) {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if games > 0 && i >= games {
                    break;
                }
                let cfg = game_cfg(base + i, &shared.knobs);
                if let Some(f) = play_game(&cfg, &shared, &mut stats, &mut None) {
                    let mut w = failures.lock().unwrap();
                    let _ = serde_json::to_writer(&mut *w, &f);
                    let _ = writeln!(w);
                    let _ = w.flush();
                    *by_class.lock().unwrap().entry(f.class).or_default() += 1;
                    let key = (f.class.to_string(), normalise(&f.message));
                    let mut d = distinct.lock().unwrap();
                    d.entry(key).and_modify(|e| e.1 += 1).or_insert((f.seed, 1));
                }
                if stats.games % 50 == 0 {
                    total.lock().unwrap().merge(&std::mem::take(&mut stats));
                }
            }
            total.lock().unwrap().merge(&stats);
        }));
    }
    // Progress lines while the workers run.
    let reporter = {
        let (total, by_class, stop) = (Arc::clone(&total), Arc::clone(&by_class), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut last = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if last.elapsed() >= Duration::from_secs(60) {
                    last = Instant::now();
                    let t = total.lock().unwrap();
                    eprintln!(
                        "[fuzz {:>6.0}s] games {} steps {} decisions {} failures {:?}",
                        started.elapsed().as_secs_f64(),
                        t.games,
                        t.steps,
                        t.decisions,
                        by_class.lock().unwrap()
                    );
                }
            }
        })
    };
    for h in handles {
        let _ = h.join();
    }
    stop.store(true, Ordering::Relaxed);
    let _ = reporter.join();
    failures.lock().unwrap().flush()?;

    let stats = total.lock().unwrap().clone();
    let by_class = by_class.lock().unwrap().clone();
    let distinct: Vec<(String, String, u64, u64)> = distinct
        .lock()
        .unwrap()
        .iter()
        .map(|((c, m), (s, n))| (c.clone(), m.clone(), *s, *n))
        .collect();
    let summary = Summary {
        commit: botbowl_data::git_commit().to_string(),
        capacity: format!("{}x{}/{}", WIDTH - 2, HEIGHT - 2, TEAM_SIZE),
        boards_spec: args.boards.clone(),
        seconds: started.elapsed().as_secs_f64(),
        stats: &stats,
        failures_by_class: by_class.clone(),
        distinct: distinct.clone(),
    };
    fs::write(
        out_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )?;
    eprintln!(
        "fuzz done in {:.0}s: {} games, {} micro-steps, {} decisions, {} rolls, {} replays; failures {:?}",
        summary.seconds, stats.games, stats.steps, stats.decisions, stats.rolls, stats.replays, by_class
    );
    for (c, m, s, n) in &distinct {
        eprintln!("  {c:<12} x{n:<6} first seed {s:<10} {m}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs(boards: &str) -> Knobs {
        Knobs {
            boards: boards_from(boards).unwrap(),
            start: None,
            policy: None,
            replay_share: 1.0,
            rolldice_share: 0.25,
            wide_share: 0.5,
            flat_share: 0.5,
            hidden_prob: 0.05,
            mcts_prob: 0.0,
            nn_prob: 0.0,
            apply_every: 2,
            serde_every: 4,
            max_steps: 400_000,
        }
    }

    fn shared(boards: &str) -> Shared {
        Shared {
            knobs: knobs(boards),
            mcts_iters: None,
            mcts_config: botbowl_mcts::MctsConfig::new(),
            nn: None,
        }
    }

    /// A handful of games over the loop's mix pass every check, and replay.
    #[test]
    fn a_few_games_pass_every_check() {
        install_panic_hook();
        let sh = shared("loop");
        let mut stats = Stats::default();
        for seed in 0..12 {
            let cfg = game_cfg(seed, &sh.knobs);
            if let Some(f) = play_game(&cfg, &sh, &mut stats, &mut None) {
                panic!("{}", serde_json::to_string_pretty(&f).unwrap());
            }
        }
        assert_eq!(stats.games, 12);
        assert_eq!(stats.game_overs, 12);
        assert_eq!(stats.replays, 12);
    }

    /// The checks see a corrupted state: they are not vacuous.
    #[test]
    fn the_checks_catch_corruption() {
        let sh = shared("loop");
        let cfg = game_cfg(3, &sh.knobs);
        let state = build_start(&GameCfg {
            start: Start::Random,
            ..cfg
        });
        let prev = Prev::of(&state);
        assert!(check_always(&state, &prev, false).is_ok());

        // A carrier who is not on the pitch.
        let mut s = state.clone();
        let id = s
            .get_players_on_pitch()
            .map(|p| p.id)
            .find(|&i| s.ball != BallState::Carried(i))
            .unwrap();
        s.unfield_player(id, botbowl_engine::core::model::DugoutPlace::Reserves)
            .unwrap();
        s.ball = BallState::Carried(id);
        assert_eq!(check_always(&s, &prev, false).unwrap_err().0, "ball");

        let mut s = state.clone();
        s.ball = BallState::OnGround(Position::new((0, 0)));
        assert_eq!(check_always(&s, &prev, false).unwrap_err().0, "ball");

        let mut s = state.clone();
        s.info.half = prev.half + 1;
        s.info.home_turn = 9;
        assert_eq!(check_always(&s, &prev, false).unwrap_err().0, "counters");

        let mut later = state.clone();
        later.home.score += 1;
        let prev_later = Prev::of(&later);
        assert_eq!(check_always(&state, &prev_later, false).unwrap_err().0, "counters");

        let mut s = state.clone();
        let id = s
            .get_players_on_pitch()
            .map(|p| p.id)
            .find(|&i| s.ball != BallState::Carried(i))
            .unwrap();
        s.unfield_player(id, botbowl_engine::core::model::DugoutPlace::Reserves)
            .unwrap();
        let moved = Prev::of(&s);
        assert!(check_always(&s, &moved, false).is_ok(), "unfielding keeps the roster");
        assert_eq!(moved.roster, prev.roster);
    }
}
