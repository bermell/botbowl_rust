//! Random-but-plausible mid-game state generation for self-play training
//! data (plan 019).
//!
//! Kickoff-started self-play with untrained bots produces homogeneous
//! states (most players never move). Instead we place the ball, then place
//! players team-by-team sampling squares from a weighted distribution, then
//! randomize the game context (half, turn, score, active team) so the value
//! head sees diverse clock/score situations.
//!
//! Structure comes from three mechanisms:
//! - **Roles**: each team is split into *line* players (the front brawl),
//!   *pocket* players (near the ball), and *wide* players (anywhere).
//! - **Engagement line**: a front is sampled a few squares ahead of the
//!   ball toward the attacking team's endzone; each team gets its own
//!   facing column there, and line players decay away from it.
//! - **Per-square decay**: distance biases divide the probability for every
//!   square of distance (`bias^-d`), so they stay effective against the
//!   quadratic growth of far-away area. A global temperature sharpens or
//!   flattens the final distribution.

use botbowl_engine::core::dices::D6Target;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{other_team, BoardDims, Coord, Direction, PlayerStats, Position, TeamType};
use botbowl_engine::core::table::{SimpleAT, Skill};
use rand::{Rng, RngCore};
use rand_chacha::ChaCha8Rng;

/// Clamp range for the multiplicative property biases (`mark_*`, `own_side`).
pub const BIAS_MIN: f32 = 0.05;
pub const BIAS_MAX: f32 = 100.0;
/// Clamp range for the per-square decay biases (`ball_distance`, `front_line`).
pub const DECAY_MIN: f32 = 1.0;
pub const DECAY_MAX: f32 = 4.0;
/// Clamp range for `temperature`.
pub const TEMP_MIN: f32 = 0.2;
pub const TEMP_MAX: f32 = 5.0;

/// Per-stat probability that a generated player's stat line is nudged off the
/// lineman baseline, and the geometric continuation probability for handing
/// out skills (so a player gets `k` skills with probability ~`p^k (1-p)`).
const MOD_STR_PROB: f32 = 0.1;
const MOD_MA_PROB: f32 = 0.1;
const MOD_AGI_PROB: f32 = 0.1;
const MOD_ARMOR_PROB: f32 = 0.1;
const MOD_PASS_PROB: f32 = 0.1;
const ADD_SKILL_PROB: f32 = 0.5;

/// Legal ranges for the nudged stats. `ag`/`pass` are D6 target numbers
/// (lower is better), `str_`/`ma`/`av` are plain characteristics.
///
/// Every upper bound stays at or under the matching `PlayerStats::MAX_*` cap,
/// which the NN encoder divides by — a generator that exceeded one would emit
/// a feature plane above 1.0 and break the side-factored encoding. Pinned by
/// `sampling_ranges_stay_under_the_engine_caps`.
const STR_RANGE: (u8, u8) = (1, 6);
const MA_RANGE: (u8, u8) = (1, 9);
const AG_RANGE: (u8, u8) = (1, PlayerStats::MAX_AG);
const AV_RANGE: (u8, u8) = (3, 11);
const PASS_RANGE: (u8, u8) = (2, 6);

/// Bias weights for [`generate_random_start`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomStartConfig {
    /// Per-square decay toward the ball: each square of distance divides the
    /// probability by this (pocket players; line players use it on the
    /// y-distance so the brawl centers near the ball laterally). 1.0 = off.
    pub ball_distance: f32,
    /// Per-square decay toward the team's front column for line players.
    /// 1.0 = off.
    pub front_line: f32,
    /// Multiplier for squares adjacent to an already-placed teammate.
    pub mark_teammate: f32,
    /// Multiplier for squares adjacent to an already-placed opponent.
    pub mark_opponent: f32,
    /// Multiplier for squares between the team's own endzone and its closest
    /// opponent.
    pub own_side: f32,
    /// Sharpens (<1) or flattens (>1) the final square distribution:
    /// `weight^(1/temperature)`.
    pub temperature: f32,
    /// Probability that the ball starts carried by a player.
    pub carried_prob: f32,
    /// Fraction of each team assigned to the line role.
    pub line_fraction: f32,
    /// Fraction of each team assigned to the pocket role (rest are wide).
    pub pocket_fraction: f32,
    /// Board to generate for; `None` uses `BoardDims::from_env()`.
    pub board_dims: Option<BoardDims>,
}

// Keep in sync with the clap defaults on `BiasArgs` in botbowl-ui.
// Values tuned 2026-07-18 with the placement visualizer (plan 020);
// generation alternates temperature 0.6 / 1.5 per game — 0.6 is the
// sharp default here, the flat 1.5 comes from `BiasArgs::temperature2`.
impl Default for RandomStartConfig {
    fn default() -> Self {
        Self {
            ball_distance: 1.30,
            front_line: 2.20,
            mark_teammate: 1.5,
            mark_opponent: 1.5,
            own_side: 1.5,
            temperature: 0.6,
            carried_prob: 0.75,
            line_fraction: 0.80,
            pocket_fraction: 0.25,
            board_dims: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Role {
    Line,
    Pocket,
    Wide,
}

/// Clamped biases plus the sampled front geometry, fixed for one generation.
struct Placement {
    ball_distance: f32,
    front_line: f32,
    mark_teammate: f32,
    mark_opponent: f32,
    own_side: f32,
    inv_temperature: f32,
    ball_pos: Position,
    carried: bool,
    /// Front column per team, indexed by `front_x_index(team)`.
    front_x: [Coord; 2],
}

fn front_x_index(team: TeamType) -> usize {
    match team {
        TeamType::Home => 0,
        TeamType::Away => 1,
    }
}

/// Generate a random mid-game state per `cfg`. All randomness comes from
/// `rng`; the returned state is seeded from it, set to
/// [`DiceMode::RollDice`], and does not print to stdout.
pub fn generate_random_start(cfg: &RandomStartConfig, rng: &mut ChaCha8Rng) -> GameState {
    let dims = cfg.board_dims.unwrap_or_else(BoardDims::from_env);

    // The ball never starts in an endzone column: a pre-placed carrier there
    // would be a touchdown the engine never awarded.
    let ball_pos = Position::new((rng.gen_range(2..=dims.width - 3), rng.gen_range(1..=dims.height - 2)));
    let carried = rng.gen::<f32>() < cfg.carried_prob.clamp(0.0, 1.0);
    // The attacker orients the front; with a carried ball it is the carrier's
    // team, with a loose ball a nominal choice.
    let attacker = if rng.gen::<bool>() {
        TeamType::Home
    } else {
        TeamType::Away
    };

    // The engagement line sits a few squares ahead of the ball toward the
    // attacker's target endzone; the defenders' column faces it one square
    // further. The carrier thus ends up a few squares behind their own line.
    let dir: Coord = if dims.endzone_x(attacker) < ball_pos.x { -1 } else { 1 };
    let offset: Coord = rng.gen_range(2..=4);
    let attacker_front = (ball_pos.x + dir * offset).clamp(2, dims.width - 3);
    let defender_front = (attacker_front + dir).clamp(2, dims.width - 3);
    let mut front_x = [0; 2];
    front_x[front_x_index(attacker)] = attacker_front;
    front_x[front_x_index(other_team(attacker))] = defender_front;

    let placement = Placement {
        ball_distance: cfg.ball_distance.clamp(DECAY_MIN, DECAY_MAX),
        front_line: cfg.front_line.clamp(DECAY_MIN, DECAY_MAX),
        mark_teammate: cfg.mark_teammate.clamp(BIAS_MIN, BIAS_MAX),
        mark_opponent: cfg.mark_opponent.clamp(BIAS_MIN, BIAS_MAX),
        own_side: cfg.own_side.clamp(BIAS_MIN, BIAS_MAX),
        inv_temperature: 1.0 / cfg.temperature.clamp(TEMP_MIN, TEMP_MAX),
        ball_pos,
        carried,
        front_x,
    };

    let mut home_roles = role_queue(sample_player_count(dims.team_size, rng), cfg);
    let mut away_roles = role_queue(sample_player_count(dims.team_size, rng), cfg);

    let mut placed: Vec<(Position, TeamType)> = Vec::with_capacity(home_roles.len() + away_roles.len());
    let mut turn_of = attacker;
    if carried {
        // The attacker places first; their first player is the carrier.
        placed.push((ball_pos, attacker));
        let roles = match attacker {
            TeamType::Home => &mut home_roles,
            TeamType::Away => &mut away_roles,
        };
        remove_carrier_role(roles);
        turn_of = other_team(attacker);
    }
    while !home_roles.is_empty() || !away_roles.is_empty() {
        let roles = match turn_of {
            TeamType::Home => &mut home_roles,
            TeamType::Away => &mut away_roles,
        };
        if !roles.is_empty() {
            let role = roles.remove(0);
            let pos = sample_square(&placement, dims, turn_of, role, &placed, rng);
            placed.push((pos, turn_of));
        }
        turn_of = other_team(turn_of);
    }

    let mut builder = GameStateBuilder::new();
    builder
        .with_board_dims(dims)
        .set_state(BuilderState::Turn { turn: 1 })
        .add_ball_pos(ball_pos);
    for (pos, team) in &placed {
        builder.add_player_details(*pos, *team, sample_player(rng, *team));
    }
    let mut state = builder.build();
    // Quiet the engine's stdout log before stepping the state below —
    // generation must not print (it runs inside the raw-mode TUI).
    state.set_logging_state(false);

    apply_game_context(&mut state, rng);

    state.set_seed(rng.next_u64());
    state.set_dice_mode(DiceMode::RollDice);
    state
}

/// Nudge `start` by -2..=2, clamped into `[min, max]`. Signed arithmetic
/// throughout: an unsigned intermediate wraps on a negative delta and the
/// clamp then silently returns `max`.
fn add_random(rng: &mut ChaCha8Rng, start: u8, (min, max): (u8, u8)) -> u8 {
    let delta: i16 = rng.gen_range(-2..=2);
    (start as i16 + delta).clamp(min as i16, max as i16) as u8
}

/// Generate a random player stat line: a lineman with each characteristic
/// independently nudged with small probability, plus a geometric number of
/// skills drawn **without replacement** from [`Skill::good_skills`].
///
/// Only `good_skills` is sampled — the engine implements a subset of
/// `Skill::ALL`, and the network has a plane for every variant regardless, so
/// widening the pool here is a one-line change once more skills have rules.
fn sample_player(rng: &mut ChaCha8Rng, team: TeamType) -> PlayerStats {
    let mut stats = PlayerStats::new_lineman(team);

    if rng.gen::<f32>() < MOD_STR_PROB {
        stats.str_ = add_random(rng, stats.str_, STR_RANGE);
    }
    if rng.gen::<f32>() < MOD_MA_PROB {
        stats.ma = add_random(rng, stats.ma, MA_RANGE);
    }
    if rng.gen::<f32>() < MOD_AGI_PROB {
        stats.ag = add_random(rng, stats.ag, AG_RANGE);
    }
    if rng.gen::<f32>() < MOD_ARMOR_PROB {
        stats.av = add_random(rng, stats.av, AV_RANGE);
    }
    if rng.gen::<f32>() < MOD_PASS_PROB {
        let pass = add_random(rng, stats.pass as u8, PASS_RANGE);
        stats.pass = D6Target::try_from(pass).expect("pass target kept inside PASS_RANGE");
    }

    let mut pool = Skill::good_skills();
    while !pool.is_empty() && rng.gen::<f32>() < ADD_SKILL_PROB {
        let index = rng.gen_range(0..pool.len());
        stats.give_skill(pool.swap_remove(index));
    }

    stats
}

/// Per-team player count, skewed toward full strength: for team size 11 the
/// counts 11..=7 have probability .50/.20/.15/.10/.05.
fn sample_player_count(team_size: usize, rng: &mut ChaCha8Rng) -> usize {
    const BASE: [f32; 5] = [0.50, 0.20, 0.15, 0.10, 0.05];
    let min_count = team_size.saturating_sub(4).max(1);
    let weights = &BASE[..team_size - min_count + 1];
    let total: f32 = weights.iter().sum();
    let mut r = rng.gen::<f32>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return team_size - i;
        }
    }
    min_count
}

/// The team's placement order: line players first (they seed the front),
/// then pocket, then wide.
fn role_queue(count: usize, cfg: &RandomStartConfig) -> Vec<Role> {
    let line = ((count as f32) * cfg.line_fraction.clamp(0.0, 1.0)).round() as usize;
    let line = line.min(count);
    let pocket = (((count as f32) * cfg.pocket_fraction.clamp(0.0, 1.0)).round() as usize).min(count - line);
    let mut queue = Vec::with_capacity(count);
    queue.extend(std::iter::repeat_n(Role::Line, line));
    queue.extend(std::iter::repeat_n(Role::Pocket, pocket));
    queue.extend(std::iter::repeat_n(Role::Wide, count - line - pocket));
    queue
}

/// The carrier fills the team's pocket slot when there is one.
fn remove_carrier_role(roles: &mut Vec<Role>) {
    if let Some(i) = roles.iter().position(|r| *r == Role::Pocket) {
        roles.remove(i);
    } else {
        roles.pop();
    }
}

fn sample_square(
    placement: &Placement,
    dims: BoardDims,
    team: TeamType,
    role: Role,
    placed: &[(Position, TeamType)],
    rng: &mut ChaCha8Rng,
) -> Position {
    let mut squares: Vec<(Position, f32)> = Vec::new();
    let mut total = 0.0f32;
    for x in 1..=dims.width - 2 {
        for y in 1..=dims.height - 2 {
            let s = Position::new((x, y));
            if placed.iter().any(|(p, _)| *p == s) {
                continue;
            }
            // A standing player on a loose ball's square is not a real
            // Blood Bowl state — they would have picked it up or bounced it.
            if !placement.carried && s == placement.ball_pos {
                continue;
            }
            let w = square_weight(placement, dims, team, role, s, placed);
            total += w;
            squares.push((s, w));
        }
    }
    // Strong decays can underflow every candidate to 0.0 (e.g. all near-ball
    // squares taken); fall back to a uniform pick.
    if total <= 0.0 {
        let i = rng.gen_range(0..squares.len());
        return squares[i].0;
    }
    let mut r = rng.gen::<f32>() * total;
    for (s, w) in &squares {
        r -= w;
        if r <= 0.0 {
            return *s;
        }
    }
    squares.last().expect("board has no free squares").0
}

fn square_weight(
    placement: &Placement,
    dims: BoardDims,
    team: TeamType,
    role: Role,
    s: Position,
    placed: &[(Position, TeamType)],
) -> f32 {
    let ball_pos = placement.ball_pos;
    let mut w = match role {
        // Pinned to the team's front column, pulled toward the ball's y so
        // the brawl forms near the ball laterally.
        Role::Line => {
            placement
                .front_line
                .powi(-i32::from((s.x - placement.front_x[front_x_index(team)]).abs()))
                * placement.ball_distance.powi(-i32::from((s.y - ball_pos.y).abs()))
        }
        Role::Pocket => placement.ball_distance.powi(-i32::from(s.distance_to(&ball_pos))),
        Role::Wide => 1.0,
    };

    let mut adjacent_teammate = false;
    let mut adjacent_opponent = false;
    for dir in Direction::all_directions_iter() {
        let n = s + *dir;
        if let Some((_, t)) = placed.iter().find(|(p, _)| *p == n) {
            if *t == team {
                adjacent_teammate = true;
            } else {
                adjacent_opponent = true;
            }
        }
    }
    if adjacent_teammate {
        w *= placement.mark_teammate;
    }
    if adjacent_opponent {
        w *= placement.mark_opponent;
    }
    if own_side_feature(dims, team, s, placed) {
        w *= placement.own_side;
    }
    w.powf(placement.inv_temperature)
}

/// True if `s` lies strictly between the team's defended endzone and the
/// opponent closest to it. Before any opponent is placed, falls back to the
/// team's own half.
fn own_side_feature(dims: BoardDims, team: TeamType, s: Position, placed: &[(Position, TeamType)]) -> bool {
    let opponent_xs = placed.iter().filter(|(_, t)| *t != team).map(|(p, _)| p.x);
    match team {
        // Home defends the high-x endzone, Away the low-x one.
        TeamType::Home => match opponent_xs.max() {
            Some(threshold) => s.x > threshold,
            None => dims.is_on_team_side(s, team),
        },
        TeamType::Away => match opponent_xs.min() {
            Some(threshold) => s.x < threshold,
            None => dims.is_on_team_side(s, team),
        },
    }
}

/// Randomize active team, turn, half, and score on a freshly built
/// turn-1-Home-active state, using only the public engine API (see plan 019:
/// the counter pattern mirrors exactly what `Half::step` produces).
fn apply_game_context(state: &mut GameState, rng: &mut ChaCha8Rng) {
    let active = if rng.gen::<bool>() {
        TeamType::Home
    } else {
        TeamType::Away
    };
    if active == TeamType::Away {
        // Hand the turn to Away through the engine so available_actions,
        // team_turn and player flags are all regenerated consistently.
        state.step_simple(SimpleAT::EndTurn);
    }

    let turn: u8 = rng.gen_range(1..=8);
    state.info.home_turn = turn;
    state.info.away_turn = if active == TeamType::Home { turn - 1 } else { turn };

    let half: u8 = rng.gen_range(1..=2);
    if half == 2 {
        state.set_half(2);
    }

    // Cap scores by elapsed game time: a touchdown drive takes ~3 turns.
    let elapsed = (half - 1) * 8 + turn - 1;
    let max_td = (elapsed / 3).min(3);
    state.home.score = rng.gen_range(0..=max_td);
    state.away.score = rng.gen_range(0..=max_td);
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::bots::{Bot, RandomBot};
    use botbowl_engine::core::model::BallState;
    use rand::SeedableRng;
    use std::collections::{HashMap, HashSet};

    fn generate(cfg: &RandomStartConfig, seed: u64) -> GameState {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        generate_random_start(cfg, &mut rng)
    }

    /// The point of `sample_player`: a generated corpus must contain players
    /// that are *not* the lineman baseline, in every varied dimension, and
    /// must hand out skills. One state is far too few (each nudge fires at
    /// 10%), so this samples across seeds.
    #[test]
    fn generated_players_vary_in_stats_and_skills() {
        let cfg = RandomStartConfig::default();
        let mut str_ = HashSet::new();
        let mut ma = HashSet::new();
        let mut ag = HashSet::new();
        let mut av = HashSet::new();
        let mut pass = HashSet::new();
        let mut skilled = 0usize;
        let mut total = 0usize;
        for seed in 0..40 {
            for p in generate(&cfg, seed).get_players_on_pitch() {
                total += 1;
                str_.insert(p.stats.str_);
                ma.insert(p.stats.ma);
                ag.insert(p.stats.ag);
                av.insert(p.stats.av);
                pass.insert(p.stats.pass as u8);
                if !p.stats.skills.is_empty() {
                    skilled += 1;
                }
            }
        }
        assert!(total > 100, "expected a decent sample, got {total} players");
        assert!(str_.len() > 1, "strength never varied: {str_:?}");
        assert!(ma.len() > 1, "movement never varied: {ma:?}");
        assert!(ag.len() > 1, "agility never varied: {ag:?}");
        assert!(av.len() > 1, "armour never varied: {av:?}");
        assert!(pass.len() > 1, "passing never varied: {pass:?}");
        // ADD_SKILL_PROB = 0.5 ⇒ ~half the players get at least one skill.
        assert!(skilled * 4 > total, "only {skilled} of {total} players got a skill");
    }

    /// Skills come from `good_skills` only, and never twice on one player —
    /// the pool is drawn without replacement.
    #[test]
    fn generated_skills_come_only_from_the_good_pool() {
        let good: HashSet<Skill> = Skill::good_skills().into_iter().collect();
        let cfg = RandomStartConfig::default();
        for seed in 0..40 {
            for p in generate(&cfg, seed).get_players_on_pitch() {
                for skill in &p.stats.skills {
                    assert!(good.contains(skill), "seed {seed}: {skill:?} is not a good skill");
                }
                assert!(p.stats.skills.len() <= good.len());
            }
        }
    }

    /// Stats stay inside their declared ranges — `add_random` used to do its
    /// arithmetic in `u16`, where a negative delta wraps and the clamp then
    /// silently returns the maximum.
    #[test]
    fn generated_stats_stay_in_range() {
        let cfg = RandomStartConfig::default();
        for seed in 0..40 {
            for p in generate(&cfg, seed).get_players_on_pitch() {
                let st = &p.stats;
                assert!((STR_RANGE.0..=STR_RANGE.1).contains(&st.str_), "str {}", st.str_);
                assert!((MA_RANGE.0..=MA_RANGE.1).contains(&st.ma), "ma {}", st.ma);
                assert!((AG_RANGE.0..=AG_RANGE.1).contains(&st.ag), "ag {}", st.ag);
                assert!((AV_RANGE.0..=AV_RANGE.1).contains(&st.av), "av {}", st.av);
                let pass = st.pass as u8;
                assert!((PASS_RANGE.0..=PASS_RANGE.1).contains(&pass), "pass {pass}");
            }
        }
    }

    /// `add_random` is symmetric around its input and respects the bounds.
    #[test]
    fn add_random_never_wraps() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut seen = HashSet::new();
        for _ in 0..500 {
            let v = add_random(&mut rng, 1, (1, 9));
            assert!((1..=3).contains(&v), "1 nudged out of range: {v}");
            seen.insert(v);
        }
        assert_eq!(seen, HashSet::from([1, 2, 3]), "clamping at the floor lost values");
    }

    /// The generator must not outrun the encoder's normalisers.
    #[test]
    fn sampling_ranges_stay_under_the_engine_caps() {
        assert!(STR_RANGE.1 <= PlayerStats::MAX_ST);
        assert!(MA_RANGE.1 <= PlayerStats::MAX_MA);
        assert!(AG_RANGE.1 <= PlayerStats::MAX_AG);
        assert!(AV_RANGE.1 <= PlayerStats::MAX_AV);
    }

    /// A config with every mechanism switched off.
    fn neutral() -> RandomStartConfig {
        RandomStartConfig {
            ball_distance: 1.0,
            front_line: 1.0,
            mark_teammate: 1.0,
            mark_opponent: 1.0,
            own_side: 1.0,
            temperature: 1.0,
            carried_prob: 0.75,
            line_fraction: 0.0,
            pocket_fraction: 0.0,
            board_dims: None,
        }
    }

    fn ball_square(state: &GameState) -> Position {
        match state.ball {
            BallState::OnGround(pos) => pos,
            BallState::Carried(id) => {
                state
                    .get_players_on_pitch()
                    .find(|p| p.id == id)
                    .expect("carrier is on the pitch")
                    .position
            }
            other => panic!("unexpected ball state: {other:?}"),
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let cfg = RandomStartConfig::default();
        assert_eq!(generate(&cfg, 7), generate(&cfg, 7));
    }

    #[test]
    fn players_on_board_and_counts() {
        let cfg = RandomStartConfig::default();
        for seed in 0..50 {
            let state = generate(&cfg, seed);
            let dims = state.board_dims;
            let mut seen = HashSet::new();
            let (mut home, mut away) = (0usize, 0usize);
            for player in state.get_players_on_pitch() {
                assert!(!dims.is_out(player.position), "seed {seed}: player out of bounds");
                assert!(seen.insert(player.position), "seed {seed}: duplicate square");
                match player.stats.team {
                    TeamType::Home => home += 1,
                    TeamType::Away => away += 1,
                }
            }
            let min_count = dims.team_size.saturating_sub(4).max(1);
            assert!((min_count..=dims.team_size).contains(&home), "seed {seed}: home={home}");
            assert!((min_count..=dims.team_size).contains(&away), "seed {seed}: away={away}");
        }
    }

    #[test]
    fn ball_rules() {
        let carried_cfg = RandomStartConfig {
            carried_prob: 1.0,
            ..Default::default()
        };
        let loose_cfg = RandomStartConfig {
            carried_prob: 0.0,
            ..Default::default()
        };
        for seed in 0..30 {
            let state = generate(&carried_cfg, seed);
            assert!(
                matches!(state.ball, BallState::Carried(_)),
                "seed {seed}: {:?}",
                state.ball
            );
            let pos = ball_square(&state);
            let dims = state.board_dims;
            assert!(
                (2..=dims.width - 3).contains(&pos.x),
                "seed {seed}: carrier in endzone column"
            );

            let state = generate(&loose_cfg, seed);
            let BallState::OnGround(pos) = state.ball else {
                panic!("seed {seed}: expected loose ball, got {:?}", state.ball);
            };
            assert!(state.get_player_at(pos).is_none(), "seed {seed}: player on loose ball");
            assert!((2..=state.board_dims.width - 3).contains(&pos.x));
            assert!((1..=state.board_dims.height - 2).contains(&pos.y));
        }
    }

    #[test]
    fn context_fields_valid() {
        let cfg = RandomStartConfig::default();
        let mut actives_seen = HashSet::new();
        let mut halves_seen = HashSet::new();
        for seed in 0..50 {
            let state = generate(&cfg, seed);
            let info = &state.info;
            assert!((1..=2).contains(&info.half), "seed {seed}");
            assert!((1..=8).contains(&info.home_turn), "seed {seed}");
            let active = if info.home_turn > info.away_turn {
                assert_eq!(info.away_turn, info.home_turn - 1, "seed {seed}");
                TeamType::Home
            } else {
                assert_eq!(info.away_turn, info.home_turn, "seed {seed}");
                TeamType::Away
            };
            assert_eq!(state.available_actions.team, Some(active), "seed {seed}");
            assert_eq!(info.team_turn, active, "seed {seed}");
            assert!(!state.is_logging(), "generated states must not print to stdout");
            actives_seen.insert(active);
            halves_seen.insert(info.half);

            let elapsed = (info.half - 1) * 8 + info.home_turn - 1;
            let max_td = (elapsed / 3).min(3);
            assert!(state.home.score <= max_td, "seed {seed}");
            assert!(state.away.score <= max_td, "seed {seed}");
        }
        assert_eq!(actives_seen.len(), 2, "both teams should appear as active");
        assert_eq!(halves_seen.len(), 2, "both halves should appear");
    }

    #[test]
    fn ball_distance_pulls_players_toward_ball() {
        let mean_ball_dist = |decay: f32| -> f32 {
            let cfg = RandomStartConfig {
                ball_distance: decay,
                pocket_fraction: 1.0, // everyone samples with the ball pull
                carried_prob: 0.0,
                ..neutral()
            };
            let (mut sum, mut n) = (0.0f32, 0usize);
            for seed in 0..20 {
                let state = generate(&cfg, seed);
                let ball = ball_square(&state);
                for player in state.get_players_on_pitch() {
                    sum += player.position.distance_to(&ball) as f32;
                    n += 1;
                }
            }
            sum / n as f32
        };
        let (pulled, free) = (mean_ball_dist(2.0), mean_ball_dist(1.0));
        assert!(
            pulled < free * 0.5,
            "per-square decay should strongly reduce mean distance to ball ({pulled} vs {free})"
        );
    }

    #[test]
    fn mark_teammate_clusters_teammates() {
        let adjacent_teammate_fraction = |bias: f32| -> f32 {
            let cfg = RandomStartConfig {
                mark_teammate: bias,
                ..neutral()
            };
            let (mut adjacent, mut n) = (0usize, 0usize);
            for seed in 0..20 {
                let state = generate(&cfg, seed);
                let players: Vec<_> = state
                    .get_players_on_pitch()
                    .map(|p| (p.position, p.stats.team))
                    .collect();
                for (pos, team) in &players {
                    let has_neighbor = players
                        .iter()
                        .any(|(p, t)| t == team && p != pos && pos.distance_to(p) == 1);
                    adjacent += has_neighbor as usize;
                    n += 1;
                }
            }
            adjacent as f32 / n as f32
        };
        assert!(
            adjacent_teammate_fraction(50.0) > adjacent_teammate_fraction(1.0),
            "high mark_teammate bias should increase teammate adjacency"
        );
    }

    /// Fraction of each team standing in its 2 most common x-columns,
    /// averaged over teams and seeds — high when a front line forms.
    fn column_concentration(cfg: &RandomStartConfig, seeds: std::ops::Range<u64>) -> f32 {
        let (mut sum, mut n) = (0.0f32, 0usize);
        for seed in seeds {
            let state = generate(cfg, seed);
            for team in [TeamType::Home, TeamType::Away] {
                let mut columns: HashMap<Coord, usize> = HashMap::new();
                let mut count = 0usize;
                for player in state.get_players_on_pitch() {
                    if player.stats.team == team {
                        *columns.entry(player.position.x).or_default() += 1;
                        count += 1;
                    }
                }
                let mut sizes: Vec<usize> = columns.values().copied().collect();
                sizes.sort_unstable_by(|a, b| b.cmp(a));
                let top2: usize = sizes.iter().take(2).sum();
                sum += top2 as f32 / count as f32;
                n += 1;
            }
        }
        sum / n as f32
    }

    #[test]
    fn front_line_forms_columns() {
        let line_cfg = RandomStartConfig {
            front_line: 3.0,
            line_fraction: 1.0,
            ..neutral()
        };
        let lined = column_concentration(&line_cfg, 0..20);
        let scattered = column_concentration(&neutral(), 0..20);
        assert!(
            lined > scattered + 0.25,
            "line role + front decay should concentrate teams in columns ({lined} vs {scattered})"
        );
    }

    #[test]
    fn temperature_sharpens_distribution() {
        let cfg_at = |temperature: f32| RandomStartConfig {
            front_line: 1.5,
            line_fraction: 1.0,
            temperature,
            ..neutral()
        };
        let sharp = column_concentration(&cfg_at(0.3), 0..30);
        let flat = column_concentration(&cfg_at(3.0), 0..30);
        assert!(
            sharp > flat,
            "low temperature should sharpen the same distribution ({sharp} vs {flat})"
        );
    }

    #[test]
    fn small_board_generation() {
        // Engine 16x9/5 is the documented small-board test floor.
        let dims = BoardDims::new(16, 9, 5);
        let cfg = RandomStartConfig {
            board_dims: Some(dims),
            ..Default::default()
        };
        for seed in 0..20 {
            let state = generate(&cfg, seed);
            assert_eq!(state.board_dims, dims, "seed {seed}");
            let mut seen = HashSet::new();
            for player in state.get_players_on_pitch() {
                assert!(!dims.is_out(player.position), "seed {seed}: player out of bounds");
                assert!(seen.insert(player.position), "seed {seed}: duplicate square");
            }
            let ball = ball_square(&state);
            assert!((2..=dims.width - 3).contains(&ball.x), "seed {seed}: ball x");
        }

        // And a full game still terminates on the small board.
        let mut state = generate(&cfg, 0);
        let mut home = RandomBot::new();
        let mut away = RandomBot::new();
        home.set_seed(ChaCha8Rng::seed_from_u64(0xA));
        away.set_seed(ChaCha8Rng::seed_from_u64(0xB));
        let mut steps = 0u32;
        while !state.info.game_over && steps < 300_000 {
            let action = match state.available_actions.team {
                Some(TeamType::Home) => home.get_action(&state),
                Some(TeamType::Away) => away.get_action(&state),
                None => break,
            };
            state.step(action).expect("engine step failed on small board");
            steps += 1;
        }
        assert!(state.info.game_over, "small-board game did not finish in {steps} steps");
    }

    #[test]
    fn random_start_is_playable() {
        // Prove the proc-stack surgery terminates: RandomBot vs RandomBot
        // reaches game_over from both half-1 and half-2 starts.
        let cfg = RandomStartConfig::default();
        let mut seed_for_half = [None, None];
        for seed in 0..32 {
            let half = generate(&cfg, seed).info.half as usize;
            if seed_for_half[half - 1].is_none() {
                seed_for_half[half - 1] = Some(seed);
            }
        }
        for seed in seed_for_half.into_iter().flatten() {
            let mut state = generate(&cfg, seed);
            let mut home = RandomBot::new();
            let mut away = RandomBot::new();
            home.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xA));
            away.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB));
            let mut steps = 0u32;
            while !state.info.game_over && steps < 300_000 {
                let action = match state.available_actions.team {
                    Some(TeamType::Home) => home.get_action(&state),
                    Some(TeamType::Away) => away.get_action(&state),
                    None => break,
                };
                state.step(action).expect("engine step failed");
                steps += 1;
            }
            assert!(
                state.info.game_over,
                "seed {seed}: game did not finish in {steps} steps"
            );
        }
    }
}
