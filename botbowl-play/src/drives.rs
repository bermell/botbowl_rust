//! Paired contested drives: a ladder rung that plays single drives from frozen random-start
//! positions instead of full games (plan 051 step 4).
//!
//! A drive is about a fifth of a game and is what the net is trained on. Each position is played
//! twice with the sides swapped and the same dice seed, so the pair folds into
//! [`crate::stats::Pentanomial`] exactly as a mirrored pair of games does. A position set is a
//! *recipe*, not states: workers regenerate each position from `(bias, board, seed)` with
//! `generate_random_start`, which is deterministic at a fixed commit, so a `GameState` never has
//! to be serialised.
//!
//! **Attacker** is the team to move in the generated position, and the defender is the other.
//! When the ball is loose, or carried by the defender, the name is only a convention. What the
//! pairing needs is that each bot plays both sides of one position, and it does whatever the
//! labels mean.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use botbowl_curriculum::generate_random_start;
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::{other_team, Action as EngineAction, BoardDims, TeamType};

use crate::board_sizes::{board_label, parse_board};
use crate::eval::{EvalGameLine, CANDIDATE_SEED_MIX, OPPONENT_SEED_MIX};
use crate::generate::RandomStartBias;

/// Plan 068: bot seeds by role rather than by bot under the divergence gate, so the attacker of
/// both games of a pair draws the same search noise (and the defender likewise). Without it, a
/// preset with root noise makes even a net against itself diverge at its first decision.
pub(crate) const ATTACKER_SEED_MIX: u64 = 0x5A5A_A5A5_5A5A_A5A5;
pub(crate) const DEFENDER_SEED_MIX: u64 = 0xA5A5_5A5A_A5A5_5A5A;

/// Turns in a half. A position needs at least [`MIN_TURNS_LEFT`] of them left for the side to
/// move, or the clock rather than the bots ends the drive and the sample carries no information.
const TURNS_PER_HALF: u8 = 8;
pub const MIN_TURNS_LEFT: u8 = 4;

/// A frozen set of positions on one board. `positions new` writes one with `seeds` only, and
/// `scripts/positions_screen.py` adds `screen` from a reference self-play rung.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PositionSet {
    /// Short name for rung labels, e.g. `contested_14x7`.
    pub name: String,
    /// The commit the set was built at. Positions regenerate identically only on a binary whose
    /// random-start generator is unchanged; this records which one that was.
    pub commit: String,
    /// Playable board, `14x7/4`.
    pub board: String,
    pub bias: RandomStartBias,
    /// Every candidate position's seed, before screening.
    pub seeds: Vec<u64>,
    #[serde(default)]
    pub screen: Option<Screen>,
}

/// The contested subset: positions whose attacker scored within `band` over `playouts` drives of
/// the reference bot against itself.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Screen {
    pub reference: String,
    pub playouts: u32,
    pub band: (f64, f64),
    pub kept: Vec<KeptPosition>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KeptPosition {
    pub seed: u64,
    pub attacker_td_rate: f64,
}

impl PositionSet {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
    }

    pub fn board_dims(&self) -> Result<BoardDims, String> {
        parse_board(&self.board, 0.0)
    }

    /// The positions a drive rung plays: the screened ones if the set has been screened, else
    /// every candidate.
    pub fn positions(&self) -> Vec<u64> {
        match &self.screen {
            Some(s) => s.kept.iter().map(|k| k.seed).collect(),
            None => self.seeds.clone(),
        }
    }

    /// What a drive rung ships to workers.
    pub fn rung(&self) -> Result<DriveRung, String> {
        let positions = self.positions();
        if positions.is_empty() {
            return Err(format!("position set {} has no positions", self.name));
        }
        Ok(DriveRung {
            set: self.name.clone(),
            bias: self.bias,
            positions,
            gate: false,
        })
    }
}

/// A drive rung's configuration as it travels to the players: enough to regenerate every
/// position. The board is the rung's own (`RungReq.board`, `Task::Eval.board`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DriveRung {
    pub set: String,
    pub bias: RandomStartBias,
    pub positions: Vec<u64>,
    /// Plan 068: play under the divergence gate ([`play_drive_game_gated`]): bots seeded by role
    /// and every line fingerprinted. False is today's drive. On the worker wire since v22.
    #[serde(default)]
    pub gate: bool,
}

/// The rung label for drives from `set` against `opponent` on `board`. The opponent part stays a
/// prefix so `vs:` detection in `eval_summary.py` and `anchor_curve.py` still finds it.
pub fn drive_rung_name(opponent: &str, set: &str, board: BoardDims) -> String {
    format!("{opponent} drives({set})@{}", board_label(board))
}

/// The position for `seed`: the same draw `random_start_trajectory` makes for a corpus game with
/// this seed, temperature alternation included, so the set comes from the training distribution.
pub fn position_state(bias: &RandomStartBias, board: BoardDims, seed: u64) -> GameState {
    let mut rs = bias.to_config();
    if seed % 2 == 1 {
        rs.temperature = bias.temperature2;
    }
    rs.board_dims = Some(board);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = generate_random_start(&rs, &mut rng);
    state.set_logging_state(false);
    state
}

/// The team to move in a fresh position.
pub fn attacker_of(state: &GameState) -> TeamType {
    state
        .available_actions
        .team
        .expect("a generated random-start position always has a team to move")
}

/// Turns the side to move has left in this half, its current one included.
pub fn turns_left(state: &GameState) -> u8 {
    let turn = match attacker_of(state) {
        TeamType::Home => state.info.home_turn,
        TeamType::Away => state.info.away_turn,
    };
    (TURNS_PER_HALF + 1).saturating_sub(turn)
}

/// Which position game `g` plays, whether the candidate is the attacker, and its dice seed.
/// Pair `g / 2` is position `(g / 2) mod n`: the even game puts the candidate on the attacking
/// side and the odd one on the defending side, with one dice seed for both. A rung longer than
/// `2n` games plays every position again with new dice. Pure in `g`, so which worker plays which
/// game cannot change the pairing.
pub fn drive_assignment(n_positions: usize, base_seed: u64, g: u32) -> (usize, bool, u64) {
    assert!(n_positions > 0, "a drive rung needs at least one position");
    let pair = g / 2;
    (
        pair as usize % n_positions,
        g % 2 == 0,
        base_seed.wrapping_add(pair as u64),
    )
}

/// Where a drive stands when it starts, and so when it is over: either side scores, the half
/// changes, or the game ends. `random_start_trajectory` (the corpus's drives) and
/// [`play_drive_game`] (the drive benchmark) both stop on it, so anything else that plays "the
/// rest of the drive" from an arbitrary state (the plan-055 override audit) ends it exactly where
/// they do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriveStart {
    pub home_score: u8,
    pub away_score: u8,
    pub half: u8,
}

impl DriveStart {
    pub fn of(state: &GameState) -> Self {
        DriveStart {
            home_score: state.home.score,
            away_score: state.away.score,
            half: state.info.half,
        }
    }

    pub fn over(&self, state: &GameState) -> bool {
        state.info.game_over
            || state.home.score != self.home_score
            || state.away.score != self.away_score
            || state.info.half != self.half
    }

    /// The drive's own touchdowns so far, `(home, away)`.
    pub fn scored(&self, state: &GameState) -> (u8, u8) {
        (state.home.score - self.home_score, state.away.score - self.away_score)
    }

    /// The drive's outcome for `team`, in the value target's units: +1 if `team` scored, -1 if
    /// the other side did, 0 if nobody has. This is `Trajectory::backfill_outcome_value`'s
    /// Home-centric score delta, clamped to [-1, 1], re-signed by `botbowl_nn::targets::value_target`.
    pub fn outcome_for(&self, state: &GameState, team: TeamType) -> f32 {
        let (home, away) = self.scored(state);
        let home_delta = (home as f32 - away as f32).clamp(-1.0, 1.0);
        match team {
            TeamType::Home => home_delta,
            // `0.0 - x`, not `-x`: a scoreless drive is 0, never -0.
            TeamType::Away => 0.0 - home_delta,
        }
    }
}

/// One drive from `position` between `candidate` and `opponent`. It ends when either side scores,
/// the half changes, the game ends, or `max_steps` runs out, which is `random_start_trajectory`'s
/// rule. The line's `home_score` / `away_score` are the drive's own touchdowns, so it folds
/// through [`crate::eval::LadderRow::record`] like a game: a candidate touchdown is a win and no
/// touchdown is a draw. `seed` is the position's seed, and `attacker` marks the line as a drive.
#[allow(clippy::too_many_arguments)]
pub fn play_drive_game(
    candidate: &mut dyn Bot,
    opponent: &mut dyn Bot,
    rung: &str,
    game: u32,
    position_seed: u64,
    state: GameState,
    candidate_attacks: bool,
    dice_seed: u64,
    max_steps: u32,
) -> EvalGameLine {
    play_drive_game_gated(
        candidate,
        opponent,
        rung,
        game,
        position_seed,
        state,
        candidate_attacks,
        dice_seed,
        max_steps,
        false,
    )
}

/// [`play_drive_game`], optionally under the plan-068 divergence gate (`gate`). Gated, the bots
/// are seeded by role ([`ATTACKER_SEED_MIX`] / [`DEFENDER_SEED_MIX`]) instead of by bot, and the
/// line carries `path`, the fingerprint of every move played ([`MovePath`]). The two games of a
/// pair share their dice, so equal fingerprints mean the two games were the same game: both bots
/// chose the same move at every decision, and the pair scores exactly ½. Ungated, the line is
/// byte-identical to [`play_drive_game`]'s.
#[allow(clippy::too_many_arguments)]
pub fn play_drive_game_gated(
    candidate: &mut dyn Bot,
    opponent: &mut dyn Bot,
    rung: &str,
    game: u32,
    position_seed: u64,
    mut state: GameState,
    candidate_attacks: bool,
    dice_seed: u64,
    max_steps: u32,
    gate: bool,
) -> EvalGameLine {
    let attacker = attacker_of(&state);
    let candidate_team = if candidate_attacks {
        attacker
    } else {
        other_team(attacker)
    };
    state.set_seed(dice_seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    let (cand_mix, opp_mix) = match (gate, candidate_attacks) {
        (false, _) => (CANDIDATE_SEED_MIX, OPPONENT_SEED_MIX),
        (true, true) => (ATTACKER_SEED_MIX, DEFENDER_SEED_MIX),
        (true, false) => (DEFENDER_SEED_MIX, ATTACKER_SEED_MIX),
    };
    candidate.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ cand_mix));
    opponent.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ opp_mix));

    let start = DriveStart::of(&state);
    let mut path = MovePath::default();
    let mut steps = 0u32;
    while !start.over(&state) && steps < max_steps {
        let action = match state.available_actions.team {
            Some(t) if t == candidate_team => candidate.get_action(&state),
            Some(_) => opponent.get_action(&state),
            None => break,
        };
        if gate {
            path.push(action);
        }
        state.step(action).expect("engine step failed during a drive");
        botbowl_mcts::MctsBot::release_stale_tree_of(&mut *candidate, &state);
        botbowl_mcts::MctsBot::release_stale_tree_of(&mut *opponent, &state);
        steps += 1;
    }
    let telemetry = botbowl_mcts::MctsBot::take_telemetry_of(candidate);
    let ended = start.over(&state);
    let (home_score, away_score) = start.scored(&state);
    EvalGameLine {
        rung: rung.to_string(),
        game,
        seed: position_seed,
        candidate_team,
        home_score,
        away_score,
        kicking_first_half: state.info.kicking_first_half,
        finished: ended,
        board: Some(board_label(state.board_dims)),
        telemetry,
        attacker: Some(attacker),
        path: gate.then_some(path.0),
    }
}

/// Plan 068: a fingerprint of the moves a drive played, in order: 64-bit FNV-1a over each action's
/// `(kind, type, x, y)` bytes. It is a function of the actions alone, the same on every platform
/// and compiler (no `std` hasher), so two workers that played the two games of a pair agree on
/// it exactly when the games did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MovePath(pub u64);

impl Default for MovePath {
    fn default() -> Self {
        MovePath(0xcbf2_9ce4_8422_2325)
    }
}

impl MovePath {
    pub fn push(&mut self, action: EngineAction) {
        let bytes = match action {
            EngineAction::Positional(at, pos) => [1u8, at as u8, pos.x as u8, pos.y as u8],
            EngineAction::Simple(at) => [2u8, at as u8, 0, 0],
        };
        for b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

/// Plan 068: the four bot instances of a lockstep pair, one per (bot, role). In the normal pair,
/// game `2k` has the candidate attacking against the opponent defending, and game `2k+1` has the
/// opponent attacking against the candidate defending. Each instance here sees exactly the states
/// its counterpart in those two games sees on the shared prefix, so its trees, and so its moves,
/// are the ones the normal pair would get.
pub struct LockstepBots<'a> {
    pub cand_att: &'a mut dyn Bot,
    pub cand_def: &'a mut dyn Bot,
    pub opp_att: &'a mut dyn Bot,
    pub opp_def: &'a mut dyn Bot,
}

/// One decision of a lockstep drive, as the probe sees it.
pub struct LockstepDecision<'s> {
    pub state: &'s GameState,
    /// Engine steps before this decision.
    pub step: u32,
    /// The attacker is to move.
    pub attacker_to_move: bool,
    /// The candidate's and the opponent's move for the side to move. On the shared prefix the
    /// two agree until the divergence. Past it (with `follow`), game `2k` plays `cand` when the
    /// attacker moves and `opp` when the defender does; the other is the move the counterfactual
    /// bot would have made there.
    pub cand: EngineAction,
    pub opp: EngineAction,
    /// False on the shared prefix, true while following game `2k` past the divergence.
    pub after_divergence: bool,
}

/// What a lockstep run of one position found.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LockstepOutcome {
    /// Engine steps on the shared prefix (every step is a decision by the side to move).
    pub prefix_steps: u32,
    /// The step at which the two bots first chose differently; `None` if they never did, in which
    /// case the two games of the pair are the same game and the pair scores exactly ½.
    pub divergence: Option<u32>,
    /// The drive reached its end (score, half, game over) on the shared prefix.
    pub ended: bool,
    /// `follow`: steps of game `2k` played past the divergence.
    pub followed_steps: u32,
}

/// Plan 068: play one position in lockstep. At every decision both bots of the side to move are
/// asked; while they agree the move is played (it is the same move in both games of the pair,
/// which share their dice), and the first disagreement is the divergence. With `follow`, game
/// `2k` (candidate attacking) is played on to the end of the drive past it, both bots still asked
/// at every decision, so the probe sees agreement over a whole drive. `probe` is called once per
/// decision, before the move is played. Seeds the bots as [`play_drive_game`] does (`role_seeds`
/// = false) or by role (true; see [`DriveGate`]).
#[allow(clippy::too_many_arguments)]
pub fn lockstep_drive(
    bots: LockstepBots<'_>,
    mut state: GameState,
    dice_seed: u64,
    max_steps: u32,
    role_seeds: bool,
    follow: bool,
    mut probe: impl FnMut(&LockstepDecision<'_>),
) -> LockstepOutcome {
    let LockstepBots {
        cand_att,
        cand_def,
        opp_att,
        opp_def,
    } = bots;
    let attacker = attacker_of(&state);
    state.set_seed(dice_seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    let (att_mix, def_mix) = if role_seeds {
        (ATTACKER_SEED_MIX, DEFENDER_SEED_MIX)
    } else {
        (CANDIDATE_SEED_MIX, OPPONENT_SEED_MIX)
    };
    // Game 2k: the candidate attacks. Game 2k+1: the opponent attacks.
    cand_att.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ att_mix));
    opp_def.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ def_mix));
    let (cand_def_mix, opp_att_mix) = if role_seeds {
        (DEFENDER_SEED_MIX, ATTACKER_SEED_MIX)
    } else {
        (CANDIDATE_SEED_MIX, OPPONENT_SEED_MIX)
    };
    cand_def.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ cand_def_mix));
    opp_att.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ opp_att_mix));

    let start = DriveStart::of(&state);
    let mut out = LockstepOutcome::default();
    let mut steps = 0u32;
    while !start.over(&state) && steps < max_steps {
        let Some(team) = state.available_actions.team else {
            break;
        };
        let attacking = team == attacker;
        let diverged = out.divergence.is_some();
        // Past the divergence game 2k is played: the candidate on attack, the opponent on defence.
        let (cand, opp) = if attacking {
            (cand_att.get_action(&state), opp_att.get_action(&state))
        } else {
            (cand_def.get_action(&state), opp_def.get_action(&state))
        };
        probe(&LockstepDecision {
            state: &state,
            step: steps,
            attacker_to_move: attacking,
            cand,
            opp,
            after_divergence: diverged,
        });
        if !diverged && cand != opp {
            out.divergence = Some(steps);
            out.prefix_steps = steps;
            if !follow {
                return out;
            }
        }
        let played = if out.divergence.is_none() || attacking {
            cand
        } else {
            opp
        };
        state.step(played).expect("engine step failed during a lockstep drive");
        for bot in [&mut *cand_att, &mut *cand_def, &mut *opp_att, &mut *opp_def] {
            botbowl_mcts::MctsBot::release_stale_tree_of(bot, &state);
        }
        steps += 1;
        if out.divergence.is_some() {
            out.followed_steps += 1;
        }
    }
    if out.divergence.is_none() {
        out.prefix_steps = steps;
        out.ended = start.over(&state);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::LadderRow;
    use botbowl_engine::bots::RandomBot;
    use botbowl_engine::scripted_bot::ScriptedBot;

    fn board() -> Option<BoardDims> {
        // The 14x7/4 tier, when the compiled capacity has room for it.
        BoardDims::try_new(16, 9, 4).ok()
    }

    #[test]
    fn assignment_pairs_each_position_across_both_sides() {
        assert_eq!(drive_assignment(3, 100, 0), (0, true, 100));
        assert_eq!(drive_assignment(3, 100, 1), (0, false, 100));
        assert_eq!(drive_assignment(3, 100, 2), (1, true, 101));
        assert_eq!(drive_assignment(3, 100, 5), (2, false, 102));
        // Past 2n games the positions come round again with new dice.
        assert_eq!(drive_assignment(3, 100, 6), (0, true, 103));
    }

    #[test]
    fn a_position_regenerates_identically_from_its_seed() {
        let Some(b) = board() else { return };
        let bias = RandomStartBias::default();
        for seed in [0, 1, 17, 12345] {
            let a = position_state(&bias, b, seed);
            let c = position_state(&bias, b, seed);
            assert!(a == c, "seed {seed} regenerated differently");
            assert_eq!(a.board_dims, b);
            assert!(turns_left(&a) >= 1 && turns_left(&a) <= TURNS_PER_HALF);
        }
    }

    /// A drive line folds like a game: the score fields are the drive's own touchdowns, the
    /// pair shares a position, and only the side changes between its two halves.
    #[test]
    fn a_mirrored_drive_pair_folds_into_one_sample() {
        let Some(b) = board() else { return };
        let bias = RandomStartBias::default();
        let mut row = LadderRow::new("scripted drives(t)@14x7/4");
        let (mut cand, mut opp) = (ScriptedBot::new(), RandomBot::new());
        for g in 0..8 {
            let (i, attacks, dice) = drive_assignment(4, 900, g);
            let seed = 40 + i as u64;
            let state = position_state(&bias, b, seed);
            let attacker = attacker_of(&state);
            let line = play_drive_game(&mut cand, &mut opp, "t", g, seed, state, attacks, dice, 20_000);
            assert_eq!(line.attacker, Some(attacker));
            assert_eq!(line.candidate_team == attacker, attacks);
            assert!(
                line.home_score + line.away_score <= 1,
                "a drive ends at its first touchdown"
            );
            assert_eq!(line.seed, seed);
            row.record(&line);
        }
        let row = row.finish();
        assert_eq!(row.games, 8);
        assert_eq!(row.pairs.pairs(), 4);
    }

    /// Plan 068: ungated, a drive is exactly today's line; gated, it carries the path, and a pair
    /// of the same deterministic bot plays one game twice: one path, exactly ½.
    #[test]
    fn a_gated_drive_fingerprints_its_moves_and_an_ungated_one_is_unchanged() {
        let Some(b) = board() else { return };
        let bias = RandomStartBias::default();
        for seed in [40u64, 41, 42] {
            let state = || position_state(&bias, b, seed);
            let plain = play_drive_game(
                &mut ScriptedBot::new(),
                &mut RandomBot::new(),
                "t",
                0,
                seed,
                state(),
                true,
                7,
                20_000,
            );
            let off = play_drive_game_gated(
                &mut ScriptedBot::new(),
                &mut RandomBot::new(),
                "t",
                0,
                seed,
                state(),
                true,
                7,
                20_000,
                false,
            );
            assert_eq!(plain, off);
            assert_eq!(off.path, None);
            let pair: Vec<EvalGameLine> = [true, false]
                .into_iter()
                .enumerate()
                .map(|(g, attacks)| {
                    play_drive_game_gated(
                        &mut ScriptedBot::new(),
                        &mut ScriptedBot::new(),
                        "t",
                        g as u32,
                        seed,
                        state(),
                        attacks,
                        7,
                        20_000,
                        true,
                    )
                })
                .collect();
            assert!(pair[0].path.is_some());
            assert_eq!(pair[0].path, pair[1].path, "seed {seed}: one game, played twice");
            assert_eq!(
                (pair[0].home_score, pair[0].away_score),
                (pair[1].home_score, pair[1].away_score)
            );
            let mut row = LadderRow::new("t").with_gate(Some(crate::eval::GateMode::Report));
            row.record(&pair[0]);
            row.record(&pair[1]);
            let gate = row.finish().gate.unwrap();
            assert_eq!((gate.identical, gate.mismatched), (1, 0));
        }
    }

    #[test]
    fn move_paths_tell_orders_and_kinds_apart() {
        use botbowl_engine::core::model::Position;
        use botbowl_engine::core::table::{PosAT, SimpleAT};
        let a = EngineAction::Positional(PosAT::Move, Position::new((3, 4)));
        let c = EngineAction::Simple(SimpleAT::EndTurn);
        let path = |acts: &[EngineAction]| {
            let mut p = MovePath::default();
            acts.iter().for_each(|&x| p.push(x));
            p
        };
        assert_eq!(path(&[a, c]), path(&[a, c]));
        assert_ne!(path(&[a, c]), path(&[c, a]));
        assert_ne!(path(&[a]), path(&[a, a]));
        assert_ne!(path(&[]), path(&[c]));
    }

    /// Plan 068: lockstep with the same deterministic bot never diverges and plays the drive to its
    /// end; against a different bot it stops at the first disagreement, and every decision before
    /// it was an agreement.
    #[test]
    fn lockstep_finds_the_first_disagreement() {
        let Some(b) = board() else { return };
        let bias = RandomStartBias::default();
        let state = position_state(&bias, b, 41);
        let (mut s1, mut s2, mut s3, mut s4) = (
            ScriptedBot::new(),
            ScriptedBot::new(),
            ScriptedBot::new(),
            ScriptedBot::new(),
        );
        let mut seen = 0;
        let same = lockstep_drive(
            LockstepBots {
                cand_att: &mut s1,
                cand_def: &mut s2,
                opp_att: &mut s3,
                opp_def: &mut s4,
            },
            state.clone(),
            7,
            20_000,
            true,
            false,
            |d| {
                assert_eq!(d.cand, d.opp);
                seen += 1;
            },
        );
        assert_eq!(same.divergence, None);
        assert!(same.ended && same.prefix_steps == seen && seen > 0);
        let (mut c1, mut c2) = (ScriptedBot::new(), ScriptedBot::new());
        let (mut r1, mut r2) = (RandomBot::new(), RandomBot::new());
        let mut log = Vec::new();
        let diff = lockstep_drive(
            LockstepBots {
                cand_att: &mut c1,
                cand_def: &mut c2,
                opp_att: &mut r1,
                opp_def: &mut r2,
            },
            state,
            7,
            20_000,
            true,
            false,
            |d| log.push(d.cand == d.opp),
        );
        let at = diff.divergence.expect("scripted and random part ways");
        assert_eq!(log.len() as u32, at + 1);
        assert!(log[..at as usize].iter().all(|&x| x) && !log[at as usize]);
    }

    #[test]
    fn a_set_plays_its_screened_positions_when_it_has_them() {
        let mut set = PositionSet {
            name: "t".into(),
            commit: "abc".into(),
            board: "14x7/4".into(),
            bias: RandomStartBias::default(),
            seeds: vec![1, 2, 3, 4],
            screen: None,
        };
        assert_eq!(set.positions(), vec![1, 2, 3, 4]);
        set.screen = Some(Screen {
            reference: "ref.onnx".into(),
            playouts: 4,
            band: (0.25, 0.75),
            kept: vec![
                KeptPosition {
                    seed: 2,
                    attacker_td_rate: 0.5,
                },
                KeptPosition {
                    seed: 4,
                    attacker_td_rate: 0.25,
                },
            ],
        });
        assert_eq!(set.positions(), vec![2, 4]);
        let json = serde_json::to_string(&set).unwrap();
        assert_eq!(serde_json::from_str::<PositionSet>(&json).unwrap(), set);
        set.screen.as_mut().unwrap().kept.clear();
        assert!(set.rung().is_err(), "an empty set is refused rather than played");
    }
}
