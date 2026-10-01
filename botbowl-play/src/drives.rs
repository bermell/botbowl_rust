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
use botbowl_engine::core::model::{other_team, BoardDims, TeamType};

use crate::board_sizes::{board_label, parse_board};
use crate::eval::{EvalGameLine, CANDIDATE_SEED_MIX, OPPONENT_SEED_MIX};
use crate::generate::RandomStartBias;

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
    mut state: GameState,
    candidate_attacks: bool,
    dice_seed: u64,
    max_steps: u32,
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
    candidate.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ CANDIDATE_SEED_MIX));
    opponent.set_seed(ChaCha8Rng::seed_from_u64(dice_seed ^ OPPONENT_SEED_MIX));

    let (start_home, start_away, start_half) = (state.home.score, state.away.score, state.info.half);
    let mut steps = 0u32;
    while !state.info.game_over
        && steps < max_steps
        && state.home.score == start_home
        && state.away.score == start_away
        && state.info.half == start_half
    {
        let action = match state.available_actions.team {
            Some(t) if t == candidate_team => candidate.get_action(&state),
            Some(_) => opponent.get_action(&state),
            None => break,
        };
        state.step(action).expect("engine step failed during a drive");
        botbowl_mcts::MctsBot::release_stale_tree_of(&mut *candidate, &state);
        botbowl_mcts::MctsBot::release_stale_tree_of(&mut *opponent, &state);
        steps += 1;
    }
    let telemetry = botbowl_mcts::MctsBot::take_telemetry_of(candidate);
    let ended = state.info.game_over
        || state.home.score != start_home
        || state.away.score != start_away
        || state.info.half != start_half;
    EvalGameLine {
        rung: rung.to_string(),
        game,
        seed: position_seed,
        candidate_team,
        home_score: state.home.score - start_home,
        away_score: state.away.score - start_away,
        kicking_first_half: state.info.kicking_first_half,
        finished: ended,
        board: Some(board_label(state.board_dims)),
        telemetry,
        attacker: Some(attacker),
    }
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
