//! One opponent-ladder game and the records it produces (plan 020).
//!
//! A rung is N full games from kickoff against a fixed opponent,
//! alternating Home/Away on a fixed seed set so candidates are compared on
//! identical situations. Each game yields one [`EvalGameLine`]; a rung's
//! lines fold into one [`LadderRow`] via [`LadderRow::record`]. Keeping the
//! fold here, next to the per-game record, is what lets the plan-041 hub
//! rebuild `report.json` from lines that arrived from many workers.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::TeamType;

const OPPONENT_SEED_MIX: u64 = 0xC3C3_C3C3_C3C3_C3C3;
const CANDIDATE_SEED_MIX: u64 = 0x3C3C_3C3C_3C3C_3C3C;

/// Per-game side-relative record (plan 023 deferred item 5): the pooled
/// rung row cannot distinguish a scoring-rate bias from a win-conversion
/// one, nor see who received the opening kickoff. One JSON line per game
/// in `--per-game-out`; field order is the file format, so don't reorder.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct EvalGameLine {
    pub rung: String,
    pub game: u32,
    pub seed: u64,
    pub candidate_team: TeamType,
    pub home_score: u8,
    pub away_score: u8,
    pub kicking_first_half: TeamType,
    pub finished: bool,
}

impl EvalGameLine {
    /// `(candidate_score, opponent_score)`.
    pub fn candidate_scores(&self) -> (u8, u8) {
        match self.candidate_team {
            TeamType::Home => (self.home_score, self.away_score),
            TeamType::Away => (self.away_score, self.home_score),
        }
    }
}

/// Which side the candidate plays in game `g` of a rung, and the game's
/// seed. Sides alternate and the seed is shared by the mirrored pair
/// `g±1`, so every candidate faces the same situations from both sides.
/// Pure in `g`, so which worker picks up which game cannot change the
/// pairing.
pub fn ladder_assignment(base_seed: u64, g: u32) -> (TeamType, u64) {
    let team = if g % 2 == 0 { TeamType::Home } else { TeamType::Away };
    (team, base_seed.wrapping_add((g / 2) as u64))
}

/// One full game from kickoff between `candidate` (playing
/// `candidate_team`) and `opponent`.
pub fn play_ladder_game(
    candidate: &mut dyn Bot,
    opponent: &mut dyn Bot,
    rung: &str,
    game: u32,
    candidate_team: TeamType,
    seed: u64,
    max_steps: u32,
) -> EvalGameLine {
    let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    candidate.set_seed(ChaCha8Rng::seed_from_u64(seed ^ CANDIDATE_SEED_MIX));
    opponent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ OPPONENT_SEED_MIX));

    let mut steps = 0u32;
    while !state.info.game_over && steps < max_steps {
        let action = match state.available_actions.team {
            Some(t) if t == candidate_team => candidate.get_action(&state),
            Some(_) => opponent.get_action(&state),
            None => break,
        };
        state.step(action).expect("engine step failed during eval game");
        steps += 1;
    }

    line_of(&state, rung, game, candidate_team, seed)
}

fn line_of(state: &GameState, rung: &str, game: u32, candidate_team: TeamType, seed: u64) -> EvalGameLine {
    EvalGameLine {
        rung: rung.to_string(),
        game,
        seed,
        candidate_team,
        home_score: state.home.score,
        away_score: state.away.score,
        kicking_first_half: state.info.kicking_first_half,
        finished: state.info.game_over,
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LectureRow {
    pub lecture: String,
    pub difficulty: String,
    pub trials: u32,
    pub successes: u32,
    pub failures: u32,
    pub timeouts: u32,
    pub success_rate: f64,
    /// The lecture's hard-coded full-pitch coordinates don't fit the
    /// compiled board — cell skipped (see plan 020 next-next steps:
    /// board-relative lecture setups).
    pub skipped_board_too_small: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct LadderRow {
    pub opponent: String,
    pub games: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    pub tds_for: u32,
    pub tds_against: u32,
    pub unfinished: u32,
    pub win_rate: f64,
    /// Per-side split (games alternate Home/Away): a Home/Away asymmetry
    /// cancels out of `win_rate` but shows up here (plan 021 open issue 5,
    /// the 0.40 mirror anomaly).
    pub wins_as_home: u32,
    pub losses_as_home: u32,
    pub wins_as_away: u32,
    pub losses_as_away: u32,
    /// Side-relative TD totals (not candidate-relative): closes the
    /// instrument gap noted in plan 023 — `tds_for/against` are pooled over
    /// both sides and so are balanced by construction in a mirror.
    pub tds_by_home: u32,
    pub tds_by_away: u32,
}

impl LadderRow {
    pub fn new(opponent: &str) -> Self {
        LadderRow {
            opponent: opponent.to_string(),
            ..Default::default()
        }
    }

    /// Fold one game into the counters. Every field is commutative, so
    /// lines can arrive in any order from any number of workers.
    /// `win_rate` is not maintained here; call [`LadderRow::finish`].
    pub fn record(&mut self, line: &EvalGameLine) {
        let (cand, opp) = line.candidate_scores();
        self.games += 1;
        self.tds_for += cand as u32;
        self.tds_against += opp as u32;
        self.tds_by_home += line.home_score as u32;
        self.tds_by_away += line.away_score as u32;
        if !line.finished {
            self.unfinished += 1;
        }
        let home = line.candidate_team == TeamType::Home;
        match cand.cmp(&opp) {
            std::cmp::Ordering::Greater => {
                self.wins += 1;
                if home {
                    self.wins_as_home += 1
                } else {
                    self.wins_as_away += 1
                }
            }
            std::cmp::Ordering::Equal => self.draws += 1,
            std::cmp::Ordering::Less => {
                self.losses += 1;
                if home {
                    self.losses_as_home += 1
                } else {
                    self.losses_as_away += 1
                }
            }
        }
    }

    /// Derive `win_rate` once all games are in.
    pub fn finish(mut self) -> Self {
        self.win_rate = if self.games > 0 {
            self.wins as f64 / self.games as f64
        } else {
            0.0
        };
        self
    }
}

/// The report card `botbowl-ui eval --out` writes.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Report {
    pub candidate: String,
    pub mcts_iters: usize,
    pub seed: u64,
    pub board_env: String,
    pub git_commit: String,
    pub git_dirty: bool,
    pub lectures: Vec<LectureRow>,
    pub ladder: Vec<LadderRow>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(g: u32, team: TeamType, h: u8, a: u8, finished: bool) -> EvalGameLine {
        EvalGameLine {
            rung: "scripted".into(),
            game: g,
            seed: 7,
            candidate_team: team,
            home_score: h,
            away_score: a,
            kicking_first_half: TeamType::Away,
            finished,
        }
    }

    /// The per-game line is a file format read by `scripts/paired_summary.py`
    /// and friends; pin the exact bytes the hand-written `writeln!` used to
    /// produce.
    #[test]
    fn eval_game_line_serializes_like_the_old_writeln() {
        let l = line(3, TeamType::Away, 1, 2, true);
        assert_eq!(
            serde_json::to_string(&l).unwrap(),
            r#"{"rung":"scripted","game":3,"seed":7,"candidate_team":"Away","home_score":1,"away_score":2,"kicking_first_half":"Away","finished":true}"#
        );
        assert_eq!(
            serde_json::from_str::<EvalGameLine>(&serde_json::to_string(&l).unwrap()).unwrap(),
            l
        );
    }

    #[test]
    fn ladder_row_folds_sides_and_results() {
        let mut row = LadderRow::new("scripted");
        row.record(&line(0, TeamType::Home, 2, 0, true)); // win as home
        row.record(&line(1, TeamType::Away, 2, 0, true)); // loss as away
        row.record(&line(2, TeamType::Home, 1, 1, false)); // draw, unfinished
        row.record(&line(3, TeamType::Away, 0, 1, true)); // win as away
        let row = row.finish();
        assert_eq!((row.games, row.wins, row.draws, row.losses), (4, 2, 1, 1));
        assert_eq!(
            (
                row.wins_as_home,
                row.losses_as_home,
                row.wins_as_away,
                row.losses_as_away
            ),
            (1, 0, 1, 1)
        );
        assert_eq!((row.tds_for, row.tds_against), (4, 3));
        assert_eq!((row.tds_by_home, row.tds_by_away), (5, 2));
        assert_eq!(row.unfinished, 1);
        assert_eq!(row.win_rate, 0.5);
    }

    #[test]
    fn ladder_assignment_mirrors_pairs() {
        assert_eq!(ladder_assignment(100, 0), (TeamType::Home, 100));
        assert_eq!(ladder_assignment(100, 1), (TeamType::Away, 100));
        assert_eq!(ladder_assignment(100, 2), (TeamType::Home, 101));
        assert_eq!(ladder_assignment(100, 5), (TeamType::Away, 102));
    }
}
