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
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{BoardDims, TeamType};
use botbowl_mcts::SearchTelemetry;

use crate::board_sizes::board_label;
use crate::trace::{ReuseTraceRow, ReuseTraceWriter};

const OPPONENT_SEED_MIX: u64 = 0xC3C3_C3C3_C3C3_C3C3;
const CANDIDATE_SEED_MIX: u64 = 0x3C3C_3C3C_3C3C_3C3C;

/// Per-game side-relative record (plan 023 deferred item 5): the pooled
/// rung row cannot distinguish a scoring-rate bias from a win-conversion
/// one, nor see who received the opening kickoff. One JSON line per game
/// in `--per-game-out`; field order is the file format, so don't reorder.
///
/// `Serialize` is hand-written (below) so that `board` is **omitted from
/// JSON when absent** — keeping every env-board line byte-identical to the
/// pre-042 format — but **always present on the binary wire** (`postcard`
/// is not self-describing, so a skipped field there is a decode error on
/// the hub). `Deserialize` is derived with `#[serde(default)]`, which reads
/// both.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct EvalGameLine {
    pub rung: String,
    pub game: u32,
    pub seed: u64,
    pub candidate_team: TeamType,
    pub home_score: u8,
    pub away_score: u8,
    pub kicking_first_half: TeamType,
    pub finished: bool,
    /// Plan 042: the playable board (`14x7/4`) when the rung named one
    /// explicitly. Absent for env-board games, so those lines are
    /// byte-identical to the pre-042 format.
    #[serde(default)]
    pub board: Option<String>,
    /// Plan 043: the candidate's search health over this game — how often it kept its tree, and
    /// what recombination cost. Absent when the candidate is not an MCTS bot (the random and
    /// scripted rungs have no search to report on). Same omit-from-JSON-when-absent rule as
    /// `board`, and for the same reason.
    ///
    /// This is the bot's own type rather than a flattened copy, so the fold into
    /// [`LadderRow::record`] is `SearchTelemetry::merge` — one implementation, used identically by
    /// `botbowl-ui eval` and by the hub rebuilding a report from workers' lines.
    #[serde(default)]
    pub telemetry: Option<SearchTelemetry>,
}

impl Serialize for EvalGameLine {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // JSON (human-readable): the trailing key only when set. Binary
        // (postcard, not self-describing): always, or the decoder starves.
        let binary = !serializer.is_human_readable();
        let with_board = self.board.is_some() || binary;
        let with_telemetry = self.telemetry.is_some() || binary;
        let n = 8 + usize::from(with_board) + usize::from(with_telemetry);
        let mut s = serializer.serialize_struct("EvalGameLine", n)?;
        s.serialize_field("rung", &self.rung)?;
        s.serialize_field("game", &self.game)?;
        s.serialize_field("seed", &self.seed)?;
        s.serialize_field("candidate_team", &self.candidate_team)?;
        s.serialize_field("home_score", &self.home_score)?;
        s.serialize_field("away_score", &self.away_score)?;
        s.serialize_field("kicking_first_half", &self.kicking_first_half)?;
        s.serialize_field("finished", &self.finished)?;
        if with_board {
            s.serialize_field("board", &self.board)?;
        }
        if with_telemetry {
            s.serialize_field("telemetry", &self.telemetry)?;
        }
        s.end()
    }
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
/// `candidate_team`) and `opponent`, on `board` (`None` = the env board).
#[allow(clippy::too_many_arguments)]
pub fn play_ladder_game(
    candidate: &mut dyn Bot,
    opponent: &mut dyn Bot,
    rung: &str,
    game: u32,
    candidate_team: TeamType,
    seed: u64,
    max_steps: u32,
    board: Option<BoardDims>,
    // Plan 043: `--trace-reuse`. `None` in every normal run, including every distributed one —
    // the aggregate in `report.json` is what a run reports, and this is for chasing an aggregate
    // that looks wrong.
    trace: Option<&ReuseTraceWriter>,
) -> EvalGameLine {
    let mut builder = GameStateBuilder::new();
    builder.set_state(BuilderState::CoinToss);
    if let Some(dims) = board {
        builder.with_board_dims(dims);
    }
    let mut state = builder.build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    candidate.set_seed(ChaCha8Rng::seed_from_u64(seed ^ CANDIDATE_SEED_MIX));
    opponent.set_seed(ChaCha8Rng::seed_from_u64(seed ^ OPPONENT_SEED_MIX));

    let mut steps = 0u32;
    let mut decision = 0u32;
    while !state.info.game_over && steps < max_steps {
        let action = match state.available_actions.team {
            Some(t) if t == candidate_team => {
                let a = candidate.get_action(&state);
                if let Some(w) = trace {
                    // Read the decision *before* stepping, so the action list is the fan the
                    // search actually faced.
                    if let Some(summary) = botbowl_mcts::MctsBot::last_search_of(&*candidate) {
                        w.write(&ReuseTraceRow::new(game, decision, &summary.reuse, &state));
                    }
                    decision += 1;
                }
                a
            }
            Some(_) => opponent.get_action(&state),
            None => break,
        };
        state.step(action).expect("engine step failed during eval game");
        steps += 1;
    }

    // Plan 043: drain rather than read. A rung reuses one bot across its games, so taking the
    // counters here is what makes this line's telemetry mean *this game*. `None` for a bot that
    // does not search — the random and scripted rungs.
    let telemetry = botbowl_mcts::MctsBot::take_telemetry_of(candidate);

    line_of(&state, rung, game, candidate_team, seed, board, telemetry)
}

fn line_of(
    state: &GameState,
    rung: &str,
    game: u32,
    candidate_team: TeamType,
    seed: u64,
    board: Option<BoardDims>,
    telemetry: Option<SearchTelemetry>,
) -> EvalGameLine {
    EvalGameLine {
        rung: rung.to_string(),
        game,
        seed,
        candidate_team,
        home_score: state.home.score,
        away_score: state.away.score,
        kicking_first_half: state.info.kicking_first_half,
        finished: state.info.game_over,
        board: board.map(board_label),
        telemetry,
    }
}

/// The rung label a multi-size ladder uses for `opponent` on `board`:
/// `scripted@14x7/4`. Single-board ladders keep the bare opponent name so
/// every downstream script keeps matching.
pub fn rung_name(opponent: &str, board: Option<BoardDims>) -> String {
    match board {
        Some(d) => format!("{opponent}@{}", board_label(d)),
        None => opponent.to_string(),
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
    /// Plan 042: the playable board this rung ran on, when the ladder named
    /// one (`14x7/4`); absent on an env-board ladder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<String>,
    /// Plan 043: the candidate's search health summed over this rung's games. Absent when the
    /// candidate does not search. Folded by [`LadderRow::record`] like every other counter here,
    /// so the hub rebuilding a report from workers' lines gets the identical number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<SearchTelemetry>,
}

impl LadderRow {
    pub fn new(opponent: &str) -> Self {
        LadderRow {
            opponent: opponent.to_string(),
            ..Default::default()
        }
    }

    /// A rung on an explicit board: named `opponent@board`, `board` set.
    pub fn on_board(opponent: &str, board: Option<BoardDims>) -> Self {
        LadderRow {
            opponent: rung_name(opponent, board),
            board: board.map(board_label),
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
        // Plan 043: every field of `SearchTelemetry` is a commutative counter, so this obeys the
        // same "any order, any number of producers" rule as the counters above.
        if let Some(t) = &line.telemetry {
            self.telemetry.get_or_insert_with(SearchTelemetry::default).merge(t);
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
    /// Plan 043: the named preset each side played under, if any. `None` means the historical
    /// per-flag configuration. Serde-defaulted so an older `report.json` still parses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_config: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opponent_config: Option<String>,
    /// Plan 043: the candidate's search health over the whole ladder — the sum of every rung's
    /// [`LadderRow::telemetry`]. Build it with [`Report::telemetry_of`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telemetry: Option<SearchTelemetry>,
}

impl Report {
    /// Sum the ladder's per-rung telemetry. `None` when no rung had a searching candidate.
    pub fn telemetry_of(ladder: &[LadderRow]) -> Option<SearchTelemetry> {
        ladder.iter().filter_map(|r| r.telemetry.as_ref()).fold(None, |acc, t| {
            let mut acc = acc.unwrap_or_default();
            acc.merge(t);
            Some(acc)
        })
    }
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
            board: None,
            telemetry: None,
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

    /// A board-tagged line adds one trailing key and nothing else, so the
    /// old readers keep working and the tag is where a grouper looks for it.
    #[test]
    fn board_tag_is_a_trailing_optional_key() {
        let mut l = line(3, TeamType::Away, 1, 2, true);
        l.board = Some("14x7/4".into());
        let s = serde_json::to_string(&l).unwrap();
        assert!(s.ends_with(r#","finished":true,"board":"14x7/4"}"#), "{s}");
        assert_eq!(serde_json::from_str::<EvalGameLine>(&s).unwrap(), l);
        let row = LadderRow::on_board("scripted", None);
        assert_eq!((row.opponent.as_str(), row.board.as_deref()), ("scripted", None));
        assert_eq!(rung_name("scripted", None), "scripted");
    }

    /// The binary wire must round-trip both an absent and a present board:
    /// `postcard` cannot skip a field, so the JSON-only omission above must
    /// not leak into it (it did once — the hub read every worker frame as
    /// "end of buffer").
    #[test]
    fn board_tag_survives_a_non_self_describing_encoding() {
        for board in [None, Some("12x5/2".to_string())] {
            let mut l = line(1, TeamType::Home, 0, 0, true);
            l.board = board;
            let bytes = postcard::to_allocvec(&l).unwrap();
            let back: EvalGameLine = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back, l);
        }
    }

    /// A sample telemetry blob, shaped like a real one: a couple of procedures and a fan.
    fn sample_telemetry() -> SearchTelemetry {
        let mut t = SearchTelemetry::default();
        for (outcome, proc, fan) in [
            (botbowl_mcts::ReuseOutcome::NoCache, "Turn", 4),
            (botbowl_mcts::ReuseOutcome::Reused, "MoveAction", 21),
            (botbowl_mcts::ReuseOutcome::AnchorMiss, "Turn", 6),
        ] {
            t.record(
                &botbowl_mcts::ReuseDecision {
                    outcome,
                    proc: Some(proc.to_string()),
                    n_actions: fan,
                    path_len: 0,
                },
                botbowl_mcts::RecombinationCounts {
                    hits: 3,
                    misses: 40,
                    probes: 43,
                    eq_checks: 300,
                    eq_hash_equal: 290,
                    eq_rejects: 297,
                    lookup_probes: 1,
                    lookup_hits: 1,
                },
            );
        }
        t
    }

    /// Plan 043's trailing field plays by the same rules as `board`: invisible in JSON when
    /// absent, so every existing line and every downstream script is untouched.
    #[test]
    fn telemetry_is_a_trailing_optional_key() {
        let mut l = line(3, TeamType::Away, 1, 2, true);
        assert!(
            !serde_json::to_string(&l).unwrap().contains("telemetry"),
            "a non-searching candidate must not add a key"
        );

        l.telemetry = Some(sample_telemetry());
        let json = serde_json::to_string(&l).unwrap();
        assert!(json.contains(r#""telemetry":{"#), "got {json}");
        assert_eq!(serde_json::from_str::<EvalGameLine>(&json).unwrap(), l);
    }

    /// The postcard trap, again. `board`'s comment explains it: a field skipped on a
    /// non-self-describing wire makes the hub read every worker frame as end-of-buffer.
    #[test]
    fn telemetry_survives_a_non_self_describing_encoding() {
        for telemetry in [None, Some(sample_telemetry())] {
            let mut l = line(1, TeamType::Home, 0, 0, true);
            l.telemetry = telemetry;
            let bytes = postcard::to_allocvec(&l).unwrap();
            let back: EvalGameLine = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back, l);
        }
    }

    /// The fold is what the hub relies on: a rung's telemetry is the sum of its games',
    /// regardless of which worker sent which line or in what order.
    #[test]
    fn ladder_row_sums_telemetry_over_its_games() {
        let with = |g: u32| {
            let mut l = line(g, TeamType::Home, 1, 0, true);
            l.telemetry = Some(sample_telemetry());
            l
        };

        let mut forward = LadderRow::new("mcts");
        forward.record(&with(0));
        forward.record(&with(1));

        let mut backward = LadderRow::new("mcts");
        backward.record(&with(1));
        backward.record(&with(0));

        assert_eq!(forward.telemetry, backward.telemetry, "the fold is order-independent");
        let t = forward.telemetry.expect("two searching games");
        assert_eq!(t.searches, 6, "three decisions per game, two games");
        assert_eq!(t.reuse.total.reused, 2);
        assert_eq!(t.reuse.by_proc["Turn"].attempts(), 4, "two Turn decisions per game");
        assert_eq!(t.recombination.hits, 18);

        // A rung whose candidate does not search stays absent rather than reading as all-zero.
        let mut plain = LadderRow::new("random");
        plain.record(&line(0, TeamType::Home, 1, 0, true));
        assert_eq!(plain.telemetry, None);
    }

    #[test]
    fn ladder_assignment_mirrors_pairs() {
        assert_eq!(ladder_assignment(100, 0), (TeamType::Home, 100));
        assert_eq!(ladder_assignment(100, 1), (TeamType::Away, 100));
        assert_eq!(ladder_assignment(100, 2), (TeamType::Home, 101));
        assert_eq!(ladder_assignment(100, 5), (TeamType::Away, 102));
    }
}
