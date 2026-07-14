//! Training-data schema + persistence for Blood Bowl MCTS self-play.
//!
//! The grand plan (steps 6–7: imitation learning, then self-play) needs a
//! way to persist, per decision the agent makes:
//!
//! - the **game state** it acted from (a decision node),
//! - the **search distribution** over actions (raw per-child visit counts,
//!   Q values, priors and *solvedness* — not a pre-normalised policy), and
//! - **values** (the search's root value *and* the backfilled drive/game
//!   outcome), so training can pick a bootstrap or Monte-Carlo target.
//!
//! Plus enough **provenance** to make a dataset reproducible against a
//! moving engine: the git commit the generating binary was built at, the
//! board capacity/dimensions, the bots, the seed.
//!
//! ## Why raw stats, not a normalised policy target
//!
//! Since `recon_mcts` gained solved-subtree pruning, a child's visit count
//! is **not** a valid posterior. A solved child (often the *best* move — a
//! touchdown solves fast) leaves the selectable set, so its visits freeze
//! while unsolved siblings keep accruing. Training code must therefore see
//! the raw `{visits, q, prior, solved}` per child and construct the policy
//! target itself (see plan 017's caveat). We deliberately store the raw
//! search output and defer target construction to training time.
//!
//! ## On-disk format
//!
//! [JSON Lines](https://jsonlines.org): one [`Trajectory`] per line. Append-
//! able, greppable, and a natural streaming unit for a shuffling data
//! loader. `GameState` JSON is verbose (full board arrays); a compact
//! binary encoding can be swapped in behind [`DatasetWriter`] later without
//! touching the schema.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action, BoardDims, TeamType, HEIGHT, TEAM_SIZE, WIDTH};

/// Schema version. Bump on any breaking change to the types below so a
/// reader can reject / migrate old files.
pub const FORMAT_VERSION: u32 = 1;

/// The git commit the *currently running* binary was built at (full SHA,
/// or `"unknown"` if git was unavailable at build time). Baked in by
/// `build.rs`. This is the commit that should be recorded on every
/// trajectory this binary produces.
pub fn git_commit() -> &'static str {
    env!("BOTBOWL_GIT_COMMIT")
}

/// Whether the working tree had uncommitted changes when this binary was
/// built. A `true` here means the recorded [`git_commit`] does not fully
/// describe the generating code — treat such datasets with suspicion.
pub fn git_dirty() -> bool {
    env!("BOTBOWL_GIT_DIRTY") == "true"
}

/// Which team acts at a node. Mirrors [`TeamType`] but lives in this crate
/// so the on-disk schema is independent of engine-internal derives.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Team {
    Home,
    Away,
}

impl From<TeamType> for Team {
    fn from(t: TeamType) -> Self {
        match t {
            TeamType::Home => Team::Home,
            TeamType::Away => Team::Away,
        }
    }
}

impl From<Team> for TeamType {
    fn from(t: Team) -> Self {
        match t {
            Team::Home => TeamType::Home,
            Team::Away => TeamType::Away,
        }
    }
}

/// Raw search statistics for one child of a decision node.
///
/// All fields are the un-normalised search output. Build a policy target
/// from these at training time — do **not** assume `visits` alone is a
/// posterior (see the module docs and `solved`).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChildStat {
    /// The engine action leading to this child.
    pub action: Action,
    /// N: descents through this child (cumulative across any reused tree
    /// within the turn). Frozen once `solved` is `true`.
    pub visits: u32,
    /// Aggregated value of the child, **Home-centric** (Home maximises,
    /// Away minimises, chance is an expectation). `None` if the child was
    /// never scored (e.g. an unexpanded chance leaf).
    pub q: Option<i64>,
    /// Domain-knowledge prior weight used by PUCT — *relative* and
    /// un-normalised. `None` for chance edges.
    pub prior: Option<f32>,
    /// The child's subtree is fully solved (exact minimax within the
    /// horizon); its `visits` are frozen. Critical for policy targets.
    pub solved: bool,
    /// The child is itself a terminal leaf (game/horizon end).
    pub terminal: bool,
}

/// One decision made by the agent, plus the search behind it. This is the
/// atomic training example.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sample {
    /// The decision node the agent acted from. Serialises fully except the
    /// RNG (reseeded on load — irrelevant for training).
    pub state: GameState,
    /// The team to move at this node.
    pub to_move: Team,
    /// The action the agent actually played (by best aggregated Q, not
    /// most-visited — see `MctsBot`).
    pub chosen_action: Action,
    /// Root children after the search, with raw per-child stats.
    pub children: Vec<ChildStat>,
    /// The search's aggregated **root** value, Home-centric. `None` if the
    /// root was never scored.
    pub root_value: Option<i64>,
    /// Total descents through the root this search.
    pub root_visits: u32,
    /// The whole root subtree was solved (a proven position — its policy
    /// target should be a sharp/one-hot over child Q, not a visit softmax).
    pub root_solved: bool,
    /// Ground-truth value target, **backfilled at trajectory end**: the
    /// eventual outcome from Home's perspective in `[-1, 1]` (see
    /// [`Outcome::z_home`]). `None` before backfilling.
    #[serde(default)]
    pub outcome_value: Option<f32>,
}

/// How a trajectory ended — the value-target ground truth.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Outcome {
    pub home_score: u8,
    pub away_score: u8,
    pub winner: Option<Team>,
    pub game_over: bool,
    /// Outcome from Home's perspective in `[-1, 1]`: `+1` Home ahead at the
    /// end of the trajectory, `-1` Away ahead, `0` level. This is the
    /// AlphaZero-style `z` target broadcast to every sample.
    pub z_home: f32,
    /// If the trajectory came from a curriculum lecture, its terminal
    /// status (`"Success"` / `"Failure"` / `"InProgress"`); else `None`.
    pub lecture_status: Option<String>,
}

impl Outcome {
    /// Build an outcome from a finished (or cut-off) game state.
    pub fn from_state(state: &GameState, lecture_status: Option<String>) -> Self {
        let home = state.home.score;
        let away = state.away.score;
        let z_home = (home as f32 - away as f32).clamp(-1.0, 1.0);
        Outcome {
            home_score: home,
            away_score: away,
            winner: state.info.winner.map(Team::from),
            game_over: state.info.game_over,
            z_home,
            lecture_status,
        }
    }
}

/// Build-time board capacity (the maximum board this binary can run — the
/// stack-allocated array size). Distinct from the runtime [`BoardDims`]
/// actually in play.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardCapacity {
    /// Engine width incl. the 2-cell OOB border (`WIDTH`).
    pub width: usize,
    /// Engine height incl. the 2-cell OOB border (`HEIGHT`).
    pub height: usize,
    /// Max players fielded per team (`TEAM_SIZE`).
    pub team_size: usize,
}

impl BoardCapacity {
    /// The capacity compiled into the current binary.
    pub fn current() -> Self {
        BoardCapacity {
            width: WIDTH,
            height: HEIGHT,
            team_size: TEAM_SIZE,
        }
    }
}

/// Provenance for a batch of samples — everything needed to reproduce or
/// filter a dataset against a moving engine.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrajectoryMeta {
    pub format_version: u32,
    /// Git commit the generating binary was built at (full SHA / `"unknown"`).
    pub git_commit: String,
    /// Working tree was dirty at build time — `git_commit` is incomplete.
    pub git_dirty: bool,
    /// Max board the generating binary supports.
    pub board_capacity: BoardCapacity,
    /// Logical board actually in play for this trajectory.
    pub board_dims: BoardDims,
    /// Where the data came from: `"self-play"`, a lecture name, etc.
    pub source: String,
    /// Human-readable descriptor of the Home bot (e.g. `"mcts(time=150ms)"`).
    pub home_bot: String,
    /// Human-readable descriptor of the Away bot.
    pub away_bot: String,
    /// The seed the trajectory was generated with, if any.
    pub seed: Option<u64>,
    /// Wall-clock creation time (unix seconds), best-effort.
    pub created_unix_secs: Option<u64>,
    /// Free-form extras: search budget, difficulty, notes, ... Kept out of
    /// the typed fields so adding one never breaks the schema.
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

impl TrajectoryMeta {
    /// Start a metadata record, stamping the current binary's git commit,
    /// board capacity, and wall-clock time. Fill in the rest with the
    /// builder-style setters.
    pub fn new(source: impl Into<String>, board_dims: BoardDims) -> Self {
        TrajectoryMeta {
            format_version: FORMAT_VERSION,
            git_commit: git_commit().to_string(),
            git_dirty: git_dirty(),
            board_capacity: BoardCapacity::current(),
            board_dims,
            source: source.into(),
            home_bot: String::new(),
            away_bot: String::new(),
            seed: None,
            created_unix_secs: SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs()),
            extra: BTreeMap::new(),
        }
    }

    pub fn with_bots(mut self, home: impl Into<String>, away: impl Into<String>) -> Self {
        self.home_bot = home.into();
        self.away_bot = away.into();
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// A sequence of decisions from one game (or lecture trial) sharing one
/// provenance record and one outcome. The unit of a JSONL line.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Trajectory {
    pub meta: TrajectoryMeta,
    pub samples: Vec<Sample>,
    pub outcome: Outcome,
}

impl Trajectory {
    pub fn new(meta: TrajectoryMeta, samples: Vec<Sample>, outcome: Outcome) -> Self {
        let mut t = Trajectory { meta, samples, outcome };
        t.backfill_outcome_value();
        t
    }

    /// Broadcast the trajectory outcome (`z_home`) onto every sample's
    /// `outcome_value`. Called by [`Trajectory::new`]; exposed for callers
    /// that mutate samples after construction.
    pub fn backfill_outcome_value(&mut self) {
        let z = self.outcome.z_home;
        for s in &mut self.samples {
            s.outcome_value = Some(z);
        }
    }
}

/// Appends [`Trajectory`] records as JSON Lines. Buffered; flush (or drop)
/// to ensure data hits disk.
pub struct DatasetWriter {
    inner: BufWriter<File>,
}

impl DatasetWriter {
    /// Open `path` for appending, creating it if absent. Existing content
    /// is preserved — new trajectories are added at the end.
    pub fn append(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(DatasetWriter {
            inner: BufWriter::new(file),
        })
    }

    /// Create (or truncate) `path` for writing from scratch.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(DatasetWriter {
            inner: BufWriter::new(file),
        })
    }

    /// Write one trajectory as a single JSON line.
    pub fn write(&mut self, trajectory: &Trajectory) -> io::Result<()> {
        serde_json::to_writer(&mut self.inner, trajectory)?;
        self.inner.write_all(b"\n")?;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl Drop for DatasetWriter {
    fn drop(&mut self) {
        let _ = self.inner.flush();
    }
}

/// Streaming reader over a JSONL dataset — one [`Trajectory`] per line.
/// Blank lines are skipped.
pub struct DatasetReader {
    lines: std::io::Lines<BufReader<File>>,
}

impl DatasetReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(DatasetReader {
            lines: BufReader::new(file).lines(),
        })
    }
}

impl Iterator for DatasetReader {
    type Item = io::Result<Trajectory>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let line = match self.lines.next()? {
                Ok(l) => l,
                Err(e) => return Some(Err(e)),
            };
            if line.trim().is_empty() {
                continue;
            }
            return Some(serde_json::from_str(&line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)));
        }
    }
}

/// Convenience: read every trajectory from a JSONL file into memory.
pub fn read_trajectories(path: impl AsRef<Path>) -> io::Result<Vec<Trajectory>> {
    DatasetReader::open(path)?.collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::Position;
    use botbowl_engine::core::table::PosAT;

    fn sample_state() -> GameState {
        GameStateBuilder::new_start_of_game()
    }

    fn dummy_sample() -> Sample {
        let state = sample_state();
        let dims = state.board_dims;
        let _ = dims;
        Sample {
            state,
            to_move: Team::Home,
            chosen_action: Action::Positional(PosAT::Move, Position::new((5, 5))),
            children: vec![ChildStat {
                action: Action::Positional(PosAT::Move, Position::new((5, 5))),
                visits: 42,
                q: Some(500),
                prior: Some(1.5),
                solved: false,
                terminal: false,
            }],
            root_value: Some(500),
            root_visits: 100,
            root_solved: false,
            outcome_value: None,
        }
    }

    #[test]
    fn git_commit_is_stamped() {
        // Either a real 40-char sha or the "unknown" fallback.
        let c = git_commit();
        assert!(c == "unknown" || c.len() >= 7, "unexpected commit stamp: {c:?}");
    }

    #[test]
    fn trajectory_backfills_outcome_value() {
        let state = sample_state();
        let dims = state.board_dims;
        let meta = TrajectoryMeta::new("self-play", dims)
            .with_bots("mcts", "random")
            .with_seed(7)
            .with_extra("budget", "iters=100");
        let outcome = Outcome {
            home_score: 1,
            away_score: 0,
            winner: Some(Team::Home),
            game_over: true,
            z_home: 1.0,
            lecture_status: None,
        };
        let traj = Trajectory::new(meta, vec![dummy_sample()], outcome);
        assert_eq!(traj.samples[0].outcome_value, Some(1.0));
    }

    #[test]
    fn jsonl_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("botbowl_data_roundtrip_{}.jsonl", std::process::id()));

        let state = sample_state();
        let dims = state.board_dims;
        let make = |z: f32, home: u8, away: u8| {
            Trajectory::new(
                TrajectoryMeta::new("self-play", dims).with_bots("mcts", "mcts"),
                vec![dummy_sample()],
                Outcome {
                    home_score: home,
                    away_score: away,
                    winner: None,
                    game_over: true,
                    z_home: z,
                    lecture_status: None,
                },
            )
        };

        {
            let mut w = DatasetWriter::create(&path).unwrap();
            w.write(&make(1.0, 1, 0)).unwrap();
            w.write(&make(-1.0, 0, 1)).unwrap();
            w.flush().unwrap();
        }

        let back = read_trajectories(&path).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].outcome.z_home, 1.0);
        assert_eq!(back[0].samples[0].outcome_value, Some(1.0));
        assert_eq!(back[1].outcome.home_score, 0);
        assert_eq!(back[0].meta.format_version, FORMAT_VERSION);
        assert_eq!(back[0].meta.board_capacity, BoardCapacity::current());
        // GameState round-trips (modulo the skipped RNG).
        assert_eq!(back[0].samples[0].state, make(1.0, 1, 0).samples[0].state);

        std::fs::remove_file(&path).ok();
    }
}
