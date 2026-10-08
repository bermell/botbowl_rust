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
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action, BoardDims, TeamType, HEIGHT, TEAM_SIZE, WIDTH};

/// Schema version. Bump on any breaking change to the types below so a
/// reader can reject / migrate old files.
pub const FORMAT_VERSION: u32 = 1;

/// `(commit, dirty)` of the checkout this binary was built from, read **at
/// process start** (first call) and cached for the life of the process.
///
/// Runs `git rev-parse HEAD` and `git status --porcelain
/// --untracked-files=no` in the workspace root baked in at compile time
/// (`CARGO_MANIFEST_DIR/..`), so the answer does not depend on the cwd.
/// *Dirty* means staged or unstaged changes to tracked files; untracked
/// files do not count (a new file only reaches the build through an edit
/// to a tracked one). `scripts/lib/git.sh`'s `require_clean_tree` uses the
/// same definition.
///
/// This replaced a build-time stamp whose `rerun-if-changed=.git/index`
/// rebuilt 9 crates on every `git status`, yet missed unstaged edits. The
/// trade-off: a binary that is *not* rebuilt after a commit stamps the new
/// HEAD. Every launcher builds via `cargo build`/`cargo run` first, which
/// recompiles on any source change, so stamp and code agree whenever the
/// binary is current.
///
/// If `git` cannot run (no git, binary copied off its checkout) the stamp
/// falls back to the commit `build.rs` saw when this crate was last
/// compiled, reported **dirty** because nothing verifies it.
pub fn git_provenance() -> (&'static str, bool) {
    static STAMP: OnceLock<(String, bool)> = OnceLock::new();
    let (commit, dirty) = STAMP.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let git = |args: &[&str]| -> Option<String> {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(args)
                .output()
                .ok()?;
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        };
        match (
            git(&["rev-parse", "HEAD"]),
            git(&["status", "--porcelain", "--untracked-files=no"]),
        ) {
            (Some(commit), Some(status)) => (commit, !status.is_empty()),
            _ => (env!("BOTBOWL_BUILD_GIT_COMMIT").to_string(), true),
        }
    });
    (commit.as_str(), *dirty)
}

/// The commit half of [`git_provenance`] (full SHA, or `"unknown"`). This
/// is the commit recorded on every trajectory this binary produces.
pub fn git_commit() -> &'static str {
    git_provenance().0
}

/// The dirty half of [`git_provenance`]. `true` means the recorded
/// [`git_commit`] does not fully describe the generating code — treat such
/// datasets with suspicion.
pub fn git_dirty() -> bool {
    git_provenance().1
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
    /// outcome of this sample's *drive* from Home's perspective in
    /// `[-1, 1]` — the score delta to the next score change (or the
    /// trajectory end), see [`Trajectory::backfill_outcome_value`].
    /// `None` before backfilling.
    #[serde(default)]
    pub outcome_value: Option<f32>,
    /// Plan 047: the action was not searched — a formation plan answered a setup
    /// placement (the gen-0 teacher shard), or one action was left after pruning
    /// (a forced decision; `MctsBot` plays it without a search). `children` then
    /// holds one visit on the chosen action and no `Q`, so the policy target is
    /// one-hot on it; `root_visits` is 1, `root_value` is `None`, and `prepare`
    /// must not drop it as an under-searched root. A formation sample lists every
    /// legal action as a child; a forced one lists only the one post-pruning
    /// action, which is how `prepare` recognises it and leaves it out of training
    /// (fewer than two children). Forced samples stay in the trajectory: replay
    /// rebuilds every state from the recorded `chosen_action`s.
    #[serde(default)]
    pub scripted: bool,
    /// Plan 060: the shape of the search behind this decision — how deep its descents went and
    /// where they ended relative to the horizon. `None` for an unsearched decision and in every
    /// corpus written before plan 060; never read by `prepare` or the trainer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree: Option<TreeStats>,
}

/// Plan 060: where one search's descents went. Counted **per descent** (each of the `descents`
/// counts once), not per node: a per-node count is dominated by thousands of one-visit leaves, a
/// per-descent count is where the budget went.
///
/// "Own" means the root's mover (the team the search plays for); a "ply" is any tree edge — a
/// decision by either side or a chance outcome. Phases are relative to the root's horizon anchor
/// (`botbowl_mcts::HorizonAnchor`): at the default `horizon_turns = 1` a line runs through the
/// rest of the mover's turn and the opponent's next turn, and stops when the mover's next turn
/// begins.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct TreeStats {
    /// Descents this search ran (a reused tree's earlier descents are not counted).
    pub descents: u32,
    /// Leaf depth in plies from the root (decision and chance edges).
    pub plies: DepthStats,
    /// Leaf depth in the root mover's own decisions only.
    pub own: DepthStats,
    /// Chance edges per descent, mean. `chance_plies_mean / plies.mean` is the share of the depth
    /// that is dice, not decisions.
    pub chance_plies_mean: f32,
    /// Where each descent stopped.
    pub ends: PhaseCounts,
    /// Descents whose leaf lies in the opponent's following turn or later (the opponent's turn
    /// counter moved past the root's), however the descent then ended.
    pub reached_opp_turn: u32,
    /// How each descent's leaf got its value.
    pub valued: LeafValueCounts,
    /// The line the search settled on: most-visited child from the root down.
    pub main_line: MainLine,
    /// The opponent has a turn inside this search's horizon. `false` at the end of a half (the
    /// opponent has played its last turn) — analyses filter those decisions out.
    pub opp_turn_follows: bool,
    /// `proc_stack_top()` at the root: what kind of decision this was.
    pub proc: Option<String>,
}

/// Mean / nearest-rank p90 / max of a per-descent depth.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct DepthStats {
    pub mean: f32,
    pub p90: u32,
    pub max: u32,
}

/// Where a line stopped, relative to the root's horizon anchor.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Still in the root mover's current turn (or, for a root in a setup, before the next turn).
    OwnTurn,
    /// In the opponent's following turn.
    OppTurn,
    /// Past the horizon: the mover's next turn began (the horizon leaf).
    Horizon,
    /// Someone scored (terminal).
    Score,
    /// The half ended.
    HalfEnd,
    /// The game ended.
    GameOver,
}

/// Descents per [`Phase`] they stopped in.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseCounts {
    pub own_turn: u32,
    pub opp_turn: u32,
    pub horizon: u32,
    pub score: u32,
    pub half_end: u32,
    pub game_over: u32,
}

impl PhaseCounts {
    pub fn record(&mut self, phase: Phase) {
        *self.slot(phase) += 1;
    }

    pub fn slot(&mut self, phase: Phase) -> &mut u32 {
        match phase {
            Phase::OwnTurn => &mut self.own_turn,
            Phase::OppTurn => &mut self.opp_turn,
            Phase::Horizon => &mut self.horizon,
            Phase::Score => &mut self.score,
            Phase::HalfEnd => &mut self.half_end,
            Phase::GameOver => &mut self.game_over,
        }
    }

    pub fn total(&self) -> u32 {
        self.own_turn + self.opp_turn + self.horizon + self.score + self.half_end + self.game_over
    }
}

/// How each descent's leaf was valued.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeafValueCounts {
    /// A fresh decision leaf inside the horizon, scored by the evaluator (net or heuristic).
    pub new_leaf: u32,
    /// A fresh chance node inside the horizon: expanded, its value waits for its outcomes.
    pub chance: u32,
    /// A known outcome (a score, the half or the game over), or an in-horizon dead end.
    pub terminal: u32,
    /// Every child of the node the descent reached was solved.
    pub solved: u32,
    /// The horizon leaf: the mover's next turn began, valued by the evaluator.
    pub horizon: u32,
}

/// The search's main line: from the root, the most-visited child at every node (the most-visited
/// outcome at chance nodes), until a node with no visited child.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct MainLine {
    pub plies: u32,
    /// The root mover's own decisions on it.
    pub own: u32,
    pub chance: u32,
    /// The phase its last node is in. `None` when the node keeps no state (`MemoryMode::GetState`).
    pub phase: Option<Phase>,
}

/// How a trajectory ended — the value-target ground truth.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Outcome {
    pub home_score: u8,
    pub away_score: u8,
    pub winner: Option<Team>,
    pub game_over: bool,
    /// Outcome from Home's perspective in `[-1, 1]`: `+1` Home ahead at the
    /// end of the trajectory, `-1` Away ahead, `0` level. Trajectory-level
    /// descriptor only — per-sample value targets are drive-relative and
    /// live in [`Sample::outcome_value`] (they are **not** this broadcast).
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

    /// Backfill each sample's `outcome_value` with the outcome of *its
    /// drive*: the Home-centric score delta from the sample's state to the
    /// first subsequent score change — or to the trajectory's final score
    /// if no one scored again — clamped to `[-1, 1]`. Called by
    /// [`Trajectory::new`]; exposed for callers that mutate samples after
    /// construction.
    ///
    /// Deliberately **not** the broadcast final-scoreline `z_home` (plan
    /// 017 caveat 2026-07-14): trajectories starting from a non-level
    /// score, or spanning several drives, would otherwise label every
    /// sample with the wrong target — and the search consumes values in a
    /// drive-relative frame (`HorizonAnchor::score_delta`), so the value
    /// head must be trained in that same frame.
    pub fn backfill_outcome_value(&mut self) {
        fn score_of(s: &GameState) -> (u8, u8) {
            (s.home.score, s.away.score)
        }
        let final_score = (self.outcome.home_score, self.outcome.away_score);
        // Walk backwards keeping the score at the end of the drive the
        // current sample belongs to: the score right after the first
        // change following it, defaulting to the trajectory's end.
        let mut drive_end = final_score;
        for i in (0..self.samples.len()).rev() {
            let cur = score_of(&self.samples[i].state);
            let next = self.samples.get(i + 1).map_or(final_score, |s| score_of(&s.state));
            if next != cur {
                drive_end = next;
            }
            let dv = (drive_end.0 as f32 - cur.0 as f32) - (drive_end.1 as f32 - cur.1 as f32);
            self.samples[i].outcome_value = Some(dv.clamp(-1.0, 1.0));
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
            scripted: false,
            tree: None,
        }
    }

    /// Plan 060: `tree` is optional both ways. A corpus written before it existed still parses
    /// (no key → `None`), and a sample without one writes no key, so readers that predate it see
    /// exactly the old line.
    #[test]
    fn tree_stats_are_an_optional_trailing_key() {
        let plain = dummy_sample();
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("\"tree\""), "no tree key without stats: {json}");
        let back: Sample = serde_json::from_str(&json).unwrap();
        assert!(back.tree.is_none());

        let mut with = dummy_sample();
        with.tree = Some(TreeStats {
            descents: 1000,
            plies: DepthStats {
                mean: 7.5,
                p90: 12,
                max: 19,
            },
            own: DepthStats {
                mean: 3.0,
                p90: 5,
                max: 8,
            },
            chance_plies_mean: 2.5,
            ends: PhaseCounts {
                own_turn: 600,
                opp_turn: 300,
                horizon: 50,
                score: 40,
                half_end: 10,
                game_over: 0,
            },
            reached_opp_turn: 380,
            valued: LeafValueCounts {
                new_leaf: 700,
                chance: 100,
                terminal: 60,
                solved: 90,
                horizon: 50,
            },
            main_line: MainLine {
                plies: 9,
                own: 4,
                chance: 3,
                phase: Some(Phase::OppTurn),
            },
            opp_turn_follows: true,
            proc: Some("Turn".to_string()),
        });
        let json = serde_json::to_string(&with).unwrap();
        assert!(json.contains("\"phase\":\"opp_turn\""), "phases are snake_case: {json}");
        let back: Sample = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tree, with.tree);
    }

    #[test]
    fn git_commit_is_stamped() {
        // Either a real 40-char sha or the "unknown" fallback.
        let c = git_commit();
        assert!(c == "unknown" || c.len() >= 7, "unexpected commit stamp: {c:?}");
    }

    #[test]
    fn git_provenance_is_read_at_runtime_from_this_checkout() {
        // Tests run from a checkout, so the runtime path (not the build.rs
        // fallback) must answer: the stamp is the checkout's live HEAD.
        let head = std::process::Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
            .output()
            .expect("git runs in the test checkout");
        let head = String::from_utf8_lossy(&head.stdout).trim().to_string();
        assert_eq!(git_commit(), head);
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

    fn dummy_sample_with_score(home: u8, away: u8) -> Sample {
        let mut s = dummy_sample();
        s.state.home.score = home;
        s.state.away.score = away;
        s
    }

    #[test]
    fn outcome_value_is_drive_relative_not_final_scoreline() {
        let dims = sample_state().board_dims;
        let meta = || TrajectoryMeta::new("random-start", dims).with_bots("mcts", "mcts");

        // Non-level start, scoreless to the end: the final-scoreline z is
        // +1 but no drive produced a score → every target must be 0.
        let outcome = Outcome {
            home_score: 1,
            away_score: 0,
            winner: Some(Team::Home),
            game_over: true,
            z_home: 1.0,
            lecture_status: None,
        };
        let traj = Trajectory::new(meta(), vec![dummy_sample_with_score(1, 0)], outcome);
        assert_eq!(traj.samples[0].outcome_value, Some(0.0));

        // Two drives: Home scores after sample 1 (0-0 → 1-0), Away scores
        // after sample 3 (1-0 → 1-1). Targets follow each sample's drive.
        let outcome = Outcome {
            home_score: 1,
            away_score: 1,
            winner: None,
            game_over: true,
            z_home: 0.0,
            lecture_status: None,
        };
        let samples = vec![
            dummy_sample_with_score(0, 0),
            dummy_sample_with_score(0, 0),
            dummy_sample_with_score(1, 0),
            dummy_sample_with_score(1, 0),
        ];
        let traj = Trajectory::new(meta(), samples, outcome);
        let values: Vec<f32> = traj.samples.iter().map(|s| s.outcome_value.unwrap()).collect();
        assert_eq!(values, vec![1.0, 1.0, -1.0, -1.0]);
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
