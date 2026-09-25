//! Wire types between `botbowl-hub` and `botbowl-worker` (plan 041).
//!
//! One websocket per worker, binary frames, `postcard` encoding. The
//! worker dials the hub, sends [`ToHub::Hello`], and from then on the hub
//! pushes [`ToWorker::Task`]s (preceded by any [`ToWorker::Model`] the
//! worker has not confirmed having) while the worker streams one
//! result frame per finished game.
//!
//! Compatibility is four checks: [`PROTOCOL_VERSION`] (bump on any
//! frame-layout change), the compiled board capacity, the *active* board the
//! worker's environment selects within that capacity, and the commit baked
//! into both binaries — exact match unless the hub was started with
//! `--allow-commit-mismatch` or the worker's commit is named in the hub's
//! allowlist file (`botbowl_hub::allowlist`). A dirty tree is refused in
//! every case unless the hub is dirty too. The rationale is in
//! `plans/041-plan--distributed-hub-and-workers.md` decision 5.

use serde::{Deserialize, Serialize};

pub use botbowl_engine::core::model::BoardDims;
pub use botbowl_play::board_sizes::SizeDist;
pub use botbowl_play::bots::{Evaluator, SearchConfig};
pub use botbowl_play::eval::EvalGameLine;
pub use botbowl_play::generate::GenerateConfig;

/// Bump on any change to the frames below.
// v3 (plan 042): `Task::Eval.board` and `GenerateConfig.board_sizes`.
// v4 (plan 043): `SearchConfig.config` (a named `MctsConfig` preset) and
// `GenerateConfig.config_name`.
// v5: `BuildInfo.env_board` + `RejectReason::Board` — the *active* board is
// part of compatibility, not just the compiled capacity.
pub const PROTOCOL_VERSION: u32 = 5;

/// Content hash of an ONNX file (BLAKE3). Model identity is bytes, never a
/// path, so two workers with the same cache can never disagree about which
/// net a task means.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModelId(pub [u8; 32]);

impl ModelId {
    pub fn of(onnx: &[u8]) -> Self {
        ModelId(*blake3::hash(onnx).as_bytes())
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(ModelId(out))
    }
}

impl std::fmt::Debug for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelId({}..)", &self.to_hex()[..12])
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Compiled board capacity `(width, height, team_size)` including the
/// engine's out-of-bounds border, i.e. `BoardDims::default()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capacity {
    pub width: u16,
    pub height: u16,
    pub team_size: u16,
}

impl Capacity {
    pub fn compiled() -> Self {
        let d = botbowl_engine::core::model::BoardDims::default();
        Capacity {
            width: d.width as u16,
            height: d.height as u16,
            team_size: d.team_size as u16,
        }
    }
}

/// What this binary was built from, and what board its environment selects.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    pub commit: String,
    pub dirty: bool,
    pub capacity: Capacity,
    /// The *active* board, `BoardDims::from_env()` — what a task that names no
    /// board of its own actually plays on. Distinct from `capacity`, which is
    /// only the compile-time ceiling: one binary built at 16x9/6 plays 12x5/3
    /// or 16x9/6 depending on `BOARD_SIZE_*` at run time, so two workers that
    /// agree on capacity can still disagree about the game. It is in the
    /// handshake so an env-board job means one board across the whole fleet.
    pub env_board: BoardDims,
}

impl BuildInfo {
    pub fn current() -> Self {
        BuildInfo {
            commit: botbowl_data::git_commit().to_string(),
            dirty: botbowl_data::git_dirty(),
            capacity: Capacity::compiled(),
            env_board: BoardDims::from_env(),
        }
    }
}

pub type TaskId = u64;

/// A fully specified bot, buildable on any worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BotSpec {
    Random,
    Scripted,
    Mcts {
        search: SearchConfig,
        evaluator: Evaluator,
        /// Required when `evaluator.needs_model()`.
        model: Option<ModelId>,
    },
}

impl BotSpec {
    pub fn model(&self) -> Option<ModelId> {
        match self {
            BotSpec::Mcts { model, .. } => *model,
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Task {
    /// Play these games of one ladder rung. Game `g`'s side and seed come
    /// from `botbowl_play::eval::ladder_assignment(seed, g)`.
    Eval {
        id: TaskId,
        rung: String,
        games: Vec<u32>,
        seed: u64,
        max_steps: u32,
        candidate: BotSpec,
        opponent: BotSpec,
        /// Plan 042: the board this rung plays on; `None` = the worker's
        /// env board (which the capacity check makes the same as the hub's).
        board: Option<BoardDims>,
    },
    /// Play these games of one corpus shard. Game `g`'s seed is
    /// `seed_base + g`, exactly as `botbowl-ui dataset --seed seed_base`
    /// numbers them, so a hub-generated shard has the same seed set as the
    /// single-process one it replaces.
    Generate {
        id: TaskId,
        shard: String,
        games: Vec<u32>,
        seed_base: u64,
        /// `cfg.model` is the hub-side path, used only for the provenance
        /// label; the bytes come from `model`.
        cfg: GenerateConfig,
        model: Option<ModelId>,
    },
}

impl Task {
    pub fn id(&self) -> TaskId {
        match self {
            Task::Eval { id, .. } | Task::Generate { id, .. } => *id,
        }
    }

    /// Every model the worker must hold before it can run this task.
    pub fn models(&self) -> Vec<ModelId> {
        match self {
            Task::Eval {
                candidate, opponent, ..
            } => candidate.model().into_iter().chain(opponent.model()).collect(),
            Task::Generate { model, .. } => model.iter().copied().collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ToHub {
    Hello {
        protocol: u32,
        token: String,
        build: BuildInfo,
        /// Rust target triple, for `Update` (phase 4).
        triple: String,
        name: String,
        cores: u16,
        ram_mb: u32,
        /// Models already in the worker's on-disk cache.
        cached_models: Vec<ModelId>,
        /// Streams the worker will run; `None` lets the hub size it.
        parallel_games: Option<u16>,
    },
    EvalGameDone {
        task: TaskId,
        line: EvalGameLine,
    },
    /// One finished trajectory: its JSON line (without the newline), zstd
    /// compressed. The hub appends the decompressed bytes verbatim, so the
    /// shard file is byte-for-byte what `DatasetWriter` writes. Empty
    /// `zstd_json` means the game legitimately produced nothing (a
    /// curriculum trial the mode skipped); the game still counts as done.
    TrajectoryDone {
        task: TaskId,
        game: u32,
        samples: u32,
        zstd_json: Vec<u8>,
    },
    /// A game (or a whole task) could not be played. The hub requeues it
    /// elsewhere; a task that fails everywhere fails the job.
    TaskFailed {
        task: TaskId,
        error: String,
    },
    Heartbeat {
        games_in_flight: u16,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    Protocol { hub: u32 },
    BadToken,
    Commit { hub: String },
    Dirty,
    Capacity { hub: Capacity },
    /// Same binary, different `BOARD_SIZE_*` in the worker's environment.
    Board { hub: BoardDims },
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::Protocol { hub } => write!(f, "protocol version mismatch (hub speaks {hub})"),
            RejectReason::BadToken => f.write_str("bad token"),
            RejectReason::Commit { hub } => write!(f, "git commit mismatch (hub is {hub}); rebuild from that commit"),
            RejectReason::Dirty => f.write_str("worker built from a dirty tree; commit and rebuild"),
            RejectReason::Capacity { hub } => write!(f, "board capacity mismatch (hub is {hub:?})"),
            RejectReason::Board { hub } => write!(
                f,
                "active board mismatch (the hub's env board is {}); export the same BOARD_SIZE_W/BOARD_SIZE_H/BOARD_PLAYERS, or submit jobs with explicit --board-sizes",
                botbowl_play::board_sizes::board_label(*hub)
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ToWorker {
    Welcome {
        /// Streams the hub will keep busy on this worker.
        parallel_games: u16,
    },
    Reject {
        reason: RejectReason,
    },
    /// ONNX bytes for a model the worker reported not having. Always
    /// precedes the first task that needs it.
    Model {
        id: ModelId,
        onnx: Vec<u8>,
    },
    Task(Task),
    /// Finish in-flight games, then expect the socket to close.
    Drain,
}

pub fn encode<T: Serialize>(msg: &T) -> Vec<u8> {
    postcard::to_allocvec(msg).expect("postcard encode")
}

pub fn decode<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, postcard::Error> {
    postcard::from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_hex_roundtrip() {
        let id = ModelId::of(b"hello");
        assert_eq!(ModelId::from_hex(&id.to_hex()), Some(id));
        assert_eq!(id.to_hex().len(), 64);
        assert!(ModelId::from_hex("zz").is_none());
    }

    #[test]
    fn frames_roundtrip() {
        let task = Task::Eval {
            id: 7,
            rung: "scripted".into(),
            games: vec![0, 1, 2],
            seed: 5,
            max_steps: 100_000,
            candidate: BotSpec::Mcts {
                search: SearchConfig::iterations(8),
                evaluator: Evaluator::Nn,
                model: Some(ModelId::of(b"net")),
            },
            opponent: BotSpec::Scripted,
            board: Some(BoardDims::default()),
        };
        let bytes = encode(&ToWorker::Task(task.clone()));
        let back: ToWorker = decode(&bytes).unwrap();
        match back {
            ToWorker::Task(t) => {
                assert_eq!(t.id(), 7);
                assert_eq!(t.models(), vec![ModelId::of(b"net")]);
            }
            other => panic!("{other:?}"),
        }
        let hello = ToHub::Hello {
            protocol: PROTOCOL_VERSION,
            token: "t".into(),
            build: BuildInfo::current(),
            triple: "x".into(),
            name: "n".into(),
            cores: 8,
            ram_mb: 16_000,
            cached_models: vec![],
            parallel_games: None,
        };
        let _: ToHub = decode(&encode(&hello)).unwrap();
    }
}
