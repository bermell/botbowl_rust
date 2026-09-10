//! Read-out of a finished search: what the bot looked at, and why it chose
//! what it chose.
//!
//! Added for plan 034's bot inspector, but deliberately free of any web
//! dependency — these are plain data structs over the search's own vocabulary,
//! so the same read-out serves a UI, a diagnostic dump, or a test assertion.
//!
//! **Q is Home-centric** on the wire, exactly as `recon_mcts` stores it
//! (plan 006), on `leaf_score`'s scale where a touchdown is ±1000. Each
//! struct also carries `q_mover`, the same number from the perspective of the
//! node's own player and rescaled to `[-1, 1]`, because that is the only form
//! that reads correctly next to a board.
//!
//! Visit counts are "descents through this node", cumulative across reused
//! trees within a turn, and `recon_mcts` freezes them once a subtree is
//! solved — so they are a search-effort measure, not a move-quality one.
//! `MctsBot` picks by aggregated Q for exactly that reason.

use std::time::Duration;

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, TeamType};

use crate::action::{BbAction, BbPlayer};

/// The `leaf_score` value of a touchdown — the scale `q_home` is on.
pub const Q_SCALE: f32 = 1000.0;

/// Aggregated statistics for one search node.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeStats {
    /// Descents through this node.
    pub visits: u32,
    /// Home-centric aggregated score. `None` when the node was never scored —
    /// an expanded chance node is deliberately unscored (plan 018).
    pub q_home: Option<i64>,
    /// `q_home` from this node's own player's perspective, rescaled so a
    /// touchdown is ±1. `Chance` nodes keep the Home frame.
    pub q_mover: Option<f32>,
    /// `recon_mcts` has proven this subtree out.
    pub solved: bool,
    /// No children: terminal, or past the search horizon.
    pub terminal: bool,
    pub player: BbPlayer,
    pub depth: usize,
    pub n_parents: usize,
    /// `None` when the node has not been expanded.
    pub n_children: Option<usize>,
    /// `proc_stack_top()` of the node's state, when the tree stores states.
    pub proc: Option<String>,
}

/// One edge out of a node, together with its child's stats.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub action: BbAction,
    pub stats: NodeStats,
}

impl Edge {
    /// PUCT prior, for player edges.
    pub fn prior(&self) -> Option<f32> {
        self.action.prior_f32()
    }

    /// Outcome probability, for chance edges.
    pub fn prob(&self) -> Option<f32> {
        self.action.prob_f32()
    }
}

/// Everything known about one completed decision.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSummary {
    /// The side that searched — the perspective the *root's* `q_mover` is in.
    pub agent: TeamType,
    pub chosen: EngineAction,
    pub root: NodeStats,
    /// Root children, sorted by visits descending.
    pub children: Vec<Edge>,
    pub elapsed: Duration,
    /// How the budget was expressed, e.g. `"2000 iterations"`.
    pub budget: String,
    /// Which value/prior source ran, e.g. `"nn"`.
    pub evaluator: String,
    /// The evaluator's own value at the root, mover-centric in `[-1, 1]`.
    /// Only the NN evaluators have one to report.
    pub evaluator_value: Option<f32>,
}

impl SearchSummary {
    /// Total visits across the root's children — the search's real work
    /// measure, which can exceed the root's own visit count.
    pub fn child_visits(&self) -> u32 {
        self.children.iter().map(|c| c.stats.visits).sum()
    }
}

/// One node of the search DAG plus a level of its children, as returned by
/// [`crate::MctsBot::explore`].
#[derive(Debug, Clone)]
pub struct NodeView {
    pub stats: NodeStats,
    /// Sorted by visits descending. Empty for an unexpanded node.
    pub children: Vec<Edge>,
    /// The board at this node. `Some` under `MemoryMode::StoreState`, which
    /// is what production uses.
    pub state: Option<GameState>,
}

/// Sign of the mover's perspective relative to Home's.
pub(crate) fn mover_sign(player: BbPlayer) -> f32 {
    match player {
        BbPlayer::Away => -1.0,
        BbPlayer::Home | BbPlayer::Chance => 1.0,
    }
}

/// `q_home` → the node player's frame, rescaled to `[-1, 1]`.
pub(crate) fn q_mover(q_home: Option<i64>, player: BbPlayer) -> Option<f32> {
    q_home.map(|q| mover_sign(player) * (q as f32) / Q_SCALE)
}
