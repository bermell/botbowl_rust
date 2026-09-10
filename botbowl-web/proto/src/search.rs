//! What the MCTS bot reports about the search behind a move.
//!
//! Q values are the engine's raw `leaf_score` scale (a touchdown is ±1000) and
//! **Home-centric end-to-end**, exactly as `recon_mcts` stores them. The
//! server also ships a mover-centric `q_display` in `[-1, 1]` so the client
//! never has to know that convention.

use serde::{Deserialize, Serialize};

use crate::action::{Action, TeamType};
use crate::dice::RollResult;

/// Which kind of node an edge leads to. The Blood Bowl search has three
/// "players": the two teams plus `Chance` for pending-roll nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodePlayer {
    Home,
    Away,
    Chance,
}

/// One edge out of a search node: either a team's action or a die outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SearchEdge {
    Player(Action),
    Chance { result: RollResult, prob: f32 },
}

impl SearchEdge {
    pub fn action(&self) -> Option<Action> {
        match self {
            SearchEdge::Player(a) => Some(*a),
            SearchEdge::Chance { .. } => None,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            SearchEdge::Player(a) => a.describe(),
            SearchEdge::Chance { result, prob } => format!("{result:?} (p={prob:.3})"),
        }
    }
}

/// Aggregated statistics for one node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeStats {
    /// Descents through this node, cumulative across reused trees this turn.
    pub visits: u32,
    /// Home-centric aggregated score, `None` when the node was never scored
    /// (an expanded chance node is deliberately unscored).
    pub q_home: Option<i64>,
    /// `q_home` from the searching agent's perspective, rescaled to `[-1, 1]`
    /// (a touchdown is ±1).
    pub q_display: Option<f32>,
    /// `recon_mcts` has proven this subtree out.
    pub solved: bool,
    /// No children at all — terminal or past the horizon.
    pub terminal: bool,
    pub player: NodePlayer,
}

/// One root child: the candidate list and the pitch heatmap are both views of
/// this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChildReport {
    pub edge: SearchEdge,
    pub stats: NodeStats,
    /// PUCT prior for the edge. Un-normalised: the scripted priors have mean
    /// ≈ 1.0 and the NN priors are `softmax × len`, same mean, wider spread.
    pub prior: Option<f32>,
    /// `visits / root_visits`, precomputed for the heatmap.
    pub visit_share: f32,
}

/// Everything the inspector shows for one bot decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchReport {
    /// Identifies the tree this report came from. The bot keeps exactly
    /// **one** tree — the most recent search's — so an inspector opened on an
    /// older move can no longer be walked. The client sends this back with
    /// [`crate::ClientMsg::ExpandNode`] and the server refuses a stale one
    /// rather than answering about the wrong position.
    pub search_id: u64,
    /// Which side searched — the perspective `q_display` is in.
    pub agent: TeamType,
    pub chosen: Action,
    pub root_visits: u32,
    pub root_q_home: Option<i64>,
    pub root_q_display: Option<f32>,
    /// Sorted by visits, descending.
    pub children: Vec<ChildReport>,
    /// Greedy principal variation from the root, by best mover-Q then visits.
    pub pv: Vec<PvStep>,
    pub elapsed_ms: u64,
    /// How the budget was expressed, e.g. `"2000 iterations"` or `"500 ms"`.
    pub budget: String,
    pub evaluator: String,
    /// The evaluator's own value at the root, mover-centric in `[-1, 1]`.
    /// Only present for the NN evaluators.
    pub evaluator_value: Option<f32>,
    /// True when the whole tree was solved and the workers stopped early.
    pub solved: bool,
}

/// One step of the principal variation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PvStep {
    pub edge: SearchEdge,
    pub stats: NodeStats,
    /// Path from the root to *and including* this edge — what
    /// [`crate::ClientMsg::ExpandNode`] takes.
    pub path: Vec<SearchEdge>,
}

/// Answer to [`crate::ClientMsg::ExpandNode`]: one node plus a bounded slice
/// of its children. The DAG is never serialised whole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeExpansion {
    /// Echo of the request, so a late answer can be matched up.
    pub search_id: u64,
    pub path: Vec<SearchEdge>,
    pub stats: NodeStats,
    pub depth: usize,
    pub n_parents: usize,
    /// `None` when the node is unexpanded.
    pub n_children: Option<usize>,
    /// Sorted by visits, descending, and truncated — see
    /// [`MAX_CHILDREN_PER_NODE`].
    pub children: Vec<ChildReport>,
    /// How many children were dropped by the truncation.
    pub children_omitted: usize,
    /// `proc_stack_top()` at this node, when the tree stores states.
    pub proc: Option<String>,
    /// The board at this node, for "step into the PV" (phase 3). Only present
    /// under `MemoryMode::StoreState`, which is what production uses.
    pub view: Option<Box<crate::view::ViewState>>,
}

/// Serialisation bound for one `ExpandNode` answer (plan 034: depth ≤ 3,
/// ≤ 12 children per node).
pub const MAX_CHILDREN_PER_NODE: usize = 12;
