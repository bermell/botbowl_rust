//! `botbowl_mcts::report` → the wire's [`SearchReport`] / [`NodeExpansion`].

use botbowl_mcts::report::{Edge, NodeStats, NodeView, SearchSummary};
use botbowl_mcts::{BbAction, BbPlayer};
use botbowl_web_proto::search as ps;

use crate::mirror;

pub fn player_to_proto(p: BbPlayer) -> ps::NodePlayer {
    match p {
        BbPlayer::Home => ps::NodePlayer::Home,
        BbPlayer::Away => ps::NodePlayer::Away,
        BbPlayer::Chance => ps::NodePlayer::Chance,
    }
}

pub fn edge_to_proto(action: &BbAction) -> ps::SearchEdge {
    match action {
        BbAction::Player { action, .. } => ps::SearchEdge::Player(mirror::action_to_proto(*action)),
        BbAction::Chance { result, prob_bits } => ps::SearchEdge::Chance {
            result: mirror::roll_result_to_proto(*result),
            prob: f32::from_bits(*prob_bits),
        },
    }
}

/// Rebuild the search-tree edge a client echoed back.
///
/// `BbAction`'s `Eq`/`Hash` ignore `prior_bits` for player edges but *include*
/// `prob_bits` for chance edges, so a player edge can be reconstructed with a
/// dummy prior while a chance edge must carry the same probability back — the
/// client returns the `f32` it was given, and JSON round-trips an `f32`
/// exactly, so the child lookup hits.
pub fn edge_from_proto(edge: &ps::SearchEdge) -> Result<BbAction, String> {
    Ok(match edge {
        ps::SearchEdge::Player(action) => BbAction::player(mirror::action_from_proto(*action), 0.0),
        ps::SearchEdge::Chance { result, prob } => BbAction::chance(mirror::roll_result_from_proto(result)?, *prob),
    })
}

pub fn stats_to_proto(stats: &NodeStats) -> ps::NodeStats {
    ps::NodeStats {
        visits: stats.visits,
        q_home: stats.q_home,
        q_display: stats.q_agent,
        solved: stats.solved,
        terminal: stats.terminal,
        player: player_to_proto(stats.player),
    }
}

fn child_to_proto(edge: &Edge, root_visits: u32) -> ps::ChildReport {
    ps::ChildReport {
        edge: edge_to_proto(&edge.action),
        stats: stats_to_proto(&edge.stats),
        prior: edge.prior(),
        visit_share: if root_visits == 0 {
            0.0
        } else {
            edge.stats.visits as f32 / root_visits as f32
        },
    }
}

/// The heatmap normalises against the busiest sibling, not the root's own
/// visit count: the root's counter is "descents through the root", which for
/// a reused tree can be far larger than the sum over the current children,
/// and normalising by it washes the colours out.
fn heat_denominator(summary: &SearchSummary) -> u32 {
    summary
        .children
        .iter()
        .map(|c| c.stats.visits)
        .max()
        .unwrap_or(0)
        .max(1)
}

pub fn summary_to_proto(search_id: u64, summary: &SearchSummary, pv: &[Edge], solved: bool) -> ps::SearchReport {
    let denom = heat_denominator(summary);
    let mut path: Vec<ps::SearchEdge> = Vec::new();
    let pv_steps = pv
        .iter()
        .map(|edge| {
            path.push(edge_to_proto(&edge.action));
            ps::PvStep {
                edge: edge_to_proto(&edge.action),
                stats: stats_to_proto(&edge.stats),
                path: path.clone(),
            }
        })
        .collect();

    ps::SearchReport {
        search_id,
        agent: mirror::team_to_proto(summary.agent),
        chosen: mirror::action_to_proto(summary.chosen),
        root_visits: summary.root.visits,
        root_q_home: summary.root.q_home,
        root_q_display: summary.root.q_agent,
        children: summary.children.iter().map(|c| child_to_proto(c, denom)).collect(),
        pv: pv_steps,
        elapsed_ms: summary.elapsed.as_millis() as u64,
        budget: summary.budget.clone(),
        evaluator: summary.evaluator.clone(),
        evaluator_value: summary.evaluator_value,
        solved,
    }
}

pub fn node_to_proto(
    search_id: u64,
    path: Vec<ps::SearchEdge>,
    node: &NodeView,
    view: Option<Box<botbowl_web_proto::view::ViewState>>,
) -> ps::NodeExpansion {
    let denom = node.children.iter().map(|c| c.stats.visits).max().unwrap_or(0).max(1);
    let total = node.children.len();
    let children: Vec<ps::ChildReport> = node
        .children
        .iter()
        .take(ps::MAX_CHILDREN_PER_NODE)
        .map(|c| child_to_proto(c, denom))
        .collect();
    ps::NodeExpansion {
        search_id,
        path,
        stats: stats_to_proto(&node.stats),
        depth: node.stats.depth,
        n_parents: node.stats.n_parents,
        n_children: node.stats.n_children,
        children_omitted: total.saturating_sub(ps::MAX_CHILDREN_PER_NODE),
        children,
        proc: node.stats.proc.clone(),
        view,
    }
}
