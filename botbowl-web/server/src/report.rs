//! `botbowl_mcts::report` → the wire's [`SearchReport`] / [`NodeExpansion`].

use botbowl_mcts::report::{Edge, NodeStats, NodeView, SearchSummary};
use botbowl_mcts::{BbAction, BbPlayer};
use botbowl_web_proto::search as ps;

use crate::mirror;

/// `None` is an unmaterialised placeholder — enumerated, never descended
/// into, so it has no mover yet (plan 035).
pub fn player_to_proto(p: Option<BbPlayer>) -> ps::NodePlayer {
    match p {
        Some(BbPlayer::Home) => ps::NodePlayer::Home,
        Some(BbPlayer::Away) => ps::NodePlayer::Away,
        Some(BbPlayer::Chance) => ps::NodePlayer::Chance,
        None => ps::NodePlayer::Pending,
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

/// How one set of sibling edges normalises, computed once per node.
struct Siblings {
    /// The busiest sibling's visits — the heatmap's denominator.
    heat: u32,
    /// Total visits over the siblings.
    visits: u32,
    /// Sum of the player edges' priors.
    prior: f32,
}

impl Siblings {
    /// The heatmap normalises against the busiest sibling, not the root's own
    /// visit count: the root's counter is "descents through the root", which
    /// for a reused tree can be far larger than the sum over the current
    /// children, and normalising by it washes the colours out.
    fn of(edges: &[Edge]) -> Self {
        Siblings {
            heat: edges.iter().map(|c| c.stats.visits).max().unwrap_or(0).max(1),
            visits: edges.iter().map(|c| c.stats.visits).sum(),
            prior: edges.iter().filter_map(Edge::prior).sum(),
        }
    }
}

fn child_to_proto(edge: &Edge, siblings: &Siblings, net_value: Option<f32>) -> ps::ChildReport {
    let prior = edge.prior();
    ps::ChildReport {
        edge: edge_to_proto(&edge.action),
        stats: stats_to_proto(&edge.stats),
        prior,
        prior_share: prior.filter(|_| siblings.prior > 0.0).map(|p| p / siblings.prior),
        visit_share: edge.stats.visits as f32 / siblings.heat as f32,
        visit_prob: if siblings.visits == 0 {
            0.0
        } else {
            edge.stats.visits as f32 / siblings.visits as f32
        },
        net_value,
    }
}

/// `child_value` is the net's value of the position behind one root edge, in
/// the searching agent's frame — the session has the net, this module does
/// not.
pub fn summary_to_proto(
    search_id: u64,
    summary: &SearchSummary,
    pv: &[Edge],
    solved: bool,
    config: &str,
    child_value: impl Fn(&BbAction) -> Option<f32>,
) -> ps::SearchReport {
    let siblings = Siblings::of(&summary.children);
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
        children: summary
            .children
            .iter()
            .map(|c| child_to_proto(c, &siblings, child_value(&c.action)))
            .collect(),
        pv: pv_steps,
        elapsed_ms: summary.elapsed.as_millis() as u64,
        budget: summary.budget.clone(),
        evaluator: summary.evaluator.clone(),
        config: config.to_string(),
        evaluator_value: summary.evaluator_value,
        solved,
        health: ps::SearchHealth {
            reuse: summary.reuse.outcome.label().to_string(),
            proc: summary.reuse.proc.clone(),
            n_actions: summary.reuse.n_actions,
            searches: summary.telemetry.searches,
            reused: summary.telemetry.reuse.total.reused,
            recomb_hits: summary.telemetry.recombination.hits,
            recomb_probes: summary.telemetry.recombination.probes,
            eq_checks: summary.telemetry.recombination.eq_checks,
            eq_rejects: summary.telemetry.recombination.eq_rejects,
        },
    }
}

pub fn node_to_proto(
    search_id: u64,
    path: Vec<ps::SearchEdge>,
    node: &NodeView,
    view: Option<Box<botbowl_web_proto::view::ViewState>>,
) -> ps::NodeExpansion {
    // Normalise over *every* child, before the truncation below.
    let siblings = Siblings::of(&node.children);
    let total = node.children.len();
    let children: Vec<ps::ChildReport> = node
        .children
        .iter()
        .take(ps::MAX_CHILDREN_PER_NODE)
        .map(|c| child_to_proto(c, &siblings, None))
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
