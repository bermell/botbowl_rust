//! The bot inspector: what the search looked at, and what it believed.
//!
//! Three views of one search, in increasing depth:
//!   * the **candidate list** — every root child with its visits, Q and prior;
//!   * the **principal variation** — the greedy line, clickable;
//!   * the **tree explorer** — one node at a time, walked on demand.
//!
//! The bot keeps exactly one tree (the most recent search's), so a report
//! from an earlier move can be read but not walked; `search_id` is what says
//! which, and the server refuses a stale one rather than answering about the
//! wrong position.

use botbowl_web_proto::msg::ClientMsg;
use botbowl_web_proto::search::{ChildReport, NodeStats, SearchEdge};
use leptos::prelude::*;

use crate::state::App;
use crate::ws;

#[component]
pub fn Inspector() -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="inspector" class:open=move || app.inspector_open.get()>
            <button class="drawer-handle" on:click=move |_| app.inspector_open.update(|o| *o = !*o)>
                {move || if app.inspector_open.get() { "▼ bot inspector" } else { "▲ bot inspector" }}
            </button>
            <Show when=move || app.inspector_open.get()>
                {move || match app.report.get() {
                    None => view! {
                        <div class="empty">
                            "No search to show. Play against the MCTS bot and its reasoning appears here after each move."
                        </div>
                    }
                        .into_any(),
                    Some(report) => {
                        let search_id = report.search_id;
                        view! {
                            <div class="inspector-body">
                                <Summary />
                                <Candidates />
                                <Pv search_id=search_id />
                                <Explorer search_id=search_id />
                            </div>
                        }
                            .into_any()
                    }
                }}
            </Show>
        </div>
    }
}

fn pct(x: f32) -> String {
    format!("{:.0}%", x * 100.0)
}

/// Q in the searching agent's frame, where ±1 is a touchdown. Positive is
/// good for the bot at every depth — the frame does not flip per ply.
fn q_text(stats: &NodeStats) -> String {
    match stats.q_display {
        Some(q) => format!("{q:+.3}"),
        None => "—".to_string(),
    }
}

#[component]
fn Summary() -> impl IntoView {
    let app = expect_context::<App>();
    move || {
        app.report.get().map(|r| {
            view! {
                <div class="summary">
                    <span class="chosen">{format!("played {}", r.chosen.describe())}</span>
                    <span>{format!("{:?} · {}", r.agent, r.budget)}</span>
                    <span>{format!("{} ms", r.elapsed_ms)}</span>
                    <span>
                        {match r.root_q_display {
                            Some(q) => format!("root Q {q:+.3}"),
                            None => "root Q —".to_string(),
                        }}
                    </span>
                    <span>{format!("{} visits", r.root_visits)}</span>
                    <span class="eval">{r.evaluator.clone()}</span>
                    {r.evaluator_value.map(|v| view! { <span class="nn">{format!("net value {v:+.3}")}</span> })}
                    {r.solved.then(|| view! { <span class="solved">"solved"</span> })}
                </div>
            }
        })
    }
}

#[component]
fn Candidates() -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="candidates">
            <h3>"Candidates"</h3>
            <table>
                <thead>
                    <tr>
                        <th>"action"</th>
                        <th>"visits"</th>
                        <th>"share"</th>
                        <th>"Q"</th>
                        <th>"prior"</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    {move || {
                        app.report
                            .get()
                            .map(|r| {
                                let chosen = r.chosen;
                                r.children
                                    .into_iter()
                                    .take(16)
                                    .map(|c: ChildReport| {
                                        let is_chosen = matches!(c.edge, SearchEdge::Player(a) if a == chosen);
                                        let square = match c.edge {
                                            SearchEdge::Player(a) => a.position(),
                                            SearchEdge::Chance { .. } => None,
                                        };
                                        view! {
                                            <tr
                                                class:chosen=is_chosen
                                                on:mouseenter=move |_| app.hover.set(square)
                                                on:mouseleave=move |_| app.hover.set(None)
                                            >
                                                <td class="action">{c.edge.describe()}</td>
                                                <td class="num">{c.stats.visits}</td>
                                                <td class="bar">
                                                    <span
                                                        class="fill"
                                                        style=format!("width: {:.1}%", c.visit_share * 100.0)
                                                    ></span>
                                                    <span class="bar-text">{pct(c.visit_share)}</span>
                                                </td>
                                                <td class="num">{q_text(&c.stats)}</td>
                                                <td class="num">
                                                    {c.prior.map(|p| format!("{p:.2}")).unwrap_or_default()}
                                                </td>
                                                <td class="flags">
                                                    {c.stats.solved.then_some("solved")}
                                                    {c.stats.terminal.then_some(" terminal")}
                                                </td>
                                            </tr>
                                        }
                                    })
                                    .collect_view()
                            })
                    }}
                </tbody>
            </table>
        </div>
    }
}

#[component]
fn Pv(search_id: u64) -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="pv">
            <h3>"Principal variation"</h3>
            <ol>
                {move || {
                    app.report
                        .get()
                        .map(|r| {
                            r.pv
                                .into_iter()
                                .map(|step| {
                                    let path = step.path.clone();
                                    let depth = path.len();
                                    view! {
                                        <li
                                            class:current=move || app.node_path.get() == path
                                            on:click={
                                                let path = step.path.clone();
                                                move |_| {
                                                    ws::send(
                                                        &ClientMsg::ExpandNode {
                                                            search_id,
                                                            path: path.clone(),
                                                            with_view: true,
                                                        },
                                                    )
                                                }
                                            }
                                        >
                                            <span class="ply">{depth}</span>
                                            <span class="edge">{step.edge.describe()}</span>
                                            <span class="q">{q_text(&step.stats)}</span>
                                            <span class="visits">{format!("{}v", step.stats.visits)}</span>
                                        </li>
                                    }
                                })
                                .collect_view()
                        })
                }}
            </ol>
            <p class="hint">"click a ply to open it below, and to preview its board on the pitch"</p>
        </div>
    }
}

#[component]
fn Explorer(search_id: u64) -> impl IntoView {
    let app = expect_context::<App>();

    let go = move |path: Vec<SearchEdge>| {
        ws::send(&ClientMsg::ExpandNode {
            search_id,
            path,
            with_view: true,
        });
    };

    // Keep the hypothetical board on the pitch in step with the explorer.
    Effect::new(move |_| {
        let board = app.node.get().and_then(|n| n.view.map(|v| *v));
        app.hypothetical.set(board);
    });

    view! {
        <div class="explorer">
            <h3>"Tree"</h3>
            <div class="breadcrumbs">
                <button on:click=move |_| go(Vec::new())>"root"</button>
                {move || {
                    let path = app.node_path.get();
                    path.iter()
                        .enumerate()
                        .map(|(i, edge)| {
                            let prefix: Vec<SearchEdge> = path[..=i].to_vec();
                            view! {
                                <button on:click={
                                    let prefix = prefix.clone();
                                    move |_| go(prefix.clone())
                                }>{edge.describe()}</button>
                            }
                        })
                        .collect_view()
                }}
                {move || {
                    (!app.node_path.get().is_empty() || app.hypothetical.get().is_some())
                        .then(|| {
                            view! {
                                <button
                                    class="back-to-live"
                                    on:click=move |_| {
                                        app.hypothetical.set(None);
                                        app.node.set(None);
                                        app.node_path.set(Vec::new());
                                    }
                                >
                                    "back to the live board"
                                </button>
                            }
                        })
                }}
            </div>
            {move || match app.node.get() {
                None => view! {
                    <p class="hint">
                        "Open the root, or a ply of the principal variation, to walk the search DAG one level at a time."
                    </p>
                }
                    .into_any(),
                Some(node) => {
                    let path = node.path.clone();
                    view! {
                        <div>
                            <div class="node-stats">
                                <span>{format!("{:?}", node.stats.player)}</span>
                                <span>{format!("depth {}", node.depth)}</span>
                                <span>{format!("{} visits", node.stats.visits)}</span>
                                <span>{format!("Q {}", q_text(&node.stats))}</span>
                                <span>{format!("{} parent(s)", node.n_parents)}</span>
                                {node.proc.clone().map(|p| view! { <span class="proc">{p}</span> })}
                                {node.stats.solved.then(|| view! { <span class="solved">"solved"</span> })}
                            </div>
                            <table class="children">
                                <tbody>
                                    {node
                                        .children
                                        .iter()
                                        .cloned()
                                        .map(|c| {
                                            let mut child_path = path.clone();
                                            child_path.push(c.edge.clone());
                                            view! {
                                                <tr on:click={
                                                    let child_path = child_path.clone();
                                                    move |_| go(child_path.clone())
                                                }>
                                                    <td class="action">{c.edge.describe()}</td>
                                                    <td class="num">{c.stats.visits}</td>
                                                    <td class="bar">
                                                        <span
                                                            class="fill"
                                                            style=format!("width: {:.1}%", c.visit_share * 100.0)
                                                        ></span>
                                                    </td>
                                                    <td class="num">{q_text(&c.stats)}</td>
                                                    <td class="num">
                                                        {c.prior.map(|p| format!("p {p:.2}")).unwrap_or_default()}
                                                    </td>
                                                </tr>
                                            }
                                        })
                                        .collect_view()}
                                </tbody>
                            </table>
                            {(node.children_omitted > 0)
                                .then(|| {
                                    view! {
                                        <p class="hint">
                                            {format!("{} more child(ren) not shown", node.children_omitted)}
                                        </p>
                                    }
                                })}
                        </div>
                    }
                        .into_any()
                }
            }}
        </div>
    }
}
