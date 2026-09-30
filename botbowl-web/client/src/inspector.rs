//! The inspector: the decision log, and what was behind any one decision.
//!
//! Left, the **decision log** — every action either side took, human or bot,
//! newest first. Right, the **detail** of the selected one (or the newest,
//! while following the game):
//!   * for every decision, the net's read of the position it was taken in —
//!     its value and its policy over the legal actions;
//!   * for an MCTS decision, the search root on top of that — every legal
//!     action with the net's probability, the search's prior, visits, Q, and
//!     the net's own value of the child position — then the principal
//!     variation and the tree explorer.
//!
//! Every record is a snapshot, so the root of any past decision stays
//! readable. Walking *below* the root needs the bot's tree, and each bot keeps
//! only its most recent one: the PV and explorer are offered for that search
//! alone, and the server refuses a stale `search_id` rather than answering
//! about the wrong position.

use botbowl_web_proto::action::{Action, TeamType};
use botbowl_web_proto::decision::{Decider, DecisionRecord, NetReadout};
use botbowl_web_proto::msg::ClientMsg;
use botbowl_web_proto::search::{NodeStats, PvStep, SearchEdge, SearchReport};
use leptos::prelude::*;

use crate::state::App;
use crate::ws;

/// The log renders at most this many rows; the rest are a count.
const LOG_ROWS: usize = 400;

#[component]
pub fn Inspector() -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="inspector" class:open=move || app.inspector_open.get()>
            <button class="drawer-handle" on:click=move |_| app.inspector_open.update(|o| *o = !*o)>
                {move || if app.inspector_open.get() { "▼ decisions" } else { "▲ decisions" }}
            </button>
            <Show when=move || app.inspector_open.get()>
                <div class="inspector-split">
                    <DecisionLog />
                    <Detail />
                </div>
            </Show>
        </div>
    }
}

fn pct(x: f32) -> String {
    format!("{:.0}%", x * 100.0)
}

/// Render a Home-centric value in `[-1, 1]` as the side it favours (plan 043).
///
/// A bare `+0.42` requires the reader to remember whose frame it is in, and the question people
/// actually ask of a value head is "who does it think scores next" — so name that side and keep
/// the magnitude as the confidence.
pub fn favours(value_home: f32) -> String {
    let team = if value_home >= 0.0 { "Home" } else { "Away" };
    format!("{team} {:.2}", value_home.abs())
}

/// Q in the searching agent's frame, where ±1 is a touchdown. Positive is
/// good for the bot at every depth — the frame does not flip per ply.
fn q_text(stats: &NodeStats) -> String {
    signed(stats.q_display)
}

fn signed(v: Option<f32>) -> String {
    v.map_or_else(|| "—".to_string(), |v| format!("{v:+.3}"))
}

fn team_tag(team: TeamType) -> &'static str {
    match team {
        TeamType::Home => "H",
        TeamType::Away => "A",
    }
}

fn who(by: &Decider) -> String {
    match by {
        Decider::Human => "you".into(),
        Decider::Bot { label } => label.split('[').next().unwrap_or(label).to_string(),
    }
}

/// The most-visited root child, which is *not* always the move played:
/// `MctsBot` picks by aggregated Q.
fn most_visited(report: &SearchReport) -> Option<Action> {
    report
        .children
        .iter()
        .max_by_key(|c| c.stats.visits)
        .and_then(|c| c.edge.action())
}

// ------------------------------------------------------------------- log

#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    All,
    Home,
    Away,
    Bots,
    Humans,
}

/// One log line — a few strings, so a long game does not clone every search
/// report each time a decision arrives.
#[derive(Clone, PartialEq)]
struct Row {
    index: u64,
    team: TeamType,
    turn: u8,
    who: String,
    action: String,
    badge: String,
    /// Something a debugger should look at: the net disliked the move, or
    /// the search played something other than its most-visited child.
    flag: Option<&'static str>,
}

impl Row {
    fn of(r: &DecisionRecord) -> Self {
        let net_rank = r.net.as_ref().and_then(|n| n.rank_of(r.action));
        let (badge, flag) = match &r.search {
            Some(s) => {
                let chosen = s.children.iter().find(|c| c.edge.action() == Some(r.action));
                let badge = format!(
                    "Q {} · {}v",
                    chosen.map_or_else(|| "—".to_string(), |c| q_text(&c.stats)),
                    s.root_visits
                );
                let flag = (most_visited(s) != Some(r.action)).then_some("≠ most visited");
                (badge, flag)
            }
            None => match net_rank {
                Some((rank, p)) => (
                    format!("net #{rank} · {}", pct(p)),
                    (rank > 3 && r.n_legal > 3).then_some("net disliked"),
                ),
                None => (String::new(), None),
            },
        };
        Row {
            index: r.index,
            team: r.team,
            turn: r.turn,
            who: who(&r.by),
            action: r.action.describe(),
            badge,
            flag,
        }
    }
}

#[component]
fn DecisionLog() -> impl IntoView {
    let app = expect_context::<App>();
    let filter = RwSignal::new(Filter::All);
    let hide_forced = RwSignal::new(true);
    // The row the detail pane is on, resolved once for the whole list.
    let shown = Memo::new(move |_| {
        app.selected
            .get()
            .or_else(|| app.decisions.with(|d| d.last().map(|r| r.index)))
    });

    let rows = Memo::new(move |_| {
        let f = filter.get();
        let hide = hide_forced.get();
        app.decisions.with(|d| {
            let matching: Vec<&DecisionRecord> = d
                .iter()
                .rev()
                .filter(|r| !hide || r.n_legal > 1)
                .filter(|r| match f {
                    Filter::All => true,
                    Filter::Home => r.team == TeamType::Home,
                    Filter::Away => r.team == TeamType::Away,
                    Filter::Bots => matches!(r.by, Decider::Bot { .. }),
                    Filter::Humans => r.by == Decider::Human,
                })
                .collect();
            let total = matching.len();
            let rows: Vec<Row> = matching.into_iter().take(LOG_ROWS).map(Row::of).collect();
            (rows, total)
        })
    });

    let filter_button = move |f: Filter, label: &'static str| {
        view! {
            <button class="pill" class:on=move || filter.get() == f on:click=move |_| filter.set(f)>
                {label}
            </button>
        }
    };

    view! {
        <div class="decision-log">
            <div class="log-controls">
                <label class="check">
                    <input
                        type="checkbox"
                        prop:checked=move || app.selected.get().is_none()
                        on:change=move |ev| {
                            if event_target_checked(&ev) {
                                app.selected.set(None);
                                app.back_to_live();
                            } else {
                                let last = app.decisions.with(|d| d.last().map(|r| r.index));
                                app.selected.set(last);
                            }
                        }
                    />
                    "follow"
                </label>
                <label class="check" title="hide decisions with only one legal action">
                    <input
                        type="checkbox"
                        prop:checked=move || hide_forced.get()
                        on:change=move |ev| hide_forced.set(event_target_checked(&ev))
                    />
                    "hide forced"
                </label>
                {filter_button(Filter::All, "all")}
                {filter_button(Filter::Home, "home")}
                {filter_button(Filter::Away, "away")}
                {filter_button(Filter::Bots, "bots")}
                {filter_button(Filter::Humans, "humans")}
            </div>
            <div class="log-rows">
                <For
                    each=move || rows.get().0
                    key=|r| (r.index, r.action.clone())
                    children=move |r: Row| {
                        let index = r.index;
                        view! {
                            <div
                                class=format!("log-row team-{}", team_tag(r.team))
                                class:selected=move || shown.get() == Some(index)
                                on:click=move |_| app.selected.set(Some(index))
                                on:dblclick=move |_| {
                                    app.selected.set(Some(index));
                                    ws::send(&ClientMsg::ShowDecision { index });
                                }
                            >
                                <span class="idx">{format!("#{index}")}</span>
                                <span class="team">{team_tag(r.team)}</span>
                                <span class="turn">{format!("t{}", r.turn)}</span>
                                <span class="who">{r.who.clone()}</span>
                                <span class="act" title=r.action.clone()>{r.action.clone()}</span>
                                <span class="badge">{r.badge.clone()}</span>
                                {r.flag.map(|f| view! { <span class="flag">{f}</span> })}
                            </div>
                        }
                    }
                />
                {move || {
                    let (rows, total) = rows.get();
                    if total == 0 {
                        Some(view! { <p class="hint">"No decisions yet."</p> }.into_any())
                    } else if total > rows.len() {
                        Some(
                            view! { <p class="hint">{format!("… and {} older", total - rows.len())}</p> }
                                .into_any(),
                        )
                    } else {
                        None
                    }
                }}
            </div>
            <p class="hint">"click to inspect · double-click to put its board on the pitch"</p>
        </div>
    }
}

// ---------------------------------------------------------------- detail

#[component]
fn Detail() -> impl IntoView {
    let app = expect_context::<App>();
    // A memo, so a decision arriving while an older one is selected does not
    // rebuild the pane (and reset its sort and scroll) — only a change of the
    // record shown does.
    let inspected = Memo::new(move |_| app.inspected());
    move || {
        let Some(record) = inspected.get() else {
            return view! {
                <div class="detail empty">
                    "Nothing decided yet. Every move either side makes appears on the left; pick one to see the net's policy and, for an MCTS move, the search behind it."
                </div>
            }
            .into_any();
        };
        let index = record.index;
        let header = format!(
            "#{index} · {:?} ({}) · half {} turn {} · {} · {} legal",
            record.team,
            who(&record.by),
            record.half,
            record.turn,
            record.proc,
            record.n_legal
        );
        let played = record.action.describe();
        let body = match record.search.clone() {
            Some(report) => {
                let walkable = app.walkable(report.agent, report.search_id);
                view! { <SearchDetail report=*report net=record.net.clone() played=record.action walkable=walkable /> }
                    .into_any()
            }
            None => match record.net.clone() {
                Some(net) => view! { <NetDetail net=net played=record.action /> }.into_any(),
                None => view! {
                    <p class="hint">"No net is seated in this game, so there is nothing to read out for this move."</p>
                }
                .into_any(),
            },
        };
        view! {
            <div class="detail">
                <div class="detail-head">
                    <span class="played">{format!("played {played}")}</span>
                    <span class="meta">{header}</span>
                    <button on:click=move |_| ws::send(&ClientMsg::ShowDecision { index })>"show its board"</button>
                    {move || {
                        (app.board_of.get() == Some(index))
                            .then(|| view! { <button class="warn" on:click=move |_| app.back_to_live()>"back to live"</button> })
                    }}
                </div>
                {body}
            </div>
        }
        .into_any()
    }
}

/// The net's view of a decision no search was run for.
#[component]
fn NetDetail(net: NetReadout, played: Action) -> impl IntoView {
    let app = expect_context::<App>();
    let rank = net.rank_of(played);
    view! {
        <div class="net-detail">
            <div class="summary">
                <span class="nn">{format!("net favours {}", favours(net.value_home))}</span>
                <span>{net.model.clone()}</span>
                <span>
                    {match rank {
                        Some((r, p)) => format!("played the net's #{r} choice ({})", pct(p)),
                        None => "played an action the net was not asked about".to_string(),
                    }}
                </span>
            </div>
            <h3>"Net policy over the legal actions"</h3>
            <table>
                <thead>
                    <tr>
                        <th>"#"</th>
                        <th>"action"</th>
                        <th>"net p"</th>
                    </tr>
                </thead>
                <tbody>
                    {net
                        .priors
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            let square = p.action.position();
                            view! {
                                <tr
                                    class:chosen=p.action == played
                                    on:mouseenter=move |_| app.hover.set(square)
                                    on:mouseleave=move |_| app.hover.set(None)
                                >
                                    <td class="num">{i + 1}</td>
                                    <td class="action">{p.action.describe()}</td>
                                    <td class="bar">
                                        <span class="fill" style=format!("width: {:.1}%", p.prob * 100.0)></span>
                                        <span class="bar-text">{format!("{:.1}%", p.prob * 100.0)}</span>
                                    </td>
                                </tr>
                            }
                        })
                        .collect_view()}
                </tbody>
            </table>
        </div>
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortBy {
    Visits,
    Q,
    Prior,
    NetP,
    NetV,
}

/// One legal action at a searched root: the search's view of it, if the
/// search kept it, and the net's.
#[derive(Clone)]
struct RootRow {
    action: Action,
    net_p: Option<f32>,
    prior: Option<f32>,
    prior_raw: Option<f32>,
    visits: Option<u32>,
    visit_prob: f32,
    visit_share: f32,
    q: Option<f32>,
    net_v: Option<f32>,
    solved: bool,
    terminal: bool,
}

fn root_rows(report: &SearchReport, net: Option<&NetReadout>) -> Vec<RootRow> {
    let mut rows: Vec<RootRow> = report
        .children
        .iter()
        .filter_map(|c| {
            let action = c.edge.action()?;
            Some(RootRow {
                action,
                net_p: net.and_then(|n| n.prob_of(action)),
                prior: c.prior_share,
                prior_raw: c.prior,
                visits: Some(c.stats.visits),
                visit_prob: c.visit_prob,
                visit_share: c.visit_share,
                q: c.stats.q_display,
                net_v: c.net_value,
                solved: c.stats.solved,
                terminal: c.stats.terminal,
            })
        })
        .collect();
    // Legal actions the search never considered: pruned before expansion.
    if let Some(net) = net {
        for p in &net.priors {
            if !rows.iter().any(|r| r.action == p.action) {
                rows.push(RootRow {
                    action: p.action,
                    net_p: Some(p.prob),
                    prior: None,
                    prior_raw: None,
                    visits: None,
                    visit_prob: 0.0,
                    visit_share: 0.0,
                    q: None,
                    net_v: None,
                    solved: false,
                    terminal: false,
                });
            }
        }
    }
    rows
}

fn sort_rows(rows: &mut [RootRow], by: SortBy) {
    let key = |r: &RootRow| -> f32 {
        match by {
            SortBy::Visits => r.visits.map_or(-1.0, |v| v as f32),
            SortBy::Q => r.q.unwrap_or(f32::NEG_INFINITY),
            SortBy::Prior => r.prior.unwrap_or(-1.0),
            SortBy::NetP => r.net_p.unwrap_or(-1.0),
            SortBy::NetV => r.net_v.unwrap_or(f32::NEG_INFINITY),
        }
    };
    rows.sort_by(|a, b| key(b).total_cmp(&key(a)));
}

#[component]
fn SearchDetail(report: SearchReport, net: Option<NetReadout>, played: Action, walkable: bool) -> impl IntoView {
    let app = expect_context::<App>();
    let sort = RwSignal::new(SortBy::Visits);
    let search_id = report.search_id;
    let rows = root_rows(&report, net.as_ref());
    let pruned = rows.iter().filter(|r| r.visits.is_none()).count();
    let top_visited = most_visited(&report);
    let net_top = net.as_ref().and_then(|n| n.priors.first().map(|p| p.action));
    let played_rank = net.as_ref().and_then(|n| n.rank_of(played));

    let header = move |by: SortBy, label: &'static str, title: &'static str| {
        view! {
            <th class="sortable" class:on=move || sort.get() == by title=title on:click=move |_| sort.set(by)>
                {label}
            </th>
        }
    };

    let summary = {
        let r = report.clone();
        view! {
            <div class="summary">
                <span>{format!("{:?} · {}", r.agent, r.budget)}</span>
                <span>{format!("{} ms", r.elapsed_ms)}</span>
                <span>{format!("root Q {}", signed(r.root_q_display))}</span>
                <span>{format!("{} root visits", r.root_visits)}</span>
                {r.evaluator_value
                    .map(|v| {
                        // `evaluator_value` is in the *searching agent's* frame; put it back
                        // into Home's so the label means the same thing everywhere.
                        let home = match r.agent {
                            TeamType::Home => v,
                            TeamType::Away => -v,
                        };
                        view! { <span class="nn">{format!("net favours {}", favours(home))}</span> }
                    })}
                {played_rank.map(|(rank, p)| view! { <span>{format!("played the net's #{rank} ({})", pct(p))}</span> })}
                {(top_visited != Some(played))
                    .then(|| view! { <span class="warn">"played ≠ most visited (the bot picks by Q)"</span> })}
                {(net_top.is_some() && net_top != Some(played))
                    .then(|| view! { <span class="warn">"search overrode the net's favourite"</span> })}
                {r.solved.then(|| view! { <span class="solved">"solved"</span> })}
                <span class="config" title="search configuration">{format!("{} · {}", r.evaluator, r.config)}</span>
            </div>
        }
    };

    view! {
        <div class="search-detail">
            {summary}
            <div class="search-grid">
                <div class="root-table">
                    <h3>
                        "Root — every legal action"
                        {(pruned > 0).then(|| format!(" ({pruned} pruned before search)"))}
                    </h3>
                    <table>
                        <thead>
                            <tr>
                                <th>"action"</th>
                                {header(SortBy::NetP, "net p", "the net's softmax over all legal actions")}
                                {header(SortBy::Prior, "prior", "the PUCT prior the search used, as a share of the searched actions")}
                                {header(SortBy::Visits, "visits", "descents through this child")}
                                <th title="share of the root's child visits — a visit-count policy target">"π"</th>
                                {header(SortBy::Q, "Q", "search value, the bot's frame, ±1 = a touchdown")}
                                {header(SortBy::NetV, "net V", "the net's value of the child position before search, same frame")}
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {move || {
                                let mut rows = rows.clone();
                                sort_rows(&mut rows, sort.get());
                                rows.into_iter()
                                    .map(|r| {
                                        let square = r.action.position();
                                        let gap = r.q.zip(r.net_v).map(|(q, v)| q - v);
                                        view! {
                                            <tr
                                                class:chosen=r.action == played
                                                class:pruned=r.visits.is_none()
                                                on:mouseenter=move |_| app.hover.set(square)
                                                on:mouseleave=move |_| app.hover.set(None)
                                            >
                                                <td class="action" title=r.action.describe()>{r.action.describe()}</td>
                                                <td class="num">{r.net_p.map(|p| format!("{:.1}%", p * 100.0)).unwrap_or_default()}</td>
                                                <td class="num" title=r.prior_raw.map(|p| format!("raw PUCT prior {p:.3}")).unwrap_or_default()>
                                                    {r.prior.map(|p| format!("{:.1}%", p * 100.0)).unwrap_or_default()}
                                                </td>
                                                <td class="bar">
                                                    <span class="fill" style=format!("width: {:.1}%", r.visit_share * 100.0)></span>
                                                    <span class="bar-text">{r.visits.map(|v| v.to_string()).unwrap_or_else(|| "pruned".into())}</span>
                                                </td>
                                                <td class="num">{r.visits.map(|_| pct(r.visit_prob)).unwrap_or_default()}</td>
                                                <td class="num">{r.visits.map(|_| signed(r.q)).unwrap_or_default()}</td>
                                                <td class="num" title=gap.map(|g| format!("Q − net V = {g:+.3}")).unwrap_or_default()>
                                                    {r.net_v.map(|v| format!("{v:+.3}")).unwrap_or_default()}
                                                </td>
                                                <td class="flags">
                                                    {(Some(r.action) == top_visited).then_some("most visited ")}
                                                    {r.solved.then_some("solved ")}
                                                    {r.terminal.then_some("terminal")}
                                                </td>
                                            </tr>
                                        }
                                    })
                                    .collect_view()
                            }}
                        </tbody>
                    </table>
                </div>
                <Health health=report.health.clone() />
                {if walkable {
                    view! {
                        <Pv search_id=search_id pv=report.pv.clone() />
                        <Explorer search_id=search_id />
                    }
                        .into_any()
                } else {
                    view! {
                        <div class="pv">
                            <h3>"Principal variation"</h3>
                            <ol>
                                {report
                                    .pv
                                    .iter()
                                    .map(|step| {
                                        view! {
                                            <li class="stale">
                                                <span class="ply">{step.path.len()}</span>
                                                <span class="edge">{step.edge.describe()}</span>
                                                <span class="q">{q_text(&step.stats)}</span>
                                                <span class="visits">{format!("{}v", step.stats.visits)}</span>
                                            </li>
                                        }
                                    })
                                    .collect_view()}
                            </ol>
                            <p class="hint">
                                "This bot has searched again since, and it keeps only its latest tree — the root and PV above are a snapshot, but the tree can no longer be walked."
                            </p>
                        </div>
                    }
                        .into_any()
                }}
            </div>
        </div>
    }
}

/// Search health (plan 043): did this decision start from the tree the last one built, and is
/// recombination earning what it costs.
///
/// Both are things you can only see over time, so each line pairs *this decision's* answer with
/// the rate so far. A reuse outcome of `anchor_miss` at a turn boundary is expected; the same
/// outcome mid-turn, or a `lookup_miss`, is not.
#[component]
fn Health(health: botbowl_web_proto::search::SearchHealth) -> impl IntoView {
    let h = health;
    let reuse_class = if h.reuse == "reused" { "ok" } else { "warn" };
    let rate = |v: Option<f32>| v.map_or_else(|| "—".to_string(), pct);
    view! {
        <div class="health">
            <h3>"Search health"</h3>
            <div class="health-row">
                <span class="k">"tree reuse"</span>
                <span class=format!("v {reuse_class}")>{h.reuse.clone()}</span>
                <span class="note">
                    {format!("{} of {} decisions ({})", h.reused, h.searches, rate(h.reuse_rate()))}
                </span>
            </div>
            <div class="health-row">
                <span class="k">"decision"</span>
                <span class="v">{h.proc.clone().unwrap_or_else(|| "—".to_string())}</span>
                <span class="note">{format!("{} actions searched", h.n_actions)}</span>
            </div>
            <div class="health-row">
                <span class="k">"recombination"</span>
                <span class="v">{rate(h.recomb_hit_rate())}</span>
                <span class="note">{format!("{} hits / {} probes", h.recomb_hits, h.recomb_probes)}</span>
            </div>
            <div class="health-row">
                <span class="k">"wasted compares"</span>
                <span class="v">{rate(h.eq_reject_rate())}</span>
                <span class="note">{format!("{} of {} state comparisons", h.eq_rejects, h.eq_checks)}</span>
            </div>
        </div>
    }
}

#[component]
fn Pv(search_id: u64, pv: Vec<PvStep>) -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="pv">
            <h3>"Principal variation"</h3>
            <ol>
                {pv
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
                    .collect_view()}
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

    // Put the node's board on the pitch when one arrives. Never *clear* it
    // from here: a past decision's board may be the one showing.
    Effect::new(move |_| {
        if let Some(board) = app.node.get().and_then(|n| n.view.map(|v| *v)) {
            app.board_of.set(None);
            app.hypothetical.set(Some(board));
        }
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
                                <button class="back-to-live" on:click=move |_| app.back_to_live()>
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
                                <thead>
                                    <tr>
                                        <th>"edge"</th>
                                        <th>"visits"</th>
                                        <th>"π"</th>
                                        <th>"Q"</th>
                                        <th>"prior"</th>
                                    </tr>
                                </thead>
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
                                                    <td class="bar">
                                                        <span
                                                            class="fill"
                                                            style=format!("width: {:.1}%", c.visit_share * 100.0)
                                                        ></span>
                                                        <span class="bar-text">{c.stats.visits}</span>
                                                    </td>
                                                    <td class="num">{pct(c.visit_prob)}</td>
                                                    <td class="num">{q_text(&c.stats)}</td>
                                                    <td class="num">
                                                        {c.prior_share.map(|p| format!("{:.1}%", p * 100.0)).unwrap_or_default()}
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
