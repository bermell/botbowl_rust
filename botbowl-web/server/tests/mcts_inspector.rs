//! Phase 2 of plan 034: the MCTS opponent reports its search, and the client
//! can walk into the tree that produced it.
//!
//! Kept to a handful of moves at a tiny budget on the committed `tiny.onnx`
//! fixture — this is a wiring test, not a search-quality one. What it pins:
//!   * a bot's `Decision` carries a `SearchReport` whose chosen action is the
//!     action actually played, and the net's read-out of the same position;
//!   * a human's `Decision` carries the net's policy over their legal actions;
//!   * root children carry visits / Q / priors and a usable heat share;
//!   * the principal variation's `path` values are accepted by `ExpandNode`,
//!     which is the whole contract behind the tree explorer;
//!   * exploring does not disturb the tree (the same node reads the same
//!     twice).

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use botbowl_web_proto::action::TeamType;
use botbowl_web_proto::decision::{Decider, NetReadout};
use botbowl_web_proto::msg::{
    BoardSpec, BotSpec, Budget, ClientMsg, GameSpec, MctsSpec, Seat, ServerMsg, StartFrom, StepMode,
};
use botbowl_web_proto::search::{SearchEdge, SearchReport};
use botbowl_web_proto::view::ViewState;
use botbowl_web_server::{compiled_capacity, router, AppState};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

type Socket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn serve() -> SocketAddr {
    let app = Arc::new(AppState {
        capacity: compiled_capacity(),
        // The committed test net: untagged, so it is offered for every board.
        models_dir: concat!(env!("CARGO_MANIFEST_DIR"), "/../../botbowl-nn/tests/fixtures").into(),
        recordings_dir: std::env::temp_dir().join("botbowl-web-test"),
        model_cache: Default::default(),
        server: "test".into(),
        // The tests drive the pacing themselves; they start from the original free-running mode.
        opts: botbowl_web_server::PlayOptions {
            initial_step_mode: botbowl_web_proto::msg::StepMode::Run,
            ..Default::default()
        },
    });
    let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router(app, None, None)).await;
    });
    addr
}

async fn send(socket: &mut Socket, msg: ClientMsg) {
    socket
        .send(Message::Text(serde_json::to_string(&msg).unwrap()))
        .await
        .unwrap();
}

async fn recv(socket: &mut Socket) -> ServerMsg {
    let frame = tokio::time::timeout(Duration::from_secs(180), socket.next())
        .await
        .expect("server went quiet")
        .expect("socket closed")
        .expect("frame");
    match frame {
        Message::Text(text) => serde_json::from_str(&text).unwrap(),
        other => panic!("unexpected frame {other:?}"),
    }
}

/// Deliberately tiny: this is a wiring test and it runs in a debug build,
/// where the search is orders of magnitude slower.
fn tiny_mcts() -> BotSpec {
    BotSpec::Mcts(MctsSpec {
        budget: Budget::Iterations(60),
        workers: Some(1),
        model: "tiny.onnx".into(),
        ..Default::default()
    })
}

/// A read-out is a distribution over the legal actions, and a value in range.
fn assert_readout(net: &NetReadout, played: botbowl_web_proto::action::Action) {
    assert!((-1.0..=1.0).contains(&net.value_home), "value {}", net.value_home);
    assert_eq!(net.model, "tiny.onnx");
    let total: f32 = net.priors.iter().map(|p| p.prob).sum();
    assert!((total - 1.0).abs() < 1e-3, "net priors sum to {total}");
    assert!(
        net.priors.windows(2).all(|w| w[0].prob >= w[1].prob),
        "net priors not sorted"
    );
    assert!(
        net.rank_of(played).is_some(),
        "the played action {played:?} is not among the legal ones"
    );
}

fn first_legal(view: &ViewState) -> botbowl_web_proto::action::Action {
    view.squares
        .iter()
        .flat_map(|sq| {
            sq.actions
                .iter()
                .map(move |at| botbowl_web_proto::action::Action::Positional(*at, sq.pos))
        })
        .chain(
            view.simple_actions
                .iter()
                .map(|a| botbowl_web_proto::action::Action::Simple(a.at)),
        )
        .next()
        .expect("the view offered no action")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_mcts_opponent_reports_the_search_behind_each_move() {
    let capacity = compiled_capacity();
    let board = BoardSpec::new(14, 7, 4);
    if board.validate(capacity).is_err() {
        eprintln!("skipped: capacity too small for 14x7");
        return;
    }
    let addr = serve().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .unwrap();
    let _lobby = recv(&mut socket).await;

    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board,
            home: Seat::Human,
            away: Seat::Bot(tiny_mcts()),
            seed: Some(5),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
            no_natural_one_turn: true,
        }),
    )
    .await;

    let mut reports: Vec<SearchReport> = Vec::new();
    let mut saw_thinking = false;
    let mut saw_net = false;
    let mut decisions = 0usize;
    // Collect a few bot decisions, then keep reading until the server goes
    // quiet (the human is on the clock). Only *then* is the cached tree the
    // one the last report describes: every search re-roots the single tree
    // the bot keeps, so inspecting has to happen while the server is idle.
    let mut idle = false;
    while reports.len() < 3 || !idle {
        match recv(&mut socket).await {
            ServerMsg::BotThinking { team, budget } => {
                saw_thinking = true;
                assert_eq!(team, TeamType::Away);
                assert!(budget.contains("mcts"), "{budget}");
            }
            ServerMsg::Decision(record) => {
                let action = record.action;
                assert_eq!(record.index as usize, decisions, "decision indices are consecutive");
                decisions += 1;
                assert!(record.n_legal > 0);
                let net = record
                    .net
                    .as_ref()
                    .expect("a net is seated, so every decision is read out");
                assert_readout(net, action);
                if record.by == Decider::Human {
                    assert_eq!(record.team, TeamType::Home);
                    assert!(record.search.is_none(), "a human decision has no search");
                    continue;
                }
                assert_eq!(record.team, TeamType::Away);
                // A heuristic bot answers its kickoff setup from a formation, without a
                // search (plan 047); those decisions carry no report.
                let Some(report) = record.search else { continue };
                let report = *report;
                assert_eq!(report.chosen, action, "the report must be about the move played");
                assert_eq!(report.agent, TeamType::Away);
                assert_eq!(report.evaluator, "nn", "the web bot always searches on the net");
                assert_eq!(report.budget, "60 iterations");
                assert!(
                    report.evaluator_value.is_some(),
                    "the NN evaluator reports its root value"
                );
                assert!(report.config.contains("1 worker(s)"), "{}", report.config);
                // Plan 043: search health rides along with every report.
                let h = &report.health;
                assert!(
                    matches!(
                        h.reuse.as_str(),
                        "reused" | "no_cache" | "anchor_miss" | "lookup_miss" | "no_path"
                    ),
                    "unexpected reuse outcome {:?}",
                    h.reuse
                );
                assert!(h.proc.is_some(), "a decision state always has a procedure on top");
                assert!(h.n_actions > 0, "a decision offers at least one legal action");
                assert!(h.searches > 0, "this decision is counted");
                assert!(h.reused <= h.searches);
                assert!(h.recomb_probes > 0, "a search must probe the registry");
                assert!(h.recomb_hits <= h.recomb_probes, "hits cannot exceed probes");
                assert!(
                    h.eq_rejects <= h.eq_checks,
                    "a rejection is a comparison that returned false"
                );
                assert!(!report.children.is_empty(), "a searched root has children");
                assert!(
                    report.children.iter().any(|c| c.stats.visits > 0),
                    "no child was ever visited"
                );
                assert!(
                    report
                        .children
                        .iter()
                        .any(|c| matches!(c.edge, SearchEdge::Player(a) if a == action)),
                    "the played action must be one of the root children"
                );
                // Sorted by visits, and the heat share is usable as an alpha.
                let visits: Vec<u32> = report.children.iter().map(|c| c.stats.visits).collect();
                assert!(
                    visits.windows(2).all(|w| w[0] >= w[1]),
                    "children not sorted: {visits:?}"
                );
                for child in &report.children {
                    assert!((0.0..=1.0).contains(&child.visit_share), "{:?}", child.visit_share);
                    assert!((0.0..=1.0).contains(&child.visit_prob), "{:?}", child.visit_prob);
                    if let SearchEdge::Player(_) = child.edge {
                        assert!(child.prior.is_some(), "a player edge carries its PUCT prior");
                        assert!(child.prior_share.is_some(), "and its share of the policy");
                    }
                    // The value head's own read of every visited decision child.
                    if child.stats.visits > 0
                        && matches!(
                            child.stats.player,
                            botbowl_web_proto::search::NodePlayer::Home | botbowl_web_proto::search::NodePlayer::Away
                        )
                    {
                        let v = child.net_value.expect("a visited decision child has a net value");
                        assert!((-1.0..=1.0).contains(&v), "{v}");
                    }
                }
                let prior_total: f32 = report.children.iter().filter_map(|c| c.prior_share).sum();
                assert!((prior_total - 1.0).abs() < 1e-3, "prior shares sum to {prior_total}");
                if report.children.iter().any(|c| c.stats.visits > 0) {
                    let visit_total: f32 = report.children.iter().map(|c| c.visit_prob).sum();
                    assert!((visit_total - 1.0).abs() < 1e-3, "visit probs sum to {visit_total}");
                }

                // Every Q in one report must be in **one** frame — the
                // searching agent's — or the inspector reads as a sign flip
                // at every ply and the bot looks like it picked the worst
                // move. Away searched here, so `q_display` is `-q_home/1000`
                // at the root, at every child, and at every PV step alike.
                let expected = |q_home: Option<i64>| q_home.map(|q| -(q as f32) / 1000.0);
                assert_eq!(report.root_q_display, expected(report.root_q_home));
                for child in &report.children {
                    assert_eq!(
                        child.stats.q_display,
                        expected(child.stats.q_home),
                        "child {:?} is in a different frame from the root",
                        child.edge
                    );
                }
                for step in &report.pv {
                    assert_eq!(
                        step.stats.q_display,
                        expected(step.stats.q_home),
                        "PV step {:?} is in a different frame from the root",
                        step.edge
                    );
                }
                reports.push(report);
            }
            ServerMsg::View(view) => {
                if view.scoreboard.game_over {
                    break;
                }
                if view.to_act == Some(TeamType::Home) && !view.bot_thinking {
                    if reports.len() >= 3 {
                        idle = true;
                    } else {
                        send(&mut socket, ClientMsg::Act(first_legal(&view))).await;
                    }
                }
            }
            // Plan 043: the live read-out of the current position.
            ServerMsg::Net(net) => {
                saw_net = true;
                assert!((-1.0..=1.0).contains(&net.value_home), "{}", net.value_home);
            }
            ServerMsg::Error(e) => panic!("server error: {e}"),
            _ => {}
        }
    }
    assert!(saw_thinking, "the UI needs a spinner signal before a search");
    assert!(saw_net, "a seated net reads out every board");

    // The principal variation is walkable: each step's `path` is exactly what
    // `ExpandNode` takes.
    // Every logged decision can put its own board back on the pitch.
    send(&mut socket, ClientMsg::ShowDecision { index: 0 }).await;
    loop {
        match recv(&mut socket).await {
            ServerMsg::DecisionBoard { index, view } => {
                assert_eq!(index, 0);
                assert_eq!(
                    view.scoreboard.half, 0,
                    "the first decision is the coin toss, before any half"
                );
                break;
            }
            ServerMsg::Error(e) => panic!("ShowDecision failed: {e}"),
            _ => {}
        }
    }
    send(
        &mut socket,
        ClientMsg::ShowDecision {
            index: decisions as u64,
        },
    )
    .await;
    loop {
        match recv(&mut socket).await {
            ServerMsg::Error(e) => {
                assert!(e.contains("no decision"), "{e}");
                break;
            }
            ServerMsg::DecisionBoard { index, .. } => panic!("decision {index} does not exist yet"),
            _ => {}
        }
    }

    let last = reports.last().unwrap();
    assert!(!last.pv.is_empty(), "a searched tree has a principal variation");
    assert_eq!(last.pv[0].path.len(), 1, "the first PV step is one edge from the root");
    // The PV must follow the search, not wander into a child that was never
    // visited. (It used to: unscored children sorted *first* for an Away
    // agent, so the line was two plies of `0 visits`.)
    if last.children.iter().any(|c| c.stats.visits > 0) {
        assert!(
            last.pv[0].stats.visits > 0,
            "the PV opened on an unvisited child: {:?}",
            last.pv[0]
        );
        assert!(
            last.pv[0].stats.q_home.is_some(),
            "the PV opened on an unscored child: {:?}",
            last.pv[0]
        );
    }

    for step in &last.pv {
        send(
            &mut socket,
            ClientMsg::ExpandNode {
                search_id: last.search_id,
                path: step.path.clone(),
                with_view: false,
            },
        )
        .await;
        loop {
            match recv(&mut socket).await {
                ServerMsg::Node(node) => {
                    assert_eq!(node.path, step.path, "the answer echoes the request");
                    assert_eq!(node.stats, step.stats, "the PV step and the node must agree");
                    assert!(node.children.len() <= botbowl_web_proto::search::MAX_CHILDREN_PER_NODE);
                    assert!(node.depth >= step.path.len() - 1);
                    break;
                }
                ServerMsg::Error(e) => panic!("ExpandNode failed for {:?}: {e}", step.path),
                _ => {}
            }
        }
    }

    // The root itself, twice, with the board attached: inspection must be
    // inert.
    let mut seen = Vec::new();
    for _ in 0..2 {
        send(
            &mut socket,
            ClientMsg::ExpandNode {
                search_id: last.search_id,
                path: Vec::new(),
                with_view: true,
            },
        )
        .await;
        loop {
            if let ServerMsg::Node(node) = recv(&mut socket).await {
                assert!(node.view.is_some(), "StoreState keeps the board on every node");
                seen.push(node);
                break;
            }
        }
    }
    assert_eq!(seen[0].stats, seen[1].stats, "walking the tree changed it");
    assert_eq!(seen[0].children, seen[1].children);

    // A path that leaves the materialised DAG is an error, not a panic.
    send(
        &mut socket,
        ClientMsg::ExpandNode {
            search_id: last.search_id,
            path: vec![SearchEdge::Player(botbowl_web_proto::action::Action::Simple(
                botbowl_web_proto::action::SimpleAT::KickoffAimMiddle,
            ))],
            with_view: false,
        },
    )
    .await;
    loop {
        match recv(&mut socket).await {
            ServerMsg::Error(e) => {
                assert!(e.contains("cached search tree"), "{e}");
                break;
            }
            ServerMsg::Node(n) => panic!("expected a miss, got {n:?}"),
            _ => {}
        }
    }

    // A stale search id is refused with an explanation rather than answered
    // about the wrong position.
    send(
        &mut socket,
        ClientMsg::ExpandNode {
            search_id: last.search_id.saturating_sub(1),
            path: Vec::new(),
            with_view: false,
        },
    )
    .await;
    loop {
        match recv(&mut socket).await {
            ServerMsg::Error(e) => {
                assert!(e.contains("superseded"), "{e}");
                break;
            }
            ServerMsg::Node(n) => panic!("a stale search must not be answered: {n:?}"),
            _ => {}
        }
    }
}

/// Two bots, one game: each keeps its own tree, so the latest search of
/// *either* side can be walked — and a bot's older search cannot.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_mcts_bots_each_keep_a_walkable_tree() {
    let capacity = compiled_capacity();
    let board = BoardSpec::new(14, 7, 4);
    if board.validate(capacity).is_err() {
        eprintln!("skipped: capacity too small for 14x7");
        return;
    }
    let addr = serve().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .unwrap();
    let _lobby = recv(&mut socket).await;

    // Manual pacing, so the session stops where we tell it to and both trees
    // are still the ones the reports describe when we walk them.
    send(&mut socket, ClientMsg::SetStepMode(StepMode::Manual)).await;
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board,
            home: Seat::Bot(tiny_mcts()),
            away: Seat::Bot(tiny_mcts()),
            seed: Some(11),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
            no_natural_one_turn: true,
        }),
    )
    .await;

    let mut latest: [Option<u64>; 2] = [None, None];
    let mut older: Option<u64> = None;
    loop {
        match recv(&mut socket).await {
            ServerMsg::View(view) => {
                assert!(view.humans.is_empty(), "nobody is seated from the browser");
                if view.scoreboard.game_over {
                    panic!("the game ended before both bots had searched");
                }
                if latest.iter().all(Option::is_some) && older.is_some() {
                    break;
                }
                if view.paused {
                    send(&mut socket, ClientMsg::StepOnce).await;
                }
            }
            ServerMsg::Decision(record) => {
                assert!(matches!(record.by, Decider::Bot { .. }));
                let report = record.search.expect("both bots search");
                let i = usize::from(record.team == TeamType::Away);
                if let Some(previous) = latest[i].replace(report.search_id) {
                    older = Some(previous);
                }
            }
            ServerMsg::Error(e) => panic!("server error: {e}"),
            _ => {}
        }
    }

    for search_id in latest.into_iter().flatten() {
        send(
            &mut socket,
            ClientMsg::ExpandNode {
                search_id,
                path: Vec::new(),
                with_view: false,
            },
        )
        .await;
        loop {
            match recv(&mut socket).await {
                ServerMsg::Node(node) => {
                    assert_eq!(node.search_id, search_id);
                    break;
                }
                ServerMsg::Error(e) => panic!("search {search_id} should be walkable: {e}"),
                _ => {}
            }
        }
    }

    send(
        &mut socket,
        ClientMsg::ExpandNode {
            search_id: older.unwrap(),
            path: Vec::new(),
            with_view: false,
        },
    )
    .await;
    loop {
        match recv(&mut socket).await {
            ServerMsg::Error(e) => {
                assert!(e.contains("superseded"), "{e}");
                break;
            }
            ServerMsg::Node(n) => panic!("a bot's older search must not be answered: {n:?}"),
            _ => {}
        }
    }
}
