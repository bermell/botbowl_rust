//! Phase 2 of plan 034: the MCTS opponent reports its search, and the client
//! can walk into the tree that produced it.
//!
//! Kept to a handful of moves at a tiny budget — this is a wiring test, not a
//! search-quality one. What it pins:
//!   * `BotMoved` carries a `SearchReport` whose chosen action is the action
//!     actually played;
//!   * root children carry visits / Q / priors and a usable heat share;
//!   * the principal variation's `path` values are accepted by `ExpandNode`,
//!     which is the whole contract behind the tree explorer;
//!   * exploring does not disturb the tree (the same node reads the same
//!     twice).

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use botbowl_web_proto::action::TeamType;
use botbowl_web_proto::msg::{
    BoardSpec, BotSpec, Budget, ClientMsg, EvaluatorSpec, GameSpec, MctsSpec, ServerMsg, StartFrom,
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
        models_dir: "models".into(),
        recordings_dir: std::env::temp_dir().join("botbowl-web-test"),
        model_cache: Default::default(),
        server: "test".into(),
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
            human: TeamType::Home,
            bot: BotSpec::Mcts(MctsSpec {
                // Deliberately tiny: this is a wiring test and it runs in a
                // debug build, where the search is orders of magnitude slower.
                budget: Budget::Iterations(60),
                workers: Some(1),
                evaluator: EvaluatorSpec::Heuristic,
                ..Default::default()
            }),
            seed: Some(5),
            start: StartFrom::CoinToss,
        }),
    )
    .await;

    let mut reports: Vec<SearchReport> = Vec::new();
    let mut saw_thinking = false;
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
            ServerMsg::BotMoved { action, report } => {
                let report = *report.expect("the MCTS bot must report its search");
                assert_eq!(report.chosen, action, "the report must be about the move played");
                assert_eq!(report.agent, TeamType::Away);
                assert_eq!(report.evaluator, "heuristic");
                assert_eq!(report.budget, "60 iterations");
                assert!(report.evaluator_value.is_none(), "only the NN evaluators have one");
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
                    if let SearchEdge::Player(_) = child.edge {
                        assert!(child.prior.is_some(), "a player edge carries its PUCT prior");
                    }
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
            ServerMsg::Error(e) => panic!("server error: {e}"),
            _ => {}
        }
    }
    assert!(saw_thinking, "the UI needs a spinner signal before a search");

    // The principal variation is walkable: each step's `path` is exactly what
    // `ExpandNode` takes.
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
