//! Phase 1 exit criterion of plan 034: a whole game, played over the real
//! websocket, on both the small trained board and the full pitch — from one
//! server binary (decision 8).
//!
//! The client here is deliberately dumb: it plays uniformly random legal
//! actions read *only* out of the `ViewState` the server derived. If the view
//! ever fails to describe a legal move, the game deadlocks and this test
//! fails — which is the property the real UI depends on.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use botbowl_web_proto::action::{Action, TeamType};
use botbowl_web_proto::msg::{BoardSpec, BotSpec, ClientMsg, GameSpec, ServerMsg, StartFrom};
use botbowl_web_proto::view::ViewState;
use botbowl_web_server::{compiled_capacity, router, AppState};
use futures_util::{SinkExt, StreamExt};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokio_tungstenite::tungstenite::Message;

/// A server on an ephemeral loopback port, torn down when the test ends.
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
        .expect("bind an ephemeral port");
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router(app, None, None)).await;
    });
    addr
}

type Socket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn connect(addr: SocketAddr) -> Socket {
    let (socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .expect("websocket upgrade");
    socket
}

async fn send(socket: &mut Socket, msg: ClientMsg) {
    socket
        .send(Message::Text(serde_json::to_string(&msg).unwrap()))
        .await
        .expect("send");
}

/// Next server message, with a timeout so a deadlock fails loudly instead of
/// hanging the suite.
async fn recv(socket: &mut Socket) -> ServerMsg {
    let frame = tokio::time::timeout(Duration::from_secs(120), socket.next())
        .await
        .expect("server went quiet for 120s — deadlock?")
        .expect("socket closed early")
        .expect("frame");
    match frame {
        Message::Text(text) => serde_json::from_str(&text).expect("a ServerMsg"),
        other => panic!("unexpected frame {other:?}"),
    }
}

/// Every legal action the view describes, positional and simple alike. This is
/// the whole point of `ViewState`: a client needs no engine to find its moves.
fn legal_actions(view: &ViewState) -> Vec<Action> {
    let mut actions: Vec<Action> = view
        .squares
        .iter()
        .flat_map(|sq| sq.actions.iter().map(move |at| Action::Positional(*at, sq.pos)))
        .collect();
    actions.extend(view.simple_actions.iter().map(|a| Action::Simple(a.at)));
    actions
}

struct Outcome {
    views: usize,
    human_decisions: usize,
    dice: usize,
    bot_moves: usize,
    home_score: u8,
    away_score: u8,
}

async fn play_a_whole_game(addr: SocketAddr, board: BoardSpec, seed: u64) -> Outcome {
    let mut socket = connect(addr).await;
    let mut rng = StdRng::seed_from_u64(seed);

    match recv(&mut socket).await {
        ServerMsg::Lobby(lobby) => {
            assert_eq!(lobby.capacity, compiled_capacity());
            assert!(
                lobby.boards.contains(&board),
                "the lobby should offer {board:?}, got {:?}",
                lobby.boards
            );
        }
        other => panic!("the first message must be the lobby, got {other:?}"),
    }

    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board,
            human: TeamType::Home,
            bot: BotSpec::Scripted,
            seed: Some(seed),
            start: StartFrom::CoinToss,
        }),
    )
    .await;

    let mut outcome = Outcome {
        views: 0,
        human_decisions: 0,
        dice: 0,
        bot_moves: 0,
        home_score: 0,
        away_score: 0,
    };

    loop {
        match recv(&mut socket).await {
            ServerMsg::View(view) => {
                outcome.views += 1;
                assert_eq!(view.dims.playable_width(), board.width);
                assert_eq!(view.dims.playable_height(), board.height);
                assert_eq!(view.squares.len(), view.dims.width as usize * view.dims.height as usize);
                outcome.home_score = view.scoreboard.home_score;
                outcome.away_score = view.scoreboard.away_score;

                if view.scoreboard.game_over || view.bot_thinking {
                    continue;
                }
                let Some(to_act) = view.to_act else { continue };
                if to_act != TeamType::Home {
                    continue;
                }

                let actions = legal_actions(&view);
                assert!(
                    !actions.is_empty(),
                    "the human was asked to act at {:?} but the view offered nothing",
                    view.proc
                );
                outcome.human_decisions += 1;
                let action = actions[rng.gen_range(0..actions.len())];
                send(&mut socket, ClientMsg::Act(action)).await;
            }
            ServerMsg::Dice(_) => outcome.dice += 1,
            ServerMsg::BotMoved { report, .. } => {
                outcome.bot_moves += 1;
                assert!(report.is_none(), "only the MCTS bot reports a search");
            }
            ServerMsg::BotThinking { team, .. } => assert_eq!(team, TeamType::Away),
            ServerMsg::GameOver {
                home_score, away_score, ..
            } => {
                outcome.home_score = home_score;
                outcome.away_score = away_score;
                return outcome;
            }
            ServerMsg::Error(e) => panic!("server error during play: {e}"),
            other => panic!("unexpected message {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_whole_game_on_the_small_board() {
    let capacity = compiled_capacity();
    let board = BoardSpec::new(14, 7, 4);
    if board.validate(capacity).is_err() {
        eprintln!("skipped: capacity {capacity:?} is too small for 14x7");
        return;
    }
    let addr = serve().await;
    let outcome = play_a_whole_game(addr, board, 7).await;
    assert!(outcome.human_decisions > 20, "{} decisions", outcome.human_decisions);
    assert!(outcome.bot_moves > 20, "{} bot moves", outcome.bot_moves);
    assert!(outcome.dice > 20, "{} dice events", outcome.dice);
    assert!(outcome.views > outcome.human_decisions);
    eprintln!(
        "14x7: {} views, {} human decisions, {} bot moves, {} rolls, {}-{}",
        outcome.views, outcome.human_decisions, outcome.bot_moves, outcome.dice, outcome.home_score, outcome.away_score
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_whole_game_at_the_compiled_capacity() {
    let capacity = compiled_capacity();
    let addr = serve().await;
    let outcome = play_a_whole_game(addr, capacity, 11).await;
    assert!(outcome.human_decisions > 20);
    eprintln!(
        "{}x{}: {} views, {} human decisions, {} bot moves, {} rolls, {}-{}",
        capacity.width,
        capacity.height,
        outcome.views,
        outcome.human_decisions,
        outcome.bot_moves,
        outcome.dice,
        outcome.home_score,
        outcome.away_score
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn undo_rewinds_across_the_bots_reply() {
    let capacity = compiled_capacity();
    let board = if BoardSpec::new(14, 7, 4).validate(capacity).is_ok() {
        BoardSpec::new(14, 7, 4)
    } else {
        capacity
    };
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board,
            human: TeamType::Home,
            bot: BotSpec::Random,
            seed: Some(3),
            start: StartFrom::CoinToss,
        }),
    )
    .await;

    // Play until the human is asked something, remembering that board.
    let mut before: Option<ViewState> = None;
    while before.is_none() {
        if let ServerMsg::View(view) = recv(&mut socket).await {
            if view.to_act == Some(TeamType::Home) && !view.scoreboard.game_over && !view.bot_thinking {
                before = Some(*view);
            }
        }
    }
    let before = before.unwrap();
    assert!(!before.can_undo, "nothing to undo before the first move");

    let action = legal_actions(&before)[0];
    send(&mut socket, ClientMsg::Act(action)).await;

    // Play on until the human is asked again — possibly several of our own
    // decisions and a bot reply later.
    let mut after: Option<ViewState> = None;
    while after.is_none() {
        if let ServerMsg::View(view) = recv(&mut socket).await {
            if view.to_act == Some(TeamType::Home) && !view.scoreboard.game_over && !view.bot_thinking {
                after = Some(*view);
            }
        }
    }
    assert!(after.unwrap().can_undo, "after acting there is something to undo");

    send(&mut socket, ClientMsg::Undo).await;
    let mut rewound: Option<ViewState> = None;
    while rewound.is_none() {
        if let ServerMsg::View(view) = recv(&mut socket).await {
            if view.to_act == Some(TeamType::Home) && !view.bot_thinking {
                rewound = Some(*view);
            }
        }
    }
    let rewound = rewound.unwrap();
    assert_eq!(rewound.squares, before.squares, "undo must restore the board exactly");
    assert_eq!(rewound.scoreboard, before.scoreboard);
    assert!(!rewound.can_undo, "the stack is empty again");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_input_is_reported_not_fatal() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;

    // Acting before there is a game.
    send(&mut socket, ClientMsg::Undo).await;
    assert!(matches!(recv(&mut socket).await, ServerMsg::Error(_)));

    // A board this binary cannot run.
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board: BoardSpec::new(120, 101, 40),
            human: TeamType::Home,
            bot: BotSpec::Random,
            seed: Some(1),
            start: StartFrom::CoinToss,
        }),
    )
    .await;
    match recv(&mut socket).await {
        ServerMsg::Error(e) => assert!(e.contains("capacity"), "{e}"),
        other => panic!("expected a capacity error, got {other:?}"),
    }

    // An odd width is rejected by the engine's own rule, before any state is
    // built (`BoardDims::new` would panic).
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board: BoardSpec::new(9, 7, 3),
            human: TeamType::Home,
            bot: BotSpec::Random,
            seed: Some(1),
            start: StartFrom::CoinToss,
        }),
    )
    .await;
    match recv(&mut socket).await {
        ServerMsg::Error(e) => assert!(e.contains("even"), "{e}"),
        other => panic!("expected a width error, got {other:?}"),
    }

    // ...and the socket still works afterwards.
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board: if BoardSpec::new(14, 7, 4).validate(compiled_capacity()).is_ok() {
                BoardSpec::new(14, 7, 4)
            } else {
                compiled_capacity()
            },
            human: TeamType::Home,
            bot: BotSpec::Random,
            seed: Some(1),
            start: StartFrom::CoinToss,
        }),
    )
    .await;
    let mut saw_view = false;
    for _ in 0..200 {
        if let ServerMsg::View(_) = recv(&mut socket).await {
            saw_view = true;
            break;
        }
    }
    assert!(saw_view, "the session recovered and started a game");
}
