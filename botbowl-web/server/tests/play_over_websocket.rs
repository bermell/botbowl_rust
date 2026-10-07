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
use botbowl_web_proto::decision::Decider;
use botbowl_web_proto::log::LogKind;
use botbowl_web_proto::msg::{BoardSpec, BotSpec, ClientMsg, GameSpec, Seat, ServerMsg, StartFrom, StepMode};
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
        // The tests drive the pacing themselves; they start from the original free-running mode.
        opts: botbowl_web_server::PlayOptions {
            initial_step_mode: botbowl_web_proto::msg::StepMode::Run,
            ..Default::default()
        },
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
        .send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
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
            home: Seat::Human,
            away: Seat::Bot(BotSpec::Random),
            seed: Some(seed),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
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
            ServerMsg::Log(entry) => {
                if entry.kind == LogKind::Roll {
                    assert!(entry.roll.is_some(), "a roll line carries its die");
                    outcome.dice += 1;
                }
            }
            ServerMsg::LogTruncated { .. } => {}
            ServerMsg::Decision(record) => {
                assert!(record.search.is_none(), "only the MCTS bot reports a search");
                assert!(record.net.is_none(), "no seat has a net to read out");
                match record.by {
                    Decider::Human => assert_eq!(record.team, TeamType::Home),
                    Decider::Bot { .. } => {
                        assert_eq!(record.team, TeamType::Away);
                        outcome.bot_moves += 1;
                    }
                }
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
            home: Seat::Human,
            away: Seat::Bot(BotSpec::Random),
            seed: Some(3),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
        }),
    )
    .await;

    // Play until the human is asked something, remembering that board and
    // how long the decision log was.
    let mut before: Option<ViewState> = None;
    let mut logged = 0u64;
    while before.is_none() {
        match recv(&mut socket).await {
            ServerMsg::View(view) => {
                if view.to_act == Some(TeamType::Home) && !view.scoreboard.game_over && !view.bot_thinking {
                    before = Some(*view);
                }
            }
            ServerMsg::Decision(_) => logged += 1,
            _ => {}
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
    let mut truncated = None;
    while rewound.is_none() {
        match recv(&mut socket).await {
            ServerMsg::View(view) => {
                if view.to_act == Some(TeamType::Home) && !view.bot_thinking {
                    rewound = Some(*view);
                }
            }
            ServerMsg::DecisionsTruncated { keep } => truncated = Some(keep),
            _ => {}
        }
    }
    assert_eq!(
        truncated,
        Some(logged),
        "undo cuts the decision log back to where it was"
    );
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
            home: Seat::Human,
            away: Seat::Bot(BotSpec::Random),
            seed: Some(1),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
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
            home: Seat::Human,
            away: Seat::Bot(BotSpec::Random),
            seed: Some(1),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
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
            home: Seat::Human,
            away: Seat::Bot(BotSpec::Random),
            seed: Some(1),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
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

/// The board the tests play on: the plan's first target, or the whole pitch
/// when this binary was built too small for it.
fn test_board() -> BoardSpec {
    let capacity = compiled_capacity();
    if BoardSpec::new(14, 7, 4).validate(capacity).is_ok() {
        BoardSpec::new(14, 7, 4)
    } else {
        capacity
    }
}

fn new_game(board: BoardSpec, seed: u64) -> ClientMsg {
    ClientMsg::NewGame(GameSpec {
        board,
        home: Seat::Human,
        away: Seat::Bot(BotSpec::Random),
        seed: Some(seed),
        start: StartFrom::CoinToss,
        home_team: "Human".into(),
        away_team: "Human".into(),
    })
}

/// `Manual` pacing: the session holds *after* each bot decision and before the
/// move is played, so the board on screen is the position the decision was
/// about, the record beside it is that decision, and nothing moves until the
/// client asks. Dice are never held — they roll through to the next decision.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn manual_pacing_holds_on_every_bot_decision() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;

    // Sent before there is a game: the pacing lives on the connection, so
    // this must be remembered rather than answered with "no game yet".
    send(&mut socket, ClientMsg::SetStepMode(StepMode::Manual)).await;
    send(&mut socket, new_game(test_board(), 7)).await;

    let mut rng = StdRng::seed_from_u64(7);
    let mut holds = 0usize;
    // Bot decisions since the last hold was released: a hold is reached by
    // exactly one, the one it holds in front of.
    let mut decisions_since_release = 0usize;
    let mut last_bot_action = None;
    let mut checked_quiet = false;

    while holds < 40 {
        match recv(&mut socket).await {
            ServerMsg::View(view) => {
                assert_eq!(view.step_mode, StepMode::Manual, "the view reports the pacing");
                if view.scoreboard.game_over {
                    break;
                }
                if view.paused {
                    holds += 1;
                    assert!(!view.bot_thinking, "a hold comes after the search, not during it");
                    assert_eq!(decisions_since_release, 1, "one bot decision per hold");
                    assert_eq!(
                        view.pending_action, last_bot_action,
                        "the view names the move the next step will play"
                    );
                    // The first hold: prove nothing moves on its own.
                    if !checked_quiet {
                        checked_quiet = true;
                        let quiet = tokio::time::timeout(Duration::from_millis(300), socket.next()).await;
                        assert!(quiet.is_err(), "a held session must not step itself: {quiet:?}");
                    }
                    decisions_since_release = 0;
                    send(&mut socket, ClientMsg::StepOnce).await;
                    continue;
                }
                assert!(view.pending_action.is_none(), "a pending move means a hold");
                if view.bot_thinking || view.to_act != Some(TeamType::Home) {
                    continue;
                }
                // Our own decision points are never held.
                let actions = legal_actions(&view);
                assert!(
                    !actions.is_empty(),
                    "asked to act at {:?} with nothing on offer",
                    view.proc
                );
                decisions_since_release = 0;
                send(&mut socket, ClientMsg::Act(actions[rng.gen_range(0..actions.len())])).await;
            }
            ServerMsg::Decision(d) if matches!(d.by, Decider::Bot { .. }) => {
                decisions_since_release += 1;
                last_bot_action = Some(d.action);
            }
            ServerMsg::Decision(_) | ServerMsg::Log(_) | ServerMsg::BotThinking { .. } => {}
            ServerMsg::GameOver { .. } => break,
            ServerMsg::Error(e) => panic!("server error while stepping: {e}"),
            other => panic!("unexpected message {other:?}"),
        }
    }

    assert!(holds >= 20, "only {holds} holds — the session was not stepping");
}

/// A rewind puts an earlier micro-step back on the board and continues from
/// it: the log and the decision log are cut to what led there, the bot's
/// decision taken at that position is searched again, and under `Run` the
/// session switches itself to `Manual` so the rewound position stays.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rewind_restores_an_earlier_step_and_holds_there() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board: test_board(),
            home: Seat::Bot(BotSpec::Random),
            away: Seat::Bot(BotSpec::Random),
            seed: Some(31),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
        }),
    )
    .await;

    // Let the bots play a while, then pick a bot decision from the middle.
    let mut decisions = Vec::new();
    let mut log_len = 0u64;
    while decisions.len() < 30 {
        match recv(&mut socket).await {
            ServerMsg::Decision(d) => decisions.push(*d),
            ServerMsg::Log(e) => {
                assert_eq!(e.index, log_len, "log lines arrive in order");
                log_len += 1;
            }
            ServerMsg::Error(e) => panic!("server error: {e}"),
            ServerMsg::GameOver { .. } => panic!("the game ended before 30 decisions"),
            _ => {}
        }
    }
    let target = decisions[12].clone();
    send(&mut socket, ClientMsg::RewindTo { step: target.step }).await;

    let mut log_keep = None;
    let mut decisions_keep = None;
    let mut redone = None;
    let held = loop {
        match recv(&mut socket).await {
            ServerMsg::LogTruncated { keep } => log_keep = Some(keep),
            ServerMsg::DecisionsTruncated { keep } => decisions_keep = Some(keep),
            // The first decision after the cut is the rewound one, taken afresh.
            ServerMsg::Decision(d) if decisions_keep.is_some() && redone.is_none() => redone = Some(*d),
            ServerMsg::View(view) if view.paused => break view,
            ServerMsg::Error(e) => panic!("server error: {e}"),
            _ => {}
        }
    };
    assert_eq!(
        decisions_keep,
        Some(12),
        "decisions taken at or after the step are gone"
    );
    let log_keep = log_keep.expect("the log was cut");
    assert!(log_keep < log_len, "the log was cut back ({log_keep} of {log_len})");
    let redone = redone.expect("the bot decided again at the rewound position");
    assert_eq!(redone.index, 12, "the redone decision takes the freed index");
    assert_eq!(redone.step, target.step, "it is taken at the rewound step");
    assert_eq!(redone.team, target.team);
    assert_eq!(
        held.step_mode,
        StepMode::Manual,
        "a rewind under Run switches to Manual so the position stays"
    );
    assert_eq!(held.pending_action, Some(redone.action));

    // Held means held.
    let quiet = tokio::time::timeout(Duration::from_millis(300), socket.next()).await;
    assert!(quiet.is_err(), "a rewound session must not move on its own: {quiet:?}");

    // And it plays on from there when stepped.
    send(&mut socket, ClientMsg::StepOnce).await;
    loop {
        match recv(&mut socket).await {
            ServerMsg::Decision(d) => {
                assert_eq!(d.index, 13, "play continues from the rewound position");
                break;
            }
            ServerMsg::Error(e) => panic!("server error: {e}"),
            _ => {}
        }
    }
}

/// Switching back to `Run` while the session is holding releases it, and the
/// game finishes on its own from there.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn run_releases_a_held_session() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;
    send(&mut socket, ClientMsg::SetStepMode(StepMode::Manual)).await;
    send(&mut socket, new_game(test_board(), 5)).await;

    // Wait for the first hold.
    loop {
        if let ServerMsg::View(view) = recv(&mut socket).await {
            if view.paused {
                break;
            }
        }
    }

    send(&mut socket, ClientMsg::SetStepMode(StepMode::Run)).await;

    // The session must now reach a human decision (or the end) unprodded.
    let mut released = false;
    for _ in 0..400 {
        if let ServerMsg::View(view) = recv(&mut socket).await {
            assert_eq!(view.step_mode, StepMode::Run);
            if view.scoreboard.game_over {
                released = true;
                break;
            }
            if !view.paused && !view.bot_thinking && view.to_act == Some(TeamType::Home) {
                released = true;
                break;
            }
        }
    }
    assert!(released, "Run did not release the hold");
}

/// `Auto` paces itself: no `StepOnce` is ever sent here, and the game still
/// makes progress.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn auto_pacing_steps_itself() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;
    send(&mut socket, ClientMsg::SetStepMode(StepMode::Auto { ms: 20 })).await;
    send(&mut socket, new_game(test_board(), 9)).await;

    let mut rng = StdRng::seed_from_u64(9);
    let mut work = 0usize;
    let mut human_decisions = 0usize;

    while work < 30 {
        match recv(&mut socket).await {
            ServerMsg::View(view) => {
                assert_eq!(view.step_mode, StepMode::Auto { ms: 20 });
                if view.scoreboard.game_over {
                    break;
                }
                if view.paused || view.bot_thinking || view.to_act != Some(TeamType::Home) {
                    continue;
                }
                let actions = legal_actions(&view);
                human_decisions += 1;
                send(&mut socket, ClientMsg::Act(actions[rng.gen_range(0..actions.len())])).await;
            }
            ServerMsg::Log(e) if e.kind == LogKind::Roll => work += 1,
            ServerMsg::Decision(d) if matches!(d.by, Decider::Bot { .. }) => work += 1,
            ServerMsg::Error(e) => panic!("server error under auto pacing: {e}"),
            _ => {}
        }
    }

    assert!(work >= 30, "auto pacing stalled after {work} steps");
    assert!(human_decisions > 0, "auto pacing never handed control back");
}

/// Two bots, nobody at the keyboard: under `Run` the game plays itself to the
/// end, every decision from both sides lands in the log in order — and the
/// socket stays responsive throughout, so switching to `Manual` mid-game
/// actually stops it (the session yields between bot moves).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_bots_play_a_whole_game_and_can_be_paused() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board: test_board(),
            home: Seat::Bot(BotSpec::Random),
            away: Seat::Bot(BotSpec::Random),
            seed: Some(21),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
        }),
    )
    .await;

    let mut decisions = 0u64;
    let mut by_team = [0usize; 2];
    let mut paused_at: Option<u64> = None;
    loop {
        match recv(&mut socket).await {
            ServerMsg::Decision(record) => {
                assert_eq!(record.index, decisions, "decisions arrive in order");
                assert!(matches!(record.by, Decider::Bot { .. }), "nobody human is seated");
                decisions += 1;
                by_team[usize::from(record.team == TeamType::Away)] += 1;
                // Early (was 50): with the engine optimised in dev builds (plan 059) the rest of
                // a random game could finish before the server saw the mode change.
                if decisions == 10 {
                    send(&mut socket, ClientMsg::SetStepMode(StepMode::Manual)).await;
                }
            }
            ServerMsg::View(view) => {
                assert!(view.humans.is_empty());
                assert!(!view.human_to_act(), "no view may offer a bot's move to the browser");
                if view.paused && paused_at.is_none() {
                    paused_at = Some(decisions);
                    // Held means held: nothing moves until we say so ...
                    let quiet = tokio::time::timeout(Duration::from_millis(300), socket.next()).await;
                    assert!(quiet.is_err(), "a held bot-vs-bot game must not step itself: {quiet:?}");
                    // ... and Run lets it finish.
                    send(&mut socket, ClientMsg::SetStepMode(StepMode::Run)).await;
                }
            }
            ServerMsg::GameOver { .. } => break,
            ServerMsg::Error(e) => panic!("server error: {e}"),
            _ => {}
        }
    }
    let paused_at = paused_at.expect("Manual never took hold of a running bot-vs-bot game");
    assert!(
        paused_at >= 10,
        "paused at decision {paused_at}, before the mode was even sent"
    );
    assert!(by_team.iter().all(|&n| n > 20), "both bots played: {by_team:?}");
}

/// A random-start drive between two random bots ends with `DriveOver` — never a whole game's
/// `GameOver` — and starts mid-turn, with each side's players wearing its team's pictures.
/// Several position seeds, so a score and a half running out both get exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_random_drive_ends_when_the_drive_does() {
    let addr = serve().await;
    for seed in 0..6u64 {
        let mut socket = connect(addr).await;
        let _lobby = recv(&mut socket).await;
        send(
            &mut socket,
            ClientMsg::NewGame(GameSpec {
                board: test_board(),
                home: Seat::Bot(BotSpec::Random),
                away: Seat::Bot(BotSpec::Random),
                seed: Some(seed),
                start: StartFrom::RandomDrive {
                    seed: Some(1000 + seed),
                },
                home_team: "Orc".into(),
                away_team: "Skaven".into(),
            }),
        )
        .await;
        let mut first_view: Option<Box<ViewState>> = None;
        loop {
            match recv(&mut socket).await {
                ServerMsg::View(v) => {
                    first_view.get_or_insert(v);
                }
                ServerMsg::DriveOver {
                    scored,
                    home_score,
                    away_score,
                    ..
                } => {
                    let first = &first_view.as_ref().unwrap().scoreboard;
                    match scored {
                        Some(TeamType::Home) => assert_eq!(home_score, first.home_score + 1),
                        Some(TeamType::Away) => assert_eq!(away_score, first.away_score + 1),
                        None => assert_eq!((home_score, away_score), (first.home_score, first.away_score)),
                    }
                    break;
                }
                ServerMsg::GameOver { .. } => panic!("a drive must end as a drive"),
                ServerMsg::Error(e) => panic!("seed {seed}: {e}"),
                _ => {}
            }
        }
        let first = first_view.unwrap();
        let pictures: Vec<&str> = first
            .squares
            .iter()
            .filter_map(|s| s.player.as_ref())
            .map(|p| p.sprite.as_str())
            .collect();
        assert!(!pictures.is_empty(), "the position has players on the pitch");
        // Random-start players are drawn from the lineman template, so they wear each team's
        // filler picture.
        assert!(pictures.iter().any(|s| s.contains("olineman1")), "{pictures:?}");
        assert!(pictures.iter().any(|s| s.contains("sklineman1")), "{pictures:?}");
    }
}

/// An unknown team is an error message, not a panic, and a known one reaches the roster.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn teams_are_chosen_by_name() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    match recv(&mut socket).await {
        ServerMsg::Lobby(lobby) => {
            assert!(lobby.teams.iter().any(|t| t.name == "Dwarf"));
            assert!(lobby.skills.iter().any(|s| s.label == "Block" && s.implemented));
        }
        other => panic!("expected the lobby, got {other:?}"),
    }
    let mut spec = GameSpec {
        board: test_board(),
        home: Seat::Human,
        away: Seat::Bot(BotSpec::Random),
        seed: Some(3),
        start: StartFrom::CoinToss,
        home_team: "Nobody".into(),
        away_team: "Dwarf".into(),
    };
    send(&mut socket, ClientMsg::NewGame(spec.clone())).await;
    match recv(&mut socket).await {
        ServerMsg::Error(e) => assert!(e.contains("Nobody"), "{e}"),
        other => panic!("expected an unknown-team error, got {other:?}"),
    }
    spec.home_team = "Human".into();
    send(&mut socket, ClientMsg::NewGame(spec)).await;
    loop {
        if let ServerMsg::View(v) = recv(&mut socket).await {
            let away = v.dugouts.iter().find(|d| d.team == TeamType::Away).unwrap();
            assert!(
                away.players.iter().all(|p| p.sprite.starts_with("iconssmall/d")),
                "{:?}",
                away.players
            );
            break;
        }
    }
}

/// `AutoSetup` plays out the whole of the human's setup with one of the
/// formations the view named, as one undoable decision — so a player who does
/// not want to place eleven pieces by hand still gets the per-player undo.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn auto_setup_finishes_the_humans_setup_as_one_undoable_decision() {
    let addr = serve().await;
    let mut socket = connect(addr).await;
    let _lobby = recv(&mut socket).await;
    send(
        &mut socket,
        ClientMsg::NewGame(GameSpec {
            board: test_board(),
            home: Seat::Human,
            away: Seat::Bot(BotSpec::Random),
            seed: Some(5),
            start: StartFrom::CoinToss,
            home_team: "Human".into(),
            away_team: "Human".into(),
        }),
    )
    .await;

    // Answer the coin toss and whatever else comes first, until the view says
    // Home is placing players.
    let mut setup_view: Option<ViewState> = None;
    while setup_view.is_none() {
        if let ServerMsg::View(view) = recv(&mut socket).await {
            if view.to_act != Some(TeamType::Home) || view.scoreboard.game_over || view.bot_thinking {
                continue;
            }
            if view.setup.as_ref().is_some_and(|s| s.team == TeamType::Home) {
                setup_view = Some(*view);
            } else {
                let action = legal_actions(&view)[0];
                send(&mut socket, ClientMsg::Act(action)).await;
            }
        }
    }
    let before = setup_view.unwrap();
    let setup = before.setup.clone().unwrap();
    assert_eq!(setup.placed, 0);
    assert_eq!(setup.team_size, test_board().team_size);
    assert!(setup.formations.contains(&"Line".to_string()), "{:?}", setup.formations);
    assert!(
        before
            .squares
            .iter()
            .any(|s| s.actions.contains(&botbowl_web_proto::PosAT::PlacePlayer)),
        "the setup view offers per-square placement"
    );

    // An unknown formation is ignored, not fatal: the socket stays up and the
    // real request after it still works.
    send(&mut socket, ClientMsg::AutoSetup("Pyramid".into())).await;
    send(&mut socket, ClientMsg::AutoSetup(setup.formations[0].clone())).await;

    let mut after: Option<ViewState> = None;
    while after.is_none() {
        if let ServerMsg::View(view) = recv(&mut socket).await {
            if view.to_act == Some(TeamType::Home) && !view.scoreboard.game_over && !view.bot_thinking {
                after = Some(*view);
            }
        }
    }
    let after = after.unwrap();
    assert!(
        after.setup.as_ref().is_none_or(|s| s.team != TeamType::Home),
        "Home's setup is over, got {:?}",
        after.setup
    );
    let fielded = after
        .squares
        .iter()
        .filter(|s| s.player.as_ref().is_some_and(|p| p.team == TeamType::Home))
        .count();
    assert_eq!(fielded, test_board().team_size, "the formation fielded a full team");
    assert!(after.can_undo);

    // One undo lands back on the very first placement prompt.
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
    assert_eq!(
        rewound.squares, before.squares,
        "undo must restore the setup prompt exactly"
    );
    assert_eq!(rewound.setup, before.setup);
}
