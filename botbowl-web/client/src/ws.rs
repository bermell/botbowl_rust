//! The websocket, as thin as it goes: JSON in, signals out.
//!
//! The socket lives in a `thread_local` rather than in Leptos storage because
//! wasm is single-threaded and `web_sys::WebSocket` is neither `Send` nor
//! `Sync` — a `RefCell` in TLS is the honest representation and keeps the
//! reactive graph free of non-`Send` values.

use std::cell::RefCell;

use botbowl_web_proto::msg::{ClientMsg, ServerMsg};
use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CloseEvent, MessageEvent, WebSocket};

use crate::state::{App, Connection, DICE_TICKER};

thread_local! {
    static SOCKET: RefCell<Option<WebSocket>> = const { RefCell::new(None) };
}

/// `ws://<this host>/ws` — the client is always served by its own server.
fn endpoint() -> String {
    let location = web_sys::window().expect("a window").location();
    let protocol = if location.protocol().as_deref() == Ok("https:") {
        "wss"
    } else {
        "ws"
    };
    let host = location.host().unwrap_or_else(|_| "127.0.0.1:8080".into());
    format!("{protocol}://{host}/ws")
}

/// Send one message. Silently drops when the socket is not open — every
/// caller is a UI event, and the UI already shows the connection state.
pub fn send(msg: &ClientMsg) {
    let json = match serde_json::to_string(msg) {
        Ok(json) => json,
        Err(_) => return,
    };
    SOCKET.with(|s| {
        if let Some(socket) = s.borrow().as_ref() {
            let _ = socket.send_with_str(&json);
        }
    });
}

pub fn connect(app: App) {
    let socket = match WebSocket::new(&endpoint()) {
        Ok(socket) => socket,
        Err(_) => {
            app.connection.set(Connection::Closed);
            app.error("could not open a websocket to the server".into());
            return;
        }
    };

    let on_open = Closure::<dyn FnMut()>::new(move || app.connection.set(Connection::Open));
    socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
    on_open.forget();

    let on_close = Closure::<dyn FnMut(CloseEvent)>::new(move |_| {
        app.connection.set(Connection::Closed);
    });
    socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));
    on_close.forget();

    let on_error = Closure::<dyn FnMut(web_sys::Event)>::new(move |_| {
        app.connection.set(Connection::Closed);
    });
    socket.set_onerror(Some(on_error.as_ref().unchecked_ref()));
    on_error.forget();

    let on_message = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
        let Some(text) = event.data().as_string() else { return };
        match serde_json::from_str::<ServerMsg>(&text) {
            Ok(msg) => handle(app, msg),
            Err(e) => app.error(format!("could not parse a server message: {e}")),
        }
    });
    socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();

    SOCKET.with(|s| *s.borrow_mut() = Some(socket));
}

fn handle(app: App, msg: ServerMsg) {
    match msg {
        ServerMsg::Lobby(lobby) => {
            app.spec.set(Some(lobby.defaults.clone()));
            app.lobby.set(Some(*lobby));
        }
        ServerMsg::View(view) => {
            // A stale view can arrive after an undo or a race; the sequence
            // number is monotonic per session, so ignore anything older.
            let fresh = app
                .view
                .with(|current| current.as_ref().is_none_or(|c| view.seq >= c.seq));
            if fresh {
                app.thinking.set(None);
                app.menu.set(None);
                app.view.set(Some(*view));
            }
        }
        ServerMsg::Dice(event) => app.dice.update(|d| {
            d.insert(0, event);
            d.truncate(DICE_TICKER);
        }),
        ServerMsg::BotThinking { team, budget } => {
            app.thinking.set(Some(format!("{team:?} thinking — {budget}")));
        }
        ServerMsg::BotMoved { report, .. } => {
            if let Some(report) = report {
                app.node.set(None);
                app.node_path.set(Vec::new());
                app.hypothetical.set(None);
                app.report.set(Some(*report));
            }
        }
        ServerMsg::Node(node) => {
            app.node_path.set(node.path.clone());
            app.node.set(Some(*node));
        }
        ServerMsg::RollPinned(roll) => app.pinned.set(roll),
        ServerMsg::Saved { path } => app.saved.set(Some(path)),
        ServerMsg::GameOver {
            winner,
            home_score,
            away_score,
        } => app.game_over.set(Some((winner, home_score, away_score))),
        ServerMsg::Error(e) => app.error(e),
    }
}
