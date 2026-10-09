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

use crate::state::{App, Connection, LOG_CAP};

thread_local! {
    static SOCKET: RefCell<Option<WebSocket>> = const { RefCell::new(None) };
}

/// `ws://<this host><this page's directory>ws` — the client is always served by its own
/// server, at `/` standalone and at `/play/` on the hub, so the socket sits next to the page.
fn endpoint() -> String {
    let location = web_sys::window().expect("a window").location();
    let protocol = if location.protocol().as_deref() == Ok("https:") {
        "wss"
    } else {
        "ws"
    };
    let host = location.host().unwrap_or_else(|_| "127.0.0.1:8080".into());
    let path = location.pathname().unwrap_or_else(|_| "/".into());
    let dir = &path[..path.rfind('/').map_or(0, |i| i + 1)];
    let dir = if dir.is_empty() { "/" } else { dir };
    format!("{protocol}://{host}{dir}ws")
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
            app.teams.set(lobby.teams.clone());
            app.pictures.set(lobby.pictures.clone());
            app.step_mode.set(lobby.step_mode);
            if let Some(ms) = lobby.step_mode.millis() {
                app.step_ms.set(ms);
            }
            app.lobby.set(Some(*lobby));
        }
        ServerMsg::Teams(teams) => {
            app.teams.set(teams);
            app.notice.set(Some("saved".into()));
        }
        ServerMsg::PictureSaved { picture, pictures } => {
            app.pictures.set(pictures);
            app.uploaded.set(Some(picture));
        }
        ServerMsg::DriveOver {
            attacker,
            scored,
            home_score,
            away_score,
        } => app.drive_over.set(Some((attacker, scored, home_score, away_score))),
        ServerMsg::View(view) => {
            // A stale view can arrive after an undo or a race; the sequence
            // number is monotonic per session, so ignore anything older.
            let fresh = app
                .view
                .with(|current| current.as_ref().is_none_or(|c| view.seq >= c.seq));
            if fresh {
                // `bot_thinking` is the view of the board the search is
                // about: the spinner stays up until the next one.
                if !view.bot_thinking {
                    app.thinking.set(None);
                }
                app.menu.set(None);
                app.selection.set(None);
                // The server is the authority on pacing — it may have clamped
                // or carried a mode across a new game.
                app.step_mode.set(view.step_mode);
                if let Some(ms) = view.step_mode.millis() {
                    app.step_ms.set(ms);
                }
                app.view.set(Some(*view));
            }
        }
        ServerMsg::Log(entry) => app.log.update(|log| {
            // Indices are reused after a rewind; the truncation message has
            // already cut the log back, this is belt and braces — unless the
            // cap below has dropped the oldest lines, in which case the index
            // is simply ahead of the length.
            if let Some(at) = log.iter().position(|e| e.index >= entry.index) {
                log.truncate(at);
            }
            log.push(entry);
            if log.len() > LOG_CAP {
                log.drain(..log.len() - LOG_CAP);
            }
        }),
        ServerMsg::LogTruncated { keep } => app.log.update(|log| {
            if let Some(at) = log.iter().position(|e| e.index >= keep) {
                log.truncate(at);
            }
        }),
        ServerMsg::BotThinking { team, .. } => app.thinking.set(Some(team)),
        // A preview of a board that has since moved on is no use.
        ServerMsg::Selection(selection) => {
            if app.view.with_untracked(|v| v.as_ref().is_some_and(|v| v.seq == selection.seq)) {
                app.selection.set(Some(*selection));
            }
        }
        ServerMsg::Decision(record) => {
            // Following the game: a new search replaces the tree the explorer
            // was walking, so drop back to the live board with it.
            if app.selected.get_untracked().is_none() && record.search.is_some() {
                app.back_to_live();
            }
            app.decisions.update(|d| {
                // After an undo the server reuses indices; the truncation
                // message has already cut the log back, this is belt and braces.
                d.truncate(record.index as usize);
                d.push(*record);
            });
        }
        ServerMsg::DecisionsTruncated { keep } => {
            app.decisions.update(|d| d.truncate(keep as usize));
            if app.selected.get_untracked().is_some_and(|i| i >= keep) {
                app.selected.set(None);
            }
            if app.board_of.get_untracked().is_some_and(|i| i >= keep) {
                app.back_to_live();
            }
        }
        ServerMsg::DecisionBoard { index, view } => {
            app.node.set(None);
            app.node_path.set(Vec::new());
            app.board_of.set(Some(index));
            app.hypothetical.set(Some(*view));
        }
        ServerMsg::Node(node) => {
            app.node_path.set(node.path.clone());
            app.node.set(Some(*node));
        }
        ServerMsg::Net(readout) => app.net_now.set(Some(*readout)),
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
