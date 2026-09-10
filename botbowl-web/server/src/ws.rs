//! The websocket handler: JSON in, JSON out, and nothing else.
//!
//! All the game logic lives on the blocking thread this spawns
//! ([`crate::session::run`]). The async side only translates frames, so a
//! multi-second MCTS search cannot stall the runtime and a dropped socket
//! tears the session down by closing its channels.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use botbowl_web_proto::msg::{ClientMsg, ServerMsg};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;

use crate::{session, AppState};

/// Outbound queue depth. A full `ViewState` is a few tens of KB and the
/// session can emit a burst of them (a kickoff resolves many rolls in a row),
/// so this is generous; the session blocks rather than dropping if it fills.
const OUT_BUFFER: usize = 256;

pub async fn handler(ws: WebSocketUpgrade, State(app): State<Arc<AppState>>) -> Response {
    ws.on_upgrade(move |socket| serve(socket, app))
}

async fn serve(socket: WebSocket, app: Arc<AppState>) {
    let (mut sink, mut stream) = socket.split();
    let (to_session, from_ws) = mpsc::channel::<ClientMsg>(64);
    let (to_ws, mut from_session) = mpsc::channel::<ServerMsg>(OUT_BUFFER);
    let errors = to_ws.clone();

    let session = tokio::task::spawn_blocking(move || session::run(app, from_ws, to_ws));

    let pump = tokio::spawn(async move {
        while let Some(msg) = from_session.recv().await {
            let json = match serde_json::to_string(&msg) {
                Ok(json) => json,
                // Serialising our own types cannot fail in practice; if it
                // ever does, say so rather than dropping the message silently.
                Err(e) => {
                    serde_json::to_string(&ServerMsg::Error(format!("server could not serialise a message: {e}")))
                        .unwrap_or_else(|_| "{\"Error\":\"serialisation failed\"}".to_string())
                }
            };
            if sink.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    });

    while let Some(frame) = stream.next().await {
        let Ok(frame) = frame else { break };
        match frame {
            Message::Text(text) => match serde_json::from_str::<ClientMsg>(&text) {
                Ok(msg) => {
                    if to_session.send(msg).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = errors
                        .send(ServerMsg::Error(format!("could not parse client message: {e}")))
                        .await;
                }
            },
            Message::Close(_) => break,
            // Ping/Pong are handled by axum; binary frames are not part of
            // the protocol.
            _ => {}
        }
    }

    // Closing the input channel ends the session loop; it may take a moment
    // if a search is in flight (a documented POC limitation — there is no
    // cancel).
    drop(to_session);
    let _ = session.await;
    pump.abort();
}
