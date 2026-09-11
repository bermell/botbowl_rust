//! The shell: connect, then show either the lobby or the game.

use leptos::prelude::*;

use crate::inspector::Inspector;
use crate::lobby::Lobby;
use crate::pitch::Game;
use crate::state::{App, Connection};
use crate::ws;

#[component]
pub fn Root() -> impl IntoView {
    let app = App::new();
    provide_context(app);
    ws::connect(app);

    view! {
        <div class="shell">
            <ConnectionBanner />
            <Errors />
            {move || {
                if app.view.get().is_some() {
                    view! { <div class="game-and-inspector"><Game /><Inspector /></div> }.into_any()
                } else {
                    view! { <Lobby /> }.into_any()
                }
            }}
        </div>
    }
}

#[component]
fn ConnectionBanner() -> impl IntoView {
    let app = expect_context::<App>();
    move || match app.connection.get() {
        Connection::Open => None,
        Connection::Connecting => Some(view! { <div class="banner">"connecting…"</div> }),
        Connection::Closed => Some(view! {
            <div class="banner error">
                "disconnected — the server stopped, or the page outlived it. Reload to reconnect."
            </div>
        }),
    }
}

#[component]
fn Errors() -> impl IntoView {
    let app = expect_context::<App>();
    move || {
        let errors = app.errors.get();
        (!errors.is_empty()).then(|| {
            view! {
                <div class="errors" on:click=move |_| app.errors.set(Vec::new())>
                    {errors.into_iter().map(|e| view! { <div class="error-line">{e}</div> }).collect_view()}
                    <div class="dismiss">"click to dismiss"</div>
                </div>
            }
        })
    }
}
