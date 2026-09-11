//! Blood Bowl web play — the browser half of `plans/034-plan--web-play-ui.md`.
//!
//! A Leptos CSR app that renders whatever the server derived and sends back
//! clicks. It depends on `botbowl-web-proto` and nothing else from the
//! workspace: no engine, no rules, no geometry (decision 3). That is what
//! keeps the wasm build trivial and puts every game decision in one
//! server-side function that can be unit-tested.

mod app;
mod inspector;
mod lobby;
mod pitch;
mod state;
mod ws;

fn main() {
    console_error_panic_hook::set_once();
    leptos::mount::mount_to_body(app::Root);
}
