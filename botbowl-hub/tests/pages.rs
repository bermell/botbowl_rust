//! The hub's browser-facing layout: `/` an index, `/status` the status page, `/play/` the web
//! play app nested whole (its own `/ws` included), and `/ws` still the workers'.

use std::time::Duration;

use axum::routing::get;
use axum::Router;
use botbowl_hub::http::request;
use botbowl_hub::{Hub, HubConfig};

async fn get_path(addr: std::net::SocketAddr, path: &str) -> (u16, String) {
    let url = format!("http://{addr}{path}");
    tokio::task::spawn_blocking(move || request("GET", &url, "", None).expect("request"))
        .await
        .unwrap()
}

async fn start(play: Option<Router>) -> std::net::SocketAddr {
    let (_hub, addr, _task) = Hub::start_with(
        HubConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            token: "t".into(),
            allow_commit_mismatch: false,
            allowed_commits: "/nonexistent".into(),
            worker_timeout: Duration::from_secs(120),
            run_dir: None,
        },
        play,
    )
    .await
    .unwrap();
    addr
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_index_status_and_play_pages_sit_side_by_side() {
    // A stand-in for the play app, which records the path it was handed.
    let play = Router::new()
        .route("/", get(|| async { "play index" }))
        .route("/ws", get(|| async { "play socket" }))
        .route("/img/x.gif", get(|| async { "sprite" }));
    let addr = start(Some(play)).await;

    let (code, body) = get_path(addr, "/").await;
    assert_eq!(code, 200);
    assert!(
        body.contains("href=\"/status\"") && body.contains("href=\"/play/\""),
        "{body}"
    );

    let (code, body) = get_path(addr, "/status").await;
    assert_eq!(code, 200);
    assert!(body.contains("botbowl hub · commit"), "{body}");

    // Relative URLs in the client resolve against `/play/` only with the slash.
    assert_eq!(get_path(addr, "/play").await.0, 308);
    assert_eq!(get_path(addr, "/play/").await, (200, "play index".into()));
    assert_eq!(get_path(addr, "/play/ws").await, (200, "play socket".into()));
    assert_eq!(get_path(addr, "/play/img/x.gif").await, (200, "sprite".into()));

    // The workers' socket is untouched: a plain GET is refused for want of an upgrade, not 404.
    let (code, _) = get_path(addr, "/ws").await;
    assert_ne!(code, 404);
    assert_ne!(code, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_play_app_play_is_absent() {
    let addr = start(None).await;
    let (code, body) = get_path(addr, "/").await;
    assert_eq!(code, 200);
    assert!(body.contains("not served"), "{body}");
    assert_eq!(get_path(addr, "/play/").await.0, 404);
}
