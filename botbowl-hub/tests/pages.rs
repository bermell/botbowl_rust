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
            allow_from: Vec::new(),
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

/// `--allow-from`: the browser-facing routes answer only the listed addresses (and loopback);
/// everyone else gets 403. `/ws` never checks — workers authenticate with the token.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn allow_from_keeps_pages_and_play_to_listed_addresses() {
    use axum::body::Body;
    use axum::extract::ConnectInfo;
    use axum::http::Request;
    use tower::ServiceExt;

    let hub = Hub::new(HubConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        token: "t".into(),
        allow_commit_mismatch: false,
        allowed_commits: "/nonexistent".into(),
        worker_timeout: Duration::from_secs(120),
        run_dir: None,
        allow_from: vec!["157.250.168.190".parse().unwrap(), "192.168.1.0/24".parse().unwrap()],
    });
    let play = Router::new().route("/", get(|| async { "play index" }));
    let router = hub.router_with(Some(play));
    let status = |from: &str, path: &str| {
        let router = router.clone();
        let peer: std::net::SocketAddr = format!("{from}:50000").parse().unwrap();
        let path = path.to_string();
        async move {
            let mut req = Request::builder().uri(path).body(Body::empty()).unwrap();
            req.extensions_mut().insert(ConnectInfo(peer));
            router.oneshot(req).await.unwrap().status().as_u16()
        }
    };
    for path in ["/", "/status", "/play/"] {
        assert_eq!(
            status("157.250.168.190", path).await,
            200,
            "{path} from the VPN address"
        );
        assert_eq!(status("192.168.1.77", path).await, 200, "{path} from the LAN");
        assert_eq!(status("127.0.0.1", path).await, 200, "{path} from loopback");
        assert_eq!(status("203.0.113.5", path).await, 403, "{path} from elsewhere");
    }
    // The API wants the token too; the gate refuses an outsider before the token is looked at.
    assert_eq!(status("157.250.168.190", "/api/status").await, 401);
    assert_eq!(status("203.0.113.5", "/api/status").await, 403);
    // The workers' socket is not gated (a plain GET is refused for not being an upgrade, not 403).
    assert_ne!(status("203.0.113.5", "/ws").await, 403);
}

#[test]
fn allow_net_parses_addresses_and_networks() {
    use botbowl_hub::AllowNet;
    let one: AllowNet = "157.250.168.190".parse().unwrap();
    assert!(one.contains("157.250.168.190".parse().unwrap()));
    assert!(!one.contains("157.250.168.191".parse().unwrap()));
    let lan: AllowNet = "192.168.0.0/16".parse().unwrap();
    assert!(lan.contains("192.168.1.146".parse().unwrap()));
    assert!(!lan.contains("10.0.0.1".parse().unwrap()));
    // An IPv4 peer on a dual-stack socket arrives IPv4-mapped.
    assert!(one.contains("::ffff:157.250.168.190".parse().unwrap()));
    let all: AllowNet = "0.0.0.0/0".parse().unwrap();
    assert!(all.contains("8.8.8.8".parse().unwrap()));
    assert!("1.2.3.4/33".parse::<AllowNet>().is_err());
    assert!("nonsense".parse::<AllowNet>().is_err());
}
