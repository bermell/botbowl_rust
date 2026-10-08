//! The hub's browser-facing layout: `/` an index, `/status` the status page, `/registry/` the
//! project registry, `/play/` the web play app nested whole (its own `/ws` included), and `/ws`
//! still the workers'.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use axum::routing::get;
use axum::Router;
use botbowl_hub::api::Counts;
use botbowl_hub::http::request;
use botbowl_hub::{Hub, HubConfig};

async fn get_path(addr: std::net::SocketAddr, path: &str) -> (u16, String) {
    let url = format!("http://{addr}{path}");
    tokio::task::spawn_blocking(move || request("GET", &url, "", None).expect("request"))
        .await
        .unwrap()
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "botbowl-hub-pages-test-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn config(run_dir: Option<PathBuf>, registry_dir: Option<PathBuf>) -> HubConfig {
    HubConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        token: "t".into(),
        allow_commit_mismatch: false,
        allowed_commits: "/nonexistent".into(),
        worker_timeout: Duration::from_secs(120),
        run_dir,
        allow_from: Vec::new(),
        rate_interval: Duration::from_secs(300),
        registry_dir,
    }
}

async fn start(play: Option<Router>) -> std::net::SocketAddr {
    start_hub(config(None, None), play).await.1
}

async fn start_hub(cfg: HubConfig, play: Option<Router>) -> (Hub, std::net::SocketAddr) {
    let (hub, addr, _task) = Hub::start_with(cfg, play).await.unwrap();
    (hub, addr)
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
        body.contains("href=\"/status\"") && body.contains("href=\"/play/\"") && body.contains("href=\"/registry/\""),
        "{body}"
    );

    let (code, body) = get_path(addr, "/status").await;
    assert_eq!(code, 200);
    assert!(body.contains("botbowl hub · commit"), "{body}");
    assert!(body.contains("href=\"/registry/\""), "{body}");
    // No --run-dir: no loop section, and no link to its status file.
    assert!(!body.contains("/run/status.md"), "{body}");
    assert_eq!(get_path(addr, "/run/status.md").await.0, 404);

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
        allow_from: vec!["157.250.168.190".parse().unwrap(), "192.168.1.0/24".parse().unwrap()],
        ..config(None, Some(tmp("gate-registry")))
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
    for path in ["/", "/status", "/play/", "/registry/"] {
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

/// `/registry/`: the registry's markdown, read from disk per request, tables rendered, files
/// linked to each other; nothing outside the directory is reachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_registry_is_browsable_and_read_per_request() {
    let root = tmp("registry");
    let dir = root.join("registry");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("DATA.md"),
        "# Data registry — corpora\n\nWhat was generated. Nets are in [NETS.md](NETS.md).\n\n\
         ## Loop\n\n| gen | samples |\n|---|---|\n| 09 | 204,956 |\n",
    )
    .unwrap();
    std::fs::write(dir.join("NETS.md"), "# Network registry\n\nOne entry per net.\n").unwrap();
    std::fs::write(dir.join("notes.txt"), "not markdown").unwrap();
    std::fs::write(root.join("secret.md"), "# outside\n").unwrap();
    let (_hub, addr) = start_hub(config(None, Some(dir.clone())), None).await;

    // Relative links need the slash.
    assert_eq!(get_path(addr, "/registry").await.0, 308);
    let (code, body) = get_path(addr, "/registry/").await;
    assert_eq!(code, 200);
    assert!(
        body.contains("Data registry — corpora") && body.contains("Network registry"),
        "{body}"
    );
    assert!(
        body.contains("href=\"DATA.md\"") && body.contains("href=\"raw/DATA.md\""),
        "{body}"
    );
    // DATA before NETS (the registry's reading order), and only markdown is listed.
    assert!(
        body.find("href=\"DATA.md\"").unwrap() < body.find("href=\"NETS.md\"").unwrap(),
        "{body}"
    );
    assert!(!body.contains("notes.txt"), "{body}");

    let (code, body) = get_path(addr, "/registry/DATA.md").await;
    assert_eq!(code, 200);
    assert!(body.contains("<table>") && body.contains("<td>204,956</td>"), "{body}");
    assert!(body.contains("<h2 id=\"loop\">"), "{body}");
    assert!(body.contains("<a href=\"NETS.md\">NETS.md</a>"), "{body}");
    assert!(body.contains("href=\"/status\""), "the hub's nav: {body}");

    let (code, body) = get_path(addr, "/registry/raw/NETS.md").await;
    assert_eq!(
        (code, body.as_str()),
        (200, "# Network registry\n\nOne entry per net.\n")
    );

    // An edit shows on the next request, no restart.
    std::fs::write(
        dir.join("NETS.md"),
        "# Network registry\n\n| net | val |\n|---|---|\n| gen09 | 0.91 |\n",
    )
    .unwrap();
    let (_, body) = get_path(addr, "/registry/NETS.md").await;
    assert!(body.contains("<td>gen09</td>"), "{body}");

    for path in [
        "/registry/notes.txt",
        "/registry/missing.md",
        "/registry/..%2Fsecret.md",
        "/registry/raw/..%2Fsecret.md",
    ] {
        assert_eq!(get_path(addr, path).await.0, 404, "{path}");
    }

    // A hub without a registry says so.
    let addr = start(None).await;
    assert_eq!(get_path(addr, "/registry/").await.0, 404);
}

/// With `--run-dir`, the page shows the loop's latest status lines and links the whole file.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_status_page_shows_the_loop_and_links_its_status_file() {
    let run = tmp("run").join("loopmix16x9v9");
    std::fs::create_dir_all(&run).unwrap();
    let lines: Vec<String> = (0..30)
        .map(|i| format!("[2026-10-08 12:{i:02}:00] gen09 generate: line {i}"))
        .collect();
    std::fs::write(run.join("status.md"), lines.join("\n") + "\n").unwrap();
    let (_hub, addr) = start_hub(config(Some(run.clone()), None), None).await;

    let (code, body) = get_path(addr, "/status").await;
    assert_eq!(code, 200);
    assert!(body.contains("loop loopmix16x9v9"), "{body}");
    assert!(body.contains("10-08 12:29 gen09 generate: line 29"), "{body}");
    assert!(!body.contains("line 3\n"), "only the tail: {body}");
    assert!(body.contains("href=\"/run/status.md\""), "{body}");

    let (code, text) = get_path(addr, "/run/status.md").await;
    assert_eq!(code, 200);
    assert_eq!(text.lines().count(), 30);
    assert!(text.starts_with("[2026-10-08 12:00:00]"), "{text}");
}

/// Throughput on the page: per worker name (connected or gone) and in total, over the windows,
/// with the interval history — the thing that tells which worker a change of pace came from.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_status_page_shows_each_workers_rate_and_its_history() {
    let (hub, addr) = start_hub(config(None, None), None).await;
    {
        let mut inner = hub.inner.lock().unwrap();
        // A hub up for an hour (ledger time is passed in, so the past can be written).
        let now = Instant::now();
        let hour = Duration::from_secs(3600);
        inner.ledger = botbowl_hub::rates::Ledger::new(now - hour, std::time::SystemTime::now() - hour);
        // Both joined when the hub started (a later connection keeps the first join time).
        inner.ledger.joined("local", now - hour);
        inner.ledger.joined("laptop", now - hour);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        inner.add_worker(botbowl_hub::state::WorkerConn {
            name: "local".into(),
            build: botbowl_hub_proto::BuildInfo::current(),
            triple: "x86_64-unknown-linux-musl".into(),
            cores: 8,
            ram_mb: 16000,
            parallel_games: 24,
            tx,
            known_models: Default::default(),
            named_models: Default::default(),
            tasks: Default::default(),
            games_done: 0,
            last_seen: now,
        });
        let gen = |samples| Counts {
            games: 1,
            records: 1,
            samples,
            ..Default::default()
        };
        // local: a game every 30 s for the whole hour; the laptop: one a minute until 20
        // minutes ago, when it left.
        for k in (1..120).rev() {
            if k % 2 == 0 && k >= 40 {
                inner
                    .ledger
                    .record(now - Duration::from_secs(30 * k), "laptop", 0, gen(20));
            }
            inner
                .ledger
                .record(now - Duration::from_secs(30 * k), "local", 0, gen(30));
        }
    }
    let (code, body) = get_path(addr, "/status").await;
    assert_eq!(code, 200);
    assert!(body.contains("workers (1 connected)"), "{body}");
    assert!(body.contains("local · 0/24 streams busy · 8 cores"), "{body}");
    assert!(body.contains("laptop · gone"), "{body}");
    assert!(body.contains("total · 0/24 streams busy"), "{body}");
    assert!(body.contains("    decisions"), "{body}");
    // The sparkline and the interval table: the laptop's column shows where it stopped.
    assert!(body.contains("games/min, "), "{body}");
    assert!(
        body.contains("by 5-min interval: generate games/min · decisions/min"),
        "{body}"
    );
    let table: Vec<&str> = body.lines().skip_while(|l| !l.contains("  ending")).take(13).collect();
    assert!(
        table[0].contains("local") && table[0].contains("laptop") && table[0].contains("total"),
        "{body}"
    );
    // Newest interval first: the laptop had nothing in it, local 2 games/min, 60 decisions/min.
    assert!(table[1].contains('–') && table[1].contains("2.0·60"), "{table:?}");

    // The same numbers in the JSON API.
    let url = format!("http://{addr}/api/status");
    let (code, json) = tokio::task::spawn_blocking(move || request("GET", &url, "t", None).unwrap())
        .await
        .unwrap();
    assert_eq!(code, 200);
    let s: botbowl_hub::api::HubStatus = serde_json::from_str(&json).unwrap();
    let local = s.throughput.workers.iter().find(|w| w.name == "local").unwrap();
    assert_eq!(local.since_start.counts.games, 119);
    assert_eq!(s.throughput.total.since_start.counts.games, 119 + 40);
    assert!(
        (local.windows[1].per_min(local.windows[1].counts.games) - 2.0).abs() < 0.1,
        "{local:?}"
    );
}
