//! `botbowl-hub`: the job queue and result sink of plan 041.
//!
//! One axum server exposes
//!
//! - `GET /ws` — the worker websocket ([`ws`]);
//! - `POST /api/jobs` (an [`api::JobRequest`]: eval or generate),
//!   `GET /api/jobs/{id}`, `GET /api/status` — the control API the
//!   `botbowl-hub job` CLI uses (bearer token);
//! - `GET /` — an index linking the pages below ([`page::render_index`]);
//! - `GET /status` — a status page for watching a run from a phone ([`page`]);
//! - `/play/` — the web play app (`botbowl-web-server`'s router, nested), when
//!   [`Hub::start_with`] is given one.
//!
//! All state is [`state::Inner`] behind one mutex; `changed` wakes anyone
//! waiting on a job.

pub mod allowlist;
pub mod api;
pub mod http;
pub mod page;
pub mod state;
pub mod ws;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::Notify;

use api::{EvalJobRequest, GenerateJobRequest, JobId, JobRequest, JobState, JobStatus, Submitted};
use state::Inner;

#[derive(Clone, Debug)]
pub struct HubConfig {
    pub bind: SocketAddr,
    pub token: String,
    /// Accept workers built from *any* commit (plan 041 decision 5). The blunt instrument,
    /// for hacking on the worker; a programme uses `allowed_commits` instead.
    pub allow_commit_mismatch: bool,
    /// Path to the untracked per-commit allowlist ([`allowlist`]). Read on every handshake, so
    /// editing it admits waiting workers on their next reconnect without restarting the hub.
    pub allowed_commits: PathBuf,
    /// Drop a worker that has not been heard from for this long, and requeue
    /// the games it was holding. Workers heartbeat every 30 s whether or not
    /// they are mid-game, so this is about a lost *machine*, not a slow one.
    pub worker_timeout: Duration,
    /// The training loop's run directory, for the status page's loop and training lines.
    pub run_dir: Option<PathBuf>,
    /// Client addresses the browser-facing routes answer (`/`, `/status`, `/api/*`, `/play/`).
    /// Empty = everyone (the default). Loopback is always allowed (the loop's own job client), and
    /// `/ws` never checks: workers dial in from anywhere and authenticate with the token. Binding
    /// to an address does not do this — a server can only bind to its own interfaces — so a
    /// tool that should be seen only from, say, an office VPN filters on the peer address here.
    pub allow_from: Vec<AllowNet>,
}

/// One entry of [`HubConfig::allow_from`]: an address, or a network in CIDR form (`10.0.0.0/8`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllowNet {
    addr: std::net::IpAddr,
    prefix: u8,
}

impl std::str::FromStr for AllowNet {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let (a, p) = s.split_once('/').map_or((s, None), |(a, p)| (a, Some(p)));
        let addr: std::net::IpAddr = a.parse().map_err(|e| format!("{s}: {e}"))?;
        let max = if addr.is_ipv4() { 32 } else { 128 };
        let prefix = match p {
            Some(p) => p.parse::<u8>().map_err(|e| format!("{s}: prefix: {e}"))?,
            None => max,
        };
        if prefix > max {
            return Err(format!("{s}: prefix {prefix} > {max}"));
        }
        Ok(AllowNet { addr, prefix })
    }
}

impl AllowNet {
    /// Whether `ip` is in this network. An IPv4 peer reaching an IPv6 socket arrives as an
    /// IPv4-mapped address and is compared as the IPv4 address it is.
    pub fn contains(&self, ip: std::net::IpAddr) -> bool {
        use std::net::IpAddr;
        let bits = |n: u128, width: u32| {
            if self.prefix == 0 {
                0
            } else {
                n >> (width - self.prefix as u32)
            }
        };
        match (self.addr, ip.to_canonical()) {
            (IpAddr::V4(a), IpAddr::V4(b)) => bits(u32::from(a) as u128, 32) == bits(u32::from(b) as u128, 32),
            (IpAddr::V6(a), IpAddr::V6(b)) => bits(u128::from(a), 128) == bits(u128::from(b), 128),
            _ => false,
        }
    }
}

/// The [`HubConfig::allow_from`] gate, in front of every route.
async fn allow_from_gate(
    State(hub): State<Hub>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let allow = &hub.cfg.allow_from;
    if allow.is_empty() || req.uri().path() == "/ws" {
        return next.run(req).await;
    }
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip());
    match peer {
        Some(ip) if ip.to_canonical().is_loopback() || allow.iter().any(|n| n.contains(ip)) => next.run(req).await,
        _ => (StatusCode::FORBIDDEN, "not available from this address\n").into_response(),
    }
}

#[derive(Clone)]
pub struct Hub {
    pub cfg: Arc<HubConfig>,
    pub inner: Arc<Mutex<Inner>>,
    pub changed: Arc<Notify>,
}

impl Hub {
    pub fn new(cfg: HubConfig) -> Self {
        Hub {
            cfg: Arc::new(cfg),
            inner: Arc::new(Mutex::new(Inner::default())),
            changed: Arc::new(Notify::new()),
        }
    }

    pub fn router(&self) -> Router {
        self.router_with(None)
    }

    /// The hub's routes, plus the web play app nested under `/play/` when one is given.
    ///
    /// The play app is a complete router of its own (its own state, its own `/ws` for game
    /// sockets), so it is nested as a service: `/play/ws` is a game, `/ws` stays the workers'.
    /// The client is built with relative URLs, which resolve against `/play/` only with the
    /// trailing slash — hence the redirect.
    pub fn router_with(&self, play: Option<Router>) -> Router {
        let has_play = play.is_some();
        let mut router = Router::new()
            .route("/", get(move || index_page(has_play)))
            .route("/status", get(status_page))
            .route("/ws", get(ws_upgrade))
            .route("/api/status", get(api_status))
            .route("/api/jobs", post(api_submit))
            .route("/api/jobs/{id}", get(api_job));
        if let Some(play) = play {
            router = router
                .route("/play", get(|| async { axum::response::Redirect::permanent("/play/") }))
                .nest_service("/play/", play);
        }
        router
            .layer(axum::middleware::from_fn_with_state(self.clone(), allow_from_gate))
            .with_state(self.clone())
    }

    /// Bind and serve in the background. Returns the bound address (useful
    /// with port 0) and the server task.
    pub async fn start(cfg: HubConfig) -> std::io::Result<(Hub, SocketAddr, tokio::task::JoinHandle<()>)> {
        Self::start_with(cfg, None).await
    }

    /// [`Hub::start`], also serving the web play app under `/play/`.
    pub async fn start_with(
        cfg: HubConfig,
        play: Option<Router>,
    ) -> std::io::Result<(Hub, SocketAddr, tokio::task::JoinHandle<()>)> {
        let hub = Hub::new(cfg);
        let listener = tokio::net::TcpListener::bind(hub.cfg.bind).await?;
        let addr = listener.local_addr()?;
        let router = hub.router_with(play);
        let reaper = hub.clone();
        let task = tokio::spawn(async move {
            // The reaper never returns, so the select ends with the server
            // and the loop is dropped with it — no task outlives its hub.
            tokio::select! {
                r = axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>()) => {
                    if let Err(e) = r {
                        eprintln!("[hub] server error: {e}");
                    }
                }
                _ = reap_loop(reaper) => {}
            }
        });
        Ok((hub, addr, task))
    }

    /// Hash every `.onnx` under `dirs` on a background thread, then name each connected worker's
    /// cached copies of them (`ToWorker::ModelName`); later connections are named on arrival.
    /// The training box's `runs/` holds every net the loop ever shipped, so this is what turns a
    /// helper box's hash-named cache back into `bbnet_..._genNN.onnx`.
    pub fn index_models(&self, dirs: Vec<PathBuf>) -> std::thread::JoinHandle<usize> {
        let inner = Arc::clone(&self.inner);
        std::thread::spawn(move || {
            let index = state::index_models(&dirs);
            let n = index.len();
            let mut inner = inner.lock().unwrap();
            for (id, path) in index {
                inner.model_index.entry(id).or_insert(path);
            }
            inner.name_cached_models();
            n
        })
    }

    pub fn submit(&self, req: JobRequest) -> std::io::Result<JobId> {
        let id = self.inner.lock().unwrap().submit(req)?;
        self.changed.notify_waiters();
        Ok(id)
    }

    pub fn submit_eval(&self, req: EvalJobRequest) -> std::io::Result<JobId> {
        self.submit(JobRequest::Eval(req))
    }

    pub fn submit_generate(&self, req: GenerateJobRequest) -> std::io::Result<JobId> {
        self.submit(JobRequest::Generate(req))
    }

    pub fn job_status(&self, id: JobId) -> Option<JobStatus> {
        self.inner.lock().unwrap().job_status(id)
    }

    /// Resolve when the job leaves `Running`.
    pub async fn wait(&self, id: JobId) -> Option<JobStatus> {
        loop {
            let notified = self.changed.notified();
            match self.job_status(id) {
                None => return None,
                Some(s) if s.state != JobState::Running => return Some(s),
                Some(_) => notified.await,
            }
        }
    }
}

fn authed(hub: &Hub, headers: &HeaderMap) -> bool {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| t == hub.cfg.token)
}

/// Drop workers that have gone silent, for as long as the hub serves.
///
/// A worker whose *process* dies closes its socket and the websocket task
/// requeues its games immediately. A worker whose *machine* goes away — a
/// laptop that sleeps, a dropped VPN — leaves an ESTABLISHED socket the hub
/// cannot tell from a healthy one, and its in-flight games are stranded: the
/// job then sits at 4791/4800 with every live worker idle until someone
/// notices. (Seen 2026-09-18: gen10 generate stalled just under three hours
/// on nine games held by a slept laptop.) Heartbeats are what distinguishes
/// the two, so act on them.
async fn reap_loop(hub: Hub) -> ! {
    let timeout = hub.cfg.worker_timeout;
    let mut tick = tokio::time::interval((timeout / 4).max(Duration::from_millis(250)));
    loop {
        tick.tick().await;
        let dropped = hub.inner.lock().unwrap().reap_stale(timeout);
        if !dropped.is_empty() {
            hub.changed.notify_waiters();
        }
    }
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(hub): State<Hub>) -> impl IntoResponse {
    // Model frames are ~2 MB; leave headroom.
    ws.max_message_size(64 << 20)
        .on_upgrade(move |socket| ws::handle(socket, hub))
}

async fn api_status(State(hub): State<Hub>, headers: HeaderMap) -> impl IntoResponse {
    if !authed(&hub, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Json(hub.inner.lock().unwrap().status()).into_response()
}

async fn api_submit(State(hub): State<Hub>, headers: HeaderMap, Json(req): Json<JobRequest>) -> impl IntoResponse {
    if !authed(&hub, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    match hub.submit(req) {
        Ok(id) => Json(Submitted { id }).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

async fn api_job(State(hub): State<Hub>, headers: HeaderMap, Path(id): Path<JobId>) -> impl IntoResponse {
    if !authed(&hub, &headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    match hub.job_status(id) {
        Some(s) => Json(s).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn index_page(has_play: bool) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        page::render_index(has_play),
    )
}

async fn status_page(State(hub): State<Hub>) -> impl IntoResponse {
    let status = hub.inner.lock().unwrap().status();
    let port = hub.cfg.bind.port();
    let run_dir = hub.cfg.run_dir.clone();
    // File reads and an `nvidia-smi` call: off the async workers.
    let html = tokio::task::spawn_blocking(move || {
        page::render_html(
            &page::gather(status, port, run_dir.as_deref()),
            std::time::SystemTime::now(),
        )
    })
    .await
    .unwrap_or_else(|e| format!("status page failed: {e}"));
    (StatusCode::OK, [("content-type", "text/html; charset=utf-8")], html)
}
