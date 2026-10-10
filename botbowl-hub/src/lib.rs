//! `botbowl-hub`: the job queue and result sink of plan 041.
//!
//! One axum server exposes
//!
//! - `GET /ws` — the worker websocket ([`ws`]);
//! - `POST /api/jobs` (an [`api::JobRequest`]: eval, generate or label),
//!   `GET /api/jobs/{id}`, `GET /api/status` — the control API the
//!   `botbowl-hub job` CLI uses (bearer token);
//! - `GET /` — an index linking the pages below ([`page::render_index`]);
//! - `GET /status` — a status page for watching a run from a phone ([`page`]), with games and
//!   decisions per minute per worker ([`rates`]); `GET /run/status.md` the loop's status file;
//! - `GET /registry/` — the project registry, rendered ([`registry`]);
//! - `/play/` — the web play app (`botbowl-web-server`'s router, nested), when
//!   [`Hub::start_with`] is given one.
//!
//! All state is [`state::Inner`] behind one mutex; `changed` wakes anyone
//! waiting on a job.

pub mod allowlist;
pub mod api;
pub mod http;
pub mod label;
pub mod page;
pub mod rates;
pub mod registry;
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

use api::{EvalJobRequest, GenerateJobRequest, JobId, JobRequest, JobState, JobStatus, LabelJobRequest, Submitted};
use botbowl_hub_proto::{LabelResult, TaskId};
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
    /// The status page's rate history interval, and how often hub.log gets a `[hub] rate ...`
    /// line (only for intervals in which something finished). Aligned to the wall clock.
    pub rate_interval: Duration,
    /// The project registry (`registry/*.md`), served at `/registry/`. Read per request.
    pub registry_dir: Option<PathBuf>,
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
        let mut inner = Inner::default();
        inner.ledger.bucket = cfg.rate_interval.max(Duration::from_secs(60));
        Hub {
            cfg: Arc::new(cfg),
            inner: Arc::new(Mutex::new(inner)),
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
            .route("/run/status.md", get(run_status))
            .route(
                "/registry",
                get(|| async { axum::response::Redirect::permanent("/registry/") }),
            )
            .route("/registry/", get(registry_index))
            .route("/registry/{file}", get(registry_doc))
            .route("/registry/raw/{file}", get(registry_raw))
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
        let rates = hub.clone();
        let task = tokio::spawn(async move {
            // The loops never return, so the select ends with the server
            // and they are dropped with it — no task outlives its hub.
            tokio::select! {
                r = axum::serve(listener, router.into_make_service_with_connect_info::<SocketAddr>()) => {
                    if let Err(e) = r {
                        eprintln!("[hub] server error: {e}");
                    }
                }
                _ = reap_loop(reaper) => {}
                _ = rate_log_loop(rates) => {}
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
        let id = match req {
            // Plan 062: a generation's shards are ~1.5 GB of JSON. Read and compress them before
            // taking the lock, so workers' results and the status page are not held up.
            JobRequest::Label(r) => {
                let inputs = label::load_inputs(&r)?;
                let (id, writes) = self.inner.lock().unwrap().submit_label(r, inputs)?;
                for w in writes {
                    self.spawn_write(w);
                }
                id
            }
            other => self.inner.lock().unwrap().submit(other)?,
        };
        self.changed.notify_waiters();
        Ok(id)
    }

    pub fn submit_label(&self, req: LabelJobRequest) -> std::io::Result<JobId> {
        self.submit(JobRequest::Label(req))
    }

    /// Plan 062: one label item's result, from a worker. A shard whose last item this was is
    /// written on its own thread (parse, label and re-serialise ~180 MB), off the lock.
    pub fn label_done(&self, worker: state::WorkerId, task: TaskId, item: u32, result: LabelResult) {
        let write = self.inner.lock().unwrap().label_done(worker, task, item, result);
        if let Some(w) = write {
            self.spawn_write(w);
        }
        self.changed.notify_waiters();
    }

    fn spawn_write(&self, w: label::ShardWrite) {
        let hub = self.clone();
        std::thread::spawn(move || {
            let result = w.run();
            hub.inner.lock().unwrap().label_written(w.job, w.unit, result);
            hub.changed.notify_waiters();
        });
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

/// Write a `[hub] rate ...` line to stderr (the loop's hub.log) each time a wall-clock interval
/// completes with results in it: per worker, games and decisions per minute. The loop's logs then
/// record each worker's speed, so a change in a generation's pace can be attributed afterwards.
async fn rate_log_loop(hub: Hub) -> ! {
    let mut tick = tokio::time::interval(Duration::from_secs(5));
    let mut last: Option<u64> = None;
    loop {
        tick.tick().await;
        let line = {
            let inner = hub.inner.lock().unwrap();
            let now = std::time::Instant::now();
            let interval = inner.ledger.interval(now);
            let previous = last.replace(interval);
            // The first interval seen is the one the hub started in: nothing completed yet.
            if previous.is_none() || previous == Some(interval) {
                continue;
            }
            inner.rate_log_line(now)
        };
        if let Some(line) = line {
            eprintln!("[hub] {line}");
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
    // A label job reads its shards before it is queued (seconds): off the async workers.
    let submitter = hub.clone();
    match tokio::task::spawn_blocking(move || submitter.submit(req)).await {
        Ok(Ok(id)) => Json(Submitted { id }).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
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

const HTML: (&str, &str) = ("content-type", "text/html; charset=utf-8");
const TEXT: (&str, &str) = ("content-type", "text/plain; charset=utf-8");

/// The last lines of the loop's `status.md` this many; the page shows the newest few.
const RUN_STATUS_LINES: usize = 2000;

/// `GET /run/status.md`: the loop's own status lines, newest last (the tail, for a long run).
async fn run_status(State(hub): State<Hub>) -> axum::response::Response {
    let Some(dir) = hub.cfg.run_dir.clone() else {
        return (StatusCode::NOT_FOUND, "this hub has no --run-dir\n").into_response();
    };
    let text = tokio::task::spawn_blocking(move || std::fs::read_to_string(dir.join("status.md")))
        .await
        .ok()
        .and_then(|r| r.ok());
    match text {
        Some(t) => {
            let lines: Vec<&str> = t.lines().collect();
            let tail = lines[lines.len().saturating_sub(RUN_STATUS_LINES)..].join("\n");
            (StatusCode::OK, [TEXT], tail + "\n").into_response()
        }
        None => (StatusCode::NOT_FOUND, "no status.md in the run directory\n").into_response(),
    }
}

/// Run a registry renderer on the blocking pool, or 404 without a registry directory.
async fn with_registry<T: Send + 'static>(
    hub: &Hub,
    f: impl FnOnce(&std::path::Path) -> Option<T> + Send + 'static,
) -> Result<T, (StatusCode, &'static str)> {
    let Some(dir) = hub.cfg.registry_dir.clone() else {
        return Err((StatusCode::NOT_FOUND, "this hub serves no registry (--registry-dir)\n"));
    };
    tokio::task::spawn_blocking(move || f(&dir))
        .await
        .ok()
        .flatten()
        .ok_or((StatusCode::NOT_FOUND, "no such registry file\n"))
}

async fn registry_index(State(hub): State<Hub>) -> axum::response::Response {
    match with_registry(&hub, |d| Some(registry::render_index(d))).await {
        Ok(html) => (StatusCode::OK, [HTML], html).into_response(),
        Err(r) => r.into_response(),
    }
}

async fn registry_doc(State(hub): State<Hub>, Path(file): Path<String>) -> axum::response::Response {
    match with_registry(&hub, move |d| registry::render_doc(d, &file)).await {
        Ok(html) => (StatusCode::OK, [HTML], html).into_response(),
        Err(r) => r.into_response(),
    }
}

async fn registry_raw(State(hub): State<Hub>, Path(file): Path<String>) -> axum::response::Response {
    match with_registry(&hub, move |d| registry::read(d, &file)).await {
        Ok(text) => (StatusCode::OK, [TEXT], text).into_response(),
        Err(r) => r.into_response(),
    }
}
