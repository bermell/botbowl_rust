//! `botbowl-hub`: the job queue and result sink of plan 041.
//!
//! One axum server exposes
//!
//! - `GET /ws` — the worker websocket ([`ws`]);
//! - `POST /api/jobs` (an [`api::JobRequest`]: eval or generate),
//!   `GET /api/jobs/{id}`, `GET /api/status` — the control API the
//!   `botbowl-hub job` CLI uses (bearer token);
//! - `GET /` — a plain-text status page for watching a run from a phone.
//!
//! All state is [`state::Inner`] behind one mutex; `changed` wakes anyone
//! waiting on a job.

pub mod api;
pub mod http;
pub mod state;
pub mod ws;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

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
    /// Accept workers built from a different commit (plan 041 decision 5).
    pub allow_commit_mismatch: bool,
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
        Router::new()
            .route("/", get(status_page))
            .route("/ws", get(ws_upgrade))
            .route("/api/status", get(api_status))
            .route("/api/jobs", post(api_submit))
            .route("/api/jobs/{id}", get(api_job))
            .with_state(self.clone())
    }

    /// Bind and serve in the background. Returns the bound address (useful
    /// with port 0) and the server task.
    pub async fn start(cfg: HubConfig) -> std::io::Result<(Hub, SocketAddr, tokio::task::JoinHandle<()>)> {
        let hub = Hub::new(cfg);
        let listener = tokio::net::TcpListener::bind(hub.cfg.bind).await?;
        let addr = listener.local_addr()?;
        let router = hub.router();
        let task = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, router).await {
                eprintln!("[hub] server error: {e}");
            }
        });
        Ok((hub, addr, task))
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

async fn status_page(State(hub): State<Hub>) -> impl IntoResponse {
    let s = hub.inner.lock().unwrap().status();
    let mut out = format!(
        "botbowl-hub  commit {}{}\n\nworkers ({}):\n",
        &s.commit[..s.commit.len().min(12)],
        if s.dirty { "-dirty" } else { "" },
        s.workers.len()
    );
    for w in &s.workers {
        out.push_str(&format!(
            "  {:20} {:>3} streams  {:>3} tasks in flight  {:>6} games  {}  seen {}s ago\n",
            w.name, w.parallel_games, w.tasks_in_flight, w.games_done, w.triple, w.last_seen_secs
        ));
    }
    out.push_str(&format!("\njobs ({}):\n", s.jobs.len()));
    for j in &s.jobs {
        out.push_str(&format!(
            "  job {}  {:?}  {:?}  {}s\n",
            j.id, j.kind, j.state, j.elapsed_secs
        ));
        for u in &j.units {
            out.push_str(&format!(
                "      {:40} {:>5}/{:<5}{}\n",
                u.name,
                u.done,
                u.total,
                if u.samples > 0 {
                    format!("  {} samples", u.samples)
                } else {
                    String::new()
                }
            ));
        }
    }
    (StatusCode::OK, [("content-type", "text/plain; charset=utf-8")], out)
}
