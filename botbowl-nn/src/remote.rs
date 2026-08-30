//! Client for the batched inference sidecar (`scripts/nn_server.py`, plan 024).
//!
//! MCTS leaf evaluation is sequential — the next leaf is only known once
//! the current one is backed up — so batch parallelism has to come from
//! *somewhere else*. Here it comes from the generation loop's independent
//! shard processes: each one issues one blocking forward at a time over a
//! Unix-domain socket, and the server batches whatever has arrived. The
//! Rust side stays synchronous and nothing above [`forward_raw`] changes.
//!
//! # Wire protocol (little-endian, `SOCK_STREAM`)
//!
//! One request, one response, per connection; each client thread owns its
//! own connection, so there are no request IDs and no multiplexing.
//!
//! ```text
//! handshake  →  magic "BBNN" | u16 version | u16 C | u16 F | u16 path_len | utf8 model_path
//! handshake  ←  u16 status | u16 model_id | u16 h | u16 w | u32 len | payload
//!               status 0: payload = f32 canary_value | f32[A*h*w] canary_policy
//!               status ≠0: payload = utf8 error message
//! request    →  u32 len | u16 model_id | u16 flags | u16 h | u16 w
//!                       | f32[C*h*w] spatial | f32[F] global      (len covers all after itself)
//! response   ←  u32 len | f32 value | f32[A*h*w] policy if flags&WANT_POLICY
//! ```
//!
//! # The canary handshake is the safety interlock
//!
//! A connection is bound to one model at handshake. For that model the
//! server runs the committed parity fixture and returns *its* result; the
//! client runs the same fixture through tract using the ONNX path it was
//! given and compares to `< 1e-3`. A mismatch means the server resolved
//! the path to different weights, a different tier, or broken kernels —
//! and the client refuses to run rather than write a corpus (or a
//! promotion-gate verdict) labelled with the wrong network.
//!
//! # Failure is a fallback, not an abort
//!
//! Any connect/IO error or timeout drops the connection and the caller
//! silently uses tract for that forward: a shard gets slower but
//! *finishes*, preserving today's "a dead shard is a warning" semantics.
//! One `NN_SERVER_FALLBACK` warning is emitted, and the count is reported
//! at exit. A *canary* mismatch is the one thing that is fatal, because
//! its failure mode is silent corruption rather than slowness.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::actions::POLICY_CHANNELS;
use crate::encode::{GLOBAL_FEATURES, SPATIAL_CHANNELS};

/// Frame magic; guards against pointing `--nn-server` at some other socket.
pub const MAGIC: [u8; 4] = *b"BBNN";
/// Protocol version. Bump on any frame-layout change.
pub const PROTOCOL_VERSION: u16 = 1;
/// `flags` bit 0 — ask for the policy map as well as the value.
pub const FLAG_WANT_POLICY: u16 = 1;
/// Read/write timeout. A wedged server must not hang a generation.
pub const IO_TIMEOUT: Duration = Duration::from_secs(5);
/// After a failure, don't hammer the socket: retry at most this often
/// (per thread).
const RECONNECT_BACKOFF: Duration = Duration::from_secs(60);
/// Canary agreement threshold — looser than the tract-vs-torch parity
/// test's 1e-4 because GPU fp32 reassociates convolution reductions.
pub const CANARY_TOL: f32 = 1e-3;

/// The committed parity fixture doubles as the canary input: it is the
/// 14x7 tier's exact shape (16x9 board + border → `h=9, w=16`), it is
/// already in the repo, and both sides already know how to read it.
const CANARY_SPATIAL_NPY: &[u8] = include_bytes!("../tests/fixtures/parity_9x16_spatial.npy");
const CANARY_GLOBAL_NPY: &[u8] = include_bytes!("../tests/fixtures/parity_9x16_global.npy");
/// Canary board dims (H, W).
pub const CANARY_H: usize = 9;
pub const CANARY_W: usize = 16;

/// `(spatial, global)` canary input — the committed parity fixture.
pub fn canary_input() -> (Vec<f32>, Vec<f32>) {
    let spatial = crate::npy::parse(CANARY_SPATIAL_NPY).expect("canary spatial fixture").as_f32();
    let global = crate::npy::parse(CANARY_GLOBAL_NPY).expect("canary global fixture").as_f32();
    (spatial, global)
}

/// Why a remote forward could not be served. Every variant means "use
/// tract for this call".
#[derive(Debug)]
pub enum RemoteError {
    /// Could not connect, or the connection was in cool-down after a
    /// previous failure.
    Unavailable(String),
    /// The server answered, but not with something we can use.
    Protocol(String),
    /// The server's canary disagrees with our tract result. Fatal —
    /// never falls back, because falling back would hide a wrong net.
    Canary(String),
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RemoteError::Unavailable(m) => write!(f, "nn-server unavailable: {m}"),
            RemoteError::Protocol(m) => write!(f, "nn-server protocol error: {m}"),
            RemoteError::Canary(m) => write!(f, "nn-server canary mismatch: {m}"),
        }
    }
}

impl std::error::Error for RemoteError {}

/// A live, handshaked connection bound to one model.
struct Conn {
    stream: UnixStream,
    model_id: u16,
}

enum Slot {
    Live(Conn),
    /// Failed at this instant; retry after [`RECONNECT_BACKOFF`].
    Dead(Instant),
}

thread_local! {
    /// One connection per (thread, client). Keyed by the client's id so
    /// two `NnEvaluator`s in one process (the eval phase's candidate and
    /// champion) get independent streams over the same socket.
    static CONNS: RefCell<HashMap<usize, Slot>> = RefCell::new(HashMap::new());
}

static NEXT_CLIENT_ID: AtomicUsize = AtomicUsize::new(0);

/// Thin, blocking client for one `(socket, model)` pair.
///
/// `Sync` without a lock: state that cannot be shared lives in a
/// `thread_local!`, so parallel MCTS workers (and, later, parallel games)
/// each hold their own stream and the server sees them as independent
/// batchable requests.
pub struct RemoteClient {
    socket: PathBuf,
    model_path: String,
    id: usize,
    /// tract's own answer on the canary input, computed once at
    /// construction and compared against every connection's handshake.
    expected_canary: (f32, Vec<f32>),
    warned: AtomicBool,
    fallbacks: AtomicU64,
    requests: AtomicU64,
}

impl std::fmt::Debug for RemoteClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RemoteClient{{socket: {:?}, model: {}}}", self.socket, self.model_path)
    }
}

impl RemoteClient {
    /// Build a client and, if the server is reachable, handshake once on
    /// the current thread to validate the canary.
    ///
    /// An *unreachable* server is not an error — the caller falls back to
    /// tract. A canary *mismatch* is.
    pub fn new(
        socket: impl AsRef<Path>,
        model_path: &str,
        expected_canary: (f32, Vec<f32>),
    ) -> Result<Self, RemoteError> {
        let client = RemoteClient {
            socket: socket.as_ref().to_path_buf(),
            model_path: model_path.to_string(),
            id: NEXT_CLIENT_ID.fetch_add(1, Ordering::Relaxed),
            expected_canary,
            warned: AtomicBool::new(false),
            fallbacks: AtomicU64::new(0),
            requests: AtomicU64::new(0),
        };
        // Eagerly handshake so a wrong-net server is caught before a
        // single sample is generated, not on some worker thread later.
        match client.open() {
            Ok(conn) => {
                CONNS.with(|c| c.borrow_mut().insert(client.id, Slot::Live(conn)));
                eprintln!(
                    "NN_SERVER connected: {} model={} (canary ok)",
                    client.socket.display(),
                    client.model_path
                );
            }
            Err(e @ RemoteError::Canary(_)) => return Err(e),
            Err(e) => client.warn_fallback(&e),
        }
        Ok(client)
    }

    /// Number of forwards that fell back to tract.
    pub fn fallback_count(&self) -> u64 {
        self.fallbacks.load(Ordering::Relaxed)
    }

    /// Number of forwards served remotely.
    pub fn served_count(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    fn warn_fallback(&self, e: &RemoteError) {
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!("NN_SERVER_FALLBACK {e} — using tract for this process; further warnings suppressed");
        }
    }

    /// Open + handshake + verify the canary.
    fn open(&self) -> Result<Conn, RemoteError> {
        let stream =
            UnixStream::connect(&self.socket).map_err(|e| RemoteError::Unavailable(format!("{e} ({:?})", self.socket)))?;
        stream.set_read_timeout(Some(IO_TIMEOUT)).ok();
        stream.set_write_timeout(Some(IO_TIMEOUT)).ok();
        let mut conn = Conn { stream, model_id: 0 };

        let path_bytes = self.model_path.as_bytes();
        let mut frame = Vec::with_capacity(12 + path_bytes.len());
        frame.extend_from_slice(&MAGIC);
        frame.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        frame.extend_from_slice(&(SPATIAL_CHANNELS as u16).to_le_bytes());
        frame.extend_from_slice(&(GLOBAL_FEATURES as u16).to_le_bytes());
        frame.extend_from_slice(&(path_bytes.len() as u16).to_le_bytes());
        frame.extend_from_slice(path_bytes);
        conn.stream.write_all(&frame).map_err(io_err)?;

        let mut head = [0u8; 12];
        conn.stream.read_exact(&mut head).map_err(io_err)?;
        let status = u16::from_le_bytes([head[0], head[1]]);
        let model_id = u16::from_le_bytes([head[2], head[3]]);
        let h = u16::from_le_bytes([head[4], head[5]]) as usize;
        let w = u16::from_le_bytes([head[6], head[7]]) as usize;
        let len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        let mut payload = vec![0u8; len];
        conn.stream.read_exact(&mut payload).map_err(io_err)?;
        if status != 0 {
            return Err(RemoteError::Protocol(format!(
                "handshake rejected (status {status}): {}",
                String::from_utf8_lossy(&payload)
            )));
        }
        conn.model_id = model_id;

        let expect_len = 4 + 4 * POLICY_CHANNELS * h * w;
        if (h, w) != (CANARY_H, CANARY_W) || len != expect_len {
            return Err(RemoteError::Canary(format!(
                "server canary is {h}x{w} / {len} B, expected {CANARY_H}x{CANARY_W} / {expect_len} B"
            )));
        }
        let value = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let policy = decode_f32(&payload[4..]);
        self.check_canary(value, &policy)?;
        Ok(conn)
    }

    fn check_canary(&self, value: f32, policy: &[f32]) -> Result<(), RemoteError> {
        let (ev, ep) = &self.expected_canary;
        let dv = (value - ev).abs();
        if dv >= CANARY_TOL {
            return Err(RemoteError::Canary(format!(
                "value {value} vs tract {ev} (|Δ| = {dv} ≥ {CANARY_TOL}) for model {}",
                self.model_path
            )));
        }
        if policy.len() != ep.len() {
            return Err(RemoteError::Canary(format!(
                "policy length {} vs tract {} for model {}",
                policy.len(),
                ep.len(),
                self.model_path
            )));
        }
        let dp = policy.iter().zip(ep).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        if dp >= CANARY_TOL {
            return Err(RemoteError::Canary(format!(
                "policy max-abs-diff {dp} ≥ {CANARY_TOL} for model {}",
                self.model_path
            )));
        }
        Ok(())
    }

    /// One blocking remote forward. `Err` means "fall back to tract".
    ///
    /// A canary mismatch discovered on a *new* connection is fatal and
    /// exits the process: it means the server is now serving different
    /// weights than the ones this corpus is being labelled with.
    pub fn forward(
        &self,
        spatial: &[f32],
        global: &[f32],
        h: usize,
        w: usize,
        want_policy: bool,
    ) -> Result<(Vec<f32>, f32), RemoteError> {
        let r = CONNS.with(|cell| {
            let mut map = cell.borrow_mut();
            match map.get(&self.id) {
                Some(Slot::Dead(t)) if t.elapsed() < RECONNECT_BACKOFF => {
                    return Err(RemoteError::Unavailable("in reconnect cool-down".into()))
                }
                Some(Slot::Live(_)) => {}
                _ => match self.open() {
                    Ok(conn) => {
                        map.insert(self.id, Slot::Live(conn));
                    }
                    Err(e @ RemoteError::Canary(_)) => {
                        eprintln!("FATAL {e}");
                        eprintln!("      refusing to generate data labelled with a network the server is not serving.");
                        std::process::exit(1);
                    }
                    Err(e) => {
                        map.insert(self.id, Slot::Dead(Instant::now()));
                        return Err(e);
                    }
                },
            }
            let Some(Slot::Live(conn)) = map.get_mut(&self.id) else {
                unreachable!("live connection inserted above");
            };
            match exchange(conn, spatial, global, h, w, want_policy) {
                Ok(out) => Ok(out),
                Err(e) => {
                    // A broken stream is unrecoverable mid-frame; drop it
                    // and let the backoff decide when to try again.
                    map.insert(self.id, Slot::Dead(Instant::now()));
                    Err(e)
                }
            }
        });
        match &r {
            Ok(_) => {
                self.requests.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                self.fallbacks.fetch_add(1, Ordering::Relaxed);
                self.warn_fallback(e);
            }
        }
        r
    }
}

/// Write one request frame and read its response.
fn exchange(
    conn: &mut Conn,
    spatial: &[f32],
    global: &[f32],
    h: usize,
    w: usize,
    want_policy: bool,
) -> Result<(Vec<f32>, f32), RemoteError> {
    let body_len = 8 + 4 * spatial.len() + 4 * global.len();
    let mut frame = Vec::with_capacity(4 + body_len);
    frame.extend_from_slice(&(body_len as u32).to_le_bytes());
    frame.extend_from_slice(&conn.model_id.to_le_bytes());
    frame.extend_from_slice(&(if want_policy { FLAG_WANT_POLICY } else { 0 }).to_le_bytes());
    frame.extend_from_slice(&(h as u16).to_le_bytes());
    frame.extend_from_slice(&(w as u16).to_le_bytes());
    encode_f32(&mut frame, spatial);
    encode_f32(&mut frame, global);
    // One write_all: the payload is ~21 KB, and splitting it costs a
    // syscall per piece on the hot path.
    conn.stream.write_all(&frame).map_err(io_err)?;

    let mut len_buf = [0u8; 4];
    conn.stream.read_exact(&mut len_buf).map_err(io_err)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let expect = if want_policy { 4 + 4 * POLICY_CHANNELS * h * w } else { 4 };
    if len != expect {
        return Err(RemoteError::Protocol(format!("response is {len} B, expected {expect} B")));
    }
    let mut payload = vec![0u8; len];
    conn.stream.read_exact(&mut payload).map_err(io_err)?;
    let value = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let policy = if want_policy { decode_f32(&payload[4..]) } else { Vec::new() };
    Ok((policy, value))
}

fn io_err(e: io::Error) -> RemoteError {
    match e.kind() {
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => {
            RemoteError::Unavailable(format!("timed out after {IO_TIMEOUT:?}"))
        }
        _ => RemoteError::Unavailable(e.to_string()),
    }
}

fn encode_f32(out: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canary_fixture_has_the_tier_shape() {
        let (spatial, global) = canary_input();
        assert_eq!(spatial.len(), SPATIAL_CHANNELS * CANARY_H * CANARY_W);
        assert_eq!(global.len(), GLOBAL_FEATURES);
    }

    #[test]
    fn f32_codec_round_trips() {
        let xs = vec![0.0f32, -1.5, 3.25, f32::MIN_POSITIVE];
        let mut buf = Vec::new();
        encode_f32(&mut buf, &xs);
        assert_eq!(buf.len(), 16);
        assert_eq!(decode_f32(&buf), xs);
    }
}
