//! The in-process batch queue (plan 066 §2): search threads hand their encoded state to a
//! preallocated slot and park; batcher threads take every ready slot, run them as one batch and
//! wake the owners.
//!
//! This is the sidecar's batching (`scripts/nn_server.py`'s `Server.serve`) moved into the search
//! process, without the socket, the frame codec or the Python loop. It is backend-agnostic — the
//! device work is an [`Executor`] — so the queue is compiled and unit-tested in every build, and
//! only the libtorch executor (`inproc.rs`) needs the `torch` feature.
//!
//! # The protocol
//!
//! * A **slot** holds one request: its canvas-staged input (spatial planes padded top-left onto the
//!   canvas, the globals, the mask of the real board — the sidecar's staging, done by the search
//!   thread itself and in parallel instead of by one Python thread) and its output row. Slots live
//!   in a fixed ring, allocated once; a slot's buffers are reused, and a slot whose last board had
//!   the same shape skips re-zeroing the padding.
//! * The owner claims a free slot, stages into it, pushes its index onto the ready list and parks:
//!   a short spin on the slot's done flag, then `thread::park` (a futex) until the batcher unparks
//!   it.
//! * A **batcher** thread (one per lane; two by default, the depth-2 pipeline the sidecar gets
//!   from its two pinned slots) takes every ready slot up to the executor's largest batch, runs
//!   them, writes each output row and wakes each owner.
//! * **No max-wait timer.** Like the sidecar, a batch is whatever arrived while the previous one
//!   was on the device. With a batch in flight, a free batcher holds the pending requests until the
//!   device is about [`JIT`] from idle (the sidecar's `JIT_S`), so they leave as one batch rather
//!   than a trickle of singles queued behind it; it never holds them with the device idle.
//!
//! # Failure is loud and never a hang
//!
//! An executor error fails every request in that batch with the message and marks the queue dead:
//! every later [`BatchQueue::forward`] returns `Err` at once. An owner that waits longer than
//! [`WAIT_TIMEOUT`] gives up with `Err` too (a wedged device must not hang a generation). The
//! caller ([`crate::eval::NnEvaluator`]) turns `Err` into a tract forward and one loud warning, as
//! it does for the sidecar.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

/// How long before the device goes idle a held batch launches (the sidecar's `JIT_S`).
pub const JIT: Duration = Duration::from_micros(250);
/// Seed of each bucket's device-time estimate (the sidecar's `eta_s` seed).
const ETA_SEED: Duration = Duration::from_micros(600);
/// How long an owner spins on its done flag before parking.
const SPIN: Duration = Duration::from_micros(20);
/// An owner waiting longer than this gives up (the request fails, the caller falls back).
pub const WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Tensor geometry of the queue: the canvas every request is staged onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Spatial channels `C`.
    pub c: usize,
    /// Global features `F`.
    pub f: usize,
    /// Policy channels `A`.
    pub a: usize,
    /// Canvas height (tensor rows, border included).
    pub canvas_h: usize,
    /// Canvas width.
    pub canvas_w: usize,
}

impl Geometry {
    /// `C·H·W` of the canvas.
    pub fn spatial_len(&self) -> usize {
        self.c * self.canvas_h * self.canvas_w
    }

    /// `H·W` of the canvas.
    pub fn plane(&self) -> usize {
        self.canvas_h * self.canvas_w
    }
}

/// One request's staged input and its output.
pub struct SlotData {
    /// Canvas-staged spatial planes, `C × canvas_h × canvas_w`.
    pub spatial: Vec<f32>,
    /// `F` globals.
    pub global: Vec<f32>,
    /// `canvas_h × canvas_w`, 1 on the request's own board.
    pub mask: Vec<f32>,
    /// The request's own board, tensor rows/cols (border included).
    pub h: usize,
    pub w: usize,
    /// Whether the caller wants the policy map back.
    pub want_policy: bool,
    /// Output: the value.
    pub value: f32,
    /// Output: the policy cropped to the request's own `A × h × w`, if asked for.
    pub policy: Vec<f32>,
    /// The board last staged into `spatial`/`mask`, so an equal one skips re-zeroing.
    staged: Option<(usize, usize)>,
    /// When the owner marked it ready, for the queue-time counter.
    ready_at: Option<Instant>,
}

impl SlotData {
    fn new() -> Self {
        SlotData {
            spatial: Vec::new(),
            global: Vec::new(),
            mask: Vec::new(),
            h: 0,
            w: 0,
            want_policy: false,
            value: 0.0,
            policy: Vec::new(),
            staged: None,
            ready_at: None,
        }
    }

    /// Stage one request onto the canvas: the sidecar's `np_s[i].fill(0); np_s[i, :, :h, :w] =
    /// spatial; np_m[i].fill(0); np_m[i, :, :h, :w] = 1`. Padding is only re-zeroed when the board
    /// differs from the one this slot held last (the cells it covers are overwritten anyway).
    fn stage(&mut self, geo: &Geometry, spatial: &[f32], global: &[f32], h: usize, w: usize) {
        let (ch, cw) = (geo.canvas_h, geo.canvas_w);
        if self.spatial.len() != geo.spatial_len() {
            self.spatial = vec![0.0; geo.spatial_len()];
            self.mask = vec![0.0; geo.plane()];
            self.staged = None;
        }
        if self.staged != Some((h, w)) {
            self.spatial.fill(0.0);
            self.mask.fill(0.0);
            for y in 0..h {
                self.mask[y * cw..y * cw + w].fill(1.0);
            }
            self.staged = Some((h, w));
        }
        for c in 0..geo.c {
            let src = &spatial[c * h * w..(c + 1) * h * w];
            let dst = &mut self.spatial[c * ch * cw..(c + 1) * ch * cw];
            for y in 0..h {
                dst[y * cw..y * cw + w].copy_from_slice(&src[y * w..(y + 1) * w]);
            }
        }
        self.global.clear();
        self.global.extend_from_slice(global);
        self.h = h;
        self.w = w;
    }
}

/// Stage one request onto fresh canvas buffers (the uncached form of the slot's staging), for
/// harnesses that build batches by hand: returns `(spatial C × ch × cw, mask ch × cw)`.
pub fn stage_canvas(geo: &Geometry, spatial: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut d = SlotData::new();
    d.stage(geo, spatial, &[], h, w);
    (d.spatial, d.mask)
}

/// Runs one batch of staged slots on the device. Implemented by the libtorch executor and by the
/// tests' CPU stand-in.
pub trait Executor: Send + Sync + 'static {
    /// Batcher lanes (threads); `run` is called with `lane < lanes()` from that lane's thread only.
    fn lanes(&self) -> usize;
    /// The largest batch `run` accepts.
    fn max_batch(&self) -> usize;
    /// The bucket a batch of `n` is padded to (its device-time estimate is kept per bucket).
    fn bucket_for(&self, n: usize) -> usize;
    /// Run `rows` as one batch on `lane`: read each row's staged input and write its `value` (and
    /// its cropped `policy` when `want_policy`, which is true if any row asked). Returns the device
    /// time of the batch.
    fn run(&self, lane: usize, rows: &mut [MutexGuard<'_, SlotData>], want_policy: bool) -> Result<Duration, String>;
}

struct Slot {
    data: Mutex<SlotData>,
    /// Set by the batcher when `data` holds the answer (or `error` is set).
    done: AtomicBool,
    error: Mutex<Option<String>>,
    owner: Mutex<Option<Thread>>,
}

struct State {
    ready: VecDeque<usize>,
    free: Vec<usize>,
    in_flight: usize,
    /// When the device is expected to go idle (meaningful while `in_flight > 0`).
    device_free_at: Instant,
    /// Per-bucket EMA of device time, by bucket size.
    eta: Vec<(usize, Duration)>,
    dead: Option<String>,
    shutdown: bool,
}

thread_local! {
    /// Plan 069 item 4b: the calling thread's request class ([`set_thread_class`]).
    static CLASS: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Plan 069 item 4b: count this thread's forwards as class `c` (1 = MC-label playouts; 0, the
/// default, = everything else). A hub worker tags its label-only streams, so one process running
/// generation and labelling side by side reports each one's positions (`label_samples=` in the
/// stats line). Threads a tagged thread spawns are untagged.
pub fn set_thread_class(c: u8) {
    CLASS.with(|k| k.set(c));
}

/// Counters for the stats line.
#[derive(Default)]
struct Counters {
    batches: AtomicU64,
    samples: AtomicU64,
    /// Requests made from threads tagged class 1 ([`set_thread_class`]).
    class1: AtomicU64,
    padded: AtomicU64,
    device_ns: AtomicU64,
    queue_ns: AtomicU64,
}

struct Shared {
    geo: Geometry,
    exec: Arc<dyn Executor>,
    slots: Vec<Slot>,
    state: Mutex<State>,
    /// Batchers wait here for ready slots (and for the device-idle moment).
    work: Condvar,
    /// Owners wait here for a free slot when every slot is taken.
    freed: Condvar,
    counters: Counters,
    hist: Mutex<Vec<u64>>,
    started: Instant,
    wait_timeout: Duration,
}

/// A snapshot of the queue's counters.
#[derive(Clone, Debug, Default)]
pub struct QueueStats {
    pub batches: u64,
    pub samples: u64,
    /// Rows computed for padding (bucket − n), summed.
    pub padded: u64,
    /// Device time summed over batches.
    pub device: Duration,
    /// Time from a request being ready to its batch launching, summed over requests.
    pub queue: Duration,
    /// Batches by size (index = n).
    pub hist: Vec<u64>,
    pub elapsed: Duration,
    /// Plan 069 item 4b: requests from class-1 (label) threads, a part of `samples`.
    pub class1: u64,
}

impl QueueStats {
    /// The sidecar's stats line, so logs read the same.
    pub fn line(&self) -> String {
        if self.batches == 0 {
            return "no batches yet".into();
        }
        let b = self.batches as f64;
        let s = self.samples.max(1) as f64;
        let top: Vec<String> = self
            .hist
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .take(8)
            .map(|(n, c)| format!("({n}, {c})"))
            .collect();
        format!(
            "batches={} samples={} mean_batch={:.2} pad={:.2} fwd {:.0}us per batch; {:.0}us GPU/sample queue={:.0}us/sample {:.0} samples/s hist=[{}]",
            self.batches,
            self.samples,
            self.samples as f64 / b,
            self.padded as f64 / s,
            self.device.as_secs_f64() * 1e6 / b,
            self.device.as_secs_f64() * 1e6 / s,
            self.queue.as_secs_f64() * 1e6 / s,
            self.samples as f64 / self.elapsed.as_secs_f64().max(1e-9),
            top.join(", ")
        ) + &if self.class1 > 0 {
            format!(" label_samples={}", self.class1)
        } else {
            String::new()
        }
    }
}

/// The queue: a slot ring and its batcher threads. Dropping it stops the batchers.
pub struct BatchQueue {
    shared: Arc<Shared>,
    batchers: Vec<JoinHandle<()>>,
}

impl BatchQueue {
    /// Start one batcher thread per executor lane over a ring of `slots` slots.
    pub fn new(exec: Arc<dyn Executor>, geo: Geometry, slots: usize) -> Self {
        Self::with_timeout(exec, geo, slots, WAIT_TIMEOUT)
    }

    /// [`new`](Self::new) with an owner wait timeout other than [`WAIT_TIMEOUT`].
    pub fn with_timeout(exec: Arc<dyn Executor>, geo: Geometry, slots: usize, wait_timeout: Duration) -> Self {
        let slots = slots.max(1);
        let shared = Arc::new(Shared {
            geo,
            exec: Arc::clone(&exec),
            slots: (0..slots)
                .map(|_| Slot {
                    data: Mutex::new(SlotData::new()),
                    done: AtomicBool::new(false),
                    error: Mutex::new(None),
                    owner: Mutex::new(None),
                })
                .collect(),
            state: Mutex::new(State {
                ready: VecDeque::new(),
                // Popped from the back: the slot freed last is reused first, so a thread tends to
                // get back a slot that already has its board staged.
                free: (0..slots).rev().collect(),
                in_flight: 0,
                device_free_at: Instant::now(),
                eta: Vec::new(),
                dead: None,
                shutdown: false,
            }),
            work: Condvar::new(),
            freed: Condvar::new(),
            counters: Counters::default(),
            hist: Mutex::new(vec![0; exec.max_batch() + 1]),
            started: Instant::now(),
            wait_timeout,
        });
        let batchers = (0..exec.lanes())
            .map(|lane| {
                let sh = Arc::clone(&shared);
                thread::Builder::new()
                    .name(format!("nn-batcher-{lane}"))
                    .spawn(move || batcher(&sh, lane))
                    .expect("spawn batcher thread")
            })
            .collect();
        BatchQueue { shared, batchers }
    }

    pub fn geometry(&self) -> Geometry {
        self.shared.geo
    }

    /// `Some(message)` once a batch has failed; every forward fails from then on.
    pub fn dead(&self) -> Option<String> {
        self.shared.state.lock().unwrap().dead.clone()
    }

    pub fn stats(&self) -> QueueStats {
        let c = &self.shared.counters;
        QueueStats {
            batches: c.batches.load(Ordering::Relaxed),
            samples: c.samples.load(Ordering::Relaxed),
            padded: c.padded.load(Ordering::Relaxed),
            device: Duration::from_nanos(c.device_ns.load(Ordering::Relaxed)),
            queue: Duration::from_nanos(c.queue_ns.load(Ordering::Relaxed)),
            hist: self.shared.hist.lock().unwrap().clone(),
            elapsed: self.shared.started.elapsed(),
            class1: c.class1.load(Ordering::Relaxed),
        }
    }

    /// One blocking forward: `(policy cropped to h × w if asked, else empty, value)`.
    ///
    /// `spatial` is the request's own `C × h × w` planes; the board must fit the canvas.
    pub fn forward(
        &self,
        spatial: &[f32],
        global: &[f32],
        h: usize,
        w: usize,
        want_policy: bool,
    ) -> Result<(Vec<f32>, f32), String> {
        let sh = &*self.shared;
        let geo = sh.geo;
        if h > geo.canvas_h || w > geo.canvas_w {
            return Err(format!(
                "board {h}x{w} does not fit the {}x{} canvas",
                geo.canvas_h, geo.canvas_w
            ));
        }
        if spatial.len() != geo.c * h * w || global.len() != geo.f {
            return Err(format!(
                "input is {} + {} floats, expected {} + {} for {h}x{w}",
                spatial.len(),
                global.len(),
                geo.c * h * w,
                geo.f
            ));
        }
        if CLASS.with(|k| k.get()) == 1 {
            sh.counters.class1.fetch_add(1, Ordering::Relaxed);
        }
        // Claim a slot.
        let idx = {
            let mut st = sh.state.lock().unwrap();
            loop {
                if let Some(e) = &st.dead {
                    return Err(e.clone());
                }
                if let Some(i) = st.free.pop() {
                    break i;
                }
                st = sh.freed.wait(st).unwrap();
            }
        };
        let slot = &sh.slots[idx];
        {
            let mut d = slot.data.lock().unwrap();
            d.stage(&geo, spatial, global, h, w);
            d.want_policy = want_policy;
            d.ready_at = Some(Instant::now());
        }
        slot.done.store(false, Ordering::Relaxed);
        *slot.error.lock().unwrap() = None;
        *slot.owner.lock().unwrap() = Some(thread::current());
        let t_ready = Instant::now();
        {
            let mut st = sh.state.lock().unwrap();
            if let Some(e) = st.dead.clone() {
                st.free.push(idx);
                return Err(e);
            }
            st.ready.push_back(idx);
        }
        sh.work.notify_one();
        // Wait: spin briefly, then park until the batcher unparks us.
        let spin_until = t_ready + SPIN;
        while !slot.done.load(Ordering::Acquire) && Instant::now() < spin_until {
            std::hint::spin_loop();
        }
        let deadline = t_ready + sh.wait_timeout;
        while !slot.done.load(Ordering::Acquire) {
            let now = Instant::now();
            if now >= deadline {
                // The slot stays with the batcher (it may still write to it); mark the queue dead
                // so nobody else waits on a wedged device.
                let msg = format!("no answer from the device in {:?}", sh.wait_timeout);
                sh.state.lock().unwrap().dead.get_or_insert(msg.clone());
                // This slot is never freed: wake every owner waiting for one, so it sees `dead`.
                sh.freed.notify_all();
                return Err(msg);
            }
            thread::park_timeout(deadline - now);
        }
        let out = match slot.error.lock().unwrap().take() {
            Some(e) => Err(e),
            None => {
                let mut d = slot.data.lock().unwrap();
                let policy = if want_policy { std::mem::take(&mut d.policy) } else { Vec::new() };
                Ok((policy, d.value))
            }
        };
        sh.state.lock().unwrap().free.push(idx);
        sh.freed.notify_one();
        out
    }
}

impl Drop for BatchQueue {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.work.notify_all();
        for b in self.batchers.drain(..) {
            let _ = b.join();
        }
    }
}

fn eta_of(st: &State, bucket: usize) -> Duration {
    st.eta
        .iter()
        .find(|(b, _)| *b == bucket)
        .map_or(ETA_SEED, |(_, d)| *d)
}

fn batcher(sh: &Shared, lane: usize) {
    let max = sh.exec.max_batch();
    loop {
        // Wait for work, and with a batch in flight hold it until the device is nearly idle.
        let (batch, t_launch) = {
            let mut st = sh.state.lock().unwrap();
            loop {
                if st.shutdown {
                    return;
                }
                if !st.ready.is_empty() {
                    if st.in_flight == 0 || st.ready.len() >= max {
                        break;
                    }
                    let launch_at = st.device_free_at.checked_sub(JIT).unwrap_or(st.device_free_at);
                    let now = Instant::now();
                    if now >= launch_at {
                        break;
                    }
                    st = sh.work.wait_timeout(st, launch_at - now).unwrap().0;
                } else {
                    st = sh.work.wait(st).unwrap();
                }
            }
            let n = st.ready.len().min(max);
            let batch: Vec<usize> = st.ready.drain(..n).collect();
            let now = Instant::now();
            let bucket = sh.exec.bucket_for(n);
            let start = if st.in_flight > 0 && st.device_free_at > now {
                st.device_free_at
            } else {
                now
            };
            st.device_free_at = start + eta_of(&st, bucket);
            st.in_flight += 1;
            // Another batcher may now be able to hold (or take) the rest.
            if !st.ready.is_empty() {
                sh.work.notify_one();
            }
            (batch, now)
        };
        let n = batch.len();
        let bucket = sh.exec.bucket_for(n);
        let mut guards: Vec<MutexGuard<'_, SlotData>> = batch.iter().map(|&i| sh.slots[i].data.lock().unwrap()).collect();
        let want_policy = guards.iter().any(|g| g.want_policy);
        let queue_ns: u64 = guards
            .iter()
            .filter_map(|g| g.ready_at)
            .map(|t| t_launch.saturating_duration_since(t).as_nanos() as u64)
            .sum();
        let dead = sh.state.lock().unwrap().dead.clone();
        let result = match dead {
            Some(e) => Err(e),
            None => sh.exec.run(lane, &mut guards, want_policy),
        };
        drop(guards);
        {
            let mut st = sh.state.lock().unwrap();
            st.in_flight -= 1;
            match &result {
                Ok(device) => {
                    // The sidecar's `eta_s += 0.1 * (fwd - eta_s)`, per bucket.
                    let old = eta_of(&st, bucket);
                    let new = old.mul_f64(0.9) + device.mul_f64(0.1);
                    match st.eta.iter_mut().find(|(b, _)| *b == bucket) {
                        Some(e) => e.1 = new,
                        None => st.eta.push((bucket, new)),
                    }
                }
                Err(e) => {
                    st.dead.get_or_insert_with(|| e.clone());
                }
            }
        }
        sh.work.notify_all();
        if result.is_err() {
            // Owners waiting for a free slot must see `dead` too, not wait for one to come back.
            sh.freed.notify_all();
        }
        if let Ok(device) = &result {
            let c = &sh.counters;
            c.batches.fetch_add(1, Ordering::Relaxed);
            c.samples.fetch_add(n as u64, Ordering::Relaxed);
            c.padded.fetch_add(bucket.saturating_sub(n) as u64, Ordering::Relaxed);
            c.device_ns.fetch_add(device.as_nanos() as u64, Ordering::Relaxed);
            c.queue_ns.fetch_add(queue_ns, Ordering::Relaxed);
            if let Some(h) = sh.hist.lock().unwrap().get_mut(n) {
                *h += 1;
            }
        }
        for &i in &batch {
            let slot = &sh.slots[i];
            if let Err(e) = &result {
                *slot.error.lock().unwrap() = Some(e.clone());
            }
            slot.done.store(true, Ordering::Release);
            if let Some(t) = slot.owner.lock().unwrap().as_ref() {
                t.unpark();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    const GEO: Geometry = Geometry {
        c: 3,
        f: 2,
        a: 2,
        canvas_h: 5,
        canvas_w: 6,
    };

    /// A CPU stand-in: value = masked sum of the spatial planes + globals; policy = the staged
    /// spatial plane `k mod C` times `k`, cropped. Every row depends only on itself, like the net.
    struct Fake {
        lanes: usize,
        max: usize,
        delay: Duration,
        fail_after: Option<usize>,
        hang: bool,
        runs: AtomicUsize,
        max_seen: AtomicUsize,
    }

    impl Fake {
        fn new(lanes: usize, max: usize) -> Self {
            Fake {
                lanes,
                max,
                delay: Duration::from_millis(1),
                fail_after: None,
                hang: false,
                runs: AtomicUsize::new(0),
                max_seen: AtomicUsize::new(0),
            }
        }
    }

    fn reference(spatial: &[f32], global: &[f32], h: usize, w: usize) -> (Vec<f32>, f32) {
        let v = spatial.iter().sum::<f32>() + global.iter().sum::<f32>();
        let mut p = Vec::new();
        for k in 0..GEO.a {
            let c = k % GEO.c;
            p.extend(spatial[c * h * w..(c + 1) * h * w].iter().map(|x| x * (k as f32 + 1.0)));
        }
        (p, v)
    }

    impl Executor for Fake {
        fn lanes(&self) -> usize {
            self.lanes
        }
        fn max_batch(&self) -> usize {
            self.max
        }
        fn bucket_for(&self, n: usize) -> usize {
            n.next_power_of_two().min(self.max)
        }
        fn run(&self, lane: usize, rows: &mut [MutexGuard<'_, SlotData>], want_policy: bool) -> Result<Duration, String> {
            assert!(lane < self.lanes);
            assert!(!rows.is_empty() && rows.len() <= self.max);
            self.max_seen.fetch_max(rows.len(), Ordering::Relaxed);
            let k = self.runs.fetch_add(1, Ordering::Relaxed);
            if self.hang {
                thread::sleep(Duration::from_millis(1500));
            }
            if self.fail_after.is_some_and(|f| k >= f) {
                return Err("device lost (test)".into());
            }
            thread::sleep(self.delay);
            for r in rows.iter_mut() {
                let (h, w) = (r.h, r.w);
                // Read only through the staged canvas + mask, as the device does.
                let mut own = vec![0.0; GEO.c * h * w];
                for c in 0..GEO.c {
                    for y in 0..h {
                        for x in 0..w {
                            let m = r.mask[y * GEO.canvas_w + x];
                            assert_eq!(m, 1.0);
                            own[c * h * w + y * w + x] = r.spatial[c * GEO.plane() + y * GEO.canvas_w + x];
                        }
                    }
                }
                // Padding must be zero in both the planes and the mask.
                for y in 0..GEO.canvas_h {
                    for x in 0..GEO.canvas_w {
                        if y >= h || x >= w {
                            assert_eq!(r.mask[y * GEO.canvas_w + x], 0.0);
                            for c in 0..GEO.c {
                                assert_eq!(r.spatial[c * GEO.plane() + y * GEO.canvas_w + x], 0.0);
                            }
                        }
                    }
                }
                let (p, v) = reference(&own, &r.global, h, w);
                r.value = v;
                if want_policy {
                    r.policy = p;
                }
            }
            Ok(self.delay)
        }
    }

    fn input(seed: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
        let s = (0..GEO.c * h * w).map(|i| ((i * 7 + seed * 13) % 17) as f32 - 8.0).collect();
        let g = (0..GEO.f).map(|i| (seed + i) as f32).collect();
        (s, g)
    }

    #[test]
    fn single_forward_matches_reference() {
        let q = BatchQueue::new(Arc::new(Fake::new(2, 8)), GEO, 4);
        let (s, g) = input(1, 3, 4);
        let (p, v) = q.forward(&s, &g, 3, 4, true).unwrap();
        let (rp, rv) = reference(&s, &g, 3, 4);
        assert_eq!(v, rv);
        assert_eq!(p, rp);
        let (p, v) = q.forward(&s, &g, 3, 4, false).unwrap();
        assert!(p.is_empty());
        assert_eq!(v, rv);
    }

    /// More concurrent callers than slots, mixed board sizes in one slot over time (so a slot is
    /// re-staged smaller after larger), every answer equal to its own reference: slot reuse,
    /// re-zeroing and wake-ups all hold, and no caller is left parked.
    #[test]
    fn slots_are_reused_and_every_owner_is_woken() {
        let fake = Arc::new(Fake::new(2, 4));
        let q = Arc::new(BatchQueue::new(fake.clone(), GEO, 3));
        let boards = [(5, 6), (2, 3), (4, 6), (3, 2), (5, 5)];
        thread::scope(|s| {
            for t in 0..12 {
                let q = &q;
                s.spawn(move || {
                    for k in 0..60 {
                        let (h, w) = boards[(t + k) % boards.len()];
                        let (sp, g) = input(t * 1000 + k, h, w);
                        let want = k % 3 == 0;
                        let (p, v) = q.forward(&sp, &g, h, w, want).unwrap();
                        let (rp, rv) = reference(&sp, &g, h, w);
                        assert_eq!(v, rv);
                        if want {
                            assert_eq!(p, rp);
                        }
                    }
                });
            }
        });
        let st = q.stats();
        assert_eq!(st.samples, 12 * 60);
        assert!(fake.max_seen.load(Ordering::Relaxed) > 1, "requests were batched");
        assert!(fake.max_seen.load(Ordering::Relaxed) <= 4);
    }

    /// Batch invariance at the queue level: a row's answer does not depend on what it was batched
    /// with (the executor is row-independent; the queue must not mix rows up).
    #[test]
    fn answers_do_not_depend_on_batch_composition() {
        let q = Arc::new(BatchQueue::new(Arc::new(Fake::new(2, 8)), GEO, 16));
        let (s0, g0) = input(42, 4, 5);
        let alone = q.forward(&s0, &g0, 4, 5, true).unwrap();
        thread::scope(|s| {
            for t in 0..8 {
                let q = &q;
                let (s0, g0, alone) = (&s0, &g0, &alone);
                s.spawn(move || {
                    for k in 0..40 {
                        if (t + k) % 2 == 0 {
                            assert_eq!(&q.forward(s0, g0, 4, 5, true).unwrap(), alone);
                        } else {
                            let (sp, g) = input(t * 100 + k, 5, 6);
                            q.forward(&sp, &g, 5, 6, false).unwrap();
                        }
                    }
                });
            }
        });
    }

    /// A failed batch fails its requests with the device's message and kills the queue: later
    /// forwards fail at once instead of waiting.
    #[test]
    fn device_failure_is_an_error_not_a_hang() {
        let mut fake = Fake::new(1, 4);
        fake.fail_after = Some(2);
        let q = Arc::new(BatchQueue::new(Arc::new(fake), GEO, 8));
        let (s, g) = input(1, 3, 3);
        q.forward(&s, &g, 3, 3, false).unwrap();
        q.forward(&s, &g, 3, 3, false).unwrap();
        let e = q.forward(&s, &g, 3, 3, false).unwrap_err();
        assert!(e.contains("device lost"), "{e}");
        let t0 = Instant::now();
        let e = q.forward(&s, &g, 3, 3, false).unwrap_err();
        assert!(e.contains("device lost"), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(1));
        assert!(q.dead().is_some());
        // Many threads at once, all get an error.
        thread::scope(|sc| {
            for _ in 0..6 {
                let q = &q;
                let (s, g) = (&s, &g);
                sc.spawn(move || assert!(q.forward(s, g, 3, 3, true).is_err()));
            }
        });
    }

    /// A device that never answers: the owner gives up after the timeout, and the queue is dead.
    #[test]
    fn wedged_device_times_out() {
        let mut fake = Fake::new(1, 2);
        fake.hang = true;
        let q = BatchQueue::with_timeout(Arc::new(fake), GEO, 2, Duration::from_millis(100));
        let (s, g) = input(0, 2, 2);
        let t0 = Instant::now();
        let e = q.forward(&s, &g, 2, 2, false).unwrap_err();
        assert!(e.contains("no answer"), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(1));
        assert!(q.forward(&s, &g, 2, 2, false).is_err());
    }

    /// More callers than slots on a wedged device: the slot owner times out and never frees its
    /// slot, so a caller waiting for one must be woken by the queue dying, not wait forever.
    #[test]
    fn callers_waiting_for_a_slot_see_a_dead_queue() {
        let mut fake = Fake::new(1, 2);
        fake.hang = true;
        let q = BatchQueue::with_timeout(Arc::new(fake), GEO, 1, Duration::from_millis(100));
        let (s, g) = input(0, 2, 2);
        let t0 = Instant::now();
        thread::scope(|sc| {
            for _ in 0..3 {
                let (q, s, g) = (&q, &s, &g);
                sc.spawn(move || assert!(q.forward(s, g, 2, 2, false).is_err()));
            }
        });
        assert!(t0.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn oversized_board_is_rejected() {
        let q = BatchQueue::new(Arc::new(Fake::new(1, 2)), GEO, 2);
        let (s, g) = input(0, 6, 6);
        assert!(q.forward(&s, &g, 6, 6, false).unwrap_err().contains("canvas"));
    }

    #[test]
    fn queue_drops_cleanly() {
        let q = BatchQueue::new(Arc::new(Fake::new(2, 2)), GEO, 2);
        let (s, g) = input(0, 2, 2);
        q.forward(&s, &g, 2, 2, false).unwrap();
        drop(q);
    }
}
