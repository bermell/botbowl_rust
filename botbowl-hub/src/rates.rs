//! Throughput: games and decisions per minute, per worker and in total.
//!
//! A generation's speed once jumped and dropped and we misattributed it (plan 058: the laptop
//! joining and leaving, not a shared memo), because nothing showed what each worker produced.
//! The [`Ledger`] keeps every result the hub accepts — when, from which worker *name*, for which
//! job, and its [`Counts`] — for [`KEEP`], and answers windowed rates and a wall-clock-aligned
//! interval history from it. Everything comes from frames workers already send (a
//! `TrajectoryDone` carries its sample count, an `EvalGameLine` its search telemetry), so the
//! worker protocol is untouched.
//!
//! Workers are keyed by name, not connection id: a reconnect is a new id under the same name, and
//! "local2 slowed down" is the statement we want to be able to make across it.
//!
//! Time is passed in (`now: Instant`) so the arithmetic is testable without sleeping.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::api::{Counts, JobId, Rate, Throughput, WorkerRates};

/// How long results are kept for the windows and the history.
pub const KEEP: Duration = Duration::from_secs(6 * 3600);
/// The windowed rates: the last 5 and 30 minutes.
pub const WINDOWS: [Duration; 2] = [Duration::from_secs(5 * 60), Duration::from_secs(30 * 60)];
/// Complete intervals kept in [`WorkerRates::history`]: three hours of 5-minute intervals.
pub const HISTORY: usize = 36;
/// The default interval, and the period of the `[hub] rate ...` line in hub.log.
pub const DEFAULT_BUCKET: Duration = Duration::from_secs(5 * 60);

struct Event {
    at: Instant,
    worker: String,
    job: JobId,
    c: Counts,
}

/// A connected worker as the page should list it.
pub struct Live<'a> {
    pub name: &'a str,
    pub streams: u32,
    pub busy: u32,
}

pub struct Ledger {
    start: Instant,
    start_wall: SystemTime,
    /// The history's interval (and the hub.log rate line's period).
    pub bucket: Duration,
    /// In arrival order, which is time order: `record` is called with the time it is called.
    events: VecDeque<Event>,
    /// Everything ever recorded per worker name, beyond [`KEEP`].
    totals: BTreeMap<String, Counts>,
    first_seen: HashMap<String, Instant>,
}

impl Default for Ledger {
    fn default() -> Self {
        Ledger::new(Instant::now(), SystemTime::now())
    }
}

impl Ledger {
    /// A ledger whose clock starts at `start`, which is `start_wall` on the wall clock.
    pub fn new(start: Instant, start_wall: SystemTime) -> Self {
        Ledger {
            start,
            start_wall,
            bucket: DEFAULT_BUCKET,
            events: VecDeque::new(),
            totals: BTreeMap::new(),
            first_seen: HashMap::new(),
        }
    }

    /// A worker connected. Its rates are over the time since its first connection, so a worker
    /// that joined two minutes ago is not reported at 2/5 of its speed.
    pub fn joined(&mut self, worker: &str, at: Instant) {
        self.first_seen.entry(worker.to_string()).or_insert(at);
    }

    /// One accepted result.
    pub fn record(&mut self, at: Instant, worker: &str, job: JobId, c: Counts) {
        self.first_seen.entry(worker.to_string()).or_insert(at);
        self.totals.entry(worker.to_string()).or_default().add(&c);
        self.events.push_back(Event {
            at,
            worker: worker.to_string(),
            job,
            c,
        });
        while self
            .events
            .front()
            .is_some_and(|e| at.saturating_duration_since(e.at) > KEEP)
        {
            self.events.pop_front();
        }
    }

    /// The results in `(from, now]` that `keep(worker, job)` accepts, where `from` is the later of
    /// `now - window` and `since`; the rate is over that span.
    pub fn window(&self, now: Instant, window: Duration, since: Instant, keep: impl Fn(&str, JobId) -> bool) -> Rate {
        let from = now.checked_sub(window).map_or(since, |t| t.max(since));
        let mut counts = Counts::default();
        for e in self.events.iter().rev() {
            if e.at <= from {
                break;
            }
            if e.at <= now && keep(&e.worker, e.job) {
                counts.add(&e.c);
            }
        }
        Rate {
            counts,
            secs: now.saturating_duration_since(from).as_secs_f64(),
        }
    }

    /// When `worker` first joined (or produced), never before the ledger's start.
    fn since(&self, worker: &str) -> Instant {
        self.first_seen.get(worker).map_or(self.start, |t| (*t).max(self.start))
    }

    /// Wall-clock seconds at instant `t`.
    fn wall(&self, t: Instant) -> f64 {
        let base = self
            .start_wall
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        base + t.saturating_duration_since(self.start).as_secs_f64()
    }

    fn bucket_secs(&self) -> u64 {
        self.bucket.as_secs().max(1)
    }

    /// The wall-clock interval `now` falls in. It changes when an interval completes, which is
    /// when the hub.log rate line is written.
    pub fn interval(&self, now: Instant) -> u64 {
        (self.wall(now) / self.bucket_secs() as f64).floor() as u64
    }

    /// Rates for every worker name that is connected or produced something in the history's
    /// span, and for the fleet.
    pub fn throughput(&self, now: Instant, live: &[Live]) -> Throughput {
        let b = self.bucket_secs();
        let cur = self.interval(now);
        let first = self.interval(self.start).max(cur.saturating_sub(HISTORY as u64));
        let n = (cur - first) as usize;
        let start_wall = self.wall(self.start);
        let now_wall = self.wall(now);
        let history_secs: Vec<u64> = (first..cur)
            .map(|k| {
                let (lo, hi) = ((k * b) as f64, ((k + 1) * b) as f64);
                (hi.min(now_wall) - lo.max(start_wall)).max(0.0).round() as u64
            })
            .collect();

        // One pass over the history's span: per-name and total interval counts.
        let mut hist: BTreeMap<&str, Vec<Counts>> = BTreeMap::new();
        let mut total_hist = vec![Counts::default(); n];
        for e in self.events.iter().rev() {
            let k = (self.wall(e.at) / b as f64).floor() as u64;
            if k < first {
                break;
            }
            if k >= cur {
                continue;
            }
            let i = (k - first) as usize;
            hist.entry(&e.worker).or_insert_with(|| vec![Counts::default(); n])[i].add(&e.c);
            total_hist[i].add(&e.c);
        }

        let mut names: BTreeSet<&str> = live.iter().map(|l| l.name).collect();
        // A worker that left is still listed while its results are recent enough to matter.
        let recent = now.checked_sub(WINDOWS[WINDOWS.len() - 1]).unwrap_or(self.start);
        names.extend(hist.keys().copied());
        names.extend(
            self.events
                .iter()
                .rev()
                .take_while(|e| e.at > recent)
                .map(|e| e.worker.as_str()),
        );

        let rates = |name: Option<&str>| -> WorkerRates {
            // The fleet's rates run from the first worker's arrival, so the total of workers that
            // joined together is the sum of their rates, not diluted by the hub's idle start.
            let first_worker = self
                .first_seen
                .values()
                .min()
                .map_or(self.start, |t| (*t).max(self.start));
            let since = name.map_or(first_worker, |w| self.since(w));
            let keep = |w: &str, _: JobId| name.is_none_or(|n| n == w);
            let total = match name {
                Some(w) => self.totals.get(w).copied().unwrap_or_default(),
                None => self.totals.values().fold(Counts::default(), |mut a, c| {
                    a.add(c);
                    a
                }),
            };
            let conns: Vec<&Live> = live.iter().filter(|l| name.is_none_or(|n| n == l.name)).collect();
            WorkerRates {
                name: name.unwrap_or("total").to_string(),
                connected: conns.len() as u32,
                streams: conns.iter().map(|l| l.streams).sum(),
                busy: conns.iter().map(|l| l.busy).sum(),
                windows: WINDOWS.iter().map(|w| self.window(now, *w, since, keep)).collect(),
                since_start: Rate {
                    counts: total,
                    secs: now.saturating_duration_since(since).as_secs_f64(),
                },
                history: match name {
                    Some(w) => hist.get(w).cloned().unwrap_or_else(|| vec![Counts::default(); n]),
                    None => total_hist.clone(),
                },
            }
        };

        Throughput {
            uptime_secs: now.saturating_duration_since(self.start).as_secs(),
            windows: WINDOWS.iter().map(|w| w.as_secs()).collect(),
            bucket_secs: b,
            history_end: cur * b,
            history_secs,
            workers: names.iter().map(|w| rates(Some(w))).collect(),
            total: rates(None),
        }
    }
}

/// `14:05`, local time.
pub fn clock(unix: u64) -> String {
    use chrono::TimeZone;
    match chrono::Local.timestamp_opt(unix as i64, 0) {
        chrono::LocalResult::Single(t) | chrono::LocalResult::Ambiguous(t, _) => t.format("%H:%M").to_string(),
        chrono::LocalResult::None => format!("@{unix}"),
    }
}

/// `6.2`, `14.0`, `210`: a rate at the precision worth reading.
pub fn num(x: f64) -> String {
    if x >= 100.0 {
        format!("{x:.0}")
    } else {
        format!("{x:.1}")
    }
}

/// The hub.log line for the newest complete interval, or `None` when nothing finished in it:
/// `rate 14:05-14:10: local 6.2 games/min 210 decisions/min (24 streams) · ... · total ...`.
pub fn log_line(t: &Throughput) -> Option<String> {
    let secs = *t.history_secs.last()?;
    let last = |w: &WorkerRates| w.history.last().copied().unwrap_or_default();
    if secs == 0 || last(&t.total).is_empty() {
        return None;
    }
    let r = |c: Counts| Rate {
        counts: c,
        secs: secs as f64,
    };
    let part = |w: &WorkerRates| {
        let c = last(w);
        let mut s = format!(
            "{} {} games/min {} decisions/min",
            w.name,
            num(r(c).per_min(c.games)),
            num(r(c).per_min(c.samples))
        );
        if c.eval_games > 0 {
            s.push_str(&format!(" + eval {} games/min", num(r(c).per_min(c.eval_games))));
        }
        if c.label_items > 0 {
            s.push_str(&format!(" + label {} samples/min", num(r(c).per_min(c.label_samples))));
        }
        s
    };
    let mut parts: Vec<String> = t
        .workers
        .iter()
        .filter(|w| !last(w).is_empty() || w.connected > 0)
        .map(|w| {
            let mut s = part(w);
            if w.connected > 0 {
                s.push_str(&format!(" ({} streams)", w.streams));
            } else {
                s.push_str(" (gone)");
            }
            s
        })
        .collect();
    parts.push(part(&t.total));
    Some(format!(
        "rate {}-{}: {}",
        clock(t.history_end.saturating_sub(t.bucket_secs)),
        clock(t.history_end),
        parts.join(" · ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen(games: u64, samples: u64) -> Counts {
        Counts {
            games,
            records: games,
            samples,
            ..Default::default()
        }
    }

    /// A ledger started at a wall-clock instant on an interval boundary (12:00:00 UTC).
    fn ledger() -> (Ledger, Instant) {
        let t0 = Instant::now();
        // 1_800_000_000 is a multiple of 300.
        let wall = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
        (Ledger::new(t0, wall), t0)
    }

    fn at(t0: Instant, secs: u64) -> Instant {
        t0 + Duration::from_secs(secs)
    }

    #[test]
    fn a_window_counts_only_its_own_results_over_its_own_span() {
        let (mut l, t0) = ledger();
        // 10 minutes of local at one game a minute, 30 samples each.
        for m in 1..=10 {
            l.record(at(t0, m * 60), "local", 0, gen(1, 30));
        }
        let now = at(t0, 600);
        let r = l.window(now, Duration::from_secs(300), t0, |_, _| true);
        assert_eq!(r.secs, 300.0);
        // (300, 600]: the games at 6..=10 minutes.
        assert_eq!(r.counts.games, 5);
        assert!((r.per_min(r.counts.games) - 1.0).abs() < 1e-9);
        assert!((r.per_min(r.counts.samples) - 30.0).abs() < 1e-9);
        // A window longer than the hub has been up is over the uptime, not the window.
        let r = l.window(now, Duration::from_secs(1800), t0, |_, _| true);
        assert_eq!(r.secs, 600.0);
        assert!((r.per_min(r.counts.games) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_late_joiner_is_rated_over_the_time_since_it_joined() {
        let (mut l, t0) = ledger();
        l.joined("local", t0);
        for m in 1..=10 {
            l.record(at(t0, m * 60), "local", 0, gen(2, 60));
        }
        // The laptop joins at minute 8 and finishes one game a minute from minute 9.
        l.joined("laptop", at(t0, 480));
        l.record(at(t0, 540), "laptop", 0, gen(1, 20));
        l.record(at(t0, 600), "laptop", 0, gen(1, 20));
        let now = at(t0, 600);
        let live = [
            Live {
                name: "local",
                streams: 24,
                busy: 24,
            },
            Live {
                name: "laptop",
                streams: 5,
                busy: 3,
            },
        ];
        let t = l.throughput(now, &live);
        let by = |n: &str| t.workers.iter().find(|w| w.name == n).unwrap().clone();
        let local = by("local");
        let laptop = by("laptop");
        assert!((local.windows[0].per_min(local.windows[0].counts.games) - 2.0).abs() < 1e-9);
        // Two games in the two minutes since it joined: 1/min, not 2 per 5 min.
        assert_eq!(laptop.windows[0].secs, 120.0);
        assert!((laptop.windows[0].per_min(laptop.windows[0].counts.games) - 1.0).abs() < 1e-9);
        assert_eq!((laptop.streams, laptop.busy, laptop.connected), (5, 3, 1));
        // The total over the last 5 minutes: local's 10 games plus the laptop's 2.
        let w = &t.total.windows[0];
        assert_eq!(w.counts.games, 12);
        assert_eq!(t.total.streams, 29);
        assert_eq!(t.total.since_start.counts.games, 22);
        assert_eq!(t.total.since_start.counts.samples, 640);
    }

    #[test]
    fn the_history_is_whole_wall_clock_intervals_and_shows_a_drop() {
        let (mut l, t0) = ledger();
        // 30 minutes: local at 4 games/min for 20 minutes, then nothing (it left).
        for s in (15..1200).step_by(15) {
            l.record(at(t0, s), "local", 0, gen(1, 30));
        }
        // local2 steady at 2 games/min the whole time.
        for s in (30..=1800).step_by(30) {
            l.record(at(t0, s), "local2", 0, gen(1, 30));
        }
        let now = at(t0, 1810);
        let live = [Live {
            name: "local2",
            streams: 24,
            busy: 24,
        }];
        let t = l.throughput(now, &live);
        assert_eq!(t.bucket_secs, 300);
        // Six complete intervals since the start; the seventh (30:00-35:00) is still running.
        assert_eq!(t.history_secs, vec![300; 6]);
        assert_eq!(t.history_end % 300, 0);
        let local = t.workers.iter().find(|w| w.name == "local").unwrap();
        // It left, but its results are recent, so it is listed, as gone.
        assert_eq!(local.connected, 0);
        let games: Vec<u64> = local.history.iter().map(|c| c.games).collect();
        // 4/min = 20 per interval for four intervals (the first misses the game at 0 s), then 0.
        assert_eq!(games, vec![19, 20, 20, 20, 0, 0]);
        let total: Vec<u64> = t.total.history.iter().map(|c| c.games).collect();
        assert_eq!(total, vec![28, 30, 30, 30, 10, 10]);
        // The 5-minute window sees the drop; the 30-minute one averages it away.
        assert!((t.total.windows[0].per_min(t.total.windows[0].counts.games) - 2.0).abs() < 0.1);
        assert!(t.total.windows[1].per_min(t.total.windows[1].counts.games) > 4.0);
    }

    #[test]
    fn a_hub_that_starts_mid_interval_has_a_partial_first_interval() {
        let t0 = Instant::now();
        let wall = UNIX_EPOCH + Duration::from_secs(1_800_000_000 + 120);
        let mut l = Ledger::new(t0, wall);
        l.record(at(t0, 60), "local", 0, gen(3, 90));
        let t = l.throughput(at(t0, 400), &[]);
        // Started at :02, so the first interval (:00-:05) covers 180 s of it.
        assert_eq!(t.history_secs, vec![180]);
        assert_eq!(t.total.history[0].games, 3);
    }

    #[test]
    fn old_results_are_dropped_but_totals_are_kept() {
        let (mut l, t0) = ledger();
        l.record(at(t0, 10), "local", 0, gen(1, 10));
        l.record(at(t0, 10 + KEEP.as_secs() + 60), "local", 0, gen(1, 10));
        assert_eq!(l.events.len(), 1);
        assert_eq!(l.totals["local"].games, 2);
    }

    #[test]
    fn a_job_filter_rates_one_job() {
        let (mut l, t0) = ledger();
        for m in 1..=5 {
            l.record(at(t0, m * 60), "local", 1, gen(2, 0));
            l.record(at(t0, m * 60), "local", 2, gen(1, 0));
        }
        let r = l.window(at(t0, 300), Duration::from_secs(300), t0, |_, j| j == 2);
        assert_eq!(r.counts.games, 5);
    }

    #[test]
    fn the_log_line_reports_the_last_interval_per_worker() {
        let (mut l, t0) = ledger();
        for s in (10..300).step_by(10) {
            l.record(at(t0, s), "local", 0, gen(1, 35));
        }
        l.record(at(t0, 100), "laptop", 0, gen(1, 30));
        let live = [Live {
            name: "local",
            streams: 24,
            busy: 20,
        }];
        let t = l.throughput(at(t0, 301), &live);
        let line = log_line(&t).unwrap();
        // 29 games in 5 minutes = 5.8/min, 29*35 = 1015 samples = 203/min.
        assert!(
            line.contains("local 5.8 games/min 203 decisions/min (24 streams)"),
            "{line}"
        );
        assert!(line.contains("laptop 0.2 games/min 6.0 decisions/min (gone)"), "{line}");
        assert!(line.contains("total 6.0 games/min 209 decisions/min"), "{line}");
        // An idle interval writes nothing.
        let t = l.throughput(at(t0, 601), &live);
        assert!(log_line(&t).is_none());
    }
}
