//! The `GET /status` page: what the box and its workers are doing, readable from a phone.
//!
//! It names things the way we talk about them — `gen03 drives vs gen21`, `vs gen13 @14x7/4` —
//! not by the paths and descriptors in rung names, which the report scripts still need verbatim
//! ([`short_label`] rewrites them for display only). With `serve --run-dir`, it also shows the
//! training loop's latest status lines and, while the box trains a net (work no worker can take
//! over), how far along that is, from the trainer's `--progress` file.

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::api::{GenStats, HubStatus, JobKind, JobState, JobStatus, Rate, Throughput, UnitStats, WorkerRates};

/// Everything the page shows, gathered first so [`render`] is a pure function.
pub struct PageInput {
    pub hub: HubStatus,
    pub port: u16,
    pub run: Option<RunInfo>,
    pub machine: Option<BoxInfo>,
}

pub struct RunInfo {
    pub name: String,
    /// The last lines of the loop's `status.md`, raw.
    pub recent: Vec<String>,
    pub status_age: Option<Duration>,
    pub training: Option<Training>,
}

pub struct Training {
    /// `gen03`.
    pub generation: String,
    pub progress: TrainProgress,
    /// Since the trainer last wrote its progress file.
    pub age: Duration,
}

/// The trainer's `--progress` JSON (`bbnn.train`).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct TrainProgress {
    pub started: f64,
    #[serde(default)]
    pub updated: f64,
    pub total_steps: Option<u64>,
    pub step: u64,
    pub train_value: Option<f64>,
    pub val_value: Option<f64>,
    pub best_step: Option<u64>,
    pub best_val_value: Option<f64>,
    pub baseline_val_value: Option<f64>,
    #[serde(default)]
    pub done: bool,
}

pub struct BoxInfo {
    pub mem_available_mb: u64,
    pub mem_total_mb: u64,
    pub swap_used_mb: u64,
    /// Utilisation %, memory used and total in MB.
    pub gpu: Option<(u32, u64, u64)>,
}

/// How many recent loop status lines and finished jobs the page keeps.
const RECENT_LINES: usize = 8;
const DONE_JOBS: usize = 6;
/// A trainer that has not written progress for this long is shown as possibly stopped.
const TRAIN_STALE: Duration = Duration::from_secs(15 * 60);

/// Gather the loop and machine sections. Blocking file reads and one `nvidia-smi` call.
pub fn gather(hub: HubStatus, port: u16, run_dir: Option<&Path>) -> PageInput {
    PageInput {
        hub,
        port,
        run: run_dir.and_then(read_run),
        machine: read_box(),
    }
}

fn read_run(dir: &Path) -> Option<RunInfo> {
    let status = dir.join("status.md");
    let text = std::fs::read_to_string(&status).ok()?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let recent = lines[lines.len().saturating_sub(RECENT_LINES)..]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let status_age = age_of(&status);
    // The newest generation with a progress file is the one training, or the one that last did.
    let mut gens: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("gen") && n[3..].chars().all(|c| c.is_ascii_digit()) && n.len() > 3)
        .collect();
    gens.sort_by_key(|n| n[3..].parse::<u32>().unwrap_or(0));
    let training = gens.iter().rev().find_map(|g| {
        let p = dir.join(g).join("train.progress.json");
        let progress: TrainProgress = serde_json::from_str(&std::fs::read_to_string(&p).ok()?).ok()?;
        Some(Training {
            generation: g.clone(),
            age: age_of(&p).unwrap_or_default(),
            progress,
        })
    });
    Some(RunInfo {
        name: dir
            .file_name()
            .map_or_else(|| dir.display().to_string(), |n| n.to_string_lossy().into_owned()),
        recent,
        status_age,
        training,
    })
}

fn age_of(p: &Path) -> Option<Duration> {
    SystemTime::now()
        .duration_since(std::fs::metadata(p).ok()?.modified().ok()?)
        .ok()
}

fn read_box() -> Option<BoxInfo> {
    let mem = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb = |key: &str| -> Option<u64> {
        mem.lines()
            .find(|l| l.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()
    };
    let gpu = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            let v: Vec<u64> = s
                .lines()
                .next()?
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            (v.len() == 3).then(|| (v[0] as u32, v[1], v[2]))
        });
    Some(BoxInfo {
        mem_available_mb: kb("MemAvailable:")? / 1024,
        mem_total_mb: kb("MemTotal:")? / 1024,
        swap_used_mb: kb("SwapTotal:")?.saturating_sub(kb("SwapFree:")?) / 1024,
        gpu,
    })
}

/// HTML-escape text.
pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The style every hub page shares: monospace, light and dark, readable at phone width.
pub const STYLE: &str = "body{margin:0;padding:12px;background:#fff;color:#111;\
     font:14px/1.5 ui-monospace,Menlo,Consolas,monospace}\
     pre{font:13px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;margin:0}\
     a{color:inherit}nav{margin:0 0 10px;opacity:.85}nav b{font-weight:600}\
     ul{padding-left:1.2em}code{background:rgba(128,128,128,.15);padding:0 .2em;border-radius:3px}\
     @media (prefers-color-scheme:dark){body{background:#111;color:#ddd}}";

/// The hub's pages, as a line of links; `current` is shown unlinked.
pub fn nav(current: &str, extra: &[(&str, &str)]) -> String {
    let mut links = vec![
        ("hub", "/"),
        ("status", "/status"),
        ("registry", "/registry/"),
        ("play", "/play/"),
    ];
    links.extend_from_slice(extra);
    let items: Vec<String> = links
        .iter()
        .map(|(name, href)| {
            if *name == current {
                format!("<b>{name}</b>")
            } else {
                format!("<a href=\"{href}\">{name}</a>")
            }
        })
        .collect();
    format!("<nav>{}</nav>", items.join(" · "))
}

/// The page, as HTML: one `<pre>` that refreshes itself every 30 s, section heads in bold.
pub fn render_html(input: &PageInput, now: SystemTime) -> String {
    let text: Vec<String> = render(input, now)
        .lines()
        .map(|l| {
            let l = esc(l);
            if !l.is_empty() && !l.starts_with(' ') {
                format!("<b>{l}</b>")
            } else {
                l
            }
        })
        .collect();
    let extra: &[(&str, &str)] = if input.run.is_some() {
        &[("status.md", "/run/status.md")]
    } else {
        &[]
    };
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <meta http-equiv=\"refresh\" content=\"30\"><title>botbowl hub</title>\
         <style>{STYLE}</style></head><body>{}<pre>{}</pre></body></html>",
        nav("status", extra),
        text.join("\n")
    )
}

/// `GET /`: what this port serves, in the status page's style.
pub fn render_index(has_play: bool) -> String {
    let play = if has_play {
        "<li><a href=\"/play/\">/play</a> — play the bots in a browser, or watch two of them; \
         ordinary games, random-start drives, and a team editor</li>"
    } else {
        "<li>/play — not served (<code>serve --no-play</code>, or no client build)</li>"
    };
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>botbowl hub</title><style>{STYLE}</style>\
         </head><body>{}<h3 style=\"margin:0 0 8px\">botbowl hub</h3><ul>\
         <li><a href=\"/status\">/status</a> — jobs, workers (games and decisions per minute, \
         per worker and in total) and the training loop</li>\
         <li><a href=\"/registry/\">/registry</a> — the project registry: corpora, nets, \
         experiments and matches</li>\
         {play}\
         <li>/ws — where workers connect (<code>botbowl-worker --hub ws://&lt;host&gt;/ws</code>)</li>\
         </ul></body></html>",
        nav("hub", &[])
    )
}

/// The page as plain text.
pub fn render(input: &PageInput, now: SystemTime) -> String {
    let s = &input.hub;
    let mut out = format!(
        "botbowl hub · commit {}{}",
        &s.commit[..s.commit.len().min(7)],
        if s.dirty { "-dirty" } else { "" }
    );
    if !s.throughput.windows.is_empty() {
        out.push_str(&format!(" · up {}", dur(Duration::from_secs(s.throughput.uptime_secs))));
    }
    if let Some(m) = &input.machine {
        out.push_str(&format!(
            "\nbox: {:.1} of {:.1} GB free, swap {:.1} GB used",
            m.mem_available_mb as f64 / 1024.0,
            m.mem_total_mb as f64 / 1024.0,
            m.swap_used_mb as f64 / 1024.0
        ));
        if let Some((util, used, total)) = m.gpu {
            out.push_str(&format!(
                " · GPU {util}% ({:.1} of {:.1} GB)",
                used as f64 / 1024.0,
                total as f64 / 1024.0
            ));
        }
    }
    out.push('\n');

    if let Some(run) = &input.run {
        out.push_str(&format!("\nloop {}", run.name));
        if let Some(a) = run.status_age {
            out.push_str(&format!(" · last status {} ago", dur(a)));
        }
        out.push('\n');
        if let Some(t) = &run.training {
            if let Some(line) = training_line(t, now) {
                out.push_str(&line);
            }
        }
        for l in &run.recent {
            out.push_str(&format!("  {}\n", status_line(l)));
        }
    }

    if s.throughput.windows.is_empty() {
        // A hub from before the rates: its connections, as they were listed then.
        out.push_str(&format!("\nworkers ({})\n", s.workers.len()));
        let name_w = s.workers.iter().map(|w| w.name.len()).max().unwrap_or(0);
        for w in &s.workers {
            out.push_str(&format!(
                "  {:name_w$}  {:>2} streams  {:>2} tasks  {:>6} games  {}  seen {}s ago\n",
                w.name, w.parallel_games, w.tasks_in_flight, w.games_done, w.triple, w.last_seen_secs
            ));
        }
    } else {
        throughput_lines(&mut out, s);
        interval_lines(&mut out, &s.throughput);
    }

    let running: Vec<&JobStatus> = s.jobs.iter().filter(|j| j.state == JobState::Running).collect();
    let mut finished: Vec<&JobStatus> = s.jobs.iter().filter(|j| j.state != JobState::Running).collect();
    finished.sort_by_key(|j| std::cmp::Reverse(j.id));
    out.push_str(&format!(
        "\njobs ({} running, {} finished)\n",
        running.len(),
        finished.len()
    ));
    for j in running.iter().copied().chain(finished.iter().take(DONE_JOBS).copied()) {
        job_lines(&mut out, j);
    }
    if finished.len() > DONE_JOBS {
        out.push_str(&format!("  … {} older finished jobs\n", finished.len() - DONE_JOBS));
    }

    if !s.dirty {
        out.push_str(&format!(
            "\njoin as a worker (this hub's token in ~/.config/botbowl/hub.token on that machine):\n\
             \x20 git fetch origin && git checkout {commit}\n\
             \x20 BOARD_SIZE_W={pw} BOARD_SIZE_H={ph} BOARD_PLAYERS={team} cargo build --release -p botbowl-worker\n\
             \x20 ./target/release/botbowl-worker --hub ws://<this-host-or-forwarded-address>:{port}/ws --name <yours>\n",
            commit = s.commit,
            pw = s.capacity.width.saturating_sub(2),
            ph = s.capacity.height.saturating_sub(2),
            team = s.capacity.team_size,
            port = input.port,
        ));
    } else {
        out.push_str(
            "\njoin as a worker: this hub runs a dirty tree, so no commit matches it exactly — start \
             the worker with --allow-commit-mismatch, or commit and restart the hub first.\n",
        );
    }
    out
}

/// Rows of the per-worker blocks: a label, then one column per window and one since the start.
const ROW_LABEL: usize = 18;
const ROW_COL: usize = 7;
/// Intervals in the table under the workers (an hour of 5-minute intervals).
const TABLE_INTERVALS: usize = 12;

/// `▁▂▃▄▅▆▇█`, scaled to the largest value; `·` for an interval with nothing in it.
pub fn spark(values: &[f64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let max = values.iter().copied().fold(0.0, f64::max);
    values
        .iter()
        .map(|&x| {
            if x <= 0.0 || max <= 0.0 {
                '·'
            } else {
                BARS[((x / max) * 7.0).round().clamp(0.0, 7.0) as usize]
            }
        })
        .collect()
}

/// The workers section: per worker name, then the fleet, games and decisions per minute over the
/// last 5 and 30 minutes and since the start, with three hours of intervals as a sparkline.
fn throughput_lines(out: &mut String, s: &HubStatus) {
    use crate::rates::num;
    let t = &s.throughput;
    out.push_str(&format!("\nworkers ({} connected)\n", s.workers.len()));
    let mut head = format!("{:<ROW_LABEL$}", "  per minute");
    for w in &t.windows {
        head.push_str(&format!("{:>ROW_COL$}", format!("{}m", w / 60)));
    }
    head.push_str(&format!("{:>ROW_COL$}", "all"));
    out.push_str(&head);
    out.push('\n');
    let block = |out: &mut String, w: &WorkerRates, detail: String| {
        out.push_str(&format!("  {}{detail}\n", w.name));
        let all = w.since_start.counts;
        let row = |out: &mut String, label: &str, pick: &dyn Fn(&crate::api::Counts) -> u64| {
            let mut l = format!("{:<ROW_LABEL$}", format!("    {label}"));
            for r in w.windows.iter().chain(std::iter::once(&w.since_start)) {
                l.push_str(&format!("{:>ROW_COL$}", num(r.per_min(pick(&r.counts)))));
            }
            out.push_str(&l);
            out.push('\n');
        };
        if all.games > 0 || all.eval_games == 0 {
            row(out, "games", &|c| c.games);
            // `--next-drive` games write two records when they score.
            if all.records != all.games {
                row(out, "records", &|c| c.records);
            }
            row(out, "decisions", &|c| c.samples);
        }
        if all.eval_games > 0 {
            row(out, "eval games", &|c| c.eval_games);
            row(out, "eval decisions", &|c| c.eval_decisions);
        }
        let per_min: Vec<f64> = w
            .history
            .iter()
            .zip(&t.history_secs)
            .map(|(c, secs)| {
                Rate {
                    counts: *c,
                    secs: *secs as f64,
                }
                .per_min(c.all_games())
            })
            .collect();
        if per_min.iter().any(|x| *x > 0.0) {
            let peak = per_min.iter().copied().fold(0.0, f64::max);
            out.push_str(&format!(
                "    {} games/min, {} in {}-min steps to {}, peak {}\n",
                spark(&per_min),
                dur(Duration::from_secs(t.bucket_secs * per_min.len() as u64)),
                t.bucket_secs / 60,
                crate::rates::clock(t.history_end),
                num(peak)
            ));
        }
    };
    for w in &t.workers {
        let conns: Vec<&crate::api::WorkerStatus> = s.workers.iter().filter(|c| c.name == w.name).collect();
        let detail = if w.connected == 0 {
            " · gone".to_string()
        } else {
            let seen = conns.iter().map(|c| c.last_seen_secs).min().unwrap_or(0);
            let cores: u32 = conns.iter().map(|c| c.cores as u32).sum();
            let triple = conns.first().map_or("", |c| c.triple.as_str());
            let mut d = format!(" · {}/{} streams busy · {cores} cores · {triple}", w.busy, w.streams);
            if conns.len() > 1 {
                d.push_str(&format!(" · {} connections", conns.len()));
            }
            d.push_str(&format!(" · seen {seen}s ago"));
            d
        };
        block(out, w, detail);
    }
    if t.workers.len() != 1 {
        let total = &t.total;
        block(out, total, format!(" · {}/{} streams busy", total.busy, total.streams));
    }
}

/// The last hour's intervals, newest first: generate games/min · decisions/min per worker. This is
/// what tells a fleet that slowed from a worker that left, after the fact.
fn interval_lines(out: &mut String, t: &Throughput) {
    use crate::rates::num;
    let n = t.history_secs.len();
    if n == 0 {
        return;
    }
    let shown = n.min(TABLE_INTERVALS);
    let recent = |w: &WorkerRates| w.history[n - shown..].iter().any(|c| !c.is_empty());
    if !recent(&t.total) {
        return;
    }
    let mut cols: Vec<&WorkerRates> = t.workers.iter().filter(|w| recent(w)).collect();
    if cols.len() != 1 {
        cols.push(&t.total);
    }
    let evals = t.total.history[n - shown..].iter().any(|c| c.eval_games > 0);
    let cell = |c: &crate::api::Counts, secs: u64| {
        let r = Rate {
            counts: *c,
            secs: secs as f64,
        };
        if c.games == 0 && c.samples == 0 {
            "–".to_string()
        } else {
            format!("{}·{}", num(r.per_min(c.games)), num(r.per_min(c.samples)))
        }
    };
    let widths: Vec<usize> = cols
        .iter()
        .map(|w| {
            (n - shown..n)
                .map(|i| cell(&w.history[i], t.history_secs[i]).chars().count())
                .chain([w.name.chars().count()])
                .max()
                .unwrap_or(0)
                + 2
        })
        .collect();
    out.push_str(&format!(
        "\nlast {} by {}-min interval: generate games/min · decisions/min{}\n",
        dur(Duration::from_secs(t.bucket_secs * shown as u64)),
        t.bucket_secs / 60,
        if evals { "; eval games/min" } else { "" }
    ));
    let mut head = "  ending".to_string();
    for (w, width) in cols.iter().zip(&widths) {
        head.push_str(&format!("{:>width$}", w.name));
    }
    if evals {
        head.push_str("   eval");
    }
    out.push_str(&head);
    out.push('\n');
    for i in (n - shown..n).rev() {
        let end = t.history_end - t.bucket_secs * (n - 1 - i) as u64;
        let mut l = format!("  {:<6}", crate::rates::clock(end));
        for (w, width) in cols.iter().zip(&widths) {
            l.push_str(&format!("{:>width$}", cell(&w.history[i], t.history_secs[i])));
        }
        if evals {
            let c = &t.total.history[i];
            let r = Rate {
                counts: *c,
                secs: t.history_secs[i] as f64,
            };
            l.push_str(&format!("{:>7}", num(r.per_min(c.eval_games))));
        }
        out.push_str(&l);
        out.push('\n');
    }
}

fn training_line(t: &Training, now: SystemTime) -> Option<String> {
    let p = &t.progress;
    if p.done {
        return None;
    }
    let mut s = format!("  training {} on this box: step {}", t.generation, p.step);
    if let Some(total) = p.total_steps.filter(|&n| n > 0) {
        s.push_str(&format!("/{total} ({:.0}%)", 100.0 * p.step as f64 / total as f64));
        let started = UNIX_EPOCH + Duration::from_secs_f64(p.started.max(0.0));
        if let Ok(elapsed) = now.duration_since(started) {
            if p.step > 0 && t.age < TRAIN_STALE {
                let left = elapsed.mul_f64(total.saturating_sub(p.step) as f64 / p.step as f64);
                s.push_str(&format!(", {} in, ~{} left", dur(elapsed), dur(left)));
            }
        }
    }
    if t.age >= TRAIN_STALE {
        s.push_str(&format!("  — no progress for {}, stopped?", dur(t.age)));
    }
    s.push('\n');
    let f = |v: Option<f64>| v.map_or("–".to_string(), |v| format!("{v:.4}"));
    s.push_str(&format!(
        "    value loss: train {}  val {}",
        f(p.train_value),
        f(p.val_value)
    ));
    if let (Some(b), Some(at)) = (p.best_val_value, p.best_step) {
        s.push_str(&format!("  (best val {b:.4} at step {at})"));
    }
    if let Some(b) = p.baseline_val_value {
        s.push_str(&format!("  warm start {b:.4}"));
    }
    s.push('\n');
    Some(s)
}

/// A `status.md` line: `[2026-10-03 06:29:52] text` becomes `10-03 06:29 text`, labels short.
fn status_line(l: &str) -> String {
    let (when, text) = match l.strip_prefix('[').and_then(|r| r.split_once("] ")) {
        Some((ts, text)) if ts.len() >= 16 => (format!("{} ", &ts[5..16]), text),
        _ => (String::new(), l),
    };
    let text = short_label(text);
    let text = match text.char_indices().nth(220) {
        Some((i, _)) => format!("{}…", &text[..i]),
        None => text,
    };
    format!("{when}{text}")
}

fn job_lines(out: &mut String, j: &JobStatus) {
    let label = j.label.clone().unwrap_or_else(|| {
        format!(
            "job {} {}",
            j.id,
            if j.kind == JobKind::Eval { "eval" } else { "generate" }
        )
    });
    // A rung with an SPRT verdict takes no more games, so it counts as complete.
    let decided = |u: &crate::api::UnitProgress| matches!(&u.stats, Some(UnitStats::Eval(e)) if e.sprt.is_some_and(|s| s.verdict != botbowl_play::stats::Verdict::Undecided));
    let total: u32 = j.units.iter().map(|u| u.total).sum();
    let done: u32 = j
        .units
        .iter()
        .map(|u| if decided(u) { u.total } else { u.done.min(u.total) })
        .sum();
    let played: u32 = j.units.iter().map(|u| u.done).sum();
    let what = if j.units.iter().any(|u| u.name.contains(" drives(")) {
        "drives"
    } else {
        "games"
    };
    let elapsed = Duration::from_secs(j.elapsed_secs);
    let per_min = |n: u64, secs: f64| if secs > 0.0 { n as f64 * 60.0 / secs } else { 0.0 };
    let num = crate::rates::num;
    let head = match &j.state {
        JobState::Running => {
            let mut h = format!("▶ {label}  {played}/{total} {what} · {}", dur(elapsed));
            // The rate over the last 5 minutes when there is one, else the whole job's.
            let recent = j.recent.filter(|r| r.counts.all_games() > 0 && r.secs > 0.0);
            let (rate, over) = match recent {
                Some(r) => (
                    r.per_min(r.counts.all_games()),
                    format!("last {}", dur(Duration::from_secs_f64(r.secs))),
                ),
                None => (per_min(played as u64, elapsed.as_secs_f64()), "so far".to_string()),
            };
            if rate > 0.0 {
                h.push_str(&format!(" · {}/min {over}", num(rate)));
                if let Some(r) = recent.filter(|r| r.counts.samples > 0) {
                    h.push_str(&format!(", {} decisions/min", num(r.per_min(r.counts.samples))));
                }
            }
            if done > 0 && done < total {
                let left = if rate > 0.0 {
                    Duration::from_secs_f64((total - done) as f64 * 60.0 / rate)
                } else {
                    elapsed.mul_f64((total - done) as f64 / done as f64)
                };
                let sprt = j.kind == JobKind::Eval
                    && j.units
                        .iter()
                        .any(|u| matches!(&u.stats, Some(UnitStats::Eval(e)) if e.sprt.is_some()));
                // An SPRT may stop early, so its estimate is an upper bound.
                h.push_str(&format!(" · {}{} left", if sprt { "at most " } else { "~" }, dur(left)));
            }
            h
        }
        JobState::Done => {
            let mut h = format!("✓ {label}  {played} {what} in {}", dur(elapsed));
            let rate = per_min(played as u64, elapsed.as_secs_f64());
            if rate > 0.0 {
                h.push_str(&format!(" · {}/min", num(rate)));
            }
            h
        }
        JobState::Failed { error } => format!("✗ {label}  failed after {}: {}", dur(elapsed), short_label(error)),
    };
    out.push_str(&format!("  {head}\n"));
    // Who played it: each worker's share of the job's games.
    let games: u64 = j.by_worker.values().map(|c| c.all_games()).sum();
    if games > 0 {
        let mut shares: Vec<(&String, u64)> = j.by_worker.iter().map(|(w, c)| (w, c.all_games())).collect();
        shares.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let parts: Vec<String> = shares
            .iter()
            .map(|(w, n)| format!("{w} {:.0}% ({n})", 100.0 * *n as f64 / games as f64))
            .collect();
        out.push_str(&format!("      by worker: {}\n", parts.join(" · ")));
    }
    match j.kind {
        JobKind::Eval => {
            let w = j
                .units
                .iter()
                .map(|u| short_label(&u.name).chars().count())
                .max()
                .unwrap_or(0);
            for u in &j.units {
                let name = short_label(&u.name);
                let pad = w.saturating_sub(name.chars().count());
                let mut l = format!("      {name}{} {:>4}/{:<4}", " ".repeat(pad), u.done, u.total);
                if let Some(UnitStats::Eval(e)) = &u.stats {
                    if e.games() > 0 {
                        let n = e.games() as f64;
                        l.push_str(&format!("  {:.3}", e.points()));
                        // Below the SPRT's own minimum the variance estimate is noise, often 0.
                        if e.pairs >= botbowl_play::stats::SPRT_MIN_PAIRS {
                            l.push_str(&format!(" ± {:.3} ({} pairs)", e.points_se, e.pairs));
                        } else if e.pairs > 0 {
                            l.push_str(&format!(" ({} pairs)", e.pairs));
                        }
                        l.push_str(&format!(
                            "  W{} D{} L{}  TD {:.2}-{:.2}",
                            e.wins,
                            e.draws,
                            e.losses,
                            e.tds_for as f64 / n,
                            e.tds_against as f64 / n
                        ));
                        if let Some(s) = e.sprt {
                            l.push_str(&format!(
                                "  SPRT {}:{} {}",
                                s.rule.s0,
                                s.rule.s1,
                                match s.verdict {
                                    botbowl_play::stats::Verdict::H1 => "H1 (better)".to_string(),
                                    botbowl_play::stats::Verdict::H0 => "H0 (not better)".to_string(),
                                    botbowl_play::stats::Verdict::Undecided =>
                                        format!("LLR {:.2} in [{:.2}, {:.2}]", s.llr, s.lower, s.upper),
                                }
                            ));
                        }
                    }
                }
                out.push_str(&l);
                out.push('\n');
            }
        }
        JobKind::Generate => {
            let mut corpus = GenStats::default();
            let mut samples = 0u64;
            for u in &j.units {
                samples += u.samples;
                if let Some(UnitStats::Generate(g)) = &u.stats {
                    corpus.merge(g);
                }
            }
            if corpus.drives > 0 {
                let d = corpus.drives as f64;
                out.push_str(&format!(
                    "      {} shards · {samples} samples · TD rate {:.3} · {:.1} steps/drive\n",
                    j.units.len(),
                    corpus.scored as f64 / d,
                    corpus.steps as f64 / d
                ));
                // The per-board breakdown is for watching a running corpus; a finished one has
                // its td_rate.json.
                if j.state == JobState::Running {
                    let boards: Vec<String> = corpus
                        .by_board
                        .iter()
                        .map(|(b, s)| {
                            format!(
                                "{} {:.2}",
                                b.split('/').next().unwrap_or(b),
                                s.scored as f64 / s.drives.max(1) as f64
                            )
                        })
                        .collect();
                    out.push_str(&format!("      TD rate by board: {}\n", boards.join(" · ")));
                }
            }
        }
    }
}

/// `2h05m`, `14m`, `40s`.
pub fn dur(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, s % 3600 / 60)
    } else if s >= 60 {
        format!("{}m", s / 60)
    } else {
        format!("{s}s")
    }
}

/// A model file as we call it: `…/bbnet_mix16x9d1k_gen02.onnx` and `anchor_mix16x9_gen13.onnx`
/// are `gen02` and `gen13`; any other model is its file stem.
pub fn short_model(path: &str) -> String {
    let base = path.rsplit('/').next().unwrap_or(path);
    let stem = base
        .strip_suffix(".onnx")
        .or_else(|| base.strip_suffix(".pt"))
        .unwrap_or(base);
    if stem.starts_with("bbnet_") || stem.starts_with("anchor_") {
        if let Some((_, g)) = stem.rsplit_once('_') {
            if g.len() > 3 && g.starts_with("gen") && g[3..].chars().all(|c| c.is_ascii_digit()) {
                return g.to_string();
            }
        }
    }
    stem.to_string()
}

/// A rung name or a status line, for reading: `vs:mcts(nn:/…/anchor_mix16x9_gen13.onnx)
/// [puct=raw(c=10)]@14x7/4` becomes `vs gen13 @14x7/4`. Model files become their
/// [`short_model`] name, other absolute paths their last two components, and the default PUCT
/// knob — on every rung, so it says nothing — is dropped.
pub fn short_label(s: &str) -> String {
    let default_puct = botbowl_mcts::PuctMode::raw().label();
    let mut s = s.replace(&format!(" [{default_puct}]"), "");
    s = s.replace(&format!("{default_puct} "), "");
    for (open, suffix) in [("mcts(nn:", ""), ("mcts(nn-value:", " (value head)")] {
        while let Some(i) = s.find(open) {
            let Some(j) = s[i..].find(')') else { break };
            let model = short_model(&s[i + open.len()..i + j]);
            // `gen13@14x7/4` reads better as `gen13 @14x7/4`.
            let gap = if s[i + j + 1..].starts_with('@') { " " } else { "" };
            s.replace_range(i..i + j + 1, &format!("{model}{suffix}{gap}"));
        }
    }
    if let Some(rest) = s.strip_prefix("vs:") {
        s = format!("vs {rest}");
    }
    let mut out = String::with_capacity(s.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        out.push_str(&short_word(word));
        word.clear();
    };
    for c in s.chars() {
        if c.is_whitespace() || "(),;'\"[]".contains(c) {
            flush(&mut word, &mut out);
            out.push(c);
        } else {
            word.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// One whitespace-free token: a path or a model file, possibly with an `@board` suffix.
fn short_word(w: &str) -> String {
    let (head, at) = match w.find('@') {
        Some(i) => (&w[..i], &w[i..]),
        None => (w, ""),
    };
    let is_model = head.ends_with(".onnx") || head.ends_with(".pt");
    let short = if is_model {
        let m = short_model(head);
        if at.is_empty() {
            m
        } else {
            format!("{m} ")
        }
    } else if head.starts_with('/') && head.matches('/').count() > 2 {
        let parts: Vec<&str> = head.rsplitn(3, '/').collect();
        format!("{}/{}", parts[1], parts[0])
    } else {
        head.to_string()
    };
    format!("{short}{at}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rung_names_read_as_we_say_them() {
        assert_eq!(
            short_label(
                "vs:mcts(nn:/home/m/repos/botbowl_rust/models/az_v7/anchor_mix16x9_gen13.onnx) [puct=raw(c=10)]@14x7/4"
            ),
            "vs gen13 @14x7/4"
        );
        assert_eq!(
            short_label("vs:mcts(nn:/x/models/az_v7/bbnet_mix16x9_gen21.onnx) drives(contested_14x7)@14x7/4"),
            "vs gen21 drives(contested_14x7)@14x7/4"
        );
        // A non-default knob stays: it is what tells two arms apart.
        assert_eq!(
            short_label("vs:mcts(nn:/x/exp055_gen01data_tau100.onnx) [puct=raw(c=2)]@16x9/6"),
            "vs exp055_gen01data_tau100 [puct=raw(c=2)]@16x9/6"
        );
        assert_eq!(short_label("scripted@14x7/4"), "scripted@14x7/4");
    }

    #[test]
    fn status_lines_lose_their_paths() {
        assert_eq!(
            status_line(
                "[2026-10-03 06:29:31] gen02 curve: anchor anchor_mix16x9_gen13.onnx@14x7/4: gen02 0.460 ± 0.034"
            ),
            "10-03 06:29 gen02 curve: anchor gen13 @14x7/4: gen02 0.460 ± 0.034"
        );
        assert_eq!(
            status_line("[2026-10-03 06:29:52] gen03 train: warm start from bbnet_mix16x9d1k_gen02.pt at lr 2e-4"),
            "10-03 06:29 gen03 train: warm start from gen02 at lr 2e-4"
        );
        assert_eq!(
            short_label("FATAL: tau100 failed, see /home/m/repos/botbowl_rust/runs/exp056/tau100/eval.log"),
            "FATAL: tau100 failed, see tau100/eval.log"
        );
    }

    #[test]
    fn training_shows_progress_and_losses() {
        let started = SystemTime::now() - Duration::from_secs(600);
        let t = Training {
            generation: "gen03".into(),
            age: Duration::from_secs(5),
            progress: TrainProgress {
                started: started.duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
                total_steps: Some(10_000),
                step: 2_500,
                train_value: Some(0.0927),
                val_value: Some(0.1152),
                best_step: Some(2_500),
                best_val_value: Some(0.1152),
                baseline_val_value: Some(0.1143),
                ..Default::default()
            },
        };
        let l = training_line(&t, SystemTime::now()).unwrap();
        assert!(l.contains("step 2500/10000 (25%)"), "{l}");
        assert!(l.contains("~30m left"), "{l}");
        assert!(l.contains("train 0.0927  val 0.1152"), "{l}");
        let done = Training {
            progress: TrainProgress {
                done: true,
                ..t.progress.clone()
            },
            ..t
        };
        assert!(training_line(&done, SystemTime::now()).is_none());
    }
}
