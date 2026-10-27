//! Resuming a job from what is already on disk (plan 041 phase 3, plan 069 item 5).
//!
//! The hub keeps no job state across a restart. It does not need to: every result it accepts is
//! appended to a file under one lock and flushed, so **the files are the record**. A job submitted
//! with `resume` reads its outputs first and plays only what is missing:
//!
//! - a **generate** shard ([`scan_shard`]): every line names its game (`meta.seed`, and game
//!   `g = seed - shard seed`), so the complete games are known exactly and the resumed job plays
//!   the other seeds — the same seed set an uninterrupted run plays, so the same corpus up to the
//!   order of lines and the search's own run-to-run nondeterminism;
//! - an **eval** per-game file ([`scan_eval`]): every `EvalGameLine` names its rung and game; the
//!   lines are folded into the rows again in file order (so an SPRT's state is exactly what it was
//!   when the hub died) and only the other games are played.
//!
//! What a cut can leave behind, and what is done with it:
//!
//! - bytes after the last newline: a line torn mid-write (`kill -9`, a node lost). Cut off.
//! - a `--next-drive` game whose first drive scored (and did not end the game) but whose second
//!   record is missing: the cut may have come between a game's two lines. The whole game is cut off
//!   and played again (only the last game of a file can be in this state: games are written whole,
//!   one after the other). A score on a half's last turn has no second record and looks the same,
//!   so such a last game is replayed too — one game, the same seed.
//! - a last line that does not parse: cut off, like a torn one.
//!
//! Anything else that does not fit — a line from another seed range, another generator, a
//! duplicate game, an unreadable line mid-file — is refused with the reason: those are not cuts,
//! and resuming onto them would mix corpora. `--truncate` (or removing the file) starts over.
//!
//! Scanning reads files and changes nothing; the cut ([`cut`]) is made under the hub's lock when
//! the job is inserted, after checking the file has not grown since it was scanned.

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use botbowl_play::eval::EvalGameLine;

/// What a generate shard's file already holds.
#[derive(Clone, Debug, Default)]
pub struct ShardScan {
    /// Games `g < games` whose every record is on disk.
    pub done: Vec<u32>,
    /// Records (lines) kept.
    pub lines: u64,
    /// Games kept with `g >= games` (a previous run asked for more games per shard): left in place.
    pub beyond: u32,
    /// The file's length when scanned (0 when it does not exist).
    pub file_len: u64,
    /// The length to keep: everything after it is a torn line or an incomplete game.
    pub keep_len: u64,
    /// What was found to cut off, for the log.
    pub cut: Vec<String>,
}

/// What an eval per-game file already holds.
#[derive(Clone, Debug, Default)]
pub struct EvalScan {
    /// Every complete line, in file order.
    pub lines: Vec<EvalGameLine>,
    pub file_len: u64,
    pub keep_len: u64,
    pub cut: Vec<String>,
}

/// The few fields of a corpus line's `meta` a resume needs.
#[derive(serde::Deserialize)]
struct MetaPeek {
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    home_bot: String,
    #[serde(default)]
    extra: BTreeMap<String, String>,
}

impl MetaPeek {
    fn drive(&self) -> u32 {
        self.extra.get("drive").and_then(|d| d.parse().ok()).unwrap_or(1)
    }

    fn start_goals(&self) -> u32 {
        self.extra
            .get("start_score")
            .and_then(|s| s.split_once('-'))
            .and_then(|(h, a)| Some(h.parse::<u32>().ok()? + a.parse::<u32>().ok()?))
            .unwrap_or(0)
    }
}

/// The `meta` object of a corpus line, read without the samples: `Trajectory` serialises `meta`
/// first, so it is the prefix `{"meta":{...}` and only that prefix is parsed (a line is ~0.5 MB,
/// its meta under 2 KB, and a generation is gigabytes).
fn peek_meta(line: &[u8]) -> Option<MetaPeek> {
    const KEY: &[u8] = b"{\"meta\":";
    let body = line.strip_prefix(KEY)?;
    let end = object_end(body)?;
    serde_json::from_slice(&body[..end]).ok()
}

/// The length of the JSON object `s` starts with (through its closing brace), or `None`.
fn object_end(s: &[u8]) -> Option<usize> {
    if s.first() != Some(&b'{') {
        return None;
    }
    let (mut depth, mut in_str, mut esc) = (0u32, false, false);
    for (i, &b) in s.iter().enumerate() {
        if in_str {
            match (esc, b) {
                (true, _) => esc = false,
                (false, b'\\') => esc = true,
                (false, b'"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// `None` when the line does not parse to the end; else whether a `--next-drive` game could have
/// a second record after it: its drive scored and the game did not end. (A score on a half's last
/// turn has no follow-on either, but the line cannot tell that apart: such a game is played again
/// when it is the last of its file, which costs one game, never a wrong corpus.)
fn may_follow(line: &[u8], meta: &MetaPeek) -> Option<bool> {
    #[derive(serde::Deserialize)]
    struct Outcome {
        home_score: u8,
        away_score: u8,
        #[serde(default)]
        game_over: bool,
    }
    #[derive(serde::Deserialize)]
    struct Whole {
        #[allow(dead_code)]
        meta: serde::de::IgnoredAny,
        #[allow(dead_code)]
        samples: serde::de::IgnoredAny,
        outcome: Outcome,
    }
    let w: Whole = serde_json::from_slice(line).ok()?;
    let scored = w.outcome.home_score as u32 + w.outcome.away_score as u32 > meta.start_goals();
    Some(scored && !w.outcome.game_over)
}

/// One game's records as found, in file order.
struct Found {
    game: u32,
    start: u64,
    lines: u32,
    /// The first record, kept only while it is the file's last game (to judge it whole).
    last_first: Option<(Vec<u8>, MetaPeek)>,
}

/// Read a generate shard written by the hub (or `botbowl-ui dataset`): which of games
/// `0..games` (seeds `seed..seed + games`) are complete, and where the file must be cut. `label` is
/// the `home_bot` every line must carry (`botbowl_play::generate::budget_label` of the job's
/// config): a shard made by another net or configuration is refused, not extended.
pub fn scan_shard(path: &Path, seed: u64, games: u32, label: &str, next_drive: bool) -> io::Result<ShardScan> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(ShardScan::default()),
        Err(e) => return Err(e),
    };
    let file_len = file.metadata()?.len();
    let bad = |n: u64, why: String| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} line {n}: {why} — not a cut this job can resume from (remove the file or pass --truncate)",
                path.display()
            ),
        )
    };
    let mut r = BufReader::with_capacity(1 << 20, file);
    let mut buf = Vec::new();
    let mut scan = ShardScan {
        file_len,
        ..Default::default()
    };
    let mut seen: HashSet<u32> = HashSet::new();
    let mut games_found: Vec<Found> = Vec::new();
    let (mut offset, mut n) = (0u64, 0u64);
    loop {
        buf.clear();
        let read = r.read_until(b'\n', &mut buf)? as u64;
        if read == 0 {
            break;
        }
        n += 1;
        if buf.last() != Some(&b'\n') {
            scan.cut.push(format!("a torn last line ({read} bytes)"));
            break;
        }
        let line = &buf[..buf.len() - 1];
        let Some(meta) = peek_meta(line) else {
            // An unreadable line is a cut only at the very end of the file.
            if offset + read == file_len {
                scan.cut.push(format!("an unreadable last line ({read} bytes)"));
                break;
            }
            return Err(bad(n, "unreadable meta".into()));
        };
        let Some(s) = meta.seed else {
            return Err(bad(n, "no meta.seed".into()));
        };
        if meta.home_bot != label {
            return Err(bad(
                n,
                format!("generated by {:?}, this job generates {label:?}", meta.home_bot),
            ));
        }
        let g = s
            .checked_sub(seed)
            .and_then(|g| u32::try_from(g).ok())
            .ok_or_else(|| bad(n, format!("seed {s} is not in this shard's range (from {seed})")))?;
        if meta.drive() >= 2 {
            // A follow-on record belongs to the game just before it.
            match games_found.last_mut() {
                Some(f) if f.game == g && f.lines == 1 => {
                    f.lines += 1;
                    f.last_first = None;
                }
                _ => {
                    return Err(bad(
                        n,
                        format!("a drive-{} record of seed {s} without its game", meta.drive()),
                    ))
                }
            }
        } else {
            if !seen.insert(g) {
                return Err(bad(n, format!("seed {s} appears twice")));
            }
            if let Some(prev) = games_found.last_mut() {
                prev.last_first = None;
            }
            games_found.push(Found {
                game: g,
                start: offset,
                lines: 1,
                last_first: Some((line.to_vec(), meta)),
            });
        }
        offset += read;
    }
    scan.keep_len = offset;
    // The last game is whole unless it is one drive that scored under `--next-drive` (its follow-on
    // may be what the cut lost: [`may_follow`]), or its last line does not parse to the end.
    if let Some(last) = games_found.last() {
        let whole = match &last.last_first {
            None => true, // two records: complete by construction
            Some((line, meta)) => match may_follow(line, meta) {
                None => false,
                Some(follows) => !(next_drive && follows),
            },
        };
        if !whole {
            scan.cut.push(format!(
                "seed {}'s game, which may be incomplete ({} record(s) on disk)",
                seed + last.game as u64,
                last.lines
            ));
            scan.keep_len = last.start;
            games_found.pop();
        }
    }
    for f in &games_found {
        scan.lines += f.lines as u64;
        if f.game < games {
            scan.done.push(f.game);
        } else {
            scan.beyond += 1;
        }
    }
    scan.done.sort_unstable();
    Ok(scan)
}

/// [`scan_shard`] for every shard of a generate request, as its shards will be played.
pub fn scan_generate(req: &crate::api::GenerateJobRequest) -> io::Result<Vec<ShardScan>> {
    req.shards
        .iter()
        .map(|s| {
            let label = botbowl_play::generate::budget_label(&s.cfg);
            scan_shard(&s.out, s.seed, s.games, &label, s.cfg.next_drive)
        })
        .collect()
}

/// Read an eval job's per-game file: every complete line, in order, and where to cut a torn tail.
pub fn scan_eval(path: &Path) -> io::Result<EvalScan> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(EvalScan::default()),
        Err(e) => return Err(e),
    };
    let file_len = file.metadata()?.len();
    let mut r = BufReader::new(file);
    let mut buf = Vec::new();
    let mut scan = EvalScan {
        file_len,
        ..Default::default()
    };
    let (mut offset, mut n) = (0u64, 0u64);
    loop {
        buf.clear();
        let read = r.read_until(b'\n', &mut buf)? as u64;
        if read == 0 {
            break;
        }
        n += 1;
        let line = buf.strip_suffix(b"\n");
        match line.map(serde_json::from_slice::<EvalGameLine>) {
            Some(Ok(l)) => scan.lines.push(l),
            None => {
                scan.cut.push(format!("a torn last line ({read} bytes)"));
                break;
            }
            Some(Err(_)) if offset + read == file_len => {
                scan.cut.push(format!("an unreadable last line ({read} bytes)"));
                break;
            }
            Some(Err(e)) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} line {n}: {e}", path.display()),
                ))
            }
        }
        offset += read;
    }
    scan.keep_len = offset;
    Ok(scan)
}

/// Cut `path` to `keep_len`, if it is still `scanned_len` long. Called under the hub's lock as the
/// job is inserted: a file that changed since it was scanned is someone else's write in progress.
pub fn cut(path: &Path, scanned_len: u64, keep_len: u64) -> io::Result<()> {
    if scanned_len == keep_len {
        return Ok(());
    }
    let f = File::options().write(true).open(path)?;
    let now = f.metadata()?.len();
    if now != scanned_len {
        return Err(io::Error::other(format!(
            "{} changed while it was read ({scanned_len} -> {now} bytes): another writer?",
            path.display()
        )));
    }
    f.set_len(keep_len)?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const LABEL: &str = "mcts(iters=2,workers=1,eval=heuristic)";

    fn line(seed: u64, drive: u32, start: &str, home: u8, away: u8) -> String {
        format!(
            r#"{{"meta":{{"seed":{seed},"home_bot":"{LABEL}","note":"a }} in a \" string {{","extra":{{"drive":"{drive}","start_score":"{start}"}}}},"samples":[{{"x":[1,2,{{"y":"}}"}}]}}],"outcome":{{"home_score":{home},"away_score":{away}}}}}"#
        ) + "\n"
    }

    fn file(tag: &str, body: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("hub-resume-{tag}-{}.jsonl", std::process::id()));
        std::fs::File::create(&p).unwrap().write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn whole_games_are_kept_and_a_torn_tail_is_cut() {
        let good = line(100, 1, "0-0", 0, 0) + &line(102, 1, "0-0", 1, 0);
        let p = file("torn", &(good.clone() + &line(101, 1, "0-0", 0, 0)[..40]));
        let s = scan_shard(&p, 100, 4, LABEL, false).unwrap();
        assert_eq!(s.done, vec![0, 2]);
        assert_eq!(s.keep_len, good.len() as u64);
        assert_eq!(s.lines, 2);
        assert_eq!(s.cut.len(), 1);
        cut(&p, s.file_len, s.keep_len).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), good);
    }

    #[test]
    fn a_scored_drive_without_its_follow_on_is_played_again_under_next_drive() {
        let body = line(100, 1, "0-0", 1, 0) + &line(100, 2, "1-0", 1, 1) + &line(101, 1, "1-1", 2, 1);
        let p = file("nd", &body);
        let s = scan_shard(&p, 100, 4, LABEL, true).unwrap();
        assert_eq!(s.done, vec![0], "seed 101 scored and has no drive-2 record: may be cut");
        assert_eq!(s.lines, 2);
        // Without --next-drive one record is the whole game.
        let s = scan_shard(&p, 100, 4, LABEL, false).unwrap();
        assert_eq!(s.done, vec![0, 1]);
        // A last drive that did not score is whole under --next-drive too.
        let p = file(
            "nd2",
            &(line(100, 1, "0-0", 1, 0) + &line(100, 2, "1-0", 1, 1) + &line(101, 1, "1-1", 1, 1)),
        );
        assert_eq!(scan_shard(&p, 100, 4, LABEL, true).unwrap().done, vec![0, 1]);
    }

    #[test]
    fn a_foreign_or_duplicated_line_is_refused() {
        let p = file("dup", &(line(100, 1, "0-0", 0, 0) + &line(100, 1, "0-0", 0, 0)));
        assert!(scan_shard(&p, 100, 4, LABEL, false)
            .unwrap_err()
            .to_string()
            .contains("twice"));
        let p = file("range", &line(99, 1, "0-0", 0, 0));
        assert!(scan_shard(&p, 100, 4, LABEL, false).is_err());
        let p = file("label", &line(100, 1, "0-0", 0, 0));
        assert!(scan_shard(&p, 100, 4, "mcts(other)", false)
            .unwrap_err()
            .to_string()
            .contains("generated by"));
        let p = file(
            "mid",
            &(String::from("{\"meta\":garbage\n") + &line(100, 1, "0-0", 0, 0)),
        );
        assert!(scan_shard(&p, 100, 4, LABEL, false).is_err());
    }

    #[test]
    fn games_beyond_the_target_are_kept_and_counted() {
        let p = file("beyond", &(line(100, 1, "0-0", 0, 0) + &line(105, 1, "0-0", 0, 0)));
        let s = scan_shard(&p, 100, 2, LABEL, false).unwrap();
        assert_eq!((s.done, s.beyond, s.lines), (vec![0], 1, 2));
    }

    #[test]
    fn a_missing_file_is_an_empty_shard() {
        let s = scan_shard(Path::new("/nonexistent/shard0.jsonl"), 0, 3, LABEL, false).unwrap();
        assert!(s.done.is_empty() && s.keep_len == 0);
    }

    #[test]
    fn cut_refuses_a_file_that_grew_since_the_scan() {
        let p = file("grew", &line(100, 1, "0-0", 0, 0));
        assert!(cut(&p, 3, 1).is_err());
    }
}
