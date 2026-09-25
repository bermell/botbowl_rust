//! A deliberate, untracked list of *other* commits a hub will accept workers from.
//!
//! Plan 041 decision 5 makes compatibility an exact commit match, because a semver we must
//! remember to bump is exactly what gets forgotten, and a mixed-commit corpus defeats the
//! per-trajectory stamping the programme relies on. That rule is right and stays the default.
//!
//! It is also wrong for one specific case: a commit that provably cannot change a game — a
//! README, a plan, a python script, a `botbowl-ui` label — still invalidates every helper box,
//! which then has to pull and rebuild before it can rejoin. This file is the escape hatch, and
//! it is built to be hard to use by accident:
//!
//! * It is not tracked (`.gitignore`), so it never travels with the code it talks about.
//! * It names the hub commit it applies to. **The moment you commit again it is stale**, the
//!   hub ignores it wholesale, and every worker is back to exact-match. Updating it is a
//!   deliberate act performed after the commit and before the hub starts.
//! * A worker admitted through it plays games stamped with the *hub's* commit (the corpus and
//!   `report.json` take their provenance from the hub, see `state.rs`). That is the whole
//!   assertion being made: "these commits are the same game". The hub logs every admission so
//!   the assertion can be audited afterwards.
//!
//! ```toml
//! # hub-allowed-commits.toml
//! hub_commit = "5f818d2"
//! allow = ["db7fc9c", "5a60d25"]   # docs + web-client only; no engine/mcts/nn/play change
//! note  = "checked with: git diff --stat db7fc9c..5f818d2 -- botbowl-engine botbowl-mcts"
//! ```
//!
//! `--allow-commit-mismatch` remains the blunt instrument: it accepts *any* commit and is for
//! developing the worker itself, not for running a programme.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The default file name, looked for in the hub's working directory. Absent = exact match only,
/// which is the state the repository ships in.
pub const DEFAULT_PATH: &str = "hub-allowed-commits.toml";

/// Shortest prefix we will compare. Below this a "commit" is too ambiguous to trust.
const MIN_HASH_LEN: usize = 7;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Doc {
    /// The commit the hub must be running for `allow` to mean anything.
    hub_commit: String,
    /// Worker commits accepted alongside the hub's own.
    #[serde(default)]
    allow: Vec<String>,
    /// Free text: why these are believed game-identical. Never parsed.
    #[serde(default)]
    note: Option<String>,
}

/// A loaded allowlist, already checked against the running commit.
#[derive(Debug, Clone)]
pub struct Allowlist {
    pub path: PathBuf,
    /// `Ok` = applies to this hub; `Err` = why it does not (stale, unreadable, malformed).
    state: Result<Vec<String>, String>,
    pub note: Option<String>,
}

impl Allowlist {
    /// Read `path` and resolve it against `hub_commit`. `Ok(None)` = no such file, i.e. the
    /// default strict behaviour; the hub says nothing about it.
    pub fn load(path: &Path, hub_commit: &str) -> Option<Allowlist> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => {
                return Some(Allowlist {
                    path: path.to_path_buf(),
                    state: Err(format!("cannot read: {e}")),
                    note: None,
                })
            }
        };
        let doc: Doc = match toml::from_str(&text) {
            Ok(d) => d,
            Err(e) => {
                return Some(Allowlist {
                    path: path.to_path_buf(),
                    state: Err(format!("cannot parse: {e}")),
                    note: None,
                })
            }
        };
        let state = if commit_eq(&doc.hub_commit, hub_commit) {
            Ok(doc.allow)
        } else {
            Err(format!(
                "written for hub commit {} but this hub is {}; ignored until it is updated",
                short(&doc.hub_commit),
                short(hub_commit)
            ))
        };
        Some(Allowlist {
            path: path.to_path_buf(),
            state,
            note: doc.note,
        })
    }

    /// Does this list admit a worker built from `commit`?
    pub fn admits(&self, commit: &str) -> bool {
        match &self.state {
            Ok(allow) => allow.iter().any(|a| commit_eq(a, commit)),
            Err(_) => false,
        }
    }

    /// One line for the hub log / status page.
    pub fn describe(&self) -> String {
        match &self.state {
            Ok(allow) if allow.is_empty() => format!("{}: applies, but lists no commits", self.path.display()),
            Ok(allow) => format!(
                "{}: also accepting workers on {}{}",
                self.path.display(),
                allow.iter().map(|c| short(c)).collect::<Vec<_>>().join(", "),
                match &self.note {
                    Some(n) => format!(" ({n})"),
                    None => String::new(),
                }
            ),
            Err(why) => format!("{}: {why}", self.path.display()),
        }
    }

    /// True when the file exists but is not in force — worth saying out loud at the moment a
    /// worker is turned away, because "I updated the allowlist" is then the wrong diagnosis.
    pub fn is_inert(&self) -> bool {
        self.state.is_err()
    }
}

/// Git hashes compare by prefix, either direction, case-insensitively: the file may hold short
/// hashes and the binaries carry full ones. Anything shorter than [`MIN_HASH_LEN`] never matches.
fn commit_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.trim().to_ascii_lowercase(), b.trim().to_ascii_lowercase());
    if a.len() < MIN_HASH_LEN || b.len() < MIN_HASH_LEN {
        return false;
    }
    a.starts_with(&b) || b.starts_with(&a)
}

fn short(c: &str) -> String {
    c.chars().take(12).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, body: &str) -> PathBuf {
        let p = dir.join(DEFAULT_PATH);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bb-allowlist-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn absent_file_is_silent() {
        let d = tmpdir("absent");
        assert!(Allowlist::load(&d.join(DEFAULT_PATH), "5f818d2abc").is_none());
    }

    #[test]
    fn admits_listed_commits_for_the_matching_hub() {
        let d = tmpdir("match");
        let p = write(&d, "hub_commit = \"5f818d2\"\nallow = [\"db7fc9c\", \"5a60d25\"]\n");
        let l = Allowlist::load(&p, "5f818d2ffffffffffffffffffffffffffffffff").unwrap();
        assert!(l.admits("db7fc9cfffffffffffffffffffffffffffffffff"));
        assert!(l.admits("5a60d25"));
        assert!(!l.admits("1f319a8"));
        assert!(!l.is_inert());
    }

    #[test]
    fn a_new_hub_commit_makes_it_stale() {
        let d = tmpdir("stale");
        let p = write(&d, "hub_commit = \"5f818d2\"\nallow = [\"db7fc9c\"]\n");
        let l = Allowlist::load(&p, "aaaaaaa1111").unwrap();
        assert!(!l.admits("db7fc9c"), "a stale list must admit nobody");
        assert!(l.is_inert());
        assert!(l.describe().contains("ignored until it is updated"));
    }

    #[test]
    fn malformed_is_inert_not_permissive() {
        let d = tmpdir("bad");
        let p = write(&d, "hub_commit = 5f818d2\n");
        let l = Allowlist::load(&p, "5f818d2").unwrap();
        assert!(l.is_inert());
        assert!(!l.admits("5f818d2"));
        let p = write(&d, "hub_commit = \"5f818d2\"\nallowed = [\"db7fc9c\"]\n");
        let l = Allowlist::load(&p, "5f818d2").unwrap();
        assert!(l.is_inert(), "a typo'd key must not silently become an empty list");
    }

    #[test]
    fn short_or_empty_hashes_never_match() {
        let d = tmpdir("short");
        let p = write(&d, "hub_commit = \"5f818d2\"\nallow = [\"\", \"ab\"]\n");
        let l = Allowlist::load(&p, "5f818d2").unwrap();
        assert!(!l.admits(""));
        assert!(!l.admits("ab"));
        assert!(!l.admits("abcdefgh"));
    }
}
