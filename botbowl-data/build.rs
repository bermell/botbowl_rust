//! Capture the git commit the binary was built at, so every trajectory
//! we later record can be pinned to the exact engine that produced it
//! (the game rules change often — a dataset is only meaningful next to
//! the commit that generated it).
//!
//! The commit is baked in at *build* time (not read at run time) on
//! purpose: it describes the code inside this binary, which is what
//! actually generated the data, regardless of where/when it runs or what
//! the working tree looks like later.

use std::process::Command;

fn main() {
    let commit = run_git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    // `--porcelain` prints one line per dirty path; empty output == clean.
    let dirty = run_git(&["status", "--porcelain"])
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);

    println!("cargo:rustc-env=BOTBOWL_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=BOTBOWL_GIT_DIRTY={dirty}");

    // Rebuild (refreshing the stamped commit) when HEAD moves or the
    // index changes. Best-effort: missing paths are simply ignored by
    // cargo. `.git/logs/HEAD` covers checkouts/commits/resets.
    for p in [".git/HEAD", ".git/index", ".git/logs/HEAD"] {
        // Paths are relative to the repo root; build.rs runs in the crate
        // dir, so reach up one level.
        println!("cargo:rerun-if-changed=../{p}");
    }
}

fn run_git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}
