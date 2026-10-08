//! Build-time *fallback* git stamp, used only when `git` cannot be run at
//! process start (see `git_provenance` in `src/lib.rs`, which is the real
//! stamp).
//!
//! Deliberately **no** `rerun-if-changed` on `.git/*`: watching
//! `.git/index` rebuilt this crate and every dependant (9 crates) on each
//! `git status`/`add`/`commit`. This script now reruns only when it is
//! edited, so the value below can be arbitrarily stale — which is why the
//! runtime fallback reports it as dirty.

use std::process::Command;

fn main() {
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=BOTBOWL_BUILD_GIT_COMMIT={commit}");
    println!("cargo:rerun-if-changed=build.rs");
}
