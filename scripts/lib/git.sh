# shellcheck shell=bash
# Shared git helpers for launchers and the loop. Source it; run from the repo root.
#
#   source "$REPO/scripts/lib/git.sh"
#   require_clean_tree || exit 1
#   echo "commit $(git rev-parse --short HEAD)$(dirty_suffix)"
#   git log -1 --format=%h -- $(game_crates)

# One definition of "dirty", shared with the Rust stamp (`botbowl_data::git_provenance`):
# staged or unstaged changes to tracked files. Untracked files do not count — a new file only
# reaches the build through an edit to a tracked one. (`git diff --quiet`, used before, missed
# staged changes.)
tree_is_clean() {
    [ -z "$(git status --porcelain --untracked-files=no)" ]
}

# Prints "-dirty" when the tree is dirty, nothing otherwise — the suffix the stamps use.
dirty_suffix() {
    tree_is_clean || echo -dirty
}

# Fails (status 1, message on stderr) on a dirty tree. Launchers call this before stamping a run.
require_clean_tree() {
    tree_is_clean && return 0
    echo "FATAL: dirty tree (staged or unstaged changes to tracked files):" >&2
    git status --short --untracked-files=no >&2
    return 1
}

# The pathspec of everything a worker's games are built from: every workspace crate
# `botbowl-worker` depends on (normal deps; it reaches `botbowl-play` and through it the engine,
# search, nets and data) plus the workspace manifest and lockfile. Two commits that agree on these
# paths play the same games. Regenerate after adding a crate to that dependency tree:
#   cargo tree -p botbowl-worker --prefix none -e normal | grep -o '(/[^)]*)' | sort -u
game_crates() {
    echo botbowl-engine botbowl-curriculum botbowl-mcts botbowl-nn botbowl-data botbowl-play \
        botbowl-hub-proto botbowl-worker recon_mcts Cargo.toml Cargo.lock
}
