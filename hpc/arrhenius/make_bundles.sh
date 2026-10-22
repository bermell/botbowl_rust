#!/usr/bin/env bash
# Plan 069 item 3: everything the Arrhenius image build needs from the internet, fetched here, so
# `build_image.sh` works on a node without internet. Run on any Linux box with internet, docker,
# cargo and uv (this one); copy the output dir to project storage on the cluster.
#
#   hpc/arrhenius/make_bundles.sh [OUT]       # default hpc/arrhenius/bundle (gitignored)
#
# OUT/
#   vendor.tar.gz       `cargo vendor --locked` of the botbowl workspace + recon_mcts/ (one dir,
#                       ~24k files: shipped as one tarball for the cluster's file-count quota)
#   cargo-vendor.toml   the source replacement for it, as a `cargo --config` file (paths for here;
#                       the image writes its own)
#   wheels/             the trainer's locked wheels for cp312 linux aarch64 (torch 2.13.0+cu126
#                       and its CUDA libraries), requirements.txt with hashes, and
#                       wheels/cuda-headers/ (nvcc + cccl wheels: the shim's crt/ and nv/target)
#   rust/               the standalone Rust installer for aarch64, same version as this box's rustc
#   base/python-3.12-bookworm-arm64.tar   `docker save` of the pinned arm64 base image (SKIP_BASE=1
#                       to leave it out when the build node can pull from Docker Hub)
#   MANIFEST            commit, versions, sizes
#
# The bundle is per commit only as far as Cargo.lock / uv.lock go: rerun it after either changes.
# Every step is skipped when its output is already there (delete it to refetch).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mkdir -p "${1:-$REPO/hpc/arrhenius/bundle}" && cd "${1:-$REPO/hpc/arrhenius/bundle}" && pwd)"
# Pinned base: python:3.12-bookworm's multi-arch index (2026-10-22). Same pin as botbowl.def.
BASE_REF="python:3.12-bookworm@sha256:903c1a88edb0a18dbc4e02ef07545d18eefea9839ee40761266a640f8d5213d6"
RUST_VERSION="${RUST_VERSION:-$(rustc --version | awk '{print $2}')}"
log() { printf '[make_bundles] %s\n' "$*" >&2; }

# ---- 1. cargo vendor ---------------------------------------------------------
if [ ! -f "$OUT/vendor.tar.gz" ]; then
    log "cargo vendor (botbowl workspace + recon_mcts)"
    rm -rf "$OUT/vendor"
    cargo vendor --locked --manifest-path "$REPO/Cargo.toml" --sync "$REPO/recon_mcts/Cargo.toml" \
        "$OUT/vendor" > /dev/null
    tar -C "$OUT" -czf "$OUT/vendor.tar.gz.part" vendor
    mv "$OUT/vendor.tar.gz.part" "$OUT/vendor.tar.gz"
fi
cat > "$OUT/cargo-vendor.toml" <<EOF
# cargo --config file: build the botbowl workspace (or recon_mcts/) from the vendored crates.
#   cargo build --offline --locked --config $OUT/cargo-vendor.toml ...
# (vendor.tar.gz unpacked beside this file; on another machine fix the path below)
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "$OUT/vendor"
EOF
[ -d "$OUT/vendor" ] || tar -C "$OUT" -xzf "$OUT/vendor.tar.gz"

# ---- 2. Python wheelhouse (aarch64, from uv.lock) ----------------------------
log "wheelhouse (cp312 linux aarch64, from train/uv.lock)"
uv run --no-project --python 3.12 --with packaging \
    python "$REPO/hpc/arrhenius/wheelhouse.py" "$OUT/wheels" --python 3.12 --machine aarch64 --glibc 2.36

# ---- 3. Rust standalone installer --------------------------------------------
RUST_TAR="rust-$RUST_VERSION-aarch64-unknown-linux-gnu.tar.xz"
mkdir -p "$OUT/rust"
if [ ! -f "$OUT/rust/$RUST_TAR" ]; then
    log "Rust $RUST_VERSION standalone installer (aarch64)"
    curl -sSfL -o "$OUT/rust/$RUST_TAR.part" "https://static.rust-lang.org/dist/$RUST_TAR"
    curl -sSfL -o "$OUT/rust/$RUST_TAR.sha256" "https://static.rust-lang.org/dist/$RUST_TAR.sha256"
    want=$(awk '{print $1}' "$OUT/rust/$RUST_TAR.sha256")
    got=$(sha256sum "$OUT/rust/$RUST_TAR.part" | awk '{print $1}')
    [ "$want" = "$got" ] || { echo "sha256 mismatch for $RUST_TAR" >&2; exit 1; }
    mv "$OUT/rust/$RUST_TAR.part" "$OUT/rust/$RUST_TAR"
fi

# ---- 4. base image (docker-archive) ------------------------------------------
BASE_TAR="$OUT/base/python-3.12-bookworm-arm64.tar"
if [ "${SKIP_BASE:-0}" != 1 ] && [ ! -f "$BASE_TAR" ]; then
    log "base image $BASE_REF (linux/arm64) → docker-archive"
    mkdir -p "$OUT/base"
    # Pulling by digest with --platform fetches the arm64 manifest only (no emulation needed).
    docker pull -q --platform linux/arm64 "$BASE_REF" > /dev/null
    id=$(docker image inspect --format '{{.Id}}' "$BASE_REF")
    arch=$(docker image inspect --format '{{.Architecture}}' "$id")
    [ "$arch" = arm64 ] || { echo "pulled $arch, not arm64" >&2; exit 1; }
    docker tag "$id" botbowl-base:python-3.12-bookworm-arm64
    docker save -o "$BASE_TAR.part" botbowl-base:python-3.12-bookworm-arm64
    mv "$BASE_TAR.part" "$BASE_TAR"
    docker rmi -f botbowl-base:python-3.12-bookworm-arm64 > /dev/null
fi

# ---- manifest ----------------------------------------------------------------
{
    echo "commit: $(git -C "$REPO" rev-parse HEAD)$(git -C "$REPO" diff --quiet HEAD -- Cargo.lock recon_mcts/Cargo.lock train/uv.lock || echo ' (lockfiles dirty)')"
    echo "made: $(date -u +%FT%TZ) on $(hostname)"
    echo "Cargo.lock sha256: $(sha256sum "$REPO/Cargo.lock" | awk '{print $1}')"
    echo "recon_mcts/Cargo.lock sha256: $(sha256sum "$REPO/recon_mcts/Cargo.lock" | awk '{print $1}')"
    echo "train/uv.lock sha256: $(sha256sum "$REPO/train/uv.lock" | awk '{print $1}')"
    echo "rust: $RUST_VERSION"
    echo "base: $BASE_REF$([ -f "$BASE_TAR" ] || echo ' (not bundled)')"
    echo "sizes:"
    (cd "$OUT" && du -sh vendor.tar.gz wheels rust base 2>/dev/null | sed 's/^/  /')
    echo "  total (without the unpacked vendor/): $(du -sch --exclude=vendor "$OUT" | tail -1 | cut -f1)"
} > "$OUT/MANIFEST"
cat "$OUT/MANIFEST"
