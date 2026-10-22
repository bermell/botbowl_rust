#!/usr/bin/env bash
# Plan 069 item 3: build botbowl.sif from this checkout's HEAD and a bundle from make_bundles.sh.
# Run on an Arrhenius Arm (GPU-partition) node, e.g. inside `interactive -A <project> -p gpu ...`.
#
#   hpc/arrhenius/build_image.sh BUNDLE_DIR OUT.sif
#
# Builds the *committed* HEAD (a shallow clone of it), never the working tree: uncommitted edits are
# refused, so the image's provenance stamp is always a real, clean commit. The bundle must match
# this checkout's lockfiles (checked against its MANIFEST). Uses the bundled base image when
# BUNDLE_DIR/base/ has it (no Docker Hub needed), else pulls the pinned one.
#
# Apptainer's scratch and cache default to /tmp and ~/.apptainer: on the cluster point both at
# project storage first (the sandbox is ~12 GB while it builds), e.g.
#   export APPTAINER_TMPDIR=/nobackup/proj/disk/<project>/apptainer/tmp
#   export APPTAINER_CACHEDIR=/nobackup/proj/disk/<project>/apptainer/cache
# Extra `apptainer build` flags (e.g. --fakeroot) go in APPTAINER_BUILD_FLAGS.
set -euo pipefail
[ $# -eq 2 ] || { sed -n '2,16p' "$0"; exit 2; }
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BUNDLE="$(cd "$1" && pwd)"
mkdir -p "$(dirname "$2")"
OUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
log() { printf '[build_image] %s\n' "$*" >&2; }

[ "$(uname -m)" = aarch64 ] || log "WARNING: $(uname -m) host — an aarch64 image builds natively only on an Arm node (here it needs qemu binfmt and is slow)"
command -v apptainer > /dev/null || { echo "apptainer not on PATH" >&2; exit 1; }
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    echo "uncommitted changes in $REPO: commit first (the image builds HEAD, and its stamp must be clean)" >&2
    exit 1
fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

for f in vendor.tar.gz wheels/requirements.txt MANIFEST; do
    [ -e "$BUNDLE/$f" ] || { echo "$BUNDLE/$f missing — run make_bundles.sh" >&2; exit 1; }
done
ls "$BUNDLE"/rust/rust-*-aarch64-unknown-linux-gnu.tar.xz > /dev/null
# recon_mcts/Cargo.lock is gitignored (regenerated per checkout) and the image only builds the botbowl
# workspace, whose Cargo.lock pins recon_mcts's dependencies; so it is not checked here.
for lock in Cargo.lock train/uv.lock; do
    want=$(sha256sum "$REPO/$lock" | awk '{print $1}')
    grep -q "^$lock sha256: $want" "$BUNDLE/MANIFEST" \
        || { echo "$lock changed since the bundle was made — rerun make_bundles.sh" >&2; exit 1; }
done

log "staging $HEAD into $BUNDLE"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git clone -q --depth 1 "file://$REPO" "$TMP/src"
[ "$(git -C "$TMP/src" rev-parse HEAD)" = "$HEAD" ] || { echo "shallow clone is not at HEAD" >&2; exit 1; }
tar -C "$TMP" -cf "$BUNDLE/src.tar" src
cp "$REPO/hpc/arrhenius/botbowl.def" "$BUNDLE/botbowl.def"
BASE="$BUNDLE/base/python-3.12-bookworm-arm64.tar"
if [ -f "$BASE" ]; then
    log "base: bundled $BASE (docker-archive)"
    sed -i -e 's|^Bootstrap: docker$|Bootstrap: docker-archive|' -e "s|^From: python:3.12-bookworm@.*|From: $BASE|" \
        "$BUNDLE/botbowl.def"
else
    log "base: pulling the pinned python:3.12-bookworm (needs Docker Hub)"
fi

cd "$BUNDLE"   # %files paths are relative to here, whichever way apptainer resolves them
log "apptainer build $OUT"
# shellcheck disable=SC2086
apptainer build ${APPTAINER_BUILD_FLAGS:-} "$OUT" botbowl.def
log "done: $OUT ($(du -h "$OUT" | cut -f1)), commit $(apptainer exec "$OUT" cat /opt/botbowl/COMMIT)"
