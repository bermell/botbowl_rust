#!/usr/bin/env bash
# Plan 069 item 3: check (or build) the workspace for aarch64 on this x86 box, C/C++/asm included.
#
# `cargo check --target aarch64-unknown-linux-gnu` alone is not enough here: tract-linalg's build
# script assembles its arm64 kernels and mimalloc-sys compiles C, both with the target's C compiler.
# This runs cargo (the host's toolchain, mounted) inside an x86 Debian bookworm container that has
# the aarch64 cross gcc 12 + glibc 2.36, the same as the image's base. No qemu, no root.
#
#   hpc/arrhenius/cross_check.sh                     # cargo check --workspace --all-targets, 20x11/7
#   hpc/arrhenius/cross_check.sh build --release -p botbowl-ui
#   TORCH_DIR=<site-packages>/torch NVCC_INCLUDE=<fetch_cuda_headers.sh dest>/include \
#     hpc/arrhenius/cross_check.sh build --release -p botbowl-ui --features botbowl-ui/torch
#   (<site-packages>: the aarch64 wheelhouse installed with `uv pip install --target`, see README)
#
# Needs `rustup target add aarch64-unknown-linux-gnu`. Env: CARGO_TARGET_DIR (default
# target/aarch64-cross), JOBS (2), BOARD_SIZE_W/H, BOARD_PLAYERS (20/11/7), RUSTFLAGS
# (-C target-cpu=neoverse-v2), TORCH_DIR / NVCC_INCLUDE / CUDA_INCLUDE (→ BOTBOWL_TORCH_DIR /
# BOTBOWL_CUDA_NVCC_INCLUDE / BOTBOWL_CUDA_INCLUDE). The torch dir is mounted with its parent (the
# wheel's nvidia/ libraries live beside it).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMG=botbowl-aarch64-cross
docker image inspect "$IMG" > /dev/null 2>&1 \
    || docker build -q -t "$IMG" -f "$REPO/hpc/arrhenius/cross.Dockerfile" "$REPO/hpc/arrhenius"
[ $# -gt 0 ] || set -- check --workspace --all-targets
CMD="$1"; shift
TD="${CARGO_TARGET_DIR:-$REPO/target/aarch64-cross}"
mkdir -p "$TD"
MOUNTS=(-v "$REPO:$REPO" -v "$HOME/.cargo:$HOME/.cargo" -v "$HOME/.rustup:$HOME/.rustup")
EXTRA=()
if [ -n "${TORCH_DIR:-}" ]; then
    SP=$(dirname "$TORCH_DIR")
    MOUNTS+=(-v "$SP:$SP:ro")
    EXTRA+=(-e "BOTBOWL_TORCH_DIR=$TORCH_DIR")
fi
if [ -n "${NVCC_INCLUDE:-}" ]; then
    MOUNTS+=(-v "$NVCC_INCLUDE:$NVCC_INCLUDE:ro")
    EXTRA+=(-e "BOTBOWL_CUDA_NVCC_INCLUDE=$NVCC_INCLUDE")
fi
if [ -n "${CUDA_INCLUDE:-}" ]; then
    IFS=: read -r -a dirs <<< "$CUDA_INCLUDE"
    for d in "${dirs[@]}"; do MOUNTS+=(-v "$d:$d:ro"); done
    EXTRA+=(-e "BOTBOWL_CUDA_INCLUDE=$CUDA_INCLUDE")
fi
[ "$TD" = "${TD#"$REPO"}" ] && MOUNTS+=(-v "$TD:$TD")
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME="$HOME" -w "$REPO" \
    "${MOUNTS[@]}" "${EXTRA[@]}" \
    -e PATH="$HOME/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e CARGO_TARGET_DIR="$TD" \
    -e BOARD_SIZE_W="${BOARD_SIZE_W:-20}" -e BOARD_SIZE_H="${BOARD_SIZE_H:-11}" \
    -e BOARD_PLAYERS="${BOARD_PLAYERS:-7}" \
    -e RUSTFLAGS="${RUSTFLAGS:--C target-cpu=neoverse-v2}" \
    "$IMG" cargo "$CMD" -j "${JOBS:-2}" --target aarch64-unknown-linux-gnu "$@"
