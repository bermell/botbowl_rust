# x86_64 container with Debian bookworm's aarch64 cross toolchain (gcc 12, glibc 2.36: the same
# compiler and C library as the arm64 python:3.12-bookworm base of botbowl.def). cross_check.sh uses
# it to `cargo check` / `cargo build` for aarch64 on an x86 box, C/C++/asm included (tract-linalg's
# arm64 kernels, mimalloc, the libtorch shim). Runs natively, no qemu. The Rust toolchain is the
# host's (~/.rustup, ~/.cargo mounted), so the container holds only the C side.
FROM debian:bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++-aarch64-linux-gnu libc6-dev-arm64-cross gcc g++ libc6-dev git ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*
ENV CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++ \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
