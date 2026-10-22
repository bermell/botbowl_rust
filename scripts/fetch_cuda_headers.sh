#!/usr/bin/env bash
# Plan 066: fetch the CUDA headers the in-process backend's C++ shim needs, without a CUDA toolkit.
#
# libtorch's CUDA headers include <cuda_runtime.h> and <cuda_fp16.h>, which need `crt/host_config.h`
# (from `nvidia-cuda-nvcc-cu12`) and `nv/target` (libcu++, from `nvidia-cuda-cccl-cu12`). The torch
# wheel only depends on `nvidia-cuda-runtime-cu12`, which ships cuda_runtime*.h without either. This
# downloads those two wheels (matching the venv's CUDA major.minor) from PyPI and unpacks only their
# `include/` into ~/.cache/botbowl/cuda_nvcc/<version>/include, where botbowl-nn/build.rs finds
# them. Nothing is installed into train/.venv. Headers only: nothing here is linked or run.
#
#   scripts/fetch_cuda_headers.sh            # version from train/.venv's nvidia-cuda-runtime-cu12
#   scripts/fetch_cuda_headers.sh 12.6       # explicit CUDA major.minor
#
# Plan 069 (offline, other machines, images):
#   CUDA_WHEEL_DIR=DIR   take the two wheels from DIR instead of PyPI, no network
#                        (hpc/arrhenius/wheelhouse.py puts them in <wheels>/cuda-headers)
#   CUDA_HEADERS_DEST=D  unpack into D/include instead of ~/.cache/botbowl/cuda_nvcc/<version>, and
#                        point BOTBOWL_CUDA_NVCC_INCLUDE at D/include (build.rs looks there first)
#   CUDA_ARCH=aarch64    the wheels' platform (default: this machine's)
#
# A box with a CUDA toolkit (an HPC module, /usr/local/cuda) does not need this: point
# BOTBOWL_CUDA_INCLUDE at the toolkit's include/ instead.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/train/.venv/bin/python}"
CU="${1:-}"
if [ -z "$CU" ]; then
    CU=$("$PY" -c "import importlib.metadata as m; v=m.version('nvidia-cuda-runtime-cu12'); print('.'.join(v.split('.')[:2]))")
fi
case "${CUDA_ARCH:-$(uname -m)}" in
    x86_64) TAG=x86_64 ;;
    aarch64) TAG=aarch64 ;;
    *) echo "unsupported arch ${CUDA_ARCH:-$(uname -m)}" >&2; exit 1 ;;
esac
DEST="${CUDA_HEADERS_DEST:-$HOME/.cache/botbowl/cuda_nvcc/$CU}"
if [ -f "$DEST/include/crt/host_config.h" ] && [ -f "$DEST/include/nv/target" ]; then
    echo "already there: $DEST/include"
    exit 0
fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST/include"
for PKG in cuda_nvcc cuda_cccl; do
    if [ -n "${CUDA_WHEEL_DIR:-}" ]; then
        # e.g. nvidia_cuda_nvcc_cu12-12.6.85-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
        WHL=$(find "$CUDA_WHEEL_DIR" -maxdepth 1 -name "nvidia_${PKG}_cu12-$CU.*$TAG*.whl" | sort -V | tail -1)
        [ -n "$WHL" ] || { echo "no nvidia-${PKG//_/-}-cu12 $CU $TAG wheel in $CUDA_WHEEL_DIR" >&2; exit 1; }
        echo "unpacking $(basename "$WHL")"
        cp "$WHL" "$TMP/$PKG.whl"
    else
        read -r VER URL < <(curl -sSL "https://pypi.org/pypi/nvidia-${PKG//_/-}-cu12/json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
want, tag = '$CU', '$TAG'
best = None
for v, files in d['releases'].items():
    if not (v == want or v.startswith(want + '.')):
        continue
    for f in files:
        n = f['filename']
        if n.endswith('.whl') and 'manylinux' in n and tag in n:
            key = tuple(int(x) for x in v.split('.') if x.isdigit())
            if best is None or key > best[0]:
                best = (key, v, f['url'])
if best is None:
    sys.exit('no nvidia-${PKG//_/-}-cu12 wheel for CUDA ' + want + ' on ' + tag)
print(best[1], best[2])
")
        echo "fetching nvidia-${PKG//_/-}-cu12 $VER ($TAG)"
        curl -sSL -o "$TMP/$PKG.whl" "$URL"
    fi
    python3 - "$TMP/$PKG.whl" "$DEST/include" "nvidia/$PKG/include/" <<'EOF'
import sys, zipfile, pathlib
whl, dest, marker = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.startswith(marker) and not n.endswith("/"):
            out = dest / n[len(marker):]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(n))
EOF
done
echo "CUDA $CU headers in $DEST/include"
