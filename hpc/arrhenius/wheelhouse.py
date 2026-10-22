"""Plan 069: download the trainer's locked wheels for another platform (aarch64 by default).

Reads `train/uv.lock` directly, so the wheelhouse holds exactly the files the lock pins (same URLs,
same sha256), not whatever an index resolves today: the package set is `uv export --frozen` (dev
group included, the project itself left out) with its markers evaluated for the target, and for
each package the best wheel for the target's tags. Writes `requirements.txt` (`name==version
--hash=sha256:…`, one hash per line, the wheel's) for an offline, hash-checked, no-resolve install:

    pip install --no-index --find-links WHEELS --no-deps --require-hashes -r WHEELS/requirements.txt

Also fetches the CUDA header wheels the libtorch shim needs (`nvidia-cuda-nvcc-cu12`,
`nvidia-cuda-cccl-cu12`, matching the lock's CUDA runtime major.minor) for
`scripts/fetch_cuda_headers.sh` to unpack offline; they are not part of requirements.txt.

    uv run --no-project --with packaging python hpc/arrhenius/wheelhouse.py OUT_DIR \
        [--python 3.12] [--machine aarch64] [--glibc 2.36]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

REPO = Path(__file__).resolve().parents[2]


def target_tags(py: tuple[int, int], machine: str, glibc: tuple[int, int]):
    plats = [f"manylinux_{glibc[0]}_{m}_{machine}" for m in range(glibc[1], 16, -1)]
    plats += [f"manylinux2014_{machine}", f"linux_{machine}"]
    tags = list(cpython_tags(py, [f"cp{py[0]}{py[1]}", "abi3", "none"], plats))
    tags += list(compatible_tags(py, f"cp{py[0]}{py[1]}", plats))
    return {t: i for i, t in enumerate(tags)}  # lower = preferred


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: str) -> None:
    if dest.is_file() and sha256_of(dest) == sha256:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    # curl, not urllib: uv's standalone Python hung in urlopen on this box.
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), url], check=True)
    if sha256_of(tmp) != sha256:
        tmp.unlink()
        sys.exit(f"sha256 mismatch for {url}")
    tmp.rename(dest)


def get_json(url: str):
    return json.loads(subprocess.run(["curl", "-sSfL", url], check=True, capture_output=True).stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--python", default="3.12")
    ap.add_argument("--machine", default="aarch64")
    ap.add_argument("--glibc", default="2.36", help="the target image's glibc (bookworm: 2.36)")
    a = ap.parse_args()
    py = tuple(int(x) for x in a.python.split("."))
    glibc = tuple(int(x) for x in a.glibc.split("."))
    tags = target_tags(py, a.machine, glibc)
    env = {
        "python_version": a.python,
        "python_full_version": a.python + ".0",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": a.machine,
        "os_name": "posix",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }

    exported = subprocess.run(
        ["uv", "export", "--frozen", "--no-emit-project", "--no-hashes", "--no-header", "--no-annotate"],
        cwd=REPO / "train", check=True, capture_output=True, text=True,
    ).stdout
    want: dict[str, str] = {}
    for line in exported.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req, _, marker = line.partition(";")
        name, _, ver = req.strip().partition("==")
        if marker.strip() and not Marker(marker.strip()).evaluate(env):
            continue
        want[canonicalize_name(name)] = ver.strip()

    lock = tomllib.loads((REPO / "train/uv.lock").read_text())
    by_key = {(canonicalize_name(p["name"]), p["version"]): p for p in lock["package"]}
    a.out.mkdir(parents=True, exist_ok=True)
    reqs, total, runtime_ver = [], 0, None
    for name, ver in sorted(want.items()):
        pkg = by_key.get((name, ver))
        if pkg is None:
            sys.exit(f"{name}=={ver} from uv export is not in uv.lock")
        best = None
        for w in pkg.get("wheels", []):
            fname = w["url"].rsplit("/", 1)[1].replace("%2B", "+")
            _, _, _, wtags = parse_wheel_filename(fname)
            rank = min((tags[t] for t in wtags if t in tags), default=None)
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, fname, w)
        if best is None:
            sys.exit(f"{name}=={ver}: no wheel in uv.lock for cp{py[0]}{py[1]} {a.machine} glibc {a.glibc}")
        _, fname, w = best
        sha = w["hash"].split(":", 1)[1]
        print(f"{name}=={ver}  {fname}", flush=True)
        fetch(w["url"], a.out / fname, sha)
        total += (a.out / fname).stat().st_size
        reqs.append(f"{name}=={ver} --hash=sha256:{sha}")
        if name == "nvidia-cuda-runtime-cu12":
            runtime_ver = ver
    (a.out / "requirements.txt").write_text(
        f"# {len(reqs)} wheels for cp{py[0]}{py[1]} linux {a.machine} (glibc {a.glibc}), from train/uv.lock\n"
        + "\n".join(reqs) + "\n"
    )

    # Header-only wheels for the shim (crt/host_config.h, nv/target): same CUDA major.minor.
    if runtime_ver is None:
        sys.exit("no nvidia-cuda-runtime-cu12 in the lock — cannot pick CUDA header wheels")
    cu = ".".join(runtime_ver.split(".")[:2])
    hdr = a.out / "cuda-headers"
    hdr.mkdir(exist_ok=True)
    for pkg in ("nvidia-cuda-nvcc-cu12", "nvidia-cuda-cccl-cu12"):
        d = get_json(f"https://pypi.org/pypi/{pkg}/json")
        best = None
        for v, files in d["releases"].items():
            if not (v == cu or v.startswith(cu + ".")):
                continue
            for f in files:
                n = f["filename"]
                if n.endswith(".whl") and "manylinux" in n and a.machine in n:
                    key = tuple(int(x) for x in v.split(".") if x.isdigit())
                    if best is None or key > best[0]:
                        best = (key, n, f)
        if best is None:
            sys.exit(f"no {pkg} wheel for CUDA {cu} on {a.machine}")
        _, n, f = best
        print(f"{pkg} (headers)  {n}", flush=True)
        fetch(f["url"], hdr / n, f["digests"]["sha256"])
        total += (hdr / n).stat().st_size
    print(f"wheelhouse: {len(reqs)} wheels + 2 header wheels, {total / 2**30:.2f} GiB in {a.out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
