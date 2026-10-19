#!/usr/bin/env python3
"""Plan 066 bit-parity harness, Python side: batch plans, the sidecar's own runner path, comparison.

The question is whether the in-process libtorch backend (`botbowl-nn/src/inproc.rs`) returns the
*same bits* as the Python sidecar (`scripts/nn_server.py`) for the same batch: same rows, same
bucket, same canvas. Both sides read the same encoded corpus states (`states.bin`, written by
`cargo run -p botbowl-nn --example inproc_parity -- dump`) and run the same batch plan; this script
runs the sidecar side through `nn_server`'s own classes (`Registry`, `warm`, `RunnerPool`,
`GraphRunner`, `Request`) exactly as `nn_server.main` sets them up, and compares outputs bit for bit.

    PY=train/.venv/bin/python
    $PY scripts/parity066.py plan    --states S.bin --out plan.bin [--seed 0]
    $PY scripts/parity066.py sidecar --states S.bin --plan plan.bin --out side.bin --model NET.onnx \\
                                     --canvas 13x22 --mode det|prod
    $PY scripts/parity066.py compare A.bin B.bin --states S.bin --plan plan.bin [--json out.json]
    $PY scripts/parity066.py graph   --model NET.onnx --canvas 13x22 --out graph.txt   # optimised IR

`--mode det` is cuDNN `deterministic=True, benchmark=False`; `--mode prod` is what the loop runs
(`benchmark=True`, set by `nn_server.main` before anything loads). `scripts/parity066.sh` drives the
whole gate (plan 066 §4 A and B).
"""

from __future__ import annotations

import argparse
import gc
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


# --------------------------------------------------------------------------- states / plan files


class States:
    """`states.bin`: header, per-channel scales, then `u16 h | u16 w | u8 raw[C*h*w] | f32 g[F]`."""

    def __init__(self, path: str):
        self.buf = np.memmap(path, dtype=np.uint8, mode="r")
        assert bytes(self.buf[:4]) == b"BBST", "not a states.bin"
        _ver, self.c, self.f, self.n = struct.unpack_from("<IIII", self.buf, 4)
        self.scales = np.frombuffer(self.buf, dtype="<f4", count=self.c, offset=20).astype(np.float32)
        self.offsets = np.empty(self.n, dtype=np.int64)
        self.hw = np.empty((self.n, 2), dtype=np.int64)
        o = 20 + 4 * self.c
        for i in range(self.n):
            h, w = struct.unpack_from("<HH", self.buf, o)
            self.offsets[i] = o
            self.hw[i] = (h, w)
            o += 4 + self.c * h * w + 4 * self.f

    def get(self, i: int):
        """`(spatial f32 (C, h, w), global f32 (F,), h, w)`: `raw / scale` in float32, as Rust's
        `encode` computes it (both are one correctly rounded IEEE division)."""
        o = int(self.offsets[i])
        h, w = (int(x) for x in self.hw[i])
        raw = np.asarray(self.buf[o + 4 : o + 4 + self.c * h * w]).reshape(self.c, h, w)
        spatial = raw.astype(np.float32) / self.scales[:, None, None]
        go = o + 4 + self.c * h * w
        g = np.frombuffer(self.buf, dtype="<f4", count=self.f, offset=go).astype(np.float32)
        return spatial.astype(np.float32), g, h, w


def write_plan(path: str, batches: list[tuple[list[int], bool]]) -> None:
    with open(path, "wb") as f:
        f.write(b"BBPL" + struct.pack("<I", len(batches)))
        for idx, want in batches:
            f.write(struct.pack("<II", len(idx), int(want)))
            f.write(np.asarray(idx, dtype="<u4").tobytes())


def read_plan(path: str) -> list[tuple[list[int], bool]]:
    data = Path(path).read_bytes()
    assert data[:4] == b"BBPL"
    (nb,) = struct.unpack_from("<I", data, 4)
    o = 8
    out = []
    for _ in range(nb):
        n, want = struct.unpack_from("<II", data, o)
        o += 8
        idx = np.frombuffer(data, dtype="<u4", count=n, offset=o).astype(np.int64).tolist()
        o += 4 * n
        out.append((idx, bool(want)))
    return out


def cmd_plan(args) -> int:
    st = States(args.states)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(st.n).tolist()
    batches = []
    k = 0
    pos = 0
    while pos < len(order):
        bi = k % len(BUCKETS)
        bucket = BUCKETS[bi]
        cycle = k // len(BUCKETS)
        # Even cycles fill the bucket; odd ones send a partial batch that pads up to it.
        lo = BUCKETS[bi - 1] + 1 if bi > 0 else 1
        n = bucket if cycle % 2 == 0 else int(rng.integers(lo, bucket + 1))
        want = (cycle // 2) % 2 == 0  # policy and value-only alternate every two cycles
        idx = order[pos : pos + n]
        pos += n
        batches.append((idx, want))
        k += 1
    write_plan(args.out, batches)
    by_bucket = defaultdict(lambda: [0, 0])
    for idx, want in batches:
        b = next(x for x in BUCKETS if x >= len(idx))
        by_bucket[b][int(want)] += len(idx)
    print(f"plan: {len(batches)} batches, {st.n} rows -> {args.out}")
    for b in BUCKETS:
        print(f"  bucket {b:2d}: {by_bucket[b][1]:6d} rows with policy, {by_bucket[b][0]:6d} value-only")
    return 0


# --------------------------------------------------------------------------- sidecar side


def setup_sidecar(model_path: str, canvas: tuple[int, int], mode: str):
    """`nn_server.main`'s setup, step for step: cuDNN flags, registry, warm, prewarm."""
    sys.path.insert(0, str(REPO / "scripts"))
    import torch

    import nn_server as ns

    if mode == "prod":
        torch.backends.cudnn.benchmark = True
    elif mode == "det":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        raise SystemExit(f"--mode {mode}: want det or prod")
    registry = ns.Registry("cuda", 4, "auto", canvas)
    ns.WEIGHTS_DIRS.append(Path(model_path).resolve().parent)
    model = registry.get(model_path)
    ns.warm(registry, model, [(ns.CANARY_H, ns.CANARY_W)], 64)
    pool = ns.RunnerPool("cuda", 64, enabled=True)
    pool.prewarm(model, [(ns.CANARY_H, ns.CANARY_W)], canvas)
    gc.collect()
    gc.freeze()
    return ns, registry, model, pool


def cmd_sidecar(args) -> int:
    canvas = tuple(int(x) for x in args.canvas.split("x"))
    ns, _registry, model, pool = setup_sidecar(args.model, canvas, args.mode)
    st = States(args.states)
    plan = read_plan(args.plan)
    a = ns.POLICY_CHANNELS
    with open(args.out, "wb") as f:
        for idx, want in plan:
            reqs = []
            for i in idx:
                s, g, h, w = st.get(i)
                reqs.append(ns.Request(conn=None, model=model, h=h, w=w, want_policy=want, spatial=s, global_=g, canvas=canvas))
            runner = pool.get(model, *canvas, len(reqs), masked=True)
            launched = runner.launch(reqs, want, 0)
            values, policies, _fwd = runner.wait(launched)
            f.write(np.ascontiguousarray(values[: len(reqs)], dtype="<f4").tobytes())
            if want:
                for k, r in enumerate(reqs):
                    # `Server.finish`'s crop, verbatim.
                    f.write(np.ascontiguousarray(policies[k].reshape(a, *canvas)[:, : r.h, : r.w], dtype="<f4").tobytes())
    print(f"sidecar ({args.mode}): {len(plan)} batches -> {args.out}")
    return 0


def cmd_graph(args) -> int:
    canvas = tuple(int(x) for x in args.canvas.split("x"))
    sys.path.insert(0, str(REPO / "scripts"))
    import torch

    import nn_server as ns

    torch.backends.cudnn.benchmark = True
    registry = ns.Registry("cuda", 4, "auto", canvas)
    model = registry.get(args.model)
    Path(args.out).write_text(str(model.masked.graph))
    print(f"graph -> {args.out}")
    return 0


# --------------------------------------------------------------------------- compare


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in representable float32 steps (sign-magnitude mapped to a monotone integer line)."""
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


def cmd_compare(args) -> int:
    st = States(args.states)
    plan = read_plan(args.plan)
    fa = np.memmap(args.a, dtype="<f4", mode="r")
    fb = np.memmap(args.b, dtype="<f4", mode="r")
    assert fa.shape == fb.shape, f"sizes differ: {fa.shape} vs {fb.shape}"
    a_ch = 32
    o = 0
    tot = defaultdict(int)
    by = defaultdict(lambda: defaultdict(int))  # (path, bucket) -> counters
    by_board = defaultdict(lambda: defaultdict(int))
    worst = {"value_abs": 0.0, "value_ulp": 0, "policy_abs": 0.0, "policy_ulp": 0}
    for idx, want in plan:
        n = len(idx)
        bucket = next(x for x in BUCKETS if x >= n)
        key = ("policy" if want else "value", bucket)
        va, vb = np.asarray(fa[o : o + n]), np.asarray(fb[o : o + n])
        o += n
        vdiff = va.view(np.uint32) != vb.view(np.uint32)
        tot["rows"] += n
        tot["value_diff_rows"] += int(vdiff.sum())
        by[key]["rows"] += n
        by[key]["value_diff_rows"] += int(vdiff.sum())
        if vdiff.any():
            worst["value_abs"] = max(worst["value_abs"], float(np.abs(va - vb).max()))
            worst["value_ulp"] = max(worst["value_ulp"], int(ulp_distance(va, vb).max()))
        for k, i in enumerate(idx):
            h, w = (int(x) for x in st.hw[i])
            bk = f"{w}x{h}"
            by_board[bk]["rows"] += 1
            by_board[bk]["value_diff_rows"] += int(vdiff[k])
        if want:
            for i in idx:
                h, w = (int(x) for x in st.hw[i])
                m = a_ch * h * w
                pa, pb = np.asarray(fa[o : o + m]), np.asarray(fb[o : o + m])
                o += m
                d = pa.view(np.uint32) != pb.view(np.uint32)
                nd = int(d.sum())
                tot["policy_rows"] += 1
                tot["policy_elems"] += m
                tot["policy_diff_elems"] += nd
                tot["policy_diff_rows"] += int(nd > 0)
                by[key]["policy_rows"] += 1
                by[key]["policy_diff_rows"] += int(nd > 0)
                by[key]["policy_diff_elems"] += nd
                by_board[f"{w}x{h}"]["policy_diff_rows"] += int(nd > 0)
                if nd:
                    worst["policy_abs"] = max(worst["policy_abs"], float(np.abs(pa - pb).max()))
                    worst["policy_ulp"] = max(worst["policy_ulp"], int(ulp_distance(pa, pb).max()))
    assert o == fa.shape[0], f"plan covers {o} floats, files hold {fa.shape[0]}"
    # Differing *bits*, the headline number: popcount of XOR over every output float.
    bits = 0
    step = 1 << 20
    for s in range(0, fa.shape[0], step):
        xor = np.asarray(fa[s : s + step]).view(np.uint32) ^ np.asarray(fb[s : s + step]).view(np.uint32)
        if xor.any():
            bits += int(np.unpackbits(xor.view(np.uint8)).sum())
    tot["differing_bits"] = bits
    tot["floats"] = int(fa.shape[0])
    print(f"compare {args.a} vs {args.b}")
    print(
        f"  rows {tot['rows']}: value differs in {tot['value_diff_rows']}; policy rows {tot['policy_rows']} "
        f"({tot['policy_elems']} logits): differs in {tot['policy_diff_rows']} rows / {tot['policy_diff_elems']} logits"
    )
    print(f"  differing bits over all {tot['floats']} output floats: {bits}")
    print(f"  worst: {worst}")
    print("  by path/bucket (rows, value-diff rows, policy-diff rows):")
    for key in sorted(by):
        c = by[key]
        print(f"    {key[0]:6s} b={key[1]:2d}: {c['rows']:6d} {c['value_diff_rows']:6d} {c['policy_diff_rows']:6d}")
    print("  boards (rows, value-diff rows, policy-diff rows): " + ", ".join(
        f"{b} {c['rows']}/{c['value_diff_rows']}/{c['policy_diff_rows']}" for b, c in sorted(by_board.items())))
    if args.json:
        Path(args.json).write_text(json.dumps({
            "a": args.a, "b": args.b, "totals": dict(tot), "worst": worst,
            "by_path_bucket": {f"{k[0]}:{k[1]}": dict(v) for k, v in by.items()},
            "by_board": {k: dict(v) for k, v in by_board.items()},
        }, indent=1))
    return 0


# Per-run fields of a trajectory: when it was written, and nothing else. Everything the search and
# the net decide — every sample, every visit count, every value, the recombination counters — must
# match.
VOLATILE = {"created_unix_secs"}


def strip(x, drop: set[str]):
    if isinstance(x, dict):
        return {k: strip(v, drop) for k, v in x.items() if k not in drop}
    if isinstance(x, list):
        return [strip(v, drop) for v in x]
    return x


def first_diff(a, b, path="") -> str | None:
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: present on one side only"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:80]!r} vs {str(b)[:80]!r}"


def cmd_corpus(args) -> int:
    """Gate C: two JSONL corpora (or two directories of them) line for line, ignoring only
    `VOLATILE` fields (+ `--ignore`). Exit 0 iff identical."""
    drop = VOLATILE | set(args.ignore or [])
    pa, pb = Path(args.a), Path(args.b)
    pairs = [(pa, pb)] if pa.is_file() else [(f, pb / f.name) for f in sorted(pa.glob("*.jsonl"))]
    lines = diffs = 0
    for fa, fb in pairs:
        la = fa.read_text().splitlines()
        lb = fb.read_text().splitlines() if fb.exists() else []
        if len(la) != len(lb):
            print(f"  {fa.name}: {len(la)} vs {len(lb)} lines")
            diffs += 1
        for i, (x, y) in enumerate(zip(la, lb)):
            lines += 1
            if x == y:
                continue
            d = first_diff(strip(json.loads(x), drop), strip(json.loads(y), drop))
            if d:
                diffs += 1
                if diffs <= args.show:
                    print(f"  {fa.name} line {i}: {d}")
    print(f"corpus {args.a} vs {args.b}: {lines} lines compared, {diffs} differ (ignoring {sorted(drop)})")
    return 0 if diffs == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--states", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("sidecar")
    p.add_argument("--states", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--canvas", default="13x22")
    p.add_argument("--mode", default="prod", choices=("det", "prod"))
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--states", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--json", default=None)
    p = sub.add_parser("graph")
    p.add_argument("--model", required=True)
    p.add_argument("--canvas", default="13x22")
    p.add_argument("--out", required=True)
    p = sub.add_parser("corpus")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--ignore", nargs="*", help="more keys to ignore wherever they appear")
    p.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
    return {"plan": cmd_plan, "sidecar": cmd_sidecar, "compare": cmd_compare, "graph": cmd_graph, "corpus": cmd_corpus}[
        args.cmd
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
