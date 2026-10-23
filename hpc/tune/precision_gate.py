#!/usr/bin/env python3
"""Reduced-precision numerics on the GPU's real kernels (plan 069 item 4): exp081, but on the files the
in-process backend loads instead of CPU autocast.

    python hpc/tune/precision_gate.py NET.onnx PREPARED_DIR [--variants fp32,default,tf32,fp16,bf16,fp16-cl]
        [--canvas 13x22] [--out numerics.md]

Each variant is the TorchScript file `inproc.rs` would load (`NET.ts`, `NET.fp16.ts`, `NET.bf16-cl.ts`, …,
from `python -m bbnn.export_ts NET.pt --dtype … [--channels-last]`), put through
`torch.jit.optimize_for_inference` on the GPU as the shim does, with the TF32 flags its precision sets
(`fp32`: both off — the reference; `default`: torch's defaults, i.e. TF32 convolutions on an H100;
`tf32`: both on). Every held-out sample (a `prepare` output dir, e.g. gen49's shard 4 with
`--policy-target cq --tau 100`, as exp081) is staged top-left onto the canvas with its mask, exactly
the queue's staging, and the policy is cropped back. Against `fp32`, per variant:

* value |Δv| mean and max; value MSE against the sample's own label, fp32 → variant;
* policy KL(fp32 ‖ variant) over the legal actions (the masked gather `prepare` stores);
* top-1 move changes, and how many of them are at a near-tie (fp32 top-2 log-prob margin < 0.1).

It is not bit parity (`scripts/parity066.sh` is, for the default) and not the last word: a variant that
passes here still goes through the value benches (`precision_gate.sh`, real queue and shim) and paired
drives. Thresholds (`--max-top1`, `--max-mse-delta`) are exp081-derived defaults: fp16 changed 0.05% of
top-1 moves on CPU and the value bench resolves ~0.003 in RMS; the search itself disagrees with itself
at 4-6% of decisions (exp079).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bbnn.data import make_loader, open_prepared
from bbnn.model import masked_policy_logits

FLAGS = {  # cudnn.allow_tf32, cuda.matmul.allow_tf32
    "fp32": (False, False),
    "default": (True, False),
    "tf32": (True, True),
}


def ts_path(net: Path, variant: str) -> Path:
    base = net.with_suffix(".ts")
    dtype = variant.split("-")[0]
    cl = variant.endswith("-cl")
    if dtype in FLAGS and not cl:
        return base
    tag = ("fp32" if dtype in FLAGS else dtype) + ("-cl" if cl else "")
    return base.with_name(f"{base.stem}.{tag}.ts")


def set_flags(variant: str):
    cudnn, matmul = FLAGS.get(variant.split("-")[0], (True, False))
    torch.backends.cudnn.allow_tf32 = cudnn
    torch.backends.cuda.matmul.allow_tf32 = matmul


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", type=Path, help="NET.onnx (its .ts variants beside it)")
    ap.add_argument("prepared", type=Path)
    ap.add_argument("--variants", default="fp32,default,tf32,fp16,bf16,fp16-cl")
    ap.add_argument("--canvas", default="13x22")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-top1", type=float, default=0.01, help="pass: top-1 changes below this fraction")
    ap.add_argument("--max-mse-delta", type=float, default=0.001, help="pass: |value MSE change| below this")
    ap.add_argument("--out", type=Path, default=None, help="also write the table here (markdown)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("precision_gate: no CUDA device (the point is the GPU's kernels)", file=sys.stderr)
        return 2
    dev = torch.device("cuda")
    ch, cw = (int(x) for x in args.canvas.split("x"))
    variants = [v for v in args.variants.split(",") if v]
    if "fp32" not in variants:
        variants.insert(0, "fp32")
    mods = {}
    for v in variants:
        p = ts_path(args.net, v)
        if not p.exists():
            print(f"precision_gate: {p} missing — python -m bbnn.export_ts {args.net.with_suffix('.pt')} "
                  f"--dtype {v.split('-')[0] if v.split('-')[0] not in FLAGS else 'fp32'}"
                  f"{' --channels-last' if v.endswith('-cl') else ''}", file=sys.stderr)
            return 2
        set_flags(v)
        m = torch.jit.load(str(p), map_location=dev).eval()
        mods[v] = torch.jit.optimize_for_inference(m)
    ds = open_prepared(str(args.prepared))
    loader = make_loader(ds, args.batch, shuffle=False)
    acc = {v: dict(n=0, dv=0.0, dv_max=0.0, kl=0.0, flips=0, flips_close=0, se=0.0, secs=0.0) for v in variants}
    close_total, se_ref = 0, 0.0
    with torch.no_grad():
        for batch in loader:
            s = batch["spatial"].to(dev)
            n, c, h, w = s.shape
            sc = torch.zeros(n, c, ch, cw, device=dev)
            sc[:, :, :h, :w] = s
            m = torch.zeros(n, 1, ch, cw, device=dev)
            m[:, :, :h, :w] = 1.0
            g = batch["global"].to(dev)
            actions, pad = batch["actions"].to(dev), batch["pad_mask"].to(dev)
            z = batch["value"].to(dev).reshape(n)
            out = {}
            for v in variants:
                set_flags(v)
                torch.cuda.synchronize()
                t = time.perf_counter()
                p, val = mods[v](sc, g, m)
                torch.cuda.synchronize()
                acc[v]["secs"] += time.perf_counter() - t
                lp = F.log_softmax(masked_policy_logits(p[:, :, :h, :w].float(), actions, pad), 1)
                out[v] = (lp, val.float().reshape(n))
            lp0, v0 = out["fp32"]
            p0, a0 = lp0.exp(), lp0.argmax(1)
            top2 = lp0.topk(2, 1).values
            close = (top2[:, 0] - top2[:, 1]) < 0.1
            close_total += int(close.sum())
            se_ref += float(((v0 - z) ** 2).sum())
            for v in variants:
                lp, vv = out[v]
                a = acc[v]
                a["n"] += n
                d = (vv - v0).abs()
                a["dv"] += float(d.sum())
                a["dv_max"] = max(a["dv_max"], float(d.max()))
                kl = (p0 * (lp0 - lp)).where(pad > 0, torch.zeros((), device=dev))
                a["kl"] += float(kl.sum())
                same = lp.argmax(1) == a0
                a["flips"] += int((~same).sum())
                a["flips_close"] += int((~same & close).sum())
                a["se"] += float(((vv - z) ** 2).sum())
    n = acc["fp32"]["n"]
    gpu = torch.cuda.get_device_name(0)
    lines = [f"Numerics of {args.net.name}'s in-process variants on {gpu} (torch {torch.__version__}), "
             f"{n} held-out samples of `{args.prepared}`, canvas {args.canvas}; reference fp32 (TF32 off). "
             f"Value MSE vs label at fp32: {se_ref / n:.5f}; {close_total} samples at a near-tie.", "",
             "| variant | value \\|Δv\\| mean / max | value MSE vs label | policy KL(fp32‖·) | top-1 changes "
             "| … at a near-tie | ms per 1k positions (eager, not graphs) | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    ok = True
    for v in variants:
        a = acc[v]
        top1 = a["flips"] / n
        dmse = a["se"] / n - se_ref / n
        verdict = "ref" if v == "fp32" else ("PASS" if top1 < args.max_top1 and abs(dmse) < args.max_mse_delta
                                              and math.isfinite(a["dv_max"]) else "FAIL")
        ok &= verdict != "FAIL"
        lines.append(f"| {v} | {a['dv'] / n:.5f} / {a['dv_max']:.4f} | {se_ref / n:.5f} → {a['se'] / n:.5f} | "
                     f"{a['kl'] / n:.2e} | {100 * top1:.2f}% ({a['flips']}) | {a['flips_close']} | "
                     f"{1e6 * a['secs'] / n:.1f} | {verdict} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        args.out.write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
