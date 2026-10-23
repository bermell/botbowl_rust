"""TorchScript export of `BBNet.forward_masked` for the in-process backend (plan 066).

    python -m bbnn.export_ts models/az_v7/bbnet_x_gen44.pt            # -> models/az_v7/bbnet_x_gen44.ts
    python -m bbnn.export_ts runs/r/gen12.pt --out /tmp/gen12.ts --canvas 13x22

The in-process backend (`botbowl-nn/src/inproc.rs`, cargo feature `torch`) runs the net inside the
Rust process through libtorch — the same binaries the sidecar (`scripts/nn_server.py`) runs. It loads
this file, not the `.pt`: C++ cannot rebuild `BBNet` from a state_dict, and a TorchScript module is
the net *as the sidecar runs it*.

**What is saved is exactly the sidecar's trace, before optimisation.** `nn_server.py`'s
`maybe_trace_masked` traces `MaskedNet(net)` (the masked canvas forward) on canvas-shaped inputs of
batch 2 and then calls `torch.jit.optimize_for_inference`, which freezes the module, folds every
BatchNorm into its convolution and fuses conv + add + ReLU on CUDA. Those folds are computed on the
device and are part of the bits the sidecar serves, so the in-process backend repeats them itself
after loading (`torch::jit::optimize_for_inference`, the C++ twin of the Python call) rather than
reading pre-folded weights: the file stays device-independent (it loads on CUDA, ROCm or the CPU),
and the folding runs through the same libtorch code on the same device as the sidecar's. Plan 066
§4 test A is what pins that this reproduces the sidecar bit for bit.

The trace is checked to be batch-general (two batch sizes, two boards on the canvas) against the
eager `forward_masked`, as the sidecar checks its own; a trace that baked in a batch size would be
silently wrong at every other bucket. Shapes come from the weights (`BBNet.shape_of`), so every
head knob (`value_conv`, `value_hidden`, `value_pool`, `policy_conv`) exports.

**Reduced-precision and channels-last variants (plan 069 item 4)**, for the H100's tensor cores:

    python -m bbnn.export_ts NET.pt --dtype fp16                   # -> NET.fp16.ts
    python -m bbnn.export_ts NET.pt --dtype bf16 --channels-last   # -> NET.bf16-cl.ts
    python -m bbnn.export_ts NET.pt --channels-last                # -> NET.fp32-cl.ts

The in-process backend loads one when asked (`BLOOD_NN_PRECISION=fp16`, `BLOOD_NN_CHANNELS_LAST=1`;
`botbowl-nn/src/inproc.rs`); the plain `NET.ts` above is untouched and stays the default. A variant
is a different module, not a flag: every BatchNorm is folded into its convolution **in fp32 here**
(after the cast, `optimize_for_inference` would fold in half precision, and `var + eps` with eps
1e-5 is subnormal in fp16), then the weights are cast and `CastNet` takes the fp32 canvas inputs the
queue stages, casts them (and, for `-cl`, makes them NHWC), and returns fp32 outputs in NCHW. So the
shim, the staging, the H2D copy and the crop are the same code for every variant. Everything runs in
the reduced dtype, the value head's pooling included (cuDNN and ATen accumulate in fp32 inside a
kernel); exp081 measured fp16/bf16 under CPU autocast, which keeps reductions in fp32, so
`hpc/tune/precision_gate.py` re-measures these files on the real kernels. The file carries its tag as
the `bbnn_variant` extra file, which the shim reads to size the load canary's tolerance.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .model import GLOBAL_FEATURES, SPATIAL_CHANNELS, BBNet

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class MaskedNet(torch.nn.Module):
    """`BBNet.forward_masked` as a module's `forward` — the sidecar's `MaskedNet`, verbatim."""

    def __init__(self, net: BBNet):
        super().__init__()
        self.net = net

    def forward(self, spatial, global_feat, mask):
        return self.net.forward_masked(spatial, global_feat, mask)


class CastNet(torch.nn.Module):
    """A variant's module: fp32 canvas inputs in, cast (and NHWC for channels-last), the folded net
    in `dtype`, fp32 NCHW outputs back — the same interface as `MaskedNet`."""

    def __init__(self, net: BBNet, dtype: torch.dtype, channels_last: bool):
        super().__init__()
        self.net = net
        self.dtype = dtype
        self.channels_last = channels_last

    def forward(self, spatial, global_feat, mask):
        s = spatial.to(self.dtype)
        m = mask.to(self.dtype)
        if self.channels_last:
            s = s.contiguous(memory_format=torch.channels_last)
            m = m.contiguous(memory_format=torch.channels_last)
        p, v = self.net.forward_masked(s, global_feat.to(self.dtype), m)
        return p.float().contiguous(), v.float()


def fold_batchnorm(net: BBNet) -> BBNet:
    """A copy of `net` (eval mode) with every BatchNorm folded into the convolution before it, in
    fp32, the BatchNorm replaced by an identity. Same function up to float reassociation."""
    net = copy.deepcopy(net).eval()
    pairs = [(net, "stem", "stem_bn"), (net, "value_conv", "value_bn")]
    pairs += [(b, c, bn) for b in net.blocks for c, bn in (("c1", "b1"), ("c2", "b2"))]
    if net.policy_conv is not None:
        pairs.append((net, "policy_conv", "policy_bn"))
    for owner, conv, bn in pairs:
        setattr(owner, conv, fuse_conv_bn_eval(getattr(owner, conv), getattr(owner, bn)))
        setattr(owner, bn, torch.nn.Identity())
    leftover = [n for n, m in net.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if leftover:
        raise RuntimeError(f"BatchNorm left unfolded: {leftover} (a new layer? add its pair to fold_batchnorm)")
    return net


def variant_tag(dtype: str, channels_last: bool) -> str:
    """The tag `inproc.rs` looks for: `fp16`, `bf16-cl`, `fp32-cl` (`""` = the plain export)."""
    if dtype == "fp32" and not channels_last:
        return ""
    return dtype + ("-cl" if channels_last else "")


def canvas_inputs(b: int, h: int, w: int, canvas: tuple[int, int], device: str = "cpu", seed: int = 0):
    """`b` random `h x w` boards embedded top-left in `canvas`, with their mask."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.zeros(b, SPATIAL_CHANNELS, *canvas)
    s[:, :, :h, :w] = torch.randn(b, SPATIAL_CHANNELS, h, w, generator=gen)
    m = torch.zeros(b, 1, *canvas)
    m[:, :, :h, :w] = 1.0
    g = torch.randn(b, GLOBAL_FEATURES, generator=gen)
    return s.to(device), g.to(device), m.to(device)


def trace_masked(net: BBNet, canvas: tuple[int, int], device: str = "cpu") -> torch.jit.ScriptModule:
    """Trace `MaskedNet(net)` exactly as `nn_server.maybe_trace_masked` does (batch 2, full canvas,
    `check_trace=False`), unoptimised, and check it is batch-general against eager."""
    module = MaskedNet(net).eval().to(device)
    with torch.no_grad():
        traced = torch.jit.trace(module, canvas_inputs(2, *canvas, canvas, device), check_trace=False)
        for b, h, w in ((1, 9, 16), (5, canvas[0], canvas[1])):
            if h > canvas[0] or w > canvas[1]:
                continue
            s, g, m = canvas_inputs(b, h, w, canvas, device, seed=b)
            tp, tv = traced(s, g, m)
            ep, ev = module(s, g, m)
            dp = (tp - ep).abs().max().item()
            dv = (tv - ev).abs().max().item()
            if dp > 1e-5 or dv > 1e-5:
                raise RuntimeError(f"masked trace disagrees with eager at b={b} {h}x{w}: dp={dp:.2e} dv={dv:.2e}")
    return traced


def parse_hw(spec: str) -> tuple[int, int]:
    h, w = spec.lower().split("x")
    return int(h), int(w)


def trace_variant(net: BBNet, canvas: tuple[int, int], dtype: str, channels_last: bool, device: str):
    """Trace `CastNet(fold_batchnorm(net))` in `dtype` on `device`, check it is batch-general
    against its own eager run, and report how far it lands from the fp32 net (printed, not gated:
    `hpc/tune/precision_gate.py` is the gate). Returns the traced module."""
    folded = fold_batchnorm(net).to(device)
    ref = MaskedNet(net).eval().to(device)
    if channels_last:
        folded = folded.to(memory_format=torch.channels_last)
    module = CastNet(folded.to(DTYPES[dtype]), DTYPES[dtype], channels_last).eval()
    # Same ops on both sides, so only a trace that baked in a shape misses; looser than the fp32
    # check only because a reduced dtype's rounding can differ between cuDNN algorithms.
    tol = 1e-5 if dtype == "fp32" else 1e-2
    with torch.no_grad():
        traced = torch.jit.trace(module, canvas_inputs(2, *canvas, canvas, device), check_trace=False)
        worst_p = worst_v = 0.0
        for b, h, w in ((1, 9, 16), (5, canvas[0], canvas[1])):
            if h > canvas[0] or w > canvas[1]:
                continue
            s, g, m = canvas_inputs(b, h, w, canvas, device, seed=b)
            tp, tv = traced(s, g, m)
            ep, ev = module(s, g, m)
            rp, rv = ref(s, g, m)
            if not (torch.isfinite(tp).all() and torch.isfinite(tv).all()):
                raise RuntimeError(f"{dtype} trace gives non-finite outputs at b={b} {h}x{w}")
            dp = (tp - ep).abs().max().item()
            dv = (tv - ev).abs().max().item()
            if dp > tol or dv > tol:
                raise RuntimeError(f"variant trace disagrees with eager at b={b} {h}x{w}: dp={dp:.2e} dv={dv:.2e}")
            mask = m.bool().expand_as(tp)
            worst_p = max(worst_p, (tp - rp).abs()[mask].max().item())
            worst_v = max(worst_v, (tv - rv).abs().max().item())
    print(f"export_ts: {variant_tag(dtype, channels_last)} vs fp32 on random inputs: "
          f"policy max |d| {worst_p:.2e}, value max |d| {worst_v:.2e} ({device})")
    return traced


def export(
    pt: Path,
    out: Path,
    canvas: tuple[int, int],
    dtype: str = "fp32",
    channels_last: bool = False,
    device: str | None = None,
) -> Path:
    state = torch.load(pt, map_location="cpu", weights_only=True)
    net = BBNet.from_state_dict(state).eval()  # `.eval()`: BatchNorm on its running statistics
    tag = variant_tag(dtype, channels_last)
    extra = {}
    if not tag:
        # The plain export, as it always was (the default backend loads it): same code and weights
        # entries as before plan 069; only the debug pickles' source paths and the serialization id
        # vary between any two exports.
        traced = trace_masked(net, canvas, "cpu")
    else:
        # A reduced dtype traces where it runs: half-precision convolutions are a GPU thing.
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        traced = trace_variant(net, canvas, dtype, channels_last, device)
        extra = {"bbnn_variant": tag}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    if extra:
        traced.save(str(tmp), _extra_files=extra)
    else:
        traced.save(str(tmp))
    tmp.replace(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", type=Path, help="the trainer's state_dict (.pt)")
    ap.add_argument(
        "--out", type=Path, default=None, help="default: the .pt's path with suffix .ts (.<variant>.ts for a variant)"
    )
    ap.add_argument(
        "--canvas",
        default="13x22",
        help="HxW the trace's example inputs are staged on (the sidecar traces at its --canvas). The trace "
        "itself is shape-general; this only mirrors the sidecar's example.",
    )
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32", help="plan 069: a reduced-precision variant")
    ap.add_argument("--channels-last", action="store_true", help="plan 069: an NHWC variant")
    ap.add_argument("--device", default=None, help="where a variant is traced (default: cuda if available)")
    args = ap.parse_args()
    tag = variant_tag(args.dtype, args.channels_last)
    out = args.out or args.pt.with_suffix(f".{tag}.ts" if tag else ".ts")
    export(args.pt, out, parse_hw(args.canvas), args.dtype, args.channels_last, args.device)
    print(f"export_ts: {args.pt} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
