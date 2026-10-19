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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GLOBAL_FEATURES, SPATIAL_CHANNELS, BBNet


class MaskedNet(torch.nn.Module):
    """`BBNet.forward_masked` as a module's `forward` — the sidecar's `MaskedNet`, verbatim."""

    def __init__(self, net: BBNet):
        super().__init__()
        self.net = net

    def forward(self, spatial, global_feat, mask):
        return self.net.forward_masked(spatial, global_feat, mask)


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


def export(pt: Path, out: Path, canvas: tuple[int, int]) -> Path:
    state = torch.load(pt, map_location="cpu", weights_only=True)
    net = BBNet.from_state_dict(state).eval()  # `.eval()`: BatchNorm on its running statistics
    traced = trace_masked(net, canvas, "cpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    traced.save(str(tmp))
    tmp.replace(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", type=Path, help="the trainer's state_dict (.pt)")
    ap.add_argument("--out", type=Path, default=None, help="default: the .pt's path with suffix .ts")
    ap.add_argument(
        "--canvas",
        default="13x22",
        help="HxW the trace's example inputs are staged on (the sidecar traces at its --canvas). The trace "
        "itself is shape-general; this only mirrors the sidecar's example.",
    )
    args = ap.parse_args()
    out = args.out or args.pt.with_suffix(".ts")
    export(args.pt, out, parse_hw(args.canvas))
    print(f"export_ts: {args.pt} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
