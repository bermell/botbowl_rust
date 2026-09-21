"""Recover a BBNet ``state_dict`` from an exported ONNX, so a champion whose
``.pt`` is gone can still be fed to `bbnn.migrate`.

    uv run python -m bbnn.from_onnx models/bb_best_14x7_gen23.onnx --out models/bb_best_14x7_gen23.pt

`export.export_onnx` runs with ``do_constant_folding=True``, which **fuses each
BatchNorm into the convolution before it**: the graph keeps one `Conv` per
conv+BN pair, carrying `W' = W·γ/√(σ²+ε)` and `b' = β − μ·γ/√(σ²+ε)` (plus the
conv's own bias through the same scale). The individual γ/β/μ/σ² are therefore
*not recoverable* — but an equivalent net is, because the fold is invertible
the other way round: keep `W'` in the conv, zero its bias, and make the
BatchNorm the affine map `y = x + b'` by setting

    γ = 1,  β = b',  μ = 0,  σ² = 1 − ε

so that `√(σ²+ε) = 1` exactly. In `eval()` mode — which is how tract, the
sidecar and `Migration.verify` all run a net — the result is bit-comparable to
the ONNX it came from. `onnx_roundtrip_is_exact` pins that against the ONNX
reference runtime.

**This is an inference-equivalent net, not the trainer's original weights.**
The BN scale now lives in the conv weights and the running statistics are
identity, so warm-starting training from it is not the same starting point the
original `.pt` would have been (BN re-estimates its statistics from the first
batch). Prefer the real `.pt` whenever it still exists; this is the fallback
for when it does not.

Nodes are matched to modules by their **name** — the TorchScript exporter
writes `/blocks.0/c1/Conv`, i.e. the module path — rather than by position in
the graph, so a wider or deeper tower needs no changes here. The name, not the
output name: the last conv's output is renamed to the graph output `policy`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .model import BBNet

# Which BatchNorm consumed each conv's output. `policy_head` has none — its
# bias survives the export as a real conv bias.
BN_OF: dict[str, str | None] = {"stem": "stem_bn", "value_conv": "value_bn", "policy_head": None}
BLOCK_BN = {"c1": "b1", "c2": "b2"}

EPS = nn.BatchNorm2d(1).eps

_NODE_NAME = re.compile(r"^/(.+)/(?:Conv|Gemm)$")


def _bn_of(module: str) -> str | None:
    """The BatchNorm module name paired with conv ``module``, or None."""
    if module in BN_OF:
        return BN_OF[module]
    head, _, leaf = module.rpartition(".")
    if head.startswith("blocks.") and leaf in BLOCK_BN:
        return f"{head}.{BLOCK_BN[leaf]}"
    raise ValueError(f"unknown conv module {module!r} — model.py grew a layer this loader does not know")


def state_dict_from_onnx(path: str | Path) -> dict[str, torch.Tensor]:
    """The `state_dict` of a net that, in eval mode, computes what ``path`` does."""
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(path)).graph
    init = {t.name: torch.from_numpy(numpy_helper.to_array(t).copy()) for t in graph.initializer}

    sd: dict[str, torch.Tensor] = {}
    for node in graph.node:
        if node.op_type not in ("Conv", "Gemm"):
            continue
        m = _NODE_NAME.match(node.name)
        if m is None:
            raise ValueError(f"cannot name the module behind {node.op_type} node {node.name!r}")
        module = m.group(1).replace("/", ".")  # ONNX writes the module path with slashes
        _, w, b = node.input
        if node.op_type == "Gemm":  # nn.Linear: weight/bias are never folded
            sd[f"{module}.weight"], sd[f"{module}.bias"] = init[w], init[b]
        else:
            weight, bias = init[w], init[b]
            sd[f"{module}.weight"] = weight
            bn = _bn_of(module)
            if bn is None:
                sd[f"{module}.bias"] = bias
                continue
            sd[f"{module}.bias"] = torch.zeros_like(bias)
            ch = int(weight.shape[0])
            sd[f"{bn}.weight"] = torch.ones(ch)
            sd[f"{bn}.bias"] = bias.clone()
            sd[f"{bn}.running_mean"] = torch.zeros(ch)
            sd[f"{bn}.running_var"] = torch.full((ch,), 1.0 - EPS)
            sd[f"{bn}.num_batches_tracked"] = torch.tensor(0)

    # Building the net the state_dict implies and loading it strictly is the
    # check that nothing is missing or spurious.
    BBNet(**shape_of_onnx(sd)).load_state_dict(sd)
    return sd


def shape_of_onnx(sd: dict[str, torch.Tensor]) -> dict:
    """Constructor kwargs implied by a recovered state_dict (any schema)."""
    embed = sd["global_fc.weight"].shape[0]
    return {
        **BBNet.shape_of(sd),
        "global_embed": int(embed),
        "spatial_ch": int(sd["stem.weight"].shape[1] - embed),
        "global_f": int(sd["global_fc.weight"].shape[1]),
        "policy_ch": int(sd["policy_head.weight"].shape[0]),
        "value_hidden": int(sd["value_fc1.weight"].shape[0]),
    }


def max_abs_diff_vs_onnx(path: str | Path, sd: dict[str, torch.Tensor], sizes=((5, 9), (9, 16)), seed: int = 0) -> float:
    """Largest output difference between the ONNX at ``path`` and the net
    ``sd`` builds, over random inputs at each ``(H, W)``.

    Uses `onnx.reference` (pure Python, no onnxruntime dependency), which is
    slow — hence the small default boards.
    """
    import numpy as np
    from onnx import load
    from onnx.reference import ReferenceEvaluator

    ref = ReferenceEvaluator(load(str(path)))
    shape = shape_of_onnx(sd)
    model = BBNet(**shape)
    model.load_state_dict(sd)
    model.eval()

    rng = np.random.default_rng(seed)
    worst = 0.0
    for (h, w) in sizes:
        spatial = rng.random((1, shape["spatial_ch"], h, w), dtype=np.float32)
        global_ = rng.random((1, shape["global_f"]), dtype=np.float32)
        p_ref, v_ref = ref.run(None, {"spatial": spatial, "global": global_})
        with torch.no_grad():
            p, v = model(torch.from_numpy(spatial), torch.from_numpy(global_))
        worst = max(worst, float(np.abs(p.numpy() - p_ref).max()), float(np.abs(v.numpy() - v_ref).max()))
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Recover a BBNet state_dict from an exported ONNX.")
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--out", type=Path, help="write the recovered bare state_dict here")
    ap.add_argument("--no-verify", action="store_true", help="skip the forward-equivalence check against the ONNX")
    ap.add_argument("--atol", type=float, default=1e-4)
    a = ap.parse_args(argv)

    sd = state_dict_from_onnx(a.onnx)
    shape = shape_of_onnx(sd)
    print(f"{a.onnx}: {shape}")
    if not a.no_verify:
        worst = max_abs_diff_vs_onnx(a.onnx, sd)
        print(f"onnx vs recovered: max |diff| {worst:.3e}")
        if worst > a.atol:
            print(f"recovery is not equivalent (> {a.atol})", file=sys.stderr)
            return 1
    if a.out:
        torch.save(sd, a.out)
        print(f"saved weights -> {a.out}")
    else:
        print("(dry run: pass --out to write)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
