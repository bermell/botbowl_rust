"""ONNX export (opset 17, dynamic batch/H/W) + parity-tensor dumping.

The parity dump runs the model at two board sizes and saves the exact
input/output tensors so the Rust `tract` evaluator can be checked against
PyTorch to < 1e-4 (proves dynamic-axes concretization works end to end).
"""

from pathlib import Path

import numpy as np
import torch

from .model import GLOBAL_FEATURES, SPATIAL_CHANNELS

# Reference board sizes for parity: default 28x17 (engine) and the 14x7
# tier's 16x9. Stored (H, W) since tensors are NCHW.
PARITY_SIZES = [(17, 28), (9, 16)]


def export_onnx(model, path, opset: int = 17):
    """Export `model` to ONNX with dynamic batch, height and width."""
    model.eval()
    dummy_spatial = torch.zeros(1, SPATIAL_CHANNELS, 17, 28)
    dummy_global = torch.zeros(1, GLOBAL_FEATURES)
    torch.onnx.export(
        model,
        (dummy_spatial, dummy_global),
        str(path),
        input_names=["spatial", "global"],
        output_names=["policy", "value"],
        dynamic_axes={
            "spatial": {0: "N", 2: "H", 3: "W"},
            "global": {0: "N"},
            "policy": {0: "N", 2: "H", 3: "W"},
            "value": {0: "N"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter — no onnxscript dep, tract-friendly graph
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))


def dump_parity(model, out_dir, sizes=PARITY_SIZES, seed: int = 0):
    """Save `(spatial, global) -> (policy, value)` reference tensors at each
    `(H, W)` for the Rust parity test. Deterministic under `seed`.
    """
    model.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for (h, w) in sizes:
        spatial = torch.randn(1, SPATIAL_CHANNELS, h, w, generator=g)
        global_ = torch.randn(1, GLOBAL_FEATURES, generator=g)
        with torch.no_grad():
            policy, value = model(spatial, global_)
        tag = f"{h}x{w}"
        np.save(out / f"parity_{tag}_spatial.npy", spatial.numpy())
        np.save(out / f"parity_{tag}_global.npy", global_.numpy())
        np.save(out / f"parity_{tag}_policy.npy", policy.numpy())
        np.save(out / f"parity_{tag}_value.npy", value.numpy())
