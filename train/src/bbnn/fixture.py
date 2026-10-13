"""Build the committed tract parity fixture: a tiny seeded, untrained model
exported to `botbowl-nn/tests/fixtures/tiny.onnx`, plus reference
input/output tensors at two board sizes.

Untrained is fine — parity only checks that tract reproduces PyTorch's
arithmetic, not that the net is any good. BatchNorm's default running
stats (mean 0, var 1) make eval-mode output deterministic.
"""

from pathlib import Path

import torch

from .export import dump_parity, export_onnx
from .model import BBNet


def fixtures_dir() -> Path:
    # train/src/bbnn/fixture.py → repo root is parents[3].
    return Path(__file__).resolve().parents[3] / "botbowl-nn" / "tests" / "fixtures"


def build(out_dir: Path | None = None):
    out_dir = out_dir or fixtures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    model = BBNet(width=16, blocks=2, global_embed=8, value_hidden=16)
    model.eval()
    export_onnx(model, out_dir / "tiny.onnx")
    dump_parity(model, out_dir)
    print(f"fixture written → {out_dir}")


# The heads fixture's boards: small, so the committed tensors stay a few hundred KB.
HEADS_SIZES = [(9, 16), (7, 12)]


def build_heads(out_dir: Path | None = None):
    """Plan 063: a tiny net with every non-default head (3x3 policy conv, mean+max value pooling,
    wider value conv and hidden layer) in ``fixtures/heads/``, for `tests/parity.rs`'s second case:
    tract must run the ops those heads add (ReduceMax, Concat, a folded BN after the policy conv).
    BatchNorm gets random statistics, so a wrong fold cannot hide behind the identity."""
    out_dir = out_dir or fixtures_dir() / "heads"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(4321)
    model = BBNet(width=16, blocks=2, global_embed=8, value_conv=12, value_pool="meanmax", value_hidden=24, policy_conv=20)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    model.eval()
    export_onnx(model, out_dir / "tiny_heads.onnx")
    dump_parity(model, out_dir, sizes=HEADS_SIZES)
    print(f"heads fixture written → {out_dir}")


if __name__ == "__main__":
    import sys

    # `--heads` rebuilds only the plan-063 heads fixture; plain runs rebuild both. The canary of
    # `scripts/nn_server.py` reads tiny's 9x16 inputs, so rebuilding tiny is never needed for heads.
    if "--heads" not in sys.argv:
        build()
    build_heads()
