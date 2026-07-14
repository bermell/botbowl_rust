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


if __name__ == "__main__":
    build()
