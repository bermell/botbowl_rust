#!/usr/bin/env python3
"""Emit a randomly-initialised BBNet (.pt + .onnx) — the gen-0 seed for a
from-scratch AlphaZero run.

`train_loop.sh`'s gen-0 bootstrap builds its first champion by training on a
*heuristic* corpus: the scripted bot is the teacher, and everything the loop
learns afterwards is downstream of that teacher's taste. This script is the
alternative AlphaZero actually specifies — start from random weights and let
search be the only source of signal.

    train/.venv/bin/python scripts/make_random_net.py \
        --out models/az/bbnet_14x7_gen00 --seed 0

Writes `<out>.pt` (a bare state_dict, which is what nn_server.py loads) and
`<out>.onnx` (what the Rust `tract` evaluator loads). The two must agree, so
they are exported from the same in-memory module.

The net is left in eval() mode with untrained BatchNorm running stats
(mean 0, var 1) — that is deliberate. Nothing has been observed yet, so the
identity normalisation is the honest prior, and the value head's tanh over a
random tower sits near zero: MCTS sees near-uniform priors and near-neutral
leaf values, which is exactly the cold start AlphaZero begins from.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "src"))

from bbnn.export import export_onnx  # noqa: E402
from bbnn.model import BBNet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path prefix; writes <out>.pt and <out>.onnx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model = BBNet(width=a.width, blocks=a.blocks)
    model.eval()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    export_onnx(model, out.with_suffix(".onnx"))

    params = sum(p.numel() for p in model.parameters())
    print(
        f"random-init net: width {a.width} blocks {a.blocks}, {params} params, seed {a.seed}\n"
        f"  {out.with_suffix('.pt')}\n  {out.with_suffix('.onnx')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
