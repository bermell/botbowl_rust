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

Plan 071 R1: any shape the trainer builds, e.g. the cluster run's 96x8 + bigger heads (exp072 arm B):

    train/.venv/bin/python scripts/make_random_net.py --out models/hpc/bbnet_gen00 --seed 0 \
        --width 96 --blocks 8 --value-conv 64 --value-pool meanmax --value-hidden 128 --policy-conv 96

`BBNet.shape_of` reads the shape back from the .pt, so every loader (the sidecar, `bbnn.export_ts`
for the in-process backend's `NET.ts`, a warm start) serves it, and `train_loop.sh`'s NO_WARM_FROM
scratch training builds the same shape (SCRATCH_ARCH_ARGS empty).

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
    # Plan 071 R1: plan 063's head knobs, the same names, defaults and meaning as `bbnn.train`'s.
    # The defaults are today's heads, so an unchanged command builds the same net bit for bit.
    ap.add_argument("--value-conv", type=int, default=32, help="channels of the value head's 1x1 conv")
    ap.add_argument("--value-hidden", type=int, default=64, help="width of the value head's hidden layer")
    ap.add_argument("--value-pool", choices=("mean", "meanmax"), default="mean",
                    help="value head pooling: global mean, or mean and max concatenated")
    ap.add_argument("--policy-conv", type=int, default=0,
                    help="0 = the policy is one 1x1 conv of the trunk; k = a 3x3 conv+BN+ReLU of k channels first")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model = BBNet(width=a.width, blocks=a.blocks, value_conv=a.value_conv, value_hidden=a.value_hidden,
                  value_pool=a.value_pool, policy_conv=a.policy_conv)
    model.eval()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out.with_suffix(".pt"))
    export_onnx(model, out.with_suffix(".onnx"))

    params = sum(p.numel() for p in model.parameters())
    heads = {k: v for k, v in BBNet.shape_of(model.state_dict()).items() if k not in ("width", "blocks")}
    print(
        f"random-init net: width {a.width} blocks {a.blocks}" + (f", heads {heads}" if heads else "")
        + f", {params} params, seed {a.seed}\n"
        f"  {out.with_suffix('.pt')}\n  {out.with_suffix('.onnx')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
