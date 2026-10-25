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


# The schema the fixture nets are *drawn* at. Later schemas get them through `bbnn.migrate`, so a
# policy-head bump keeps every old weight (and the tests that play the fixture net — policy-only
# playouts in `mc_label` / `override_audit` — keep playing the same games): the v10 fixture is
# the v9 seed net with Secure the Ball's zero channel inserted.
DRAWN_SCHEMA = 9


def _drawn(**kwargs) -> BBNet:
    """A `BBNet` drawn from the current RNG at `DRAWN_SCHEMA`'s policy width, migrated to today's."""
    from .migrate import SCHEMAS

    return BBNet(policy_ch=SCHEMAS[DRAWN_SCHEMA]["A"], schema_version=DRAWN_SCHEMA, **kwargs)


def _migrated(old: BBNet) -> BBNet:
    from .migrate import migrate, shape_of

    # The migration's verifier builds nets (drawing init weights): keep the global RNG where the
    # draw left it, so the parity inputs drawn next are the ones the old schema drew.
    with torch.random.fork_rng():
        sd, _, _ = migrate(old.state_dict(), log=lambda *_: None)
        model = BBNet(**shape_of(sd))
        model.load_state_dict(sd)
    return model


def build(out_dir: Path | None = None):
    out_dir = out_dir or fixtures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1234)
    model = _migrated(_drawn(width=16, blocks=2, global_embed=8, value_hidden=16))
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
    model = _drawn(width=16, blocks=2, global_embed=8, value_conv=12, value_pool="meanmax", value_hidden=24, policy_conv=20)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    model = _migrated(model)
    model.eval()
    export_onnx(model, out_dir / "tiny_heads.onnx")
    dump_parity(model, out_dir, sizes=HEADS_SIZES)
    print(f"heads fixture written → {out_dir}")


def build_ts(out_dir: Path | None = None):
    """Plan 066: TorchScript exports of both committed fixture nets (`tiny.ts`, `heads/tiny_heads.ts`)
    for the in-process backend's tests. Built *from the committed ONNX* (`bbnn.from_onnx`: BatchNorm
    comes back folded, an eval-equivalent net), so they need no seed to line up with it — the tests'
    canary (the ONNX's tract answer, `< 1e-3`) checks the pairing."""
    from .export_ts import trace_masked
    from .from_onnx import shape_of_onnx, state_dict_from_onnx

    out_dir = out_dir or fixtures_dir()
    for onnx in (out_dir / "tiny.onnx", out_dir / "heads" / "tiny_heads.onnx"):
        sd = state_dict_from_onnx(onnx)
        net = BBNet(**shape_of_onnx(sd))
        net.load_state_dict(sd)
        net.eval()
        trace_masked(net, (13, 22), "cpu").save(str(onnx.with_suffix(".ts")))
        print(f"torchscript fixture written → {onnx.with_suffix('.ts')}")


if __name__ == "__main__":
    import sys

    # `--ts` (plan 066) writes only the TorchScript twins of the committed ONNX fixtures.
    if "--ts" in sys.argv:
        build_ts()
        raise SystemExit(0)
    # `--heads` rebuilds only the plan-063 heads fixture; plain runs rebuild both. The canary of
    # `scripts/nn_server.py` reads tiny's 9x16 inputs, so rebuilding tiny is never needed for heads.
    if "--heads" not in sys.argv:
        build()
    build_heads()
