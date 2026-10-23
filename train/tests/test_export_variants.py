"""Plan 069 item 4: `bbnn.export_ts`'s reduced-precision / channels-last variants.

A variant must be the same function as the plain export up to rounding: BatchNorm folded in fp32,
fp32 canvas in, fp32 NCHW out, and its tag in the file for the shim. The numerics of fp16/bf16 on
the real kernels are `hpc/tune/precision_gate.py`'s job; here only what a CPU can pin.
"""

import zipfile

import pytest
import torch

from bbnn.export_ts import MaskedNet, canvas_inputs, export, fold_batchnorm, variant_tag
from bbnn.model import BBNet

CANVAS = (13, 22)


def _live_bn(net, seed=0):
    """Non-trivial BatchNorm statistics and affines, so a wrong fold shows."""
    g = torch.Generator().manual_seed(seed)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.rand(n, generator=g) - 0.5)
            m.running_var.copy_(torch.rand(n, generator=g) * 1.5 + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.rand(n, generator=g) - 0.5)
    return net.eval()


@pytest.mark.parametrize("heads", [{}, dict(value_conv=16, value_pool="meanmax", value_hidden=32, policy_conv=24)])
def test_fold_batchnorm_is_the_same_function(heads):
    torch.manual_seed(0)
    net = _live_bn(BBNet(width=24, blocks=2, **heads))
    folded = fold_batchnorm(net)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in folded.modules())
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules()), "the original is untouched"
    s, g, m = canvas_inputs(3, 9, 16, CANVAS, seed=1)
    with torch.no_grad():
        p0, v0 = MaskedNet(net)(s, g, m)
        p1, v1 = MaskedNet(folded)(s, g, m)
    assert (p0 - p1).abs().max() < 1e-4
    assert (v0 - v1).abs().max() < 1e-5


def test_variant_tags():
    assert variant_tag("fp32", False) == ""
    assert variant_tag("fp32", True) == "fp32-cl"
    assert variant_tag("fp16", False) == "fp16"
    assert variant_tag("bf16", True) == "bf16-cl"


@pytest.mark.parametrize("dtype,cl,tol", [("fp32", True, 1e-4), ("bf16", False, 0.25), ("bf16", True, 0.25)])
def test_variant_file_round_trips(tmp_path, dtype, cl, tol):
    torch.manual_seed(0)
    net = _live_bn(BBNet(width=24, blocks=2))
    pt = tmp_path / "net.pt"
    torch.save(net.state_dict(), pt)
    out = tmp_path / f"net.{variant_tag(dtype, cl)}.ts"
    try:
        export(pt, out, CANVAS, dtype, cl, device="cpu")
    except RuntimeError as e:  # a CPU build without the dtype's convolution kernels
        pytest.skip(f"{dtype} on this CPU: {e}")
    with zipfile.ZipFile(out) as z:
        tags = [n for n in z.namelist() if n.endswith("extra/bbnn_variant")]
        assert tags and z.read(tags[0]).decode() == variant_tag(dtype, cl)
    loaded = torch.jit.load(str(out))
    s, g, m = canvas_inputs(4, 11, 20, CANVAS, seed=3)
    with torch.no_grad():
        p, v = loaded(s, g, m)
        rp, rv = MaskedNet(net)(s, g, m)
    assert p.dtype == torch.float32 and v.dtype == torch.float32
    assert p.is_contiguous(), "NCHW out, whatever the variant computes in"
    mask = m.bool().expand_as(p)
    assert (p - rp).abs()[mask].max() < tol
    assert (v - rv).abs().max() < tol


def test_plain_export_has_no_variant_tag(tmp_path):
    net = _live_bn(BBNet(width=16, blocks=1))
    pt = tmp_path / "net.pt"
    torch.save(net.state_dict(), pt)
    out = export(pt, tmp_path / "net.ts", CANVAS)
    with zipfile.ZipFile(out) as z:
        assert not any(n.endswith("extra/bbnn_variant") for n in z.namelist())
