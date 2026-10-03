"""Plan 050 step 1: the WDL value head.

What has to hold for the A/B to mean anything: the scalar head is untouched by default, the WDL
head's exported value is the scalar the search already reads, and the two heads are scored on
the same number.
"""

import torch

from bbnn.model import BBNet
from bbnn.train import compute_losses, wdl_target

from test_value_weighting import _batch


def test_targets_are_one_hot_on_outcomes_and_keep_the_mean_on_blends():
    t = wdl_target(torch.tensor([[1.0], [0.0], [-1.0], [0.5], [-0.25]]))
    assert torch.allclose(t[:3], torch.eye(3))
    assert torch.allclose(t.sum(dim=1), torch.ones(5))
    assert torch.allclose(t[:, 0] - t[:, 2], torch.tensor([1.0, 0.0, -1.0, 0.5, -0.25]))


def test_wdl_exports_the_implied_scalar():
    torch.manual_seed(0)
    net = BBNet(spatial_ch=3, global_f=3, policy_ch=1, width=8, blocks=1, value_head="wdl").eval()
    s, g = torch.randn(4, 3, 5, 6), torch.randn(4, 3)
    _, v = net(s, g)
    _, logits = net(s, g, wdl=True)
    p = torch.softmax(logits, dim=1)
    assert v.shape == (4, 1)
    assert torch.allclose(v, p[:, 0:1] - p[:, 2:3])
    mask = torch.ones(4, 1, 5, 6)
    _, vm = net.forward_masked(s, g, mask)
    assert torch.allclose(v, vm, atol=1e-5)


def test_shape_of_recognises_the_head():
    wdl = BBNet(spatial_ch=3, global_f=3, policy_ch=1, width=8, blocks=1, value_head="wdl")
    scalar = BBNet(spatial_ch=3, global_f=3, policy_ch=1, width=8, blocks=1)
    assert BBNet.shape_of(wdl.state_dict())["value_head"] == "wdl"
    assert "value_head" not in BBNet.shape_of(scalar.state_dict())


def test_both_heads_report_the_scalar_mse():
    torch.manual_seed(0)
    batch = _batch([1.0, 0.0, -1.0, 1.0], [1.0, 0.5, 0.25, 1.0])
    batch["spatial"] = torch.randn(4, 3, 4, 6)
    for head in ("scalar", "wdl"):
        net = BBNet(spatial_ch=3, global_f=3, policy_ch=1, width=8, blocks=1, value_head=head).eval()
        _, loss, _, mse = compute_losses(net, batch, "cpu", per_drive_value_weight=True)
        with torch.no_grad():
            _, v = net(batch["spatial"], batch["global"])
        w = batch["weight"]
        expect = (w * (v - batch["value"]) ** 2).sum() / w.sum()
        assert torch.allclose(mse, expect, atol=1e-6)
        if head == "scalar":
            assert torch.allclose(loss, mse)
        else:
            assert loss > 0 and not torch.allclose(loss, mse)
