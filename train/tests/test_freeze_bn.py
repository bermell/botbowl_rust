"""Review finding 2: `--freeze-bn` trains under inference-time normalisation."""
import torch

from bbnn.model import BBNet
from bbnn.train import compute_losses, set_train_mode

from test_value_weighting import _batch


def _net():
    torch.manual_seed(0)
    net = BBNet(spatial_ch=3, global_f=3, policy_ch=1, width=8, blocks=1)
    # Non-trivial running statistics, as a warm start has.
    net.train()
    with torch.no_grad():
        for _ in range(5):
            net(torch.randn(8, 3, 4, 6) * 3 + 1, torch.randn(8, 3))
    return net


def test_frozen_bn_keeps_running_stats_and_matches_eval():
    net = _net()
    before = {k: v.clone() for k, v in net.state_dict().items() if "running" in k}
    batch = _batch([1.0, 0.0, -1.0, 1.0], [1.0] * 4)
    batch["spatial"] = torch.randn(4, 3, 4, 6)
    set_train_mode(net, freeze_bn=True)
    _, v_train = net(batch["spatial"], batch["global"])
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    pl, vl, _, _ = compute_losses(net, batch, "cpu")
    (pl + vl).backward()
    opt.step()
    after = {k: v for k, v in net.state_dict().items() if "running" in k}
    assert all(torch.equal(before[k], after[k]) for k in before)
    # Training forward uses the running statistics: the same as eval mode.
    net2 = _net().eval()
    with torch.no_grad():
        _, v_eval = net2(batch["spatial"], batch["global"])
    assert torch.allclose(v_train, v_eval, atol=1e-6)


def test_unfrozen_training_updates_running_stats():
    net = _net()
    before = net.blocks[0].b1.running_mean.clone()
    set_train_mode(net, freeze_bn=False)
    with torch.no_grad():
        net(torch.randn(4, 3, 4, 6) * 5, torch.randn(4, 3))
    assert not torch.equal(before, net.blocks[0].b1.running_mean)
