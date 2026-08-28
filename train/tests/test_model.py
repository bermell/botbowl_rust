import torch

from bbnn.model import (
    GLOBAL_FEATURES,
    POLICY_CHANNELS,
    SPATIAL_CHANNELS,
    BBNet,
    masked_policy_logits,
)


def test_forward_shapes_at_two_board_sizes():
    model = BBNet()
    model.eval()
    for (h, w) in [(17, 28), (9, 16)]:
        spatial = torch.zeros(3, SPATIAL_CHANNELS, h, w)
        global_ = torch.zeros(3, GLOBAL_FEATURES)
        with torch.no_grad():
            policy, value = model(spatial, global_)
        assert policy.shape == (3, POLICY_CHANNELS, h, w)
        assert value.shape == (3, 1)
        assert torch.all(value >= -1) and torch.all(value <= 1)


def test_masked_policy_logits_gather():
    # One sample, A=30, H=2, W=2. Craft a policy map with known values.
    policy = torch.full((1, POLICY_CHANNELS, 2, 2), -5.0)
    policy[0, 10, 1, 0] = 7.0     # positional channel 10 at (y=1, x=0)
    policy[0, 20, 0, 1] = 3.0     # simple channel 20 max cell
    # Actions: [channel, y, x, is_simple]
    actions = torch.tensor(
        [[[10, 1, 0, 0], [20, 0, 0, 1], [0, 0, 0, 0]]], dtype=torch.long
    )  # (1, 3, 4); 3rd is padding
    pad_mask = torch.tensor([[True, True, False]])
    logits = masked_policy_logits(policy, actions, pad_mask)
    assert logits.shape == (1, 3)
    assert abs(logits[0, 0].item() - 7.0) < 1e-6      # positional cell
    assert abs(logits[0, 1].item() - 3.0) < 1e-6      # simple = channel max
    assert logits[0, 2].item() < -1e8                  # padded


def test_overfit_tiny_random_batch_drives_policy_loss_down():
    # Pure plumbing check: a fixed random batch should be memorisable.
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = BBNet(width=16, blocks=2, global_embed=8, value_hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    spatial = torch.randn(4, SPATIAL_CHANNELS, 9, 16)
    global_ = torch.randn(4, GLOBAL_FEATURES)
    actions = torch.zeros(4, 3, 4, dtype=torch.long)
    actions[..., 0] = torch.tensor([0, 5, 10])  # distinct channels
    actions[..., 3] = 0
    pad_mask = torch.ones(4, 3, dtype=torch.bool)
    target = torch.zeros(4, 3)
    target[:, 0] = 1.0  # always the first action
    value_t = torch.zeros(4, 1)

    first = None
    for _ in range(60):
        opt.zero_grad()
        policy_out, value_out = model(spatial, global_)
        logits = masked_policy_logits(policy_out, actions, pad_mask)
        logsm = F.log_softmax(logits, dim=1)
        pl = -(target * logsm).sum(dim=1).mean()
        vl = F.mse_loss(value_out, value_t)
        (pl + vl).backward()
        opt.step()
        if first is None:
            first = pl.item()
    assert pl.item() < first * 0.5, f"policy loss did not drop: {first} -> {pl.item()}"


def test_resolve_device_cpu_and_auto_never_raise():
    """`auto` must always yield a usable device, and `cpu` must stay cpu.

    The training host's GPU is sm_61 while some torch wheels only ship sm_75+
    kernels, so `auto` has to survive a CUDA build that advertises a device it
    cannot actually launch on.
    """
    from bbnn.train import resolve_device

    assert resolve_device("cpu").type == "cpu"

    dev = resolve_device("auto")
    assert dev.type in ("cpu", "cuda")
    # Whatever it picked must genuinely run a kernel.
    x = torch.zeros(4, 4, device=dev)
    assert (x @ x).sum().item() == 0.0


def test_resolve_device_explicit_cuda_is_never_a_silent_cpu_fallback():
    """An explicit --device cuda must either work or fail loudly — never CPU."""
    from bbnn.train import resolve_device

    try:
        dev = resolve_device("cuda")
    except RuntimeError as e:
        assert "unusable" in str(e)
    else:
        assert dev.type == "cuda"
