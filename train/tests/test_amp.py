"""Plan 071 R3: ``--amp bf16`` autocasts the training forward to bfloat16. Off (the default) must be
today's fp32 training bit for bit — `test_scratch_train.py`'s pinned digest covers the seeded runs;
here the default is also checked against an explicit ``amp="off"``, and bf16 is pinned to be
seeded-deterministic, to really run in bf16, to keep fp32 weights, and to land near the fp32 run.
CPU autocast, so the test runs anywhere."""
import pytest
import torch

from bbnn.data import make_loader, open_prepared
from bbnn.model import BBNet
from bbnn.train import autocast_context, compute_losses, seed_everything, train

from test_scratch_train import _corpus


def _run(data, val, **kw):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return train(data, val_dir=val, max_steps=10, eval_every=5, lr=1e-3, seed=0, width=8, blocks=2,
                     device="cpu", select_on="combined", batch_size=8, **kw)
    finally:
        torch.set_num_threads(threads)


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_amp_off_is_the_default_bit_for_bit(tmp_path):
    data, val = _corpus(tmp_path)
    assert _same(_run(data, val), _run(data, val, amp="off"))


def test_bf16_is_seeded_deterministic_close_to_fp32_and_weights_stay_fp32(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    fp32 = _run(data, val)
    b1 = _run(data, val, amp="bf16")
    b2 = _run(data, val, amp="bf16")
    assert "amp: bf16 autocast on the training forward" in capsys.readouterr().out
    assert _same(b1, b2), "a seeded bf16 run must repeat exactly"
    assert not _same(fp32, b1), "bf16 changed nothing: autocast did not engage"
    for k, v in b1.state_dict().items():
        assert v.dtype == fp32.state_dict()[k].dtype, k  # master weights and the export stay fp32
    # Ten steps at lr 1e-3 from the same init: the two runs differ by rounding only.
    p32 = torch.cat([p.detach().flatten() for p in fp32.parameters()])
    p16 = torch.cat([p.detach().flatten() for p in b1.parameters()])
    # The shared init: `train` seeds, then builds the net before anything else draws.
    seed_everything(0)
    p0 = torch.cat([p.detach().flatten() for p in BBNet(width=8, blocks=2).parameters()])
    moved = (p32 - p0).abs().mean().item()
    d = (p32 - p16).abs()
    assert 0 < moved < 2 * 10 * 1e-3, moved  # p0 really is the init: ten Adam steps from it
    # Adam steps are ~lr each, so 10 steps bound any coordinate whose gradient sign rounding flipped,
    # and on average bf16 must follow fp32's trajectory far more than it departs from it.
    assert d.max().item() < 2 * 10 * 1e-3
    assert d.mean().item() < 0.5 * moved, (d.mean().item(), moved)


def test_bf16_runs_the_forward_in_bf16_and_returns_fp32_losses(tmp_path):
    data, _ = _corpus(tmp_path)
    batch = next(iter(make_loader(open_prepared(data, augment=False), 8, shuffle=False)))
    torch.manual_seed(0)
    net = BBNet(width=8, blocks=1)
    seen = []
    net.stem.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    out32 = compute_losses(net, batch, "cpu")
    out16 = compute_losses(net, batch, "cpu", amp="bf16")
    assert seen == [torch.float32, torch.bfloat16]
    assert all(t.dtype == torch.float32 for t in out16)
    assert all(torch.isfinite(t).all() for t in out16)
    assert out32[0].item() == pytest.approx(out16[0].item(), rel=2e-2)
    assert out32[1].item() == pytest.approx(out16[1].item(), rel=2e-2, abs=1e-3)


def test_unknown_amp_mode_is_refused(tmp_path):
    with pytest.raises(ValueError):
        autocast_context("cpu", "fp16")
    data, val = _corpus(tmp_path)
    with pytest.raises(ValueError):
        _run(data, val, amp="fp8")
