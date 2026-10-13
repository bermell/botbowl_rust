"""Plan 063 / exp072: training a fresh net of any shape — the arch knobs reach the net, a warmup +
cosine lr schedule, DataLoader workers, and one validation pass that yields the pooled and the
per-board numbers together. Every new option defaults to the loop's behaviour."""
import math
import re

import numpy as np
import pytest
import torch

from bbnn.data import make_loader, open_prepared
from bbnn.model import GLOBAL_FEATURES, SPATIAL_CHANNELS, BBNet
from bbnn.train import evaluate, evaluate_by_group, lr_factor, train

from test_multi_dims import write_corpus

HEADS = dict(value_conv=8, value_pool="meanmax", value_hidden=12, policy_conv=6)


def _corpus(tmp_path, dims=((6, 4), (8, 5))):
    """Train and val dirs, each a mixed corpus of ``dims`` (w, h) boards."""
    for part, n in (("train", 48), ("val", 20)):
        for (w, h) in dims:
            d = tmp_path / part / f"dims_{w}x{h}"
            write_corpus(d, n + w, h, w, c=SPATIAL_CHANNELS)
            g = np.random.default_rng(n + w).standard_normal((n + w, GLOBAL_FEATURES)).astype(np.float32)
            np.save(d / "global.npy", g)
            np.save(d / "value.npy", np.random.default_rng(w).uniform(-1, 1, n + w).astype(np.float32))
    return tmp_path / "train", tmp_path / "val"


def test_lr_factor_warms_up_then_decays_to_the_floor():
    total, warm = 1000, 100
    assert lr_factor(0, total, warm) == pytest.approx(1 / 100)
    assert lr_factor(49, total, warm) == pytest.approx(0.5)
    assert lr_factor(99, total, warm) == pytest.approx(1.0)
    assert lr_factor(100, total, warm) == pytest.approx(1.0)
    mid = 100 + (1000 - 100) // 2
    assert lr_factor(mid, total, warm) == pytest.approx(0.5, abs=1e-3)
    assert lr_factor(total, total, warm) == pytest.approx(0.0, abs=1e-9)
    assert lr_factor(total, total, warm, min_ratio=0.05) == pytest.approx(0.05)
    seq = [lr_factor(s, total, warm) for s in range(warm, total + 1)]
    assert all(a >= b for a, b in zip(seq, seq[1:])), "cosine must be monotone after warmup"
    assert lr_factor(5, total, 0) == pytest.approx(0.5 * (1 + math.cos(math.pi * 5 / 1000)))


def test_one_validation_pass_gives_pooled_and_per_board_numbers_exactly(tmp_path):
    _, val = _corpus(tmp_path)
    torch.manual_seed(0)
    net = BBNet(width=8, blocks=1)
    ds = open_prepared(val)
    pooled_loader = make_loader(ds, 8, shuffle=False)
    pooled, per = evaluate_by_group(net, pooled_loader, "cpu", True)
    assert pooled == evaluate(net, pooled_loader, "cpu", True)  # bit-identical, same additions
    for name, g in zip(ds.names, ds.groups):
        assert per[name] == evaluate(net, make_loader(g, 8, shuffle=False), "cpu", True), name


def test_a_fresh_net_takes_the_arch_knobs_and_the_schedule(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    out = tmp_path / "scratch.pt"
    train(data, val_dir=val, max_steps=12, eval_every=6, lr=1e-3, seed=0, width=8, blocks=2,
          device="cpu", select_on="combined", out=out, lr_schedule="cosine", warmup_steps=3,
          batch_size=8, **HEADS)
    log = capsys.readouterr().out
    assert "lr schedule: cosine, warmup 3 steps, peak 0.001 -> 0 over 12 steps" in log
    shape = BBNet.shape_of(torch.load(out))
    assert shape == {"width": 8, "blocks": 2, **HEADS}
    assert "val@dims_6x4" in log and "val@dims_8x5" in log


def test_constant_schedule_is_the_default_and_leaves_lr_alone(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    train(data, val_dir=val, max_steps=4, eval_every=2, lr=1e-3, seed=0, width=8, blocks=1,
          device="cpu", select_on="combined", batch_size=8)
    assert "lr schedule" not in capsys.readouterr().out


def test_loader_workers_serve_every_sample_once_per_epoch_in_single_board_batches(tmp_path):
    """Workers move the loading, not the data: an epoch is still every sample exactly once, in
    batches of one board. (Not the same *order* as ``workers=0`` at a seed: torch's multi-process
    iterator calls the sampler's ``__iter__`` twice, drawing the shuffle once more. ``workers=0``
    stays today's loader exactly.)"""
    data, _ = _corpus(tmp_path)
    ds = open_prepared(data, augment=False)
    seen = []
    for batch in make_loader(ds, 8, shuffle=True, workers=2):
        assert batch["spatial"].shape[0] <= 8
        seen.extend(batch["value"].flatten().tolist())
    want = [float(ds[i]["value"].item()) for i in range(len(ds))]
    assert sorted(seen) == sorted(want)


def test_training_runs_with_loader_workers(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    train(data, val_dir=val, max_steps=6, eval_every=3, lr=1e-3, seed=0, width=8, blocks=1, device="cpu",
          select_on="combined", batch_size=8, loader_workers=2)
    assert re.search(r"restored best-val weights: step [36] ", capsys.readouterr().out)


# sha256 over the weights of the two seeded runs below, as trained by the code before plan 063
# (6f0bb7a, torch 2.13.0, CPU): the new options at their defaults must not move a single bit —
# including the RNG draws the old per-board val passes made.
OLD_RUNS = "8d81abc643efb1c4f4fab0eb7b574095dc534f0cb7156d090bfedb2490c6788b"


@pytest.mark.skipif(not torch.__version__.startswith("2.13."), reason="digest pinned at torch 2.13")
def test_seeded_runs_train_the_same_weights_as_before_plan_063(tmp_path):
    import hashlib

    data, val = _corpus(tmp_path)
    torch.manual_seed(5)
    init = tmp_path / "init.pt"
    torch.save(BBNet(width=8, blocks=2).state_dict(), init)
    m1 = train(data, val_dir=val, max_steps=20, eval_every=5, eval_at=(2,), lr=1e-3, seed=0, width=8, blocks=2,
               device="cpu", select_on="combined", batch_size=8, per_drive_value_weight=True, value_weight=0.36)
    m2 = train(data, val_dir=val, init=init, epochs=2, eval_every=7, lr=2e-4, seed=1, device="cpu",
               select_on="combined", batch_size=8, freeze_bn=True, per_drive_value_weight=True)
    h = hashlib.sha256()
    for m in (m1, m2):
        for k, v in m.state_dict().items():
            h.update(k.encode())
            h.update(v.contiguous().numpy().tobytes())
    assert h.hexdigest() == OLD_RUNS


def test_a_warm_start_keeps_the_init_shape_over_the_flags(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    init = tmp_path / "init.pt"
    torch.manual_seed(0)
    torch.save(BBNet(width=8, blocks=2, **HEADS).state_dict(), init)
    model = train(data, val_dir=val, init=init, max_steps=2, eval_every=2, lr=1e-4, seed=0, device="cpu",
                  select_on="combined", batch_size=8)
    assert BBNet.shape_of(model.state_dict()) == {"width": 8, "blocks": 2, **HEADS}
    assert re.search(r"model: width 8, blocks 2", capsys.readouterr().out)
