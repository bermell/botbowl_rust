"""Plan 054 §7.1: the warm start is a restore candidate, and early and final steps are validated."""
import re

import numpy as np
import torch

from bbnn.model import GLOBAL_FEATURES, SPATIAL_CHANNELS
from bbnn.train import train

from test_multi_dims import write_corpus


def _corpus(tmp_path):
    for part, n in (("train", 64), ("val", 32)):
        d = tmp_path / part / "dims_6x4"
        write_corpus(d, n, 4, 6, c=SPATIAL_CHANNELS)
        g = np.random.default_rng(n).standard_normal((n, GLOBAL_FEATURES)).astype(np.float32)
        np.save(d / "global.npy", g)
    return tmp_path / "train", tmp_path / "val"


def _fit_init(tmp_path, data, val):
    init = tmp_path / "init.pt"
    train(data, val_dir=val, max_steps=40, eval_every=20, lr=1e-3, seed=0, width=8, blocks=1,
          out=init, device="cpu", select_on="combined")
    return init


def test_a_fine_tune_that_only_gets_worse_restores_the_warm_start(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    init = _fit_init(tmp_path, data, val)
    capsys.readouterr()
    # lr 5.0 wrecks the net within a step; the only good candidate is the warm start itself.
    model = train(data, val_dir=val, init=init, max_steps=6, eval_every=3, lr=5.0, seed=1,
                  device="cpu", select_on="combined", init_candidate=True)
    log = capsys.readouterr().out
    assert "step       0  (warm start, a restore candidate)" in log
    assert re.search(r"restored best-val weights: step 0 ", log)
    start = torch.load(init)
    assert all(torch.equal(start[k], v) for k, v in model.state_dict().items())


def test_without_the_flag_the_warm_start_is_never_restored(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    init = _fit_init(tmp_path, data, val)
    capsys.readouterr()
    train(data, val_dir=val, init=init, max_steps=6, eval_every=3, lr=5.0, seed=1,
          device="cpu", select_on="combined")
    assert not re.search(r"restored best-val weights: step 0 ", capsys.readouterr().out)


def test_eval_at_adds_early_checkpoints_and_the_final_step(tmp_path, capsys):
    data, val = _corpus(tmp_path)
    train(data, val_dir=val, max_steps=11, eval_every=5, eval_at=(2,), lr=1e-3, seed=0, width=8,
          blocks=1, device="cpu", select_on="policy")
    steps = [int(m) for m in re.findall(r"^step\s+(\d+)", capsys.readouterr().out, re.M)]
    assert steps == [2, 5, 10, 11]
