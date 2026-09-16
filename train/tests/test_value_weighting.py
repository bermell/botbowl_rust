"""Plan 036 W1/W4: the value-loss knobs must be inert until asked for.

The arms of an A/B are only comparable if the baseline arm is literally the
old code path, so the two things worth pinning are that the defaults change
nothing and that the weighted form stays on the plain MSE's scale.
"""

import torch

from bbnn.train import compute_losses


class _ConstModel(torch.nn.Module):
    """Emits a fixed value and uniform policy logits — the loss arithmetic is
    what is under test, not the tower."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, spatial, global_):
        n = spatial.shape[0]
        policy = torch.zeros(n, 1, spatial.shape[-2], spatial.shape[-1])
        return policy, torch.full((n, 1), self.value)


def _batch(values, weights):
    n = len(values)
    return {
        "spatial": torch.zeros(n, 3, 4, 6),
        "global": torch.zeros(n, 3),
        "value": torch.tensor(values, dtype=torch.float32).view(n, 1),
        "weight": torch.tensor(weights, dtype=torch.float32).view(n, 1),
        "actions": torch.zeros(n, 1, 4, dtype=torch.long),
        "policy": torch.ones(n, 1),
        "pad_mask": torch.ones(n, 1, dtype=torch.bool),
        "chosen": torch.zeros(n, dtype=torch.long),
    }


def test_unweighted_is_the_plain_mse():
    model = _ConstModel(0.0)
    batch = _batch([1.0, -1.0, 0.5], [0.5, 0.25, 0.25])
    _, vl, _ = compute_losses(model, batch, "cpu")
    assert torch.allclose(vl, torch.tensor((1.0 + 1.0 + 0.25) / 3))


def test_uniform_weights_reproduce_the_plain_mse():
    # Normalising by w.sum() rather than N is what keeps the weighted form on
    # the same scale, so --value-weight means one thing in both arms.
    model = _ConstModel(0.0)
    batch = _batch([1.0, -1.0, 0.5], [0.1, 0.1, 0.1])
    _, plain, _ = compute_losses(model, batch, "cpu")
    _, weighted, _ = compute_losses(model, batch, "cpu", per_drive_value_weight=True)
    assert torch.allclose(plain, weighted)


def test_weighting_reweights_by_drive_length():
    # Two drives: one row at weight 1 (a 1-position drive, error 1) and four
    # rows at weight 1/4 (a 4-position drive, error 2). Unweighted, the long
    # drive supplies 4/5 of the rows; weighted, the two drives are equal, so
    # the loss is the mean of their squared errors.
    model = _ConstModel(0.0)
    batch = _batch([1.0, 2.0, 2.0, 2.0, 2.0], [1.0, 0.25, 0.25, 0.25, 0.25])
    _, plain, _ = compute_losses(model, batch, "cpu")
    _, weighted, _ = compute_losses(model, batch, "cpu", per_drive_value_weight=True)
    assert torch.allclose(plain, torch.tensor((1.0 + 4 * 4.0) / 5))
    assert torch.allclose(weighted, torch.tensor((1.0 + 4.0) / 2))
    assert weighted < plain
