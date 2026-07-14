import torch

from bbnn.data import flip_y
from bbnn.model import POLICY_CHANNELS, masked_policy_logits


def random_actions(n, k, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    actions = torch.zeros(n, k, 4, dtype=torch.long)
    actions[..., 0] = torch.randint(0, POLICY_CHANNELS, (n, k), generator=g)
    actions[..., 1] = torch.randint(0, h, (n, k), generator=g)
    actions[..., 2] = torch.randint(0, w, (n, k), generator=g)
    actions[..., 3] = torch.randint(0, 2, (n, k), generator=g)
    return actions


def test_flip_y_is_consistent_with_the_gather():
    # The augmentation contract: gathering logits from a y-flipped map with
    # y-flipped action cells must equal the un-flipped gather — for
    # positional cells (moved) and simple cells (channel max, invariant).
    torch.manual_seed(0)
    n, k, h, w = 3, 7, 5, 6
    policy = torch.randn(n, POLICY_CHANNELS, h, w)
    actions = random_actions(n, k, h, w, seed=1)
    pad_mask = torch.ones(n, k, dtype=torch.bool)
    pad_mask[:, -1] = False

    flipped_policy, flipped_actions = flip_y(policy, actions)
    original = masked_policy_logits(policy, actions, pad_mask)
    flipped = masked_policy_logits(flipped_policy, flipped_actions, pad_mask)
    assert torch.allclose(original, flipped)


def test_flip_y_is_involutive_and_leaves_simple_actions_alone():
    torch.manual_seed(0)
    n, k, h, w = 2, 5, 9, 16
    spatial = torch.randn(n, 37, h, w)
    actions = random_actions(n, k, h, w, seed=2)

    once_s, once_a = flip_y(spatial, actions)
    twice_s, twice_a = flip_y(once_s, once_a)
    assert torch.equal(twice_s, spatial)
    assert torch.equal(twice_a, actions)

    simple = actions[..., 3] == 1
    assert torch.equal(once_a[..., 1][simple], actions[..., 1][simple])
    positional = ~simple
    assert torch.equal(once_a[..., 1][positional], (h - 1) - actions[..., 1][positional])
    # Channel and x never change.
    assert torch.equal(once_a[..., 0], actions[..., 0])
    assert torch.equal(once_a[..., 2], actions[..., 2])
