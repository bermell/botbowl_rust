"""The plan-017 value/policy tower and the legal-action logit gather.

The gather (`masked_policy_logits`) mirrors the Rust evaluator
(`botbowl-nn/src/eval.rs`) exactly: a positional action reads one cell of
its policy channel; a simple action takes that channel's spatial max. No
masking or softmax lives inside the ONNX graph — the network only emits
raw `(N, A, H, W)` policy logits and a scalar value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match botbowl-nn/src/encode.rs and actions.rs.
SPATIAL_CHANNELS = 37
GLOBAL_FEATURES = 15
POLICY_CHANNELS = 30


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class BBNet(nn.Module):
    """Global-feature embedding broadcast over the board, concatenated with
    the spatial planes, through a conv tower to a spatial policy head and a
    pooled scalar value head.

    Outputs:
    - ``policy``: ``(N, POLICY_CHANNELS, H, W)`` raw logits.
    - ``value``:  ``(N, 1)`` in ``[-1, 1]`` (mover-centric).
    """

    def __init__(
        self,
        spatial_ch: int = SPATIAL_CHANNELS,
        global_f: int = GLOBAL_FEATURES,
        policy_ch: int = POLICY_CHANNELS,
        width: int = 64,
        blocks: int = 6,
        global_embed: int = 16,
        value_hidden: int = 64,
    ):
        super().__init__()
        self.global_fc = nn.Linear(global_f, global_embed)
        self.stem = nn.Conv2d(spatial_ch + global_embed, width, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(blocks))
        self.policy_head = nn.Conv2d(width, policy_ch, 1)
        self.value_conv = nn.Conv2d(width, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, spatial, global_feat):
        n = spatial.shape[0]
        h = spatial.shape[2]
        w = spatial.shape[3]
        g = F.relu(self.global_fc(global_feat))          # (N, embed)
        g = g.view(n, -1, 1, 1).expand(-1, -1, h, w)     # (N, embed, H, W)
        x = torch.cat([spatial, g], dim=1)               # (N, C+embed, H, W)
        x = F.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        policy = self.policy_head(x)                     # (N, A, H, W)
        v = F.relu(self.value_bn(self.value_conv(x)))    # (N, 32, H, W)
        v = v.mean(dim=(2, 3))                           # ReduceMean → (N, 32)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))                # (N, 1)
        return policy, v


def masked_policy_logits(policy, actions, pad_mask):
    """Gather per-legal-action logits from a spatial policy map.

    Mirrors the Rust gather: positional → single cell, simple → channel
    spatial max. Padded slots are set to ``-1e9`` so they vanish under a
    subsequent softmax.

    Args:
        policy:   ``(N, A, H, W)`` raw logits.
        actions:  ``(N, K, 4)`` long: ``[channel, y, x, is_simple]``.
        pad_mask: ``(N, K)`` bool, ``True`` for real actions.

    Returns:
        ``(N, K)`` logits.
    """
    n, a, h, w = policy.shape
    k = actions.shape[1]
    chan = actions[..., 0].clamp(0, a - 1)
    y = actions[..., 1].clamp(0, h - 1)
    x = actions[..., 2].clamp(0, w - 1)
    is_simple = actions[..., 3].bool()
    n_idx = torch.arange(n, device=policy.device)[:, None].expand(n, k)

    pos_logit = policy[n_idx, chan, y, x]                # (N, K)
    flat = policy.reshape(n, a, h * w)
    gathered = flat[n_idx, chan]                         # (N, K, H*W)
    simple_logit = gathered.max(dim=-1).values           # (N, K)

    logit = torch.where(is_simple, simple_logit, pos_logit)
    return logit.masked_fill(~pad_mask, -1e9)
