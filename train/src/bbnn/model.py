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

# Must match botbowl-nn/src/encode.rs and actions.rs (nn_schema_version 8).
SPATIAL_CHANNELS = 61
GLOBAL_FEATURES = 18
POLICY_CHANNELS = 30
# The schema the action/encoder layout above is at. Stored in every checkpoint
# as the ``schema_version`` buffer: v8 (per-player setup) re-laid the policy
# channels without changing a single tensor shape, so shapes alone can no
# longer say which schema a state_dict is at. `bbnn.migrate` reads it.
SCHEMA_VERSION = 8


class SchemaError(ValueError):
    """A checkpoint is not at the schema this code is at — run `bbnn.migrate`."""


def check_schema(state_dict) -> None:
    """Raise `SchemaError` unless ``state_dict`` carries the current schema
    marker. A checkpoint without one predates v8."""
    v = state_dict.get("schema_version")
    if v is None:
        raise SchemaError(
            f"checkpoint has no schema_version (predates v8) — migrate it first: "
            f"python -m bbnn.migrate old.pt --out new.pt --onnx new.onnx"
        )
    if int(v) != SCHEMA_VERSION:
        raise SchemaError(f"checkpoint is at schema v{int(v)}, this code is at v{SCHEMA_VERSION} — run bbnn.migrate")


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
        schema_version: int = SCHEMA_VERSION,
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
        # Not a parameter and unused by `forward` (so absent from the ONNX
        # export); it rides in the state_dict so a loader can tell a v8
        # checkpoint from a v7 one of identical shape.
        self.register_buffer("schema_version", torch.tensor(schema_version, dtype=torch.int32))

    @staticmethod
    def shape_of(state_dict) -> dict:
        """``{"width", "blocks"}`` implied by a saved state_dict, so a loader
        can build a matching net without being told the architecture (plan
        032 #9 trains wider/deeper nets next to the 64x6 default). Every
        other constructor argument is pinned by the encoder/action schema."""
        width = int(state_dict["stem.weight"].shape[0])
        blocks = len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})
        return {"width": width, "blocks": blocks}

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs) -> "BBNet":
        """Build the net a state_dict was saved from and load it (strict).
        Refuses a checkpoint at another schema with a message naming the fix,
        instead of torch's "missing key: schema_version"."""
        check_schema(state_dict)
        model = cls(**cls.shape_of(state_dict), **kwargs)
        model.load_state_dict(state_dict)
        return model

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

    def forward_masked(self, spatial, global_feat, mask):
        """`forward` for boards embedded top-left in a larger zero canvas.

        ``mask`` is ``(N, 1, H, W)``: 1 on each sample's own ``h × w``, 0 on the padding. Zeroing
        the padding after every layer is exactly what the unpadded net's zero padding does at the
        real edge, and the value head's mean runs over the real cells only. So every sample gets
        its own board's `forward` result (up to float reassociation) on the shared canvas, and one
        canvas lets the sidecar batch every board size together. Crop the policy to ``h × w``.
        Keep this in step with `forward`: `test_model.py` pins that they agree.
        """
        n = spatial.shape[0]
        h = spatial.shape[2]
        w = spatial.shape[3]
        g = F.relu(self.global_fc(global_feat))
        g = g.view(n, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([spatial, g], dim=1) * mask
        x = F.relu(self.stem_bn(self.stem(x))) * mask
        for b in self.blocks:
            y = F.relu(b.b1(b.c1(x))) * mask
            y = b.b2(b.c2(y)) * mask
            x = F.relu(x + y)
        policy = self.policy_head(x)
        v = F.relu(self.value_bn(self.value_conv(x))) * mask
        v = v.sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
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
