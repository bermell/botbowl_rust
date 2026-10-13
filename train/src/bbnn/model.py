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

# Must match botbowl-nn/src/encode.rs and actions.rs (nn_schema_version 9).
SPATIAL_CHANNELS = 61
GLOBAL_FEATURES = 18
POLICY_CHANNELS = 32
# The schema the action/encoder layout above is at. Stored in every checkpoint
# as the ``schema_version`` buffer: v8 (per-player setup) re-laid the policy
# channels without changing a single tensor shape, so shapes alone can no
# longer say which schema a state_dict is at. `bbnn.migrate` reads it.
# v9 appended UseSkill / DontUseSkill (A 30 -> 32).
SCHEMA_VERSION = 9


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

    Plan 050: ``value_head="wdl"`` replaces the tanh scalar with a softmax over the drive outcome
    as the mover sees it, class order ``[self scores, nobody scores, opponent scores]``. The
    ``value`` output stays the scalar the search reads, ``P(self) - P(opp)``, so every consumer
    (ONNX export, the sidecar, the Rust evaluator) is unchanged. ``wdl=True`` on ``forward`` /
    ``forward_masked`` returns the three logits instead, for the trainer's loss.

    Plan 063 (exp072), the head knobs — every default is the net the loop has always trained, so a
    64x6 checkpoint keeps its exact state_dict and `shape_of` reads each knob back from the weights:
    - ``value_conv``: channels of the value head's 1x1 conv (32);
    - ``value_hidden``: width of the value head's hidden layer (64);
    - ``value_pool``: ``"mean"`` (global mean over the board) or ``"meanmax"`` (mean and max
      concatenated, doubling the hidden layer's input: a local threat, a loose ball or a free
      carrier, is a max, and a mean over a 198-cell board dilutes it);
    - ``policy_conv``: 0 = the policy logits are one 1x1 conv of the trunk; ``k > 0`` puts a 3x3
      conv + BN + ReLU of ``k`` channels (``policy_conv`` / ``policy_bn``) in front of that 1x1
      (``policy_head``), giving the policy capacity of its own that the value head does not share.
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
        value_head: str = "scalar",
        value_conv: int = 32,
        value_pool: str = "mean",
        policy_conv: int = 0,
    ):
        super().__init__()
        if value_head not in ("scalar", "wdl"):
            raise ValueError(f"value_head must be scalar or wdl, got {value_head!r}")
        if value_pool not in ("mean", "meanmax"):
            raise ValueError(f"value_pool must be mean or meanmax, got {value_pool!r}")
        if policy_conv < 0 or value_conv < 1 or value_hidden < 1:
            raise ValueError(
                f"bad head widths: policy_conv {policy_conv}, value_conv {value_conv}, value_hidden {value_hidden}"
            )
        self.value_head = value_head
        self.value_pool = value_pool
        self.global_fc = nn.Linear(global_f, global_embed)
        self.stem = nn.Conv2d(spatial_ch + global_embed, width, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(width)
        self.blocks = nn.ModuleList(ResidualBlock(width) for _ in range(blocks))
        # Off (0) registers nothing and draws nothing from the RNG, so a default net's init and
        # state_dict are exactly the old constructor's (`test_arch.py` pins the seeded init).
        if policy_conv:
            self.policy_conv = nn.Conv2d(width, policy_conv, 3, padding=1)
            self.policy_bn = nn.BatchNorm2d(policy_conv)
        else:
            self.policy_conv = None
        self.policy_head = nn.Conv2d(policy_conv or width, policy_ch, 1)
        self.value_conv = nn.Conv2d(width, value_conv, 1)
        self.value_bn = nn.BatchNorm2d(value_conv)
        self.value_fc1 = nn.Linear(value_conv * (2 if value_pool == "meanmax" else 1), value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 3 if value_head == "wdl" else 1)
        # Not a parameter and unused by `forward` (so absent from the ONNX
        # export); it rides in the state_dict so a loader can tell a v8
        # checkpoint from a v7 one of identical shape.
        self.register_buffer("schema_version", torch.tensor(schema_version, dtype=torch.int32))

    @staticmethod
    def shape_of(state_dict) -> dict:
        """The constructor kwargs implied by a saved state_dict, so a loader
        can build a matching net without being told the architecture (plan
        032 #9 trains wider/deeper nets next to the 64x6 default; plan 063
        adds the head knobs). Always ``width`` and ``blocks``; a head knob
        only when it is not the default, so today's nets read back as
        ``{"width", "blocks"}`` exactly as before. Every other constructor
        argument is pinned by the encoder/action schema."""
        width = int(state_dict["stem.weight"].shape[0])
        blocks = len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})
        shape = {"width": width, "blocks": blocks}
        if int(state_dict["value_fc2.weight"].shape[0]) == 3:
            shape["value_head"] = "wdl"
        value_conv = int(state_dict["value_conv.weight"].shape[0])
        value_hidden, fc1_in = (int(d) for d in state_dict["value_fc1.weight"].shape)
        if value_conv != 32:
            shape["value_conv"] = value_conv
        if fc1_in == 2 * value_conv:
            shape["value_pool"] = "meanmax"
        elif fc1_in != value_conv:
            raise ValueError(f"value_fc1 takes {fc1_in} inputs from a {value_conv}-channel value conv")
        if value_hidden != 64:
            shape["value_hidden"] = value_hidden
        if "policy_conv.weight" in state_dict:
            shape["policy_conv"] = int(state_dict["policy_conv.weight"].shape[0])
        return shape

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs) -> "BBNet":
        """Build the net a state_dict was saved from and load it (strict).
        Refuses a checkpoint at another schema with a message naming the fix,
        instead of torch's "missing key: schema_version"."""
        check_schema(state_dict)
        model = cls(**cls.shape_of(state_dict), **kwargs)
        model.load_state_dict(state_dict)
        return model

    def _policy_out(self, x, mask=None):
        """The policy head on the trunk ``x``. ``mask`` re-zeroes the canvas padding after the
        3x3 conv, as the unpadded net's zero padding would be at the real edge."""
        if self.policy_conv is not None:
            x = F.relu(self.policy_bn(self.policy_conv(x)))
            if mask is not None:
                x = x * mask
        return self.policy_head(x)

    def _pool(self, v, mask=None):
        """Global pooling of the value features: the mean, or (``meanmax``) the mean and the max
        concatenated. With a ``mask`` the mean runs over the real cells only; the max needs no
        correction, since ``v`` is a ReLU output (>= 0) whose padding is already zeroed."""
        if mask is None:
            mean = v.mean(dim=(2, 3))                    # ReduceMean -> (N, C)
        else:
            mean = v.sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1.0)
        if self.value_pool == "mean":
            return mean
        return torch.cat([mean, v.amax(dim=(2, 3))], dim=1)  # (N, 2C)

    def _value_out(self, v, wdl):
        """The value head's last layer: the scalar, or (``wdl``) the three logits."""
        out = self.value_fc2(v)
        if self.value_head == "scalar":
            return torch.tanh(out)
        if wdl:
            return out
        p = torch.softmax(out, dim=1)
        return p[:, 0:1] - p[:, 2:3]                     # (N, 1): P(self) - P(opp)

    def forward(self, spatial, global_feat, wdl=False):
        n = spatial.shape[0]
        h = spatial.shape[2]
        w = spatial.shape[3]
        g = F.relu(self.global_fc(global_feat))          # (N, embed)
        g = g.view(n, -1, 1, 1).expand(-1, -1, h, w)     # (N, embed, H, W)
        x = torch.cat([spatial, g], dim=1)               # (N, C+embed, H, W)
        x = F.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        policy = self._policy_out(x)                     # (N, A, H, W)
        v = F.relu(self.value_bn(self.value_conv(x)))    # (N, value_conv, H, W)
        v = self._pool(v)                                # (N, value_conv), or 2x for meanmax
        v = F.relu(self.value_fc1(v))
        return policy, self._value_out(v, wdl)           # (N, 1), or (N, 3) logits

    def forward_masked(self, spatial, global_feat, mask, wdl=False):
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
        policy = self._policy_out(x, mask)
        v = F.relu(self.value_bn(self.value_conv(x))) * mask
        v = self._pool(v, mask)
        v = F.relu(self.value_fc1(v))
        return policy, self._value_out(v, wdl)


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
