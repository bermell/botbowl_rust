"""Load one prepared board-dims directory and collate ragged action lists.

Reads the `.npy` files the Rust `prepare` step writes (see
`botbowl-nn/src/bin/prepare.rs`). All samples in a dims-dir share one
spatial shape, so only the ragged legal-action list needs padding.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def flip_y(spatial, actions):
    """Apply Blood Bowl's left-right symmetry: mirror the board across its
    long axis, i.e. flip the tensor H dim (= engine y). The x-mirror is NOT
    a symmetry — x is the attacking direction, already spent on the
    Home/Away canonicalisation in `botbowl-nn/src/perspective.rs`.

    Value, global features and simple-action logits (channel spatial max)
    are invariant; positional action cells flip their y. Works on a single
    sample ``(C, H, W)`` / ``(K, 4)`` or a batch ``(N, C, H, W)`` /
    ``(N, K, 4)``.
    """
    h = spatial.shape[-2]
    flipped = spatial.flip(-2)
    actions = actions.clone()
    positional = actions[..., 3] == 0
    actions[..., 1] = torch.where(positional, (h - 1) - actions[..., 1], actions[..., 1])
    return flipped, actions


class PreparedDataset(Dataset):
    """``augment=True`` applies a random y-flip per sample per access —
    fresh flips every epoch, nothing duplicated on disk."""

    def __init__(self, dims_dir, augment=False):
        self.augment = augment
        d = Path(dims_dir)
        # Memory-mapped: the spatial planes outgrow RAM long before anything
        # else; the OS pages slices in on demand. `__getitem__` copies its
        # slice out.
        self.spatial = np.load(d / "spatial.npy", mmap_mode="r")  # (N, C, H, W) u8
        self.global_ = np.load(d / "global.npy")             # (N, F) f32
        self.value = np.load(d / "value.npy")                # (N,) f32
        # Plan 036 W4: per-sample weight for the value term, 1/len(drive).
        # Optional — corpora prepared before it exists fall back to 1.0, which
        # is what an unweighted MSE already does, so an old prepared dir still
        # trains and `--per-drive-value-weight` on one is a no-op rather than a
        # crash.
        wpath = d / "weight.npy"
        self.weight = np.load(wpath) if wpath.exists() else np.ones(len(self.value), dtype=np.float32)
        self.chosen = np.load(d / "chosen.npy")              # (N,) i64
        self.actions = np.load(d / "actions.npy")            # (M, 4) i64
        self.policy = np.load(d / "policy.npy")              # (M,) f32
        self.offsets = np.load(d / "action_offsets.npy")     # (N+1,) i64
        with open(d / "manifest.json") as f:
            self.manifest = json.load(f)
        assert self.spatial.shape[0] == len(self.value)
        assert len(self.weight) == len(self.value)
        assert len(self.offsets) == len(self.value) + 1

        # Since schema v4 `spatial.npy` holds the encoder's *raw* integer
        # counts as u8 (4x smaller on disk) and the manifest carries the
        # per-channel divisors that turn them into what the network eats.
        # Those divisors are emitted by `botbowl-nn/src/encode.rs`, which is
        # the single source of feature layout — never hardcode them here, or
        # training and inference normalise differently and nothing says so.
        scales = self.manifest.get("spatial_scales")
        if scales is None:
            raise ValueError(
                f"{d}/manifest.json has no 'spatial_scales' — it was prepared "
                f"at nn_schema_version "
                f"{self.manifest.get('nn_schema_version')}, before the u8 "
                f"corpus (v4). Re-run `prepare`."
            )
        assert len(scales) == self.spatial.shape[1], (
            f"{len(scales)} scales for {self.spatial.shape[1]} channels"
        )
        # (C, 1, 1) so it broadcasts over a single (C, H, W) sample.
        self.spatial_scale = torch.tensor(scales, dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return self.spatial.shape[0]

    def __getitem__(self, i):
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        # np.array copies the slice out of the read-only mmap (from_numpy
        # rejects non-writable arrays).
        spatial = torch.from_numpy(np.array(self.spatial[i])).float().div_(self.spatial_scale)
        actions = torch.from_numpy(self.actions[lo:hi]).long()         # (K_i, 4)
        if self.augment and torch.rand(()) < 0.5:
            spatial, actions = flip_y(spatial, actions)
        return {
            "spatial": spatial,
            "global": torch.from_numpy(self.global_[i]).float(),
            "value": torch.tensor([self.value[i]], dtype=torch.float32),
            "weight": torch.tensor([self.weight[i]], dtype=torch.float32),
            "chosen": int(self.chosen[i]),
            "actions": actions,
            "policy": torch.from_numpy(self.policy[lo:hi]).float(),    # (K_i,)
        }


def collate(batch):
    """Stack fixed-shape tensors; pad ragged action lists to ``K_max``.

    Returns a dict with:
        spatial  (N, C, H, W)
        global   (N, F)
        value    (N, 1)
        weight   (N, 1)         f32, per-sample value-loss weight
        actions  (N, K_max, 4)  long, padded with 0
        policy   (N, K_max)     f32, padded with 0
        pad_mask (N, K_max)     bool, True for real actions
        chosen   (N,)           long, local index of played action
    """
    n = len(batch)
    k_max = max(b["actions"].shape[0] for b in batch)

    spatial = torch.stack([b["spatial"] for b in batch])
    global_ = torch.stack([b["global"] for b in batch])
    value = torch.stack([b["value"] for b in batch])
    weight = torch.stack([b["weight"] for b in batch])
    chosen = torch.tensor([b["chosen"] for b in batch], dtype=torch.long)

    actions = torch.zeros(n, k_max, 4, dtype=torch.long)
    policy = torch.zeros(n, k_max, dtype=torch.float32)
    pad_mask = torch.zeros(n, k_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b["actions"].shape[0]
        actions[i, :k] = b["actions"]
        policy[i, :k] = b["policy"]
        pad_mask[i, :k] = True

    return {
        "spatial": spatial,
        "global": global_,
        "value": value,
        "weight": weight,
        "actions": actions,
        "policy": policy,
        "pad_mask": pad_mask,
        "chosen": chosen,
    }
