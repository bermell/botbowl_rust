"""Load prepared board-dims directories and collate ragged action lists.

Reads the `.npy` files the Rust `prepare` step writes (see
`botbowl-nn/src/bin/prepare.rs`). All samples in one dims-dir share one
spatial shape, so within a dir only the ragged legal-action list needs
padding.

Plan 042: a mixed-board-size corpus prepares into several ``dims_WxH``
subdirs. `MultiDimsDataset` puts them behind one index space and
`PerDimsBatchSampler` draws every batch from a single dir, so shapes never
need padding across boards (padding a small board up to a large one would
change the `oob` statistics the net reads). Each dir's share of the batches
is its share of the samples. `open_prepared` picks the right class from the
path, and `make_loader` the right `DataLoader` shape.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


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


def find_dims_dirs(path):
    """A prepared dims dir (holds ``spatial.npy``) → ``[path]``; otherwise
    its ``dims_*`` children that do, sorted. Raises if neither."""
    p = Path(path)
    if (p / "spatial.npy").exists():
        return [p]
    dirs = sorted(d for d in p.glob("dims_*") if (d / "spatial.npy").exists())
    if not dirs:
        raise FileNotFoundError(
            f"{p}: neither a prepared dims dir (no spatial.npy) nor a parent of dims_* dirs"
        )
    return dirs


class MultiDimsDataset(Dataset):
    """Several `PreparedDataset`s of different board shapes behind one index
    space: global index ``i`` maps to ``(group, local)`` through cumulative
    offsets. Samples of different groups have different spatial shapes, so
    a plain shuffled `DataLoader` would fail in `collate`; pair it with
    `PerDimsBatchSampler`.
    """

    def __init__(self, dims_dirs, augment=False):
        dims_dirs = [Path(d) for d in dims_dirs]
        if not dims_dirs:
            raise ValueError("MultiDimsDataset needs at least one dims dir")
        self.groups = [PreparedDataset(d, augment=augment) for d in dims_dirs]
        self.names = [d.name for d in dims_dirs]
        self.offsets = np.cumsum([0] + [len(g) for g in self.groups])
        # One net trains on all of them, so the layout must agree.
        first = self.groups[0]
        for name, g in zip(self.names, self.groups):
            if g.spatial.shape[1] != first.spatial.shape[1]:
                raise ValueError(
                    f"{name}: {g.spatial.shape[1]} channels vs {first.spatial.shape[1]} in {self.names[0]}"
                )
            if not torch.equal(g.spatial_scale, first.spatial_scale):
                raise ValueError(f"{name}: spatial_scales differ from {self.names[0]}")
            if g.manifest.get("nn_schema_version") != first.manifest.get("nn_schema_version"):
                raise ValueError(f"{name}: nn_schema_version differs from {self.names[0]}")
        self.manifest = first.manifest
        self.spatial_scale = first.spatial_scale

    def __len__(self):
        return int(self.offsets[-1])

    def group_of(self, i):
        """``(group index, local index)`` for global index ``i``."""
        g = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return g, int(i - self.offsets[g])

    def __getitem__(self, i):
        g, j = self.group_of(i)
        return self.groups[g][j]

    def group_sizes(self):
        """``{dims name: sample count}``, in dir order."""
        return {n: len(g) for n, g in zip(self.names, self.groups)}


class PerDimsBatchSampler(Sampler):
    """Batches of global indices, each from one group of a `MultiDimsDataset`.

    Every epoch: shuffle each group's indices (when ``shuffle``), cut them
    into batches, then shuffle the *order of batches* across groups. A group
    therefore contributes batches in proportion to its sample count, and an
    epoch visits every sample exactly once. The last batch of each group may
    be short unless ``drop_last``.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _group_batches(self, g):
        n = len(self.dataset.groups[g])
        base = int(self.dataset.offsets[g])
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        idx = idx + base
        batches = [idx[k : k + self.batch_size].tolist() for k in range(0, n, self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __iter__(self):
        batches = []
        for g in range(len(self.dataset.groups)):
            batches.extend(self._group_batches(g))
        if self.shuffle:
            order = torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in order]
        return iter(batches)

    def __len__(self):
        total = 0
        for g in self.dataset.groups:
            n = len(g)
            total += n // self.batch_size if self.drop_last else -(-n // self.batch_size)
        return total


def open_prepared(path, augment=False):
    """A `PreparedDataset` for one dims dir, a `MultiDimsDataset` for a
    directory of several. `train_loop.sh` passes ``prepared_train/`` itself
    since plan 042; a single ``dims_*`` path still works as before."""
    dirs = find_dims_dirs(path)
    if len(dirs) == 1:
        return PreparedDataset(dirs[0], augment=augment)
    return MultiDimsDataset(dirs, augment=augment)


def make_loader(ds, batch_size, shuffle):
    """The `DataLoader` that fits ``ds``: a per-group batch sampler for a
    mixed corpus, the plain shuffled loader otherwise."""
    if isinstance(ds, MultiDimsDataset):
        return DataLoader(ds, batch_sampler=PerDimsBatchSampler(ds, batch_size, shuffle=shuffle), collate_fn=collate)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


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
