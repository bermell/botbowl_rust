"""Plan 042: a mixed-board-size corpus prepares into several `dims_*` dirs and
the trainer must consume all of them — `train_loop.sh` used to take the first
one and silently drop the rest. Batches must never mix board shapes.
"""

import json

import numpy as np
import pytest
import torch

from bbnn.data import (
    MultiDimsDataset,
    PerDimsBatchSampler,
    PreparedDataset,
    collate,
    find_dims_dirs,
    make_loader,
    open_prepared,
)


def write_corpus(d, n, h, w, c=5, scales=None, schema=7):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(n * 31 + h * 7 + w)
    np.save(d / "spatial.npy", rng.integers(0, 9, size=(n, c, h, w), dtype=np.uint8))
    np.save(d / "global.npy", rng.standard_normal((n, 3)).astype(np.float32))
    np.save(d / "value.npy", rng.standard_normal(n).astype(np.float32))
    np.save(d / "chosen.npy", np.zeros(n, dtype=np.int64))
    np.save(d / "actions.npy", np.zeros((n, 4), dtype=np.int64))
    np.save(d / "policy.npy", np.ones(n, dtype=np.float32))
    np.save(d / "action_offsets.npy", np.arange(n + 1, dtype=np.int64))
    manifest = {"nn_schema_version": schema, "spatial_scales": scales or [1.0] * c}
    (d / "manifest.json").write_text(json.dumps(manifest))


def mixed(tmp_path, sizes=((7, 4, 6), (13, 9, 16), (5, 5, 10))):
    for n, h, w in sizes:
        write_corpus(tmp_path / f"dims_{w}x{h}", n, h, w)
    return tmp_path


def test_find_dims_dirs_accepts_a_dims_dir_or_its_parent(tmp_path):
    root = mixed(tmp_path)
    dirs = find_dims_dirs(root)
    assert [d.name for d in dirs] == ["dims_10x5", "dims_16x9", "dims_6x4"]
    assert find_dims_dirs(root / "dims_16x9") == [root / "dims_16x9"]
    with pytest.raises(FileNotFoundError):
        find_dims_dirs(tmp_path / "nothing_here")


def test_open_prepared_picks_the_class_from_the_path(tmp_path):
    root = mixed(tmp_path)
    assert isinstance(open_prepared(root / "dims_16x9"), PreparedDataset)
    ds = open_prepared(root)
    assert isinstance(ds, MultiDimsDataset)
    assert len(ds) == 7 + 13 + 5
    assert ds.group_sizes() == {"dims_10x5": 5, "dims_16x9": 13, "dims_6x4": 7}


def test_global_index_maps_to_the_right_group_and_sample(tmp_path):
    ds = open_prepared(mixed(tmp_path))
    # Groups are in sorted dir order: 10x5 (5), 16x9 (13), 6x4 (7).
    assert ds.group_of(0) == (0, 0)
    assert ds.group_of(4) == (0, 4)
    assert ds.group_of(5) == (1, 0)
    assert ds.group_of(17) == (1, 12)
    assert ds.group_of(18) == (2, 0)
    assert ds.group_of(24) == (2, 6)
    for i in range(len(ds)):
        g, j = ds.group_of(i)
        assert torch.equal(ds[i]["spatial"], ds.groups[g][j]["spatial"])
    assert ds[6]["spatial"].shape == (5, 9, 16)
    assert ds[20]["spatial"].shape == (5, 4, 6)


def test_batches_never_mix_shapes_and_cover_every_sample_once(tmp_path):
    ds = open_prepared(mixed(tmp_path))
    sampler = PerDimsBatchSampler(ds, batch_size=4, shuffle=True)
    torch.manual_seed(0)
    seen = []
    for batch in sampler:
        groups = {ds.group_of(i)[0] for i in batch}
        assert len(groups) == 1, f"batch spans groups {groups}"
        assert 1 <= len(batch) <= 4
        seen.extend(batch)
    assert sorted(seen) == list(range(len(ds)))
    # ceil(5/4) + ceil(13/4) + ceil(7/4) = 2 + 4 + 2
    assert len(sampler) == 8
    # A group's share of batches is its share of samples: the biggest group
    # gets the most batches.
    per_group = {}
    for batch in PerDimsBatchSampler(ds, batch_size=4, shuffle=False):
        g = ds.group_of(batch[0])[0]
        per_group[g] = per_group.get(g, 0) + 1
    assert per_group == {0: 2, 1: 4, 2: 2}


def test_make_loader_collates_homogeneous_batches(tmp_path):
    ds = open_prepared(mixed(tmp_path))
    loader = make_loader(ds, batch_size=4, shuffle=True)
    torch.manual_seed(1)
    shapes = set()
    total = 0
    for batch in loader:
        shapes.add(tuple(batch["spatial"].shape[1:]))
        total += batch["spatial"].shape[0]
    assert shapes == {(5, 5, 10), (5, 9, 16), (5, 4, 6)}
    assert total == len(ds)
    # Single dir: the plain loader, as before.
    single = make_loader(open_prepared(tmp_path / "dims_16x9"), batch_size=4, shuffle=False)
    assert sum(b["spatial"].shape[0] for b in single) == 13


def test_groups_must_share_the_encoder_layout(tmp_path):
    write_corpus(tmp_path / "dims_10x5", 3, 5, 10, scales=[1.0] * 5)
    write_corpus(tmp_path / "dims_16x9", 3, 9, 16, scales=[2.0] * 5)
    with pytest.raises(ValueError, match="spatial_scales"):
        MultiDimsDataset(find_dims_dirs(tmp_path))
    write_corpus(tmp_path / "dims_16x9", 3, 9, 16, scales=[1.0] * 5, schema=6)
    with pytest.raises(ValueError, match="nn_schema_version"):
        MultiDimsDataset(find_dims_dirs(tmp_path))
    write_corpus(tmp_path / "dims_16x9", 3, 9, 16, c=6, scales=[1.0] * 6)
    with pytest.raises(ValueError, match="channels"):
        MultiDimsDataset(find_dims_dirs(tmp_path))


def test_collate_is_unchanged_for_one_group(tmp_path):
    ds = open_prepared(mixed(tmp_path))
    batch = collate([ds[i] for i in range(5, 9)])
    assert batch["spatial"].shape == (4, 5, 9, 16)
