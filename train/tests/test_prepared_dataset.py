"""`PreparedDataset` must undo the u8 corpus encoding exactly.

Since nn_schema_version 4 `spatial.npy` holds the Rust encoder's raw integer
counts and `manifest.json` carries the per-channel divisors. Getting the
normalisation wrong here is train/inference skew that nothing else catches:
the loss still goes down, the net is just fed different numbers than
`NnEvaluator` feeds it in the search.
"""

import json

import numpy as np
import pytest
import torch

from bbnn.data import PreparedDataset


def write_corpus(d, n=3, c=5, h=4, w=6, scales=None, schema=4):
    rng = np.random.default_rng(0)
    spatial = rng.integers(0, 9, size=(n, c, h, w), dtype=np.uint8)
    np.save(d / "spatial.npy", spatial)
    np.save(d / "global.npy", rng.standard_normal((n, 3)).astype(np.float32))
    np.save(d / "value.npy", rng.standard_normal(n).astype(np.float32))
    np.save(d / "chosen.npy", np.zeros(n, dtype=np.int64))
    # One legal action per sample keeps the CSR trivial.
    np.save(d / "actions.npy", np.zeros((n, 4), dtype=np.int64))
    np.save(d / "policy.npy", np.ones(n, dtype=np.float32))
    np.save(d / "action_offsets.npy", np.arange(n + 1, dtype=np.int64))
    manifest = {"nn_schema_version": schema}
    if scales is not None:
        manifest["spatial_scales"] = scales
    (d / "manifest.json").write_text(json.dumps(manifest))
    return spatial


def test_spatial_is_divided_by_the_manifest_scales(tmp_path):
    scales = [1.0, 10.0, 8.0, 6.0, 12.0]
    spatial = write_corpus(tmp_path, scales=scales)
    ds = PreparedDataset(tmp_path)

    assert ds.spatial.dtype == np.uint8, "corpus should stay u8 on disk"
    for i in range(len(ds)):
        got = ds[i]["spatial"]
        want = torch.from_numpy(spatial[i].astype(np.float32)) / torch.tensor(scales).view(-1, 1, 1)
        assert got.dtype == torch.float32
        assert torch.equal(got, want), f"sample {i} normalised wrong"


def test_a_pre_v4_corpus_is_rejected_rather_than_silently_misread(tmp_path):
    # Without scales the planes are raw counts an order of magnitude off, and
    # training would happily converge on the wrong inputs.
    write_corpus(tmp_path, scales=None, schema=3)
    with pytest.raises(ValueError, match="spatial_scales"):
        PreparedDataset(tmp_path)


def test_scale_length_must_match_the_channel_count(tmp_path):
    write_corpus(tmp_path, c=5, scales=[1.0, 1.0])
    with pytest.raises(AssertionError):
        PreparedDataset(tmp_path)
