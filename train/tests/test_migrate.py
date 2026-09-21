"""`bbnn.migrate`: trained weights survive a schema bump without a retrain.

The contract is *function preservation*: a v6 net and its v7 migration must
produce identical outputs whenever the v7 input is the v6 input with zeros in
the inserted slots. That is what lets a champion keep playing at its current
strength the generation the encoder changes.
"""

import subprocess
import sys

import pytest
import torch

from bbnn import migrate as mig
from bbnn.model import GLOBAL_FEATURES, SPATIAL_CHANNELS, BBNet


def v6_state_dict(seed=0, width=16, blocks=2):
    torch.manual_seed(seed)
    m = BBNet(spatial_ch=59, global_f=15, width=width, blocks=blocks, global_embed=8, value_hidden=16)
    # Non-trivial BatchNorm running stats, so eval-mode parity is a real check.
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(4, 59, 9, 16), torch.randn(4, 15))
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_schema_is_detected_from_shapes_and_unknown_shapes_are_refused():
    sd = v6_state_dict()
    assert mig.detect_schema(sd) == 6
    assert mig.CURRENT == 7
    assert mig.SCHEMAS[7] == {"C": SPATIAL_CHANNELS, "F": GLOBAL_FEATURES}
    bad = dict(sd)
    bad["global_fc.weight"] = torch.zeros(8, 99)
    with pytest.raises(ValueError, match="no known schema"):
        mig.detect_schema(bad)


def test_v6_to_v7_is_function_preserving_and_loads_strictly():
    old = v6_state_dict()
    new, src, dst = mig.migrate(old, log=lambda *_: None)
    assert (src, dst) == (6, 7)
    assert mig.detect_schema(new) == 7

    # Strict load into the production-shaped constructor.
    model = BBNet(**mig.shape_of(new))
    model.load_state_dict(new, strict=True)
    assert model.stem.weight.shape[1] == 61 + 8
    assert model.global_fc.weight.shape == (8, 18)

    # Every tensor that did not change shape is bit-identical.
    for k, v in old.items():
        if v.shape == new[k].shape:
            assert torch.equal(v, new[k]), k
    # The inserted columns are zero and sit where the encoder appends.
    assert torch.equal(new["stem.weight"][:, 59:61], torch.zeros(16, 2, 3, 3))
    assert torch.equal(new["stem.weight"][:, 61:], old["stem.weight"][:, 59:])
    assert torch.equal(new["global_fc.weight"][:, 15:], torch.zeros(8, 3))

    # Outputs agree on embedded inputs — the verifier's claim, re-asserted.
    step = mig.path_to(6, 7)[0]
    assert step.verify(old, new) < 1e-5


def test_migrated_net_ignores_the_new_features_until_trained():
    new, _, _ = mig.migrate(v6_state_dict(), log=lambda *_: None)
    model = BBNet(**mig.shape_of(new))
    model.load_state_dict(new)
    model.eval()
    g = torch.Generator().manual_seed(3)
    s = torch.rand(2, 61, 9, 16, generator=g)
    f = torch.rand(2, 18, generator=g)
    s0, f0 = s.clone(), f.clone()
    s0[:, 59:] = 0
    f0[:, 15:] = 0
    with torch.no_grad():
        p1, v1 = model(s, f)
        p0, v0 = model(s0, f0)
    assert torch.allclose(p1, p0, atol=1e-6) and torch.allclose(v1, v0, atol=1e-6)


def test_migration_is_idempotent_at_the_target_and_refuses_downgrades():
    old = v6_state_dict()
    new, _, _ = mig.migrate(old, log=lambda *_: None)
    again, src, dst = mig.migrate(new, log=lambda *_: None)
    assert (src, dst) == (7, 7)
    assert all(torch.equal(again[k], new[k]) for k in new)
    with pytest.raises(ValueError, match="downwards"):
        mig.path_to(7, 6)
    with pytest.raises(ValueError, match="no migration"):
        mig.path_to(7, 8)


def test_registry_is_a_contiguous_chain_to_the_current_schema():
    # Every registered schema older than CURRENT must reach CURRENT one step
    # at a time, and every step must verify on random weights.
    for v in sorted(mig.SCHEMAS):
        if v == mig.CURRENT:
            continue
        steps = mig.path_to(v, mig.CURRENT)
        assert [s.src for s in steps] == list(range(v, mig.CURRENT))
        assert [s.dst for s in steps] == list(range(v + 1, mig.CURRENT + 1))


def test_insert_zero_slices_keeps_the_old_columns_in_place():
    t = torch.arange(2 * 4).float().view(2, 4)
    out = mig.insert_zero_slices(t, 1, 3, 2)
    assert out.shape == (2, 6)
    assert torch.equal(out[:, :3], t[:, :3])
    assert torch.equal(out[:, 3:5], torch.zeros(2, 2))
    assert torch.equal(out[:, 5:], t[:, 3:])
    assert torch.equal(mig.insert_zero_slices(t, 1, 3, 0), t)


def test_cli_migrates_a_file_and_exports_onnx(tmp_path):
    src = tmp_path / "v6.pt"
    torch.save(v6_state_dict(), src)
    out = tmp_path / "v7.pt"
    onnx = tmp_path / "v7.onnx"
    r = subprocess.run(
        [sys.executable, "-m", "bbnn.migrate", str(src), "--out", str(out), "--onnx", str(onnx)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "v6 -> v7" in r.stdout and "verified" in r.stdout
    assert mig.detect_schema(torch.load(out)) == 7
    assert onnx.exists() and onnx.stat().st_size > 0
    listing = subprocess.run([sys.executable, "-m", "bbnn.migrate", "--list"], capture_output=True, text=True)
    assert "v6 -> v7" in listing.stdout
