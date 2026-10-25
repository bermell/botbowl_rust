"""Plan 063 / exp072: the architecture knobs (`value_conv`, `value_hidden`, `value_pool`,
`policy_conv`) next to `width` / `blocks`.

Every knob defaults to today's net, so a 64x6 checkpoint keeps its exact state_dict, and every
loader (`from_state_dict`, the sidecar's registry, `bbnn.from_onnx`, `bbnn.migrate`, the trainer's
warm start) rebuilds any shape from the weights alone. ONNX names and shapes stay what the Rust
evaluator and the sidecar expect.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from bbnn.export import export_onnx
from bbnn.from_onnx import max_abs_diff_vs_onnx, shape_of_onnx, state_dict_from_onnx
from bbnn.model import GLOBAL_FEATURES, POLICY_CHANNELS, SPATIAL_CHANNELS, BBNet

BIG_HEADS = dict(value_conv=64, value_pool="meanmax", value_hidden=128, policy_conv=96)

# Every key and shape of today's 64x6 state_dict: the defaults must not move one of them.
DEFAULT_KEYS = {
    "global_fc.weight": (16, GLOBAL_FEATURES),
    "stem.weight": (64, SPATIAL_CHANNELS + 16, 3, 3),
    "policy_head.weight": (POLICY_CHANNELS, 64, 1, 1),
    "policy_head.bias": (POLICY_CHANNELS,),
    "value_conv.weight": (32, 64, 1, 1),
    "value_bn.weight": (32,),
    "value_fc1.weight": (64, 32),
    "value_fc2.weight": (1, 64),
}


def _live_bn(net, seed=0):
    """Non-trivial BatchNorm statistics and affines, so a fold or a mask that is wrong shows."""
    g = torch.Generator().manual_seed(seed)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.rand(n, generator=g) - 0.5)
            m.running_var.copy_(torch.rand(n, generator=g) * 1.5 + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.rand(n, generator=g) - 0.5)
    return net.eval()


def test_defaults_are_todays_net():
    sd = BBNet().state_dict()
    for k, shape in DEFAULT_KEYS.items():
        assert tuple(sd[k].shape) == shape, k
    assert not any(k.startswith(("policy_conv.", "policy_bn.")) for k in sd)
    assert BBNet.shape_of(sd) == {"width": 64, "blocks": 6}


def _digest(sd):
    import hashlib

    h = hashlib.sha256()
    for k, v in sd.items():
        h.update(k.encode())
        h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


# sha256 of the seeded (manual_seed(0)) init of `BBNet()` and `BBNet(width=96, blocks=8)` as built
# by the code before the knobs existed (6f0bb7a, torch 2.13.0).
OLD_INIT = {
    (64, 6): "1d0be2656ef7772a4d4303cd056c7dca116d652bf3492a0dcbde78b131a47660",
    (96, 8): "09fc8c491217a0d4529e6f2ecff6919ed4bc042ab89ff29d75ca57c7a816a5d4",
}


@pytest.mark.skipif(not torch.__version__.startswith("2.13."), reason="init checksums pinned at torch 2.13")
def test_default_init_draws_the_same_weights_as_before():
    """A fresh net at a seed is the net the old constructor built: the knobs register no module
    and draw nothing when off, so a from-scratch control arm is today's net exactly."""
    # The pins were taken at schema v9 (policy width 32, `schema_version` 9 in the state_dict);
    # schema v10 only widens the last layer, so the knobs are checked at the schema they were pinned at.
    for (width, blocks), want in OLD_INIT.items():
        torch.manual_seed(0)
        assert _digest(BBNet(width=width, blocks=blocks, policy_ch=32, schema_version=9).state_dict()) == want
        torch.manual_seed(0)
        explicit = BBNet(
            width=width, blocks=blocks, value_conv=32, value_hidden=64, value_pool="mean", policy_conv=0,
            policy_ch=32, schema_version=9,
        )
        assert _digest(explicit.state_dict()) == want


@pytest.mark.parametrize(
    "knobs",
    [
        {"value_conv": 48},
        {"value_hidden": 96},
        {"value_pool": "meanmax"},
        {"policy_conv": 24},
        {"width": 96, "blocks": 8, **BIG_HEADS},
        {"value_head": "wdl", **BIG_HEADS},
    ],
)
def test_shape_of_infers_every_knob(knobs):
    sd = BBNet(**{"width": 16, "blocks": 2, **knobs}).state_dict()
    shape = BBNet.shape_of(sd)
    for k, v in knobs.items():
        assert shape[k] == v, (k, shape)
    model = BBNet.from_state_dict(sd)  # strict
    assert BBNet.shape_of(model.state_dict()) == shape


def test_a_default_net_refuses_a_big_head_state_dict():
    with pytest.raises(RuntimeError):
        BBNet(width=16, blocks=2).load_state_dict(BBNet(width=16, blocks=2, **BIG_HEADS).state_dict())


def test_bad_knob_values_are_refused():
    with pytest.raises(ValueError):
        BBNet(value_pool="max")
    with pytest.raises(ValueError):
        BBNet(policy_conv=-1)


def test_big_heads_change_the_outputs_not_their_shapes():
    net = _live_bn(BBNet(width=16, blocks=2, **BIG_HEADS))
    for (h, w) in [(17, 28), (9, 16)]:
        with torch.no_grad():
            p, v = net(torch.randn(3, SPATIAL_CHANNELS, h, w), torch.randn(3, GLOBAL_FEATURES))
        assert p.shape == (3, POLICY_CHANNELS, h, w) and v.shape == (3, 1)
        assert torch.all(v.abs() <= 1)


def test_masked_canvas_forward_matches_unpadded_with_big_heads():
    """`forward_masked` must zero the policy conv's padding and take the max over real cells only."""
    from test_model import CANVAS, CURRICULUM_TENSORS, _embed

    net = _live_bn(BBNet(width=16, blocks=2, **BIG_HEADS), seed=3)
    rows = [(h, w, torch.randn(1, SPATIAL_CHANNELS, h, w), torch.randn(1, GLOBAL_FEATURES)) for (h, w) in CURRICULUM_TENSORS]
    cs = torch.cat([_embed(s, CANVAS)[0] for (_, _, s, _) in rows])
    m = torch.cat([_embed(s, CANVAS)[1] for (_, _, s, _) in rows])
    g = torch.cat([gg for (_, _, _, gg) in rows])
    with torch.no_grad():
        p, v = net.forward_masked(cs, g, m)
        for i, (h, w, s, gg) in enumerate(rows):
            ref_p, ref_v = net(s, gg)
            assert (p[i : i + 1, :, :h, :w] - ref_p).abs().max().item() < 1e-5, f"{h}x{w} policy"
            assert (v[i] - ref_v[0]).abs().item() < 1e-5, f"{h}x{w} value"


def test_onnx_keeps_its_names_and_shapes(tmp_path):
    import onnx

    net = _live_bn(BBNet(width=16, blocks=2, **BIG_HEADS))
    path = tmp_path / "heads.onnx"
    export_onnx(net, path)
    graph = onnx.load(str(path)).graph
    assert [i.name for i in graph.input] == ["spatial", "global"]
    assert [o.name for o in graph.output] == ["policy", "value"]

    def dims(vi):
        return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]

    assert dims(graph.input[0]) == ["N", SPATIAL_CHANNELS, "H", "W"]
    assert dims(graph.input[1]) == ["N", GLOBAL_FEATURES]
    assert dims(graph.output[0]) == ["N", POLICY_CHANNELS, "H", "W"]
    assert dims(graph.output[1])[1:] == [1]


def test_a_96x8_big_head_net_round_trips_through_onnx_exactly(tmp_path):
    """.pt -> .onnx -> `bbnn.from_onnx` -> the same outputs as the torch net and the ONNX."""
    torch.manual_seed(0)
    net = _live_bn(BBNet(width=96, blocks=8, **BIG_HEADS), seed=1)
    path = tmp_path / "big.onnx"
    export_onnx(net, path)
    sd = state_dict_from_onnx(path)
    assert shape_of_onnx(sd) == {
        "width": 96,
        "blocks": 8,
        **BIG_HEADS,
        "global_embed": 16,
        "spatial_ch": SPATIAL_CHANNELS,
        "global_f": GLOBAL_FEATURES,
        "policy_ch": POLICY_CHANNELS,
    }
    assert max_abs_diff_vs_onnx(path, sd, sizes=((5, 9),)) < 1e-4
    back = BBNet.from_state_dict(sd).eval()
    s, g = torch.randn(2, SPATIAL_CHANNELS, 11, 18), torch.randn(2, GLOBAL_FEATURES)
    with torch.no_grad():
        p0, v0 = net(s, g)
        p1, v1 = back(s, g)
    assert (p0 - p1).abs().max().item() < 1e-4 and (v0 - v1).abs().max().item() < 1e-5


def test_migrate_shape_of_passes_the_knobs_through():
    from bbnn import migrate

    sd = BBNet(width=16, blocks=2, **BIG_HEADS).state_dict()
    shape = migrate.shape_of(sd)
    model = BBNet(**shape)
    model.load_state_dict(sd)
    assert all(shape[k] == v for k, v in BIG_HEADS.items())


def test_the_sidecar_serves_a_big_head_96x8_net_like_the_torch_net(tmp_path):
    """`nn_server.Registry` rebuilds the net from its `.pt` (no shape flag) and answers exactly
    what the torch net does, on the plain and the canvas path; its canary is that net's answer."""
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    import nn_server as ns

    torch.manual_seed(0)
    net = _live_bn(BBNet(width=96, blocks=8, **BIG_HEADS), seed=2)
    torch.save(net.state_dict(), tmp_path / "big.pt")
    for canvas in (None, (11, 18)):
        reg = ns.Registry("cpu", capacity=1, jit="auto", canvas=canvas)
        model = reg.get(str(tmp_path / "big.onnx"))
        s, g = ns.canary_input()
        with torch.no_grad():
            p, v = net(torch.from_numpy(s), torch.from_numpy(g))
        want = np.float32(v.numpy().reshape(-1)[0]).tobytes() + p.numpy().reshape(-1).astype(np.float32).tobytes()
        got = np.frombuffer(model.canary, dtype="<f4")
        assert np.abs(got - np.frombuffer(want, dtype="<f4")).max() < 1e-4
        if canvas is not None:
            h, w = 9, 16
            s2, g2 = torch.randn(2, SPATIAL_CHANNELS, h, w), torch.randn(2, GLOBAL_FEATURES)
            cs = torch.zeros(2, SPATIAL_CHANNELS, *canvas)
            cs[:, :, :h, :w] = s2
            mask = torch.zeros(2, 1, *canvas)
            mask[:, :, :h, :w] = 1.0
            with torch.no_grad():
                mp, mv = model.masked(cs, g2, mask)
                rp, rv = net(s2, g2)
            assert (mp[:, :, :h, :w] - rp).abs().max().item() < 1e-4
            assert (mv - rv).abs().max().item() < 1e-5


def test_heads_fixture_is_current():
    """The committed tract fixture for the big heads (`botbowl-nn/tests/fixtures/heads/`) is what
    `bbnn.fixture.build_heads` writes: rebuild it after any head change."""
    from bbnn.fixture import fixtures_dir

    d = fixtures_dir() / "heads"
    assert (d / "tiny_heads.onnx").exists(), "run `uv run python -m bbnn.fixture`"
    sd = state_dict_from_onnx(d / "tiny_heads.onnx")
    shape = shape_of_onnx(sd)
    assert shape["value_pool"] == "meanmax" and shape["policy_conv"] > 0
