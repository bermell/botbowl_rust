"""`bbnn.from_onnx`: an exported net can be turned back into a state_dict.

The export folds BatchNorm into the convolutions, so the recovery cannot get
the original γ/β/μ/σ² back — only a net that computes the same thing in eval
mode. That is the contract these tests pin, because it is all `bbnn.migrate`
needs to carry a champion whose `.pt` is gone across a schema bump.
"""

import torch

from bbnn import migrate as mig
from bbnn.export import export_onnx
from bbnn.from_onnx import max_abs_diff_vs_onnx, shape_of_onnx, state_dict_from_onnx
from bbnn.model import GLOBAL_FEATURES, POLICY_CHANNELS, SPATIAL_CHANNELS, BBNet


def trained_ish_net(seed=0, width=16, blocks=2):
    """A small net with non-identity BatchNorm statistics — with the default
    stats (mean 0, var 1, γ 1, β 0) the fold is a no-op and the recovery would
    pass without doing anything."""
    torch.manual_seed(seed)
    m = BBNet(width=width, blocks=blocks, global_embed=8, value_hidden=16)
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(4, SPATIAL_CHANNELS, 9, 16), torch.randn(4, GLOBAL_FEATURES))
    return m.eval()


def test_recovered_net_matches_the_onnx_it_came_from(tmp_path):
    model = trained_ish_net()
    path = tmp_path / "net.onnx"
    export_onnx(model, path)

    sd = state_dict_from_onnx(path)
    assert shape_of_onnx(sd) == {
        "width": 16,
        "blocks": 2,
        "global_embed": 8,
        "spatial_ch": SPATIAL_CHANNELS,
        "global_f": GLOBAL_FEATURES,
        "policy_ch": POLICY_CHANNELS,
        "value_hidden": 16,
    }
    assert max_abs_diff_vs_onnx(path, sd, sizes=((5, 9),)) < 1e-4


def test_recovered_net_matches_the_torch_model_it_came_from(tmp_path):
    """The stronger statement: equality with the *PyTorch* model, not just
    with the ONNX runtime's reading of it."""
    model = trained_ish_net(seed=3)
    path = tmp_path / "net.onnx"
    export_onnx(model, path)
    back = BBNet(**shape_of_onnx(state_dict_from_onnx(path)))
    back.load_state_dict(state_dict_from_onnx(path))
    back.eval()

    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for (h, w) in [(9, 16), (7, 14)]:
            spatial = torch.rand(2, SPATIAL_CHANNELS, h, w, generator=g)
            global_ = torch.rand(2, GLOBAL_FEATURES, generator=g)
            p0, v0 = model(spatial, global_)
            p1, v1 = back(spatial, global_)
            assert torch.allclose(p0, p1, atol=1e-4)
            assert torch.allclose(v0, v1, atol=1e-4)


def test_a_recovered_v6_net_still_migrates_to_v7(tmp_path):
    """The whole point: ONNX-only champion → state_dict → schema bump."""
    torch.manual_seed(11)
    old = BBNet(spatial_ch=59, global_f=15, width=16, blocks=2, global_embed=8, value_hidden=16)
    old.train()
    with torch.no_grad():
        for _ in range(3):
            old(torch.randn(4, 59, 9, 16), torch.randn(4, 15))
    old.eval()
    path = tmp_path / "v6.onnx"
    export_onnx(old, path)  # exports at the model's own v6 widths

    sd = state_dict_from_onnx(path)
    assert mig.detect_schema(sd) == 6
    out, src, dst = mig.migrate(sd, log=lambda *_: None)
    assert (src, dst) == (6, 7)
    assert mig.detect_schema(out) == 7
