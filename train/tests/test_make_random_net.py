"""Plan 071 R1: `scripts/make_random_net.py` builds gen 0 in any shape the trainer builds. The head
knobs reach the net, `BBNet.shape_of` reads them back, the in-process backend's `NET.ts` exports
from the .pt, the ONNX export matches the .pt, and with no head flags the net is today's, bit for
bit."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from bbnn.export_ts import export
from bbnn.model import BBNet

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_random_net.py"
HEADS = ["--value-conv", "16", "--value-pool", "meanmax", "--value-hidden", "24", "--policy-conv", "12"]


def _make(tmp_path, name, *args):
    out = tmp_path / name
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), "--seed", "3", *args], check=True,
                   capture_output=True)
    return out.with_suffix(".pt"), out.with_suffix(".onnx")


def test_head_knobs_round_trip_through_shape_of_and_the_ts_export(tmp_path):
    pt, onnx = _make(tmp_path, "heads", "--width", "16", "--blocks", "2", *HEADS)
    state = torch.load(pt, weights_only=True)
    assert BBNet.shape_of(state) == {"width": 16, "blocks": 2, "value_conv": 16, "value_pool": "meanmax",
                                     "value_hidden": 24, "policy_conv": 12}
    assert onnx.stat().st_size > 0
    ts = export(pt, tmp_path / "heads.ts", (13, 22))  # traces and checks the trace against eager
    assert torch.jit.load(str(ts)) is not None


def test_onnx_export_matches_the_pt(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pt, onnx = _make(tmp_path, "heads", "--width", "16", "--blocks", "2", *HEADS)
    net = BBNet.from_state_dict(torch.load(pt, weights_only=True)).eval()
    sess = ort.InferenceSession(str(onnx))
    feeds = {i.name: torch.randn(*[2 if not isinstance(d, int) else d for d in i.shape]).numpy()
             for i in sess.get_inputs()}
    with torch.no_grad():
        want = net(*[torch.from_numpy(v) for v in feeds.values()])
    for got, w in zip(sess.run(None, feeds), want):
        assert abs(torch.from_numpy(got) - w).max() < 1e-4


def test_no_head_flags_is_the_old_net_bit_for_bit(tmp_path):
    pt, _ = _make(tmp_path, "plain", "--width", "16", "--blocks", "2")
    state = torch.load(pt, weights_only=True)
    assert BBNet.shape_of(state) == {"width": 16, "blocks": 2}
    torch.manual_seed(3)  # what the script did before R1: seed, then BBNet(width, blocks)
    old = BBNet(width=16, blocks=2).state_dict()
    assert state.keys() == old.keys() and all(torch.equal(state[k], old[k]) for k in old)
