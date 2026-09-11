"""`scripts/nn_server.py` over a real socket, on the CPU, with a tiny net.

The Rust side has live tests (`botbowl-nn/tests/remote.rs`) that need a
running server and a GPU box; these pin the parts of the server that are
independent of the device — wire framing, the canary handshake, the
model_id interlock, batch invariance, and that a request split across
several `send` calls still parses — so a refactor of the event loop cannot
break them silently.
"""

from __future__ import annotations

import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import nn_server as ns  # noqa: E402
from bbnn.model import GLOBAL_FEATURES, POLICY_CHANNELS, SPATIAL_CHANNELS, BBNet  # noqa: E402

H, W = ns.CANARY_H, ns.CANARY_W


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A CPU server on a tiny random net, plus the module it serves."""
    torch.manual_seed(0)
    module = BBNet(width=8, blocks=1).eval()
    d = tmp_path_factory.mktemp("srv")
    torch.save(module.state_dict(), d / "tiny.pt")
    sock = str(d / "s.sock")
    registry = ns.Registry("cpu", capacity=1, jit="off")
    pool = ns.RunnerPool("cpu", max_batch=8, enabled=False)
    server = ns.Server(registry, pool, ns.Stats(), max_batch=8, max_wait_us=0)
    t = threading.Thread(target=server.serve, args=(sock,), daemon=True)
    t.start()
    for _ in range(200):
        if Path(sock).exists():
            break
        time.sleep(0.01)
    yield sock, str(d / "tiny.onnx"), module, server
    server.stopping.set()
    t.join(timeout=5)


def recv_all(s: socket.socket, n: int) -> bytes:
    out = b""
    while len(out) < n:
        chunk = s.recv(n - len(out))
        if not chunk:
            raise ConnectionError("EOF")
        out += chunk
    return out


def connect(sock: str, model_path: str) -> tuple[socket.socket, int, bytes]:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)
    path = model_path.encode()
    s.sendall(struct.pack("<4sHHHH", ns.MAGIC, ns.PROTOCOL_VERSION, SPATIAL_CHANNELS, GLOBAL_FEATURES, len(path)) + path)
    status, model_id, h, w, n = struct.unpack("<HHHHI", recv_all(s, 12))
    payload = recv_all(s, n)
    assert status == ns.STATUS_OK, payload
    assert (h, w) == (H, W)
    return s, model_id, payload


def request(s: socket.socket, model_id: int, spatial: np.ndarray, global_: np.ndarray, want_policy: bool):
    flags = ns.FLAG_WANT_POLICY if want_policy else 0
    body = struct.pack("<HHHH", model_id, flags, H, W) + spatial.astype("<f4").tobytes() + global_.astype("<f4").tobytes()
    s.sendall(struct.pack("<I", len(body)) + body)
    (n,) = struct.unpack("<I", recv_all(s, 4))
    payload = recv_all(s, n)
    value = struct.unpack("<f", payload[:4])[0]
    policy = np.frombuffer(payload[4:], dtype="<f4") if want_policy else None
    return value, policy


def reference(module, spatial, global_):
    with torch.no_grad():
        p, v = module(torch.from_numpy(spatial)[None], torch.from_numpy(global_)[None])
    return v.item(), p.reshape(-1).numpy()


def sample(seed: int):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((SPATIAL_CHANNELS, H, W), dtype=np.float32),
        rng.standard_normal(GLOBAL_FEATURES, dtype=np.float32),
    )


def test_canary_is_the_models_answer_on_the_committed_fixture(served):
    sock, model, module, _ = served
    s, _, canary = connect(sock, model)
    spatial, global_ = ns.canary_input()
    v, p = reference(module, spatial[0], global_[0])
    got_v = struct.unpack("<f", canary[:4])[0]
    got_p = np.frombuffer(canary[4:], dtype="<f4")
    assert abs(got_v - v) < 1e-5
    assert got_p.shape == (POLICY_CHANNELS * H * W,)
    assert np.abs(got_p - p).max() < 1e-5
    s.close()


def test_value_only_and_policy_responses(served):
    sock, model, module, _ = served
    s, model_id, _ = connect(sock, model)
    spatial, global_ = sample(1)
    v, p = reference(module, spatial, global_)
    got_v, got_p = request(s, model_id, spatial, global_, want_policy=False)
    assert got_p is None and abs(got_v - v) < 1e-5
    got_v, got_p = request(s, model_id, spatial, global_, want_policy=True)
    assert abs(got_v - v) < 1e-5
    assert np.abs(got_p - p).max() < 1e-5
    s.close()


def test_a_request_split_across_sends_still_parses(served):
    sock, model, module, _ = served
    s, model_id, _ = connect(sock, model)
    spatial, global_ = sample(2)
    body = struct.pack("<HHHH", model_id, 0, H, W) + spatial.tobytes() + global_.tobytes()
    frame = struct.pack("<I", len(body)) + body
    for piece in (frame[:3], frame[3:100], frame[100:]):
        s.sendall(piece)
        time.sleep(0.02)
    (n,) = struct.unpack("<I", recv_all(s, 4))
    (got_v,) = struct.unpack("<f", recv_all(s, n))
    assert abs(got_v - reference(module, spatial, global_)[0]) < 1e-5
    s.close()


def test_model_id_mismatch_drops_the_connection(served):
    sock, model, _, _ = served
    s, model_id, _ = connect(sock, model)
    spatial, global_ = sample(3)
    body = struct.pack("<HHHH", model_id + 1, 0, H, W) + spatial.tobytes() + global_.tobytes()
    s.sendall(struct.pack("<I", len(body)) + body)
    assert s.recv(4) == b""  # clean EOF, not a response
    s.close()


def test_bad_magic_is_rejected_with_a_status(served):
    sock, _, _, _ = served
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock)
    s.sendall(struct.pack("<4sHHHH", b"NOPE", ns.PROTOCOL_VERSION, SPATIAL_CHANNELS, GLOBAL_FEATURES, 0))
    status, *_rest, n = struct.unpack("<HHHHI", recv_all(s, 12))
    recv_all(s, n)
    assert status == ns.STATUS_BAD_MAGIC
    assert s.recv(1) == b""
    s.close()


def test_concurrent_clients_are_batched_and_batch_invariant(served):
    sock, model, module, server = served
    n_clients, per_client = 6, 25
    errors: list[str] = []
    batches0, samples0 = server.stats.batches, server.stats.samples

    def client(i: int):
        try:
            s, model_id, _ = connect(sock, model)
            for k in range(per_client):
                spatial, global_ = sample(100 + i * per_client + k)
                got_v, got_p = request(s, model_id, spatial, global_, want_policy=(k % 5 == 0))
                v, p = reference(module, spatial, global_)
                if abs(got_v - v) > 1e-5:
                    errors.append(f"client {i} req {k}: value {got_v} vs {v}")
                if got_p is not None and np.abs(got_p - p).max() > 1e-5:
                    errors.append(f"client {i} req {k}: policy differs")
            s.close()
        except Exception as e:  # pragma: no cover
            errors.append(f"client {i}: {e!r}")

    threads = [threading.Thread(target=client, args=(i,)) for i in range(n_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors[:5]
    # The point of the server: with several clients in flight some
    # requests share a batch, so fewer batches than samples.
    assert server.stats.samples - samples0 >= n_clients * per_client
    assert server.stats.batches - batches0 < n_clients * per_client
