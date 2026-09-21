"""Carry trained weights across encoder/architecture schema bumps, one step at a time.

    uv run python -m bbnn.migrate models/bbnet_14x7_genNN.pt --out models/bbnet_v7.pt [--onnx models/bbnet_v7.onnx]
    uv run python -m bbnn.migrate --list

A checkpoint is a bare ``state_dict`` (``nn_server.py`` loads it directly), so
its schema is *detected from tensor shapes* (`detect_schema`) rather than
stored. `migrate` then applies every registered `Migration` from that version
up to ``--to`` (default: the schema ``model.py`` is at), in order, like a chain
of git commits — each step is a pure function of the state_dict before it and
is verified in isolation, so a step never has to know about any other.

Every step is **function-preserving** where the architecture allows it: new
input channels/features are inserted with zero weights, so the migrated net
computes exactly what the old one did until training moves those weights.
That is what makes a champion usable across a schema bump without a from-
scratch retrain. The generic verifier (`Migration.verify`) proves it by
running both nets on random inputs and comparing outputs; a step that cannot
be exactly preserving must say so with ``exact=False`` and its own check.

Adding a schema bump: bump `SCHEMAS`, write one `Migration` (usually a few
lines with the helpers below), register it, and `test_migrate.py` checks the
chain is contiguous and every step verifies.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .model import GLOBAL_FEATURES, SPATIAL_CHANNELS, BBNet

# ---------------------------------------------------------------------------
# Schema registry: what the encoder emitted at each nn_schema_version.
# Mirrors botbowl-nn/src/bin/prepare.rs's version comments. Only versions whose
# checkpoints still exist need an entry.

SCHEMAS: dict[int, dict] = {
    6: {"C": 59, "F": 15},
    7: {"C": 61, "F": 18},
}
CURRENT = max(v for v, s in SCHEMAS.items() if (s["C"], s["F"]) == (SPATIAL_CHANNELS, GLOBAL_FEATURES))

StateDict = dict[str, torch.Tensor]


def detect_schema(sd: StateDict) -> int:
    """The schema a state_dict was trained at, from its input-side shapes.

    ``stem.weight`` is ``[width, C + global_embed, 3, 3]`` and
    ``global_fc.weight`` is ``[global_embed, F]``, so ``C`` and ``F`` fall out
    exactly; an unknown ``(C, F)`` pair is an error, never a guess.
    """
    embed = sd["global_fc.weight"].shape[0]
    f = sd["global_fc.weight"].shape[1]
    c = sd["stem.weight"].shape[1] - embed
    for v, s in SCHEMAS.items():
        if (s["C"], s["F"]) == (c, f):
            return v
    raise ValueError(f"no known schema has C={c} F={f} (known: {SCHEMAS})")


def shape_of(sd: StateDict) -> dict:
    """Constructor kwargs implied by a state_dict, any schema."""
    embed = sd["global_fc.weight"].shape[0]
    return {
        **BBNet.shape_of(sd),
        "global_embed": int(embed),
        "spatial_ch": int(sd["stem.weight"].shape[1] - embed),
        "global_f": int(sd["global_fc.weight"].shape[1]),
        "value_hidden": int(sd["value_fc1.weight"].shape[0]),
    }


# ---------------------------------------------------------------------------
# Generic surgeries. Each returns a new tensor; nothing is modified in place.


def insert_zero_slices(t: torch.Tensor, dim: int, at: int, n: int) -> torch.Tensor:
    """Insert ``n`` zero slices into ``t`` along ``dim`` before index ``at``.

    On a weight's *input* dimension this is the function-preserving way to
    add inputs: the old inputs keep their columns, the new ones contribute 0.
    """
    if n <= 0:
        return t.clone()
    shape = list(t.shape)
    shape[dim] = n
    zeros = t.new_zeros(shape)
    head, tail = t.narrow(dim, 0, at), t.narrow(dim, at, t.shape[dim] - at)
    return torch.cat([head, zeros, tail], dim=dim)


def insert_zero_inputs(sd: StateDict, key: str, at: int, n: int) -> StateDict:
    """`insert_zero_slices` on the input dim (1) of ``sd[key]``."""
    out = dict(sd)
    out[key] = insert_zero_slices(sd[key], 1, at, n)
    return out


def embed_inputs(spatial: torch.Tensor, global_: torch.Tensor, c_at: int, c_n: int, f_at: int, f_n: int):
    """Lift an old-schema input into the new one with zeros in the inserted
    slots — the input-side twin of `insert_zero_inputs`, used to verify."""
    s = insert_zero_slices(spatial, 1, c_at, c_n) if c_n else spatial
    g = insert_zero_slices(global_, 1, f_at, f_n) if f_n else global_
    return s, g


# ---------------------------------------------------------------------------
# Migrations


@dataclass(frozen=True)
class Migration:
    """One schema step. ``apply`` maps a ``src``-schema state_dict to a
    ``dst``-schema one. ``embed`` maps a ``src`` input to the ``dst`` input the
    migrated net should agree on (defaults to the identity, i.e. shapes did
    not change). ``exact`` says the step is function-preserving and the
    generic verifier should demand equality."""

    src: int
    dst: int
    description: str
    apply: Callable[[StateDict], StateDict]
    embed: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
    exact: bool = True

    def verify(
        self, old: StateDict, new: StateDict, trials: int = 4, atol: float = 1e-5, rtol: float = 1e-5, seed: int = 0
    ) -> float:
        """Build both nets, run them on random inputs, return the max abs
        output difference. Raises when ``exact`` and it exceeds
        ``atol + rtol * max|output|``.

        The nets are in eval mode with the checkpoint's own BatchNorm stats,
        which is how tract and the sidecar run them.

        **The bound has to scale with the outputs.** A zero-column insertion is
        exact in arithmetic — in float64 the two nets agree bit-for-bit — but in
        float32 a wider conv accumulates in a different order, so the difference
        grows with the magnitude of what is being accumulated. A freshly
        initialised net's logits are O(1) and a fixed 1e-5 held; a trained
        champion's reach ±140, where the same float32 noise is 5e-5. Judging
        that absolutely would reject exactly the checkpoints this module exists
        to carry, while a genuinely non-preserving step is off by O(1) and is
        still caught by a mile.
        """
        a = BBNet(**shape_of(old))
        a.load_state_dict(old)
        b = BBNet(**shape_of(new))
        b.load_state_dict(new)
        a.eval()
        b.eval()
        g = torch.Generator().manual_seed(seed)
        worst = 0.0
        scale = 0.0
        sa, fa = SCHEMAS[self.src]["C"], SCHEMAS[self.src]["F"]
        with torch.no_grad():
            for (h, w) in [(9, 16), (7, 14), (11, 18)][:trials]:
                spatial = torch.rand(2, sa, h, w, generator=g)
                global_ = torch.rand(2, fa, generator=g)
                pa, va = a(spatial, global_)
                s2, g2 = self.embed(spatial, global_) if self.embed else (spatial, global_)
                pb, vb = b(s2, g2)
                worst = max(worst, (pa - pb).abs().max().item(), (va - vb).abs().max().item())
                scale = max(scale, pa.abs().max().item(), va.abs().max().item())
        bound = atol + rtol * scale
        if self.exact and worst > bound:
            raise AssertionError(
                f"v{self.src}->v{self.dst} is not function-preserving: "
                f"max |diff| {worst:.3e} > {bound:.3e} (max |out| {scale:.3g})"
            )
        return worst


def _v6_to_v7(sd: StateDict) -> StateDict:
    """Plan 042 / schema v7: two geometry planes appended to the spatial
    channels (59, 60) and three size globals appended to the features
    (15..17). The stem sees ``[spatial(C) | global_embed]``, so the new planes
    go in *before* the embedding columns, not at the end of the tensor."""
    c6, f6 = SCHEMAS[6]["C"], SCHEMAS[6]["F"]
    sd = insert_zero_inputs(sd, "stem.weight", at=c6, n=SCHEMAS[7]["C"] - c6)
    sd = insert_zero_inputs(sd, "global_fc.weight", at=f6, n=SCHEMAS[7]["F"] - f6)
    return sd


def _embed_6_to_7(spatial, global_):
    return embed_inputs(spatial, global_, SCHEMAS[6]["C"], 2, SCHEMAS[6]["F"], 3)


MIGRATIONS: list[Migration] = [
    Migration(
        6,
        7,
        "geometry planes dist_to_us_endzone/dist_to_sideline (C 59->61) and "
        "playable_w/playable_h/team_size globals (F 15->18), zero-initialised",
        _v6_to_v7,
        _embed_6_to_7,
    ),
]


def path_to(src: int, dst: int) -> list[Migration]:
    """The chain of steps from ``src`` to ``dst``, or raise if it is broken."""
    steps = []
    v = src
    by_src = {m.src: m for m in MIGRATIONS}
    while v != dst:
        if v > dst:
            raise ValueError(f"cannot migrate downwards, v{src} -> v{dst}")
        m = by_src.get(v)
        if m is None:
            raise ValueError(f"no migration registered from v{v} (have {sorted(by_src)})")
        steps.append(m)
        v = m.dst
    return steps


def migrate(sd: StateDict, to: int | None = None, verify: bool = True, log=print) -> tuple[StateDict, int, int]:
    """Apply every step from the detected schema to ``to``. Returns
    ``(state_dict, from_version, to_version)``. Idempotent at the target."""
    to = CURRENT if to is None else to
    src = detect_schema(sd)
    cur = {k: v.detach().clone() for k, v in sd.items()}
    for m in path_to(src, to):
        nxt = m.apply(cur)
        if verify:
            worst = m.verify(cur, nxt)
            log(f"v{m.src} -> v{m.dst}: {m.description}  [verified, max |diff| {worst:.1e}]")
        else:
            log(f"v{m.src} -> v{m.dst}: {m.description}")
        cur = nxt
    if detect_schema(cur) != to:
        raise AssertionError(f"chain ended at v{detect_schema(cur)}, wanted v{to}")
    return cur, src, to


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Migrate a trained .pt across schema bumps, one step at a time.")
    ap.add_argument("checkpoint", nargs="?", type=Path, help="bare state_dict .pt")
    ap.add_argument("--out", type=Path, help="write the migrated state_dict here (bare, sidecar-loadable)")
    ap.add_argument("--onnx", type=Path, help="also export the migrated net to ONNX")
    ap.add_argument("--to", type=int, default=None, help=f"target schema (default: current, v{CURRENT})")
    ap.add_argument("--no-verify", action="store_true", help="skip the forward-equivalence check per step")
    ap.add_argument("--list", action="store_true", help="print the registered migrations and exit")
    a = ap.parse_args(argv)

    if a.list:
        print(f"schemas: {SCHEMAS}  (model.py is at v{CURRENT})")
        for m in MIGRATIONS:
            print(f"  v{m.src} -> v{m.dst}{'' if m.exact else '  [not exact]'}: {m.description}")
        return 0
    if a.checkpoint is None:
        ap.error("checkpoint is required (or --list)")

    sd = torch.load(a.checkpoint, map_location="cpu")
    if not isinstance(sd, dict) or "stem.weight" not in sd:
        print(f"{a.checkpoint}: not a bare BBNet state_dict", file=sys.stderr)
        return 2
    out, src, dst = migrate(sd, a.to, verify=not a.no_verify)
    if src == dst:
        print(f"{a.checkpoint}: already at v{dst}, nothing to do")
    else:
        print(f"{a.checkpoint}: v{src} -> v{dst} ({len(path_to(src, dst))} step(s))")
    if a.out:
        torch.save(out, a.out)
        print(f"saved weights -> {a.out}")
    if a.onnx:
        from .export import export_onnx

        model = BBNet(**shape_of(out))
        model.load_state_dict(out)
        export_onnx(model, a.onnx)
        print(f"exported ONNX -> {a.onnx}")
    if not a.out and not a.onnx and src != dst:
        print("(dry run: pass --out and/or --onnx to write)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
