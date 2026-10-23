#!/usr/bin/env python3
"""Padding waste of the one-canvas batch, and what more canvases would buy (plan 069 item 4).

    python3 hpc/tune/canvas_waste.py SHARD.jsonl [SHARD.jsonl ...] [--canvas 13x22] [--max-canvases 4]

The in-process queue stages every board top-left onto one canvas (13x22 for the 20x11/7 capacity) so
every board size batches together; the convolutions then run over every canvas cell, padding
included. This reads the boards a corpus actually played (`meta.board_dims`, tensor size with the
border) weighted by decisions (forwards per decision are ~flat across boards: the search budget is
fixed), and reports per board the share of forwards and the padded fraction, then for 1..K canvases
the best set (brute force over the boards' own heights x widths, the full canvas always included):
the GPU cells per forward relative to one canvas, and each canvas's share of the forwards.

Read the result with two caveats. GPU time is proportional to cells only once batches are large
(an H100 at batch >= ~256); at small batches launch overhead dominates and padding is nearly free.
And k canvases mean k queues (or k bucket sets in one queue): each gets only its share of the
streams, so at a fixed stream count every batch is smaller — a canvas with 5% of the forwards needs
20x the streams for the same batch. Fast: a regex over each line, no JSON parse of the states.
"""

from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter
from pathlib import Path

DIMS = re.compile(rb'"board_dims":\{"width":(\d+),"height":(\d+)(?:,"team_size":(\d+))?\}')
DECISION = b'"chosen_action"'


def board_counts(paths: list[Path], max_lines: int = 0) -> Counter:
    """(h, w) tensor dims -> decisions, over every record of every shard."""
    out: Counter = Counter()
    for p in paths:
        with open(p, "rb") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                m = DIMS.search(line)
                if m:
                    out[(int(m.group(2)), int(m.group(1)))] += line.count(DECISION)
    return out


def best_sets(boards: Counter, full: tuple[int, int], k: int):
    """The k-canvas set (full canvas included) that minimises the weighted canvas cells."""
    hs = sorted({h for h, _ in boards} | {full[0]})
    ws = sorted({w for _, w in boards} | {full[1]})
    cands = [(h, w) for h in hs for w in ws if (h, w) != full and h <= full[0] and w <= full[1]]
    best = None
    for extra in itertools.combinations(cands, k - 1):
        canv = sorted([full, *extra], key=lambda c: c[0] * c[1])
        cost, share = 0, Counter()
        for (h, w), n in boards.items():
            c = next(c for c in canv if c[0] >= h and c[1] >= w)
            cost += n * c[0] * c[1]
            share[c] += n
        if best is None or cost < best[0]:
            best = (cost, canv, share)
    return best


def report(boards: Counter, full: tuple[int, int], kmax: int) -> str:
    total = sum(boards.values())
    used = sum(n * h * w for (h, w), n in boards.items())
    one = total * full[0] * full[1]
    lines = [f"Canvas {full[0]}x{full[1]}: {total} decisions over {len(boards)} boards; the real cells are "
             f"**{100 * used / one:.1f}%** of the canvas cells the GPU computes (padding {100 * (1 - used / one):.1f}%).", "",
             "| board (tensor HxW, border incl.) | share of forwards | cells | padded on the canvas |",
             "|---|---|---|---|"]
    for (h, w), n in sorted(boards.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {h}x{w} ({w - 2}x{h - 2} board) | {100 * n / total:.1f}% | {h * w} | "
                     f"{100 * (1 - h * w / (full[0] * full[1])):.0f}% |")
    lines += ["", "| canvases | GPU cells per forward vs one canvas | padding left | canvases (share of forwards) |",
              "|---|---|---|---|"]
    for k in range(1, kmax + 1):
        res = best_sets(boards, full, k)
        if not res:
            break
        cost, canv, share = res
        desc = ", ".join(f"{h}x{w} ({100 * share[(h, w)] / total:.0f}%)" for h, w in canv)
        lines.append(f"| {k} | {cost / one:.3f} | {100 * (1 - used / cost):.1f}% | {desc} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--canvas", default="13x22", help="HxW (tensor, border included)")
    ap.add_argument("--max-canvases", type=int, default=4)
    ap.add_argument("--max-lines", type=int, default=0, help="records per shard (0 = all)")
    args = ap.parse_args()
    h, w = (int(x) for x in args.canvas.lower().split("x"))
    boards = board_counts(args.shards, args.max_lines)
    if not boards:
        print("no board_dims found")
        return 1
    print(report(boards, (h, w), args.max_canvases), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
