#!/usr/bin/env python3
"""Plan 064: is the lone mover's start-dependence the frontier effect? If the square the player
wants is out of reach from every start, the consistent move is "as far toward it as possible", and
destinations differ by construction. Per agent: share of moves of the full move fan's length (the
longest move any square in its fan allowed), the value's best square's distance from the start, and
how often the value's best square is the same for every start of a scenario.

    scripts/lone_mover_frontier.py runs/exp075/lone_mover.TAG.jsonl [...]
"""
import json
import sys
from collections import defaultdict


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    print(path)
    by_scn = defaultdict(list)
    for r in rows:
        by_scn[r["scenario"]].append(r)
    same_vbest = sum(len({tuple(r["value_best"]) for r in rs}) == 1 for rs in by_scn.values())
    vb_len = [cheb(tuple(r["value_best"]), tuple(r["start"])) for r in rows if r["agent"] == "policy"]
    print(f"  value's best square identical from every start: {same_vbest}/{len(by_scn)} scenarios; "
          f"its distance from the start: mean {sum(vb_len) / len(vb_len):.2f}")
    for a in sorted({r["agent"] for r in rows}):
        rs = [r for r in rows if r["agent"] == a and not r["fell"]]
        length = [cheb(tuple(r["last"]), tuple(r["start"])) for r in rs]
        ge6 = sum(l >= 6 for l in length) / len(length)
        ge7 = sum(l >= 7 for l in length) / len(length)
        print(f"  {a:<12} move length mean {sum(length) / len(length):.2f}; >= 6 squares {ge6:.1%}, "
              f">= 7 (needs a GFI at MA 6) {ge7:.1%}")

    # The hypothesis with the frontier taken out: among the starts from which the agent's modal
    # destination is within 5 squares (reachable with a square to spare, no GFI at MA 6), how many
    # go there, and how many GFI'd although the mode was that close?
    print("  -- starts with the agent's mode <= 5 squares away")
    for a in sorted({r["agent"] for r in rows}):
        by = defaultdict(list)
        for r in rows:
            if r["agent"] == a and not r["fell"]:
                by[r["scenario"]].append(r)
        n = k = gfi = 0
        for rs in by.values():
            cnt = defaultdict(int)
            for r in rs:
                cnt[tuple(r["last"])] += 1
            mode = max(cnt, key=cnt.get)
            for r in rs:
                if cheb(mode, tuple(r["start"])) <= 5:
                    n += 1
                    k += tuple(r["last"]) == mode
                    gfi += cheb(tuple(r["last"]), tuple(r["start"])) >= 7
        print(f"  {a:<12} n {n:>4}  at the mode {k / max(n, 1):6.1%}  GFI'd instead {gfi / max(n, 1):6.1%}")
    print(f"  fell (failed GFI): {sum(r['fell'] for r in rows)} of {len(rows)} activations")
