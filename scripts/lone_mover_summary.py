#!/usr/bin/env python3
"""Plan 064: the lone-mover rows in more detail — how far from the agent's modal destination each
start ended (0 / 1 / 2+ squares), how far the destination is from the start, and where it lies
relative to the ball, per agent.

    scripts/lone_mover_summary.py runs/exp075/lone_mover.TAG.jsonl [...]
"""
import json
import sys
from collections import Counter, defaultdict


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["agent"], r["scenario"])].append(r)
    agents = sorted({r["agent"] for r in rows})
    print(path)
    for a in agents:
        dist = Counter()
        travel = []
        for (ag, _), rs in by.items():
            if ag != a:
                continue
            ok = [r for r in rs if not r["fell"]]
            mode = Counter(tuple(r["last"]) for r in ok).most_common(1)[0][0]
            for r in ok:
                d = cheb(tuple(r["last"]), mode)
                dist[min(d, 2)] += 1
                travel.append(cheb(tuple(r["last"]), tuple(r["start"])))
        n = sum(dist.values())
        print(f"  {a:<12} starts {n:>4}  at mode {dist[0] / n:6.1%}  1 square off {dist[1] / n:6.1%}  "
              f"2+ off {dist[2] / n:6.1%}   mean move length {sum(travel) / len(travel):.2f}")
