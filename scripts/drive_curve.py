#!/usr/bin/env python3
"""Plan 051: a loop's per-generation SPRT results, one line per rung, oldest generation first.

    drive_curve.py runs/<run> [--sub drives|p1] [--last N]

Reads runs/<run>/genNN/<sub>/report.json (train_loop.sh's EVAL_VENUE=drives writes `drives`, and
`p1` for the full-game confirmation of a drive H1). Each point is the paired score with its SE,
the pairs it rests on and the SPRT verdict, so successive generations against the same reference
read side by side:

    contested_14x7@14x7/4: gen03 0.531±0.018 H1/206 · gen04 0.512±0.020 -/250(cap)
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def rung_key(row: dict) -> str:
    """`drives(contested_14x7)@14x7/4` for a drive rung, the board for a full-game rung."""
    m = re.search(r"drives\(([^)]*)\)@(\S+)", row["opponent"])
    if m:
        return f"{m.group(1)}@{m.group(2)}"
    return row.get("board") or "env"


def point(row: dict) -> str:
    s = row.get("sprt") or {}
    verdict = s.get("verdict", "")
    pairs = s.get("pairs") or sum((row.get("pairs") or {}).get("counts") or [0])
    tag = {"H1": "H1", "H0": "H0"}.get(verdict, "-")
    return f"{row['points']:.3f}±{row.get('points_se', float('nan')):.3f} {tag}/{pairs}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--sub", default="drives")
    p.add_argument("--last", type=int, default=8, help="generations per line")
    a = p.parse_args()

    series = defaultdict(list)
    for gen in sorted(Path(a.run_dir).glob("gen[0-9]*"), key=lambda d: int(d.name[3:])):
        rep = gen / a.sub / "report.json"
        if not rep.exists():
            continue
        for row in json.load(open(rep))["ladder"]:
            if row.get("games"):
                series[rung_key(row)].append(f"{gen.name} {point(row)}")
    if not series:
        print(f"no {a.sub} reports yet")
        return 0
    for key, pts in series.items():
        print(f"{key}: " + " · ".join(pts[-a.last:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
