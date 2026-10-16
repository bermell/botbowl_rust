#!/usr/bin/env python3
"""Plan 064: a probe's violations (`gain < -tau`) split by decision kind, per edit — dominance and
skills rows. `tau` is read from the summary next to the rows.

    scripts/probe_breakdown.py runs/exp075/dominance.TAG.jsonl [--edits remove_opp,knock_down_opp]
"""
import argparse
import json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("rows", nargs="+")
ap.add_argument("--edits", default="")
ap.add_argument("--classes", default="beneficial")
args = ap.parse_args()
want = set(e for e in args.edits.split(",") if e)
classes = set(args.classes.split(","))
for path in args.rows:
    tau = json.load(open(path.replace(".jsonl", ".summary.json")))["summary"]["tau"]
    cells = defaultdict(lambda: [0, 0])
    kinds = set()
    for line in open(path):
        r = json.loads(line)
        if "class" in r and r["class"] not in classes:
            continue
        e = r["edit"]
        if want and e not in want:
            continue
        key = (e, r["kind"], r.get("status", ""))
        kinds.add(r["kind"])
        cells[key][0] += 1
        cells[key][1] += r["gain"] < -tau
    print(f"{path}  tau {tau:.4f}   (n, share beyond tau)")
    for key in sorted(cells):
        n, k = cells[key]
        if n >= 20:
            print(f"  {key[0]:<16} {key[1]:<11} {key[2]:<8} n {n:>6}  {k / n:6.1%}")
