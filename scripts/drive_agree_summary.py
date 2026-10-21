#!/usr/bin/env python3
"""Plan 068: summarise `botbowl-ui drive-agree` rows (one JSONL file per bot pair and set).

For each file:
  - full search: share of positions whose drive never diverges (the pair is one game, exactly 1/2),
    the step of the first divergence, and the per-decision agreement on the shared prefix
    (searched decisions, fan > 1: 1 - divergences / decisions — the hazard of a divergence);
  - bare-policy agreement at the same decisions, and how well it predicts search agreement;
  - each prescreen (a cheap lockstep from the same position and dice): its never-diverge rate and
    the cross-table against the full search's;
  - with --follow rows: agreement over every searched decision of the drive.

    scripts/drive_agree_summary.py runs/exp079/agree_*.jsonl
"""
import json
import math
import sys


def se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def pct(k, n):
    return f"{k / n:.3f} ± {se(k / n, n):.3f} ({k}/{n})" if n else "-"


def summarise(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print(f"{path}: empty")
        return
    r0 = rows[0]
    short = lambda m: m.rsplit("/", 1)[-1].replace(".onnx", "").replace("bbnet_mix16x9v9_", "")
    print(f"== {path}: {short(r0['candidate'])} vs {short(r0['opponent'])} on {r0['set']} ({r0['board']}), {len(rows)} positions")
    full = [r for r in rows if r.get("full")]
    if full:
        never = sum(1 for r in full if r["full"]["divergence"] is None)
        ended = sum(1 for r in full if r["full"]["divergence"] is None and r["full"]["ended"])
        print(f"  full search: never diverges {pct(never, len(full))}  (of which the drive ended: {ended})")
        divs = sorted(r["full"]["divergence"] for r in full if r["full"]["divergence"] is not None)
        if divs:
            print(f"    first divergence at step: median {divs[len(divs) // 2]}, quartiles {divs[len(divs) // 4]}-{divs[3 * len(divs) // 4]}, max {divs[-1]}")
        # Searched decisions on the shared prefix, the divergent one included.
        pre = [d for r in full for d in r["decisions"] if d[7] == 0 and d[2] > 1]
        n_dis = sum(1 for d in pre if d[3] == 0)
        print(f"    per searched decision on the prefix: agree {pct(len(pre) - n_dis, len(pre))}")
        first = [next((d for d in r["decisions"] if d[2] > 1), None) for r in full]
        first = [d for d in first if d is not None]
        print(f"    first searched decision disagrees: {pct(sum(1 for d in first if d[3] == 0), len(first))}")
        # Policy argmax agreement, and what it predicts.
        pa = [d for d in pre if d[4] == 1]
        pd = [d for d in pre if d[4] == 0]
        print(f"    bare-policy argmax agrees: {pct(len(pa), len(pre))}")
        print(f"      search disagrees | policy agrees:    {pct(sum(1 for d in pa if d[3] == 0), len(pa))}")
        print(f"      search disagrees | policy disagrees: {pct(sum(1 for d in pd if d[3] == 0), len(pd))}")
        own_a = sum(d[5] for d in pre)
        own_b = sum(d[6] for d in pre)
        print(f"    search plays its own policy argmax: cand {pct(own_a, len(pre))}, opp {pct(own_b, len(pre))}")
        fol = [d for r in full for d in r["decisions"] if d[2] > 1]
        if any(d[7] for r in full for d in r["decisions"]):
            n_dis = sum(1 for d in fol if d[3] == 0)
            print(f"    every searched decision (--follow): agree {pct(len(fol) - n_dis, len(fol))}; policy agrees {pct(sum(d[4] for d in fol), len(fol))}")
        secs = [r["full"]["secs"] for r in full]
        steps = sum(r["full"]["prefix_steps"] + r["full"]["followed_steps"] for r in full)
        searched = sum(1 for r in full for d in r["decisions"] if d[2] > 1)
        print(f"    time: {sum(secs) / len(secs):.0f} s per position, {sum(secs) / max(searched, 1):.2f} s per searched decision (2 searches), {steps} steps")
    kinds = sorted({p["iters"] for r in rows for p in r["prescreen"]})
    for it in kinds:
        name = "policy" if it == 0 else f"{it} descents"
        ps = [(r, next(p for p in r["prescreen"] if p["iters"] == it)) for r in rows if any(p["iters"] == it for p in r["prescreen"])]
        never = sum(1 for _, p in ps if p["divergence"] is None)
        secs = sum(p["secs"] for _, p in ps) / len(ps)
        print(f"  prescreen {name}: never diverges {pct(never, len(ps))}, {secs:.1f} s per position")
        both = [(r, p) for r, p in ps if r.get("full")]
        if both:
            t = [[0, 0], [0, 0]]  # t[pre_never][full_never]
            for r, p in both:
                t[p["divergence"] is None][r["full"]["divergence"] is None] += 1
            print(f"    cross-table (rows prescreen, cols full): never/never {t[1][1]}  never/div {t[1][0]}  div/never {t[0][1]}  div/div {t[0][0]}")
            skip = t[1][1] + t[1][0]
            if skip:
                print(f"    skipping prescreen-never positions would drop {skip}/{len(both)}, of which {t[1][0]} full-divergent (lost information)")


for f in sys.argv[1:]:
    summarise(f)
