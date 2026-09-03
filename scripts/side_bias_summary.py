#!/usr/bin/env python3
"""Pool the Home/Away side split across every eval report on disk (plan 027).

A candidate's win rate cannot see a side bias: seats alternate Home/Away, so a
side advantage cancels out of the headline number. That is why this went
unnoticed behind plan 021's "mirror anomaly", which is a *seat* statistic and a
different effect. Every LadderRow carries the per-side breakdown needed to
recover it:

    home-side wins = wins_as_home + losses_as_away
    away-side wins = wins_as_away + losses_as_home

Lopsided rungs carry no side information — if the candidate beats `random`
30-0 the split is exactly 15/15 by construction, not by balance — so rows
where one seat took everything are excluded from the pooled figure.

Usage:  scripts/side_bias_summary.py [glob ...]
"""
import glob
import json
import math
import os
import sys

DEFAULT_GLOBS = [
    "runs/exp-search/*.json",
    "runs/exp-priors/*.json",
    "runs/loop14x7/gen*/report.json",
]


def rows(patterns):
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            try:
                doc = json.load(open(path))
            except Exception:
                continue
            if doc.get("_skipped"):
                continue
            for r in doc.get("ladder") or []:
                n = r.get("games") or 0
                if not n:
                    continue
                wh, lh = r.get("wins_as_home", 0), r.get("losses_as_home", 0)
                wa, la = r.get("wins_as_away", 0), r.get("losses_as_away", 0)
                yield {
                    "run": os.path.basename(os.path.dirname(path)),
                    "arm": os.path.basename(path)[:-5],
                    "opp": r.get("opponent", "?")[:30],
                    "n": n,
                    "home_w": wh + la,
                    "away_w": wa + lh,
                    "td_h": r.get("tds_by_home"),
                    "td_a": r.get("tds_by_away"),
                    # A shutout carries no side signal (see module docstring).
                    "informative": not (wh + wa == 0 or lh + la == 0),
                }


def wilson(k, n, z=1.96):
    """Wilson score interval — behaves at the extremes where normal-approx doesn't."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    pats = sys.argv[1:] or DEFAULT_GLOBS
    data = list(rows(pats))
    if not data:
        print("no reports found")
        return
    print(f"{'run':14} {'arm':22} {'opponent':30} {'N':>4} {'Hw':>4} {'Aw':>4} {'Away%':>6}  note")
    for d in data:
        dec = d["home_w"] + d["away_w"]
        pct = f"{d['away_w']/dec:.0%}" if dec else "-"
        note = "" if d["informative"] else "shutout - no side signal"
        print(
            f"{d['run']:14} {d['arm']:22} {d['opp']:30} {d['n']:4} "
            f"{d['home_w']:4} {d['away_w']:4} {pct:>6}  {note}"
        )

    for label, subset in (
        ("ALL rows", data),
        ("informative rows only", [d for d in data if d["informative"]]),
        ("mirror-like arms (exp-* only)", [d for d in data if d["informative"] and d["run"].startswith("exp-")]),
    ):
        h = sum(d["home_w"] for d in subset)
        a = sum(d["away_w"] for d in subset)
        n = h + a
        if not n:
            continue
        p = a / n
        se = math.sqrt(0.25 / n)
        z = (p - 0.5) / se
        lo, hi = wilson(a, n)
        th = sum(d["td_h"] or 0 for d in subset)
        ta = sum(d["td_a"] or 0 for d in subset)
        print(f"\n{label}: Home {h}  Away {a}  (n={n} decided)")
        print(f"  Away share {p:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  z={z:+.2f}")
        if th + ta:
            print(f"  TDs: Home {th}  Away {ta}  -> Away {ta/(th+ta):.1%}")


if __name__ == "__main__":
    main()
