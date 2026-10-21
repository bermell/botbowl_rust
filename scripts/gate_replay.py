#!/usr/bin/env python3
"""Plan 068: replay a drive rung's per-game lines in pair order and run both sequential tests.

Reads `eval.games.jsonl` (lines from `eval` / `job eval`, gated: they carry `path`), pairs games
2k/2k+1, splits identical pairs (equal move fingerprints) from divergent ones, and replays

  - the all-pairs SPRT (today's drive rung), and
  - the divergent-pairs SPRT (`--drive-gate sprt`),

with `botbowl_play::stats::Sprt`'s normalised LLR and its 8-pair guard, reporting where each first
crossed a bound in pairs played (all kinds) and in searches (sum of both bots' would be twice the
candidate's; the candidate's `telemetry.searches` is what the lines carry).

    scripts/gate_replay.py runs/exp079/val_null/eval.games.jsonl [--sprt 0.5:0.55]
"""
import argparse
import json
import math
from collections import defaultdict

MIN_PAIRS = 8


def half_points(line):
    me, them = (line["home_score"], line["away_score"]) if line["candidate_team"] == "Home" else (line["away_score"], line["home_score"])
    return 2 if me > them else (1 if me == them else 0)


class Penta:
    def __init__(self):
        self.c = [0] * 5

    def add(self, k):
        self.c[k] += 1

    def n(self):
        return sum(self.c)

    def mean(self):
        n = self.n()
        return sum(i / 4 * c for i, c in enumerate(self.c)) / n if n else 0.5

    def var(self):
        n = self.n()
        if not n:
            return 0.0
        m = self.mean()
        return sum((i / 4 - m) ** 2 * c for i, c in enumerate(self.c)) / n

    def se(self):
        n = self.n()
        return math.sqrt(self.var() / n) if n >= 2 else 0.0


def llr(p, s0, s1):
    n = p.n()
    if not n:
        return 0.0
    v = p.var()
    if v <= 0:
        s = (s0 + s1) / 2
        v = s * (1 - s)
    return n * (s1 - s0) * (2 * p.mean() - s0 - s1) / (2 * v)


def verdict(p, s0, s1, a, b):
    if p.n() < MIN_PAIRS:
        return None
    lo, hi = math.log(b / (1 - a)), math.log((1 - b) / a)
    x = llr(p, s0, s1)
    return "H1" if x >= hi else ("H0" if x <= lo else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lines", nargs="+")
    ap.add_argument("--sprt", default="0.5:0.55")
    args = ap.parse_args()
    parts = [float(x) for x in args.sprt.split(":")]
    s0, s1 = parts[:2]
    a, b = (parts[2], parts[3]) if len(parts) == 4 else (0.05, 0.05)
    for path in args.lines:
        by_rung = defaultdict(lambda: defaultdict(dict))
        for l in open(path):
            if l.strip():
                try:
                    g = json.loads(l)
                except json.JSONDecodeError:
                    print(f"  (skipped a torn line: the run was stopped mid-write)")
                    continue
                by_rung[g["rung"]][g["game"] // 2][g["game"] % 2] = g
        for rung, by_pair in sorted(by_rung.items()):
            replay(f"{path} [{rung}]", by_pair, args.sprt, s0, s1, a, b)


def replay(name, by_pair, rule, s0, s1, a, b):
    if True:
        allp, divp = Penta(), Penta()
        ident = mism = unknown = 0
        searches = 0
        cross_all = cross_div = None
        played = 0
        for k in sorted(by_pair):
            pr = by_pair[k]
            if len(pr) < 2:
                continue
            x, y = pr[0], pr[1]
            played += 1
            hx, hy = half_points(x), half_points(y)
            allp.add(hx + hy)
            for g in (x, y):
                searches += ((g.get("telemetry") or {}).get("searches") or 0)
            px, py = x.get("path"), y.get("path")
            if px is not None and px == py:
                ident += 1
                mism += hx + hy != 2
            else:
                unknown += px is None or py is None
                divp.add(hx + hy)
            if cross_all is None and (v := verdict(allp, s0, s1, a, b)):
                cross_all = (v, played, searches)
            if cross_div is None and (v := verdict(divp, s0, s1, a, b)):
                cross_div = (v, played, searches)
        n = allp.n()
        pd = divp.n() / n if n else 0.0
        print(f"== {name}: {n} pairs, {ident} identical, {divp.n()} divergent"
              + (f", {mism} MISMATCHED" if mism else "") + (f", {unknown} without path" if unknown else ""))
        print(f"  p_div {pd:.3f} ± {math.sqrt(pd * (1 - pd) / n) if n else 0:.3f}   "
              f"s_div {divp.mean():.3f} ± {divp.se():.3f}   implied {0.5 + pd * (divp.mean() - 0.5):.3f}   "
              f"all-pairs {allp.mean():.3f} ± {allp.se():.3f}")
        print(f"  candidate searches {searches} ({searches / max(n, 1):.0f} per pair, both bots ~2x)")
        fmt = lambda c: f"{c[0]} after {c[1]} pairs ({c[2]} cand searches)" if c else f"undecided after {n} pairs (LLR {{}})"
        print(f"  SPRT {rule} all pairs:       " + (fmt(cross_all) if cross_all else f"undecided after {n} (LLR {llr(allp, s0, s1):.2f})"))
        print(f"  SPRT {rule} divergent pairs: " + (fmt(cross_div) if cross_div else f"undecided after {n} (LLR {llr(divp, s0, s1):.2f}, {divp.n()} divergent)"))


main()
