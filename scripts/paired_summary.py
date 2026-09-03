#!/usr/bin/env python3
"""Score a ladder rung by *seed pair* instead of by game (plan 027).

`botbowl-ui eval` already pairs its games: `eval.rs:329-330` sets
`candidate_team = Home if g % 2 == 0 else Away` and `seed = base + g / 2`, so
games `2i` and `2i+1` are the same situation played from both sides. That is
textbook common random numbers — and then the report throws it away, pooling
W/D/L over games as if they were independent.

That matters here because the side effect is large: Away wins ~57-62% of decided
games (see `side_bias_summary.py`), so a big share of any single game's outcome
is "which side did the candidate draw", not "is the candidate better". Averaging
the two games of a pair cancels that term exactly, which is the entire point of
having drawn them in the first place.

This prints both estimators on the same data so the difference is visible:

  unpaired  mean over games,      SE from the game-level spread
  paired    mean over seed pairs, SE from the pair-level spread

Both estimate the same quantity. The paired SE should be smaller; how much
smaller is exactly how much power the current summary is leaving on the table.

Usage:  scripts/paired_summary.py runs/exp-search/*.games.jsonl
"""
import collections
import glob
import json
import math
import os
import sys


def score(row):
    """Candidate's points in this game: 1 win, 0.5 draw, 0 loss."""
    h, a = row["home_score"], row["away_score"]
    cand, opp = (h, a) if row.get("candidate_team") == "Home" else (a, h)
    if cand > opp:
        return 1.0
    if cand < opp:
        return 0.0
    return 0.5


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    m = sum(xs) / n
    if n < 2:
        return (m, float("nan"), n)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return (m, math.sqrt(var / n), n)


def report(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return

    by_rung = collections.defaultdict(list)
    for r in rows:
        by_rung[r.get("rung", "?")].append(r)

    print(f"\n=== {os.path.basename(path)} ===")
    for rung, rs in by_rung.items():
        games = [score(r) for r in rs]
        gm, gse, gn = mean_se(games)

        pairs = collections.defaultdict(list)
        for r in rs:
            pairs[r["seed"]].append(r)
        # Only seeds played from both sides carry the cancellation; a lone
        # game is exactly the side-contaminated observation we are removing.
        full = [v for v in pairs.values() if len(v) == 2 and {x.get("candidate_team") for x in v} == {"Home", "Away"}]
        pscores = [sum(score(x) for x in v) / 2 for v in full]
        pm, pse, pn = mean_se(pscores)
        dropped = len(pairs) - len(full)

        print(f"  rung {rung[:40]}")
        print(f"    unpaired  {gm:.3f} +/- {gse:.3f}   (n={gn} games)")
        if pn:
            print(f"    paired    {pm:.3f} +/- {pse:.3f}   (n={pn} pairs)"
                  + (f"   [{dropped} incomplete pair(s) dropped]" if dropped else ""))
            if gse and pse and not math.isnan(pse) and pse > 0:
                print(f"    -> paired SE is {gse/pse:.2f}x tighter"
                      f"; equivalent to {((gse/pse)**2 - 1)*100:+.0f}% more games")
            # A pair scoring 0.5 means the candidate won one side and lost the
            # other: pure side effect, zero evidence either way about skill.
            split = sum(1 for p in pscores if p == 0.5)
            print(f"    pairs split 1-1 (side decided it): {split}/{pn} = {split/pn:.0%}")
        else:
            print("    paired    (no complete pairs)")


def main():
    pats = sys.argv[1:] or ["runs/exp-search/*.games.jsonl"]
    files = [f for p in pats for f in sorted(glob.glob(p))]
    if not files:
        print("no per-game files found — arms need --per-game-out")
        return
    for f in files:
        report(f)


if __name__ == "__main__":
    main()
