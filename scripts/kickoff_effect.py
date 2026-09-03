#!/usr/bin/env python3
"""Decompose the Home/Away bias into a coin effect and a last-turn effect.

`side_bias_summary.py` establishes *that* Away wins more. This says *why*, from
the per-game rows written by `botbowl-ui eval --per-game-out`, which record the
seed, the candidate's side, side-relative scores, and `kicking_first_half`.

Three questions, in order:

1. **Is the coin fair?** `kicking_first_half` should be Home half the time. The
   toss winner always picks Receive (`scripted.rs:52`), so the kicking team is
   the toss *loser*. A skew here means the bias is upstream of any strategy.

2. **Does kicking win games?** `game_procs.rs:90` gives the receiving team the
   first turn of each round, so the kicking team takes the *last* turn of each
   half. If the last word matters, kickers out-win receivers regardless of side.

3. **Is there a residual side effect?** After conditioning on who kicked, does
   Away still win more? That would be a genuine Home/Away asymmetry in the
   board, setup, or bot — not an artefact of the toss.

Usage:  scripts/kickoff_effect.py runs/exp-search/*.games.jsonl
"""
import collections
import glob
import json
import math
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rate(label, k, n):
    if not n:
        print(f"  {label:34} (no decided games)")
        return
    p = k / n
    lo, hi = wilson(k, n)
    z = (p - 0.5) / math.sqrt(0.25 / n)
    print(f"  {label:34} {p:6.1%}  95% CI [{lo:5.1%}, {hi:5.1%}]  n={n:4}  z={z:+.2f}")


def main():
    pats = sys.argv[1:] or ["runs/exp-search/*.games.jsonl"]
    files = [f for p in pats for f in sorted(glob.glob(p))]
    rows = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print("no per-game rows found (arms may not have run yet)")
        return

    print(f"{len(rows)} games from {len(files)} file(s)\n")

    # 1. coin fairness
    kick = collections.Counter(r.get("kicking_first_half") for r in rows)
    n = sum(kick.values())
    print("1. Coin fairness — who kicked in half 1 (toss loser kicks):")
    for team, c in sorted(kick.items(), key=lambda kv: str(kv[0])):
        print(f"     {str(team):6} {c:4}  ({c/n:.1%})")
    rate("Home kicks first half", kick.get("Home", 0), n)

    decided = [r for r in rows if r.get("home_score") != r.get("away_score")]
    print(f"\n   ({len(rows) - len(decided)} draws excluded from win rates below)")

    def home_won(r):
        return r["home_score"] > r["away_score"]

    # 2. last-turn effect: does the kicking team win more?
    print("\n2. Last-turn effect — the kicking team takes the last turn of each half:")
    kick_wins = sum(
        1 for r in decided
        if (r.get("kicking_first_half") == "Home") == home_won(r)
    )
    rate("kicking team win rate", kick_wins, len(decided))

    # 3. residual side effect, overall and conditioned on who kicked
    print("\n3. Side effect — Away win rate, overall and conditioned on the toss:")
    away_all = sum(1 for r in decided if not home_won(r))
    rate("Away (all games)", away_all, len(decided))
    for who in ("Home", "Away"):
        sub = [r for r in decided if r.get("kicking_first_half") == who]
        aw = sum(1 for r in sub if not home_won(r))
        rate(f"Away | {who} kicked first half", aw, len(sub))

    print("\nReading it: a skew in (1) means the toss is the problem. A high rate in")
    print("(2) with (3) flat means the last turn is worth having and the sides are")
    print("fine. A residual skew in (3) inside both toss branches means a genuine")
    print("Home/Away asymmetry that the coin cannot explain.")


if __name__ == "__main__":
    main()
