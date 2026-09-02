#!/usr/bin/env python3
"""One-line summary of an `botbowl-ui eval` report JSON, plus the promotion gate.

    eval_summary.py report.json               # print summary line
    eval_summary.py report.json --gate 0.55   # also exit 0 if the vs: rung
                                              # scores >= gate, 2 if below,
                                              # 3 if the report has no vs: rung

# The gate scores points, not wins (changed 2026-09-02)

`LadderRow.win_rate` is `wins / games`, which counts a **draw exactly as a
loss**. That is not a strict standard, it is a biased one, and it is biased
hardest in the one place the gate is applied — the head-to-head against a
champion of near-equal strength, where draws are most common.

Measured on gen01: 0% draws vs random, 10-13% vs the fixed rungs, **23% vs
the champion**. Two consequences of scoring that with `wins/games`:

* Two *identical* nets score `(1 - draw_rate) / 2` = **0.383**, not 0.50.
* Clearing 0.55 at a 23% draw rate needs 16.5 wins to 6.5 losses — ~72% of
  decided games, roughly **160 Elo per generation**. AlphaZero-style loops
  gate near 55% in *points*, which is about 35 Elo.

So the gate now uses the standard score `(W + D/2) / N`. gen01 measured
W15 D7 L8: 0.50 by wins (REJECTED), **0.617 by points**. Note the evidence
is weak either way — 15-8 on decided games is p = 0.21 — so this threshold
is a decision rule, not a significance test. Both numbers are printed, so
any verdict can be re-derived from the line.
"""
import argparse
import json
import sys


def points(row: dict) -> float:
    """Standard score `(W + D/2) / N` — a draw is half a win, not a loss."""
    n = row["games"]
    if not n:
        return 0.0
    return (row["wins"] + 0.5 * row["draws"]) / n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("--gate", type=float, default=None)
    a = p.parse_args()

    with open(a.report) as f:
        r = json.load(f)

    parts = []
    vs_row = None
    for row in r["ladder"]:
        # Home-*side* wins are wins_as_home + losses_as_away: the two home/away
        # pairs are the same games counted from the candidate's perspective,
        # not two independent measurements (plan 023).
        home_side = row["wins_as_home"] + row["losses_as_away"]
        away_side = row["losses_as_home"] + row["wins_as_away"]
        side = ""
        if "tds_by_home" in row:
            side = f", side H{home_side}-{away_side} TD {row['tds_by_home']}:{row['tds_by_away']}"
        # `pts` is what the gate reads; `win_rate` is kept in the line so a
        # verdict stays re-derivable and old status lines stay comparable.
        parts.append(
            f"{row['opponent']} pts {points(row):.3f} (w {row['win_rate']:.2f}) "
            f"(W{row['wins']} D{row['draws']} L{row['losses']} "
            f"TD {row['tds_for']}:{row['tds_against']}, "
            f"home {row['wins_as_home']}-{row['losses_as_home']} "
            f"away {row['wins_as_away']}-{row['losses_as_away']}{side})"
        )
        if row["opponent"].startswith("vs:"):
            vs_row = row
    print(" | ".join(parts) if parts else "empty ladder")

    if a.gate is None:
        return 0
    if vs_row is None:
        print("gate requested but report has no vs: rung", file=sys.stderr)
        return 3
    return 0 if points(vs_row) >= a.gate else 2


if __name__ == "__main__":
    sys.exit(main())
