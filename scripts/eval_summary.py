#!/usr/bin/env python3
"""One-line summary of an `botbowl-ui eval` report JSON, plus the promotion gate.

    eval_summary.py report.json               # print summary line
    eval_summary.py report.json --gate 0.55   # also exit 0 if the vs: rung
                                              # win_rate >= gate, 2 if below,
                                              # 3 if the report has no vs: rung
"""
import argparse
import json
import sys


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
        parts.append(
            f"{row['opponent']} {row['win_rate']:.2f} "
            f"(W{row['wins']} D{row['draws']} L{row['losses']} "
            f"TD {row['tds_for']}:{row['tds_against']}, "
            f"home {row['wins_as_home']}-{row['losses_as_home']} "
            f"away {row['wins_as_away']}-{row['losses_as_away']})"
        )
        if row["opponent"].startswith("vs:"):
            vs_row = row
    print(" | ".join(parts) if parts else "empty ladder")

    if a.gate is None:
        return 0
    if vs_row is None:
        print("gate requested but report has no vs: rung", file=sys.stderr)
        return 3
    return 0 if vs_row["win_rate"] >= a.gate else 2


if __name__ == "__main__":
    sys.exit(main())
