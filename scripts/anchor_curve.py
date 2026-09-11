#!/usr/bin/env python3
"""The gateless loop's strength curve: every generation vs the frozen anchor (plan 030).

    anchor_curve.py [RUN_DIR] [--anchor bbnet_14x7_gen03.onnx] [--window 3]
                    [--regress 0.10] [--plateau-delta 0.05] [--plateau-gens 6]
                    [--summary GEN]

Per generation it takes the vs-anchor row from `genNN/anchor.json` (a backfill,
`anchor_backfill.sh`) or else from `genNN/report.json` (the loop's own eval — the
gated era's vs-champion rung counts too whenever the champion *was* the anchor,
so gen05-07 join the curve for free). Only `--evaluator nn` rows are used.

Reads, as agreed in plan 030 §Decisions 2026-09-10:
  * a single generation's score is a point on a curve, not a verdict (40 games,
    SE ≈ 0.07);
  * the rolling mean pools the last --window generations' games (120 games at
    the default, SE ≈ 0.04) — that is the number to read;
  * two *advisory* flags, never an automatic rollback:
      REGRESSION  rolling mean ≥ --regress below the best rolling mean so far
      PLATEAU     the best rolling mean over the last --plateau-gens
                  generations is not ≥ --plateau-delta above the best before them
    Both mean "come and look"; every net stays on disk and rollback is by hand.

--summary GEN prints one line for status.md instead of the table.
"""
import argparse
import glob
import json
import math
import os
import re
import sys


def points_and_var(row):
    """Mean points (W + D/2)/N and the per-game sample variance."""
    n = row["games"]
    if not n:
        return float("nan"), float("nan")
    w, d, l = row["wins"], row["draws"], row["losses"]
    m = (w + 0.5 * d) / n
    ss = w * (1 - m) ** 2 + d * (0.5 - m) ** 2 + l * (0 - m) ** 2
    var = ss / (n - 1) if n > 1 else float("nan")
    return m, var


def anchor_row(report, anchor):
    for row in report.get("ladder", []):
        opp = row["opponent"]
        if opp.startswith("vs:mcts(nn:") and os.path.basename(anchor) in opp:
            return row
    return None


def heuristic_row(report):
    for row in report.get("ladder", []):
        if row["opponent"] == "mcts-heuristic":
            return row
    return None


def load(run_dir, anchor):
    gens = []
    for d in sorted(glob.glob(os.path.join(run_dir, "gen[0-9][0-9]"))):
        m = re.search(r"gen(\d\d)$", d)
        if not m:
            continue
        g = int(m.group(1))
        row, src, heur = None, None, None
        for fname in ("anchor.json", "report.json"):
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            try:
                rep = json.load(open(p))
            except (OSError, ValueError):
                continue
            if heur is None:
                heur = heuristic_row(rep)
            if row is None:
                r = anchor_row(rep, anchor)
                if r is not None:
                    row, src = r, fname
        if row is None:
            continue
        pts, var = points_and_var(row)
        gens.append({
            "gen": g, "src": src, "n": row["games"], "pts": pts, "var": var,
            "w": row["wins"], "d": row["draws"], "l": row["losses"],
            "heur": points_and_var(heur)[0] if heur else float("nan"),
        })
    return gens


def rolling(gens, window):
    """Games-weighted pooled mean over the trailing window; SE from pooled variance."""
    out = []
    for i in range(len(gens)):
        span = gens[max(0, i - window + 1): i + 1]
        n = sum(x["n"] for x in span)
        w = sum(x["w"] for x in span)
        d = sum(x["d"] for x in span)
        m = (w + 0.5 * d) / n if n else float("nan")
        l = n - w - d
        ss = w * (1 - m) ** 2 + d * (0.5 - m) ** 2 + l * m ** 2
        se = math.sqrt(ss / (n - 1) / n) if n > 1 else float("nan")
        out.append((m, se, n, len(span)))
    return out


def flags(gens, roll, a):
    """Advisory flags evaluated at the last generation."""
    out = []
    full = [r for r in roll if r[3] >= a.window]
    if not full:
        return out
    cur = roll[-1][0]
    best = max(r[0] for r in full)
    if cur <= best - a.regress:
        out.append(f"REGRESSION: rolling {cur:.3f} is {best - cur:.3f} below the best rolling {best:.3f}")
    if len(full) > a.plateau_gens:
        recent = max(r[0] for r in full[-a.plateau_gens:])
        before = max(r[0] for r in full[:-a.plateau_gens])
        if recent < before + a.plateau_delta:
            out.append(f"PLATEAU: best rolling over the last {a.plateau_gens} gens {recent:.3f} "
                       f"vs {before:.3f} before them (< +{a.plateau_delta:.2f})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", default="runs/loop14x7")
    p.add_argument("--anchor", default="bbnet_14x7_gen03.onnx")
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--regress", type=float, default=0.10)
    p.add_argument("--plateau-delta", type=float, default=0.05)
    p.add_argument("--plateau-gens", type=int, default=6)
    p.add_argument("--summary", type=int, default=None, metavar="GEN",
                   help="one status.md line for this generation instead of the table")
    a = p.parse_args()

    gens = load(a.run_dir, a.anchor)
    if not gens:
        print(f"no vs-{a.anchor} rows under {a.run_dir}", file=sys.stderr)
        return 1
    roll = rolling(gens, a.window)
    fl = flags(gens, roll, a)

    if a.summary is not None:
        i = next((k for k, g in enumerate(gens) if g["gen"] == a.summary), None)
        if i is None:
            print(f"gen{a.summary:02d} has no anchor row yet", file=sys.stderr)
            return 1
        g, (rm, rse, rn, rk) = gens[i], roll[i]
        se = math.sqrt(g["var"] / g["n"]) if g["n"] > 1 else float("nan")
        full = [r[0] for r in roll[: i + 1] if r[3] >= a.window]
        best = f"{max(full):.3f}" if full else f"n/a (window not full until gen{gens[0]['gen'] + a.window - 1:02d})"
        line = (f"anchor {os.path.basename(a.anchor)}: gen{g['gen']:02d} {g['pts']:.3f} ± {se:.3f} "
                f"(W{g['w']} D{g['d']} L{g['l']}, n={g['n']}) | rolling{rk} {rm:.3f} ± {rse:.3f} (n={rn})"
                f" | best rolling {best} | heuristic rung {g['heur']:.3f}")
        if i == len(gens) - 1 and fl:
            line += " | " + "; ".join(fl)
        print(line)
        return 0

    print(f"vs {a.anchor}  ({a.run_dir}); rolling window {a.window} gens, games-pooled")
    print(f"{'gen':>5} {'src':>12} {'n':>4} {'pts':>6} {'±SE':>6} {'W-D-L':>10} {'roll':>6} {'±SE':>6} {'n':>4} {'heur':>6}")
    for g, (rm, rse, rn, rk) in zip(gens, roll):
        se = math.sqrt(g["var"] / g["n"]) if g["n"] > 1 else float("nan")
        rs = f"{rm:.3f}" if rk >= a.window else f"({rm:.3f})"
        print(f"{g['gen']:>5} {g['src']:>12} {g['n']:>4} {g['pts']:.3f} {se:.3f} "
              f"{g['w']:>3}-{g['d']:>2}-{g['l']:<3} {rs:>7} {rse:.3f} {rn:>4} {g['heur']:.3f}")
    for f in fl:
        print("!! " + f)
    if not fl:
        print("flags: none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
