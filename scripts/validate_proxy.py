#!/usr/bin/env python3
"""Score ranking proxies against the gold results (plan 051 step 2 / step 5).

    validate_proxy.py --gold runs/validation [--proxy P2=runs/plan051_proxy/p2 ...] [--sprt 0.5:0.55]

Gold: `<gold>/<pair>/report.json` + `eval.games.jsonl`, fixed-N full games (scripts/plan051_gold.sh).
Each pair has one rung per board. P1, the SPRT on paired full games, is computed **offline** by
replaying the gold lines in game order through the same test the binary runs. Proxies:
`<dir>/<pair>/r<k>/report.json`, one rung per board, matched to gold by board.

Cost is the candidate's MCTS iterations (`telemetry.iterations`), summed over the games or
drives a decision used. It counts the same work on any machine, unlike wall-clock: the gold jobs
shared the box, so their wall times are confounded. Both seats search at one budget, so the
opponent's share scales with it.

Acceptance (plan 051 step 2), over R runs per pair:
  (a) no gap pair with gold >= 0.55 gets the wrong sign,
  (b) at most one false H1 across all zero-gap runs,
  (c) mean cost to decision <= 1/3 of P1's on the same pairs.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict


def penta_stats(counts):
    n = sum(counts)
    if n == 0:
        return 0.5, 0.0, 0
    m = sum(i / 4 * c for i, c in enumerate(counts)) / n
    var = sum((i / 4 - m) ** 2 * c for i, c in enumerate(counts)) / n
    return m, var, n


class Sprt:
    """The same test as `botbowl_play::stats::Sprt`, for offline replay."""

    MIN_PAIRS = 8

    def __init__(self, spec):
        parts = [float(x) for x in spec.split(":")]
        self.s0, self.s1 = parts[0], parts[1]
        self.alpha, self.beta = (parts[2], parts[3]) if len(parts) == 4 else (0.05, 0.05)
        self.lower = math.log(self.beta / (1 - self.alpha))
        self.upper = math.log((1 - self.beta) / self.alpha)

    def llr(self, counts):
        m, var, n = penta_stats(counts)
        if n == 0:
            return 0.0
        if var <= 0:
            s = (self.s0 + self.s1) / 2
            var = s * (1 - s)
        return n * (self.s1 - self.s0) * (2 * m - self.s0 - self.s1) / (2 * var)

    def verdict(self, counts):
        if sum(counts) < self.MIN_PAIRS:
            return "Undecided"
        x = self.llr(counts)
        return "H1" if x >= self.upper else "H0" if x <= self.lower else "Undecided"


def half_points(line):
    h, a = line["home_score"], line["away_score"]
    cand, opp = (h, a) if line["candidate_team"] == "Home" else (a, h)
    return 2 if cand > opp else 1 if cand == opp else 0


def iters(line):
    return (line.get("telemetry") or {}).get("iterations", 0)


def replay_sprt(lines, sprt):
    """P1: feed games in index order, fold pairs, stop at a verdict. Returns
    (verdict, games, iterations, points over the games used)."""
    lines = sorted(lines, key=lambda l: l["game"])
    counts = [0] * 5
    pending = {}
    used = cost = pts = 0
    verdict = "Undecided"
    for l in lines:
        used += 1
        cost += iters(l)
        hp = half_points(l)
        pts += hp / 2
        k = l["game"] // 2
        if k in pending:
            counts[pending.pop(k) + hp] += 1
            verdict = sprt.verdict(counts)
            if verdict != "Undecided":
                break
        else:
            pending[k] = hp
    return verdict, used, cost, pts / used if used else float("nan")


def board_of(row):
    return row.get("board") or "env"


def load_gold(root):
    gold = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "report.json"))):
        pair = os.path.basename(os.path.dirname(d))
        rep = json.load(open(d))
        lines = defaultdict(list)
        gpath = os.path.join(os.path.dirname(d), "eval.games.jsonl")
        if os.path.exists(gpath):
            for line in open(gpath):
                if line.strip():
                    l = json.loads(line)
                    lines[l["rung"]].append(l)
        for row in rep["ladder"]:
            m, var, n = penta_stats(row.get("pairs", {}).get("counts", [0] * 5))
            se = math.sqrt(var / n) if n > 1 else float("nan")
            gold[(pair, board_of(row))] = {
                "points": row.get("points", (row["wins"] + row["draws"] / 2) / max(row["games"], 1)),
                "se": row.get("points_se", se),
                "games": row["games"],
                "cost": (row.get("telemetry") or {}).get("iterations", 0),
                "lines": lines.get(row["opponent"], []),
            }
    return gold


def load_proxy(root):
    """{(pair, board): [run, ...]} with each run's verdict, points, samples and cost."""
    out = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "*", "r*", "report.json"))):
        pair = os.path.basename(os.path.dirname(os.path.dirname(d)))
        for row in json.load(open(d))["ladder"]:
            s = row.get("sprt") or {}
            out[(pair, board_of(row))].append({
                "verdict": s.get("verdict", "Undecided"),
                "points": row.get("points", float("nan")),
                "games": row["games"],
                "cost": (row.get("telemetry") or {}).get("iterations", 0),
            })
    return out


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def is_zero_gap(pair):
    return pair.startswith("zero")


def judge(name, keys, gold, runs_of, p1_cost, against="P1"):
    """Apply the acceptance rule to one proxy. `runs_of(key)` is that proxy's list of runs."""
    wrong = false_h1 = zero_runs = 0
    costs, p1 = [], []
    xs, ys = [], []
    for key in keys:
        g = gold[key]
        runs = runs_of(key)
        for r in runs:
            if is_zero_gap(key[0]):
                zero_runs += 1
                false_h1 += r["verdict"] == "H1"
            elif g["points"] >= 0.55 and r["verdict"] == "H0":
                wrong += 1
            elif g["points"] <= 0.45 and r["verdict"] == "H1":
                wrong += 1
            costs.append(r["cost"])
            p1.append(p1_cost[key])
        if runs:
            xs.append(g["points"])
            ys.append(sum(r["points"] for r in runs) / len(runs))
    ratio = (sum(costs) / sum(p1)) if p1 and sum(p1) else float("nan")
    ok = wrong == 0 and false_h1 <= 1 and ratio <= 1 / 3
    if against != "P1":
        # P1 is the yardstick the others are costed against, not a candidate for adoption.
        print(
            f"{name:>4}: wrong sign {wrong}, false H1 {false_h1}/{zero_runs} zero-gap runs, "
            f"rank corr {spearman(xs, ys):.2f}, cost vs {against} {ratio:.2f}"
        )
        return
    print(
        f"{name:>4}: wrong sign {wrong}, false H1 {false_h1}/{zero_runs} zero-gap runs, "
        f"rank corr {spearman(xs, ys):.2f}, cost vs {against} {ratio:.2f}  "
        f"-> {'ACCEPT' if ok else 'pass (a)-(b), fail (c): pre-screen only' if wrong == 0 and false_h1 <= 1 else 'REJECT'}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--proxy", action="append", default=[], help="NAME=DIR")
    p.add_argument("--sprt", default="0.5:0.55")
    a = p.parse_args()
    sprt = Sprt(a.sprt)

    gold = load_gold(a.gold)
    if not gold:
        print(f"no gold under {a.gold}", file=sys.stderr)
        return 1
    proxies = {}
    for spec in a.proxy:
        name, d = spec.split("=", 1)
        proxies[name] = load_proxy(d)

    keys = sorted(gold)
    p1 = {}
    print(f"{'pair':24} {'board':8} {'gold pts':>14} {'n':>4} | {'P1 verdict':>10} {'games':>5} {'cost':>6}", end="")
    for name in proxies:
        print(f" | {name}: verdicts / mean n / cost", end="")
    print()
    for key in keys:
        g = gold[key]
        v, used, cost, _ = replay_sprt(g["lines"], sprt) if g["lines"] else ("n/a", 0, 0, 0)
        p1[key] = cost
        ratio = cost / g["cost"] if g["cost"] else float("nan")
        print(f"{key[0]:24} {key[1]:8} {g['points']:.3f} ± {g['se']:.3f} {g['games']:>4} | {v:>10} {used:>5} {ratio:>6.2f}", end="")
        for name, runs in proxies.items():
            rs = runs.get(key, [])
            if rs:
                verdicts = ",".join(r["verdict"] for r in rs)
                mean_n = sum(r["games"] for r in rs) / len(rs)
                mean_cost = sum(r["cost"] for r in rs) / len(rs) / cost if cost else float("nan")
                print(f" | {verdicts} / {mean_n:.0f} / {mean_cost:.2f}", end="")
            else:
                print(" | -", end="")
        print()
    print("(P1 cost is a fraction of the full gold match; a proxy's cost is a fraction of P1's.)")

    # P1 as its own proxy: one replay per pair, so R = 1.
    def p1_runs(key):
        g = gold[key]
        if not g["lines"]:
            return []
        v, used, cost, pts = replay_sprt(g["lines"], sprt)
        return [{"verdict": v, "points": pts, "games": used, "cost": cost}]

    judge("P1", keys, gold, p1_runs, {k: gold[k]["cost"] for k in keys}, against="the full match")
    for name, runs in proxies.items():
        judge(name, keys, gold, lambda k, r=runs: r.get(k, []), p1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
