#!/usr/bin/env python3
"""Compare PUCT selection arms from `botbowl-ui convergence` dumps (plan 026).

    puct_sweep_summary.py runs/puct/*.jsonl

Groups rows by the `puct` arm label and reports, per arm, at each budget:

  agree   pairwise top-1 agreement between independent repeats. The primary
          number: it degrades at BOTH extremes (lock-in gives different early
          leaders; uniform visits give an arbitrary argmax), so it has an
          interior optimum and cannot be gamed.
  TV      mean total-variation distance between repeats. Secondary, and
          GAMEABLE on its own -- it goes to 0 as visits approach uniform,
          which is why `peak` and `H` are printed beside it.
  peak    mean share of visits on the most-visited child. The guard: an arm
          that "wins" on TV while peak collapses toward 1/n is not deciding
          anything.
  H       normalised entropy of the visit distribution, 0 = one-hot,
          1 = uniform. Same guard, scale-free across differing action counts.
  |dv|    mean run-to-run difference in root value, in [-1,1] units.

Reuses `policy_target` from convergence_summary.py, which ports
botbowl-nn/src/targets.rs (solved children freeze their visit counts, so raw
visits are the wrong thing to compare).
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convergence_summary import policy_target, top1, tv  # noqa: E402


def entropy(probs):
    """Shannon entropy normalised by log(n), so arms with different action
    counts are comparable. 0 = one-hot, 1 = uniform."""
    ps = [p for p in probs if p > 0]
    if len(ps) < 2:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--value-scale", type=float, default=1000.0)
    a = ap.parse_args()

    # cells[arm][state][budget][repeat] = row
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in a.jsonl:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                arm = r.get("puct", "unknown")
                cells[arm][r["state_seed"]][r["budget"]][r["repeat"]] = r

    def armkey(name):
        """Sort raw arms before normalised, each by ascending c."""
        try:
            c = float(name.split("c=")[1].split(",")[0].rstrip(")"))
        except (IndexError, ValueError):
            c = 0.0
        return (0 if "raw" in name else 1, c)

    for arm in sorted(cells, key=armkey):
        by_state = cells[arm]
        budgets = sorted({b for s in by_state.values() for b in s})
        print(f"\n=== {arm}   ({len(by_state)} states) ===")
        print(f"{'budget':>7} {'agree':>7} {'TV':>7} {'peak':>7} {'H':>6} {'|dv|':>7}")
        for b in budgets:
            ag, tvs, pk, hh, dv = [], [], [], [], []
            for _, by_budget in sorted(by_state.items()):
                rows = by_budget.get(b, {})
                pts = {}
                for rep, r in rows.items():
                    pt = policy_target(r)
                    if pt:
                        pts[rep] = pt
                    ch = r["children"]
                    tot = sum(c["visits"] for c in ch)
                    if tot:
                        shares = [c["visits"] / tot for c in ch]
                        pk.append(max(shares))
                        hh.append(entropy(shares))
                ks = sorted(pts)
                for i in range(len(ks)):
                    for j in range(i + 1, len(ks)):
                        tvs.append(tv(pts[ks[i]], pts[ks[j]]))
                        ag.append(1.0 if top1(pts[ks[i]]) == top1(pts[ks[j]]) else 0.0)
                rk = sorted(rows)
                for i in range(len(rk)):
                    for j in range(i + 1, len(rk)):
                        vi, vj = rows[rk[i]]["root_value"], rows[rk[j]]["root_value"]
                        if vi is not None and vj is not None:
                            dv.append(abs(vi - vj) / a.value_scale)
            m = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
            print(f"{b:>7} {m(ag):>7.2f} {m(tvs):>7.4f} {m(pk):>7.3f} {m(hh):>6.3f} {m(dv):>7.4f}")

    print("\nRead `agree` as the headline; `TV` alone is gameable (-> 0 as visits go")
    print("uniform), so a low TV is only meaningful if `peak` has not collapsed.")


if __name__ == "__main__":
    main()
