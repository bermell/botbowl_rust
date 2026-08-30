#!/usr/bin/env python3
"""Analyse a `botbowl-ui convergence` dump (plan 025).

Answers: at what iteration budget does the search output stop changing by more
than the search's own run-to-run nondeterminism?

    convergence_summary.py runs/convergence/nn_value.jsonl

Key definitions, both from plan 025:

  signal(t) = mean over repeat pairs of TV( policy_t^i , policy_ref^j ), i != j
  floor     = mean over repeat pairs of TV( policy_ref^i , policy_ref^j )

Both are computed from *independent* runs, so both carry exactly one unit of
run-to-run variance and the crossing point is not biased by it. X* is the
smallest budget where signal(t) <= floor.

The policy target mirrors `botbowl-nn/src/targets.rs::policy_target`
(SolvedRootPolicy::OneHot) rather than using raw visits: a solved child's visit
count is frozen, and the fastest-solving child is often the *best* move, so
`pi ~ visits` is actively wrong.
"""
import argparse
import json
from collections import defaultdict
from statistics import median


def mover_q(q, mover):
    if q is None:
        return None
    return q if mover == "Home" else -q


def policy_target(row):
    """Port of targets.rs::policy_target, OneHot on a solved root.

    Returns {action_key: prob} — keyed by action, NOT index, because
    recon_mcts enumerates children in HashMap order which differs per run.
    """
    ch = row["children"]
    if not ch:
        return None
    mover = row["to_move"]
    keys = [json.dumps(c["action"], sort_keys=True) for c in ch]

    def argmax_q(filter_solved):
        best_i, best_q = None, None
        for i, c in enumerate(ch):
            if filter_solved is not None and c["solved"] != filter_solved:
                continue
            q = mover_q(c["q"], mover)
            if q is None:
                continue
            if best_q is None or q > best_q:
                best_i, best_q = i, q
        return best_i

    if row["root_solved"]:
        best = argmax_q(None)
        if best is None:
            best = max(range(len(ch)), key=lambda i: ch[i]["visits"])
        probs = [0.0] * len(ch)
        probs[best] = 1.0
        return dict(zip(keys, probs))

    counts = [float(c["visits"]) for c in ch]
    if any(c["solved"] for c in ch):
        unsolved = [c["visits"] for c in ch if not c["solved"]]
        max_unsolved = float(max(unsolved)) if unsolved else 0.0
        bs = argmax_q(True)
        if bs is not None:
            counts[bs] = max(counts[bs], max_unsolved)
    total = sum(counts)
    if total <= 0:
        return None
    return dict(zip(keys, [c / total for c in counts]))


def tv(p, q):
    """Total-variation distance over the union of action keys."""
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def top1(p):
    return max(p, key=p.get) if p else None


def pctl(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--value-scale", type=float, default=1000.0,
                    help="divide root_value by this to reach the [-1,1] target domain")
    a = ap.parse_args()

    # cells[state][budget][repeat] = row
    cells = defaultdict(lambda: defaultdict(dict))
    for path in a.jsonl:
      with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            cells[r["state_seed"]][r["budget"]][r["repeat"]] = r

    budgets = sorted({b for s in cells.values() for b in s})
    ref = budgets[-1]
    print(f"{len(cells)} states, budgets {budgets}, reference {ref}\n")

    per_budget = defaultdict(list)      # budget -> [signal per state]
    floor_all, value_sig = [], defaultdict(list)
    top1_agree = defaultdict(list)
    xstars, solved_at, strata = [], [], {}

    for sidx, by_budget in sorted(cells.items()):
        if ref not in by_budget:
            continue
        refs = {rep: policy_target(r) for rep, r in by_budget[ref].items()}
        refs = {k: v for k, v in refs.items() if v}
        if len(refs) < 2:
            continue

        # Noise floor: distance between independent runs at the reference.
        f = [tv(refs[i], refs[j]) for i in refs for j in refs if i < j]
        floor_s = sum(f) / len(f)
        floor_all.append(floor_s)

        any_row = next(iter(by_budget[ref].values()))
        strata[sidx] = (any_row["n_legal_actions"], any_row["half"], any_row["root_solved"])

        # First budget at which the root was already solved.
        solved_b = next((b for b in budgets
                         if any(r["root_solved"] for r in by_budget.get(b, {}).values())), None)
        solved_at.append(solved_b)

        sig_by_b, xstar = {}, None
        for b in budgets:
            if b not in by_budget:
                continue
            pts = {rep: policy_target(r) for rep, r in by_budget[b].items()}
            pts = {k: v for k, v in pts.items() if v}
            if not pts:
                continue
            # Cross-repeat only (i != j) so signal and floor are comparable.
            d = [tv(pts[i], refs[j]) for i in pts for j in refs if i != j]
            if not d:
                continue
            sig = sum(d) / len(d)
            sig_by_b[b] = sig
            per_budget[b].append(sig)

            t1 = [1.0 if top1(pts[i]) == top1(refs[j]) else 0.0
                  for i in pts for j in refs if i != j]
            top1_agree[b].append(sum(t1) / len(t1))

            vs = [abs(by_budget[b][i]["root_value"] - by_budget[ref][j]["root_value"]) / a.value_scale
                  for i in by_budget[b] for j in by_budget[ref]
                  if i != j and by_budget[b][i]["root_value"] is not None
                  and by_budget[ref][j]["root_value"] is not None]
            if vs:
                value_sig[b].append(sum(vs) / len(vs))

        for b in budgets:
            if b in sig_by_b and sig_by_b[b] <= floor_s:
                xstar = b
                break
        xstars.append(xstar if xstar is not None else ref)

    floor = sum(floor_all) / len(floor_all) if floor_all else float("nan")
    print(f"{'budget':>8} {'signal':>8} {'floor':>8} {'ratio':>7} {'top1':>6} {'|dv|':>7}")
    for b in budgets:
        s = sum(per_budget[b]) / len(per_budget[b]) if per_budget[b] else float("nan")
        v = sum(value_sig[b]) / len(value_sig[b]) if value_sig[b] else float("nan")
        t = sum(top1_agree[b]) / len(top1_agree[b]) if top1_agree[b] else float("nan")
        mark = "  <-- at floor" if s <= floor else ""
        print(f"{b:>8} {s:>8.4f} {floor:>8.4f} {s/floor:>7.2f} {t:>6.2f} {v:>7.4f}{mark}")

    print(f"\nnoise floor (mean over states, TV between independent runs at {ref}): {floor:.4f}")
    print(f"X* per state: median {median(xstars):.0f}, p75 {pctl(xstars,0.75)}, "
          f"p90 {pctl(xstars,0.90)}, max {max(xstars)}")
    print(f"  -> plan 025 decision rule (p90): X = {pctl(xstars,0.90)}")

    n_solved = sum(1 for s in solved_at if s is not None)
    print(f"\nroots solved at some budget: {n_solved}/{len(solved_at)}")
    if n_solved:
        vals = [s for s in solved_at if s is not None]
        print(f"  first budget at which solved: median {median(vals):.0f}, min {min(vals)}")

    # Stratify X* by legal-action count.
    print("\nX* by legal-action count:")
    buckets = defaultdict(list)
    for x, sidx in zip(xstars, sorted(cells)):
        if sidx in strata:
            n = strata[sidx][0]
            key = "1-5" if n <= 5 else "6-15" if n <= 15 else "16-30" if n <= 30 else "31+"
            buckets[key].append(x)
    for k in ("1-5", "6-15", "16-30", "31+"):
        if buckets[k]:
            print(f"  {k:>6}: n={len(buckets[k]):>3}  median {median(buckets[k]):>6.0f}  "
                  f"p90 {pctl(buckets[k],0.90):>6}")


if __name__ == "__main__":
    main()
