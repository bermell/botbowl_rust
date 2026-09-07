#!/usr/bin/env python3
"""Plan 032 #7 gate: does a Q-informed policy target agree with deep search more
often than the visit target does?

Reuses the plan-028 Stage 0 convergence dump (`botbowl-ui convergence`: 50
random-start states x 3 independent repeats x budgets up to 16000, gen03 net,
production raw c=10). No search is run here. For every state, the 16000-budget
repeats are the reference; a candidate target built from a *lower*-budget
repeat is scored by top-1 agreement with the reference, over cross-repeat pairs
(i != j) exactly as `convergence_summary.py` scores the visit target (0.69 at
budget 1000 in plan 028's table).

Two references are reported, because they answer different questions:
  ref=visits  argmax of the reference run's visit target (plan 028's number)
  ref=played  the reference run's `chosen_action` = argmax mover-Q, i.e. what a
              16k search actually plays. This is the one that matters for a
              policy head that is meant to imitate strong play.

Candidate targets (all built from the same 1000-iteration root):
  visits         targets.rs::policy_target, OneHot on a solved root (baseline)
  argmaxq        one-hot on the played move (argmax mover-Q)
  cq(tau)        completed-Q: softmax(ln prior + q_mover / tau), unvisited /
                 unscored children completed with the visit-weighted mean of
                 the visited children's Q. tau in Q points (1000 = one TD).
  cqv(tau)       same, but the prior term is replaced by ln(visits + 1) — the
                 search's own breadth decision instead of the net's prior.

    scripts/audit_q_target.py runs/exp-conv/s0a-raw-c10.jsonl --budget 1000
"""
import argparse
import json
import math
from collections import defaultdict

from convergence_summary import policy_target as visit_target


def key(c):
    return json.dumps(c["action"], sort_keys=True)


def mover_q(q, mover):
    if q is None:
        return None
    return q if mover == "Home" else -q


def completed_q_target(row, tau, use_visits_as_prior=False, prior_weight=1.0):
    ch = row["children"]
    if not ch:
        return None
    mover = row["to_move"]
    qs = [mover_q(c["q"], mover) for c in ch]
    visited = [(c["visits"], q) for c, q in zip(ch, qs) if q is not None and c["visits"] > 0]
    if visited:
        tot = sum(v for v, _ in visited)
        fill = sum(v * q for v, q in visited) / tot
    else:
        scored = [q for q in qs if q is not None]
        if not scored:
            return None
        fill = sum(scored) / len(scored)
    logits = []
    for c, q in zip(ch, qs):
        qq = fill if q is None or c["visits"] == 0 else q
        if use_visits_as_prior:
            base = math.log(c["visits"] + 1.0)
        else:
            p = c.get("prior")
            base = math.log(max(p, 1e-6)) if p is not None else 0.0
        logits.append(prior_weight * base + qq / tau)
    m = max(logits)
    w = [math.exp(l - m) for l in logits]
    z = sum(w)
    return {key(c): x / z for c, x in zip(ch, w)}


def argmax_q_target(row):
    ch = row["children"]
    if not ch:
        return None
    mover = row["to_move"]
    best, bq = None, None
    for i, c in enumerate(ch):
        q = mover_q(c["q"], mover)
        if q is None:
            continue
        if bq is None or q > bq:
            best, bq = i, q
    if best is None:
        return None
    return {key(c): (1.0 if i == best else 0.0) for i, c in enumerate(ch)}


def top1(p):
    return max(p, key=p.get) if p else None


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--taus", default="25,50,100,200,400,800")
    a = ap.parse_args()

    cells = defaultdict(lambda: defaultdict(dict))
    for path in a.jsonl:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cells[r["state_seed"]][r["budget"]][r["repeat"]] = r
    budgets = sorted({b for s in cells.values() for b in s})
    ref_b = budgets[-1]
    taus = [float(t) for t in a.taus.split(",")]

    cands = {"visits": visit_target, "argmaxq": argmax_q_target}
    for t in taus:
        cands[f"cq({t:g})"] = lambda r, t=t: completed_q_target(r, t)
    for t in taus:
        cands[f"cqv({t:g})"] = lambda r, t=t: completed_q_target(r, t, use_visits_as_prior=True)

    # name -> list of per-pair agreement, by reference kind; plus fan-width strata
    agree = {k: defaultdict(list) for k in ("visits", "played")}
    strata = {k: defaultdict(lambda: defaultdict(list)) for k in ("visits", "played")}
    tv_same = defaultdict(list)  # TV to the same-kind target built at the reference budget
    n_states = 0

    def bucket(n):
        return "<=10" if n <= 10 else "11-30" if n <= 30 else "31-60" if n <= 60 else ">60"

    for seed, by_b in sorted(cells.items()):
        if ref_b not in by_b or a.budget not in by_b:
            continue
        refs = by_b[ref_b]
        rows = by_b[a.budget]
        n_states += 1
        fan = bucket(next(iter(refs.values()))["n_legal_actions"])
        ref_visits = {j: top1(visit_target(r)) for j, r in refs.items()}
        ref_played = {j: key({"action": r["chosen_action"]}) for j, r in refs.items()}
        for name, fn in cands.items():
            for i, r in rows.items():
                p = fn(r)
                if not p:
                    continue
                t = top1(p)
                for j in refs:
                    if i == j:
                        continue
                    agree["visits"][name].append(1.0 if t == ref_visits[j] else 0.0)
                    agree["played"][name].append(1.0 if t == ref_played[j] else 0.0)
                    strata["visits"][name][fan].append(1.0 if t == ref_visits[j] else 0.0)
                    strata["played"][name][fan].append(1.0 if t == ref_played[j] else 0.0)
                    pr = fn(refs[j])
                    if pr:
                        tv_same[name].append(tv(p, pr))

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    print(f"{n_states} states, candidate budget {a.budget}, reference {ref_b}, cross-repeat pairs\n")
    print(f"{'target':<12}{'top1 vs ref visits':>20}{'top1 vs ref played':>20}{'TV same-kind':>14}")
    for name in cands:
        print(
            f"{name:<12}{mean(agree['visits'][name]):>20.3f}"
            f"{mean(agree['played'][name]):>20.3f}{mean(tv_same[name]):>14.3f}"
        )
    for refk in ("played", "visits"):
        print(f"\ntop-1 vs ref {refk}, by legal-action count at the root")
        fans = ["<=10", "11-30", "31-60", ">60"]
        print(f"{'target':<12}" + "".join(f"{f:>10}" for f in fans))
        counts = {f: len(strata[refk]["visits"][f]) for f in fans}
        print(f"{'(pairs)':<12}" + "".join(f"{counts[f]:>10}" for f in fans))
        for name in cands:
            print(f"{name:<12}" + "".join(f"{mean(strata[refk][name][f]):>10.3f}" for f in fans))


if __name__ == "__main__":
    main()
