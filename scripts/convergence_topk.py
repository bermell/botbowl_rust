#!/usr/bin/env python3
"""At what search budget does the root *decision* stop changing? (exp057, follows plan 025/045)

    convergence_topk.py runs/exp057/conv_*.jsonl

Reads `botbowl-ui convergence` dumps: the same state searched at a ladder of budgets, R
independent repeats per cell, the largest budget being the reference. Per board, per probe depth
(`advance`) and per budget, it reports against the reference, always pairing *independent* runs
(repeat i at budget t against repeat j at the reference, i != j), so the noise floor (reference
against reference) carries exactly the same run-to-run variance:

  top1    P(the bot's pick at t == the reference's pick)
  in3     P(the pick at t is among the reference's top 3)
  top3    mean |top-3 set at t intersect reference top-3 set| / 3   (roots with >= 3 children)
  regret  how much worse the pick at t is, judged by the reference run's own Q (mover frame,
          1000 = one TD): Q_ref(ref pick) - Q_ref(pick at t). Picking a near-equal alternative
          costs ~0; this is the decision-relevant one.
  cq100 / cq20  total variation between the training targets (completed-Q, tau 100 / tau 20, a port
          of botbowl-nn's completed_q_target) at t and at the reference

Ranking is the bot's own rule (`pick_best_action`): scored children first, then mover-frame Q,
then visits. Roots with a single legal action are dropped.

Rows from different selection rules (the `puct` field, e.g. plan 053's `…@gumbel16_iters`) are
separate arms at the same budget, all scored against the one reference: the largest budget of the
plain rule. So a Gumbel run can be scored against exp057's PUCT references by passing both files.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import product


def key(action):
    return json.dumps(action, sort_keys=True)


def mover_q(q, mover):
    return None if q is None else (q if mover == "Home" else -q)


def ranked(row):
    """Children's action keys in the bot's preference order."""
    mover = row["to_move"]
    def k(c):
        q = mover_q(c["q"], mover)
        return (1 if q is not None else 0, q if q is not None else 0, c["visits"])
    return [key(c["action"]) for c in sorted(row["children"], key=k, reverse=True)]


def q_of(row):
    mover = row["to_move"]
    return {key(c["action"]): mover_q(c["q"], mover) for c in row["children"]}


def cq_target(row, tau):
    """Port of targets.rs::completed_q_target (OneHot on a solved root)."""
    ch, mover = row["children"], row["to_move"]
    keys = [key(c["action"]) for c in ch]
    qs = [mover_q(c["q"], mover) for c in ch]
    if row["root_solved"]:
        best = max((i for i, q in enumerate(qs) if q is not None), key=lambda i: qs[i], default=None)
        return None if best is None else {keys[best]: 1.0}
    scored = [(c, q) for c, q in zip(ch, qs) if q is not None]
    if not scored:
        return None
    w = sum(c["visits"] for c, q in scored if c["visits"] > 0)
    fill = (sum(c["visits"] * q for c, q in scored if c["visits"] > 0) / w) if w > 0 else sum(q for _, q in scored) / len(scored)
    done = [q if (q is not None and c["visits"] > 0) else fill for c, q in zip(ch, qs)]
    logits = [(math.log(max(c["prior"], 1e-6)) if c.get("prior") is not None else 0.0) + q / tau for c, q in zip(ch, done)]
    m = max(logits)
    ws = [math.exp(l - m) for l in logits]
    z = sum(ws)
    return {k: x / z for k, x in zip(keys, ws)}


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def arm_label(puct):
    """`` for the plain rule, else the preset or rule that tells this arm apart."""
    if "@" in puct:
        return puct.split("@", 1)[1]
    return "" if puct in ("", "puct=raw(c=10)") else puct


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def compare(a, r):
    """Metrics of run `a` against reference run `r` (same state)."""
    ra, rr = ranked(a), ranked(r)
    qr = q_of(r)
    out = {"top1": float(key(a["chosen_action"]) == key(r["chosen_action"])),
           "in3": float(key(a["chosen_action"]) in rr[:3])}
    if len(rr) >= 3 and len(ra) >= 3:
        out["top3"] = len(set(ra[:3]) & set(rr[:3])) / 3
    best, pick = qr.get(key(r["chosen_action"])), qr.get(key(a["chosen_action"]))
    if best is not None and pick is not None:
        out["regret"] = best - pick
    for tau in (100, 20):
        pa, pr = cq_target(a, tau), cq_target(r, tau)
        if pa and pr:
            out[f"cq{tau}"] = tv(pa, pr)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--wide", type=int, default=10, help="fan above this counts as wide")
    a = p.parse_args()

    cells = defaultdict(lambda: defaultdict(dict))  # (board, advance, seed) -> budget -> repeat -> row
    for f in a.files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r["n_legal_actions"] >= 2:
                    arm = (r["budget"], arm_label(r.get("puct", "")))
                    cells[(r.get("board", "env"), r.get("advance", 0), r["state_seed"])][arm][r["repeat"]] = r

    groups = defaultdict(list)
    for (board, adv, seed), by_budget in cells.items():
        groups[(board, adv)].append(by_budget)
        groups[(board, "all")].append(by_budget)

    metrics = ["top1", "in3", "top3", "regret", "cq100", "cq20"]
    for (board, adv), states in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        budgets = sorted({b for s in states for b in s}, key=lambda a: (a[1], a[0]))
        ref = max((a for a in budgets if a[1] == ""), default=budgets[-1])
        fans = [len(next(iter(s[ref].values()))["children"]) for s in states if s.get(ref)]
        print(f"\n== {board}, advance {adv}: {len(states)} states (fan median {sorted(fans)[len(fans)//2] if fans else '?'}, "
              f"wide >{a.wide}: {sum(f > a.wide for f in fans)}), reference {ref[0]} descents ==")
        print(f"{'arm':>22} " + " ".join(f"{m:>7}" for m in metrics) + f" {'wide top1':>9} {'ms':>7}")
        for t in budgets:
            acc = defaultdict(list)
            wide = []
            ms = []
            for s in states:
                if t not in s or ref not in s:
                    continue
                for (i, ra), (j, rr) in product(s[t].items(), s[ref].items()):
                    if i == j:
                        continue
                    c = compare(ra, rr)
                    for m, v in c.items():
                        acc[m].append(v)
                    if len(rr["children"]) > a.wide:
                        wide.append(c["top1"])
                ms.extend(r["elapsed_ms"] for r in s[t].values())
            label = f"{t[1] + ' ' if t[1] else ''}{t[0]}" + (" (floor)" if t == ref else "")
            print(f"{label:>22} " + " ".join(f"{mean(acc[m]):7.3f}" if m != "regret" else f"{mean(acc[m]):7.1f}" for m in metrics)
                  + f" {mean(wide):9.3f} {mean(ms):7.0f}")
    print("\nThe reference row is the noise floor: two independent reference searches against each other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
