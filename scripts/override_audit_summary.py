#!/usr/bin/env python3
"""Summarise `botbowl-ui override-audit` rows (plan 055 §3 phase 2).

    scripts/override_audit_summary.py audit.jsonl [more.jsonl ...]

Rows group by (model, search config, iterations); each group gets its own report. Every value is
in the mover's frame and TD units (+1 = the mover scores the drive). MC is the policy-only drive
outcome from the move, averaged over the row's paired playouts; V is the net's value at the first
decision after the move (averaged over the move's own dice), V(s) the net's value at the decision.

Read every MC as "under the policy's own continuation": both sides play policy-only after the
move, so MC is Q^pi of the bare policy, not the move's value under good play. That is the right
quantity for "is this override a one-step improvement over the policy", and it means a move that
pays off only with a follow-up the policy will not find is scored against the search.

- **Ledger:** an override helps / hurts when its paired MC(a_s) - MC(a_p) is beyond +-2 SE, else
  it is *unresolved*. At 64 playouts the per-row SE is ~0.12-0.14, so "unresolved" mostly measures
  statistical power, not near-ties: read the mean difference, not the shares. The mean is over
  decisions with its SE across them, which already holds the playouts' sampling noise (rows from
  one trajectory are not independent, so the SE is somewhat optimistic). Split by the search move's kind, kind change, fan, phase, board and the Q gap
  the search acted on (Q(a_s) - Q(a_p)).
- **H2:** [V(a_s) - MC(a_s)] - [V(a_p) - MC(a_p)], the same with the search's Q, and the gain the
  search predicted (Q(a_s) - Q(a_p)) against the gain it got. Caution: a_s is chosen *because* its
  Q is high, so this excess is positive even from unbiased zero-mean noise (H1's winner's curse).
  A positive number does not separate H2 from H1; compare it with the size of curse the measured
  noise predicts, and with the controls' Q(a_p) - MC(a_p).
- **H1:** V(s) - MC(s) over every row (MC(s) = MC(a_p): policy-only from s plays a_p), weighted
  by 1 / keep_prob back to all decisions, with the MC sampling variance subtracted.
- **H5:** |MC(a_s) - MC(a_p)| and its sampling-corrected spread. The raw quantiles are inflated by
  sampling noise; the deconvolved "real effect sd" (with a bootstrap SE) is the H5 number.
"""
import json
import math
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def wmean_se(xs, ws=None):
    """Weighted mean and its standard error (effective-n form); (nan, nan) when empty."""
    ws = [1.0] * len(xs) if ws is None else ws
    pairs = [(x, w) for x, w in zip(xs, ws) if x is not None and not math.isnan(x)]
    if not pairs:
        return math.nan, math.nan
    sw = sum(w for _, w in pairs)
    m = sum(x * w for x, w in pairs) / sw
    if len(pairs) < 2:
        return m, math.nan
    var = sum(w * (x - m) ** 2 for x, w in pairs) / sw
    n_eff = sw * sw / sum(w * w for _, w in pairs)
    return m, math.sqrt(var * n_eff / (n_eff - 1) / n_eff) if n_eff > 1 else math.nan


def quantiles(xs, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return [math.nan] * len(qs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs]


def fmt(m, se=None, w=7):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return " " * (w - 3) + "n/a"
    s = f"{m:+.3f}"
    if se is not None and not math.isnan(se):
        s += f" ± {se:.3f}"
    return s


def verdict(r):
    d, se = r["diff_mean"], r["diff_se"]
    if se is None or se == 0:
        return "tie" if d == 0 else ("help" if d > 0 else "hurt")
    if d > 2 * se:
        return "help"
    if d < -2 * se:
        return "hurt"
    return "tie"


def q_gap(r):
    qs, qp = r["search"]["q"], r["policy"]["q"]
    return None if qs is None or qp is None else qs - qp


def fan_bucket(n):
    for hi, name in ((2, "2"), (5, "3-5"), (15, "6-15"), (40, "16-40")):
        if n <= hi:
            return name
    return "41+"


def gap_bucket(g):
    if g is None:
        return "unscored"
    for hi, name in ((0.01, "<0.01"), (0.03, "0.01-0.03"), (0.1, "0.03-0.1"), (0.3, "0.1-0.3")):
        if g < hi:
            return name
    return ">=0.3"


def phase(r):
    if r["mid_activation"]:
        return "mid-activation"
    return "turn start" if r["turn_start"] else "mid-turn"


def ledger_line(label, rs):
    n = len(rs)
    v = defaultdict(int)
    for r in rs:
        v[verdict(r)] += 1
    m, se = wmean_se([r["diff_mean"] for r in rs])
    absd = sum(abs(r["diff_mean"]) for r in rs) / n
    return (f"  {label:<26s} {n:5d}  {100 * v['help'] / n:5.1f}% {100 * v['hurt'] / n:5.1f}% "
            f"{100 * v['tie'] / n:5.1f}%   {fmt(m, se):>16s}   {absd:.3f}")


def ledger(title, rs, key, order=None):
    groups = defaultdict(list)
    for r in rs:
        groups[key(r)].append(r)
    keys = order if order else sorted(groups, key=lambda k: -len(groups[k]))
    print(f"\n  by {title}")
    for k in keys:
        if groups.get(k):
            print(ledger_line(str(k), groups[k]))


def section(name):
    print(f"\n== {name} " + "=" * max(0, 90 - len(name)))


def report(label, rows):
    over = [r for r in rows if r["override"]]
    ctrl = [r for r in rows if not r["override"]]
    playouts = sorted({r["playouts"] for r in rows})
    boards = sorted({r["board"] for r in rows})
    print(f"\n#### {label}")
    print(f"rows {len(rows)}: {len(over)} overrides, {len(ctrl)} controls; playouts {playouts}; boards {', '.join(boards)}")
    # Override rate among searched decisions, from the controls' keep probability.
    if ctrl and all(r["keep_prob"] < 1 for r in ctrl):
        est_non = sum(1 / r["keep_prob"] for r in ctrl)
        print(f"estimated override rate {len(over) / (len(over) + est_non):.3f} (controls reweighted by 1/keep_prob)")
    elif ctrl:
        print(f"override rate {len(over) / len(rows):.3f} (every decision kept)")

    if over:
        section("override ledger: MC(a_s) - MC(a_p), paired")
        print(f"  {'':<26s} {'n':>5s}   help   hurt  unres   {'mean diff ± SE':>16s}   mean|diff|")
        print(ledger_line("all overrides", over))
        ledger("search move kind", over, lambda r: r["search"]["kind"])
        ledger("kind change (policy -> search)", over,
               lambda r: "same kind" if r["search"]["kind"] == r["policy"]["kind"] else "kind changes")
        ledger("fan width", over, lambda r: fan_bucket(r["fan"]), ["2", "3-5", "6-15", "16-40", "41+"])
        ledger("phase", over, phase, ["turn start", "mid-turn", "mid-activation"])
        ledger("board", over, lambda r: r["board"])
        ledger("Q gap Q(a_s) - Q(a_p)", over, lambda r: gap_bucket(q_gap(r)),
               ["<0.01", "0.01-0.03", "0.03-0.1", "0.1-0.3", ">=0.3", "unscored"])
        ledger("search move's prior rank", over,
               lambda r: str(r["search"]["prior_rank"]) if r["search"]["prior_rank"] < 3 else "3+", ["0", "1", "2", "3+"])

        section("H2: the net's error on the search's picks vs the policy's (overrides; positive is "
                "expected from H1 noise alone, see the docstring)")
        ev_s = [r["search"]["v_mean"] - r["search"]["mc_mean"] for r in over]
        ev_p = [r["policy"]["v_mean"] - r["policy"]["mc_mean"] for r in over]
        both_q = [r for r in over if r["search"]["q"] is not None and r["policy"]["q"] is not None]
        eq_s = [r["search"]["q"] - r["search"]["mc_mean"] for r in both_q]
        eq_p = [r["policy"]["q"] - r["policy"]["mc_mean"] for r in both_q]
        print(f"  V(a_s) - MC(a_s)                       {fmt(*wmean_se(ev_s))}")
        print(f"  V(a_p) - MC(a_p)                       {fmt(*wmean_se(ev_p))}")
        print(f"  difference (V excess of search pick)   {fmt(*wmean_se([a - b for a, b in zip(ev_s, ev_p)]))}")
        print(f"  Q(a_s) - MC(a_s)                       {fmt(*wmean_se(eq_s))}   (n {len(both_q)})")
        print(f"  Q(a_p) - MC(a_p)                       {fmt(*wmean_se(eq_p))}")
        print(f"  difference (Q excess of search pick)   {fmt(*wmean_se([a - b for a, b in zip(eq_s, eq_p)]))}")
        pred = [q_gap(r) for r in both_q]
        real = [r["diff_mean"] for r in both_q]
        print(f"  predicted gain Q(a_s) - Q(a_p)         {fmt(*wmean_se(pred))}")
        print(f"  realised gain MC(a_s) - MC(a_p)        {fmt(*wmean_se(real))}")
        vpred = [r["search"]["v_mean"] - r["policy"]["v_mean"] for r in over]
        print(f"  net's one-ply gain V(a_s) - V(a_p)     {fmt(*wmean_se(vpred))}")
        if len(pred) > 2:
            mp, mr = sum(pred) / len(pred), sum(real) / len(real)
            cov = sum((p - mp) * (q - mr) for p, q in zip(pred, real))
            vp = sum((p - mp) ** 2 for p in pred)
            vr = sum((q - mr) ** 2 for q in real)
            slope = cov / vp if vp > 0 else math.nan
            corr = cov / math.sqrt(vp * vr) if vp > 0 and vr > 0 else math.nan
            print(f"  realised on predicted: slope {slope:+.3f}, corr {corr:+.3f}")

        section("H5: how big are the overrides really? |MC(a_s) - MC(a_p)|")
        absd = [abs(r["diff_mean"]) for r in over]
        q = quantiles(absd)
        print("  quantiles p10/p25/p50/p75/p90: " + " / ".join(f"{x:.3f}" for x in q))
        for t in (0.02, 0.05, 0.1):
            print(f"  share with |diff| < {t:<4}: {sum(a < t for a in absd) / len(absd):.3f}")
        d = [r["diff_mean"] for r in over]
        md = sum(d) / len(d)
        var = sum((x - md) ** 2 for x in d) / max(1, len(d) - 1)
        def effect_sd(rs):
            dd = [r["diff_mean"] for r in rs]
            mm = sum(dd) / len(dd)
            vv = sum((x - mm) ** 2 for x in dd) / max(1, len(dd) - 1)
            ss = sum((r["diff_se"] or 0.0) ** 2 for r in rs) / len(rs)
            return math.sqrt(max(0.0, vv - ss)), vv, ss
        true_sd, var, samp = effect_sd(over)
        import random
        rng = random.Random(0)
        boot = [effect_sd([rng.choice(over) for _ in over])[0] for _ in range(200)] if len(over) > 2 else []
        bsd = (sum((b - sum(boot) / len(boot)) ** 2 for b in boot) / (len(boot) - 1)) ** 0.5 if len(boot) > 1 else math.nan
        print(f"  spread of diffs: sd {math.sqrt(var):.3f}, of which sampling {math.sqrt(samp):.3f} -> "
              f"real effect sd ~{true_sd:.3f} ± {bsd:.3f} (bootstrap)")

    section("H1: value head vs Monte Carlo truth, V(s) - MC(s) (all rows, reweighted to all decisions)")
    ws = [1.0 / r["keep_prob"] for r in rows]
    err = [r["v_state"] - r["policy"]["mc_mean"] for r in rows]
    se2 = [r["policy"]["mc_se"] ** 2 if r["policy"]["mc_se"] is not None else 0.0 for r in rows]
    m, se = wmean_se(err, ws)
    sw = sum(ws)
    var = sum(w * (e - m) ** 2 for e, w in zip(err, ws)) / sw
    samp = sum(w * s for s, w in zip(se2, ws)) / sw
    net_var = var - samp
    print(f"  bias mean[V(s) - MC(s)]       {fmt(m, se)}")
    print(f"  var[V(s) - MC(s)]             {var:.4f}  (sd {math.sqrt(var):.3f})")
    print(f"  MC sampling var mean[SE^2]    {samp:.4f}  (sd {math.sqrt(samp):.3f})")
    print(f"  net error var (difference)    {net_var:.4f}  -> RMS net error ~{math.sqrt(max(0.0, net_var + m * m)):.3f} "
          f"(sd {math.sqrt(max(0.0, net_var)):.3f})")
    print("  error quantiles p10/p25/p50/p75/p90: " + " / ".join(f"{x:+.3f}" for x in quantiles(err)))
    for name, sub in (("overrides only", over), ("controls only", ctrl)):
        if len(sub) > 1:
            e = [r["v_state"] - r["policy"]["mc_mean"] for r in sub]
            mm, ss = wmean_se(e)
            v = sum((x - mm) ** 2 for x in e) / (len(e) - 1)
            sp = sum((r["policy"]["mc_se"] or 0.0) ** 2 for r in sub) / len(sub)
            print(f"  {name:<16s} n {len(sub):5d}  bias {fmt(mm, ss)}  net sd ~{math.sqrt(max(0.0, v - sp)):.3f}")
    gaps = [abs(g) for g in (q_gap(r) for r in over) if g is not None]
    if gaps:
        net_sd = math.sqrt(max(0.0, net_var))
        print("  |Q gap| the search acted on, p25/p50/p75/p90: " + " / ".join(f"{x:.3f}" for x in quantiles(gaps, (0.25, 0.5, 0.75, 0.9))))
        print(f"  overrides with |Q gap| below the net's error sd ({net_sd:.3f}): {sum(g < net_sd for g in gaps) / len(gaps):.3f}")

    if ctrl:
        section("controls (search plays the policy's move): calibration of that move")
        print(f"  V(a_p) - MC(a_p)   {fmt(*wmean_se([r['policy']['v_mean'] - r['policy']['mc_mean'] for r in ctrl]))}")
        cq = [r for r in ctrl if r["policy"]["q"] is not None]
        print(f"  Q(a_p) - MC(a_p)   {fmt(*wmean_se([r['policy']['q'] - r['policy']['mc_mean'] for r in cq]))}")
        print(f"  root - MC(s)       {fmt(*wmean_se([r['root_value'] - r['policy']['mc_mean'] for r in ctrl if r['root_value'] is not None]))}")
        by = defaultdict(list)
        for r in ctrl:
            by[phase(r)].append(r["v_state"] - r["policy"]["mc_mean"])
        for k in ("turn start", "mid-turn", "mid-activation"):
            if by.get(k):
                print(f"  V(s) - MC(s), {k:<15s} n {len(by[k]):4d}  {fmt(*wmean_se(by[k]))}")

    unfinished = sum(r["policy"]["unfinished"] + (r["search"]["unfinished"] if r["override"] else 0) for r in rows)
    if unfinished:
        print(f"\nwarning: {unfinished} playouts hit --max-steps (scored 0)")
    ms = [r.get("search_ms") for r in rows if r.get("search_ms") is not None]
    if ms:
        tot = [r["elapsed_ms"] for r in rows]
        print(f"\ncost per row: {sum(tot) / len(tot) / 1000:.1f}s (search {sum(ms) / len(ms) / 1000:.1f}s)")


def main(paths):
    rows = load(paths)
    if not rows:
        sys.exit("no rows")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"].rsplit("/", 1)[-1], r["search_config"], r["search_iters"])].append(r)
    for (model, cfg, iters), rs in sorted(groups.items()):
        report(f"{model}  search {cfg}@{iters}", rs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
