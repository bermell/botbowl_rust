#!/usr/bin/env python3
"""Plan 067: the MC value labels' noise, measured directly from a top-up's records.

`botbowl-ui mc-label --top-up K` (or `botbowl-hub job label --top-up K`) writes, beside each
topped-up shard, `topup/<shard>`: one line per trajectory with every sample's prior n-playout mean
(`prior`) and the mean of the K playouts it added (`added`). The two are independent estimates of
the same quantity (the policy's MC value of the state; disjoint dice), so with per-playout variance
s2(x) at state x,

    E[(prior - added)^2] = s2 (1/n + 1/K),

and the noise variance of an m-playout label, averaged over states, is

    E[s2] / m = mean((prior - added)^2) / (1/n + 1/K) / m        (n = K = 8, m = 8: mean(d^2) / 2).

That is the part of a value MSE measured against m-playout labels that no net can remove. Printed
per sample (unweighted) and with the trainer's per-drive weighting (each trajectory's samples
weighted 1/len, as `bbnn.train --per-drive-value-weight` weights val_value), overall and per board,
with standard errors clustered by trajectory (samples of one drive are not independent). Also the
mean of prior - added, which must be ~0: a clear offset means the two halves are not the same
quantity (another net, another backend's argmax flips, a seed that was reused).

    scripts/label_noise.py runs/exp078/mc16/gen44/topup/shard4.jsonl [...] [--val-mse 0.080]
"""
import argparse
import json
import math
import sys
from collections import defaultdict


def cluster_mean(groups):
    """Weighted mean of per-sample values and its cluster-robust SE.

    `groups` is a list of (weights, values) per trajectory. The estimator is
    sum(w*v)/sum(w); its SE treats trajectories as the independent units (a ratio
    estimator's linearisation: residuals summed per cluster).
    """
    sw = sum(sum(w) for w, _ in groups)
    if sw == 0:
        return float("nan"), float("nan")
    m = sum(sum(wi * vi for wi, vi in zip(w, v)) for w, v in groups) / sw
    g = len(groups)
    if g < 2:
        return m, float("nan")
    r = [sum(wi * (vi - m) for wi, vi in zip(w, v)) for w, v in groups]
    var = g / (g - 1) * sum(x * x for x in r) / sw ** 2
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", help="topup/<shard>.jsonl files")
    ap.add_argument("--labels", type=int, default=None,
                    help="report the noise of an m-playout label (default: the prior's n)")
    ap.add_argument("--val-mse", type=float, default=None,
                    help="a val value MSE (against n-playout labels) to put the noise next to")
    args = ap.parse_args()

    recs = []
    for path in args.records:
        with open(path) as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    if not recs:
        sys.exit("no records")
    ns = {(r["prior_playouts"], r["added_playouts"]) for r in recs}
    if len(ns) != 1:
        sys.exit(f"mixed playout counts {sorted(ns)}: pass one top-up's records")
    (n, k), = ns
    m = args.labels or n
    scale = 1.0 / ((1.0 / n + 1.0 / k) * m)   # mean(d^2) * scale = E[s2] / m

    def summary(rs):
        unw = [([1.0] * len(r["prior"]), [(a - b) ** 2 * scale for a, b in zip(r["prior"], r["added"])]) for r in rs]
        per_drive = [([1.0 / len(r["prior"])] * len(r["prior"]), v) for r, (_, v) in zip(rs, unw) if r["prior"]]
        diff = [([1.0] * len(r["prior"]), [a - b for a, b in zip(r["prior"], r["added"])]) for r in rs]
        lab = [([1.0] * len(r["prior"]), [(a * n + b * k) / (n + k) for a, b in zip(r["prior"], r["added"])]) for r in rs]
        mean_lab, _ = cluster_mean(lab)
        lab2 = [(w, [(x - mean_lab) ** 2 for x in v]) for w, v in lab]
        return {
            "trajectories": len(rs),
            "samples": sum(len(r["prior"]) for r in rs),
            "noise": cluster_mean(unw),
            "noise_per_drive": cluster_mean(per_drive),
            "offset": cluster_mean(diff),
            "label_var": cluster_mean(lab2)[0],
        }

    def line(name, s):
        nz, se = s["noise"]
        nd, sed = s["noise_per_drive"]
        off, seo = s["offset"]
        return (f"{name:>10}  {s['trajectories']:5d} traj {s['samples']:7d} samples  "
                f"noise {nz:.4f} ± {se:.4f}  per-drive {nd:.4f} ± {sed:.4f}  "
                f"offset {off:+.4f} ± {seo:.4f}  label var {s['label_var']:.4f}")

    print(f"# {len(args.records)} file(s); prior {n} playouts + added {k}; noise = E[s2]/{m}, "
          f"the variance of a {m}-playout label around the state's MC value (clustered SE by trajectory)")
    total = summary(recs)
    by_board = defaultdict(list)
    for r in recs:
        by_board[r.get("board", "?")].append(r)
    for board in sorted(by_board, key=lambda b: (len(by_board[b]), b)):
        if len(by_board[board]) >= 5:
            print(line(board, summary(by_board[board])))
    print(line("all", total))
    nz, se = total["noise"]
    nd, sed = total["noise_per_drive"]
    for mm in sorted({m, 8, 16}):
        print(f"  a {mm:2d}-playout label: noise {nz * m / mm:.4f} ± {se * m / mm:.4f} "
              f"(per-drive {nd * m / mm:.4f} ± {sed * m / mm:.4f})")
    if args.val_mse is not None:
        print(f"  val value MSE {args.val_mse:.4f}: label noise (per-drive weighting, as val_value) is "
              f"{nd / args.val_mse:.0%} of it; the rest, {args.val_mse - nd:.4f}, is the net's error vs the MC value")
    print(f"LABEL_NOISE n={n} k={k} m={m}: {nz:.4f} ± {se:.4f} per sample, {nd:.4f} ± {sed:.4f} per drive "
          f"over {total['samples']} samples / {total['trajectories']} trajectories; "
          f"offset {total['offset'][0]:+.4f} ± {total['offset'][1]:.4f}")


if __name__ == "__main__":
    main()
