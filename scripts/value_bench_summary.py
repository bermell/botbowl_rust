#!/usr/bin/env python3
"""Plan 056 §2: summarise `botbowl-ui value-bench` outputs — the value head against MC truth.

    scripts/value_bench_summary.py [NAME=]out.jsonl [[NAME=]other.jsonl ...]

Per net, overall and by board and by phase:
- bias  mean[V(s) - MC(s)] with its SE;
- RMS   the value head's per-state error with MC's own sampling noise removed:
        sqrt(mean[(V - MC)^2] - mean[mc_se^2]), SE by the delta method;
- scatter  the RMS left after the bias, sqrt(RMS^2 - bias^2);
- self-check  max |V - v_ref| when the net is the one the benchmark was frozen from (should be ~0:
  `v_ref` is the audit's own V(s) on it; anything else means the replay or the net is not the one).

With two or more outputs, every later net is also compared to the FIRST on the states both scored:
the paired difference of squared errors (the MC noise cancels, so it resolves far smaller changes
than two separate RMS figures), the RMS ratio, and the bias change. One `VALUE_BENCH` line per net
is for scripts.
"""
import json
import math
import sys
from collections import OrderedDict


def load(arg):
    name, _, path = arg.rpartition("=")
    rows = OrderedDict()
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return (name or path.rsplit("/", 1)[-1].removesuffix(".jsonl")), rows


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return math.nan, math.nan
    m = sum(xs) / n
    if n < 2:
        return m, math.nan
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def stats(rs):
    err = [r["v"] - r["mc"] for r in rs]
    sq = [e * e for e in err]
    noise = sum(r.get("mc_se", 0.0) ** 2 for r in rs) / max(len(rs), 1)
    bias, bias_se = mean_se(err)
    mse, mse_se = mean_se(sq)
    var = max(mse - noise, 0.0)
    rms = math.sqrt(var)
    rms_se = mse_se / (2 * rms) if rms > 0 else math.nan
    scatter = math.sqrt(max(var - bias * bias, 0.0))
    return dict(n=len(rs), bias=bias, bias_se=bias_se, rms=rms, rms_se=rms_se, scatter=scatter, noise=noise)


def fmt(s):
    return (f"n {s['n']:5d}  RMS {s['rms']:.3f} ± {s['rms_se']:.3f}  bias {s['bias']:+.3f} ± {s['bias_se']:.3f}"
            f"  scatter {s['scatter']:.3f}")


def groups(rows, key):
    out = OrderedDict()
    for r in sorted(rows, key=lambda r: str(r.get(key))):
        out.setdefault(str(r.get(key)), []).append(r)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    nets = [load(a) for a in sys.argv[1:]]
    lines = []
    for name, rows in nets:
        rs = list(rows.values())
        s = stats(rs)
        print(f"== {name}  ({rs[0].get('model', '?') if rs else '?'})")
        print(f"  all             {fmt(s)}   (MC noise var {s['noise']:.4f} removed)")
        for key in ("board", "phase"):
            for k, g in groups(rs, key).items():
                print(f"  {k:<15} {fmt(stats(g))}")
        refs = [abs(r["v"] - r["v_ref"]) for r in rs if r.get("v_ref") is not None]
        if refs:
            same = sum(d < 0.0015 for d in refs)
            print(f"  self-check: |V - v_ref| max {max(refs):.4f}, {same}/{len(refs)} within 0.0015"
                  + ("  (this is the benchmark's own net)" if same > 0.99 * len(refs) else ""))
        print()
        lines.append(f"VALUE_BENCH {name}: n {s['n']} RMS {s['rms']:.4f} ± {s['rms_se']:.4f}"
                     f" bias {s['bias']:+.4f} ± {s['bias_se']:.4f} scatter {s['scatter']:.4f}")
    if len(nets) > 1:
        ref_name, ref = nets[0]
        print(f"== paired against {ref_name} (same states, the MC noise cancels)")
        for name, rows in nets[1:]:
            ids = [i for i in rows if i in ref]
            if not ids:
                print(f"  {name}: no states in common")
                continue
            a = [ref[i] for i in ids]
            b = [rows[i] for i in ids]
            d_sq = [(y["v"] - y["mc"]) ** 2 - (x["v"] - x["mc"]) ** 2 for x, y in zip(a, b)]
            d_err = [y["v"] - x["v"] for x, y in zip(a, b)]
            dm, dse = mean_se(d_sq)
            sa, sb = stats(a), stats(b)
            ratio = sb["rms"] / sa["rms"] - 1 if sa["rms"] > 0 else math.nan
            db, dbse = mean_se(d_err)
            print(f"  {name:<20} n {len(ids)}  dMSE {dm:+.4f} ± {dse:.4f}  RMS {sa['rms']:.3f} -> {sb['rms']:.3f}"
                  f" ({100 * ratio:+.1f}%)  bias {sa['bias']:+.3f} -> {sb['bias']:+.3f} (d {db:+.3f} ± {dbse:.3f})")
            lines.append(f"VALUE_BENCH_PAIRED {name} vs {ref_name}: n {len(ids)} dMSE {dm:+.4f} ± {dse:.4f}"
                         f" dRMS {100 * ratio:+.1f}% dbias {db:+.4f} ± {dbse:.4f}")
        print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
