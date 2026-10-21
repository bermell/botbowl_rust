#!/usr/bin/env python3
"""exp080 / plan 069 §1: per-descent and per-forward table from perf_search_bench.sh outputs.

    scripts/exp080_table.py OUT_DIR [OUT_DIR ...]

Reads every TAG.stat (perf stat -x, output) with its TAG.counts, subtracts the start-up of the
heur/null arms (TAG `*_0` in the same dir; the in-process arms are counted after the load), and
prints instructions (user, kernel), cycles, IPC, effective clock and CPU-µs per descent and per
forward.
"""
import glob
import os
import re
import sys


def stat(path):
    v = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.strip().split(",")
        try:
            val = float(p[0])
        except ValueError:
            continue
        name = p[2] if p[2] else p[3]
        if name == "msec":
            name = p[3]
            val = float(p[0] + "." + p[1]) if p[1].isdigit() else val
        v[name] = val
    return v


def counts(path):
    c = {}
    for line in open(path):
        for kv in line.split():
            if "=" in kv:
                k, x = kv.split("=", 1)
                try:
                    c[k] = float(x)
                except ValueError:
                    pass
    return c


def main():
    rows = []
    for d in sys.argv[1:]:
        base = {}
        for f in glob.glob(os.path.join(d, "*_0.stat")):
            base[os.path.basename(f).split("_")[0]] = stat(f)
        for f in sorted(glob.glob(os.path.join(d, "*.stat"))):
            tag = os.path.basename(f)[:-5]
            if tag.endswith("_0") or not os.path.exists(f[:-5] + ".counts"):
                continue
            s, c = stat(f), counts(f[:-5] + ".counts")
            kind = tag.split("_")[0]
            b = base.get(kind, {}) if kind in ("heur", "null") else {}
            iu = s.get("instructions:u", 0) - b.get("instructions:u", 0)
            ik = s.get("instructions:k", 0) - b.get("instructions:k", 0)
            cu = s.get("cycles:u", 0) - b.get("cycles:u", 0)
            ck = s.get("cycles:k", 0) - b.get("cycles:k", 0)
            ms = s.get("task-clock", 0) - b.get("task-clock", 0)
            de, fw = c.get("descents", 0), c.get("forwards", 0)
            if not de:
                continue
            ghz = (cu + ck) / (ms * 1e6) if ms else 0
            rows.append((d, tag, iu, ik, cu, ck, ms, de, fw, ghz, c.get("decisions", 0), c.get("searches", 0)))
    print(f"{'dir/tag':38s} {'Mins/desc u':>11s} {'k':>6s} {'IPC u':>6s} {'GHz':>5s} {'us/desc':>8s} "
          f"{'fw/desc':>7s} {'Mins/fw u':>9s} {'k':>6s} {'us/fw':>7s} {'fw/search':>9s}")
    for d, tag, iu, ik, cu, ck, ms, de, fw, ghz, dec, se in rows:
        name = os.path.basename(d.rstrip("/")) + "/" + tag
        us_desc = ms * 1e3 / de if de else 0
        print(f"{name:38s} {iu / de / 1e6:11.3f} {ik / de / 1e6:6.3f} {iu / cu if cu else 0:6.2f} {ghz:5.2f} "
              f"{us_desc:8.1f} {fw / de if de else 0:7.3f} "
              f"{(iu / fw / 1e6) if fw else 0:9.3f} {(ik / fw / 1e6) if fw else 0:6.3f} "
              f"{(ms * 1e3 / fw) if fw else 0:7.1f} {fw / se if se else 0:9.1f}")


if __name__ == "__main__":
    main()
