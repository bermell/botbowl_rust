#!/usr/bin/env python3
"""Plan 039: did the fine-tune start learning again?

The primary column is **spread** — max-min of each validation series across the
whole run. Since gen04 the production fine-tune has moved val_policy by ~0.005
and val_value by ~0.009 over fifteen epochs, which is why its restore step is
noise. An arm that fixes mechanism 1 has to move more than that; an arm that
does not has not fixed it, whatever its absolute loss says.

`restore_frac` is where the restore landed as a fraction of the budget. On a
flat curve it is uniform noise, so values near 0 or 1 across arms mean nothing
on their own — read them next to the spread.

    scripts/exp039_report.py runs/exp039
"""

import re
import sys
from pathlib import Path

STEP = re.compile(r"^step\s+(\d+).*?val_policy\s+([\d.]+)\s+val_value\s+([\d.]+)")
RESTORE = re.compile(r"^restored best-val weights: step (\d+)", re.M)

# What production currently does, for the comparison that matters.
REF = {"val_policy": 0.0050, "val_value": 0.0077, "label": "gen07 production"}


def parse(path):
    txt = path.read_text()
    rows = [(int(m[1]), float(m[2]), float(m[3])) for l in txt.splitlines() if (m := STEP.match(l))]
    m = RESTORE.search(txt)
    if not rows or not m:
        return None
    vp = [r[1] for r in rows]
    vv = [r[2] for r in rows]
    rs = int(m[1])
    at = {s: (a, b) for s, a, b in rows}
    return {
        "arm": path.name.removesuffix(".train.log"),
        "n": len(rows),
        "vp_spread": max(vp) - min(vp),
        "vv_spread": max(vv) - min(vv),
        "vp_at": at[rs][0],
        "vv_at": at[rs][1],
        "vp_best": min(vp),
        "restore": rs,
        "last": rows[-1][0],
    }


def main(argv):
    out = Path(argv[1] if len(argv) > 1 else "runs/exp039")
    rows = [r for r in (parse(p) for p in sorted(out.glob("*.train.log"))) if r]
    if not rows:
        print(f"no finished arms in {out}")
        return 1
    order = {"w3_warm": 0, "wide_warm": 1, "wide_scratch": 2}
    rows.sort(key=lambda r: order.get(r["arm"], 9))
    w = max(len(r["arm"]) for r in rows)
    print(f"{'arm':<{w}}  {'ckpts':>5}  {'vp_spread':>10}  {'vv_spread':>10}  "
          f"{'val_pol@r':>10}  {'val_val@r':>10}  {'restore':>9}  {'frac':>5}")
    for r in rows:
        frac = r["restore"] / r["last"] if r["last"] else 0
        print(f"{r['arm']:<{w}}  {r['n']:>5}  {r['vp_spread']:>10.4f}  {r['vv_spread']:>10.4f}  "
              f"{r['vp_at']:>10.4f}  {r['vv_at']:>10.4f}  {r['restore']:>9}  {frac:>5.2f}")
    print()
    print(f"reference — {REF['label']}: vp_spread {REF['val_policy']:.4f}, "
          f"vv_spread {REF['val_value']:.4f} (a flat curve; restore step is noise)")
    print()
    base = next((r for r in rows if r["arm"] == "w3_warm"), None)
    for r in rows:
        verdict = "MOVED" if r["vp_spread"] > 3 * REF["val_policy"] else "flat"
        extra = ""
        if base and r is not base:
            extra = f", val_policy@restore {r['vp_at'] - base['vp_at']:+.4f} vs w3_warm"
        print(f"  {r['arm']:<{w}} {verdict}{extra}")
    print()
    print("A 'flat' arm has not addressed mechanism 1 and buys no games.")
    print("A lower val_policy than w3_warm is necessary but not sufficient — plan 032's")
    print("ground rule: val numbers select the arm, games decide it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
