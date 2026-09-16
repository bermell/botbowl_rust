#!/usr/bin/env python3
"""Plan 036 E1-E3: turn the arms' training logs into the plan's table.

Reports, per arm, exactly what the plan's Experiments section asks for:

    restore_step, val_policy@restore, val_value@restore, val_value@end

plus two derived columns that are the actual success signals:

    drift  = val_value@end - val_value@restore
             how far the value head runs away after its optimum. E2's signal
             is that this shrinks.
    pgap   = policy_optimum_step - restore_step
             the selection-rule question. Plan 036: if this still exceeds
             ~5k steps after W1-W4, the restore wants splitting between the
             heads.

The val_* numbers select the arm for E4; they do not decide it (plan 032's
ground rule). Nothing here plays a game.

    scripts/exp036_report.py runs/exp036
"""

import re
import sys
from pathlib import Path

STEP = re.compile(
    r"^step\s+(\d+)\s+.*?val_policy\s+([\d.]+)\s+val_value\s+([\d.]+)\s+val_top1\s+([\d.]+)"
)
RESTORE = re.compile(r"^restored best-val weights: step (\d+) epoch (\d+)")
POLICY_OPT = re.compile(r"^policy-only optimum: step (\d+) \(val_policy ([\d.]+)\)")


def parse(path):
    """One arm's log -> its row, or None if it never finished."""
    steps = []
    restore_step = policy_step = None
    for line in path.read_text().splitlines():
        m = STEP.match(line)
        if m:
            steps.append((int(m[1]), float(m[2]), float(m[3])))
            continue
        m = RESTORE.match(line)
        if m:
            restore_step = int(m[1])
            continue
        m = POLICY_OPT.match(line)
        if m:
            policy_step = int(m[1])
    if restore_step is None or not steps:
        return None
    at = {s: (vp, vv) for s, vp, vv in steps}
    vp_r, vv_r = at.get(restore_step, (float("nan"), float("nan")))
    _, vv_end = steps[-1][1], steps[-1][2]
    return {
        "arm": path.name.removesuffix(".train.log"),
        "restore_step": restore_step,
        "val_policy": vp_r,
        "val_value": vv_r,
        "val_value_end": vv_end,
        "drift": vv_end - vv_r,
        "pgap": (policy_step - restore_step) if policy_step is not None else None,
        "checkpoints": len(steps),
    }


def main(argv):
    out = Path(argv[1] if len(argv) > 1 else "runs/exp036")
    rows = [r for r in (parse(p) for p in sorted(out.glob("*.train.log"))) if r]
    if not rows:
        print(f"no finished arms in {out}")
        return 1

    # The baseline is the reference every other row is read against, so put it
    # first whatever it sorts as.
    rows.sort(key=lambda r: (r["arm"] != "baseline", r["arm"]))
    base = rows[0] if rows[0]["arm"] == "baseline" else None

    w = max(len(r["arm"]) for r in rows)
    print(f"{'arm':<{w}}  {'restore':>8}  {'val_pol':>8}  {'val_val':>8}  "
          f"{'val@end':>8}  {'drift':>8}  {'pgap':>7}  {'vs base':>8}")
    for r in rows:
        later = ""
        if base and base["restore_step"]:
            # E1's success signal: "the value minimum moves >= 2x later".
            later = f"{r['restore_step'] / base['restore_step']:.2f}x"
        pgap = "n/a" if r["pgap"] is None else f"{r['pgap']:+d}"
        print(f"{r['arm']:<{w}}  {r['restore_step']:>8}  {r['val_policy']:>8.4f}  "
              f"{r['val_value']:>8.4f}  {r['val_value_end']:>8.4f}  {r['drift']:>+8.4f}  "
              f"{pgap:>7}  {later:>8}")

    print()
    print("restore = step the best-val (policy+value) checkpoint came from")
    print("drift   = val_value@end - val_value@restore (E2 wants this smaller)")
    print("pgap    = policy-only optimum - restore step (>5k => split the restore)")
    print("vs base = restore step relative to the baseline arm (E1 wants >= 2x)")
    if any(r["checkpoints"] < 3 for r in rows):
        print("\nNOTE: an arm has <3 validation points — --eval-every is too coarse "
              "for the window, and the restore step is quantised to it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
