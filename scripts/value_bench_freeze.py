#!/usr/bin/env python3
"""Plan 056 §2: freeze override-audit rows into a value benchmark, one line per distinct state.

Every audit row's `policy.mc_*` is MC(s): the policy-only drive outcome from s, under the auditing
net's policy, in the mover's frame (override_audit.rs). Rows of the same state (two searches over
the same seeded sample) are merged; their playouts share dice seeds, so they are not independent,
and the merge averages both the mean and the SE instead of pooling them.

    scripts/value_bench_freeze.py runs/exp065/g05_f1000.jsonl runs/exp065/g05_f4000.jsonl \
        --out runs/value_bench/g05_gen06.jsonl

Each line: id (`corpus:line:sample`), corpus, line (1-based), sample, board, mover, half, turn,
phase (turn_start / mid_turn / mid_activation), fan, mc, mc_se, playouts, mc_model (whose policy
the MC played), v_ref (that net's own V(s), which `botbowl-ui value-bench` reproduces on it).
"""
import argparse
import json
import os
import sys
from collections import OrderedDict


def phase_of(r):
    if r["mid_activation"]:
        return "mid_activation"
    return "turn_start" if r["turn_start"] else "mid_turn"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    states = OrderedDict()
    for path in a.rows:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["corpus"], int(r["decision"].split(":")[1]), r["sample"])
            states.setdefault(key, []).append(r)
    models = {r["model"] for rs in states.values() for r in rs}
    if len(models) != 1:
        sys.exit(f"rows come from {len(models)} nets ({sorted(models)}); a benchmark is one policy's MC")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    differ = 0
    with open(a.out, "w") as f:
        for (corpus, line, sample), rs in states.items():
            r = rs[0]
            mcs = [x["policy"]["mc_mean"] for x in rs]
            differ += len(set(mcs)) > 1
            f.write(json.dumps({
                "id": f"{corpus}:{line}:{sample}",
                "corpus": corpus,
                "line": line,
                "sample": sample,
                "board": r["board"],
                "mover": r["mover"],
                "half": r["half"],
                "turn": r["turn"],
                "phase": phase_of(r),
                "fan": r["fan"],
                "mc": sum(mcs) / len(mcs),
                "mc_se": sum(x["policy"]["mc_se"] for x in rs) / len(rs),
                "playouts": r["playouts"],
                "mc_model": r["model"],
                "v_ref": r["v_state"],
            }) + "\n")
    print(f"froze {len(states)} states from {sum(map(len, states.values()))} rows into {a.out}"
          f" ({differ} merged states whose MC differed between rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
