#!/usr/bin/env python3
"""Plan 031 D3 — is the NN encoding partially observable for the active activation?

Hashes every prepared sample's (spatial, global) bytes, groups the collisions, and
asks whether rows with a *byte-identical input tensor* carry different legal-action
sets (`actions.npy` sliced by `action_offsets.npy`) or different value targets.

An "ambiguous" group is a collision group whose members do not all share the same
legal-action set: the network literally cannot tell those states apart, so the
policy mask it is trained under is not a function of its input.

Usage:
    python3 scripts/audit_encoding_collisions.py PREPARED_DIMS_DIR [--examples N]

Reads everything with mmap; peak RSS is O(N) in the 16-byte digests, not in
`spatial.npy` (which is multiple GB).
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Mirrors botbowl-nn/src/actions.rs (PosAT -> 0..14, SimpleAT -> 14..30).
POS_AT = [
    "StartMove", "StartBlitz", "StartPass", "StartFoul", "SelectPosition",
    "Push", "FollowUp", "StartHandoff", "Handoff", "Pass", "Move", "Foul",
    "StartBlock", "Block",
]
SIMPLE_AT = [
    "SelectBothDown", "SelectPow", "SelectPush", "SelectPowPush", "SelectSkull",
    "UseReroll", "DontUseReroll", "EndPlayerTurn", "EndTurn", "Heads", "Tails",
    "Kick", "Receive", "SetupLine", "EndSetup", "KickoffAimMiddle",
]


def chan_name(c):
    return POS_AT[c] if c < len(POS_AT) else SIMPLE_AT[c - len(POS_AT)]


def act_str(row):
    c, y, x, simple = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    return chan_name(c) if simple else f"{chan_name(c)}({x},{y})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_dir", type=Path)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048, help="rows hashed per read")
    ap.add_argument("--cache", type=Path, default=None,
                    help="npy file to store/reuse the per-row digests (rehashing is the slow part)")
    args = ap.parse_args()

    d = args.prepared_dir
    manifest = json.loads((d / "manifest.json").read_text())
    spatial = np.load(d / "spatial.npy", mmap_mode="r")
    glob = np.load(d / "global.npy", mmap_mode="r")
    actions = np.load(d / "actions.npy", mmap_mode="r")
    offsets = np.load(d / "action_offsets.npy", mmap_mode="r")
    value = np.load(d / "value.npy", mmap_mode="r")

    n = spatial.shape[0]
    assert glob.shape[0] == n and value.shape[0] == n and offsets.shape[0] == n + 1
    assert offsets[-1] == actions.shape[0], "CSR offsets do not cover actions.npy"
    print(f"N={n} spatial={spatial.shape} global={glob.shape} "
          f"actions={actions.shape} M={int(offsets[-1])}", file=sys.stderr)

    # --- 1. hash each row's spatial+global bytes ---
    if args.cache and args.cache.exists():
        dig = np.load(args.cache)
        assert dig.shape[0] == n, "cached digest count does not match N"
        print(f"reusing cached digests from {args.cache}", file=sys.stderr)
    else:
        dig = np.zeros((n, 16), dtype=np.uint8)
        for lo in range(0, n, args.chunk):
            hi = min(lo + args.chunk, n)
            sp = np.ascontiguousarray(spatial[lo:hi]).view(np.uint8).reshape(hi - lo, -1)
            gl = np.ascontiguousarray(glob[lo:hi]).view(np.uint8).reshape(hi - lo, -1)
            for i in range(hi - lo):
                h = hashlib.blake2b(digest_size=16)
                h.update(sp[i].tobytes())
                h.update(gl[i].tobytes())
                dig[lo + i] = np.frombuffer(h.digest(), dtype=np.uint8)
            if (lo // args.chunk) % 20 == 0:
                print(f"  hashed {hi}/{n}", file=sys.stderr)
        if args.cache:
            np.save(args.cache, dig)

    groups = defaultdict(list)
    for i, dg in enumerate(dig.tobytes()[j:j + 16] for j in range(0, 16 * n, 16)):
        groups[dg].append(i)
    colliding = [rows for rows in groups.values() if len(rows) > 1]

    # --- 2. classify collision groups ---
    off = np.asarray(offsets)
    val = np.asarray(value, dtype=np.float64)

    def legal_set(i):
        a = actions[off[i]:off[i + 1]]
        return frozenset(map(tuple, np.asarray(a).tolist()))

    ambiguous = []          # groups with >1 distinct legal set
    same_set_groups = []    # collision groups where every member shares the legal set
    for rows in colliding:
        sets = {}
        for i in rows:
            sets.setdefault(legal_set(i), []).append(i)
        if len(sets) > 1:
            ambiguous.append((rows, sets))
        else:
            same_set_groups.append(rows)

    rows_in_collision = sum(len(r) for r in colliding)
    rows_in_ambiguous = sum(len(r) for r, _ in ambiguous)

    # --- 3. value-target variance decomposition ---
    overall_var = float(val.var())
    def within_var(group_lists):
        tot, cnt = 0.0, 0
        for rows in group_lists:
            v = val[rows]
            tot += float(((v - v.mean()) ** 2).sum())
            cnt += len(rows)
        return (tot / cnt if cnt else float("nan")), cnt

    w_all, c_all = within_var(colliding)
    w_amb, c_amb = within_var([r for r, _ in ambiguous])
    w_same, c_same = within_var(same_set_groups)

    # Global irreducible MSE: every row predicted by its group mean
    # (singletons contribute 0).
    resid = 0.0
    for rows in colliding:
        v = val[rows]
        resid += float(((v - v.mean()) ** 2).sum())
    irreducible_mse = resid / n

    print()
    print(f"prepared dir : {d}")
    print(f"git_commit   : {manifest.get('git_commit')} dirty={manifest.get('git_dirty')}")
    print(f"total rows N : {n}")
    print(f"distinct tensors            : {len(groups)}")
    print(f"collision groups (size>1)   : {len(colliding)}")
    print(f"rows in a collision group   : {rows_in_collision} ({rows_in_collision/n:.4%})")
    print(f"ambiguous groups            : {len(ambiguous)}")
    print(f"rows in an ambiguous group  : {rows_in_ambiguous} ({rows_in_ambiguous/n:.4%})")
    print()
    print(f"value var, overall                       : {overall_var:.6f}")
    print(f"value var within collision groups        : {w_all:.6f}  (n={c_all})")
    print(f"  ... within ambiguous groups            : {w_amb:.6f}  (n={c_amb})")
    print(f"  ... within same-legal-set groups       : {w_same:.6f}  (n={c_same})")
    print(f"irreducible value MSE from exact dupes   : {irreducible_mse:.6f} "
          f"({irreducible_mse/overall_var:.4%} of overall var)")

    # --- 4. characterise the ambiguous groups ---
    diff_counter = defaultdict(int)   # channel-name -> # of ambiguous groups where it differs
    size_hist = defaultdict(int)
    for rows, sets in ambiguous:
        size_hist[len(rows)] += 1
        keys = list(sets.keys())
        union = set().union(*keys)
        inter = set(keys[0]).intersection(*keys[1:]) if len(keys) > 1 else set(keys[0])
        for a in union - inter:
            diff_counter[chan_name(int(a[0]))] += 1

    print()
    print("action types that appear in some but not all members of an ambiguous group")
    print("(count = number of ambiguous groups in which that action type differs):")
    for name, cnt in sorted(diff_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>16} : {cnt}")

    print()
    print("ambiguous group size histogram:", dict(sorted(size_hist.items())))

    # --- 4b. what *kind* of decision point is each member? ---
    # Signature = the set of action-type names offered. Two members with the
    # same signature but different squares = a genuinely same-kind ambiguity
    # (the "missing action-type / mid-activation state" hypothesis); different
    # signatures = two different procedure-stack states sharing a board.
    cat = defaultdict(int)
    same_sig_groups = 0
    disjoint_groups = 0
    idx_gaps = defaultdict(int)
    for rows, sets in ambiguous:
        sigs = [tuple(sorted({chan_name(int(a[0])) for a in k})) for k in sets]
        cat[tuple(sorted(sigs))] += 1
        if len(set(sigs)) == 1:
            same_sig_groups += 1
        keys = list(sets.keys())
        if all(not (set(a) & set(b)) for i, a in enumerate(keys) for b in keys[i + 1:]):
            disjoint_groups += 1
        idx_gaps[min(rows[j + 1] - rows[j] for j in range(len(rows) - 1))] += 1

    print()
    print(f"ambiguous groups whose members offer the SAME action-type signature : "
          f"{same_sig_groups} ({same_sig_groups/max(1,len(ambiguous)):.2%})")
    print(f"ambiguous groups whose members' legal sets are pairwise DISJOINT    : "
          f"{disjoint_groups} ({disjoint_groups/max(1,len(ambiguous)):.2%})")
    print("min row-index gap inside an ambiguous group (1 = consecutive samples):",
          dict(sorted(idx_gaps.items())[:8]),
          "... " + str(sum(v for k, v in idx_gaps.items() if k > 8)) + " groups with gap>8")

    print()
    print("top 20 ambiguous-group categories (multiset of member action-type signatures):")
    for sig, cnt in sorted(cat.items(), key=lambda kv: -kv[1])[:20]:
        pretty = "  ||  ".join("+".join(s) for s in sig)
        print(f"  {cnt:>7}  {pretty}")

    print()
    print(f"--- {min(args.examples, len(ambiguous))} example ambiguous groups ---")
    for gi, (rows, sets) in enumerate(ambiguous[:args.examples]):
        print(f"\n[{gi}] rows={rows} values={[float(val[i]) for i in rows]}")
        keys = list(sets.keys())
        inter = set(keys[0]).intersection(*keys[1:])
        print(f"    shared actions ({len(inter)}): "
              f"{sorted(act_str(a) for a in inter)[:12]}")
        for k, members in sets.items():
            extra = sorted(act_str(a) for a in (set(k) - inter))
            print(f"    rows {members}: n_legal={len(k)} unique={extra}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
