#!/usr/bin/env python3
"""Plan 031 D4: how prior-dominated is the search under `--evaluator nn`?

NN priors are `softmax(logits) * len(legal)` (`botbowl-nn/src/eval.rs`), so a
confident action among 60 legal ones gets `P ~ 30` while the scripted priors
`PUCT_C = 10` was tuned against live in `[0.2, 10]` (`botbowl-mcts/src/priors.rs`).
If that spread makes PUCT follow the prior rather than search, the visit
distribution under `nn` collapses onto the prior's argmax and most children
never get a second visit.

`nn-value` is the control: identical leaf values, scripted priors. Feed it two
`botbowl-ui convergence` dumps of the same states at the same budget:

    scripts/audit_root_priors.py runs/audit031/d4-nn.jsonl runs/audit031/d4-nnvalue.jsonl

Per root we report

  visit_entropy      Shannon entropy (nats) of visits normalised over children.
  norm_entropy       …divided by log(n_children), so fan widths are comparable.
  singleton_share    share of children left at <= 1 visit — the FPU sweep only.
  top_prior_share    visits of the argmax-prior child / total child visits.
  q_eq_prior         argmax-mover-Q child == argmax-prior child.

`q` in a `ChildStat` is **Home-centric** (`botbowl-data/src/lib.rs`: "Home
maximises, Away minimises"), so it is put in the mover's frame before any
argmax — `pick_best_action` applies exactly this sign.
"""
import json
import math
import statistics
import sys
from collections import defaultdict

# Fan-width strata, matching plan 031 D1/D2 so the diagnostics line up.
BUCKETS = [(0, 10, "<=10"), (11, 30, "11-30"), (31, 60, "31-60"), (61, 10**9, ">60")]


def bucket(n):
    for lo, hi, label in BUCKETS:
        if lo <= n <= hi:
            return label
    raise AssertionError(n)


def root_metrics(row):
    """Per-root summary, or None if the root is degenerate (one child)."""
    children = row["children"]
    if len(children) < 2:
        return None
    sign = 1 if row["to_move"] == "Home" else -1

    visits = [c["visits"] for c in children]
    total = sum(visits)
    if total == 0:
        return None
    entropy = 0.0
    for v in visits:
        if v > 0:
            p = v / total
            entropy -= p * math.log(p)

    # Priors are None on chance edges; a root is always a player node, so a
    # missing prior here means something is wrong rather than "skip it".
    priors = [c["prior"] for c in children]
    assert all(p is not None for p in priors), f"root with a chance edge: seed {row['state_seed']}"
    top_prior = max(range(len(children)), key=lambda i: priors[i])

    # Unscored children (q is None) rank below every scored one, as in
    # pick_best_action's `unwrap_or((0, 0, 0))`.
    def q_key(i):
        q = children[i]["q"]
        return (0, 0) if q is None else (1, sign * q)

    top_q = max(range(len(children)), key=q_key)

    return {
        "n": len(children),
        "bucket": bucket(len(children)),
        "root_visits": row["root_visits"],
        "child_visits": total,
        "visit_entropy": entropy,
        "norm_entropy": entropy / math.log(len(children)),
        "singleton_share": sum(1 for v in visits if v <= 1) / len(visits),
        "top_prior_share": visits[top_prior] / total,
        # Both raw shares above are mechanically larger at small fan widths,
        # so they are unreadable across strata on their own. Divide out the
        # uniform baseline 1/n: 1.0 means "the argmax-prior child got exactly
        # its uniform share of the visits / agreed with argmax-Q exactly as
        # often as chance", and anything well above 1 is prior domination.
        "top_prior_lift": (visits[top_prior] / total) * len(children),
        "q_eq_prior_lift": float(top_q == top_prior) * len(children),
        "top_prior_value": max(priors),
        "prior_spread": max(priors) - min(priors),
        "q_eq_prior": float(top_q == top_prior),
    }


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(paths, max_roots):
    """Read `Sample`s out of generation shards as if they were probe rows.

    The probe above can only see *turn-start* roots — `generate_random_start`
    hands back the beginning of a team turn — while production searches from
    every decision, most of them mid-activation with a different fan shape.
    The corpus has the real distribution, and it also carries a natural
    control: gen04 was generated with `--evaluator nn-value` and gen05-07
    with `--evaluator nn`, off the *same* champion (`bbnet_14x7_gen03.onnx`)
    at the same 1000 iterations (`runs/loop14x7/status.md`). Different seed
    bases, so this is unpaired — but it is production scale.

    Streams and stops at `max_roots`; a shard is ~110 MB of JSON.
    """
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                game = json.loads(line)
                for s in game["samples"]:
                    info = game["meta"]
                    rows.append(
                        {
                            "state_seed": info.get("seed"),
                            "repeat": 0,
                            "budget": 1000,
                            "n_legal_actions": len(s["children"]),
                            "to_move": s["to_move"],
                            "children": s["children"],
                            "root_visits": s["root_visits"],
                        }
                    )
                    if len(rows) >= max_roots:
                        return rows
    return rows


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def table(name, metrics):
    keys = [
        "norm_entropy",
        "singleton_share",
        "top_prior_share",
        "top_prior_lift",
        "q_eq_prior",
        "q_eq_prior_lift",
        "prior_spread",
    ]
    print(f"\n### {name}  (n_roots = {len(metrics)})")
    header = f"{'stratum':<8} {'roots':>5} " + " ".join(f"{k[:13]:>13}" for k in keys)
    print(header)
    print("-" * len(header))
    by_bucket = defaultdict(list)
    for m in metrics:
        by_bucket[m["bucket"]].append(m)
    for _, _, label in BUCKETS:
        ms = by_bucket.get(label)
        if not ms:
            continue
        print(f"{label:<8} {len(ms):>5} " + " ".join(f"{mean([m[k] for m in ms]):>13.4f}" for k in keys))
    print(f"{'ALL':<8} {len(metrics):>5} " + " ".join(f"{mean([m[k] for m in metrics]):>13.4f}" for k in keys))


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    # `--corpus NAME=shard,shard,... [--corpus ...]` reads generation shards
    # instead of probe dumps; `--max-roots N` caps each corpus arm.
    corpus_args, probe_paths, max_roots = [], [], 40000
    i = 0
    while i < len(argv):
        if argv[i] == "--corpus":
            corpus_args.append(argv[i + 1])
            i += 2
        elif argv[i] == "--max-roots":
            max_roots = int(argv[i + 1])
            i += 2
        else:
            probe_paths.append(argv[i])
            i += 1

    arms = {}
    for path in probe_paths:
        rows = load(path)
        name = path.split("/")[-1].removesuffix(".jsonl")
        metrics = [m for m in (root_metrics(r) for r in rows) if m is not None]
        arms[name] = (rows, metrics, len(rows) - len(metrics))
    for spec in corpus_args:
        name, _, paths = spec.partition("=")
        rows = load_corpus(paths.split(","), max_roots)
        metrics = [m for m in (root_metrics(r) for r in rows) if m is not None]
        arms[name] = (rows, metrics, len(rows) - len(metrics))

    for name, (rows, metrics, skipped) in arms.items():
        table(name, metrics)
        print(f"  ({skipped} of {len(rows)} rows skipped: single-child or zero-visit roots)")

    # Legal-action-count distribution: plan 032 #4 sizes Dirichlet alpha as
    # ~10 / mean legal actions, so this is the number that item needs.
    counts = sorted(m["n"] for _, (_, ms, _) in arms.items() for m in ms)
    if counts:
        q = lambda p: counts[min(len(counts) - 1, int(p * len(counts)))]  # noqa: E731
        print(
            f"\nn_legal over all roots: mean {mean(counts):.2f}  median {q(0.5)}  "
            f"p10 {q(0.10)}  p90 {q(0.90)}  max {counts[-1]}"
        )
        print(f"  => Dirichlet alpha ~ 10 / mean_legal = {10 / mean(counts):.3f}")

    # Paired comparison: two probe arms share a seed base, so compare cell by
    # cell rather than pooling — the states are the same, only the prior
    # source differs. Corpus arms cannot be paired this way (every sample in
    # a game carries the same `meta.seed`, and the generations used different
    # seed bases), so they are reported unpaired above and excluded here.
    probe_names = [p.split("/")[-1].removesuffix(".jsonl") for p in probe_paths]
    if len(probe_names) == 2:
        na, nb = probe_names
        ra, rb = arms[na][0], arms[nb][0]
        key = lambda r: (r["state_seed"], r["repeat"], r["budget"])  # noqa: E731
        ma = {key(r): root_metrics(r) for r in ra}
        mb = {key(r): root_metrics(r) for r in rb}
        shared = [k for k in ma if k in mb and ma[k] and mb[k]]
        print(f"\n### paired {na} vs {nb}  ({len(shared)} shared roots)")
        for field in ("norm_entropy", "singleton_share", "top_prior_lift", "q_eq_prior_lift"):
            da = mean([ma[k][field] for k in shared])
            db = mean([mb[k][field] for k in shared])
            diffs = [ma[k][field] - mb[k][field] for k in shared]
            se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
            print(f"  {field:<18} {na} {da:.4f}   {nb} {db:.4f}   diff {mean(diffs):+.4f} +- {se:.4f}")
        # Same states, so a differing child count means the two arms did not
        # actually search the same roots and the pairing is a lie.
        mismatched = sum(1 for k in shared if ma[k]["n"] != mb[k]["n"])
        print(f"  roots whose child count differs between arms: {mismatched} (should be 0)")


if __name__ == "__main__":
    main(sys.argv[1:])
