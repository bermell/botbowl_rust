#!/usr/bin/env python3
"""Offline corpus diagnostics for plan 031 items D1, D2 and D6.

Streams the generation shards one JSONL line (= one trajectory / drive) at a
time and prints markdown tables. Peak RSS is O(1) in corpus size: a line is
parsed, folded into integer/float accumulators, and dropped. Never load a
whole shard.

Run with the training venv (numpy only, no scipy):

    train/.venv/bin/python scripts/audit_corpus_stats.py \
        --runs runs/loop14x7 --gens 5 6 7 --workers 4 [--json out.json]

Conventions verified against the Rust source before anything is computed here
(see the header of each section below and the results write-up):

* `ChildStat.q` and `Sample.root_value` are **Home-centric** on the leaf-score
  scale (`botbowl-mcts/src/score.rs`, `dynamics.rs::pick_best_action` applies
  `q_sign = +1` for Home / `-1` for Away). Under `Evaluator::Nn` / `NnValue`
  they are the drive-relative value x1000, so `root_value/1000` is directly
  comparable to `outcome_value`. Under `Evaluator::Heuristic` they are the
  *absolute* `leaf_score` (score delta x1000 + ball control x10 + carrier
  distance), which is NOT on the outcome scale -- heuristic shards are
  reported separately and excluded from the D1 headline.
* `Sample.outcome_value` is **Home-centric** in [-1, 1]: the drive-relative
  score delta (`botbowl-data/src/lib.rs::backfill_outcome_value`).
* The mover's frame is `+1` for Home, `-1` for Away
  (`botbowl-nn/src/targets.rs::value_target`).
* The pi target is `botbowl-nn/src/targets.rs::policy_target` under
  `SolvedRootPolicy::OneHot` (prepare's default; `train_loop.sh` passes no
  `--solved-root`), with `--min-root-visits 0`. Reimplemented verbatim below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# ---------------------------------------------------------------------------
# conventions / small helpers
# ---------------------------------------------------------------------------

N_BINS = 10  # [-1,-0.8) ... [0.8,1.0]
FAN_BUCKETS = ("<=10", "11-30", "31-60", ">60")


def fan_bucket(n: int) -> str:
    if n <= 10:
        return "<=10"
    if n <= 30:
        return "11-30"
    if n <= 60:
        return "31-60"
    return ">60"


def value_bin(v: float) -> int:
    """Bin a value in [-1,1] into 10 equal bins; [0.8,1.0] is closed."""
    i = int((v + 1.0) / 0.2)
    return 0 if i < 0 else (N_BINS - 1 if i >= N_BINS else i)


BIN_LABELS = [
    f"[{-1.0 + 0.2 * i:+.1f},{-1.0 + 0.2 * (i + 1):+.1f}" + (")" if i < N_BINS - 1 else "]")
    for i in range(N_BINS)
]


def mover_sign(to_move: str) -> int:
    return 1 if to_move == "Home" else -1


def shard_kind(meta: dict) -> str:
    """'heuristic' if either bot ran the scripted evaluator, else 'nn'."""
    bots = (meta.get("home_bot", ""), meta.get("away_bot", ""))
    if any("eval=heuristic" in b for b in bots):
        return "heuristic"
    if any("eval=nn:" in b for b in bots):
        return "nn"
    if any("eval=nn-value:" in b for b in bots):
        return "nn-value"
    return "other"


def avg_ranks(xs):
    """Average ranks (1-based), ties averaged. Pure stdlib."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rho with tie-corrected ranks. None if either side is constant."""
    ra, rb = avg_ranks(a), avg_ranks(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    da = [x - ma for x in ra]
    db = [x - mb for x in rb]
    va = sum(x * x for x in da)
    vb = sum(x * x for x in db)
    if va <= 0.0 or vb <= 0.0:
        return None
    return sum(x * y for x, y in zip(da, db)) / math.sqrt(va * vb)


def policy_target(sample: dict):
    """Verbatim port of botbowl-nn/src/targets.rs::policy_target(OneHot)."""
    children = sample["children"]
    n = len(children)
    if n == 0:
        return None
    sign = mover_sign(sample["to_move"])

    def argmax_q(filter_solved):
        best_i, best_q = None, None
        for i, c in enumerate(children):
            if filter_solved is not None and c["solved"] != filter_solved:
                continue
            q = c["q"]
            if q is None:
                continue
            q *= sign
            # `max_by_key` keeps the LAST maximum in Rust's Iterator::max_by_key.
            if best_q is None or q >= best_q:
                best_i, best_q = i, q
        return best_i

    if sample["root_solved"]:
        best = argmax_q(None)
        if best is None:
            best = last_argmax(n, lambda i: children[i]["visits"])
        probs = [0.0] * n
        probs[best] = 1.0
        return probs

    counts = [float(c["visits"]) for c in children]
    if any(c["solved"] for c in children):
        unsolved = [c["visits"] for c in children if not c["solved"]]
        max_unsolved = float(max(unsolved)) if unsolved else 0.0
        bs = argmax_q(True)
        if bs is not None:
            counts[bs] = max(counts[bs], max_unsolved)
    total = sum(counts)
    if total <= 0.0:
        return None
    return [c / total for c in counts]


def last_argmax(n, key):
    """Rust's `Iterator::max_by*` semantics: the LAST maximum wins."""
    best_i, best_k = None, None
    for i in range(n):
        k = key(i)
        if best_k is None or k >= best_k:
            best_i, best_k = i, k
    return best_i


def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0.0)


# ---------------------------------------------------------------------------
# per-shard accumulation
# ---------------------------------------------------------------------------


def new_acc():
    return {
        # D1: (kind, strat, sval, bin) ->
        #     [n, sum_outcome, sum_brier, sum_gap, sum_v, sum_v2, sum_vo]
        "d1": defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # D1 range check: (kind) -> [n, n_abs_gt_1, n_saturated]
        "d1range": defaultdict(lambda: [0, 0, 0]),
        # (kind) -> {bot label: n drives}
        "bots": defaultdict(lambda: defaultdict(int)),
        # D2: (kind, fan) -> counters
        "d2": defaultdict(
            lambda: {
                "roots": 0,
                "single_child": 0,
                "lt2_scored": 0,
                "tie": 0,
                "tie_denom": 0,
                "agree": 0,
                "agree_denom": 0,
                "sp_sum": 0.0,
                "sp_n": 0,
                "sp_undef": 0,
                "ent_sum": 0.0,
                "entnorm_sum": 0.0,
                "ent_n": 0,
                "root_solved": 0,
                "no_prior": 0,
            }
        ),
        # D6
        "d6": defaultdict(
            lambda: {
                "lines": 0,
                "samples": 0,
                "empty_lines": 0,
                "ov_home": defaultdict(int),
                "ov_mover": defaultdict(int),
                "ov_missing": 0,
                "root_value_missing": 0,
            }
        ),
        # D6 drive ends: (kind, start_half, start_turn) -> {class: n}
        "ends": defaultdict(lambda: defaultdict(int)),
        # diagnostic for the clock classification: (class) -> {(ht,at): n}
        "endturns": defaultdict(lambda: defaultdict(int)),
        # (kind, class) -> [drives, samples, n_ov_zero, n_ov_nonzero]
        "endsamples": defaultdict(lambda: [0, 0, 0, 0]),
    }


def merge(dst, src):
    for k, v in src["d1"].items():
        d = dst["d1"][k]
        for i in range(len(v)):
            d[i] += v[i]
    for k, v in src["d1range"].items():
        d = dst["d1range"][k]
        for i in range(len(v)):
            d[i] += v[i]
    for k, v in src["bots"].items():
        for b, n in v.items():
            dst["bots"][k][b] += n
    for k, v in src["d2"].items():
        d = dst["d2"][k]
        for kk, vv in v.items():
            d[kk] += vv
    for k, v in src["d6"].items():
        d = dst["d6"][k]
        for kk, vv in v.items():
            if isinstance(vv, dict):
                for c, n in vv.items():
                    d[kk][c] += n
            else:
                d[kk] += vv
    for k, v in src["ends"].items():
        for c, n in v.items():
            dst["ends"][k][c] += n
    for k, v in src["endturns"].items():
        for c, n in v.items():
            dst["endturns"][k][c] += n
    for k, v in src["endsamples"].items():
        d = dst["endsamples"][k]
        for i in range(len(v)):
            d[i] += v[i]
    return dst


def process_shard(path: str):
    acc = new_acc()
    d1, d1r, d2, d6, ends, endturns, bots = (
        acc["d1"],
        acc["d1range"],
        acc["d2"],
        acc["d6"],
        acc["ends"],
        acc["endturns"],
        acc["bots"],
    )
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            traj = json.loads(line)
            meta = traj["meta"]
            kind = shard_kind(meta)
            extra = meta.get("extra", {})
            samples = traj["samples"]
            outcome = traj["outcome"]

            bots[kind][meta.get("home_bot", "")] += 1
            bots[kind][meta.get("away_bot", "")] += 1
            g6 = d6[kind]
            g6["lines"] += 1
            g6["samples"] += len(samples)
            if not samples:
                g6["empty_lines"] += 1

            # ---- D6 drive-end classification -----------------------------
            start_score = extra.get("start_score")
            start_half = extra.get("start_half")
            start_turn = None
            if start_half is not None:
                try:
                    start_turn = max(int(extra["start_home_turn"]), int(extra["start_away_turn"]))
                except (KeyError, ValueError):
                    start_turn = None
            if start_score is not None:
                sh, sa = (int(x) for x in start_score.split("-"))
                scored = (outcome["home_score"], outcome["away_score"]) != (sh, sa)
                if scored:
                    cls = "score"
                elif outcome["game_over"]:
                    cls = "clock_gameover"
                else:
                    cls = "clock_half"
                ends[(kind, start_half, start_turn)][cls] += 1
                es = acc["endsamples"][(kind, cls)]
                es[0] += 1
                es[1] += len(samples)
                for s in samples:
                    if s.get("outcome_value") == 0.0:
                        es[2] += 1
                    else:
                        es[3] += 1
                if samples:
                    info = samples[-1]["state"]["info"]
                    endturns[cls][(info["home_turn"], info["away_turn"])] += 1
                else:
                    endturns[cls][("nosamples", "nosamples")] += 1

            # ---- per-sample ----------------------------------------------
            for s in samples:
                sign = mover_sign(s["to_move"])
                ov = s.get("outcome_value")
                rv = s.get("root_value")
                children = s["children"]
                nc = len(children)
                fb = fan_bucket(nc)
                info = s["state"]["info"]
                turn = max(info["home_turn"], info["away_turn"])

                if ov is None:
                    g6["ov_missing"] += 1
                else:
                    g6["ov_home"][round(ov, 3)] += 1
                    g6["ov_mover"][round(sign * ov, 3)] += 1
                if rv is None:
                    g6["root_value_missing"] += 1

                # ---- D1 --------------------------------------------------
                if rv is not None and ov is not None:
                    raw = sign * rv / 1000.0
                    r = d1r[kind]
                    r[0] += 1
                    if abs(raw) > 1.0:
                        r[1] += 1
                    if abs(raw) >= 1.0:
                        r[2] += 1
                    v = max(-1.0, min(1.0, raw))
                    o = sign * ov
                    b = value_bin(v)
                    brier = ((v - o) / 2.0) ** 2
                    gap = v - o
                    for strat, sval in (("all", "-"), ("fan", fb), ("turn", turn)):
                        cell = d1[(kind, strat, sval, b)]
                        cell[0] += 1
                        cell[1] += o
                        cell[2] += brier
                        cell[3] += gap
                        cell[4] += v
                        cell[5] += v * v
                        cell[6] += v * o

                # ---- D2 --------------------------------------------------
                c2 = d2[(kind, fb)]
                c2["roots"] += 1
                if s["root_solved"]:
                    c2["root_solved"] += 1
                if nc <= 1:
                    c2["single_child"] += 1
                    continue

                qs = [(i, sign * c["q"]) for i, c in enumerate(children) if c["q"] is not None]
                if len(qs) < 2:
                    c2["lt2_scored"] += 1
                else:
                    qsorted = sorted((q for _, q in qs), reverse=True)
                    c2["tie_denom"] += 1
                    if qsorted[0] == qsorted[1]:
                        c2["tie"] += 1

                # argmax mover-Q with the played-move key: (scored, q, visits)
                best_played = last_argmax(
                    nc,
                    lambda i: (
                        (1, sign * children[i]["q"], children[i]["visits"])
                        if children[i]["q"] is not None
                        else (0, 0, 0)
                    ),
                )
                best_visits = last_argmax(nc, lambda i: children[i]["visits"])
                c2["agree_denom"] += 1
                if best_played == best_visits:
                    c2["agree"] += 1

                priors = [c["prior"] for c in children]
                if any(p is None for p in priors):
                    c2["no_prior"] += 1
                else:
                    rho = spearman(priors, [c["visits"] for c in children])
                    if rho is None:
                        c2["sp_undef"] += 1
                    else:
                        c2["sp_sum"] += rho
                        c2["sp_n"] += 1

                probs = policy_target(s)
                if probs is not None:
                    h = entropy(probs)
                    c2["ent_sum"] += h
                    c2["entnorm_sum"] += h / math.log(nc)
                    c2["ent_n"] += 1
    # defaultdicts don't pickle with lambdas -> plain nested structures
    return to_plain(acc)


def to_plain(acc):
    return {
        "d1": {repr(k): v for k, v in acc["d1"].items()},
        "d1range": {repr(k): v for k, v in acc["d1range"].items()},
        "d2": {repr(k): dict(v) for k, v in acc["d2"].items()},
        "d6": {
            repr(k): {kk: (dict(vv) if isinstance(vv, defaultdict) else vv) for kk, vv in v.items()}
            for k, v in acc["d6"].items()
        },
        "bots": {k: dict(v) for k, v in acc["bots"].items()},
        "ends": {repr(k): dict(v) for k, v in acc["ends"].items()},
        "endturns": {k: {repr(kk): vv for kk, vv in v.items()} for k, v in acc["endturns"].items()},
        "endsamples": {repr(k): v for k, v in acc["endsamples"].items()},
    }


def from_plain(plain):
    acc = new_acc()
    import ast

    for k, v in plain["d1"].items():
        acc["d1"][ast.literal_eval(k)] = list(v)
    for k, v in plain["d1range"].items():
        acc["d1range"][ast.literal_eval(k)] = list(v)
    for k, v in plain["d2"].items():
        cell = acc["d2"][ast.literal_eval(k)]
        for kk, vv in v.items():
            cell[kk] = vv
    for k, v in plain["d6"].items():
        cell = acc["d6"][ast.literal_eval(k)]
        for kk, vv in v.items():
            if isinstance(vv, dict):
                for c, n in vv.items():
                    cell[kk][float(c)] = n
            else:
                cell[kk] = vv
    for k, v in plain.get("bots", {}).items():
        for b, n in v.items():
            acc["bots"][k][b] += n
    for k, v in plain["ends"].items():
        for c, n in v.items():
            acc["ends"][ast.literal_eval(k)][c] += n
    for k, v in plain["endturns"].items():
        for c, n in v.items():
            acc["endturns"][k][ast.literal_eval(c)] += n
    for k, v in plain.get("endsamples", {}).items():
        d = acc["endsamples"][ast.literal_eval(k)]
        for i in range(len(v)):
            d[i] += v[i]
    return acc


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def report(per_gen, out):
    w = out.write
    gens = sorted(per_gen)
    kinds_order = ["nn", "heuristic", "nn-value", "other"]

    # ------------------------------ D1 ---------------------------------
    w("## D1 - root-value calibration\n\n")
    w("Scale check (`|mover-sign root_value/1000| > 1`, i.e. off the outcome scale) "
      "and saturation (`|root_value| == 1000`):\n\n")
    w("| gen | shard kind | roots | frac \\|v\\|>1 | frac \\|v\\|>=1 (saturated) |\n|---|---|---:|---:|---:|\n")
    for g in gens:
        for kind in kinds_order:
            r = per_gen[g]["d1range"].get(kind)
            if not r:
                continue
            w(f"| gen{g:02d} | {kind} | {r[0]:,} | {r[1] / r[0]:.4f} | {r[2] / r[0]:.4f} |\n")
    w("\n")

    w("Per-sample least-squares fit `outcome ~ a + b * v` (mover frame, v clipped to [-1,1]). "
      "Calibrated = (a, b) close to (0, 1); b < 1 = flatter than the diagonal; a < 0 = optimism.\n\n")
    w("| shard kind | stratum | " + " | ".join(f"gen{g:02d} b / a" for g in gens) + " |\n")
    w("|---|---|" + "---|" * len(gens) + "\n")
    for kind in kinds_order:
        if not any(k[0] == kind for g in gens for k in per_gen[g]["d1"]):
            continue
        for strat, sval, label in [("all", "-", "overall")] + [("fan", f, f"fan {f}") for f in FAN_BUCKETS]:
            cells = []
            for g in gens:
                n = s_v = s_v2 = s_o = s_vo = 0.0
                for b in range(N_BINS):
                    c = per_gen[g]["d1"].get((kind, strat, sval, b))
                    if not c:
                        continue
                    n += c[0]
                    s_o += c[1]
                    s_v += c[4]
                    s_v2 += c[5]
                    s_vo += c[6]
                if n < 2:
                    cells.append("-")
                    continue
                var = s_v2 - s_v * s_v / n
                cov = s_vo - s_v * s_o / n
                if var <= 0:
                    cells.append("-")
                    continue
                slope = cov / var
                inter = (s_o - slope * s_v) / n
                cells.append(f"{slope:.3f} / {inter:+.3f}")
            w(f"| {kind} | {label} | " + " | ".join(cells) + " |\n")
    w("\n")

    for kind in kinds_order:
        if not any(k[0] == kind for g in gens for k in per_gen[g]["d1"]):
            continue
        w(f"### D1 calibration, `{kind}` shards - overall\n\n")
        w("| bin (mover root_value/1000) | " + " | ".join(
            f"gen{g:02d} n / mean(outcome) / Brier" for g in gens) + " |\n")
        w("|---|" + "---|" * len(gens) + "\n")
        for b in range(N_BINS):
            cells = []
            for g in gens:
                c = per_gen[g]["d1"].get((kind, "all", "-", b))
                if not c or c[0] == 0:
                    cells.append("-")
                else:
                    cells.append(f"{c[0]:,} / {c[1] / c[0]:+.3f} / {c[2] / c[0]:.3f}")
            w(f"| {BIN_LABELS[b]} | " + " | ".join(cells) + " |\n")
        w("\n")

        w(f"### D1 mean signed gap E[v - outcome], `{kind}` shards\n\n")
        w("| stratum | " + " | ".join(f"gen{g:02d} gap (n)" for g in gens) + " |\n")
        w("|---|" + "---|" * len(gens) + "\n")
        rows = [("all", "-")] + [("fan", f) for f in FAN_BUCKETS]
        turns = sorted({k[2] for g in gens for k in per_gen[g]["d1"] if k[0] == kind and k[1] == "turn"})
        rows += [("turn", t) for t in turns]
        for strat, sval in rows:
            cells = []
            for g in gens:
                n = sum(per_gen[g]["d1"].get((kind, strat, sval, b), [0, 0, 0, 0])[0] for b in range(N_BINS))
                s = sum(per_gen[g]["d1"].get((kind, strat, sval, b), [0, 0, 0, 0])[3] for b in range(N_BINS))
                cells.append("-" if n == 0 else f"{s / n:+.3f} ({n:,})")
            label = "overall" if strat == "all" else f"{strat} {sval}"
            w(f"| {label} | " + " | ".join(cells) + " |\n")
        w("\n")

        for strat, vals in (("fan", FAN_BUCKETS),):
            w(f"### D1 calibration by {strat} width, `{kind}` shards (mean outcome per bin)\n\n")
            for g in gens:
                w(f"\ngen{g:02d}:\n\n")
                w("| bin | " + " | ".join(f"{v} (n)" for v in vals) + " |\n")
                w("|---|" + "---|" * len(vals) + "\n")
                for b in range(N_BINS):
                    cells = []
                    for v in vals:
                        c = per_gen[g]["d1"].get((kind, strat, v, b))
                        cells.append("-" if not c or c[0] == 0 else f"{c[1] / c[0]:+.3f} ({c[0]:,})")
                    w(f"| {BIN_LABELS[b]} | " + " | ".join(cells) + " |\n")
            w("\n")

    # ------------------------------ D2 ---------------------------------
    w("## D2 - tie rate and self-distillation\n\n")
    for kind in kinds_order:
        if not any(k[0] == kind for g in gens for k in per_gen[g]["d2"]):
            continue
        w(f"### `{kind}` shards\n\n")
        w("| gen | fan | roots | 1-child | <2 scored | tie rate | top1 agree | mean rho(prior,visits) | "
          "mean H(pi) nats | mean H/ln(n) | root solved |\n")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for g in gens:
            tot = None
            for fb in FAN_BUCKETS + ("ALL",):
                if fb == "ALL":
                    c = tot
                else:
                    c = per_gen[g]["d2"].get((kind, fb))
                    if c is None:
                        continue
                    if tot is None:
                        tot = dict(c)
                    else:
                        for kk, vv in c.items():
                            tot[kk] += vv
                if c is None:
                    continue
                w("| gen{:02d} | {} | {:,} | {:,} | {:,} | {} | {} | {} | {} | {} | {} |\n".format(
                    g, fb, c["roots"], c["single_child"], c["lt2_scored"],
                    fmt(c["tie"] / c["tie_denom"]) if c["tie_denom"] else "-",
                    fmt(c["agree"] / c["agree_denom"]) if c["agree_denom"] else "-",
                    fmt(c["sp_sum"] / c["sp_n"]) if c["sp_n"] else "-",
                    fmt(c["ent_sum"] / c["ent_n"]) if c["ent_n"] else "-",
                    fmt(c["entnorm_sum"] / c["ent_n"]) if c["ent_n"] else "-",
                    fmt(c["root_solved"] / c["roots"]) if c["roots"] else "-",
                ))
        w("\n")

    # ------------------------------ D6 ---------------------------------
    w("## D6 - corpus composition\n\n")
    w("Bot labels found in `meta.home_bot`/`meta.away_bot` (the evaluator each generation really ran):\n\n")
    w("| gen | bot label | bot-slots (2 per drive) |\n|---|---|---:|\n")
    for g in gens:
        for label, n in sorted(
            ((b, n) for v in per_gen[g]["bots"].values() for b, n in v.items()), key=lambda kv: -kv[1]
        ):
            short = label.replace("/home/mattias/repos/botbowl_rust/models/", "")
            w(f"| gen{g:02d} | `{short}` | {n:,} |\n")
    w("\n")
    w("| gen | shard kind | drives | samples | samples/drive | empty drives |\n|---|---|---:|---:|---:|---:|\n")
    for g in gens:
        tot_s = sum(v["samples"] for v in per_gen[g]["d6"].values())
        for kind in kinds_order:
            v = per_gen[g]["d6"].get(kind)
            if not v:
                continue
            w("| gen{:02d} | {} | {:,} | {:,} ({:.1%}) | {:.1f} | {:,} |\n".format(
                g, kind, v["lines"], v["samples"], v["samples"] / tot_s if tot_s else 0,
                v["samples"] / v["lines"] if v["lines"] else 0, v["empty_lines"]))
    w("\n")

    w("Value-target (`outcome_value`) class split, per sample:\n\n")
    w("| gen | shard kind | frame | -1 | 0 | +1 | other |\n|---|---|---|---:|---:|---:|---:|\n")
    for g in gens:
        for kind in kinds_order:
            v = per_gen[g]["d6"].get(kind)
            if not v:
                continue
            for frame, key in (("home", "ov_home"), ("mover", "ov_mover")):
                d = v[key]
                n = sum(d.values())
                if not n:
                    continue
                other = n - d.get(-1.0, 0) - d.get(0.0, 0) - d.get(1.0, 0)
                w("| gen{:02d} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:,} |\n".format(
                    g, kind, frame, d.get(-1.0, 0) / n, d.get(0.0, 0) / n, d.get(1.0, 0) / n, other))
    w("\n")

    w("Drive-end classification (one JSONL line = one drive):\n\n")
    w("| gen | shard kind | drives | score | clock (half ends) | clock (game over) |\n"
      "|---|---|---:|---:|---:|---:|\n")
    for g in gens:
        for kind in kinds_order:
            rows = [(k, v) for k, v in per_gen[g]["ends"].items() if k[0] == kind]
            if not rows:
                continue
            n = sum(sum(v.values()) for _, v in rows)
            sc = sum(v.get("score", 0) for _, v in rows)
            ch = sum(v.get("clock_half", 0) for _, v in rows)
            cg = sum(v.get("clock_gameover", 0) for _, v in rows)
            w("| gen{:02d} | {} | {:,} | {:.3f} | {:.3f} | {:.3f} |\n".format(
                g, kind, n, sc / n, ch / n, cg / n))
    w("\n")

    w("Drive length and value-target by drive class (all shard kinds and generations pooled) - "
      "the check that `outcome_value == 0` means 'clock', not 'no score':\n\n")
    w("| class | drives | samples | samples/drive | frac samples with outcome_value == 0 |\n"
      "|---|---:|---:|---:|---:|\n")
    for cls in ("score", "clock_half", "clock_gameover"):
        d = [0, 0, 0, 0]
        for g in gens:
            for k, v in per_gen[g]["endsamples"].items():
                if k[1] == cls:
                    for i in range(4):
                        d[i] += v[i]
        if not d[0]:
            continue
        w("| {} | {:,} | {:,} | {:.1f} | {:.4f} |\n".format(
            cls, d[0], d[1], d[1] / d[0], d[2] / max(1, d[2] + d[3])))
    w("\n")

    w("Clock-ended fraction by start turn (all shard kinds and generations pooled):\n\n")
    w("| start half | start turn | drives | score | clock total |\n|---|---|---:|---:|---:|\n")
    pooled = defaultdict(lambda: defaultdict(int))
    for g in gens:
        for k, v in per_gen[g]["ends"].items():
            for c, nn in v.items():
                pooled[(k[1], k[2])][c] += nn
    for key in sorted(pooled, key=lambda t: (str(t[0]), -1 if t[1] is None else t[1])):
        v = pooled[key]
        n = sum(v.values())
        clock = v.get("clock_half", 0) + v.get("clock_gameover", 0)
        w("| {} | {} | {:,} | {:.3f} | {:.3f} |\n".format(key[0], key[1], n, v.get("score", 0) / n, clock / n))
    w("\n")

    w("Turn counters `(home_turn, away_turn)` on the LAST sample of a drive, by class "
      "(pooled) - the check that 'clock' really means the turn track ran out:\n\n")
    for cls in ("clock_half", "clock_gameover", "score"):
        agg = defaultdict(int)
        for g in gens:
            for k, n in per_gen[g]["endturns"].get(cls, {}).items():
                agg[k] += n
        if not agg:
            continue
        n = sum(agg.values())
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:6]
        w(f"* `{cls}` (n={n:,}): " + ", ".join(f"{k}={v / n:.3f}" for k, v in top) + "\n")
    w("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/loop14x7")
    ap.add_argument("--gens", nargs="+", type=int, default=[5, 6, 7])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", default=None, help="dump raw aggregates here")
    ap.add_argument("--load", default=None, help="skip the pass, report from this json")
    ap.add_argument("--out", default=None, help="write the markdown here instead of stdout")
    args = ap.parse_args()

    if args.load:
        with open(args.load) as fh:
            raw = json.load(fh)
        per_gen = {int(g): from_plain(p) for g, p in raw.items()}
    else:
        import glob

        jobs = []
        for g in args.gens:
            for p in sorted(glob.glob(os.path.join(args.runs, f"gen{g:02d}", "shard*.jsonl"))):
                jobs.append((g, p))
        if not jobs:
            sys.exit("no shards found")
        per_gen = {g: new_acc() for g in args.gens}
        with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as ex:
            futs = {ex.submit(process_shard, p): (g, p) for g, p in jobs}
            for fut, (g, p) in futs.items():
                merge(per_gen[g], from_plain(fut.result()))
                print(f"done {p}", file=sys.stderr)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({str(g): to_plain(a) for g, a in per_gen.items()}, fh)

    out = open(args.out, "w") if args.out else sys.stdout
    report(per_gen, out)
    if args.out:
        out.close()


if __name__ == "__main__":
    main()
