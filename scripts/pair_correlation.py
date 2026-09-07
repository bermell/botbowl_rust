#!/usr/bin/env python3
"""How much does Home/Away pairing actually buy? (plan 031, D10)

`scripts/paired_summary.py` prints the paired and unpaired SE for one arm.
This one quantifies the *ratio* — and, more importantly, pools it across arms,
because a single arm's variance ratio at n=60 pairs is far too noisy to size
anything with.

    scripts/pair_correlation.py 'runs/exp-data/*.games.jsonl' ...

Definitions, for one rung of n = 2m games drawn as m Home/Away seed pairs:

    unpaired  mean over games,   SE_u = sqrt(V_g / n),  V_g = game-level variance
    paired    mean over pairs,   SE_p = sqrt(V_p / m),  V_p = pair-level variance
    ratio     r = SE_u / SE_p = sqrt(V_g / (2 V_p))

r > 1 means pairing helps. Note what r = 1 means: pairing buys *nothing*, which
is exactly what you get when the two games of a pair are independent
(V_p = V_g/2 then, algebraically). So r is a direct read of the within-pair
correlation rho = corr(score | candidate Home, score | candidate Away):

    V_p = V_g (1 + rho) / 2   =>   rho = 1/r^2 - 1

Negative rho (r > 1) is the side-bias cancellation the pairing was designed for.

Pooling across arms is done as a **ratio of pooled variances** — sum of squares
over sum of degrees of freedom for each estimator separately, then one ratio —
not as a mean of per-arm ratios. Two reasons: a variance ratio is a skewed
statistic whose per-arm estimate at df=59 carries ~+/-9% noise on r, so
averaging ratios both adds bias and hides how weak each one is; and pooling SS
weights each arm by its information rather than by its arm-ness. Arms differ in
mean, but each arm's SS is taken about its own mean, so that is not a problem.

The CI on the pooled ratio is a nonparametric bootstrap over *pairs* (the
independent unit), which correctly handles the fact that V_g and V_p are
computed from the same games and are therefore not independent.
"""
import collections
import glob
import json
import math
import os
import random
import sys


def score(row):
    h, a = row["home_score"], row["away_score"]
    cand, opp = (h, a) if row.get("candidate_team") == "Home" else (a, h)
    return 1.0 if cand > opp else (0.0 if cand < opp else 0.5)


def load_pairs(path):
    """-> {rung: [(home_game_score, away_game_score), ...]} for complete pairs."""
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    out = {}
    by_rung = collections.defaultdict(list)
    for r in rows:
        by_rung[r.get("rung", "?")].append(r)
    for rung, rs in by_rung.items():
        by_seed = collections.defaultdict(dict)
        for r in rs:
            by_seed[r["seed"]][r.get("candidate_team")] = score(r)
        out[rung] = ([(v["Home"], v["Away"]) for v in by_seed.values()
                      if "Home" in v and "Away" in v],
                     len(rs))
    return out


def stats(pairs):
    """(V_g, df_g, SS_g, V_p, df_p, SS_p, mean, rho)."""
    games = [x for p in pairs for x in p]
    ps = [(a + b) / 2 for a, b in pairs]
    n, m = len(games), len(ps)
    gm, pm = sum(games) / n, sum(ps) / m
    ss_g = sum((x - gm) ** 2 for x in games)
    ss_p = sum((x - pm) ** 2 for x in ps)
    v_g, v_p = ss_g / (n - 1), ss_p / (m - 1)
    r = math.sqrt(v_g / (2 * v_p)) if v_p > 0 else float("nan")
    rho = 1 / r ** 2 - 1 if v_p > 0 else float("nan")
    return dict(n=n, m=m, mean=pm, ss_g=ss_g, df_g=n - 1, ss_p=ss_p, df_p=m - 1,
                v_g=v_g, v_p=v_p, se_u=math.sqrt(v_g / n), se_p=math.sqrt(v_p / m),
                r=r, rho=rho)


def pooled_ratio(arms):
    """arms: list of pair-lists. Ratio of pooled variances."""
    ss_g = ss_p = df_g = df_p = 0.0
    for pairs in arms:
        s = stats(pairs)
        ss_g += s["ss_g"]; df_g += s["df_g"]
        ss_p += s["ss_p"]; df_p += s["df_p"]
    v_g, v_p = ss_g / df_g, ss_p / df_p
    return math.sqrt(v_g / (2 * v_p)), v_g, v_p, df_g, df_p


def bootstrap(arms, n_boot=4000, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        resampled = []
        ok = True
        for pairs in arms:
            rs = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            if len({(a + b) / 2 for a, b in rs}) < 2:
                ok = False
            resampled.append(rs)
        if not ok:
            continue
        out.append(pooled_ratio(resampled)[0])
    out.sort()
    lo = out[int(0.025 * len(out))]
    hi = out[int(0.975 * len(out))]
    return lo, hi


def main():
    pats = sys.argv[1:] or ["runs/exp-data/*.games.jsonl", "runs/exp-search/*.games.jsonl"]
    files = [f for p in pats for f in sorted(glob.glob(p))]
    named = []
    for f in files:
        for rung, (pairs, n_rows) in load_pairs(f).items():
            if len(pairs) >= 10:
                named.append((os.path.basename(f).replace(".games.jsonl", ""), pairs, n_rows))

    print(f"{'arm':<24} {'games':>6} {'pairs':>6} {'mean':>7} {'SE_unp':>8} {'SE_pair':>8} "
          f"{'ratio':>7} {'rho':>7}")
    for name, pairs, n_rows in named:
        s = stats(pairs)
        print(f"{name:<24} {s['n']:>6} {s['m']:>6} {s['mean']:>7.3f} {s['se_u']:>8.4f} "
              f"{s['se_p']:>8.4f} {s['r']:>7.3f} {s['rho']:>+7.3f}")

    def pool(sel, label):
        arms = [p for n, p, _ in named if sel(n)]
        if not arms:
            return
        r, v_g, v_p, df_g, df_p = pooled_ratio(arms)
        lo, hi = bootstrap(arms)
        rho = 1 / r ** 2 - 1
        print(f"\n{label}: {len(arms)} arms, df_g={int(df_g)} df_p={int(df_p)}")
        print(f"  pooled V_game={v_g:.4f}  V_pair={v_p:.4f}")
        print(f"  pooled ratio r = {r:.3f}  (bootstrap 95% CI [{lo:.3f}, {hi:.3f}])"
              f"   implied rho = {rho:+.3f}")
        print(f"  SE at 120 games: unpaired {math.sqrt(v_g/120):.4f} -> "
              f"paired {math.sqrt(v_p/60):.4f}")
        # games-weighted mean of the per-arm ratios, for contrast
        num = den = 0.0
        for n, p, _ in named:
            if sel(n):
                s = stats(p)
                num += s["r"] * s["n"]; den += s["n"]
        print(f"  (games-weighted mean of per-arm ratios: {num/den:.3f})")

    pool(lambda n: True, "POOLED over all arms")
    pool(lambda n: n.startswith("s"), "POOLED over exp-data strength arms only")
    pool(lambda n: n.startswith("mirror"), "POOLED over exp-search mirror (null) arms only")


if __name__ == "__main__":
    main()
