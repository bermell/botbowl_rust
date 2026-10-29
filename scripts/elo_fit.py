#!/usr/bin/env python3
"""Plan 071 R4: Bradley–Terry (Elo scale) over every paired drive result of a run, with SEs.

    elo_fit.py runs/<run> [--group board|set|all] [--anchor NAME] [--fixed-only]
               [--json F] [--csv F] [--png F] [--summary NET] [--nearest NET --among A,B,.. --k 2]

Reads every `genNN/drives*/report.json` (train_loop.sh's `drives` = DRIVE_REF, `drives_origin` =
ORIGIN_REF, `drives_ladder_<net>` = DRIVE_LADDER) plus any `--report` files. Each ladder row with a
pentanomial `pairs.counts` = [n0, n1, n2, n3, n4] is a set of pairs whose scores are 0, ¼, ½, ¾, 1
for the candidate (plan 051's pair scoring: a pair is the mean of the two games from swapped sides,
so `points` = Σ i·n_i / 4n). Full-game rows (no pairs) are skipped.

**The model.** E[s_ij] = 1 / (1 + 10^(-(R_i - R_j)/400)) for a pair between nets i and j, fitted
by maximum quasi-likelihood: each pair contributes s log p + (1 - s) log(1 - p), the Bernoulli
likelihood with the fractional pair score as outcome (Newton on the exact Hessian). The anchor is
fixed at 0. Pair scores are less variable than a single Bernoulli game (two games averaged, draws),
so the covariance is the inverse Fisher information scaled by the Pearson dispersion
φ = Σ (s - p)² / (p(1 - p)) / (pairs - parameters) — the quasi-binomial SE, which for a single
match reduces to plan 051's pentanomial points_se. `--no-dispersion` uses φ = 1.

`--prior-pairs P` (default 1) adds P drawn pairs to every observed edge so that a clean sweep stays
finite; with 200 pairs per edge it moves a rating by < 1%. Set 0 for the raw MLE.

Nets are named by file stem; the run's own nets (`bbnet_<tier>_genNN`, the candidates' tier) are
shortened to `genNN`, so a foreign gen58 yardstick never merges with the run's own gen58.

SPRT-stopped rows (DRIVE_REF with DRIVE_SPRT) are included by default — the task is every pair the
run played — but their stopping time depends on the result; `--fixed-only` drops them (plan 071
§3.5: the fit wants fixed pairs).

**Groups.** `board` (default): one fit per board (`20x11/7`, `16x9/6`), all contested sets on
that board pooled. `set`: one per `set@board`. `all`: every result in one fit.

**Modes.** Default prints a table per group (net, gen, Elo, SE, pairs, opponents). `--summary NET`
prints one line: NET's Elo per group, and its gain over the net ten generations back
(`Δ10`, SE from the covariance) — §3.6's plateau signal. `--nearest NET --among A,B --k K` prints
the K nets among the list whose pooled (`--group all`) Elo is nearest NET's, one per line; unrated
candidates come last, newest first (train_loop.sh's DRIVE_LADDER anchor choice).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

K = math.log(10) / 400.0  # d logit / d Elo
PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
OWN_RE = re.compile(r"^(?:bbnet|anchor)_(.+)_(gen\d+)$")


# -- reading -----------------------------------------------------------------------------------


@dataclass
class Result:
    """Pairs of one rung: `a` (candidate) vs `b` (opponent), pentanomial counts from a's side."""

    a: str
    b: str
    counts: tuple[int, ...]
    board: str
    set: str
    sprt: bool = False
    source: str = ""

    @property
    def n(self) -> int:
        return sum(self.counts)

    @property
    def s(self) -> float:
        return sum(c * v for c, v in zip(self.counts, PAIR_SCORES))

    @property
    def s2(self) -> float:
        return sum(c * v * v for c, v in zip(self.counts, PAIR_SCORES))


def model_stem(text: str) -> str | None:
    """The net in `mcts(nn:/path/x.onnx)@cfg` or `vs:mcts(nn:/path/x.onnx) [cfg] ...` (file stem)."""
    m = re.search(r"nn:([^)\s]+)", text)
    if not m:
        return None
    return Path(m.group(1)).name.removesuffix(".onnx").removesuffix(".pt")


def bot_name(text: str) -> str:
    """A non-nn opponent: the bot spec up to its first space (`vs:` dropped)."""
    return text.removeprefix("vs:").split(" ")[0]


def read_report(path: Path) -> list[Result]:
    rep = json.loads(path.read_text())
    cand = model_stem(rep.get("candidate", "")) or bot_name(rep.get("candidate", "?"))
    out = []
    for row in rep.get("ladder", []):
        counts = (row.get("pairs") or {}).get("counts")
        if not counts or sum(counts) == 0:
            continue
        opp = row.get("opponent", "")
        m = re.search(r"drives\(([^)]*)\)@(\S+)", opp)
        board = row.get("board") or (m.group(2) if m else "env")
        set_name = m.group(1) if m else "games"
        b = model_stem(opp) or bot_name(opp)
        out.append(
            Result(cand, b, tuple(int(c) for c in counts), board, set_name, bool(row.get("sprt")), str(path))
        )
    return out


def gen_dirs(run: Path) -> list[Path]:
    return sorted(
        (d for d in run.glob("gen[0-9]*") if d.is_dir() and re.fullmatch(r"gen\d+", d.name)),
        key=lambda d: int(d.name[3:]),
    )


def collect(run: Path | None, extra: list[Path]) -> list[Result]:
    reports = []
    if run is not None:
        for g in gen_dirs(run):
            reports += sorted(p for p in g.glob("drives*/report.json"))
    reports += extra
    res = []
    for p in reports:
        try:
            res += read_report(p)
        except (OSError, ValueError, KeyError) as e:
            print(f"elo_fit: skipping {p}: {e}", file=sys.stderr)
    return res


def own_tier(results: list[Result]) -> str | None:
    """The run's own tier: the most common one among the candidates' `bbnet_<tier>_genNN` stems."""
    tiers = Counter(m.group(1) for r in results if (m := OWN_RE.match(r.a)))
    return tiers.most_common(1)[0][0] if tiers else None


def shorten(results: list[Result]) -> list[Result]:
    tier = own_tier(results)

    def name(stem: str) -> str:
        m = OWN_RE.match(stem)
        return m.group(2) if m and m.group(1) == tier else stem

    for r in results:
        r.a, r.b = name(r.a), name(r.b)
    return results


def gen_of(name: str) -> int | None:
    m = re.fullmatch(r"gen(\d+)", name)
    return int(m.group(1)) if m else None


# -- the fit -----------------------------------------------------------------------------------


@dataclass
class Fit:
    anchor: str
    names: list[str]  # index order; names[0] is the anchor
    elo: np.ndarray  # per name, anchor = 0
    cov: np.ndarray  # (n, n), row/col 0 zero
    dispersion: float
    pairs: dict[str, int]
    opponents: dict[str, set[str]]
    dropped: list[str] = field(default_factory=list)  # not connected to the anchor

    def se(self, name: str) -> float:
        i = self.names.index(name)
        return float(math.sqrt(max(self.cov[i, i], 0.0)))

    def rating(self, name: str) -> float:
        return float(self.elo[self.names.index(name)])

    def diff(self, a: str, b: str) -> tuple[float, float]:
        """R_a - R_b and its SE."""
        i, j = self.names.index(a), self.names.index(b)
        var = self.cov[i, i] + self.cov[j, j] - 2 * self.cov[i, j]
        return float(self.elo[i] - self.elo[j]), float(math.sqrt(max(var, 0.0)))


def expected(d_elo):
    return 1.0 / (1.0 + np.power(10.0, -np.asarray(d_elo) / 400.0))


def components(edges: set[tuple[str, str]], nodes: set[str]) -> dict[str, int]:
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    comp, c = {}, 0
    for n in sorted(nodes):
        if n in comp:
            continue
        stack = [n]
        comp[n] = c
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in comp:
                    comp[y] = c
                    stack.append(y)
        c += 1
    return comp


def fit(results: list[Result], anchor: str, prior_pairs: float = 1.0, dispersion: bool = True) -> Fit:
    """Bradley–Terry on the Elo scale with `anchor` = 0 (see the module docstring)."""
    results = [r for r in results if r.a != r.b]
    nodes = {r.a for r in results} | {r.b for r in results}
    if anchor not in nodes:
        raise ValueError(f"anchor {anchor} played no pairs here")
    comp = components({(r.a, r.b) for r in results}, nodes)
    keep = {n for n in nodes if comp[n] == comp[anchor]}
    dropped = sorted(nodes - keep)
    results = [r for r in results if r.a in keep]
    names = [anchor] + sorted(keep - {anchor}, key=lambda n: (gen_of(n) is None, gen_of(n) or 0, n))
    idx = {n: i for i, n in enumerate(names)}
    # Aggregate per ordered edge (i < j): n, Σs, Σs² from i's side.
    agg: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    pairs: Counter = Counter()
    opps: dict[str, set[str]] = defaultdict(set)
    for r in results:
        i, j = idx[r.a], idx[r.b]
        n, s, s2 = r.n, r.s, r.s2
        if i > j:  # flip to the lower index's side: s -> n - s, s² -> Σ(1 - x)² = n - 2s + s²
            i, j, s, s2 = j, i, n - s, n - 2 * s + s2
        e = agg[(i, j)]
        e[0] += n
        e[1] += s
        e[2] += s2
        pairs[r.a] += n
        pairs[r.b] += n
        opps[r.a].add(r.b)
        opps[r.b].add(r.a)
    E = np.array(list(agg.keys()), dtype=int).reshape(-1, 2)
    N = np.array([v[0] for v in agg.values()])
    S = np.array([v[1] for v in agg.values()])
    S2 = np.array([v[2] for v in agg.values()])
    Np, Sp = N + prior_pairs, S + 0.5 * prior_pairs  # the prior: drawn pairs on every edge

    m = len(names)
    r = np.zeros(m)

    def grad_hess(r):
        p = expected(r[E[:, 0]] - r[E[:, 1]])
        g_edge = K * (Sp - Np * p)  # d loglik / d r_i (minus for r_j)
        w = K * K * Np * p * (1 - p)
        g = np.zeros(m)
        np.add.at(g, E[:, 0], g_edge)
        np.add.at(g, E[:, 1], -g_edge)
        H = np.zeros((m, m))  # the negative Hessian (information)
        np.add.at(H, (E[:, 0], E[:, 0]), w)
        np.add.at(H, (E[:, 1], E[:, 1]), w)
        np.add.at(H, (E[:, 0], E[:, 1]), -w)
        np.add.at(H, (E[:, 1], E[:, 0]), -w)
        return g, H, p

    for _ in range(200):
        g, H, _ = grad_hess(r)
        step = np.zeros(m)
        if m > 1:
            step[1:] = np.linalg.solve(H[1:, 1:], g[1:])
        step = np.clip(step, -400, 400)  # damp the first steps of a lopsided fit
        r += step
        if np.max(np.abs(step)) < 1e-9:
            break
    g, H, p = grad_hess(r)
    cov = np.zeros((m, m))
    if m > 1:
        cov[1:, 1:] = np.linalg.inv(H[1:, 1:])
    phi = 1.0
    dof = int(N.sum()) - (m - 1)
    if dispersion and dof > 0:
        # Σ over pairs of (s - p)² / (p(1-p)) from the edge sums: Σs² - 2pΣs + n p².
        pearson = (S2 - 2 * p * S + N * p * p) / np.maximum(p * (1 - p), 1e-12)
        phi = float(pearson.sum() / dof)
        cov *= phi
    return Fit(anchor, names, r, cov, phi, dict(pairs), dict(opps), dropped)


# -- grouping and anchor choice ----------------------------------------------------------------


def group_key(r: Result, how: str) -> str:
    return {"board": r.board, "set": f"{r.set}@{r.board}", "all": "all"}[how]


def auto_anchor(results: list[Result], run: Path | None) -> str:
    """gen00 if it played; else the net gen01 played in its `drives` sub (the init); else the
    lowest-numbered own generation."""
    nodes = {r.a for r in results} | {r.b for r in results}
    if "gen00" in nodes:
        return "gen00"
    for r in results:
        if r.a == "gen01" and Path(r.source).parent.name == "drives" and r.b in nodes:
            return r.b
    gens = sorted((gen_of(n), n) for n in nodes if gen_of(n) is not None)
    if gens:
        return gens[0][1]
    return sorted(nodes)[0]


def fit_groups(results, how, anchor, prior_pairs, dispersion) -> dict[str, Fit]:
    by = defaultdict(list)
    for r in results:
        by[group_key(r, how)].append(r)
    fits = {}
    for key in sorted(by):
        rs = by[key]
        nodes = {r.a for r in rs} | {r.b for r in rs}
        a = anchor if anchor in nodes else None
        if a is None:
            a = auto_anchor(rs, None)
            print(f"elo_fit: {key}: anchor {anchor} played no pairs here, anchored at {a}", file=sys.stderr)
        fits[key] = fit(rs, a, prior_pairs, dispersion)
        if fits[key].dropped:
            print(f"elo_fit: {key}: not connected to {a}: {', '.join(fits[key].dropped)}", file=sys.stderr)
    return fits


# -- output ------------------------------------------------------------------------------------


def table(key: str, f: Fit) -> str:
    lines = [f"== {key}: anchor {f.anchor} = 0, dispersion {f.dispersion:.3f}, "
             f"{sum(f.pairs.values()) // 2} pairs, {len(f.names)} nets =="]
    lines.append(f"{'net':<28} {'Elo':>8} {'SE':>6} {'pairs':>6}  opponents")
    for n in f.names:
        opps = ",".join(sorted(f.opponents.get(n, ()), key=lambda x: (gen_of(x) is None, gen_of(x) or 0, x)))
        lines.append(f"{n:<28} {f.rating(n):>+8.1f} {f.se(n):>6.1f} {f.pairs.get(n, 0):>6}  {opps}")
    return "\n".join(lines)


def rows(fits: dict[str, Fit]):
    for key, f in fits.items():
        for n in f.names:
            yield {"group": key, "net": n, "gen": gen_of(n), "elo": round(f.rating(n), 2),
                   "se": round(f.se(n), 2), "pairs": f.pairs.get(n, 0)}


def summary(fits: dict[str, Fit], net: str, back: int = 10) -> str:
    parts = []
    g = gen_of(net)
    for key, f in fits.items():
        if net not in f.names:
            continue
        s = f"{key}: {net} {f.rating(net):+.0f}±{f.se(net):.0f}"
        if g is not None:
            # the oldest own net at most `back` generations earlier (gen g-back when it played)
            olds = sorted((gen_of(n), n) for n in f.names if gen_of(n) is not None and g - back <= gen_of(n) < g)
            ref = olds[0][1] if olds else None
            if ref is not None:
                d, se = f.diff(net, ref)
                s += f" (Δ{g - gen_of(ref)} {d:+.0f}±{se:.0f})"
        parts.append(s)
    return " | ".join(parts) if parts else f"{net}: no pairs yet"


def nearest(fit_all: Fit | None, net: str, among: list[str], k: int) -> list[str]:
    rated, unrated = [], []
    for c in among:
        if fit_all is not None and net in fit_all.names and c in fit_all.names:
            rated.append((abs(fit_all.rating(c) - fit_all.rating(net)), -(gen_of(c) or 0), c))
        else:
            unrated.append((-(gen_of(c) or 0), c))
    picked = [c for *_, c in sorted(rated)] + [c for _, c in sorted(unrated)]
    return picked[:k]


def plot(fits: dict[str, Fit], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("elo_fit: matplotlib not installed, no plot", file=sys.stderr)
        return False
    keys = list(fits)
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.6), squeeze=False, sharey=True)
    for ax, key in zip(axes[0], keys):
        f = fits[key]
        pts = sorted((gen_of(n), f.rating(n), f.se(n)) for n in f.names if gen_of(n) is not None)
        if pts:
            x, y, e = zip(*pts)
            ax.errorbar(x, y, yerr=e, fmt="o-", ms=4, lw=1.5, elinewidth=1, capsize=0, color="#2a6fdb")
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("generation")
        ax.grid(True, color="#dddddd", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel(f"Elo (anchor {fits[keys[0]].anchor} = 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir", nargs="?", type=Path)
    p.add_argument("--report", type=Path, action="append", default=[], help="extra report.json")
    p.add_argument("--group", choices=["board", "set", "all"], default="board")
    p.add_argument("--anchor", default="auto", help="net fixed at 0 Elo (default: gen00, else the init)")
    p.add_argument("--fixed-only", action="store_true", help="drop SPRT-stopped rows")
    p.add_argument("--prior-pairs", type=float, default=1.0)
    p.add_argument("--no-dispersion", action="store_true")
    p.add_argument("--json", type=Path)
    p.add_argument("--csv", type=Path)
    p.add_argument("--png", type=Path)
    p.add_argument("--summary", metavar="NET")
    p.add_argument("--back", type=int, default=10, help="--summary's Δ span in generations")
    p.add_argument("--nearest", metavar="NET")
    p.add_argument("--among", default="")
    p.add_argument("--k", type=int, default=2)
    a = p.parse_args(argv)

    results = shorten(collect(a.run_dir, a.report))
    if a.fixed_only:
        results = [r for r in results if not r.sprt]
    among = [x for x in re.split(r"[,\s]+", a.among) if x]
    if not results:
        if a.nearest:
            print("\n".join(nearest(None, a.nearest, among, a.k)))
            return 0
        print("no drive pairs yet")
        return 0
    anchor = auto_anchor(results, a.run_dir) if a.anchor == "auto" else a.anchor
    disp = not a.no_dispersion

    if a.nearest:
        fits = fit_groups(results, "all", anchor, a.prior_pairs, disp)
        print("\n".join(nearest(fits.get("all"), a.nearest, among, a.k)))
        return 0

    fits = fit_groups(results, a.group, anchor, a.prior_pairs, disp)
    if a.summary:
        print(summary(fits, a.summary, a.back))
    else:
        print("\n\n".join(table(k, f) for k, f in fits.items()))
    if a.json:
        a.json.write_text(json.dumps({k: {"anchor": f.anchor, "dispersion": f.dispersion,
                                          "dropped": f.dropped, "nets": [r for r in rows({k: f})]}
                                      for k, f in fits.items()}, indent=1))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["group", "net", "gen", "elo", "se", "pairs"])
            w.writeheader()
            w.writerows(rows(fits))
    if a.png:
        plot(fits, a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
