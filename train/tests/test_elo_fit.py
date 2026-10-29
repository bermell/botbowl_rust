"""Plan 071 R4: `scripts/elo_fit.py` recovers known ratings from synthetic pentanomial pair results,
its SEs are calibrated, and it reads a run directory the way train_loop.sh lays it out."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import elo_fit  # noqa: E402

TRUE = {"gen00": 0.0, "gen01": 60.0, "gen02": 110.0, "gen04": 190.0, "gen08": 240.0, "gen16": 330.0}


def simulate_pairs(rng, d_elo, n, draw=0.2):
    """Pentanomial counts for n pairs; each game a draw w.p. `draw`, else a win w.p. chosen so the
    expected game score (and so the pair score) is exactly the Elo expectation."""
    p = float(elo_fit.expected(d_elo))
    w = (p - draw / 2) / (1 - draw)
    assert 0 <= w <= 1
    g = rng.choice([0.0, 0.5, 1.0], size=(n, 2), p=[(1 - draw) * (1 - w), draw, (1 - draw) * w])
    quarters = np.rint(g.sum(axis=1) * 2).astype(int)  # pair score in quarters, 0..4
    return tuple(int(c) for c in np.bincount(quarters, minlength=5))


def ladder_results(rng, n=200):
    """Each net plays its parent and the geometric anchors below it, as DRIVE_REF=parent +
    DRIVE_LADDER=geo would; plus one origin match."""
    names = list(TRUE)
    res = []
    for i, a in enumerate(names[1:], start=1):
        opps = {names[i - 1]} | set(names[: max(i - 2, 0)][-2:])
        for b in sorted(opps):
            res.append(elo_fit.Result(a, b, simulate_pairs(rng, TRUE[a] - TRUE[b], n), "20x11/7", "s"))
    res.append(elo_fit.Result("gen16", "gen00", simulate_pairs(rng, TRUE["gen16"], n), "20x11/7", "s"))
    return res


def test_recovers_known_ratings_within_their_se():
    rng = np.random.default_rng(1)
    f = elo_fit.fit(ladder_results(rng), "gen00")
    assert f.anchor == "gen00" and f.rating("gen00") == 0 and f.se("gen00") == 0
    for name, true in TRUE.items():
        if name == "gen00":
            continue
        assert abs(f.rating(name) - true) < 3 * f.se(name), (name, f.rating(name), f.se(name))
        assert 5 < f.se(name) < 40
    # Pair scores (two games averaged, 20% draws) are far less variable than one Bernoulli game.
    assert 0.25 < f.dispersion < 0.6


def test_standard_errors_are_calibrated():
    """Across replications the z-scores of every net have unit spread: the dispersion-scaled
    inverse information is the right covariance for pentanomial pair means."""
    rng = np.random.default_rng(7)
    zs = []
    for _ in range(60):
        f = elo_fit.fit(ladder_results(rng, n=150), "gen00", prior_pairs=0)
        zs += [(f.rating(n) - TRUE[n]) / f.se(n) for n in TRUE if n != "gen00"]
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.15
    assert 0.85 < zs.std() < 1.15


def test_one_match_reproduces_the_pentanomial_points_se():
    """A single match: the fitted SE is plan 051's pentanomial points_se, mapped to Elo."""
    counts = (10, 30, 80, 50, 30)
    f = elo_fit.fit([elo_fit.Result("b", "a", counts, "x", "s")], "a", prior_pairs=0)
    n = sum(counts)
    xs = np.repeat(elo_fit.PAIR_SCORES, counts)
    mean, se = xs.mean(), xs.std(ddof=1) / math.sqrt(n)
    assert f.rating("b") == pytest.approx(-400 * math.log10(1 / mean - 1), abs=1e-6)
    d_elo_d_score = 400 / math.log(10) / (mean * (1 - mean))
    assert f.se("b") == pytest.approx(se * d_elo_d_score, rel=0.01)


def test_a_sweep_stays_finite_under_the_prior():
    res = [elo_fit.Result("strong", "weak", (0, 0, 0, 0, 200), "x", "s")]
    f = elo_fit.fit(res, "weak", prior_pairs=1)
    assert 500 < f.rating("strong") < 2000
    assert math.isfinite(f.se("strong"))


def test_disconnected_nets_are_dropped_not_fitted():
    res = [
        elo_fit.Result("b", "a", (0, 10, 20, 10, 0), "x", "s"),
        elo_fit.Result("d", "c", (0, 10, 20, 10, 0), "x", "s"),
    ]
    f = elo_fit.fit(res, "a")
    assert f.names == ["a", "b"] and f.dropped == ["c", "d"]


def write_report(path: Path, cand: str, rows):
    """A `job eval` report.json in the shape train_loop.sh's drive subs hold."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ladder = []
    for opp, board, counts, sprt in rows:
        row = {
            "opponent": f"vs:mcts(nn:/m/{opp}.onnx) [chance_all_partial] drives(contested_{board})@{board}/7",
            "board": f"{board}/7",
            "pairs": {"counts": list(counts)},
            "points": sum(c * v for c, v in zip(counts, elo_fit.PAIR_SCORES)) / sum(counts),
        }
        if sprt:
            row["sprt"] = {"verdict": "H0"}
        ladder.append(row)
    path.write_text(json.dumps({"candidate": f"mcts(nn:/m/{cand}.onnx)@chance_all_partial", "ladder": ladder}))


def test_reads_a_run_dir_names_nets_and_groups_by_board(tmp_path, capsys):
    run = tmp_path / "run"
    rng = np.random.default_rng(3)
    t = "hpc2011"
    own = lambda g: f"bbnet_{t}_gen{g:02d}"  # noqa: E731
    true = {0: 0.0, 1: 80.0, 2: 150.0, 3: 200.0, 4: 260.0}
    yard = 230.0  # the foreign gen58 yardstick
    for g in range(1, 5):
        gd = run / f"gen{g:02d}"
        rows = [(own(g - 1) if g > 1 else "random_init", "20x11", simulate_pairs(rng, true[g] - true[g - 1], 200), True)]
        write_report(gd / "drives" / "report.json", own(g), rows)
        for a in (1, 2):
            if a < g - 1:
                write_report(gd / f"drives_ladder_gen{a:02d}" / "report.json", own(g),
                             [(own(a), "20x11", simulate_pairs(rng, true[g] - true[a], 200), False)])
        write_report(gd / "drives_origin" / "report.json", own(g),
                     [("bbnet_mix16x9v9_gen58", "20x11", simulate_pairs(rng, true[g] - yard, 200), False),
                      ("bbnet_mix16x9v9_gen58", "16x9", simulate_pairs(rng, true[g] - yard, 200), False)])
    # gen00 never played, so the anchor is the init gen01 played in its `drives` sub.
    assert elo_fit.main([str(run), "--json", str(tmp_path / "e.json"), "--csv", str(tmp_path / "e.csv")]) == 0
    out = json.loads((tmp_path / "e.json").read_text())
    assert set(out) == {"20x11/7", "16x9/7"}
    big = {r["net"]: r for r in out["20x11/7"]["nets"]}
    assert out["20x11/7"]["anchor"] == "random_init"
    # The run's own nets are genNN; the foreign gen58 keeps its stem and does not merge with them.
    assert {"gen01", "gen02", "gen03", "gen04", "bbnet_mix16x9v9_gen58"} <= set(big)
    for g in range(1, 5):
        assert abs(big[f"gen{g:02d}"]["elo"] - true[g]) < 3 * big[f"gen{g:02d}"]["se"]
    assert big["gen04"]["gen"] == 4
    # 16x9 has no init in it: anchored at its lowest own gen, with a note on stderr.
    assert out["16x9/7"]["anchor"] == "gen01"
    assert "anchored at gen01" in capsys.readouterr().err
    assert (tmp_path / "e.csv").read_text().startswith("group,net,gen,elo,se,pairs")

    # --fixed-only drops the SPRT-stopped parent rows: gen01 then has no route to random_init.
    elo_fit.main([str(run), "--fixed-only", "--json", str(tmp_path / "f.json")])
    fixed = json.loads((tmp_path / "f.json").read_text())
    assert "random_init" not in {r["net"] for r in fixed["20x11/7"]["nets"]}

    capsys.readouterr()
    elo_fit.main([str(run), "--summary", "gen04", "--back", "2"])
    line = capsys.readouterr().out.strip()
    assert line.startswith("16x9/7: gen04") and "| 20x11/7: gen04" in line and "(Δ2 " in line

    # Ladder choice: the anchors whose pooled rating sits nearest the parent's; unrated ones last.
    elo_fit.main([str(run), "--nearest", "gen04", "--among", "gen01,gen02,gen16", "--k", "2"])
    assert capsys.readouterr().out.split() == ["gen02", "gen01"]
    elo_fit.main([str(run), "--nearest", "gen09", "--among", "gen01 gen02 gen08", "--k", "2"])
    assert capsys.readouterr().out.split() == ["gen08", "gen02"]


def test_empty_run_prints_a_placeholder(tmp_path, capsys):
    (tmp_path / "gen01").mkdir()
    assert elo_fit.main([str(tmp_path)]) == 0
    assert "no drive pairs yet" in capsys.readouterr().out
    elo_fit.main([str(tmp_path), "--nearest", "gen03", "--among", "gen01,gen02", "--k", "1"])
    assert capsys.readouterr().out.split() == ["gen02"]
