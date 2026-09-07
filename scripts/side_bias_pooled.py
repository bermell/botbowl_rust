#!/usr/bin/env python3
"""Plan 032 #5 (side bias): pool every per-game log and split the Home/Away
result by who kicked in the first half.

Every paired match already plays each seed from both seats, so the *candidate*
effect cancels when games are grouped by seat rather than by candidate: the
Home share of points across a pooled set of logs is a side-bias estimate on
its own, no dedicated mirror needed. Splitting by `kicking_first_half`
separates "Home is favoured" from "receiving first is favoured".

    scripts/side_bias_pooled.py runs/exp032/*.games.jsonl runs/exp-data/*.games.jsonl
"""
import argparse
import json
import math
from collections import defaultdict


def points_home(r):
    h, a = r["home_score"], r["away_score"]
    return 1.0 if h > a else 0.5 if h == a else 0.0


def summarise(name, rows):
    n = len(rows)
    if n == 0:
        return
    p = sum(points_home(r) for r in rows) / n
    w = sum(1 for r in rows if r["home_score"] > r["away_score"])
    d = sum(1 for r in rows if r["home_score"] == r["away_score"])
    var = sum((points_home(r) - p) ** 2 for r in rows) / max(n - 1, 1)
    se = math.sqrt(var / n)
    z = (p - 0.5) / se if se > 0 else float("nan")
    td_h = sum(r["home_score"] for r in rows)
    td_a = sum(r["away_score"] for r in rows)
    print(f"  {name:<34} n={n:<5} Home pts {p:.3f} ± {se:.3f}  z={z:+.2f}   W{w} D{d} L{n - w - d}   TD {td_h}:{td_a}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--per-file", action="store_true")
    a = ap.parse_args()

    rows = []
    for path in a.logs:
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("finished", True):
                        r["_file"] = path
                        rows.append(r)

    # A paired match replays each seed with the seats swapped. When both seats
    # hold the *same deterministic* bot (scripted mirror, random mirror, a
    # single-worker NN mirror with no knob difference) the two games are the
    # same game and count once; otherwise the pair are distinct games.
    # Decided per log, not per pair: two distinct games can coincide (a 0-0
    # draw both ways is common), so only a log where *every* pair coincides
    # is a deterministic mirror.
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[(r["_file"], r["seed"])].append(r)
    same = defaultdict(lambda: [0, 0])
    for (f, _), pair in by_seed.items():
        same[f][1] += 1
        if len(pair) == 2 and all(
            pair[0][k] == pair[1][k] for k in ("home_score", "away_score", "kicking_first_half")
        ) and pair[0]["candidate_team"] != pair[1]["candidate_team"]:
            same[f][0] += 1
    mirror_logs = {f for f, (n_same, n) in same.items() if n >= 20 and n_same == n}
    if mirror_logs:
        deduped = []
        for (f, _), pair in by_seed.items():
            deduped.extend(pair[:1] if f in mirror_logs else pair)
        for f in sorted(mirror_logs):
            print(f"{f.split('/')[-1]}: every seed pair is the same game (deterministic bot in both seats) — counted once")
        rows = deduped
    print(f"{len(rows)} finished games from {len(a.logs)} logs\n")
    print("pooled, by seat:")
    summarise("all", rows)
    by_kick = defaultdict(list)
    for r in rows:
        by_kick[r.get("kicking_first_half", "?")].append(r)
    for k in sorted(by_kick):
        summarise(f"kicking_first_half={k}", by_kick[k])
    # Receiving-first share, seat-agnostic: did the team that received the
    # first-half kickoff win?
    recv = []
    for r in rows:
        k = r.get("kicking_first_half")
        if k not in ("Home", "Away"):
            continue
        ph = points_home(r)
        recv.append(ph if k == "Away" else 1.0 - ph)
    if recv:
        n = len(recv)
        p = sum(recv) / n
        se = math.sqrt(sum((x - p) ** 2 for x in recv) / max(n - 1, 1) / n)
        print(f"  {'receiving team (seat-agnostic)':<34} n={n:<5} pts {p:.3f} ± {se:.3f}  z={(p - 0.5) / se:+.2f}")

    if a.per_file:
        print("\nper log:")
        by_file = defaultdict(list)
        for r in rows:
            by_file[r["_file"]].append(r)
        for f in a.logs:
            summarise(f.split("/")[-1].replace(".games.jsonl", ""), by_file[f])


if __name__ == "__main__":
    main()
