#!/usr/bin/env python3
"""Screen a drive-rung position set down to its contested positions (plan 051 step 4).

    positions_screen.py --set cfgs/positions/cand_14x7.json --games runs/screen/eval.games.jsonl \\
        --reference models/az_v7/bbnet_mix16x9_gen21.onnx --band 0.25:0.75 \\
        --out cfgs/positions/contested_14x7.json

`--games` is the per-game file of a drive rung over the *unscreened* set, with the reference bot
on both sides:

    botbowl-hub job eval --evaluator nn --model REF --vs-evaluator nn --vs-model REF \\
        --positions cand_14x7.json --vs-games $((4 * N)) --skip-fixed-rungs ...

Every drive is a playout of its position, whichever side the "candidate" held, because both
sides are the reference. The attacker (the team to move) scored if its own touchdown count went
up. A position is kept when its attacker's scoring rate over at least `--min-playouts` drives
lies inside the band. Both bots scoring every time, or failing every time, means the position
cannot separate two bots.
"""

import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, help="the unscreened position set (`botbowl-ui positions`)")
    p.add_argument("--games", required=True, help="per-game JSONL of the reference self-play drive rung")
    p.add_argument("--reference", required=True, help="the reference bot, recorded in the screen")
    p.add_argument("--band", default="0.25:0.75", help="attacker TD rate to keep, LO:HI inclusive")
    p.add_argument("--min-playouts", type=int, default=4)
    p.add_argument("--name", default=None, help="name of the screened set; default `<set name>_contested`")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    lo, hi = (float(x) for x in a.band.split(":"))
    with open(a.set) as f:
        pos_set = json.load(f)
    if pos_set.get("screen"):
        print(f"{a.set} is already screened", file=sys.stderr)
        return 1
    candidates = set(pos_set["seeds"])

    scored = defaultdict(int)
    played = defaultdict(int)
    with open(a.games) as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            attacker = g.get("attacker")
            if attacker is None or g["seed"] not in candidates:
                continue
            tds = g["home_score"] if attacker == "Home" else g["away_score"]
            played[g["seed"]] += 1
            scored[g["seed"]] += int(tds > 0)

    if not played:
        print(f"no drive lines for this set's seeds in {a.games}", file=sys.stderr)
        return 1
    kept = []
    short = 0
    for seed in pos_set["seeds"]:
        n = played.get(seed, 0)
        if n < a.min_playouts:
            short += 1
            continue
        rate = scored[seed] / n
        if lo <= rate <= hi:
            kept.append({"seed": seed, "attacker_td_rate": rate})

    playouts = min(played[s] for s in pos_set["seeds"] if played.get(s, 0) >= a.min_playouts) if kept else 0
    pos_set["name"] = a.name or f"{pos_set['name']}_contested"
    pos_set["screen"] = {"reference": a.reference, "playouts": playouts, "band": [lo, hi], "kept": kept}
    with open(a.out, "w") as f:
        json.dump(pos_set, f, indent=2)

    rates = [scored[s] / played[s] for s in played]
    print(
        f"kept {len(kept)} of {len(pos_set['seeds'])} positions (band {lo}:{hi}, "
        f"{short} with fewer than {a.min_playouts} playouts); attacker TD rate over all screened "
        f"positions {sum(scored.values()) / sum(played.values()):.3f}, "
        f"share at 0 {sum(r == 0 for r in rates) / len(rates):.2f}, at 1 {sum(r == 1 for r in rates) / len(rates):.2f}"
        f" -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
