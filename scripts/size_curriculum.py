#!/usr/bin/env python3
"""Plan 042: move the board-size centre when the generation-side TD rate says
the bots can already score at the sizes around it.

    size_curriculum.py RUN_DIR --gen G [--advance 0.75] [--step 1.25]
                       [--max-area 144] [--min-drives 200] [--band 0.85]

Reads
  RUN_DIR/size_centre.txt          the current centre (playable area); created
                                   from --init-centre when missing
  RUN_DIR/genGG/td_rate.json       `td_rate.py --json` for this generation,
                                   with a per-board breakdown
  RUN_DIR/size_baseline.json       optional: `td_rate.py --json` of a reference
                                   corpus (heuristic MCTS at every size) — the
                                   TD rate a board "should" have regardless of
                                   the net; without it the rule is absolute

Rule (one line in status.md, one row appended to size_curve.tsv):
  Pool the drives of every board whose playable area is >= --band * centre
  ("at or above the centre"). Their TD/drive, divided by the same boards'
  baseline TD/drive when a baseline exists, is the *relative* rate. If it is
  >= --advance over at least --min-drives drives, the centre becomes
  min(centre * --step, --max-area); otherwise it stays. Never moves down —
  the uniform floor in the sampler keeps every size in the corpus anyway, and
  a regression shows up on the per-board eval rungs, which is where the
  operator decides.

Why TD rate and not the vs-scripted ladder: the ladder is the honest
measurement but costs games (~600 for a 0.03 effect, plan 032 D10), while
the corpus's own conversion rate is free, already tracked (td_rate.py), and
is exactly the sparse-reward signal the small boards exist to provide. The
ladder confirms per size; this only decides where the *next* corpus is
centred. Everything here is advisory in the plan-030 sense: the operator can
overwrite size_centre.txt by hand at any phase boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def area_of(board: str) -> float:
    """`14x7/4` -> 98.0 (playable area)."""
    wh = board.split("/")[0]
    w, h = wh.lower().split("x")
    return float(int(w) * int(h))


def pooled(per_board: dict, boards: list[str]) -> tuple[int, int]:
    drives = sum(per_board[b]["drives"] for b in boards)
    tds = sum(per_board[b]["home_tds"] + per_board[b]["away_tds"] for b in boards)
    return drives, tds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--init-centre", type=float, default=98.0, help="centre to create when none is stored")
    ap.add_argument("--advance", type=float, default=0.75, help="relative TD rate that moves the centre")
    ap.add_argument("--step", type=float, default=1.25, help="multiply the centre by this on advance")
    ap.add_argument("--max-area", type=float, default=144.0, help="never centre above this (the build's capacity)")
    ap.add_argument("--min-drives", type=int, default=200, help="drives needed at/above the centre before deciding")
    ap.add_argument("--band", type=float, default=0.85, help="boards with area >= band*centre count as at-or-above")
    ap.add_argument("--dry-run", action="store_true", help="decide and print, write nothing")
    a = ap.parse_args()

    run = Path(a.run_dir)
    centre_file = run / "size_centre.txt"
    if centre_file.exists():
        centre = float(centre_file.read_text().strip())
    else:
        centre = a.init_centre
        if not a.dry_run:
            centre_file.write_text(f"{centre:g}\n")

    gen_dir = run / f"gen{a.gen:02d}"
    td_path = gen_dir / "td_rate.json"
    if not td_path.exists():
        print(f"size centre {centre:g} unchanged: no {td_path.name} for gen{a.gen:02d}")
        return 0
    td = json.loads(td_path.read_text())
    per_board = td.get("per_board") or {}
    if not per_board:
        print(f"size centre {centre:g} unchanged: {td_path.name} has no per-board breakdown")
        return 0

    baseline = None
    bpath = run / "size_baseline.json"
    if bpath.exists():
        baseline = (json.loads(bpath.read_text()).get("per_board")) or None

    above = [b for b in per_board if area_of(b) >= a.band * centre]
    if not above:
        print(f"size centre {centre:g} unchanged: no drives at or above {a.band:g}x the centre")
        return 0
    drives, tds = pooled(per_board, above)
    rate = tds / drives if drives else 0.0
    if baseline is not None:
        common = [b for b in above if b in baseline and baseline[b]["drives"] > 0]
        if common:
            bd, bt = pooled(baseline, common)
            base_rate = bt / bd if bd else 0.0
            rel = rate / base_rate if base_rate > 0 else 0.0
            how = f"TD/drive {rate:.3f} vs baseline {base_rate:.3f} = {rel:.2f}"
        else:
            rel, how = rate, f"TD/drive {rate:.3f} (no baseline for these boards)"
    else:
        rel, how = rate, f"TD/drive {rate:.3f} (absolute, no baseline)"

    decision = "hold"
    new_centre = centre
    if drives < a.min_drives:
        why = f"only {drives} drives at/above centre (< {a.min_drives})"
    elif rel >= a.advance:
        new_centre = min(centre * a.step, a.max_area)
        decision = "advance" if new_centre > centre else "capped"
        why = f"{rel:.2f} >= {a.advance:g}"
    else:
        why = f"{rel:.2f} < {a.advance:g}"

    line = (
        f"size centre {centre:g} -> {new_centre:g} ({decision}: {why}; {how} over {drives} drives "
        f"on {len(above)} board(s) {' '.join(sorted(above, key=area_of))})"
    )
    print(line)
    if not a.dry_run:
        centre_file.write_text(f"{new_centre:g}\n")
        with open(run / "size_curve.tsv", "a") as f:
            f.write(f"{a.gen}\t{centre:g}\t{new_centre:g}\t{decision}\t{rate:.4f}\t{rel:.4f}\t{drives}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
