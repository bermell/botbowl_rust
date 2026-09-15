#!/usr/bin/env python3
"""Touchdown rate of a generation's self-play corpus.

    train/.venv/bin/python scripts/td_rate.py runs/az14x7/gen01
    train/.venv/bin/python scripts/td_rate.py runs/az14x7/gen01/shard0.jsonl ...

`--mode random-start` plays one *drive*: it starts from a random mid-game
placement and stops as soon as either score changes or the half ends
(`dataset.rs::random_start_trajectory`). So the scoreline in `outcome` is
absolute, and the drive's own touchdowns are `outcome - meta.extra.start_score`
— the shard logs' `score=H-A` line alone cannot tell you this, because it
does not carry the starting score.

What the numbers mean for a from-scratch run: a drive that ends without a
touchdown ran out of half, so `scored` is the share of drives the bots
actually converted. A cold net should sit low and the rate should climb as
the value head learns that reaching the end zone is what the +1 is for. A
*falling* rate with a rising anchor score means the loop is learning to
defend faster than it learns to attack.

Streaming and O(1) in corpus size: `meta` is at the head of each line and
`outcome` at the tail, so neither the samples nor the whole line are parsed.
A generation is ~850 MB across 8 shards and takes a few seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# `"start_score":"1-1"` lives in meta.extra, within the first ~2 KB of a line;
# `"outcome":{...}` is the last object on it. Both are unique per line.
START_RE = re.compile(rb'"start_score":"(\d+)-(\d+)"')
OUTCOME_RE = re.compile(rb'"outcome":\{"home_score":(\d+),"away_score":(\d+)')

HEAD = 4096   # bytes of the line scanned for meta.extra
TAIL = 256    # bytes scanned for the outcome object


def scan(path: Path) -> dict:
    acc = {"drives": 0, "scored": 0, "home_tds": 0, "away_tds": 0, "unparsed": 0}
    with open(path, "rb") as f:
        for line in f:
            m0 = START_RE.search(line, 0, HEAD)
            m1 = OUTCOME_RE.search(line, max(0, len(line) - TAIL))
            if not m0 or not m1:
                acc["unparsed"] += 1
                continue
            h = int(m1.group(1)) - int(m0.group(1))
            a = int(m1.group(2)) - int(m0.group(2))
            acc["drives"] += 1
            acc["home_tds"] += h
            acc["away_tds"] += a
            if h or a:
                acc["scored"] += 1
    return acc


def summarize(acc: dict) -> str:
    n = acc["drives"]
    if not n:
        return "no drives"
    tds = acc["home_tds"] + acc["away_tds"]
    # A drive stops on the score, so scored == tds except in the rare double
    # score; printing it only when it diverges keeps the status line short and
    # still surfaces the anomaly.
    odd = "" if acc["scored"] == tds else f", scored {acc['scored'] / n:.3f}"
    warn = f", {acc['unparsed']} UNPARSED" if acc["unparsed"] else ""
    return (
        f"TD/drive {tds / n:.3f} over {n} drives "
        f"(H {acc['home_tds'] / n:.3f} A {acc['away_tds'] / n:.3f})" + odd + warn
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="shard .jsonl files, or a generation dir")
    ap.add_argument("--per-shard", action="store_true", help="also print a line per shard")
    ap.add_argument("--json", default=None, help="write the aggregate here")
    a = ap.parse_args()

    shards: list[Path] = []
    for p in map(Path, a.paths):
        shards.extend(sorted(p.glob("shard*.jsonl")) if p.is_dir() else [p])
    if not shards:
        print("no shards found", file=sys.stderr)
        return 1

    total = {"drives": 0, "scored": 0, "home_tds": 0, "away_tds": 0, "unparsed": 0}
    for s in shards:
        acc = scan(s)
        if a.per_shard:
            print(f"{s.name}: {summarize(acc)}")
        for k in total:
            total[k] += acc[k]

    print(summarize(total))
    if a.json:
        Path(a.json).write_text(json.dumps(total, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
