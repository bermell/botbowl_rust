#!/usr/bin/env python3
"""Plan 060: how deep does the search see, and into whose turn?

Reads the per-decision `tree` block (botbowl-data `TreeStats`) that every searched sample carries
since plan 060 and prints the tables plan 060 section 3 reads:

* by the number of the mover's decisions left in its current turn (from the trajectory: later
  samples of the same team in the same half and turn) -- `0` is the turn-ending decision;
* by the procedure on top of the stack at the root;
* by board.

Decisions where the opponent has no turn inside the horizon (`opp_turn_follows = false`, the end
of a half) are filtered out and counted. Unsearched decisions (forced moves, formation setups)
carry no `tree` block and are skipped, but still count as decisions when computing "left".

Shares are per descent (summed over searches, then divided by the summed descents); depth means
are per descent too; p90 and max are per-search values averaged over searches.

    python3 scripts/tree_stats.py runs/<run>/gen07/shard*.jsonl [--top-procs 12]

Columns:
  n         searched decisions in the row
  reach_opp share of descents whose leaf lies in the opponent's following turn or later
  end_opp   share of descents that stopped inside the opponent's turn
  horizon   share of descents that stopped at the horizon (the mover's next turn began)
  term      share that stopped at a known outcome (score / half end / game over)
  depth     leaf depth in plies (decision + chance edges): mean / mean p90 / mean max
  own       leaf depth in the mover's own decisions, mean
  chance    chance edges' share of all plies
  ml_plies  main line (most-visited path) length in plies, mean
  ml_opp    share of searches whose main line reaches the opponent's turn or the horizon
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    if path.endswith(".zst"):
        import zstandard  # optional dependency, only for compressed shards

        return zstandard.open(path, "rt")
    return open(path)


class Acc:
    """Sums over searches; every share and mean is derived at print time."""

    def __init__(self) -> None:
        self.n = 0
        self.descents = 0
        self.plies = 0.0
        self.own = 0.0
        self.chance = 0.0
        self.p90 = 0
        self.max = 0
        self.reach_opp = 0
        self.end_opp = 0
        self.horizon = 0
        self.term = 0
        self.ml_plies = 0
        self.ml_opp = 0

    def add(self, t: dict) -> None:
        d = t["descents"]
        if d == 0:
            return
        ends = t["ends"]
        self.n += 1
        self.descents += d
        self.plies += t["plies"]["mean"] * d
        self.own += t["own"]["mean"] * d
        self.chance += t["chance_plies_mean"] * d
        self.p90 += t["plies"]["p90"]
        self.max += t["plies"]["max"]
        self.reach_opp += t["reached_opp_turn"]
        self.end_opp += ends["opp_turn"]
        self.horizon += ends["horizon"]
        self.term += ends["score"] + ends["half_end"] + ends["game_over"]
        ml = t["main_line"]
        self.ml_plies += ml["plies"]
        self.ml_opp += ml.get("phase") in ("opp_turn", "horizon")

    def row(self) -> list[str]:
        if self.n == 0:
            return ["0"] + ["-"] * 9
        d, n = self.descents, self.n
        pct = lambda x: f"{100.0 * x / d:5.1f}%"
        return [
            str(n),
            pct(self.reach_opp),
            pct(self.end_opp),
            pct(self.horizon),
            pct(self.term),
            f"{self.plies / d:.2f} / {self.p90 / n:.1f} / {self.max / n:.1f}",
            f"{self.own / d:.2f}",
            f"{100.0 * self.chance / self.plies:5.1f}%" if self.plies else "-",
            f"{self.ml_plies / n:.1f}",
            f"{100.0 * self.ml_opp / n:5.1f}%",
        ]


HEAD = ["n", "reach_opp", "end_opp", "horizon", "term", "depth mean/p90/max", "own", "chance", "ml_plies", "ml_opp"]


def left_bucket(k: int) -> str:
    if k <= 2:
        return str(k)
    if k <= 4:
        return "3-4"
    if k <= 9:
        return "5-9"
    return "10+"


LEFT_ORDER = ["0", "1", "2", "3-4", "5-9", "10+"]


def decisions_left(samples: list[dict]) -> list[int]:
    """For each sample, how many later samples the same team takes in the same half and turn."""
    keys = []
    for s in samples:
        info = s["state"]["info"]
        team = s["to_move"]
        turn = info["home_turn"] if team == "Home" else info["away_turn"]
        keys.append((team, info["half"], turn))
    left = [0] * len(samples)
    seen: dict = defaultdict(int)
    for i in range(len(samples) - 1, -1, -1):
        left[i] = seen[keys[i]]
        seen[keys[i]] += 1
    return left


def board_label(meta: dict) -> str:
    b = meta.get("board_dims") or {}
    if not b:
        return "?"
    return f"{b['width'] - 2}x{b['height'] - 2}/{b['team_size']}"


def table(title: str, label: str, rows: list[tuple[str, Acc]]) -> None:
    print(f"\n## {title}\n")
    head = [label] + HEAD
    body = [[name] + acc.row() for name, acc in rows]
    widths = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
    fmt = lambda r: "| " + " | ".join(c.rjust(w) for c, w in zip(r, widths)) + " |"
    print(fmt(head))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in body:
        print(fmt(r))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpora", nargs="+", help="trajectory JSONL files (.jsonl, .jsonl.gz, .jsonl.zst)")
    ap.add_argument("--top-procs", type=int, default=12, help="procedures shown, most frequent first")
    args = ap.parse_args()

    total = Acc()
    by_left: dict[str, Acc] = defaultdict(Acc)
    by_proc: dict[str, Acc] = defaultdict(Acc)
    by_board: dict[str, Acc] = defaultdict(Acc)
    trajectories = decisions = without_tree = no_opp_turn = 0
    for path in args.corpora:
        with open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traj = json.loads(line)
                trajectories += 1
                samples = traj.get("samples", [])
                board = board_label(traj.get("meta", {}))
                for s, left in zip(samples, decisions_left(samples)):
                    decisions += 1
                    t = s.get("tree")
                    if t is None:
                        without_tree += 1
                        continue
                    if not t.get("opp_turn_follows", True):
                        no_opp_turn += 1
                        continue
                    total.add(t)
                    by_left[left_bucket(left)].add(t)
                    by_proc[t.get("proc") or "<none>"].add(t)
                    by_board[board].add(t)

    print(f"# Tree statistics (plan 060): {len(args.corpora)} file(s), {trajectories} trajectories")
    print(
        f"\n{decisions} decisions: {total.n} searched and kept, {without_tree} without a tree block "
        f"(unsearched, or a pre-plan-060 corpus), {no_opp_turn} filtered (no opponent turn inside the horizon)."
    )
    if total.n == 0:
        print("\nNothing to report.")
        return 1
    table("All kept decisions", "", [("all", total)])
    table(
        "By the mover's decisions left in its turn (0 = the turn-ending decision)",
        "left",
        [(k, by_left[k]) for k in LEFT_ORDER if k in by_left],
    )
    procs = sorted(by_proc.items(), key=lambda kv: -kv[1].n)
    table(f"By procedure at the root (top {args.top_procs})", "proc", procs[: args.top_procs])
    table("By board", "board", sorted(by_board.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
