#!/usr/bin/env python3
"""Draw the kickoff setups the loop's bots chose (plan 047's per-player setup), as ASCII boards.

    scripts/show_setups.py runs/loopmix16x9v9/gen02/shard0.jsonl [--n 3] [--board 16x9]

For each `--next-drive` record (meta.extra.drive = 2), the first state after both setups are done:
`H`/`A` for the two teams (lower case for a player with skills), `o` for the ball, `|` on the
halfway line. Each team's skills are listed under the board.
"""
import argparse
import json


def proc_top(state):
    ps = state.get("proc_stack") or []
    return next(iter(ps[-1])) if ps and isinstance(ps[-1], dict) else str(ps[-1]) if ps else "?"


def render(state):
    d = state["board_dims"]
    w, h = d["width"], d["height"]
    grid = [["." for _ in range(w + 2)] for _ in range(h + 2)]
    skills = {"Home": [], "Away": []}
    for p in filter(None, state.get("fielded_players") or []):
        pos = p.get("position") or {}
        x, y = pos.get("x"), pos.get("y")
        if x is None or not (0 <= y < h + 2 and 0 <= x < w + 2):
            continue
        team = p["stats"]["team"]
        sk = p["stats"].get("skills") or []
        ch = "H" if team == "Home" else "A"
        grid[y][x] = ch.lower() if sk else ch
        if sk:
            skills[team].append(f"{p['stats'].get('role', '?')}[{','.join(sk)}]")
    ball = state.get("ball") or {}
    bpos = ball.get("position") if isinstance(ball, dict) else None
    if isinstance(bpos, dict) and grid[bpos["y"]][bpos["x"]] == ".":
        grid[bpos["y"]][bpos["x"]] = "o"
    mid = (w + 2) // 2
    lines = []
    for row in grid[1:-1]:
        cells = row[1:-1]
        lines.append(" ".join(cells[: mid - 1]) + " | " + " ".join(cells[mid - 1 :]))
    return lines, skills, f"{w}x{h}/{d['team_size']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shard")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--board", default=None, help="only this playable board, e.g. 16x9")
    a = ap.parse_args()
    shown = 0
    for line in open(a.shard):
        t = json.loads(line)
        if t["meta"].get("extra", {}).get("drive") != "2":
            continue
        d = t["meta"]["board_dims"]
        if a.board and f"{d['width']}x{d['height']}" != a.board:
            continue
        # The first decision after both setups: the kickoff is placed or the receiving turn begins.
        after = next((s["state"] for s in t["samples"] if not proc_top(s["state"]).lower().startswith("setup")), None)
        if after is None:
            continue
        lines, skills, label = render(after)
        o = t.get("outcome") or {}
        print(f"seed {t['meta'].get('seed')}  {label}  drive outcome {o.get('home_score')}-{o.get('away_score')}"
              f"  (start {t['meta']['extra'].get('start_score')})")
        print("\n".join("   " + l for l in lines))
        for team in ("Home", "Away"):
            if skills[team]:
                print(f"   {team}: {'; '.join(skills[team])}")
        print()
        shown += 1
        if shown >= a.n:
            break


if __name__ == "__main__":
    main()
