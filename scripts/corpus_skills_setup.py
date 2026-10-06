#!/usr/bin/env python3
"""How a loop's bots use the new rules (plan 058): skill choices and kickoff setups, per generation.

    scripts/corpus_skills_setup.py runs/loopmix16x9v9 [--gens 1-5]

Per generation (raw shards, all eight):
- records: random-start drives (drive 1) and --next-drive records (drive 2: both kickoff setups,
  the kick and the drive), and the touchdown rate of each;
- optional-skill decisions (`UseSkill` / `DontUseSkill`): by the procedure that asked (the skill),
  how often it was offered and how often the bot used it;
- setup decisions in drive-2 records: setup actions per setup, and how many players each side put
  on the line of scrimmage (the column next to the halfway line), as a crude shape of the formation.
"""
import argparse
import collections
import glob
import json
import os
import sys


def action_name(a):
    if isinstance(a, dict):
        if "Simple" in a:
            return a["Simple"]
        if "Positional" in a:
            return a["Positional"][0]
    return str(a)


def proc_top(state):
    ps = state.get("proc_stack") or []
    if not ps:
        return "?"
    top = ps[-1]
    return next(iter(top)) if isinstance(top, dict) else str(top)


def gen_dirs(run, spec):
    ds = sorted(glob.glob(os.path.join(run, "gen[0-9][0-9]")))
    if spec:
        a, _, b = spec.partition("-")
        lo, hi = int(a), int(b or a)
        ds = [d for d in ds if lo <= int(d[-2:]) <= hi]
    return [d for d in ds if os.path.exists(os.path.join(d, ".generated"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--gens", default=None, help="e.g. 1-5")
    a = ap.parse_args()
    for d in gen_dirs(a.run, a.gens):
        drives = collections.Counter()
        tds = collections.Counter()
        offered = collections.Counter()
        used = collections.Counter()
        setup_actions = collections.Counter()
        setups = 0
        los = []
        for shard in sorted(glob.glob(os.path.join(d, "shard[0-9].jsonl"))):
            for line in open(shard):
                if not line.strip():
                    continue
                t = json.loads(line)
                drive = int(t["meta"].get("extra", {}).get("drive", "1"))
                drives[drive] += 1
                o = t.get("outcome") or {}
                start = t["meta"].get("extra", {}).get("start_score", "0-0").split("-")
                try:
                    scored = (o.get("home_score", 0) + o.get("away_score", 0)) > (int(start[0]) + int(start[1]))
                except (ValueError, TypeError):
                    scored = False
                tds[drive] += scored
                in_setup = False
                for s in t["samples"]:
                    name = action_name(s["chosen_action"])
                    if name in ("UseSkill", "DontUseSkill"):
                        skill = proc_top(s["state"])
                        offered[skill] += 1
                        used[skill] += name == "UseSkill"
                    if drive == 2 and proc_top(s["state"]).lower().startswith("setup"):
                        setup_actions[name] += 1
                        if not in_setup:
                            setups += 1
                            in_setup = True
                    elif in_setup:
                        in_setup = False
                        # The formation the setup ended in: players on the line of scrimmage.
                        st = s["state"]
                        w = st["board_dims"].get("width") if isinstance(st.get("board_dims"), dict) else None
                        if w:
                            mid = w // 2
                            n = 0
                            for p in st.get("fielded_players") or []:
                                pos = p.get("position") if isinstance(p, dict) else None
                                if isinstance(pos, dict) and pos.get("x") in (mid, mid + 1):
                                    n += 1
                            los.append(n)
        n1, n2 = drives[1], drives[2]
        print(f"== {os.path.basename(d)}: {n1} drives (TD rate {tds[1] / max(n1, 1):.2f}), "
              f"{n2} next-drive records (TD rate {tds[2] / max(n2, 1):.2f})")
        if offered:
            print("   optional skills by the procedure that asked (offered -> used):  " + "  ".join(
                f"{k} {offered[k]}->{used[k]} ({used[k] / offered[k]:.0%})" for k, _ in offered.most_common()))
        else:
            print("   optional skills: none offered")
        if setups:
            print(f"   setups: {setups}, {sum(setup_actions.values()) / setups:.1f} setup decisions each; "
                  f"actions {dict(setup_actions.most_common(6))}"
                  + (f"; players on the line of scrimmage after setup: mean {sum(los) / len(los):.1f}" if los else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
