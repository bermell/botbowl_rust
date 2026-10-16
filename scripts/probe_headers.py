#!/usr/bin/env python3
"""Plan 064: one line per probe summary in a directory — suite, net, commit, dirty, rows, seconds.

    scripts/probe_headers.py runs/exp075
"""
import glob
import json
import sys

for f in sorted(glob.glob(f"{sys.argv[1]}/*.summary.json")):
    h = json.load(open(f))["header"]
    print(f"{h['suite']:<11} {h['tag']:<28} {h['commit'][:10]} dirty={h['dirty']!s:<5} "
          f"rows {h['rows']:>7} {h['elapsed_s']:>7.0f}s")
