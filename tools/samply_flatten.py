#!/usr/bin/env python3
"""Flatten samply processed-profile JSON + symbolicate via atos.

Usage: python3 samply_flatten.py prof.json binary [thread_index]
"""
import json, subprocess, sys
from collections import Counter

path = sys.argv[1]
binary = sys.argv[2]
thread_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None
LOAD_ADDR = 0x100000000  # macOS arm64 __TEXT default

with open(path) as f:
    prof = json.load(f)

if thread_idx is None:
    thread_idx = max(range(len(prof["threads"])),
                     key=lambda i: prof["threads"][i]["samples"].get("length", 0))
th = prof["threads"][thread_idx]
strings = th["stringArray"]
funcs = th["funcTable"]["name"]
frames_func = th["frameTable"]["func"]
frames_addr = th["frameTable"]["address"]
stk_frame = th["stackTable"]["frame"]
stk_prefix = th["stackTable"]["prefix"]

# Map stack idx → address of leaf frame.
def frame_addr_for_stack(s):
    return frames_addr[stk_frame[s]]

# Collect every distinct address that appears anywhere in any sampled stack.
addrs = set()
samples_stack = th["samples"]["stack"]
total = sum(1 for s in samples_stack if s is not None)
for s in samples_stack:
    if s is None: continue
    cur = s
    while cur is not None:
        addrs.add(frames_addr[stk_frame[cur]])
        cur = stk_prefix[cur]

# Symbolicate via atos.
sorted_addrs = sorted(addrs)
hex_addrs = [hex(LOAD_ADDR + a) for a in sorted_addrs]
print(f"# symbolicating {len(hex_addrs)} addresses via atos ...", file=sys.stderr)
proc = subprocess.run(
    ["atos", "-o", binary, "-l", hex(LOAD_ADDR)] + hex_addrs,
    capture_output=True, text=True, check=True,
)
sym_lines = proc.stdout.strip().splitlines()
sym = dict(zip(sorted_addrs, sym_lines))

def clean(s):
    # Strip module suffix "(in binary) (file.rs:N)" and the trailing hash.
    s = s.split(" (in ")[0]
    # Strip generic-hash suffix ::h<hex>
    if "::h" in s:
        head, _, tail = s.rpartition("::h")
        if len(tail) == 16 and all(c in "0123456789abcdef" for c in tail):
            s = head
    return s

self_counts = Counter()
incl_counts = Counter()
for s in samples_stack:
    if s is None: continue
    leaf = clean(sym[frame_addr_for_stack(s)])
    self_counts[leaf] += 1
    seen = set()
    cur = s
    while cur is not None:
        name = clean(sym[frames_addr[stk_frame[cur]]])
        if name not in seen:
            seen.add(name)
            incl_counts[name] += 1
        cur = stk_prefix[cur]

print(f"# thread: {th['name']!r}  total_samples: {total}")

def fmt(counts, n_top, header):
    print(f"\n## {header}")
    print(f"{'pct':>6}  {'samples':>8}  name")
    for name, c in counts.most_common(n_top):
        print(f"{100*c/total:6.2f}  {c:8d}  {name}")

fmt(self_counts, 30, "Self-time top 30")
fmt(incl_counts, 30, "Inclusive-time top 30")

import re
groups = {
    "PathFinder::player_paths": re.compile(r"player_paths|::pathing::"),
    "GameState::clone (incl. inner clones)": re.compile(
        r"botbowl_engine::core::gamestate::GameState.*Clone|"
        r"::core::gamestate.*::clone\b"
    ),
    "drop_in_place [Option<FieldedPlayer>; 22]": re.compile(
        r"drop_in_place.*FieldedPlayer.*22"
    ),
    "AvailableActions clone/drop": re.compile(
        r"AvailableActions|FullPitch.*clone|FullPitch.*drop"
    ),
    "BloodBowlDynamics::apply_action": re.compile(r"BloodBowlDynamics.*apply_action"),
    "GameState::micro_step": re.compile(r"GameState.*micro_step|micro_step"),
    "BloodBowlDynamics::select_node / puct / priors": re.compile(
        r"BloodBowlDynamics.*select_node|puct_value|prior_for"
    ),
    "score_leaf / leaf_score / FF": re.compile(
        r"BloodBowlDynamics.*score_leaf|leaf_score|optimistic_leaf_score"
    ),
    "BloodBowlDynamics::available_actions / should_prune": re.compile(
        r"BloodBowlDynamics.*available_actions|should_prune"
    ),
    "Tree internals (recon_mcts)": re.compile(r"recon_mcts::"),
    "RwLock / parking_lot": re.compile(r"RwLock|parking_lot"),
    "HashMap / HashSet / hashbrown": re.compile(r"HashMap|HashSet|hashbrown|RawTable"),
    "Arc::* / drop_in_place Arc": re.compile(r"alloc::sync::Arc|drop_in_place.*Arc"),
    "alloc / dealloc / malloc / free": re.compile(
        r"::alloc::|__rust_alloc|__rust_dealloc|malloc|free|posix_memalign"
    ),
    "memcpy / memmove": re.compile(r"memcpy|memmove|memset"),
}
print("\n## Plan-011 grouped inclusive shares (any frame in stack matches)")
print(f"{'pct':>6}  {'samples':>8}  group")
for g, pat in groups.items():
    hit = sum(c for name, c in incl_counts.items() if pat.search(name))
    print(f"{100*hit/total:6.2f}  {hit:8d}  {g}")
