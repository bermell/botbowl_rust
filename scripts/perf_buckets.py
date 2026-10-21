#!/usr/bin/env python3
"""Split a `perf record -g` profile of a generation run into plan 069 §1's buckets.

    perf script -i perf.data -F comm,tid,period,ip,sym,dso | scripts/perf_buckets.py [--top N]

Each sample is weighted by its period (record with `-e instructions:u -c N` for instruction
shares, `-e cycles:u` for time) and assigned by its call chain, checked in this order:

  encoder    any frame in `botbowl_nn::encode` (the path planes' pathfinder included: reported
             as its own sub-line when a `pathing`/`PathFinder` frame sits under the encoder);
  inference  the rest of the NN client: `NnEvaluator` (memo key compare and clone, priors gather,
             softmax), the batch queue, the in-process executor, and every frame in libtorch /
             libc10 / CUDA libraries (the batcher threads);
  search     recon_mcts, botbowl_mcts and botbowl_engine (stepping, cloning, hashing, eq,
             path-finding for moves, the allocator under them);
  rest       everything else (game driver, JSON, process start-up).

LBR call stacks are at most 32 frames deep, so a frame that names the bucket may be cut off a very
deep chain; the `unattributed` line counts samples whose chain has no frame of ours at all.
"""
import argparse
import collections
import re
import sys

NN_DSO = re.compile(r"lib(torch|c10|cuda|cudart|cudnn|cublas|nvrtc|nvJitLink|caffe2|gomp|shm)", re.I)
ENC = re.compile(r"botbowl_nn::encode::")
PATH = re.compile(r"pathing|PathFinder|player_paths")
INF = re.compile(r"botbowl_nn::(eval|batch_queue|inproc|actions|perspective)|NnEvaluator|bbnn_|inproc_shim")
SEARCH = re.compile(r"recon_mcts|botbowl_mcts|botbowl_engine")


def samples(stream):
    head, frames = None, []
    for line in stream:
        line = line.rstrip("\n")
        if not line.strip():
            if head is not None:
                yield head, frames
            head, frames = None, []
            continue
        if head is None:
            head = line
            continue
        parts = line.strip().split(None, 1)
        rest = parts[1] if len(parts) > 1 else ""
        m = re.match(r"(.*) \((.*)\)$", rest)
        sym, dso = (m.group(1), m.group(2)) if m else (rest, "")
        frames.append((sym, dso))
    if head is not None:
        yield head, frames


def bucket(frames):
    syms = [s for s, _ in frames]
    for i, s in enumerate(syms):
        if ENC.search(s):
            return "encoder/pathfinder" if any(PATH.search(x) for x in syms[:i]) else "encoder/other"
    if any(INF.search(s) for s in syms) or any(NN_DSO.search(d) for _, d in frames):
        return "inference/client+batcher"
    if any(SEARCH.search(s) for s in syms):
        if any(PATH.search(s) for s in syms):
            return "search/pathfinder"
        return "search/other"
    if any("botbowl" in s for s in syms):
        return "rest"
    return "rest/unattributed"


def period_of(head):
    # `comm tid period` with -F comm,tid,period: the period is the last integer field.
    nums = [t for t in head.split() if t.isdigit()]
    return int(nums[-1]) if nums else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="also list the top N leaf symbols per bucket")
    ap.add_argument("--after", default=None,
                    help="skip samples until one whose chain matches this regex (e.g. run_search: "
                         "drop the start-up, libtorch load and CUDA graph capture); needs time order")
    args = ap.parse_args()
    tot = collections.Counter()
    leaves = collections.defaultdict(collections.Counter)
    n = skipped = 0
    started = args.after is None
    after = re.compile(args.after) if args.after else None
    for head, frames in samples(sys.stdin):
        if not frames:
            continue
        if not started:
            if any(after.search(s) for s, _ in frames):
                started = True
            else:
                skipped += period_of(head)
                continue
        p = period_of(head)
        b = bucket(frames)
        tot[b] += p
        leaves[b][frames[0][0]] += p
        n += 1
    total = sum(tot.values()) or 1
    print(f"samples {n}, weight {total}" + (f", skipped before --after: {skipped}" if args.after else ""))
    for b in sorted(tot, key=lambda k: -tot[k]):
        print(f"{b:28s} {100.0 * tot[b] / total:6.2f}%")
    for top in ("encoder", "inference", "search", "rest"):
        s = sum(v for k, v in tot.items() if k.split("/")[0] == top)
        print(f"= {top:26s} {100.0 * s / total:6.2f}%")
    if args.top:
        for b in sorted(tot, key=lambda k: -tot[k]):
            print(f"\n-- {b}: top leaf symbols")
            for sym, v in leaves[b].most_common(args.top):
                print(f"   {100.0 * v / total:6.2f}%  {sym[:140]}")


if __name__ == "__main__":
    main()
