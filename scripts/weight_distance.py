#!/usr/bin/env python3
"""Relative L2 distance between two saved nets, per layer and overall.

    scripts/weight_distance.py models/bbnet_14x7_gen02.pt models/bbnet_14x7_gen03.pt ...

Answers "how far did this training run actually move the weights?" (plan 031, D5b).
For each consecutive pair of files given on the command line it prints

    ||theta_B - theta_A||_2 / ||theta_A||_2

per parameter tensor, per block group, and for the whole network. Learnable
parameters and BatchNorm buffers (running_mean/running_var/num_batches_tracked)
are reported separately: buffers drift under BN momentum whether or not the
gradient moved anything, so mixing them into the headline number overstates
how much learning happened.

Everything is loaded with map_location="cpu" — this never touches the GPU.
"""
import collections
import math
import re
import sys

import torch


BUFFER_SUFFIXES = ("running_mean", "running_var")
# A step counter, not a state: it dominates any norm it is included in.
SKIP_SUFFIXES = ("num_batches_tracked",)


def is_buffer(key):
    return key.endswith(BUFFER_SUFFIXES)


def group_of(key):
    """Coarse block name: blocks.3.conv1.weight -> blocks.3, stem_bn.weight -> stem."""
    m = re.match(r"^(blocks\.\d+)\.", key)
    if m:
        return m.group(1)
    head = key.split(".")[0]
    return head.replace("_bn", "").replace("_fc", "_fc")


def load(path):
    obj = torch.load(path, map_location="cpu")
    # Tolerate a wrapper dict around the state_dict.
    if isinstance(obj, dict) and not any(isinstance(v, torch.Tensor) for v in obj.values()):
        for k in ("state_dict", "model", "model_state_dict", "weights"):
            if k in obj:
                obj = obj[k]
                break
    return obj


def rel(num_sq, den_sq):
    if den_sq <= 0:
        return float("nan")
    return math.sqrt(num_sq) / math.sqrt(den_sq)


def compare(path_a, path_b):
    a, b = load(path_a), load(path_b)
    assert set(a) == set(b), f"key mismatch: {set(a) ^ set(b)}"

    rows = []
    g_num = collections.defaultdict(float)
    g_den = collections.defaultdict(float)
    tot = {"param": [0.0, 0.0], "buffer": [0.0, 0.0]}

    for k in a:
        if k.endswith(SKIP_SUFFIXES):
            continue
        ta, tb = a[k].double(), b[k].double()
        d = float(((tb - ta) ** 2).sum())
        n = float((ta ** 2).sum())
        kind = "buffer" if is_buffer(k) else "param"
        rows.append((k, kind, d, n, rel(d, n), ta.numel()))
        tot[kind][0] += d
        tot[kind][1] += n
        if kind == "param":
            g = group_of(k)
            g_num[g] += d
            g_den[g] += n

    print(f"\n=== {path_b.split('/')[-1]}  vs  {path_a.split('/')[-1]} ===")
    print("  per-group (learnable parameters only)")
    print(f"    {'group':<16} {'n_params':>10} {'rel L2':>10}")
    for g in g_num:
        npar = sum(r[5] for r in rows if r[1] == "param" and group_of(r[0]) == g)
        print(f"    {g:<16} {npar:>10} {rel(g_num[g], g_den[g]):>10.4f}")
    print(f"    {'ALL PARAMS':<16} {sum(r[5] for r in rows if r[1]=='param'):>10} "
          f"{rel(*tot['param']):>10.4f}")
    print(f"    {'BN buffers':<16} {sum(r[5] for r in rows if r[1]=='buffer'):>10} "
          f"{rel(*tot['buffer']):>10.4f}")
    return rows, rel(*tot["param"]), rel(*tot["buffer"])


def main():
    paths = sys.argv[1:]
    if len(paths) < 2:
        print(__doc__)
        return
    for i in range(len(paths) - 1):
        compare(paths[i], paths[i + 1])


if __name__ == "__main__":
    main()
