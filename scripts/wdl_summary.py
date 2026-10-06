#!/usr/bin/env python3
"""Plan 050 step 1: score value heads on held-out drives, all on the scalar the search reads.

    wdl_summary.py --val runs/exp059/wdl/prepared_val ARM=PT [ARM=PT ...]

`--val` must be prepared with `--value-blend 1.0`, so the label is the raw drive outcome
(-1 / 0 / +1, mover frame) and not half the generator's own value head. Every net is scored by
the per-drive-weighted MSE of its exported value (`P(self) - P(opp)` for a WDL head) against
that label, pooled and per board. Arms named `X_sN` are grouped by `X` (mean ± sd over seeds).
For WDL nets it also prints the plan's sanity checks: the argmax confusion matrix and the
reliability of `P(nobody scores)`.
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "src"))
from bbnn.data import MultiDimsDataset, collate, make_loader, open_prepared  # noqa: E402
from bbnn.model import BBNet  # noqa: E402
from bbnn.train import evaluate  # noqa: E402

CLASSES = ("self", "none", "opp")


def wdl_checks(net, loader, device):
    """Confusion matrix (rows: outcome, cols: argmax) and P(none) reliability in 5 bins."""
    conf = torch.zeros(3, 3)
    bins = [[0.0, 0.0, 0] for _ in range(5)]  # sum predicted, sum empirical, count
    net.eval()
    with torch.no_grad():
        for b in loader:
            _, logits = net(b["spatial"].to(device), b["global"].to(device), wdl=True)
            p = torch.softmax(logits, dim=1).cpu()
            truth = (1 - b["value"].view(-1).round()).long().clamp(0, 2)  # +1 -> self(0), 0 -> none(1), -1 -> opp(2)
            for t, a in zip(truth, p.argmax(dim=1)):
                conf[t, a] += 1
            for pn, t in zip(p[:, 1], truth):
                i = min(int(pn * 5), 4)
                bins[i][0] += float(pn)
                bins[i][1] += float(t == 1)
                bins[i][2] += 1
    rows = ["      outcome\\argmax " + " ".join(f"{c:>7}" for c in CLASSES)]
    for i, c in enumerate(CLASSES):
        rows.append(f"      {c:>19} " + " ".join(f"{int(x):7d}" for x in conf[i]))
    rel = " · ".join(f"{s / n:.2f}->{e / n:.2f} (n={n})" for s, e, n in bins if n)
    rows.append(f"      P(none) predicted->observed by bin: {rel}")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("arms", nargs="+", help="NAME=path/to/net.pt")
    a = p.parse_args()

    ds = open_prepared(a.val, augment=False)
    loader = make_loader(ds, 256, shuffle=False)
    groups = {}
    if isinstance(ds, MultiDimsDataset):
        groups = {n: DataLoader(g, batch_size=256, shuffle=False, collate_fn=collate) for n, g in zip(ds.names, ds.groups)}

    pooled = defaultdict(list)
    by_board = defaultdict(lambda: defaultdict(list))
    for spec in a.arms:
        name, path = spec.split("=", 1)
        net = BBNet.from_state_dict(torch.load(path, map_location=a.device)).to(a.device)
        _, mse, _ = evaluate(net, loader, a.device, per_drive_value_weight=True)
        arm = re.sub(r"_s\d+$", "", name)
        pooled[arm].append(mse)
        for g, gl in groups.items():
            by_board[arm][g].append(evaluate(net, gl, a.device, per_drive_value_weight=True)[1])
        print(f"{name:12} {net.value_head:6} val MSE vs drive outcome {mse:.5f}")
        if net.value_head == "wdl":
            print(wdl_checks(net, loader, a.device))

    print("\nper arm (mean ± sd over seeds), per-drive-weighted MSE of the exported value vs the outcome:")
    for arm, xs in pooled.items():
        sd = statistics.stdev(xs) if len(xs) > 1 else float("nan")
        print(f"  {arm:8} {statistics.mean(xs):.5f} ± {sd:.5f}  (n={len(xs)})")
    if groups:
        print("\nper board (mean over seeds):")
        names = list(groups)
        print("  " + " " * 8 + " ".join(f"{n.removeprefix('dims_'):>8}" for n in names))
        for arm in pooled:
            print(f"  {arm:8}" + " ".join(f"{statistics.mean(by_board[arm][n]):8.4f}" for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
