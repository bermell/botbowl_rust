#!/usr/bin/env python3
"""Did training absorb the search's improvement? Score several nets on one prepared val dir.

Per net: CE vs the prepared policy target, KL(target||pred) = CE - H(target), mean log P(played
move), top-1 == played, value MSE (per-drive weighted). The played move under Gumbel generation
(gumbel_scale=1) is a sample from the search's improved policy, so a net trained on this data
should give it more probability than the net that generated it did.

    scripts/absorb_probe.py --val DIR NAME=net.pt [NAME=net.pt ...]
"""
import argparse, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "src"))
from bbnn.data import open_prepared, make_loader  # noqa: E402
from bbnn.model import BBNet, masked_policy_logits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--summary", action="store_true",
                    help="also print one line: every later arm minus the first (the reference)")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    ds = open_prepared(a.val, augment=False)
    loader = make_loader(ds, a.batch, shuffle=False)
    print(f"val: {len(ds)} samples")
    rows = []
    print(f"{'net':28s} {'CE':>7s} {'H(tgt)':>7s} {'KL':>7s} {'logP(played)':>13s} {'P(played)':>10s} {'top1=played':>12s} {'valMSE':>7s} {'top1=tgt':>9s}")
    for arm in a.arms:
        name, path = arm.split("=", 1)
        sd = torch.load(path, map_location="cpu")
        net = BBNet.from_state_dict(sd).to(a.device).eval()
        tot = dict(ce=0.0, h=0.0, lp=0.0, p=0.0, acc=0.0, acc_t=0.0, vse=0.0, w=0.0)
        n = 0
        with torch.no_grad():
            for b in loader:
                pol, val = net(b["spatial"].to(a.device), b["global"].to(a.device))
                logits = masked_policy_logits(pol, b["actions"].to(a.device), b["pad_mask"].to(a.device))
                logsm = F.log_softmax(logits, dim=1)
                tgt = b["policy"].to(a.device)
                ce = -(tgt * logsm).sum(dim=1)
                h = -(tgt * torch.log(tgt.clamp_min(1e-30))).sum(dim=1)
                chosen = b["chosen"].to(a.device)
                lp = logsm.gather(1, chosen[:, None]).squeeze(1)
                w = b["weight"].to(a.device)
                vt = b["value"].to(a.device)
                k = ce.shape[0]
                tot["ce"] += ce.sum().item(); tot["h"] += h.sum().item(); tot["lp"] += lp.sum().item()
                tot["p"] += lp.exp().sum().item()
                tot["acc"] += (logits.argmax(1) == chosen).float().sum().item()
                tot["acc_t"] += (logits.argmax(1) == tgt.argmax(1)).float().sum().item()
                tot["vse"] += (w * (val - vt) ** 2).sum().item(); tot["w"] += w.sum().item()
                n += k
        rows.append((name, (tot["ce"] - tot["h"]) / n, tot["lp"] / n, tot["p"] / n, tot["acc"] / n,
                     tot["vse"] / tot["w"]))
        print(f"{name:28s} {tot['ce']/n:7.4f} {tot['h']/n:7.4f} {(tot['ce']-tot['h'])/n:7.4f} {tot['lp']/n:13.4f} {tot['p']/n:10.4f} {tot['acc']/n:12.4f} {tot['vse']/tot['w']:7.4f} {tot['acc_t']/n:9.4f}")
    if a.summary and len(rows) > 1:
        r0 = rows[0]
        for r in rows[1:]:
            print(f"ABSORB {r[0]} vs {r0[0]}: dlogP(played) {r[2] - r0[2]:+.4f}  dP(played) {r[3] - r0[3]:+.4f}  "
                  f"dtop1 {r[4] - r0[4]:+.4f}  dKL(target||net) {r[1] - r0[1]:+.4f}  dvalMSE {r[5] - r0[5]:+.4f}")


if __name__ == "__main__":
    main()
