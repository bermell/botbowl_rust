"""How do the policy and value losses split the gradient reaching the shared trunk?

gen12 (the net gen13 trains from) on gen13's held-out shard 4, prepared at cq tau 100 / 50 / 20.
Loss exactly as the loop trains: policy CE mean over samples + 0.25 * per-drive-weighted value MSE,
BN in eval mode (--freeze-bn).
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from bbnn.data import open_prepared, make_loader
from bbnn.model import BBNet, masked_policy_logits

torch.manual_seed(0)
torch.set_num_threads(6)
ckpt = torch.load(sys.argv[1], map_location="cpu")
sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
VW = 0.25
BUCKETS = [(2, 2), (3, 5), (6, 20), (21, 100), (101, 10**9)]


def bucket(k):
    for lo, hi in BUCKETS:
        if lo <= k <= hi:
            return f"{lo}-{hi if hi < 10**9 else ''}"


for d in sys.argv[2:]:
    model = BBNet.from_state_dict(sd)
    model.eval()
    trunk = [p for n, p in model.named_parameters() if n.startswith(("stem", "blocks", "global_fc"))]
    ds = open_prepared(d)
    loader = make_loader(ds, 256, shuffle=False)
    per = defaultdict(lambda: defaultdict(float))
    gp2 = gv2 = dot = 0.0
    nb = 0
    tot = defaultdict(float)
    for batch in loader:
        policy_out, value_out = model(batch["spatial"], batch["global"])
        logits = masked_policy_logits(policy_out, batch["actions"], batch["pad_mask"])
        logsm = F.log_softmax(logits, dim=1)
        tgt = batch["policy"]
        ce = -(tgt * logsm).sum(1)
        ent = -(tgt * torch.log(tgt.clamp_min(1e-12))).sum(1)
        p = logsm.exp()
        gnorm = ((p - tgt) * batch["pad_mask"]).norm(dim=1)  # |dCE/dlogits| per sample
        w = batch["weight"]
        se = (value_out - batch["value"]) ** 2
        policy_loss = ce.mean()
        value_loss = (w * se).sum() / w.sum()
        # per-sample value gradient wrt the pre-tanh output, as weighted in the loss (x N to compare
        # with the policy's per-sample 1/N share): 2 (v - z) (1 - v^2) w N / sum w
        n = len(w)
        vg = (2 * (value_out - batch["value"]).abs() * (1 - value_out**2) * w * n / w.sum()).squeeze(1) * VW
        k = batch["pad_mask"].sum(1)
        for i in range(n):
            b = bucket(int(k[i]))
            s = per[b]
            s["n"] += 1
            s["kl"] += float(ce[i] - ent[i])
            s["ent"] += float(ent[i])
            s["pg"] += float(gnorm[i])
            s["vg"] += float(vg[i])
            s["top1_tgt"] += float(tgt[i].max())
        gp = torch.autograd.grad(policy_loss, trunk, retain_graph=True)
        gv = torch.autograd.grad(VW * value_loss, trunk)
        gp_f = torch.cat([g.flatten() for g in gp])
        gv_f = torch.cat([g.flatten() for g in gv])
        gp2 += float(gp_f.norm()) ; gv2 += float(gv_f.norm())
        dot += float(F.cosine_similarity(gp_f, gv_f, dim=0))
        tot["pl"] += float(policy_loss); tot["kl"] += float((ce - ent).mean()); tot["vl"] += float(value_loss)
        nb += 1
    print(f"== {d.split('/')[-1]}  ({nb} batches of 256)")
    print(f"policy CE {tot['pl']/nb:.4f} (KL part {tot['kl']/nb:.4f})  value MSE {tot['vl']/nb:.4f} (x{VW} = {VW*tot['vl']/nb:.4f})")
    print(f"trunk grad norm per batch: policy {gp2/nb:.4f}  {VW}*value {gv2/nb:.4f}  ratio p/v {gp2/gv2:.2f}  cosine {dot/nb:+.3f}")
    print(f"{'legal':>8} {'share':>6} {'tgt H':>6} {'tgt max':>7} {'KL':>7} {'|p-pi|':>7} {'value g':>7}")
    tn = sum(s["n"] for s in per.values())
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi if hi < 10**9 else ''}"
        s = per.get(b)
        if not s:
            continue
        c = s["n"]
        print(f"{b:>8} {c/tn:6.1%} {s['ent']/c:6.3f} {s['top1_tgt']/c:7.3f} {s['kl']/c:7.4f} {s['pg']/c:7.4f} {s['vg']/c:7.4f}")
