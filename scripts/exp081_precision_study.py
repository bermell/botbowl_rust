"""bf16/fp16 inference numerics vs fp32 for a loop net (plan 069): gen48 on gen48's held-out shard 4."""
import sys, math
import torch, torch.nn.functional as F
from bbnn.data import open_prepared, make_loader
from bbnn.model import BBNet, masked_policy_logits
torch.manual_seed(0); torch.set_num_threads(3)
ck = torch.load(sys.argv[1], map_location="cpu"); sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
net = BBNet.from_state_dict(sd).eval()
ds = open_prepared(sys.argv[2]); loader = make_loader(ds, 256, shuffle=False)
def run(batch, mode):
    with torch.no_grad():
        if mode == "fp32":
            p, v = net(batch["spatial"], batch["global"])
        else:
            dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]
            with torch.autocast("cpu", dtype=dt):
                p, v = net(batch["spatial"], batch["global"])
            p, v = p.float(), v.float()
        lg = masked_policy_logits(p, batch["actions"], batch["pad_mask"])
        return F.log_softmax(lg, 1), v.squeeze(1)
modes = ["bf16", "fp16"]
acc = {m: dict(n=0, dv_abs=0.0, dv_max=0.0, kl=0.0, top1=0, se_ref=0.0, se_m=0.0, flips_close=0, close=0) for m in modes}
for batch in loader:
    lp0, v0 = run(batch, "fp32"); z = batch["value"].squeeze(-1) if batch["value"].dim() > 1 else batch["value"]
    p0 = lp0.exp(); a0 = lp0.argmax(1)
    top2 = lp0.topk(2, 1).values; close = (top2[:, 0] - top2[:, 1]) < 0.1  # log-prob margin < 0.1
    for m in modes:
        lp, v = run(batch, m); s = acc[m]; n = len(v)
        s["n"] += n; dv = (v - v0).abs(); s["dv_abs"] += float(dv.sum()); s["dv_max"] = max(s["dv_max"], float(dv.max()))
        s["kl"] += float((p0 * (lp0 - lp)).where(batch["pad_mask"] > 0, torch.zeros(())).sum())
        same = (lp.argmax(1) == a0); s["top1"] += int(same.sum())
        s["close"] += int(close.sum()); s["flips_close"] += int((~same & close).sum())
        s["se_ref"] += float(((v0 - z) ** 2).sum()); s["se_m"] += float(((v - z) ** 2).sum())
for m, s in acc.items():
    n = s["n"]
    print(f"{m}: n {n}  value |dv| mean {s['dv_abs']/n:.5f} max {s['dv_max']:.4f}  "
          f"MSE vs label fp32 {s['se_ref']/n:.5f} -> {m} {s['se_m']/n:.5f}  "
          f"policy KL(fp32||{m}) {s['kl']/n:.6f}  top-1 agree {s['top1']/n:.4f}  "
          f"top-1 flips {n - s['top1']} ({s['flips_close']} of them where the fp32 top-2 log-margin < 0.1, {s['close']} such)")
