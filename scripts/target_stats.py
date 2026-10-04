#!/usr/bin/env python3
"""Per-decision target statistics on a Gumbel-generated corpus shard.

For every multi-child root: KL(target || prior) and argmax-change rate for several candidate
targets, how often the played move / the deterministic Gumbel pick leaves the prior's favourite,
prior sharpness, Q spread. Q is Home-centric in points (TD = 1000); prior is softmax x len.
"""
import json, math, sys
import numpy as np

C_VISIT, C_SCALE = 50.0, 0.1


def mover_q(q, mover):
    return None if q is None else (q if mover == "Home" else -q)


def completed(ch, mover):
    qs = [mover_q(c["q"], mover) for c in ch]
    vis = [(c["visits"], q) for c, q in zip(ch, qs) if q is not None and c["visits"] > 0]
    if vis:
        tot = sum(v for v, _ in vis); fill = sum(v * q for v, q in vis) / tot
    else:
        sc = [q for q in qs if q is not None]; fill = sum(sc) / len(sc) if sc else 0.0
    return np.array([fill if (q is None or c["visits"] == 0) else q for c, q in zip(ch, qs)], float)


def softmax(x):
    x = x - x.max(); w = np.exp(x); return w / w.sum()


def kl(p, q):
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))).sum())


def targets(ch, mover):
    lp = np.log(np.maximum(np.array([c["prior"] for c in ch], float), 1e-12))
    prior = softmax(lp)
    q = completed(ch, mover)
    n = np.array([c["visits"] for c in ch], float)
    out = {"prior": prior}
    for tau in (100, 50, 30, 20):
        out[f"cq{tau}"] = softmax(lp + q / tau)
    rng = q.max() - q.min()
    for floor in (1000, 300, 0):
        qh = (q - q.min()) / max(rng, floor, 1e-9)
        out[f"gum_f{floor}"] = softmax(lp + (C_VISIT + n.max()) * C_SCALE * qh)
    out["visits"] = n / n.sum() if n.sum() > 0 else prior
    return out, q, n, prior


def main(path, limit):
    rows = []
    with open(path) as f:
        for ti, line in enumerate(f):
            if ti >= limit:
                break
            t = json.loads(line)
            for s in t["samples"]:
                ch = s["children"]
                if len(ch) < 2 or s["root_solved"]:
                    continue
                mover = s["to_move"]
                tg, q, n, prior = targets(ch, mover)
                chosen = json.dumps(s["chosen_action"], sort_keys=True)
                keys = [json.dumps(c["action"], sort_keys=True) for c in ch]
                ci = keys.index(chosen) if chosen in keys else -1
                pa = int(prior.argmax())
                vis_idx = np.where(n > 0)[0]
                top2 = np.sort(q[vis_idx])[::-1][:2] if len(vis_idx) >= 2 else None
                r = {
                    "nch": len(ch), "nvis": int((n > 0).sum()), "maxn": float(n.max()),
                    "pmax": float(prior.max()), "hprior": float(-(prior[prior > 0] * np.log(prior[prior > 0])).sum()),
                    "chosen_ne_prior": int(ci != pa), "chosen_is_vis": int(ci >= 0 and n[ci] > 0),
                    "qgap_top2": float(top2[0] - top2[1]) if top2 is not None else float("nan"),
                    "qrange_vis": float(q[vis_idx].max() - q[vis_idx].min()) if len(vis_idx) else 0.0,
                    "best_q_ne_prior": int(vis_idx[np.argmax(q[vis_idx])] != pa) if len(vis_idx) else 0,
                }
                for k, p in tg.items():
                    if k == "prior":
                        continue
                    r[f"kl_{k}"] = kl(p, prior)
                    r[f"arg_{k}"] = int(p.argmax() != pa)
                    r[f"h_{k}"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                    r[f"pchosen_{k}"] = float(p[ci]) if ci >= 0 else float("nan")
                r["pchosen_prior"] = float(prior[ci]) if ci >= 0 else float("nan")
                rows.append(r)
    keys = rows[0].keys()
    A = {k: np.array([r[k] for r in rows], float) for k in keys}
    N = len(rows)
    print(f"{path}: {N} multi-child unsolved roots from {limit} trajectories")
    print(f"children: p50 {np.median(A['nch']):.0f} p90 {np.percentile(A['nch'],90):.0f}; visited: p50 {np.median(A['nvis']):.0f} p90 {np.percentile(A['nvis'],90):.0f}; maxN p50 {np.median(A['maxn']):.0f}")
    print(f"prior: pmax p50 {np.median(A['pmax']):.3f}, frac pmax>0.9 {np.mean(A['pmax']>0.9):.3f}, >0.99 {np.mean(A['pmax']>0.99):.3f}; H(prior) mean {A['hprior'].mean():.3f} p50 {np.median(A['hprior']):.3f}")
    print(f"played move != prior argmax: {A['chosen_ne_prior'].mean():.3f}   (played move was visited: {A['chosen_is_vis'].mean():.3f})")
    print(f"best-Q visited child != prior argmax: {A['best_q_ne_prior'].mean():.3f}; top-2 Q gap p50 {np.nanmedian(A['qgap_top2']):.0f} p90 {np.nanpercentile(A['qgap_top2'],90):.0f}; visited Q range p50 {np.median(A['qrange_vis']):.0f}")
    print(f"P(played move) under prior: mean {np.nanmean(A['pchosen_prior']):.3f} p50 {np.nanmedian(A['pchosen_prior']):.3f}")
    print(f"{'target':10s} {'KL mean':>8s} {'KL p50':>8s} {'KL p90':>8s} {'argmax!=prior':>14s} {'H mean':>8s} {'P(played) mean':>15s}")
    for k in ("cq100", "cq50", "cq30", "cq20", "gum_f1000", "gum_f300", "gum_f0", "visits"):
        print(f"{k:10s} {A['kl_'+k].mean():8.4f} {np.median(A['kl_'+k]):8.4f} {np.percentile(A['kl_'+k],90):8.4f} {A['arg_'+k].mean():14.3f} {A['h_'+k].mean():8.3f} {np.nanmean(A['pchosen_'+k]):15.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150)
