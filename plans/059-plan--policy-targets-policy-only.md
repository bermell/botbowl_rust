# Plan 059 — policy targets again, judged by policy-only drives

**Status:** exp069 done 2026-10-08 12:30 (§4). No target clearly beats τ=100 after one step; τ=50 is the only candidate (value intact, log P(played) 5x, policy-only +0.009 ± 0.005). One-step policy-only drives are too blunt to settle it (§5).

## 1. Why

The v9 loop learns its policy slowly: absorption dP(played) is +0.001 to +0.004 per generation and
dtop1 is about 0, while the value side moves every generation (dvalMSE −0.005, the value bench).
The loop trains on cq τ=100. That verdict is from before the mean backup (plan 055, `ce4eda1`):
- exp059 (PUCT, 2026-10-03): no target beat τ=100; Gumbel σ was below it;
- plan 054 / exp063 (Gumbel, 2026-10-05): τ=50 at lr 5e-5 led the absorption probe but drew 0.510
  in drives vs τ=100's 0.531; Gumbel σ had the biggest log P(played) gain (+0.025) but cost 0.03 of
  mean P(played).
Both ran while the search did not beat its own policy (E5: policy-only 0.517 vs Gumbel@1000) and
Q was biased upward by the max backup. A soft target was then the safe copy of a search that had
little to teach. Now the search beats its policy (0.582 at 1000 descents) and the per-generation net
check finds overrides that pay, so a sharper target may hand more of it to the net.

**The user (2026-10-08): no anchor benchmark for this; the quick numbers, and policy-only matches as
the fast play signal.**

## 2. The targets

- **cq τ** (the loop's): `softmax(ln prior + Q/τ)`, Q in points (1 TD = 1000). At τ=100 a move
  100 points (0.1 TD) better gets e times the probability. One temperature for every root.
- **Gumbel σ** (Gumbel MuZero, Danihelka et al. 2022; the target that belongs to our root search):
  `softmax(ln prior + σ(q̂))`, q̂ the completed Q min-max normalised over this root's children,
  σ = (c_visit + max visits) · c_scale · q̂ (c_visit 50, c_scale 0.1). Its sharpness follows each
  root's own spread and the visits spent; `min_range` (Q points) floors the spread so a near-tie
  is not read as decisive.

## 3. exp069

One generation step replayed: parent gen06, the window the loop's gen07 trained on (gen05-07,
MC-labelled), the loop's recipe (lr 5e-5, `--freeze-bn`, 3 epochs, restore on combined val).

| arm | target |
|---|---|
| loop07 | cq τ=100 (the loop's own gen07) |
| cq100r | cq τ=100 re-run: the training noise of every column |
| cq50 | cq τ=50 |
| cq30 | cq τ=30 |
| gumbel | Gumbel σ, c_visit 50, c_scale 0.1, min_range 50 |

Per arm, against the parent:
- **absorption probe** on gen07's held-out shards, prepared once with cq τ=100 so the columns
  compare; P(played), log P(played) and top-1 = played do not depend on the target;
- **value bench** (paired);
- **policy-only paired drives** (`cfgs/policy_only.toml` on both sides, the prior's argmax; tract on
  the CPU, one forward per decision): 600 pairs per board on the contested 14x7 and 16x9 sets.
  Timing: 200 drives in 93 s on 4 threads. The first reading, gen06 vs gen05 policy-only on 16x9:
  0.497 ± 0.021 (100 pairs).

Policy-only drives see only the policy. The value bench covers the value head and the net check
(not run here) whether a sharper policy still serves the search.

## 4. Results (exp069, 2026-10-08; parent gen06, the gen05-07 window)

Training on the free GPU: 32 min per arm (the first, cq100r, took 101 min sharing it with gen08's
generation). Best-val checkpoints: cq100r step 25000, cq50 24000, cq30 31188 (the last), gumbel
31000.

| arm | Δ log P(played) | Δ P(played) | Δ top-1 = played | value bench dRMS (paired) | value bias | policy-only 14x7 | policy-only 16x9 | policy-only mean |
|---|---|---|---|---|---|---|---|---|
| loop07 (τ=100) | +0.003 | +0.002 | +0.002 | −1.4% | +0.001 | 0.498 ± 0.007 | 0.502 ± 0.008 | 0.500 ± 0.005 |
| cq100r (τ=100 re-run) | +0.002 | +0.001 | +0.001 | −1.7% | −0.002 | 0.503 ± 0.007 | 0.501 ± 0.008 | 0.502 ± 0.005 |
| **cq50** | **+0.016** | −0.001 | +0.002 | **−1.4%** | −0.007 | 0.500 ± 0.007 | **0.517 ± 0.008** | **0.509 ± 0.005** |
| cq30 | +0.028 | −0.010 | +0.001 | ±0.0% | +0.012 | 0.503 ± 0.009 | 0.489 ± 0.010 | 0.496 ± 0.007 |
| gumbel σ | −0.050 | −0.090 | −0.012 | +1.7% (worse) | +0.013 | 0.495 ± 0.010 | 0.509 ± 0.010 | 0.502 ± 0.007 |

(Absorption on gen07's held-out shards, every arm scored on the same cq-100 prepared set; the
gen06 parent's raw numbers: log P(played) −0.946, P(played) 0.614, top-1 0.680. Value bench: the
v9 MC benchmark, paired vs gen06 (RMS 0.2078). Policy-only: 600 pairs per board vs gen06,
`cfgs/policy_only.toml` both sides, contested positions.)

- **The τ=100 control reproduces** (loop07 vs cq100r): ±0.001 on the probe columns, 0.3% on the
  value bench, 0.002 in policy-only play. The quick numbers carry little training noise.
- **Sharper τ trades mean P(played) for log P(played)**, as in plan 054: the net stops giving the
  played move near-zero probability (log up 5-10x) but moves mass off the argmax (τ=30: −0.010).
- **The shared trunk pays for it:** τ=30 loses the whole value-bench gain (0.0% vs −1.4%), and
  Gumbel σ makes the value head worse than its parent (+1.7%). τ=50 keeps it (−1.4%).
- **Gumbel σ (c_scale 0.1, min_range 50) is too sharp** for this data: a near one-hot target,
  every probe column down, KL to the cq-100 target 0.24 (others 0.08).
- **Policy-only play:** the loop's own step moves the bare policy by 0.000. Only τ=50 is above
  (16x9 0.517, +2 SE; mean 0.509, +1.7 SE). With five arms and two boards one 2-SE reading is
  expected by chance, so this is a lead, not a result.

## 5. What one-step policy-only drives can and cannot see

A one-generation fine-tune changes the argmax in ~0.1-0.2% of decisions (Δ top-1). Policy-only
play is the argmax, so two adjacent nets play the same move almost everywhere and their drives
can only differ where the argmax moved. A real but small policy improvement is therefore nearly
invisible after one step at 600 pairs. Remedies: compare across several generations (the
lineage check below), or score policy-only play at a sampled policy (temperature 1), which
exposes the whole distribution, at the cost of noise.

**Lineage check:** does the loop's policy improve over several generations at all? Policy-only,
600 pairs per board (`runs/exp069/po_lineage/`):

| gen07 vs | 14x7 | 16x9 | mean | per generation |
|---|---|---|---|---|
| gen04 (3 generations back) | 0.511 ± 0.008 | 0.512 ± 0.009 | 0.512 ± 0.006 | ~+0.004 |
| the init net, g056 gen04 v9 (7 back) | 0.507 ± 0.009 | 0.528 ± 0.010 | 0.518 ± 0.007 | ~+0.0025 |

So the loop's bare policy does improve under τ=100, but at ~0.003-0.004 per generation: too
little to see in one step, and the first generations on the new rules (gen01-03) apparently gave
some of it back. τ=50's one-step reading (+0.009 ± 0.005) would be two to three times that rate,
if it holds.
