# Network registry — trained nets, how they were made, every benchmark number they have

One entry per net that matters now. Benchmarks here are numbers about **one** net (value bench,
absorption, net check, validation losses); results of two nets playing each other are in
[MATCHES.md](MATCHES.md). Corpora are in [DATA.md](DATA.md). **Keep this current:** add the
entry when a net is trained and its numbers as they arrive (see the root CLAUDE.md).

Benchmarks, defined once:
- **value bench:** RMS / bias of the value head against Monte Carlo truth on
  `runs/value_bench/v9_gen01_val.jsonl` (3023 states, MC noise removed; `scripts/value_bench.sh`).
  "paired" = the change against the net it was trained from, with its SE.
- **absorption** (`scripts/absorb_probe.py`): on its own generation's held-out shards, against
  the net that generated them: Δ log P(played), Δ P(played), Δ top-1 = played, Δ KL(target‖net),
  Δ value MSE. The KL column is against the training target in force (cq τ=100 through gen08,
  τ=50 from gen09).
- **net check** (`scripts/net_check.sh`, plan 055 §6): the net, generating, on the next
  generation's corpus (which it generated and was not trained on): search gain over its own
  policy per decision at 64 / 250 / 1000 / 4000 descents (want MONOTONE), the realised/predicted
  slope (1 = the search's Q gaps pay in full), and the value RMS / bias on that corpus.
- **val:** the trainer's restored best checkpoint, `val_policy + val_value` on the window's
  held-out shards. Not comparable across gen03 → gen04 (forced samples left the val set) or
  across policy targets (the target entropy differs).

## The v9 loop, `runs/loopmix16x9v9` (plan 058), `models/az_v7/bbnet_mix16x9v9_genNN.{pt,onnx}`

Recipe, all generations (`scripts/launch_plan058.sh`): warm start from the previous generation's
`.pt` at lr 5e-5, 3 epochs, `--freeze-bn --eval-at 250,500`, `--select-on combined
--eval-every 1000`, `--value-weight 0.25 --per-drive-value-weight`; `prepare --policy-target cq
--tau 100` (**τ=50 for gen09-11**, back to 100 from gen12), `--value-blend 1.0` on the MC-labelled shards (8 policy-only
playouts per sample); window = the last 3 generations' train shards (0-3, 5, 6), val = their
shards 4, 7; samples with fewer than two children dropped from gen04's training on.

| net | trained from | data window | commit (loop) | target | best checkpoint (val) | absorption: Δ logP(played) / P(played) / top-1 / KL / valMSE | value bench RMS, bias (paired vs parent) | net check (as generator of the next gen): gain @64/250/1000/4000; slope; corpus value RMS, bias |
|---|---|---|---|---|---|---|---|---|
| **init** `bbnet_mix16x9g056_gen04_v9` | g056 gen04 (= g056 gen03's weights) migrated v7→v9 with `bbnn.migrate` | — | — | — | — | — | 0.209 (as gen01's generator) | on gen01: +0.002/+0.004/+0.008/+0.007 MONOTONE; 0.72; 0.211, −0.012 |
| gen01 | init | gen01 | ae721b7 | cq 100 | step 10965, ep 3 (0.7660) | +0.0064 / +0.0029 / +0.0089 / −0.0057 / −0.0555 | 0.218, −0.017 (+4.3%, worse) | on gen02: +0.001/+0.001/+0.003/+0.005; 0.66; 0.218, −0.001 |
| gen02 | gen01 | gen01-02 | ae721b7 | cq 100 | step 21636, ep 3 (0.7669) | +0.0068 / +0.0041 / +0.0025 / −0.0042 / −0.0111 | 0.212, −0.005 (−2.9%) | on gen03: +0.003/+0.003/+0.006/+0.010; 0.76; 0.221, +0.009 |
| gen03 | gen02 | gen01-03 | ae721b7 | cq 100 | step 32000, ep 2 (0.7598) | +0.0025 / +0.0009 / −0.0004 / −0.0007 / −0.0059 | 0.214, −0.015 (+0.9%, flat) | on gen04: +0.002/+0.004/+0.007/+0.009; 0.74; 0.223, −0.022 |
| gen04 | gen03 | gen02-04 | 407b335 | cq 100, forced samples dropped | step 25000, ep 2 (0.9240) | +0.0051 / +0.0040 / +0.0017 / −0.0020 / −0.0053 | 0.209, −0.006 (−2.3%) | on gen05: +0.002/+0.003/+0.003/+0.007; 0.43; 0.218, −0.008 |
| gen05 | gen04 | gen03-05 | 407b335 | cq 100 | step 25000, ep 2 (0.9066) | +0.0022 / +0.0013 / −0.0003 / +0.0003 / −0.0048 | 0.206, +0.004 (−1.2%) | on gen06: +0.002/+0.004/+0.004/+0.010; 0.75; 0.223, +0.014 |
| gen06 | gen05 | gen04-06 | 43a45c6 | cq 100 | step 18000, ep 1 (0.8995) | +0.0042 / +0.0021 / +0.0031 / −0.0000 / −0.0052 | 0.208, +0.016 (+0.7%, flat) | on gen07: +0.001/+0.003/+0.010/+0.017; 0.64; 0.218, +0.044 |
| gen07 | gen06 | gen05-07 | 43a45c6 | cq 100 | step 23000, ep 2 (0.8907) | +0.0032 / +0.0024 / +0.0016 / +0.0011 / +0.0029 | 0.205, +0.001 (−1.4%) | on gen08: +0.001/+0.004/+0.004/+0.012; **0.94**; 0.196, −0.002 |
| gen08 | gen07 | gen06-08 (gen08 partial) | 3c25109 | cq 100 | step 25000, ep 2 (0.8881) | −0.0018 / −0.0019 / −0.0023 / +0.0023 / −0.0045 (probe set ~40% size) | **0.204**, +0.002 (−0.6%) | on gen09: +0.001/+0.002/+0.006/+0.007; 0.59; 0.201, +0.007 |
| gen09 | gen08 | gen07-09 | f75eda5 | **cq 50** (first) | step 22000, ep 2 (0.8965; τ=50 target, not comparable) | **+0.0118** / +0.0007 / −0.0002 / −0.0032 (vs the τ=50 target) / +0.0004 | 0.204, +0.004 (+0.2%, flat) | on gen10: +0.001/+0.003/+0.004/+0.006; 0.58; 0.229, +0.019 |
| gen10 | gen09 | gen08-10 | f75eda5 | cq 50 | step 27000, ep 2 (0.8943) | +0.0048 / −0.0002 / +0.0010 / −0.0009 / −0.0012 | **0.203**, +0.003 (−0.7%) | on gen11: +0.001/+0.003/+0.004/+0.005; 0.67; **0.184**, −0.002 |
| gen11 | gen10 | gen09-11 | f75eda5 | cq 50 | step 32000, ep 2 (0.8938) | +0.0024 / −0.0006 / −0.0006 / −0.0001 / −0.0039 | 0.205, +0.012 (**+1.1%, worse**) | on gen12: +0.001/+0.002/+0.002/+0.004 (weakest yet); 0.76; 0.195, +0.012 |
| gen12 | gen11 | gen10-12 | 71827c9 | **cq 100** (back) | step 24000, ep 2 (0.8825) | −0.0053 / +0.0020 / −0.0013 / +0.0021 / +0.0012 (parent trained at τ=50) | **0.201**, −0.012 (**−1.9%**, best of the run) | on gen13: +0.001/+0.004/**+0.008**/**+0.012** MONOTONE; **1.08** (the run's best); 0.190, −0.012 |
| gen13 | gen12 | gen11-13 | 71827c9 | cq 100 | step 27000, ep 2 (0.8749) | +0.0005 / +0.0008 / +0.0002 / +0.0018 / −0.0020 | **0.199**, +0.008 (−0.9%, best of the run) | on gen14: +0.001/+0.001/+0.006/**+0.016** MONOTONE (largest @4000 of the run); 0.45; 0.198, −0.011 |
| gen14 | gen13 | gen12-14 | 71827c9 | cq 100 | step 26000, ep 2 (0.8722) | +0.0009 / +0.0013 / +0.0001 / +0.0018 / −0.0007 | **0.199**, −0.009 (−0.1%, flat) | on gen15: +0.001/+0.003/+0.004/+0.006 MONOTONE; 0.74; 0.177, −0.023 |
| gen15 | gen14 | gen13-15 | 71827c9 (relaunched on e423b43, same game code) | cq 100 | step 14000, ep 1 (0.8603) | −0.0020 / +0.0005 / +0.0002 / +0.0018 / +0.0002 | **0.197**, +0.007 (−0.9%, best of the run) | on gen16: +0.002/+0.002/+0.003/+0.003 MONOTONE but flat (the weakest of the run); **0.25**, overrides pay from no Q gap; 0.208, +0.002 |
| gen16 | gen15 | gen14-16 | 188c6d4 (relaunch; chance toggles off, same games) | cq 100 | step 33453, ep 3, the last step (0.8549) | −0.0007 / +0.0014 / −0.0009 / +0.0017 / +0.0001 | 0.198, −0.001 (+0.3%, flat) | (on gen17: pending) |

Match results for these nets (drives vs the anchor at gen03 and gen06; policy-only lineage) are in
[MATCHES.md](MATCHES.md).

## Experiment arms (not in the loop)

| net | made by | recipe | numbers |
|---|---|---|---|
| `runs/exp069/{cq100r,cq50,cq30,gumbel}/net.{pt,onnx}` | exp069 (plan 059), 2026-10-08; script `scripts/exp069_targets_policy_only.sh` at 81b5842, `prepare` binary 43a45c6 | gen06 → one step on the gen05-07 window, the loop's recipe with only the policy target changed: cq τ 100 (re-run) / 50 / 30, Gumbel σ (c_visit 50, c_scale 0.1, min_range 50) | in [EXPERIMENTS.md](EXPERIMENTS.md#exp069) (absorption, value bench) and [MATCHES.md](MATCHES.md) (policy-only vs gen06) |

## References still in use

| net | what | origin |
|---|---|---|
| `models/az_v7/bbnet_mix16x9d1k_gen04_v9` | the **drive benchmark reference** (`DRIVE_REF`) of the v9 loop | d1k loop gen04 (1000 descents, PUCT era, plan 051), migrated v7→v9 2026-10-06; never trained on the v9 rules |
| `models/az_v7/anchor_mix16x9_gen13_v9` | `ANCHOR` (unused: `ANCHOR_EVERY=0`, drives only) | mix16x9 gen13, migrated |
| `models/az_v7/plan056_armF{,_v9}` | plan 056 arm F (MC labels), the g056 loop's init | `runs/exp067/arms/F`; value bench (pre-v9 set) RMS 0.236, bias −0.001 |
| `models/az_v7/bbnet_mix16x9g056_gen0{1..4}` | the g056 loop (plan 056 §7), pre-v9 | gen02 = gen01's weights, gen04 = gen03's (restored their init); numbers in plan 056 §7 |

Nets before these (az14x7v6, mix16x9, vl0, d1k, the Gumbel loop) are documented in their plans
and are not usable on the v9 rules without `bbnn.migrate`.
