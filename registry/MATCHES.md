# Match registry — two bots playing each other

Every head-to-head result, newest first. Points are the **first-named** bot's (win 1, draw ½);
"± " is the paired SE. Single-net numbers (value bench, absorption, net check) are in
[NETS.md](NETS.md); the experiments these belong to are in [EXPERIMENTS.md](EXPERIMENTS.md).
**Keep this current:** add a row for every match that finishes (see the root CLAUDE.md).

Defaults unless a row says otherwise:
- **drives** = paired contested drives (plan 051): `runs/loopmix16x9g/positions/contested_{14x7,16x9}_gen04g.json`,
  each position played twice with the seats swapped and the same dice seed;
- **Gumbel f1000** = `cfgs/gumbel16_f1000.toml` (Gumbel root m=16, q floor 1000) at 1000 descents,
  mean backup; **policy-only** = `cfgs/policy_only.toml` (`gumbel_m = 1`, no noise: the prior's
  argmax) at `--mcts-iters 8`;
- SPRT 0.5:0.55 when a row says H0/H1 (H0 = not shown 0.05 better, not "worse").

## v9 rules (2026-10-06 on)

### Policy-only lineage (plan 059; `runs/exp069/po_lineage/run.sh`; botbowl-ui at 3c25109, tract on the CPU)

| A | vs B | 14x7 | 16x9 | mean | pairs/board | date |
|---|---|---|---|---|---|---|
| v9 gen07 | init (g056 gen04 v9) | 0.507 ± 0.009 | 0.528 ± 0.010 | 0.518 ± 0.007 | 600 | 2026-10-08 |
| v9 gen07 | v9 gen04 | 0.511 ± 0.008 | 0.512 ± 0.009 | 0.512 ± 0.006 | 600 | 2026-10-08 |

### exp069: one-step arms vs their parent gen06, policy-only both sides (plan 059; botbowl-ui 3c25109)

| A | vs B | 14x7 | 16x9 | mean | pairs/board |
|---|---|---|---|---|---|
| loop v9 gen07 (cq τ=100) | v9 gen06 | 0.498 ± 0.007 | 0.502 ± 0.008 | 0.500 ± 0.005 | 600 |
| exp069 cq100r (τ=100 re-run) | v9 gen06 | 0.503 ± 0.007 | 0.501 ± 0.008 | 0.502 ± 0.005 | 600 |
| **exp069 cq50** | v9 gen06 | 0.500 ± 0.007 | **0.517 ± 0.008** | **0.509 ± 0.005** | 600 |
| exp069 cq30 | v9 gen06 | 0.503 ± 0.009 | 0.489 ± 0.010 | 0.496 ± 0.007 | 600 |
| exp069 gumbel σ | v9 gen06 | 0.495 ± 0.010 | 0.509 ± 0.010 | 0.502 ± 0.007 | 600 |
| v9 gen06 | v9 gen05 (timing test) | — | 0.497 ± 0.021 | — | 100 |

### The loop's drive benchmark (Gumbel f1000 both sides, vs `bbnet_mix16x9d1k_gen04_v9`, SPRT, cap 800 drives per set)

| A | vs B | 14x7 | 16x9 | commit | date |
|---|---|---|---|---|---|
| v9 gen06 | d1k gen04 v9 | 0.483 ± 0.023 (H0, 116 pairs) | 0.505 ± 0.021 (H0, 158 pairs) | 43a45c6 | 2026-10-08 |
| v9 gen03 | d1k gen04 v9 | 0.449 ± 0.035 (H0, 68 pairs) | 0.481 ± 0.025 (H0, 101 pairs) | ae721b7 | 2026-10-07 |

## Pre-v9 rules (for reference; the nets need migrating and the results do not carry over)

| A | vs B | 14x7 | 16x9 | mean | settings | source |
|---|---|---|---|---|---|---|
| arm F search | arm A search (same policy, other value head) | 0.538 ± 0.013 | 0.528 ± 0.013 | **0.533 ± 0.009** | Gumbel f1000 both, 300 pairs/board | plan 056, exp068 |
| arm F search@1000 | arm F policy-only | 0.578 ± 0.014 | 0.586 ± 0.016 | **0.582 ± 0.011** | 300 pairs/board, seed 68000 | plan 056, exp068 |
| g_gen05 search@250 / @1000 / @4000 | g_gen05 policy-only | 0.527 / 0.522 / 0.577 | 0.523 / 0.563 / 0.603 | **0.525 / 0.542 / 0.590** | mean backup, 300 pairs/board, seed 66000 | plan 055, exp066 |
| g056 gen03 (= the v9 init's weights) | d1k gen04 | 0.516 ± 0.012 (H0, 373) | 0.508 ± 0.018 (H0, 154) | | the g056 loop's benchmark | plan 056 §7 |
| g056 gen01 | d1k gen04 | 0.506 ± 0.018 (H0, 200) | **0.583 ± 0.029 (H1, 68)** | | the g056 loop's benchmark | plan 056 §7 |

Older matches (the PUCT and early Gumbel eras, exp048-exp063) are in plans 048-054 and are not
repeated here.
