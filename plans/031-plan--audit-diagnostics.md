# Audit diagnostics: cheap checks before spending games

**Status:** **Run 2026-09-07.** All ten items executed; results are recorded under each one and the
re-ranking they imply is folded into plan 032. Written 2026-09-07 from the methodology audit. Every
item here was under ~1 h of machine time, most ran offline on the shards already in
`runs/loop14x7/`, and each one either killed or promoted a specific entry in plan 032.

## What the audit changed

Three of the four hypotheses these diagnostics were written to confirm turned out to be **wrong**,
and the two largest findings were not on the list at all.

| item | expectation going in | what was measured | verdict |
|---|---|---|---|
| D1 root-Q calibration | minimax-over-noise optimism, flattening with fan width | optimism **+0.133-0.140**, but the fan-width signature is absent. A follow-up settles it: the bare leaf is at **+0.031** and **the search adds +0.101** (z=52) | **confirmed by other means** |
| D2 tie rate | ~85% tied roots (from 8x3 pure-td) | **13.5%** in the nn half; the cross-generation test is **vacuous** (one frozen generator net). Real defect found elsewhere: top-1 label agreement **0.588** | **wrong** |
| D3 encoding ambiguity | missing action-type plane | 12.78% ambiguous, but **zero** of them are action-type; it is the procedure stack, and the measured cost is 0 | **wrong remedy** |
| D4 prior domination | search follows the NN prior | real but partial: prior lift **1.98 → 3.11** in production, **1.08 → 5.54** on matched states; not the collapse the Read described | **partly** |
| D5 weight movement | 2e-4 barely moves the net | it moves it **1.4-1.7× further** than the last fine-tune that *did* win a rung | **wrong, and backwards** |
| D6 corpus composition | ~33% hedge by samples | 33% confirmed — but **29.1% of train vs 47.5% of val**; and the generator net has been **frozen at gen03 since gen04** | **confirmed + two surprises** |
| D7 y-flip augmentation cost | "expected small" — maybe drop it from the policy loss | **`--no-augment` is worse** (+0.0092 best `val_policy`, +0.0995 by step 110k). It is regularising, not correcting | **wrong** |
| D8 mid-procedure scoring | "below 0.1% is close" | exactly **0** of 804,757 scored leaves, and unreachable by the engine's own contract | **confirmed, and stronger** |
| D9 NN mirror equivariance | plan 027's 56% Away share might be a bug | **green** at every budget including production 1000, fixture and champion | **not a bug** |
| D10 pair correlation | ~1.10× SE inflation | **1.003**, CI [0.962, 1.060] — there is none | **wrong** |

The two unplanned findings, both from D8's counters (18 games, 1,436,287 `score_leaf` calls):

1. **Case 4 is unreachable, not merely rare.** `score_leaf`'s mid-procedure branch, documented in
   its own comment as a known compromise, is taken **0 times in 804,757 scored leaves** — and the
   engine's own contract explains why: `step_with_roll_or_action` loops until `NeedAction` /
   `NeedRoll` / `GameOver`, so a state between procedure steps is never handed back. The compromise
   costs nothing and the "principled fix" it proposes has nothing to fix.
2. **The search almost never sees a touchdown.** The exact-outcome carve-out fires on 3.4% of
   scored leaves, but **27,284 of 27,385 are `game_over`** and only **101** are the
   `chance_past_horizon` branch the code comments call "exactly where every in-search touchdown
   lands" — 0.013% of scored leaves. So effectively 100% of what the search backs up is NN estimate,
   with almost no exact anchor to pull a maximum back toward the truth. That is the context for
   D1's +0.10.

> **Correction (2026-09-07).** Two earlier drafts of this section quoted this run at different
> points — first "the carve-out never fires", then "6.5%" — both read off partial runs. The
> `exact_outcome` rate is extremely lumpy (a random start late in half 2 reaches `game_over` inside
> a 1-turn horizon; a mid-drive start never does), so it needs a run sized for it. **3.4% is an
> order of magnitude, not an estimate.** The case-4 zero is unaffected: it is a structural result,
> not a rate.

Common inputs: `runs/loop14x7/gen0{5,6,7}/shard*.jsonl` (every `Sample` carries `state`, `children[]`
with `visits/q/prior/solved`, `root_value`, `root_visits`, `outcome_value`), the prepared dirs, the
`.pt` files in `models/`, and `runs/loop14x7/status.md`. One `scripts/audit_corpus_stats.py` can
serve D1, D2, D6 and D10 — stream `DatasetReader`-compatible JSONL, one pass, print tables.

## D1 — Is the search's root value calibrated against the drive outcome? (minimax-over-noise)

**Why.** `backprop_scores` (`botbowl-mcts/src/dynamics.rs`, player branch) takes the max/min over
child Q with no averaging, and each leaf is one NN forward. A max over 30-100 noisy estimates is
biased toward the noise. FPU = parent Q then hands that maximum to every unexplored sibling.

**How.** Per generation, mover-sign `root_value/1000` and `outcome_value`; bin the former into
[-1,-0.8), ..., [0.8,1]; report mean outcome, count and Brier per bin. Stratify by
`len(children)` (≤10, 11-30, 31-60, >60) and by turn.

**Read.** Calibrated: bin means sit on the diagonal. Minimax bias: the curve is flatter than the
diagonal and flattens further as fan width grows. Also report the mean signed gap
`E[root_value/1000 − outcome_value]` for the mover — positive = optimism.

**Decides.** plan 032 #2 (mean backup). If the calibration curve is on the diagonal at all fan
widths, demote #2 below #3.

### Result (2026-09-07) — **optimism confirmed at +0.13, and the follow-up localises it in the backup**

`scripts/audit_corpus_stats.py`, full pass over gen03-gen07, all 8 shards each (40 shards, 4.0 GB)
in 34 s wall / 29 MB peak RSS. One sample dropped in 630,383 (a null `root_value`).

**Scale caveat that decides which shards D1 can use.** Under `Nn`/`NnValue`, `score_leaf` returns
the **anchor-relative** value ×1000 — the same drive-relative frame as `outcome_value`. Under
`Heuristic` it returns bare `leaf_score`, which is **absolute**: 34-36% of heuristic-shard roots
have `|root_value/1000| > 1`, versus **0.0000%** of nn roots. The heuristic third of the corpus is
not on the outcome scale at all and is excluded from every headline below.

**Mean signed gap `E[v − outcome]`, mover frame, nn shards** (positive = optimism):

| stratum | gen05 | gen06 | gen07 |
|---|---|---|---|
| **overall** | **+0.136** | **+0.140** | **+0.133** |
| fan ≤10 | +0.127 | +0.131 | +0.122 |
| fan 11-30 | +0.225 | +0.232 | +0.223 |
| fan 31-60 | +0.131 | +0.129 | +0.127 |
| fan >60 | **+0.095** | **+0.106** | **+0.108** |

Per-sample fit `outcome ≈ a + b·v`: slope **0.934 / 0.946 / 0.948** overall, moving the *wrong
way* with fan width (0.91 at ≤10 → 1.02-1.04 at >60). The mover's base rate is `E[outcome] = +0.209`,
so the search claims +0.345 against a true +0.209. Brier is worst in the middle bins (0.15-0.17)
and best at the saturated ends (0.04-0.06).

**Read, in the plan's own terms.** *"Calibrated: bin means on the diagonal"* — **no**, every
positive bin sits 0.10-0.24 below it while the negative bins sit essentially on it: the defect is
a one-sided **offset**, not a two-sided compression. *"Flatter than the diagonal **and flattens
further as fan width grows**"* — first half barely (6% flat), **second half no**: slope rises
toward 1.0 as the fan widens and the gap *falls*. The plan's stated signature is absent.

**Sign-convention check** (gen05 shard0, 19,002 roots). Fitting each mover separately gives slope
0.935 (Home) / 0.900 (Away) and gap +0.150 / +0.119 — both positive, so no sign is inverted. The
decisive row: **in the raw Home frame the gap is +0.008, essentially zero.** The optimism is
purely a *mover-frame* effect, which is equally consistent with a max-over-noise backup and with a
mover-centric value head that is simply biased high (`perspective.rs` makes the net's output
mover-centric by construction).

### Follow-up (2026-09-07): the bare value head, no search — **it is the backup**

`scripts/audit_value_head_bias.py`. Run `bbnet_14x7_gen03.pt` — the net that generated the data —
over `gen07/shard4` (an `nn` shard) with **no search**, and subtract. The prepared `value.npy` is
`targets.rs::value_target` (mover-frame drive outcome) and the value head is mover-centric, so the
two compare directly; row alignment is asserted, not assumed (`prepare` dropped 0 of 18,990, and
the shard's outcomes match `value.npy` to 0.000000).

| on the identical 18,990 rows | gap vs drive outcome | SE |
|---|---|---|
| **bare NN leaf value, no search** | **+0.0311** | 0.0045 |
| **1000-iteration search root value** | **+0.1319** | 0.0045 |
| **added by the search** (paired) | **+0.1008** | 0.0019 (**z = 52**) |

The bare head is nearly calibrated — gap +0.031, and its slope is **1.056**, i.e. slightly
*steeper* than the diagonal, not flatter. **76.4% of the search's optimism is manufactured by the
search itself.**

**Replicated on every generation the frozen gen03 net produced, and across both prior sources**
(83,432 rows in total; each generation's shard4, all scored by `bbnet_14x7_gen03.pt`, the net that
generated them):

| gen | generator | rows | bare leaf gap | search gap | **added by search** | share |
|---|---|---:|---:|---:|---:|---:|
| gen04 | `nn-value` — **scripted** priors | 21,343 | +0.0671 | +0.1578 | **+0.0908** (z=50) | 57.5% |
| gen05 | `nn` — learned priors | 21,967 | +0.0362 | +0.1226 | **+0.0865** (z=51) | 70.5% |
| gen06 | `nn` | 21,132 | +0.0447 | +0.1391 | **+0.0944** (z=52) | 67.9% |
| gen07 | `nn` | 18,990 | +0.0311 | +0.1319 | **+0.1008** (z=52) | 76.4% |

**The added optimism is +0.086 to +0.101 everywhere, and gen04 — which used scripted priors —
shows the same +0.091.** So it is not a prior artefact: swapping the entire prior source changes it
by less than its spread across generations. The bare leaf gap varies more (+0.031 to +0.067)
because the same net is being scored on four different state distributions. That is exactly the
signature of a backup-rule effect rather than a prior or value-head effect.

**Alignment is asserted, not assumed.** `prepare` drops a sample when either target is missing, and
gen06 shard4 has exactly one such row (a root with no policy target). `read_shard` mirrors
`targets.rs`'s drop rule and the script asserts the shard's outcomes equal `value.npy` elementwise —
a silent one-row shift would otherwise compare one state's search against another state's leaf.

By fan width, the added optimism is +0.090 (≤10), **+0.220 (11-30)**, +0.042 (31-60), +0.066
(>60) — it peaks at *intermediate* width. That is coherent with max-over-noise rather than against
it: at 1000 iterations a >60-child root gives each child ~1 visit so there is barely anything to
take a maximum over, and a ≤10-child root searches each child deeply enough for Q to converge
toward true minimax. The 11-30 band is where there are enough children to maximise over and too
few visits to resolve them. The plan's prediction of *monotone* flattening with fan width was the
wrong prediction for the mechanism, because fan width at a fixed budget confounds with per-child
depth.

**Decides. Plan 032 #2 (mean backup) is NOT demoted — it is promoted, and its mechanism is now
measured rather than argued.** The demotion gate ("calibration on the diagonal at every fan width")
is not met; its own "+0.05 to +0.10 *if D1 shows optimism*" clause is met at +0.133-0.140; and the
follow-up rules out the competing explanation (a biased value head) that would have aimed the work
at the wrong layer. Record the expected direction as a prediction to check: a mean backup should
remove most of the +0.10, and if it does not, the mechanism is something else.

## D2 — Tie rate and policy self-distillation

**Why.** With `--evaluator nn`, PUCT visits at a root whose children tie on Q are proportional to
the net's own prior, and the policy target is normalised visits (`targets.rs`). The head can learn
to reproduce itself. Plan 028 leaned on "84.8% tied roots", which was measured on 8x3 pure-td.

**How.** Per root: (a) exact tie between the top-2 mover-Q children (yes/no), (b) Spearman
(prior, visits) over children, (c) entropy of the π target, (d) top-1 agreement between argmax-Q
(the played move) and argmax-visits (the label). Report by generation gen05→gen07 and by fan width.

**Read.** Self-distillation signature: rising prior/visit correlation and falling π entropy across
generations at constant tie rate. A 14x7 tie rate above ~30% on its own justifies root noise
(plan 032 #4) and Q-informed targets (#7).

### Result (2026-09-07) — **the tie rate is 13.5%, not 85%; and the gen05→07 test is vacuous**

π is `policy_target(sample, SolvedRootPolicy::OneHot)` ported verbatim from `targets.rs`
(including Rust's *last*-max `max_by_key` tie behaviour); `prepare`'s defaults are `onehot` and
`--min-root-visits 0`, and `train_loop.sh` passes neither flag. Roots with one child are excluded
from the tie/Spearman columns and counted separately.

**`nn` shards (0-4):**

| gen | fan | roots | tie rate | top-1 agree | ρ(prior,visits) | H(π) nats |
|---|---|---:|---:|---:|---:|---:|
| gen05 | ALL | 104,412 | **0.135** | 0.588 | 0.435 | 1.211 |
| gen06 | ALL | 104,883 | **0.133** | 0.584 | 0.438 | 1.221 |
| gen07 | ALL | 101,213 | **0.138** | 0.592 | 0.436 | 1.212 |
| gen07 | ≤10 | 72,053 | 0.096 | 0.711 | 0.433 | 0.501 |
| gen07 | 11-30 | 12,286 | 0.199 | 0.595 | 0.430 | 1.373 |
| gen07 | 31-60 | 4,722 | 0.145 | **0.222** | 0.392 | 3.025 |
| gen07 | >60 | 12,152 | 0.245 | **0.251** | 0.471 | 3.240 |

**`heuristic` hedge shards (5-7), a third of every corpus:** tie rate **0.578-0.585 overall** and
**0.886 at >60 children**; ρ 0.22-0.23; H(π) 1.53-1.54.

**Read, in the plan's own terms.**

- *"A tie rate above ~30% justifies #4 and #7."* — **not met in the nn half: 13.5 / 13.3 / 13.8%.**
  Plan 028's "84.8% tied roots" was measured on 8x3 pure-td and does **not** carry over. Only the
  widest nn bucket reaches 24.5-24.9%, still under the bar. **The tie problem is real but it lives
  in the hedge**, at 58% overall and 89% on wide roots — and that is a third of every corpus.
- *"Self-distillation signature across gen05→07."* — **not observable, and not because it is
  absent: all three generations were generated by the same frozen `bbnet_14x7_gen03.onnx`.** No
  candidate has passed the gate since gen03. ρ = 0.435/0.438/0.436, H(π) = 1.211/1.221/1.212, tie
  = 0.135/0.133/0.138 — flat to three decimals, exactly as one fixed generator must be. **The
  plan's gen05→gen07 test is vacuous on this corpus**, and any future self-distillation
  measurement must first arrange a *changing* generator net.
- The measurable version is **gen04 → gen05** — same gen03 net, priors switched scripted → learned
  — and it *does* show the signature as a one-step level shift: **ρ 0.364 → 0.435, H(π) 1.261 →
  1.211, tie rate 0.127 → 0.135 (≈constant)**. That is the mechanism the plan describes, caught at
  its onset. Whether it *compounds* cannot be tested until two consecutive generations run
  different generator nets.

**The strongest finding here was not on the list.** Top-1 agreement between the move actually
played (argmax mover-Q, `pick_best_action`'s key) and the π label (argmax visits) is only
**0.588 / 0.584 / 0.592** overall — and **0.22-0.25 on roots with >30 children**. On wide roots the
policy label points at a different action from the one the bot played three times out of four.
That is a far larger label defect than the tie rate, and it is exactly the regime #7's
Gumbel/completed-Q argument targets.

**Decides.** **Plan 032 #4 (root Dirichlet noise) is NOT promoted above #3** — its gate ("a high
tie rate *or* falling π entropy") fails on both limbs: 13.5% is not >30%, and π entropy is flat
across gen05→07 (the only entropy fall in the data is the gen04→gen05 regime switch, not a
within-regime trend). #4 stays where it is. **Plan 032 #7 (Q-informed policy target) gains
independent support on a stronger statistic than the plan proposed**: 41% top-1 disagreement
overall, 78% on roots with >30 children.

## D3 — Is the encoding partially observable for the active activation?

**Why.** `encode.rs` has no plane for the active player's action type (Move/Blitz/Pass/Handoff/
Foul) or "blitz block already thrown"; only team-level `blitz_available` etc. Two states with
identical tensors can have different legal sets and values.

**How.** In `prepared_train/dims_16x9`: hash each `spatial`+`global` row; for colliding rows compare
the legal-action sets (`actions.npy` slices via `action_offsets`) and value targets. Count
collisions with differing legal sets, and the value-target variance inside collision groups vs
overall.

**Read.** >1% of samples in ambiguous groups → add an action-type one-hot on the active square
(plan 032 #8). Also tells us the irreducible value MSE from ambiguity.

### Result (2026-09-07) — **the trigger fires, the named remedy is wrong**

`scripts/audit_encoding_collisions.py`, over `gen07/prepared_train/dims_16x9` (N = 350,595,
commit `3ea5238`, not dirty). The `.npy` layout was verified against `prepare.rs` / `npy.rs` /
`actions.rs` rather than assumed, and the encoding is on a discrete lattice (every value is an
indicator, a tackle-zone count, or a small integer over a compile-time constant), so bit-exact
hashing is the right equivalence — there are no near-duplicates it could miss.

| quantity | value |
|---|---|
| distinct `(spatial, global)` tensors | 328,136 (93.59%) |
| rows in a collision group | 44,804 (**12.7794%**) |
| rows in an **ambiguous** group (identical tensor, differing legal set) | 44,800 (**12.7783%**) |
| ambiguous groups with **pairwise-disjoint** legal sets | 22,343 / 22,343 = **100.00%** |
| max row-index span within a group | **3** (22,229 groups of size 2, 114 of size 3) |
| value-target variance within collision groups | **0.000000** (overall variance 0.647) |

**What actually differs is the procedure stack, not the action type.** Every collision is two
adjacent phases of one block/push/reroll chain over an unchanged board: block-die-select →
push-square → follow-up → resume moving, plus reroll prompts. Typical groups:

```
{SelectBothDown, SelectPow, UseReroll}  vs  {Push(2,7)}
{FollowUp(2,5), FollowUp(3,4)}          vs  {EndPlayerTurn, Move(1,1) … }   (39 actions)
{DontUseReroll, UseReroll}              vs  {Push(3,7), Push(4,6), Push(4,7)}
```

**Not one `Start*` action appears in the differing-action histogram.** Since all groups are
pairwise disjoint, an action present in any member would necessarily appear there — so no
ambiguous row is an activation-selection node, and Move-vs-Blitz never causes a collision. The
plan's hypothesis is disconfirmed on its own data.

**And the measured cost is zero for both heads.** `train.py:91-94` takes a per-sample masked
log-softmax over the gathered legal actions only; because the colliding members' supports are
100% disjoint, one shared logit tensor satisfies both targets *exactly and simultaneously*, so
the irreducible policy loss is 0, not merely small. The 0.000 value variance is partly
tautological and must be read as such — `value_target` is the drive outcome and all collisions
are intra-drive, so this corpus **cannot** charge the value head even if a cost exists.

**A blind spot D3 cannot measure, worth carrying forward.** In *search*, `value_home_i64` scores
mid-block-chain leaves, and the block dice are visible only through the legal-action set, which
the tower never sees — a leaf after a Skull and a leaf after a Pow get the same value as the
pre-block node. That is a genuine leaf-scoring information gap, it is closer in spirit to D8, and
it needs an online measurement.

**Decides.** **Rewrites plan 032 #8 rather than promoting it.** Drop the action-type one-hot: it
would encode a distinction that never collides. If anything is added it should be a ~5-value
pending-decision-phase one-hot, which would separate 100% of these groups — ranked as a cheap
speculative capacity tidy-up with a measured payoff of 0 on this corpus, not as a correctness
fix. Nothing in the current plateau is explained by encoding ambiguity.

**Caveats.** `prepare.rs` does no dedup and no augmentation (the y-flip is per-sample at batch
time in `data.py`), so 12.78% is the clean as-stored figure; as-trained it can shift both ways.
gen07 `prepared_train` only; the mechanism is structural so gen05/06 should match, but that is
inference, not measurement.

## D4 — How prior-dominated is the search under `nn`?

**Why.** NN priors are softmax×len (`eval.rs`); `PUCT_C=10` was tuned for scripted priors in
[0.2, 10]. Under `nn` a confident action gets P≈30 and the tail P≈0.

**How.** `BLOOD_MCTS_DEBUG_ROOT=1` on 20 random-start states (same seeds) at 1000 iters, once with
`--evaluator nn` and once with `nn-value`, same gen03 net. Record per root: visit entropy, share
of children with ≤1 visit, visits of the argmax-prior child, and whether argmax-Q == argmax-prior.

**Read.** Under `nn`: entropy far below `nn-value`, most children at exactly 1 visit, argmax-Q
== argmax-prior in most roots → the search is following the prior, and plan 032 #3 (retune `c`,
FPU reduction) moves up. Also gives the per-root `n_legal` distribution the α for #4 needs.

### Result (2026-09-07) — **prior domination is real and large, but not the collapse the Read describes**

`scripts/audit_root_priors.py`. No engine change was needed: `botbowl-ui convergence` already
re-searches fixed random-start states and dumps every child's visits/q/**prior**. Two independent
arms, because each answers a different question.

**Raw `top_prior_share` is unreadable across strata** — with `n` children the uniform baseline is
`1/n`, so a small root scores high mechanically. Everything below reports **lift** = share ÷ (1/n),
where **1.0 = uniform** and higher = prior domination.

**(a) Controlled paired probe** — 150 states × 2 repeats, same seeds and same gen03 net in both
arms, only the prior source differs (`--evaluator nn` vs `nn-value`). 300 roots, fan width 11-60:

| metric (1.0 = uniform) | `nn` | `nn-value` | paired diff |
|---|---|---|---|
| `top_prior_lift` | **5.54** | **1.08** | **+4.45 ± 0.39** |
| `q_eq_prior_lift` | 7.06 | 1.15 | +5.91 ± 0.94 |
| `singleton_share` (≤1 visit) | 0.087 | 0.032 | +0.055 ± 0.0065 |
| `norm_entropy` | 0.654 | 0.724 | −0.071 ± 0.0072 |

**(b) Production-scale natural A/B** — gen04 was generated with `--evaluator nn-value` and gen07
with `nn`, off the **same champion** `bbnet_14x7_gen03.onnx` at the same 1000 iterations (D6
confirms). Unpaired (different seed bases) but it is the real distribution of decision types,
~23k roots per arm:

| stratum | `top_prior_lift` gen04 → gen07 | `singleton_share` | `norm_entropy` |
|---|---|---|---|
| ≤10 | 1.09 → 1.67 | 0.068 → 0.087 | 0.434 → 0.422 |
| 11-30 | 1.13 → **3.55** | 0.079 → 0.118 | 0.519 → 0.491 |
| 31-60 | 3.53 → 4.10 | 0.054 → 0.068 | 0.815 → 0.808 |
| >60 | 5.89 → **8.10** | 0.112 → **0.168** | 0.802 → 0.770 |
| **ALL** | **1.98 → 3.11** | 0.075 → 0.103 | 0.526 → 0.510 |

**The prior distributions themselves** (200k child priors per arm, production shards):

| | scripted (`nn-value`) | learned (`nn`) |
|---|---|---|
| distinct values | **4** — `{0.2, 1.0, 5.0, 10.0}` | 185,406 (continuous) |
| median | 1.0 | 0.83 |
| p1 / p99 | 0.20 / 5.0 | 0.098 / 3.66 |
| **max** | **10.0** | **57.9** |
| p99/p1 | 25× | 37× |

On the probe's *turn-start activation* roots the scripted prior degenerates further still — **two
values, 97.7% of children at exactly 1.0**, i.e. essentially uniform. `priors.rs`'s positional
multipliers do not apply to `Start*` actions. So on that class of root, `PUCT_C = 10` was tuned
against **no prior shaping at all**, and `c · P · √N/(1+N)` was effectively a pure visit-count
exploration bonus.

**Read, in the plan's own terms — all three limbs fail:**

- *"entropy far below `nn-value`"* — **no.** 10% lower on the probe (0.654 vs 0.724), 3% lower in
  production (0.510 vs 0.526). Consistently lower, nowhere near "far below".
- *"most children at exactly 1 visit"* — **no.** 8.7% (probe) / 10.3% (production). Over 90% of
  children get a second visit.
- *"argmax-Q == argmax-prior in most roots"* — **no.** 20% (probe) / 42% (production). Note this
  metric is the weakest of the three anyway: a *trained* policy head should agree with the search
  more often than a scripted one, so it conflates "the search follows the prior" with "the prior is
  right". `singleton_share` and entropy are the discriminating measures, and they say the search is
  not collapsed.

**What is true instead.** The NN prior concentrates early visits far more than the scripted one —
the argmax-prior child takes 5.5× its uniform share of visits against 1.1× under scripted priors on
matched states, and 3.1× vs 2.0× across the production distribution — and it does so with a top
tail (max 57.9) nearly 6× longer than the scripted prior's hard ceiling of 10.0. The search is
substantially more prior-led, without starving the tail.

**Decides. Plan 032 #3 (retune `PUCT_C`, add FPU reduction) is supported and stays where it is —
but on the strength of the concentration and range change, not the collapse the item's gate was
written around.** Its gate ("if root visit entropy under `nn` is not far below `nn-value`, drop to
#6") reads as failed on a literal reading; that gate was mis-specified, because entropy at fixed
budget is dominated by the visit-count term and is insensitive to exactly the change that occurred.
Replace the gate with the measured one: **the prior's dynamic range went from a 4-level ladder
capped at 10.0 to a continuous distribution reaching 57.9, and `c` has never been re-tuned for it.**

**`n_legal` for #4's α** — production distribution, not the probe's (the probe only sees turn-start
roots): **mean 20.2, median 6, p10 2, p90 73, max 98.** A fixed α = 10/mean = 0.49 is a poor fit to
a distribution this bimodal; use a **per-root α = 10/n_legal** (KataGo-style) rather than one
constant. Note also that plan 032 #7's premise, "30-100 children", describes only the top ~15% of
roots — the median root has **6**.

## D5 — Did the warm-start fine-tunes move the champion at all?

> Partly answered by plan 029 stage 3 (2026-09-07): warm-started W1/W3 restored at step 2,500 /
> 20,000 and W3 vs W1 = 0.487; gen05-07 restored at epoch 0-1 of 10. (b) below is still worth
> the ten minutes — it puts a number on how far 2e-4 actually moves the weights.

**Why.** Since `3ea5238` each candidate is `gen03.pt` + Adam 2e-4 + early best-val restore on data
gen03 generated. Parity vs gen03 is what that predicts.

**How.** (a) From `gen06/train.log`, `gen07/train.log`: `warm-start baseline val_value` vs the
`restored best-val weights` line and the epoch/step it restored at. (b)
`‖θ_genN − θ_gen03‖₂ / ‖θ_gen03‖₂` per layer for N=6,7, and the same for gen03 vs gen02 as a
reference for what a "real" step looks like. Five lines of torch.

**Read.** Restored val within noise of baseline and relative weight movement well under gen02→gen03's
→ the plateau is the fine-tune configuration. Then plan 032 #1's warm-started pair should run at
lr 1e-3 as well as 2e-4, and `WARM_LR` becomes a live knob.

### Result (2026-09-07) — **the condition is NOT met; the hypothesis fails in the opposite direction**

`scripts/weight_distance.py`. Three corrections to this item's own prose first, each verified
against the files:

1. **gen05 did not warm-start from gen03.** `gen05/train.log:2` and `status.md:109`: `WARM_FROM=latest`
   picked **gen04**. gen04/06/07 did start from gen03.
2. **gen02 → gen03 is not a from-scratch reference.** `gen03/train.log:2` shows gen03 was itself
   the *first* 2e-4 warm start. gen01/gen02 were the last from-scratch trains, and they are used
   below as the real "two unrelated nets" scale.
3. **`runs/exp-data/arm_init.pt` is a fresh random init, not gen03.** W1/W3 were measured against
   `models/bbnet_14x7_gen03.pt`, which is what their train logs say they loaded.

**(a)** All four fine-tunes restore at epoch 0-2 of 10, with restored `val_value` only 0.002-0.012
below the warm-start baseline. gen04 selected on `val_value` (0.3934); gen05-07 on
`val_policy+val_value` (~1.85) — **different quantities, never compare them.** In every run the
two heads' optima disagree: `val_value` bottoms at epoch 0-2 then climbs to ~0.47 by epoch 9 while
`val_policy` falls throughout. That is the concrete justification for `--eval-every`.

**(b) Relative weight movement** `‖θ_B − θ_A‖₂ / ‖θ_A‖₂`, learnable params only (BN buffers
reported separately, `num_batches_tracked` excluded):

| pair | regime | steps before restore | **rel L2** |
|---|---|---|---|
| gen03 → gen06 | warm 2e-4 | ~33k | **0.414** |
| gen03 → gen07 | warm 2e-4 | ~22k | **0.334** |
| gen04 → gen05 | warm 2e-4 | ~22k | 0.268 |
| **gen02 → gen03 (the reference)** | warm 2e-4 | ~11k | **0.245** |
| gen03 → gen04 | warm 2e-4 | ~11k | 0.188 |
| gen03 → W1 / W3 (plan 029) | warm 2e-4 | 2.5k / 20k | 0.108 / 0.315 |
| gen01 ↔ gen02, D1 ↔ D3 | two independent from-scratch fits | — | **1.26 / 1.94** |

The per-group shape is uniform — every group moves, deeper blocks slightly more than the stem,
nothing frozen, nothing dominant.

**Read.** Clause 1 (restored val within noise) **holds**. Clause 2 **fails, and in the opposite
direction**: gen06 moved **1.70×** and gen07 **1.36×** as far as gen02→gen03, the one fine-tune
that actually won its rung (0.650). Even gen04, the shortest of the four, moved 77% as far.
Movement scales cleanly with optimizer steps across both the production loop and plan 029's
fixed-step arms, which cross-validates both.

So the picture is not "2e-4 cannot move the net". It is **"2e-4 moves the net 20-40% of the way to
an unrelated network and nothing measurable happens."**

This also qualifies plan 029's conclusion 4 ("warm-started training barely moves the net"): that
was inferred from *turnover step counts*, and in weight space it is not true. What is true is the
weaker statement that the *validation optimum arrives early*.

**Decides.** **Demotes plan 032 #1(b)** — the lr 1e-3 warm arm. Its premise is that 2e-4 is too
small a step, and it is not; raising `WARM_LR` would push the weights further into a region
per-epoch validation already calls worse. **`WARM_LR` should not become a live knob on this
evidence.** #1(a) (`d7 vs gen03`) is untouched and now cleaner — D5 removes the competing
explanation "the fine-tunes never moved". Confirms the no-experiment adoption of `--eval-every`.
Weakly promotes #2/#3 by elimination: the net is being moved substantially and strength does not
follow, so the bottleneck is more likely in what the search does with the net than in how the net
is fit.

## D6 — Corpus composition facts the plans got wrong or left open

- Which generation first ran `--evaluator nn`: the gen05 `generate:` status line
  (`train_loop.sh` prints the evaluator). Fix plan 028/029 accordingly.
- Heuristic-hedge share **by samples** (not games) per generation: heuristic drives are shorter
  (~24 vs ~28 samples/drive), so the 37.5% game share is ~33% of samples; record the real number
  and the per-class value-target split for each half of the corpus.
- Fraction of drives ending at the half boundary (label 0 that is "clock", not "no score"), by
  `start_turn`.
- `NN_SERVER_FALLBACK` count and `mean_batch` per shard for gen05-07 (`shard*.log`,
  `nn_server.log`): confirms every compared generation ran the same numerics path.

### Result (2026-09-07)

**Which generation first ran `--evaluator nn`: gen05.** Two independent sources agree —
`status.md`'s `generate:` lines (gen01-04 read `nn-value`, gen05-07 read `nn`) and every shard's
own `meta.home_bot`/`meta.away_bot` over all 40 shards. **Plan 029's inventory table is correct.
Plan 028** §"The policy label is finally consumed" reads *"Until gen05 … the loop played
`--evaluator nn-value`"*, which parses as gen05 still being `nn-value`; it should read **"up to and
including gen04"**.

**Also, and load-bearing for everything else in this audit: the generator net has been frozen at
`bbnet_14x7_gen03.onnx` since gen04.** gen05, gen06 and gen07 are three samples of *one*
generator, not a trajectory. That is what makes D2's cross-generation test vacuous.

**Heuristic-hedge share by samples** — the plan's ~33% prediction is confirmed to the decimal
(its per-drive numbers were off: 34.8 vs 28.9 measured, not 28 vs 24):

| gen | nn samples (s/drive) | heur samples (s/drive) | **heur share of samples** |
|---|---|---|---|
| gen05 | 104,412 (34.8) | 51,961 (28.9) | **33.2%** |
| gen06 | 104,883 (35.0) | 51,457 (28.6) | **32.9%** |
| gen07 | 101,213 (33.7) | 51,044 (28.4) | **33.5%** |

**But the split is not uniform across the train/val partition** (`train_loop.sh`: train = shards
0-3,5,6; val = shards 4,7), measured from the shard logs' `wrote N / M samples` lines:

| gen | heur share of **train** | heur share of **val** |
|---|---|---|
| gen05 | 29.3% | **44.7%** |
| gen06 | 28.9% | **45.0%** |
| gen07 | 29.1% | **47.5%** |

**The pool that best-val restore selects on is ~45-48% scripted-heuristic play, against a 29%
training mix.** Holding out whole shards is right (samples within a game are correlated), and the
`VAL_SHARDS="4 7"` comment shows the intent was for both kinds to be *represented* — the defect is
that representation is not proportional representation.

**Sized, 2026-09-07 — smaller than it first looks, and it does *not* explain the early restore.**
Scoring each half separately (`prepare` on gen07 shard4 and shard7, then `train.py::evaluate`):

| net | pool | `val_policy` | `val_value` | `val_top1` |
|---|---|---|---|---|
| gen03 | nn half (shard4) | 1.4243 | 0.3938 | 0.5524 |
| gen03 | heuristic half (shard7) | **1.5577** | 0.3936 | 0.5002 |
| gen07 | nn half | **1.3845** | 0.3892 | 0.5564 |
| gen07 | heuristic half | **1.5611** | 0.3908 | 0.4961 |

Two facts decide the reading. (1) `val_value` is **identical across the halves** (0.3938 vs
0.3936), so the mixture is nearly invisible to the value term. (2) Over the gen07 fine-tune the net
**improved on the nn half** (1.4243 → 1.3845) while **slightly degrading on the heuristic half**
(1.5577 → 1.5611) — they move in opposite directions, so the mixture weight is not a harmless
constant offset. The current pool reports roughly **half** the policy progress the deployment
distribution (nn vs nn) actually saw.

**But it is not the cause of "restores at epoch 0-1 of 10".** Across gen05/06/07, `val_policy`
falls monotonically through epoch 9 (swing ~0.010) while `val_value` climbs 0.39 → 0.48 (swing
~0.09). The value term dominates the combined criterion by ~9×, and it is what forces the early
restore — so re-weighting a pool the value term cannot see would barely move the restore point.
**The early restore is value-head overfitting**, which is what `--eval-every` addresses.

So this is selection *hygiene* of modest size — it understates policy progress by about half and
gives contradictory training signal 47.5% of the vote — not a load-bearing bug, and **not** the
confound for D5 / plan 029's finding that an earlier draft of this section claimed.

**Value-target class split.** The two halves label differently: the hedge carries ~10 points more
label-0 mass (0.39 vs 0.29) and ~10 points less mover-`+1`. Incidental but material for D9/#11:
the nn half is Home-frame **Away-skewed** in all three generations (−1 mass 0.359-0.385 vs +1 mass
0.327-0.365) while the hedge is near even — the same direction as plan 027's unlocalised 56% Away
share.

**Drives ending at the half boundary — the classification is exact, not a proxy.**
`dataset.rs::random_start_trajectory` stops a drive on `score changed || half changed`, so a drive
ending with the score unchanged can only have run out of turns. Two independent checks: **100.00%**
of the 5,278 non-scoring drives have `(home_turn, away_turn) == (8,8)` on their last sample, and
`outcome_value == 0` ⟺ "non-scoring drive" agree on **100.0000%** of samples in both directions.
**So there is no "genuinely no score" class distinct from clock in this corpus: every label-0 is
clock.** Clock is 18.8-20.1% of nn drives and 25.4-26.6% of heuristic drives; clock drives are much
longer (45-50 samples vs 28), so 19-27% of *drives* become 28-40% of *samples*. By start turn
(pooled gen03-07) clock rises 7.2% → **65.2%** from turn 1 to turn 8, and start turn 8 alone —
12.5% of drives — supplies **36%** of all clock drives.

**Numerics path: identical across gen05-07.** Zero `NN_SERVER_FALLBACK` in all 24 shard logs;
shards 0-4 each end `fell_back_to_tract=0`. Per-shard `served` counts sum **exactly** to each
server session's `samples=` total (117,757,530 / 117,856,464 / 113,407,537), which is the proof
that no forward escaped the sidecar. `mean_batch` 3.54 / 3.59 / 3.58, pad 0.06. (`mean_batch` is
per-server-session, not per-shard — one server serves all five nn shards of a generate phase.)

**Decides.** **Plan 032 #6 (heuristic hedge ablation) clears its gate** (33% ≫ 20%) and stays in
the queue; the by-sample number also sharpens its health check, since the hedge is where the
label-0 mass and the Q-tie mass both concentrate. The val/train mixture mismatch is recorded as a
**low-priority cleanup, not an adopt-now item** — #6 would dissolve it entirely by removing the
heuristic half, so fixing the split first would only have to be undone. If it is fixed, the cheapest
form is to select on the nn half alone and keep shard 7 as a monitoring readout, since deployment is
nn vs nn. And the numerics path is **removed** as an explanation for the flat gate results.

## D7 — What does y-flip augmentation cost the policy head?

**Why.** The search's chance collapse is `Direction::up()` for every bounce/scatter/deviate, so
policy targets near a sideline with a loose ball are y-asymmetric; the augmentation flips them.

**How.** Train D1 twice from `runs/exp-data/arm_init.pt`, `--no-augment` vs default, same seed and
steps (36 min each), compare `val_policy` and `val_top1` on the shared holdout. Optional: restrict
the comparison to samples with `ball_on_ground` set.

**Read.** If `--no-augment` is not worse on `val_policy`, drop augmentation from the policy loss
(keep it for the value head) or make the chance collapse y-covariant. Expected small.

### Result (2026-09-07) — **`--no-augment` is worse; keep the augmentation**

Only one train was needed, not two: `runs/exp-data/d1.pt` **is** the augmented arm (`train_arm`
never passes `--no-augment`, and `train.py`'s default is `augment=True`). So the control already
existed and this cost ~36 min of GPU, not the ~75 min budgeted. The `--no-augment` arm is
`runs/audit031/d7-noaug.*`, identical in every other respect — same `arm_init.pt`, same
`--seed 20260906`, same 110,000 steps, same `--eval-every 2500`, same `prep_d1` pool, same gen07
holdout. Init assertion passes: both log `epoch -1 (warm-start baseline) | val_value 0.6410`.

| metric (44 checkpoints each) | augment (`d1`) | `--no-augment` | Δ |
|---|---|---|---|
| best `val_policy` | **1.4740** @ 42,500 | 1.4832 @ 25,000 | **+0.0092 worse** |
| best `val_value` | **0.4074** @ 5,000 | 0.4101 @ 5,000 | +0.0027 worse |
| best `val_top1` | 0.5260 @ 20,000 | **0.5280** @ 10,000 | −0.0020 *better* |
| best combined (what the loop selects on) | **1.8979** @ 17,500 | 1.9128 @ **7,500** | **+0.0149 worse** |
| …its `val_policy` | **1.4781** | 1.4971 | +0.0190 worse |

**Read: `--no-augment` is worse on `val_policy`, so the plan's condition is not met — do not drop
the augmentation, and do not spend effort making the chance collapse y-covariant on this evidence.**

**What the augmentation is actually doing is regularising, not correcting.** The gap is negligible
early and grows without bound late:

| step | 2,500 | 7,500 | 17,500 | 50,000 | 110,000 |
|---|---|---|---|---|---|
| Δ `val_policy` (no-aug − aug) | +0.0023 | −0.0084 | +0.0079 | **+0.0334** | **+0.0995** |

Without augmentation the net reaches its best `val_policy` at 25,000 steps and its selected
checkpoint at **7,500**; with it, 42,500 and 17,500. So the y-flip is buying delayed overfitting on
a fixed corpus, which is the ordinary reason to augment — not repairing a y-asymmetry.

**The one result that leans the plan's way, reported because it is the interesting half.** Best
`val_top1` is marginally *better* without augmentation (0.5280 vs 0.5260). Cross-entropy and
top-1 disagree in sign, which is what you would expect if the hypothesised y-asymmetry is real but
small: flipping genuinely y-asymmetric policy targets blurs the argmax slightly, while the
regularisation more than pays for it in likelihood. So the plan's mechanism may well exist — it is
just an order of magnitude smaller than the benefit it would have to beat.

**Caveat on power.** One seed pair. The `val_policy` differences (0.009-0.019 nats) sit in a range
where D10's lesson applies and there is no across-seed noise floor for `val_*` — plan 032 #8 (seed
variance) would supply one. The *overfitting* difference (+0.0995 by 110k) is far too large to be
seed noise and is the part to rely on.

**Decides.** Closes the question — no plan 032 item is created. If the y-covariance of the chance
collapse is ever revisited, it should be motivated by a search-side measurement, not by this
training comparison.

## D8 — How often is a mid-procedure state scored out of distribution?

**Why.** `score_leaf` case 4 (no team, no pending roll, not game over) asks the NN about a state
type never present in training.

**How.** Add a counter next to the existing `BLOOD_MCTS_STATS` registry counters, run 20 games.

**Read.** Below 0.1% of forwards: close. Otherwise advance through them in `apply_action`'s
quiescent loop.

### Result (2026-09-07) — **exactly 0. The case is unreachable by construction.**

`LEAF_STATS` in `dynamics.rs` (see D8's code commit) tallies `score_leaf` calls by the four cases
its own comment documents, plus how many reached a real network forward and how many were answered
by the exact-outcome carve-out. `leaf_case` mirrors `score_leaf`'s branching and a unit test pins
the two together. Dumped by `BLOOD_MCTS_LEAF_STATS=1`, kept separate from `BLOOD_MCTS_STATS=1`
because that one also walks the whole DAG for a depth histogram — more expensive than the search
itself at 1000 iterations.

Run: `dataset --mode random-start --games 20 --mcts-iters 1000 --evaluator nn --model gen03.onnx`,
production settings. **Stopped at 18 of 20 games** — the answer was not going to change (see below).

| counter | count | share |
|---|---:|---:|
| total `score_leaf` calls | 1,436,287 | |
| case 1 `chance_unscored` (pending roll, in horizon) | 631,530 | 44.0% of calls |
| **scored leaves** | **804,757** | |
| case 3 `player_decision` | 777,372 | 96.6% of scored |
| case 2 `terminal` (`game_over`) | 27,284 | 3.4% of scored |
| case 1b `chance_past_horizon` | 101 | 0.013% of scored |
| **case 4 `mid_procedure`** | **0** | **0.000%** |
| NN forwards | 777,372 | |
| **NN forwards on a case-4 state** | **0** | **0.000%** |

**Read: 0.000% ≪ 0.1%. Closed — no action, and the "principled fix" the code comment proposes
(advancing through them in `apply_action`'s quiescent loop) is not needed because there is nothing
to advance through.**

**Why it is zero, which is more useful than the number.** Case 4 is "no pending roll, not game
over, and no team owns a decision". The engine cannot hand back such a state:
`gamestate.rs::step_with_roll_or_action` is a loop over `micro_step` that only returns on
`NeedAction`, `NeedRoll` or `GameOver`, re-entering on `RunAgain`. There is no way to observe the
procedure stack *between* steps — the state machine always comes to rest waiting for something. On
top of that, `apply_action`'s quiescent-advance loop walks further through any decision that is
scripted (`scripted_player_pick`) or has a single legal action. So case 4 would require the engine
to report `NeedAction` with `available_actions.team == None`, which no well-formed procedure
produces. **The comment's "rare" was optimistic in the wrong direction: the case is unreachable,
not rare.**

**The counters are live, not dead.** A `Score TD` lecture — where a touchdown *is* reachable inside
the horizon — fires the neighbouring branches: `chance_past_horizon=13`, `terminal=24`,
`exact_outcome=37` (1.13% of scored leaves) out of 5,252 calls. So a zero in the random-start run
is a measurement, not a broken counter.

### Incidental, and the part worth carrying forward: **the search almost never sees a touchdown**

`exact_outcome` — the carve-out that returns the true ±1000 instead of asking the net — fired
27,385 times, 3.4% of scored leaves. But of those, **27,284 are `game_over` and only 101 are
`chance_past_horizon`**, the branch the code comments call *"exactly where every in-search
touchdown lands"*. So when the search reaches a known outcome it is almost always because the
**game ended**, not because someone scored.

Two consequences:

1. **Terminal reward is negligible as a training signal inside the tree.** 101 touchdown-shaped
   leaves in 804,757 scored ones is 0.013%. Effectively 100% of what the search backs up is NN
   estimate, which is the context for D1's finding that the backup adds +0.10 of optimism: there
   are almost no exact anchors to pull the maximum back toward the truth.
2. **The rate is lumpy and this run cannot pin it down.** All 27,284 `game_over` leaves arrived in a
   handful of games (the counter sat unchanged across several consecutive games, then jumped), and
   the cumulative share drifted 6.5% → 4.4% → 3.4% purely as the denominator grew. That is expected
   — a random start late in half 2 reaches game-over inside a 1-turn horizon and a mid-drive start
   never does — but it means **3.4% is an order of magnitude, not an estimate.** If the number ever
   matters, size a run for it; do not quote this one.

> Two earlier drafts of this section quoted "the carve-out never fires" and then "6.5%", both from
> partial runs. The lesson is in point 2: this statistic needs its own sample size, and reading it
> off a run sized for a different question produces a different answer every time you look.

## D9 — NN-evaluator exact-mirror test

**Why.** `botbowl-mcts/tests/mirror_search_exact.rs` proves search equivariance only for the
heuristic evaluator. Plan 027 left a pooled 56% Away share (z≈3.4 pair-corrected) in NN full
games unlocalised.

**How.** Add an arm of `search_mirrors_exactly_at_budget_{20,200}` using `Evaluator::Nn` on
`tiny.onnx` and on the champion, `TieBreak::Mover`, `deterministic_hash`. ~2 h including the test.

**Read.** Green: the residual is tie-break variance or turn-order (plan 032 #11 mirror games
decide). Red: a real bug in the NN path, fix before anything else.

### Result (2026-09-07) — **GREEN at every budget, including production**

`botbowl-mcts/tests/mirror_search_exact.rs` gained `Evaluator::Nn` arms sharing
`mirror_nn_evaluator.rs`'s `BLOOD_NN_MIRROR_MODEL` knob (default: the committed `tiny.onnx`, so
CI stays runnable with `models/` gitignored). Run:

```sh
BLOOD_NN_MIRROR_MODEL=$PWD/models/bbnet_14x7_gen03.onnx \
  cargo test --release -p botbowl-mcts --test mirror_search_exact -- --ignored nn_search
```

| arm | states | fixture `tiny.onnx` | champion `gen03.onnx` |
|---|---|---|---|
| budget 5 | 40 | pass | pass |
| budget 20 | 40 | pass | pass |
| budget 200 | 20 | pass | pass |
| **budget 1000 (production)** | 20 | pass | **pass** (82 s) |

Every assertion is exact equality, not a tolerance: root pick mirrors, `root_value` is the exact
negation, `root_visits` matches, and every child's visits / q / solved / terminal agree. A
`budget 1000` arm was added beyond what the plan asked for, because that is the budget every
plan-027 side-bias measurement was taken at and the lower arms could in principle stay green
while a bias only expressed itself deep in the tree.

**Cost:** the plan budgeted ~2 h. Actual: **under 2 minutes** for the fixture arms, 82 s for the
production-budget arm against the champion.

**Read: the NN search path is exactly mirror-equivariant.** Plan 027's residual 56% Away share is
therefore *not* a bug in the NN path — it is tie-break variance or turn-order, and plan 032 #11's
mirror games are what decide it.

**Decides.** Unblocks plan 032 #11 (it was gated on this) and rules out the "fix before anything
else" branch. Nothing in the queue moves up.

## D10 — Pair correlation on the per-game logs we already have

**Why.** Every z in plans 027/029 treats paired games as independent; plan 027 measured a 1.10×
SE inflation on one arm only.

**How.** `scripts/paired_summary.py runs/exp-data/*.games.jsonl runs/exp-search/*.games.jsonl`;
report the paired/unpaired SE ratio per arm and the pooled ratio.

**Read.** Use the pooled ratio to restate plan 029's z-scores and to size every match in plan 032.

### Result (2026-09-07) — **there is no pair correlation; the inflation factor is nil**

`scripts/pair_correlation.py`. Seven `.games.jsonl` files exist (4 in `exp-data`, 3 mirrors in
`exp-search`), 820 games, every Home/Away pair complete. Nine `exp-search` arms (`e1`, `e1b`,
`e2a`-`e2f`, `e3`) plus `exp-priors`, `exp-conv` and every `report.json` rung have **no per-game
log** and could not be re-analysed at all.

Per-arm `r = SE_unpaired / SE_paired` spreads 0.906–1.083 — pure noise (at df ≈ 59 the SD of
`log r` is ≈ 0.09, so ±1 SD is about ×÷1.10), and four of seven arms sit below 1.

Pooled as a **ratio of pooled variances** (sum SS over summed df separately per estimator, then
one ratio; bootstrap over pairs for the CI — chosen over averaging ratios, which is biased and
weights arms by count rather than information):

| pool | arms | V_game | V_pair | **pooled r** | 95% CI | implied ρ |
|---|---|---|---|---|---|---|
| all | 7 | 0.2011 | 0.1000 | **1.003** | **[0.962, 1.060]** | −0.005 |
| exp-data strength arms | 4 | 0.1935 | 0.0964 | 1.002 | [0.944, 1.080] | −0.004 |
| exp-search mirror (true-null) arms | 3 | 0.2118 | 0.1052 | 1.003 | [0.939, 1.100] | −0.007 |

All three sub-pools agree to three decimals, including the true-null mirrors where no skill
signal can contaminate the variance. Games-weighted mean of the per-arm ratios agrees at 1.007.
Sanity check: pooled `V_game = 0.2011` against a theoretical 0.200 at a 20% draw rate.

This does not contradict plan 027's 1.10×, which corrected the pooled **Away-share** z — a
different statistic on a different unit. Plan 027's own model predicted only 1.07× for match
points and flagged it as unmeasured. This is that measurement. Pairing remains free and strictly
correct (it removes a bias term), it just buys no variance: a pair splits 1-1 in 28-48% of cases,
but Blood Bowl's per-game dice variance swamps the side term.

**Plan 029's z-scores, recomputed from the logs (not rescaled) — all four arms had logs:**

| arm | mean | plan 029 (unpaired) | **corrected (paired)** |
|---|---|---|---|
| D3 vs D1 | 0.600 | z +2.50, p 0.012 | **z +2.69, p 0.007** |
| D7 vs D1 | 0.662 | z +4.18 | **z +4.00, p 6e-5** |
| W3 vs W1 | 0.487 | z −0.30 | **z −0.32, p 0.75** |
| M1 vs D1 | 0.546 | z +1.15 | **z +1.06, p 0.29** |
| D7 − M1 (volume at fixed diversity) | +0.117 | z +2.10, p 0.036 | **z +1.97, p 0.049** |
| D3vD1 − W3vW1 (regime difference) | +0.113 | z +1.94, p 0.052 | **z +2.08, p 0.037** |

**Every plan 029 conclusion survives.** The two borderline calls cross 0.05 in opposite
directions: "volume at fixed diversity" now sits on the line (so plan 029's conclusion 2 should
read "the larger and the only nominally significant term" rather than established), and the
regime difference slightly strengthens.

**A methodological finding that matters more than any of the above.** The four `exp-data` arms
used **disjoint seed bases** (96/97/98/99 M), so no cross-arm contrast could be differenced per
seed. That is the one place common random numbers still has real power here — the *situation*
term is shared across arms even though it is not shared within a Home/Away pair.

**Decides.** **Demotes the "~0.045 after pair correlation" inflation from plan 032's ground
rules** — the correction is nil, and every quoted SE and z in plans 027/029 stands. Does not
re-rank any item on strength grounds. Promotes two ground-rule changes: **share one seed base
across arms that will be differenced**, and **re-size or drop #5 and #6**, whose stated ±0.03
expected effect is a factor of two below what 120-240 games can resolve. Standing requirement:
every future eval passes `--per-game-out`.

Power at `V_pair = 0.1000` (5% two-sided, 80% power): detecting 0.55 needs **628 games**; 0.56,
436; 0.58, 245; 0.60, 157. At 120 games a match has 80% power only against ~0.60 or larger, and
even after plan 032's pre-committed 120-game extension a true 0.55 effect remains undecided.

## Order and cost — estimated vs actual (2026-09-07)

| step | estimated | **actual** | note |
|---|---|---|---|
| D6, D10 | minutes | ~35 s + minutes | the gen03-07 stats pass is 34 s wall over 4.0 GB, 29 MB peak RSS |
| D1, D2 | ~20 min + one pass | same pass | D1's decisive follow-up cost a further ~5 min |
| D5 | 10 min | ~10 min | |
| D3 | 15 min | ~7 min | blake2b-128 over 7.5 GB of mmapped `spatial` |
| D4 | ~30 min | ~35 min | needed **no code**: `botbowl-ui convergence` already dumps per-child priors |
| D8 | 20 min | ~2 h 20 (stopped at 18/20 games) | the counter was 20 min; the run is the cost |
| D9 | ~2 h | **~2 min** | 6 s on the fixture, 82 s for the production-budget arm on the champion |
| D7 | ~75 min GPU | **~36 min** | only one arm was needed — `runs/exp-data/d1.pt` already *is* the augmented control |

Two estimates were badly wrong in opposite directions, and both mistakes are worth remembering.
**D9 was over-estimated 60×** because the estimate assumed the test had to be written from scratch;
it was a parameterisation of an existing file. **D8 was under-estimated 7×** because the estimate
priced the *counter* and not the *run* — instrumentation is cheap, the games that exercise it are
not.

## Artefacts

Scripts, all reusable and committed under `scripts/`:

| script | serves |
|---|---|
| `audit_corpus_stats.py` | D1, D2, D6 — streams shard JSONL one line at a time, 4 workers |
| `audit_value_head_bias.py` | D1 follow-up — bare leaf value vs search root on identical rows |
| `audit_root_priors.py` | D4 — root visit/prior concentration, probe dumps or corpus shards |
| `audit_encoding_collisions.py` | D3 — byte-identical tensor groups with differing legal sets |
| `weight_distance.py` | D5 — per-layer relative L2 between checkpoints |
| `pair_correlation.py` | D10 — paired vs unpaired SE over `*.games.jsonl` |

Data under `runs/audit031/`. Code changes: `LEAF_STATS` + `BLOOD_MCTS_LEAF_STATS` in
`botbowl-mcts/src/dynamics.rs` (D8), `Evaluator::Nn` arms in
`botbowl-mcts/tests/mirror_search_exact.rs` (D9).

## Method notes worth keeping

Three mistakes were made and caught during this audit; each one has a cheap guard.

1. **Read a rate off a partial run twice and got two different answers** (D8's `exact_outcome`:
   "zero", then 6.5%, then 3.4%). A statistic that is lumpy across games needs its own sample size;
   a run sized for a different question will not do.
2. **Row alignment between a prepared `.npy` dir and its source shard is not free.** `prepare` drops
   samples with a missing target, and gen06 shard4 has exactly one. The assertion caught it; without
   it the script would have compared one state's search against another state's leaf, silently, for
   every row after the first drop.
3. **A gate can be mis-specified.** D4's ("visit entropy far below the control") and D2's
   cross-generation test both measured quantities that could not respond to the effect in question —
   the latter because gen05-07 share one frozen generator net. Check that a gate *can* move before
   trusting that it did not.
