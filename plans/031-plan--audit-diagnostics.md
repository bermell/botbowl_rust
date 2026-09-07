# Audit diagnostics: cheap checks before spending games

**Status:** **Run 2026-09-07.** All ten items executed; results are recorded under each one and the
re-ranking they imply is folded into plan 032. Written 2026-09-07 from the methodology audit. Every
item here was under ~1 h of machine time, most ran offline on the shards already in
`runs/loop14x7/`, and each one either killed or promoted a specific entry in plan 032.

## What the audit changed

Three of the four hypotheses these diagnostics were written to confirm turned out to be **wrong**,
and the two largest findings were not on the list at all.

| item | expectation going in | what was measured |
|---|---|---|
| D1 root-Q calibration | minimax-over-noise optimism | see D1 |
| D2 tie rate | ~85% tied roots (from 8x3 pure-td) | see D2 |
| D3 encoding ambiguity | missing action-type plane | 12.78% ambiguous, but **zero** of them are action-type; it is the procedure stack, and the measured cost is 0 |
| D4 prior domination | search follows the NN prior | real but partial: prior lift 1.98 → 3.11, not the collapse the Read described |
| D5 weight movement | 2e-4 barely moves the net | it moves it **1.4-1.7× further** than the last fine-tune that *did* win a rung |
| D8 mid-procedure scoring | "below 0.1% is close" | exactly **0** — and the counter surfaced that `exact_outcome` is **0 too** |
| D9 NN mirror equivariance | plan 027's 56% Away share might be a bug | **green** at every budget including production 1000 |
| D10 pair correlation | ~1.10× SE inflation | **1.003**, CI [0.962, 1.060] — there is none |

The two unplanned findings, both from D8's counters:

1. **The search never reaches a known outcome in production generation.** Over a 20-game
   random-start run at 1000 iterations, the exact-outcome carve-out fired **zero** times. Every
   leaf value the search backs up is an NN estimate; no terminal reward enters the tree at all.
   (The counter is not dead — a `Score TD` lecture fires it at 1.13% of scored leaves.)
2. **Case 4 does not exist.** `score_leaf`'s mid-procedure branch, documented as a known
   compromise, is never taken in either workload.

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

**`val_value` — the quantity best-val restore selects on — is measured on a pool that is ~45%
scripted-heuristic play, against a 29% training mix.** That is a live confound for D5 and for plan
029's "restored at epoch 0-1" finding, and it is a cheap fix.

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
label-0 mass and the Q-tie mass both concentrate. **New cheap item for #1's "also adopt now"
list:** the val pool is 45% hedge against a 29% train mix, so best-val restore selects on a
different distribution than it trains on — fix that before any more warm-start LR tuning. And the
numerics path is **removed** as an explanation for the flat gate results.

## D7 — What does y-flip augmentation cost the policy head?

**Why.** The search's chance collapse is `Direction::up()` for every bounce/scatter/deviate, so
policy targets near a sideline with a loose ball are y-asymmetric; the augmentation flips them.

**How.** Train D1 twice from `runs/exp-data/arm_init.pt`, `--no-augment` vs default, same seed and
steps (36 min each), compare `val_policy` and `val_top1` on the shared holdout. Optional: restrict
the comparison to samples with `ball_on_ground` set.

**Read.** If `--no-augment` is not worse on `val_policy`, drop augmentation from the policy loss
(keep it for the value head) or make the chance collapse y-covariant. Expected small.

## D8 — How often is a mid-procedure state scored out of distribution?

**Why.** `score_leaf` case 4 (no team, no pending roll, not game over) asks the NN about a state
type never present in training.

**How.** Add a counter next to the existing `BLOOD_MCTS_STATS` registry counters, run 20 games.

**Read.** Below 0.1% of forwards: close. Otherwise advance through them in `apply_action`'s
quiescent loop.

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

## Order and cost

| step | machine time | needs |
|---|---|---|
| D6, D10 | minutes | shards, logs |
| D1, D2 | ~20 min script + one pass over gen05-07 | shards |
| D5 | 10 min | `.pt` files, train logs |
| D3 | 15 min | one prepared dir |
| D4 | ~30 min | gen03 net, 20 states |
| D8 | 20 min | one counter |
| D9 | ~2 h | test code |
| D7 | ~75 min GPU | `arm_init.pt`, D1 pool, holdout |

Write results into this file under each item, then update the ranking in plan 032.
