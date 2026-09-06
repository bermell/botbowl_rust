# How many MCTS iterations does the training data actually need?

**Status:** **SUPERSEDED 2026-09-06.** Both headline findings have since been overturned by re-measurement, exactly as this plan's own provisional caveat anticipated.

- *"The search does not converge; run-to-run disagreement increases with budget."* **Refuted** by plan 028 Stage 0 on current code: signal to a 16000-iteration reference falls monotonically 0.6145 -> 0.0958, top-1 agreement climbs 0.45 -> 0.91, value precision improves 13x. The non-convergence was the pre-`e107f06` search, not the algorithm.
- *"Past ~500 iterations extra compute makes the policy target worse."* **Refuted** by plan 027: strength improves 250 -> 1000 at 0.700 (p~0.002), and plan 028 shows top-1 still climbing to 16000.

What survives, and it is the durable contribution: the *method*. Snapshot one long run at checkpoints, measure distance to a deep reference rather than to the previous checkpoint, and use run-to-run distance as a noise floor so "converged" is a threshold rather than an eyeball judgement. Plan 028 reused it unchanged. The ensemble result (avg of 2x500 beating 1x1000 on label TV) has not been re-checked post-fix and should not be assumed to survive either.

<details><summary>Original status (2026-08-30)</summary>

**Status:** **Run 2026-08-30 — see Results below. The headline outcome is the fourth row of the outcomes table: the noise floor is large, and it *grows* with budget.** (drafted 2026-08-29, while the plan-022 loop runs gen02). Needs the machine idle — it competes directly with generation for CPU. Cheap once it runs: ~40 min at 4-way parallelism for the headline number.

</details>

`MCTS_ITERS=1000` has been the generation budget since plan 020 and has never been justified by measurement. It is the single largest lever on generation cost — wall time is very close to linear in it — so being wrong in either direction is expensive:

- **Too high** → we burn compute deepening a search whose output stopped changing. At fixed total compute that is games we did not play, i.e. corpus diversity given up for nothing.
- **Too low** → every policy and value target in the corpus is noisy, and no amount of data volume fixes a biased label.

This plan measures which it is, directly, by studying how the search output converges as a function of iterations.

There is circumstantial reason to expect the budget is already past its useful point: plan 020 measured TD visibility as **flat (≈2.3%) across a 16× iteration budget** at this tier, and found 16k iters "not worth it" on wall-clock grounds alone. That was a downstream symptom; this measures the mechanism.

## The core idea

The policy target *is* the search's visit distribution, so we are not measuring a proxy — we are measuring the training label itself. Run a long search from a fixed state, snapshot the root statistics at increasing checkpoints, and ask when the distribution stops moving.

Two design choices make this cheap and honest:

**Snapshot one long run; do not run N separate budgets.** A single 32k-iteration search snapshotted at 100/200/500/1k/2k/4k/8k/16k/32k yields the whole curve for the price of the longest run. Running nine separate searches would cost ~2× more and, because the search is not reproducible (below), would confound budget differences with run-to-run noise.

**Measure distance to a reference, not to the previous checkpoint.** Successive differences `d(π_t, π_{t-1})` are the intuitive metric and they mislead: a distribution drifting slowly and steadily looks converged at every individual step. Use the 32k distribution as the reference and plot `d(π_t, π_32k)`.

## The calibration that makes the result meaningful

Plan 020 recorded, as a gotcha, that **MctsBot games are not reproducible from seeds** — `recon_mcts`'s std `HashMap`s randomise tie-break order per process. Two searches of the *same* state at the *same* budget genuinely differ.

That is normally an annoyance. Here it is the instrument: it supplies a **noise floor**, and turns "nothing major happens after X" from an eyeball judgement into a threshold.

Run each state **R ≥ 3 times independently**. At each checkpoint compute both:

- **signal** — `d(π_t, π_ref)`, distance from the checkpoint to the reference, and
- **floor** — `d(π_t^{(i)}, π_t^{(j)})`, distance between two independent runs at that same checkpoint.

**X\* is the smallest checkpoint at which signal falls to the floor.** Beyond it, extra iterations produce changes indistinguishable from the search's own nondeterminism — which is exactly the operational definition of "nothing major happens", and it cannot be argued with by staring at a curve.

A useful sanity property: the floor is itself informative. If the between-run distance at 1000 iters is already large, the current corpus's labels carry that much noise regardless of anything else.

## What to measure (four quantities, not one)

### 1. The policy target — via `targets::policy_target`, **not** raw visits

`botbowl-nn/src/targets.rs:1-19` documents the trap: `recon_mcts` **freezes a child's visit count the moment its subtree is solved**, and the fastest-solving child is often the *best* move ("a touchdown solves in ~10 descents while mediocre siblings keep accruing"). So `π ∝ visits` is "actively wrong", and a raw-visit distribution will appear to converge for reasons that have nothing to do with search saturation.

Use `policy_target(sample, solved_root)` — the same construction the trainer consumes, including the partially-solved hybrid rule. Metric: **total-variation distance** (½·L1) over children, which is bounded in [0,1], symmetric, and has no zero-support pathology (unlike KL).

### 2. The root value — probably the more decision-relevant curve

Plan 020's hybrid experiment concluded the **value head is the bottleneck and learned priors are not** (`nn-value` 0.83 vs full-`nn` 0.75 TDs/game — "the gen-0 value head actively steers the search away from scoring lines"). If value quality limits net strength, `root_value` convergence matters more than policy convergence. It is on a different scale (Home-centric i64, ×1000) and converges on its own schedule. Metric: `|v_t − v_ref|` rescaled to the [-1,1] value-target domain.

### 3. The played action — a *different* quantity from the policy target

`botbowl-data/src/lib.rs` notes `chosen_action` is chosen "by best aggregated Q, **not** most-visited". So the move the generator actually plays converges on argmax-mover-`Q`, while the training label converges on the visit-based construction. These are different statistics of the same tree and need not stabilise together. Track top-1 agreement with the reference for both. Expect argmax to stabilise **earlier** — it only needs the ordering of the top two children, not the shape of the whole distribution.

This matters for interpretation: if argmax stabilises at 200 but the policy target needs 2000, then the *games* are as good at 200 while the *labels* are not — and which you optimise for depends on whether you are generating trajectories or targets. For this corpus we are doing both at once.

### 4. When `root_solved` fires

The search already stops early on a solved tree (`dynamics.rs:1119-1122`: `if tree.is_solved() { break; }`). For states that solve, every iteration budgeted past the solve point is definitionally free — so the *distribution* of solve iteration counts tells us how much of the budget is already being returned unused, and how much of the corpus is in the `SolvedRootPolicy` path at all.

## Experimental design

**States.** Sample from the *same* generator the corpus uses — `botbowl-curriculum/src/random_start.rs`, `--mode random-start`, with the plan-021 tuned biases (the CLI defaults) — at seeds **disjoint from every corpus** (the loop uses `10_000_000 + G·10⁶ + K·10⁵`; take a base well outside, e.g. `90_000_000`). Anything else measures convergence on a state distribution we do not train on.

**N ≈ 50 states, R = 3 repeats, checkpoints 100…32k.** 50 is enough to see the shape of the distribution of X\* and to stratify coarsely; it is not enough for tight per-stratum confidence intervals, which is fine — the decision is "is 1000 roughly right", not "is it 900 or 1100".

**Stratify** when reporting, because a single X\* is a compromise over a heterogeneous population. Expect a near-TD position with one obvious move and an open-midfield position with 40 legal actions to converge decades apart. Report X\* by:
- number of legal root actions (the dominant driver),
- turn number within the drive (clock pressure sharpens the tree),
- whether the root solved at all.

**Evaluator: `nn-value` with the current champion**, matching generation exactly. Convergence depends on the value function guiding the search — a better net converges faster and to a different place. Also run a `heuristic` arm as a control, since 3/8 of every generation's shards are heuristic and may want a different budget.

**Workers: 1**, matching generation (`train_loop.sh` passes no `--mcts-workers`). Note the budget is split across workers (`dynamics.rs:1104-1108`), so this experiment's conclusions are per-worker-count and would need redoing if plan 024's Stage 4 changes it.

## The decision rule, stated in advance

Fix this before looking at the data, so the result cannot be rationalised after the fact:

> Let X\*(s) be the smallest checkpoint where signal ≤ floor for state s, on the **policy-target TV distance**. Adopt **X = the 90th percentile of X\*(s)** across the sampled states.

The 90th percentile, not the median: under-searching a hard state biases its label, and biased labels are worse than fewer labels (plan 021: "label frame purity beats label volume" — 152k drive-bounded samples beat 520k full-game ones). Report the median too; the **gap between median and p90 is the argument for an adaptive budget** below.

## Outcomes and what each would mean

| Result | Reading | Action |
|---|---|---|
| p90 X\* ≪ 1000 (say ≤ 300) | The budget is mostly wasted | Cut it; reinvest in games/generation. Confirm with the ablation. |
| p90 X\* ≈ 1000 | 1000 was a good guess | Leave it; stop wondering. Still worth the adaptive check. |
| p90 X\* ≫ 1000 | Labels are noisy — a real finding, and it partly indicts every corpus to date | Raise it, or accept noisier labels deliberately, but stop treating existing value targets as clean. |
| Floor is large at every checkpoint | Search nondeterminism dominates | Convergence is unanswerable as posed; the label noise floor becomes the headline result and the question becomes how to reduce it. |

That last row is the one to watch: it is a real possible outcome, not a hedge, and it would redirect the work entirely.

## Implementation note: separate runs, not snapshots (deviation, 2026-08-30)

The plan above specifies snapshotting **one** long search at checkpoints, for
~2x less compute. The implementation instead re-searches the same state once per
budget, via the existing public `MctsBot::get_action_with_record`
(`botbowl-mcts/src/dynamics.rs:1329`). Reasons:

- Snapshotting requires threading a checkpoint sink through `run_search`'s
  marker macro in `dynamics.rs` — the file that carries the recombination-purity
  and plan-013 deadlock invariants. Separate runs need **zero** changes to
  `botbowl-mcts`.
- The statistical cost is nil *because of how the decision rule is defined*.
  Signal is measured between independent runs anyway (checkpoint run `i` vs
  reference run `j`, `i != j`), and so is the floor (reference run `i` vs `j`).
  Both carry exactly one unit of run-to-run variance, so the crossing point is
  unbiased. Snapshotting would have made signal *correlated* with the reference
  (same tree) while the floor stayed independent — which would actually have
  biased the comparison in favour of early convergence.
- The compute cost is the sum of the budget ladder rather than its max: 31,800
  vs 16,000 iterations per (state, repeat), i.e. ~2x. At ~40 min for the whole
  sweep that is affordable.

The reference budget is **16k** (16x the operating point), not 32k, which keeps
the sweep inside the ~40 min estimate.

## Implementation sketch

A new read-only subcommand — `botbowl-ui convergence` — beside `dataset` and `eval` (`botbowl-ui/src/cli.rs`). It should **not** touch `MctsBot`'s production path.

Per (state, repeat): build the tree once, then step it in segments between checkpoints, extracting root statistics at each boundary. The extraction already exists — `dynamics.rs:1340-1376` builds `children: Vec<ChildStat>` plus `root_value`/`root_visits`/`root_solved` from a tree snapshot. Factor that into a function callable mid-search rather than only at the end; that is the whole of the required engine-side change, and it is a pure refactor with no behaviour change (guard it with a test that the end-of-search `Sample` is byte-identical before and after).

Output one JSONL row per (state, repeat, checkpoint) carrying the full `ChildStat` vector, `root_value`, `root_visits`, `root_solved` and the stratification keys. **Persist the raw stats, not just the computed distances** — the metrics above are all recomputable offline from that, and a second question ("what about top-3 agreement?") should not require re-running the search. Analysis in `scripts/` with numpy, matching how `eval_summary.py` already post-processes `report.json`.

**Cost.** ~1.95 ms/iteration under `nn-value` at this tier (2.6 ms/forward × ~0.75 forwards/iteration, both measured — see plan 024). 32k iterations ≈ 62 s per run; 50 states × 3 repeats ≈ 2.6 h single-threaded, **≈ 40 min at 4-way parallelism**. Add the heuristic control at roughly a tenth of that.

## Results (2026-08-30, 52 states x 3 repeats, 14x7, gen01 champion)

Ran both arms: `nn-value` with `bbnet_14x7_gen01.onnx` (48 min) and a `heuristic`
control (7 min). Raw data in `runs/convergence/`, analysis in
`scripts/convergence_summary.py`. **Raw data deleted 2026-09-01** along
with the rest of the plan-023 investigation's runs — this measurement
used the `gen01` champion under the pre-`e107f06` buggy search, and
`gen01` itself has since been retired for learned side-miscalibration
(see plan 023's postscript), so the non-convergence finding below should
be treated as provisional until re-checked post-fix.

**The search does not converge. Run-to-run disagreement *increases* with budget.**

| budget | TV between runs | top-1 agree | peak share | | TV (heur) | top-1 (heur) |
|---|---|---|---|---|---|---|
| 100 | 0.193 | 0.59 | 0.449 | | 0.126 | 0.60 |
| 200 | 0.218 | 0.64 | 0.544 | | 0.179 | 0.56 |
| **500** | 0.257 | **0.67** | 0.629 | | 0.220 | **0.64** |
| **1000** *(current)* | 0.287 | 0.65 | 0.667 | | 0.239 | 0.60 |
| 2000 | 0.339 | 0.56 | 0.664 | | 0.258 | 0.53 |
| 4000 | 0.308 | 0.57 | 0.686 | | 0.295 | 0.53 |
| 8000 | 0.329 | 0.54 | 0.716 | | 0.325 | 0.56 |
| 16000 | 0.383 | 0.55 | 0.742 | | 0.377 | 0.54 |

Read the columns together — that is where the finding is:

- **Peak share rises monotonically** (0.45 -> 0.74): more search concentrates the
  visit distribution onto a single action. The search gets *more confident*.
- **Top-1 agreement peaks at ~500 and then falls** (0.67 -> 0.55): it does not
  get more *right*, it gets more confidently *different*.
- **TV between independent runs therefore grows** (0.19 -> 0.38).

At 16k, only **21 of 52 states** have all three repeats picking the same top
action, while the mean peak share is 0.742. Sharper labels, less reproducible.

**It is the search, not the net.** The heuristic control shows the same shape,
so this is not the gen-0/gen-1 value head being miscalibrated. The mechanism is
near-tied alternatives plus PUCT's winner-take-all dynamic: with genuinely equal
values, whichever child takes an early lead accumulates the rest, and the lead is
decided by `recon_mcts`'s per-process `HashMap` tie-break order (plan 020's
non-reproducibility gotcha). More iterations *amplify* an arbitrary early lead
rather than resolving it. This is the mechanism behind plan 020's observation
that "84.8% of decisions have all-tied children Q".

**The value estimate is stable.** `|dv|` between runs sits at 0.03-0.05 across
every budget, flat. So the *value* signal is reproducible; only the *policy*
label is not. Given plan 020 already found the value head to be the bottleneck,
and the value target is the drive outcome rather than `root_value`, the noisy
quantity is the one we were least relying on — but it is also the one the policy
head is trained on directly.

**Zero roots solved** in 52 states at any budget, so the `SolvedRootPolicy` path
is irrelevant for random-start states at this tier.

### What this means for `MCTS_ITERS`

The original hypothesis — "the distribution converges early, so we are wasting
compute" — is **refuted, but the conclusion survives in a stronger form**: the
distribution never converges, and past ~500 iterations extra compute makes the
policy target *worse* as a training label (equally accurate top-1, more
confidently wrong, higher variance).

**Recommendation: `MCTS_ITERS` 1000 -> 500.** Top-1 reproducibility is
equal-or-better (0.67 vs 0.65 nn-value; 0.64 vs 0.60 heuristic), TV noise is
lower, and generation cost halves. That is a ~2x speedup for a constant change,
available before any of plan 024's engineering.

**Caveat that this experiment cannot settle:** a lower budget also means weaker
*play*, so the trajectory distribution changes. Label reproducibility is not the
only axis. The equal-total-compute ablation below is what decides it; this
experiment says where to aim it (500 vs 1000, not 4000).

### Follow-up questions this raises

1. **Average k independent short searches instead of one long one — measured,
   not speculative (2026-08-30).** At identical compute, averaging the visit
   distributions of 2x500 beats a single 1x1000, against a held-out third run:

   | label recipe | cost | TV vs held-out | top-1 vs held-out |
   |---|---|---|---|
   | 1 x 500 (single) | 500 | 0.2701 | 0.65 |
   | **avg of 2 x 500** | **1000** | **0.2374** | **0.68** |
   | 1 x 1000 (single) | 1000 | 0.2818 | 0.67 |

   ~16% less label noise for the same compute. **Why more iterations cannot do
   this:** PUCT is self-reinforcing — among tied children, whichever takes an
   early lead attracts more visits, widening the lead. Tie-break noise is
   *amplified* by depth, not averaged away. It is a Polya urn: running one urn
   longer converges to a *random* limit, not the mean. Within a tree, iterations
   are positively correlated (same early lead); across trees they are
   independent, so averaging cuts variance as 1/k.

   The independence assumption is confirmed by the numbers: if noise were fully
   independent, avg-of-2 vs a held-out single should sit at sqrt(0.75) = 0.866 of
   the single-run distance, predicting 0.2339 against 0.2374 measured — within
   1.5%. Note the gain is in distribution *shape* (TV 0.282 -> 0.237), not argmax
   (0.67 -> 0.68), which is what the cross-entropy policy loss actually consumes.

   Untested: play strength (two 500-searches may pick worse moves than one
   1000-search), the interaction with anchor-gated tree reuse
   (`dynamics.rs:1051`), and the optimal k. Implementation is contained — call
   `get_action_with_record` k times in the dataset generator and merge child
   visits by action; `botbowl-mcts` is untouched.

   A deterministic tie-break would attack the same root cause differently and is
   worth comparing.
2. **Should the policy target be softened?** A sharp label that is 45% likely to
   name a different action on a re-run may be worse than an explicitly softened
   one. Temperature on the visit distribution is a one-line change with a
   measurable effect on `val_top1`.
3. **What is the label-noise ceiling on `val_top1`?** Two independent labels
   agree ~0.65 of the time at 1000 iters. If the label distribution has modal
   probability `p`, pairwise agreement is `sum p_i^2` and the best achievable
   predictor accuracy is `max p_i` — so ~0.65 pairwise implies a ceiling
   somewhere around 0.75-0.80. Training currently reaches `val_top1` ~0.50, so
   there is real headroom; the net is **not** yet at the ceiling. Worth
   measuring properly with more repeats before investing in policy-head capacity.

## Follow-ons this sets up

**Adaptive budgets — potentially the bigger win.** If X\* is broadly distributed (a large median-to-p90 gap), then *no* fixed budget is efficient: it over-searches the easy majority to serve the hard tail. The natural answer is a dynamic stopping rule — stop when the policy target has been stable for K checkpoints — which spends the budget where it changes the label. Prior art: KataGo's playout-cap randomisation, which additionally exploits the fact that only a fraction of positions need a full-strength target at all. This experiment produces exactly the data needed to design that rule, and its own noise floor supplies the stability threshold.

**The confirming ablation.** Convergence is a *screen*, not the final answer. The real question is compute allocation: at fixed total compute, a lower budget buys more games. Plan 020 found data *shape* beat data *scale*, and plan 021 found label *purity* beat label *volume*, so this trade has repeatedly not gone the intuitive way. Settle it with two corpora at **equal total compute** (e.g. 4800 games @ 1000 iters vs 9600 @ 500), each trained with best-val restore and compared on the standard rungs. That is one extra generation of wall time, and it is worth spending only once the convergence screen says where to aim.

## Caveats

- **The answer has a shelf life.** Convergence depends on the value function, so it moves as the champion improves. Treat X as a per-generation check (cheap enough to re-run), not a constant to be baked in.
- **It is board-tier-specific.** 14x7/4-player only; every tier needs its own measurement, and plan 017's larger tiers will differ substantially.
- **Non-reproducibility is load-bearing here.** If `recon_mcts` ever becomes deterministic, the floor collapses to zero and the decision rule needs replacing with an explicit tolerance.
- **Solved states may dominate the easy tail** and drag the median down while telling us nothing about the hard states that matter. This is why the rule uses p90 and why solvedness is a stratification key.

## Cross-references

- plan 020 — the flat-TD-visibility-across-16×-budget result that motivates this; the value-head-is-the-bottleneck finding; the cost table.
- plan 021 — label purity over label volume (why p90, not median); the drive-bounded corpus and tuned biases this samples states from.
- plan 022 — the loop whose `MCTS_ITERS` knob this sets.
- plan 024 — measured per-forward and per-iteration costs; note its Stage 4 would change the worker count and invalidate this measurement's premise.
- `botbowl-nn/src/targets.rs` — the solved-aware policy target construction that must be used as the metric.
