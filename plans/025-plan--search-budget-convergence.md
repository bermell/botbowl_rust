# How many MCTS iterations does the training data actually need?

**Status:** Proposed (drafted 2026-08-29, while the plan-022 loop runs gen02). Needs the machine idle — it competes directly with generation for CPU. Cheap once it runs: ~40 min at 4-way parallelism for the headline number.

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

## Implementation sketch

A new read-only subcommand — `botbowl-ui convergence` — beside `dataset` and `eval` (`botbowl-ui/src/cli.rs`). It should **not** touch `MctsBot`'s production path.

Per (state, repeat): build the tree once, then step it in segments between checkpoints, extracting root statistics at each boundary. The extraction already exists — `dynamics.rs:1340-1376` builds `children: Vec<ChildStat>` plus `root_value`/`root_visits`/`root_solved` from a tree snapshot. Factor that into a function callable mid-search rather than only at the end; that is the whole of the required engine-side change, and it is a pure refactor with no behaviour change (guard it with a test that the end-of-search `Sample` is byte-identical before and after).

Output one JSONL row per (state, repeat, checkpoint) carrying the full `ChildStat` vector, `root_value`, `root_visits`, `root_solved` and the stratification keys. **Persist the raw stats, not just the computed distances** — the metrics above are all recomputable offline from that, and a second question ("what about top-3 agreement?") should not require re-running the search. Analysis in `scripts/` with numpy, matching how `eval_summary.py` already post-processes `report.json`.

**Cost.** ~1.95 ms/iteration under `nn-value` at this tier (2.6 ms/forward × ~0.75 forwards/iteration, both measured — see plan 024). 32k iterations ≈ 62 s per run; 50 states × 3 repeats ≈ 2.6 h single-threaded, **≈ 40 min at 4-way parallelism**. Add the heuristic control at roughly a tenth of that.

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
