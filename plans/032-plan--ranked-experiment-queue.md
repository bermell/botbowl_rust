# Ranked experiment queue and open questions

**Status:** Live (written 2026-09-07 from the methodology audit). This is the single place the
open programme lives; completed plans point here. Ranking is by expected strength gain per hour
of machine time, not novelty. Re-rank after plan 031's diagnostics land — the "gate" column says
which diagnostic can demote an item before it costs any games.

Ground rules carried over from plans 027/029, **as corrected by plan 031 D10 (2026-09-07)**:

- Paired Home/Away on a shared seed base, points (W + D/2)/N, one variable per arm, commit before
  launching, results decided by games not by `val_*`.
- **SE ≈ 0.041 at 120 games, paired or unpaired.** The measured pair correlation is **zero**
  (`r = 1.003`, 95% CI [0.962, 1.060] over 813 games in seven per-game logs), so the old
  "~0.045 after pair correlation" inflation is dropped and every SE and z quoted in plans 027/029
  stands as written. Score with `scripts/paired_summary.py` anyway — pairing is free and removes
  the side-bias term from the point estimate; it just buys no variance.
- **Use one shared seed base across arms that will be differenced against each other.** Plan 029's
  four `exp-data` arms used disjoint bases (96/97/98/99 M), so no cross-arm contrast could be
  differenced per seed. That is where the remaining variance reduction actually lives: the
  *situation* term is shared across arms even though it is not shared within a Home/Away pair.
- **Always pass `--per-game-out`.** Nine of the twelve `runs/exp-search/` arms, all of
  `runs/exp-priors/` and `runs/exp-conv/`, and every `runs/loop14x7/gen*/report.json` rung could
  not be re-analysed at all for lack of one.
- **Size against real power, not against the 120-game floor.** At `V_pair = 0.1000` (5%
  two-sided, 80% power): detecting 0.55 needs **628 games**; 0.56, 436; 0.58, 245; 0.60, 157. At
  120 games a match has 80% power only against ~0.60 or larger. The pre-committed extension rule
  still holds — a match in [0.53, 0.58] gets 120 more games on the same pair, not a claim — but
  note that 240 games reaches only SE ≈ 0.029, so **a true 0.55 effect stays undecided even after
  the extension**. Any item whose expected effect is ±0.03 must be resized to ~600 games or
  dropped rather than run at 120 and called ambiguous.

## Re-ranking after plan 031 (2026-09-07)

The diagnostics landed and moved four items. The order below is the new one; each item's own
section carries the evidence.

| was | now | item | why it moved |
|---|---|---|---|
| 1 | **1** | D7 vs champion gen03 | unchanged — no gate, both nets exist, answers a strategic question cheaply. **#1(b) (lr 1e-3 warm arm) is dropped**: D5 shows 2e-4 already moves the net *further* than the last fine-tune that won a rung. |
| 2 | **2** | Mean backup instead of minimax | **mechanism now measured, not argued.** D1's follow-up ran the generating net over the same 18,990 rows with no search: bare leaf gap +0.031, search gap +0.132, **search adds +0.1008 (SE 0.0019, z=52)**. 76% of the optimism is made by the backup. Its demotion gate is not met. |
| 7 | **3** | Q-informed policy target | **promoted two places.** D2 found a label defect much larger than the tie rate it was gated on: top-1 agreement between the move actually played and the visit label is **0.588 overall and 0.22-0.25 above 30 children**. |
| 3 | **4** | Re-tune `PUCT_C`, add FPU reduction | supported, but its gate was mis-specified and is **replaced** — see the item. The measured change is the prior's dynamic range, not a visit-entropy collapse. |
| 11 | **5** | Residual side bias in NN full games | **unblocked**: D9 is green at every budget including production 1000, so the mirror games can now be read as a variance question rather than a possible bug hunt. |
| 4 | 6 | Root Dirichlet noise + visit-temperature | **gate failed on both limbs** — nn tie rate is 13.5% not >30%, and π entropy is flat. Also: use a **per-root α = 10/n_legal**, not a constant; the root fan distribution is bimodal (median 6, p90 73). |
| 6 | 7 | Heuristic hedge ablation | gate cleared (33% of samples ≫ 20%), but **must be resized**: D10 says a ±0.03 effect needs ~600 games, not 120. |
| 5 | 8 | Training-seed variance | unchanged in kind, same resizing problem as #7. |
| 9 | 9 | Capacity at fixed data | unchanged. |
| 10 | 10 | High-budget strength | unchanged; still best run after #2 changes the backup. |
| 8 | **11** | Encoder additions | **demoted and rewritten.** D3: the trigger fires numerically (12.78% ambiguous) but **zero** of 22,343 ambiguous groups are distinguished by action type, and the measured cost is 0 for both heads. |

## The queue

### 1. D7 vs champion gen03 — does a from-scratch retrain on the whole corpus beat eight incremental generations?

- **What is already known (plan 029 stage 3, 2026-09-07).** The warm-started pair ran: W3 vs W1
  = **0.487** (z=−0.30) against D3 vs D1 = 0.600 from scratch. The from-scratch data curve does
  not transfer to the loop's `--init gen03.pt --lr 2e-4` regime. W1 restored at step **2,500**
  and W3 at 20,000, so a low-LR warm start extracts a small, quickly exhausted amount from any
  pool; the plateau is at least as consistent with the incremental regime as with data starvation.
- **Mechanism.** If D7 (from scratch, gen01-07, 0.662 vs D1) beats gen03 (eight incremental
  generations), the loop should periodically retrain from scratch on the accumulated corpus
  rather than only fine-tune forward — a bigger change than any window setting.
- **Arms.** (a) `d7 vs gen03`, 120 games at production settings (`--evaluator nn`, 1000 iters);
  both `.onnx` exist. ~~(b) a warm start at `--lr 1e-3` vs gen03.~~ **(b) is dropped (plan 031 D5).**
  Its premise was that 2e-4 is too small a step. It is not: at the restore point gen06 has travelled
  **1.70×** and gen07 **1.36×** as far from gen03 as gen02→gen03 did — and gen02→gen03 is the one
  fine-tune that actually won its rung (0.650). Raising `WARM_LR` would push the weights further
  into a region per-epoch validation already calls worse. **`WARM_LR` stays fixed.**
- **Gate.** None — runs first. Plan 031 D5 answered the weight-distance question: the fine-tunes
  move the net 20-40% of the way to an unrelated network (two independent from-scratch fits sit at
  1.26-1.94) and nothing measurable happens. That *removes* the competing explanation "the
  fine-tunes never moved" and makes (a) the cleaner test.
- **Cost.** (a) ~5 h; (b) 40 min + 5 h.
- **Decide.** D7 ≥ 0.55 vs gen03 → add a periodic from-scratch retrain (every k generations,
  `--eval-every`, `--select-on combined`) to the plan-030 rebuild. D7 ≤ 0.50 → neither data nor
  regime explains the plateau; #2/#3 move to the top.
- **Also adopt now, no experiment needed:** `--eval-every` in `train_loop.sh`. Plan 029 stage 3
  found gen05-07 all restored at epoch 0-1 of 10 under per-epoch validation, i.e. the loop has
  been shipping past-peak weights for lack of checkpoint resolution. **Done 2026-09-07:**
  `EVAL_EVERY` (default 2500 steps) is passed to the warm-start `bbnn.train` call; the gen00
  bootstrap call is unchanged.
- **Run (launched 2026-09-07 21:51, `scripts/exp032_s1_d7_vs_gen03.sh`, out `runs/exp032/`).**
  `runs/exp-data/d7.onnx` vs `models/bbnet_14x7_gen03.onnx`, 120 games, seed base 32000000,
  6 parallel, GPU sidecar (`nn_server.py --device cuda`), `--per-game-out`. Pace: 12 games in
  32 min → ~5.3 h. Result: pending.

### 2. Mean backup instead of minimax with an NN leaf

- **Mechanism.** Player-node backup is max/min over child Q (`dynamics.rs`, `backprop_scores`);
  with a noisy learned leaf this is a max-over-noise estimator and it also sets FPU (= parent Q)
  to the most optimistic sibling. Averaging (visit-weighted child mean, AlphaZero-style) is the
  standard remedy. Chance nodes already average.
- **Change.** `BackupMode::{Minimax, Mean}` on `BloodBowlDynamics`, env/flag plumbed like
  `PuctMode`. Keep the known-outcome carve-out (exact ±1000 leaves). Note the mean changes the Q
  scale FPU and `PUCT_C` see, so run #3's `c` sweep on the winner.
- **Gate — cleared, and the mechanism is now measured (plan 031 D1 + follow-up).** The demotion
  gate was "calibration on the diagonal at every fan width". It is not met: the search is optimistic
  by **+0.133 to +0.140** of a drive outcome in the mover's frame, stable across gen05/06/07. More
  importantly, D1's follow-up separates the two candidate causes that D1 alone could not. Running
  the *generating* net over the same 18,990 rows with **no search at all**:

  | on identical rows | gap vs drive outcome | SE |
  |---|---|---|
  | bare NN leaf value | **+0.0311** | 0.0045 |
  | 1000-iteration search root | **+0.1319** | 0.0045 |
  | **added by the search** (paired) | **+0.1008** | 0.0019 (**z = 52**) |

  The leaf is nearly calibrated (slope 1.056, i.e. slightly *steeper* than the diagonal).
  **76.4% of the search's optimism is manufactured by the backup**, which is exactly this item's
  thesis. **Replicated on all four generations of the frozen gen03 net (83,432 rows): added
  +0.0908 / +0.0865 / +0.0944 / +0.1008 for gen04-07, every one at z ≈ 50.** gen04 used
  **scripted** priors and shows the same +0.091, so this is not a prior artefact — swapping the
  entire prior source moves it less than its spread across generations. Note the plan-031 prediction of *monotone* flattening with fan width was wrong and does
  not count against the item: the added optimism peaks at **intermediate** fan width (+0.220 at
  11-30 vs +0.066 at >60), which is what max-over-noise predicts once you account for fan width
  confounding with per-child depth at a fixed budget.
- **Cost.** ~half a day of code + tests (existing `backprop_*` unit tests pin minimax; add mean
  variants), 120 games ~5 h.
- **Expected.** +0.05 to +0.10. **Pre-commit this prediction:** a mean backup should remove most of
  the measured +0.101, and `scripts/audit_value_head_bias.py` re-run on a mean-backup corpus is the
  cheap check *before* spending the 120 games. If the optimism does not fall, the mechanism is
  something else and the games are not worth playing.
- **Abandon.** Mean ≤ 0.50 vs minimax at 120 games.
- **Implemented 2026-09-07** (`BackupMode::{Minimax, Mean}` on `BloodBowlDynamics`; env
  `BLOOD_MCTS_BACKUP=mean`, `MctsBot::with_backup`, `eval --backup/--vs-backup`; `dataset` reads
  the env and stamps `backup=mean` into `budget_label`). The mean is the visit-weighted mean of
  child Q in `i128` integer arithmetic (plain mean of scored children if no child has visits), so
  it stays deterministic and **mirror-exact**: `tests/mirror_search_exact.rs` gained mean-backup
  arms at budget 20 (default suite) and 200 (heuristic + NN, `#[ignore]`), all green. Unit tests
  pin order-independence and the ±negation symmetry.
- **Run (queued behind stage 1, `scripts/exp032_s2_mean_backup.sh`).** Part A: two 150-game
  random-start corpora on seed 32100000, `BLOOD_MCTS_BACKUP=minimax|mean`, each through
  `audit_value_head_bias.py` — the pre-committed check that mean backup pulls the search-added
  optimism from +0.10 toward the bare leaf's +0.03. Part B: `gen03 --backup mean` vs `gen03`
  minimax, 120 games, seed base 32000000. Result: pending.

### 3. Re-tune `PUCT_C` under learned priors; add FPU reduction

- **Mechanism.** NN priors are softmax×len, so the exploration term's spread is ~3× the scripted
  one at the top and ~0 in the tail; `c=10` was tuned for scripted priors and plan 026's sweep
  used the heuristic evaluator. FPU has no reduction, so the first sweep is breadth-first by
  construction (plan 026 §successor experiment).
- **Arms.** `--puct-c 3 | 10 | 30` under `--evaluator nn`, 120 games each vs `c=10`; then
  `fpu = parent_Q − k·√(Σ visited priors)` at one `k`, 120 games. Flags exist for `c`; FPU
  reduction is ~30 lines in `puct_value`.
- **Gate — the original one was mis-specified; here is the replacement (plan 031 D4).** The stated
  gate was "if root visit entropy under `nn` is not far below `nn-value`, drop to #6". Measured,
  entropy is only 3-10% lower — but **entropy at a fixed budget is dominated by the visit-count term
  and is insensitive to exactly the change that occurred**, so that gate tests the wrong thing.
  What D4 actually measured, on matched states with the same net and only the prior source differing:

  | | scripted (`nn-value`) | learned (`nn`) |
  |---|---|---|
  | distinct prior values (production) | **4** — `{0.2, 1.0, 5.0, 10.0}` | continuous |
  | max prior | **10.0** | **57.9** |
  | `top_prior_lift` (1.0 = uniform), matched states | **1.08** | **5.54** |
  | `top_prior_lift`, production distribution | 1.98 | 3.11 |

  On **turn-start activation roots the scripted prior is two values with 97.7% of children at
  exactly 1.0** — `priors.rs`'s positional multipliers do not apply to `Start*` actions — so on that
  whole class of root `c = 10` was tuned against *no prior shaping at all*. **The gate is now: the
  prior went from a 4-level ladder capped at 10.0 to a continuous distribution reaching 57.9, and
  `c` has never been re-tuned for it. That is met.**
- **What D4 does *not* support.** The search is not "following the prior" in the collapsed sense:
  8-10% of children sit at ≤1 visit (not "most"), and argmax-Q == argmax-prior in 20-42% of roots
  (not "most"). So expect a tuning gain, not a rescue.
- **Cost.** ~15 h of games total.
- **Expected.** +0.03 to +0.06 (plan 026 saw c=30 at 0.593 under the heuristic, p=0.08). Note D10's
  power table: a +0.03 effect is **not resolvable at 120 games** — size the `c` sweep accordingly or
  treat it as a screen whose winner then gets a properly-sized match.
- **Abandon.** All arms within 1 SE of each other.
- **Implemented 2026-09-07.** FPU reduction in the Leela/KataGo form, `parent_Q − k·√(visited
  prior share)` (`BloodBowlDynamics.fpu_reduction`, env `BLOOD_MCTS_FPU_REDUCTION`, `eval
  --fpu-reduction/--vs-fpu-reduction`); `k = 0` is byte-identical to the shipped plain FPU. The
  prior share is accumulated in `f64` in a fixed child order, so it is mirror-exact too (verified
  at k=300 via `BLOOD_NN_MIRROR_FPU_K`). `--puct-c/--vs-puct-c` already existed.
- **Run (queued behind stage 2, `scripts/exp032_s3_puct_fpu.sh`).** Four 120-game screens, gen03
  both sides, seed base 32000000, backup rule for all arms chosen from stage 2 (mean if ≥ 0.55,
  else minimax): `c=3 vs c=10`, `c=30 vs c=10`, `k=100 vs k=0`, `k=300 vs k=0`. Screens only —
  a winner gets a sized match. Result: pending.

### 4. Exploration in self-play: root Dirichlet noise + visit-temperature sampling for the first k decisions of a drive

- **Mechanism.** Generation is argmax-Q with no noise (`pick_best_action`); diversity comes only
  from random starts and dice. The value head never sees what non-greedy alternatives were worth
  and the policy target never sees a root that was pushed off its prior.
- **Change.** In `dataset` only (never `eval`): Dirichlet(α) mixed into root priors with
  α ≈ 10 / mean legal actions (plan 031 D4 gives the number), ε=0.25; sample the played action
  ∝ visits for the first k=2 decisions of a drive, argmax after. Apply noise only when
  `state == root_state` — a per-search constant like the horizon anchor, so recombination purity
  holds and the cached `prior_bits` in `BbAction` never differ between paths.
- **Gate — FAILED on both limbs (plan 031 D2), so this is not promoted.** The gate was "a high tie
  rate *or* falling π entropy promotes this above #3". The nn tie rate is **13.5 / 13.3 / 13.8%**,
  not >30% — plan 028's "84.8% tied roots" was measured on 8x3 pure-td and does not carry over. And
  π entropy is **flat** across gen05→07 (1.211 / 1.221 / 1.212), necessarily so: **all three
  generations were generated by the same frozen `bbnet_14x7_gen03.onnx`**, so that comparison was
  vacuous. The one real entropy fall in the data is the gen04→gen05 regime switch (scripted →
  learned priors: ρ 0.364 → 0.435, H(π) 1.261 → 1.211), not a within-regime trend.
- **Correction to the α recipe.** D4's production root fan distribution is **mean 20.2, median 6,
  p10 2, p90 73, max 98** — bimodal. A single α = 10/mean = 0.49 fits neither mode; use a
  **per-root α = 10/n_legal** (KataGo-style).
- **Where the ties actually are.** The heuristic hedge, not the nn half: 58% overall and **89% at
  >60 children**, across a third of every corpus. If ties are the motivation, #7 (hedge ablation)
  addresses them more directly than root noise does.
- **Cost.** One generation A/B: two corpora from the same champion and seeds (noisy vs greedy),
  train both from the same init, play 120 games. ~14 h.
- **Expected.** Standard ingredient; modest on its own, larger once #2 removes the ties that
  make visits ∝ prior.
- **Abandon.** Noisy-corpus net ≤ greedy-corpus net at 120 games.

### 5. Training-seed variance of strength (the floor every result is compared against)

- **Mechanism.** No two nets have ever been trained on identical data with different seeds and
  played. Plan 029's +0.06 per doubling and every gate verdict are read against an unknown floor.
- **Arms.** Two D1 trains from two `--epochs 0` inits (different `--seed`), 110k steps, then
  head-to-head 120 games. Optionally a third to get a spread.
- **Cost.** ~1.2 h GPU + ~5 h games.
- **Expected.** Informational. If |Δ| > 0.05, plan 029's D7-vs-D3 step is inside noise and every
  future arm needs a replicate or 240 games.
- **Resizing (plan 031 D10).** 120 games gives SE 0.041, so this arm can only distinguish "the seed
  floor is above ~0.60" from "it is not" — which is not the question. To bound the floor at the
  ±0.05 level that would actually change how other arms are read, it needs **~300 games**; to bound
  it at ±0.03, ~600. Either run it at 300 and report a bound, or accept that it answers only
  "is the floor catastrophic?".
- **Also cheap and worth folding in:** plan 031 D5 already measured the *weight-space* floor —
  two independent from-scratch fits sit at rel-L2 **1.26-1.94** while every warm-start fine-tune
  sits at 0.19-0.41. That does not give strength variance, but it does say the two seeds will be
  genuinely different nets, not near-copies.

### 6. Heuristic hedge ablation

- **Mechanism.** One third of every corpus is heuristic-MCTS play with scripted-prior policy
  targets (plan 029 correction). It was a plan-020 hedge against a bad gen-0 net and has never
  been re-examined; it now teaches the policy head the scripted prior a third of the time.
- **Arms.** `HEUR_SHARDS=""` (all 8 nn) vs current, one generation each from the same champion,
  same seeds; train from the same init; 120 games. Also check corpus health (TDs/drive,
  scoreless %, class split) — the hedge may still be protecting value-target balance.
- **Gate — cleared (plan 031 D6).** The gate was "below 20% of samples → skip". Measured by
  samples: **33.2 / 32.9 / 33.5%** for gen05/06/07 — the plan's ~33% prediction confirmed to the
  decimal (its per-drive figures were off: 34.8 vs 28.9 measured, not 28 vs 24).
- **What D6 and D2 add to the health check.** The hedge is where two defects concentrate:
  - **Ties.** Heuristic roots tie at **58%** overall and **89% at >60 children**, against 13.5% in
    the nn half. If #6 (root noise) is motivated by ties, this item addresses them more directly.
  - **Labels.** The hedge carries ~10 points more label-0 (clock) mass (0.39 vs 0.29) and ~10 points
    less mover-`+1`, and its policy targets teach the *scripted* prior. So it is not a neutral
    diluent — it teaches something different.
  - **Selection.** Because `VAL_SHARDS="4 7"` is 1 nn + 1 heuristic while `TRAIN_SHARDS` is 4 + 2,
    the val pool is **47.5% hedge against a 29.1% train mix**. Measured, the two halves score very
    differently on `val_policy` (1.4243 vs 1.5577 for gen03) and **identically** on `val_value`
    (0.3938 vs 0.3936), and over the gen07 fine-tune the net *improved* on the nn half while
    *degrading* on the heuristic half. Removing the hedge dissolves this mismatch entirely — which
    is why fixing the split separately is listed as a low-priority cleanup, not an adopt-now item.
- **Cost.** ~14 h. **Expected.** ±0.03 — **which D10 says 120 games cannot resolve.** Size at
  ~600 games, or run it as a corpus-health study (tie rate, label split, TDs/drive, scoreless %)
  and only play games if the health numbers move. **Abandon.** Health metrics degrade.

### 7. Q-informed policy target (Gumbel / completed-Q style)

- **Mechanism.** At 1000 iterations over 30-100 children the visit target is mostly the FPU
  sweep plus prior. `π ∝ softmax(logit(prior) + σ(q_mover))` with completed Q for unvisited
  children extracts the search's *value* information into the target and is robust at low
  visit counts. `q` per child is already recorded, so this is a `targets.rs` change and a
  retrain on the existing corpus.
- **Promoted to #3 by plan 031 D2, on a statistic stronger than the one this item was written
  around.** The visit target does not merely tie — it points somewhere else. Top-1 agreement
  between **the move the bot actually played** (argmax mover-Q, `pick_best_action`'s key) and **the
  π label** (argmax visits) is:

  | fan width | top-1 agreement |
  |---|---|
  | ≤10 | 0.711 |
  | 11-30 | 0.595 |
  | 31-60 | **0.222** |
  | >60 | **0.251** |
  | **all** | **0.588** |

  On roots with >30 children the label points at a different action from the played one **three
  times out of four** — which is precisely the low-visit regime completed-Q is designed for. Note
  also that "30-100 children" describes only the top ~15% of roots (D4: median fan is 6), so the
  gain concentrates on a minority of samples; weight the expectation accordingly.
- **Gate.** Rerun plan 028 Stage 0 on 50 states (~1 h): top-1 agreement of the new target with
  the 16k reference must beat raw visits' 0.69. No gain → drop.
- **Cost.** ~2 h code + 1 h probe + 40 min train + 5 h games.
- **Expected.** +0.03 to +0.08 on the policy head; compounds with #4.
- **Gate, part 1 — offline on the existing plan-028 dump (2026-09-07, `scripts/audit_q_target.py`).**
  No search needed: `runs/exp-conv/s0a-raw-c10.jsonl` already holds 50 states × 3 repeats at
  budgets 100…16000 with per-child `prior`/`visits`/`q`, so any target built from the 1000-budget
  rows can be scored against the 16k repeats (cross-repeat pairs, 300 per row). Two references:
  *ref visits* (argmax of the 16k visit target — plan 028's statistic) and *ref played* (the 16k
  search's `chosen_action` = argmax mover-Q — what strong play actually does, and the one a
  policy head should imitate). Candidates from the same 1000-iteration root: `visits` (the
  shipped target), `argmaxq` (one-hot on the played move), `cq(τ)` = softmax(ln prior + q/τ)
  with unvisited children completed by the visit-weighted mean Q (τ in Q points, 1000 = one
  TD), `cqv(τ)` = the same with ln(visits+1) in place of the prior.

  | target | top-1 vs ref visits | top-1 vs ref played |
  |---|---|---|
  | visits (shipped) | 0.687 | 0.677 |
  | argmaxq | 0.620 | 0.660 |
  | cq(50) | 0.647 | 0.700 |
  | **cq(100)** | **0.693** | **0.740** |
  | cq(200) | 0.660 | 0.680 |
  | cqv(50) | 0.680 | 0.720 |

  Reading: completed-Q at τ=100 is the only candidate that beats the shipped target on *both*
  references, +0.06 against the played move. But 50 states means that is ≈3 states changing
  side, so it is suggestive, not a pass — and more importantly **every root in this dump has
  4-21 legal actions** (the convergence probe samples turn-start activation roots), i.e. none of
  it is in the >30-fan regime where D2 put the defect. The gate cannot be decided on this file.
- **Gate, part 2 — wide-fan probe (launched 2026-09-07 22:30, `scripts/exp032_s7_widefan_probe.sh`).**
  `botbowl-ui convergence` gained `--advance N` (play N production decisions from the random
  start so the probed root is mid-turn) and `--min-legal M` (skip roots whose *pruned* fan —
  measured with a 2-iteration root expansion, i.e. what the search sees — is below M). Run:
  200 seeds, `--advance 1 --min-legal 30`, budgets 1000 + 16000, 3 repeats, gen03, raw c=10,
  seed base 91000000. Incidental finding from the smoke run: after one activation the pruned
  fan is **bimodal** — of 40 seeds, 16 had fan 1 (pruning collapses the move fan to a single
  square), 14 had 2-22, and 10 (25%) had ≥30. So "wide fan" is a quarter of second decisions,
  not the norm, and the first decision after activation is often forced. Result: pending.

### 8. Encoder additions

- **Mechanism.** No plane for the active player's action type or blitz-block-thrown (partial
  observability, plan 031 D3); no coordinate planes (receptive field 27 covers 16 columns but
  not 28, so this matters from the 20x11 tier up); `kicking_first_half`/receiver flag absent
  (irrelevant in drive-bounded training, relevant in full-game eval).
- **Rewritten and demoted to last by plan 031 D3.** The gate ">1% of samples in ambiguous groups"
  fires — **12.78%** of gen07's 350,595 rows are byte-identical tensors with differing legal sets —
  but **the remedy this item named is wrong**, and D3's step 4 is what shows it:
  - **Zero of 22,343 ambiguous groups are distinguished by the active player's action type.** No
    `Start*` action appears anywhere in the differing-action histogram (and since 100% of groups
    have pairwise-disjoint legal sets, one that mattered would have to appear). Move-vs-Blitz never
    causes a collision. **Do not add the action-type one-hot.**
  - The real ambiguity is the **procedure stack**: adjacent phases of one block/push/reroll chain
    over an unchanged board (block-die-select → push-square → follow-up → resume moving). Max
    row-index span within a group is **3**; there are no long-range duplicates at all.
  - **The measured cost is zero for both heads.** Value-target variance within collision groups is
    **0.000000** (though partly tautological — collisions are intra-drive and the target is the
    drive outcome, so this corpus *cannot* charge the value head). And because the supports are
    100% disjoint and `train.py:91-94` takes a per-sample masked log-softmax over the gathered legal
    actions only, one shared logit tensor satisfies both members' targets **exactly and
    simultaneously** — the irreducible policy loss is 0, not merely small.
- **Change, if anything.** A ~5-value **pending-decision-phase** one-hot (block-die-select /
  push-square / follow-up / reroll-prompt / normal activation) would separate 100% of these groups.
  Rank it as a speculative capacity tidy-up with a measured payoff of **0** on this corpus, not as a
  correctness fix. The coordinate planes are unaffected by D3 and still matter from the 20x11 tier
  up (receptive field 27 covers 16 columns but not 28).
- **The one real information gap D3 could not measure, and the reason to keep this item alive at
  all:** in *search*, `value_home_i64` scores mid-block-chain leaves, and the block dice are visible
  only through the legal-action set, which the tower never sees — so a leaf after a Skull and a leaf
  after a Pow get **the same value** as the pre-block node. That is a leaf-scoring gap, it is
  invisible to a corpus whose value target is drive-constant, and it needs an online measurement.
- **Cost.** ~6 h. **Expected.** Small on 14x7, and D3 puts the corpus-measurable part at zero.

### 9. Capacity at fixed data

- **Mechanism.** BBNet is 0.48 M params; plan 024 showed a 2-3× net is nearly free on the
  sidecar. Never tried. Run after #1-#4 so the search, not the net, is not the ceiling.
- **Arms.** width 96 / blocks 8 on the D3 pool from a fresh init; 120 games vs the width-64 D3.
- **Cost.** ~1.5 h GPU + 5 h games. **Abandon.** No gain on D3 data.

### 10. High-budget strength (plan 028 C1/C2)

- 8000 and 16000 iterations vs 1000, same net, 60 games (the predicted effect is large). Only
  informs evaluation/label quality, not the loop's budget. ~15 h. Run after #2 changes the
  backup, since the flat spot may be a minimax artefact.

### 11. Residual side bias in NN full games

- Plan 027 left a pooled 56% Away share (z≈3.4 pair-corrected) with a net in the game and 50.5%
  without. **Plan 031 D9 is green and this is unblocked** — `mirror_search_exact.rs` now has
  `Evaluator::Nn` arms at budgets 5/20/200 **and at the production budget of 1000**, all exact-equality
  (root pick mirrors, `root_value` is the exact negation, every child's visits/q/solved/terminal
  agree), passing against both the committed `tiny.onnx` fixture and `bbnet_14x7_gen03.onnx`. So the
  NN search path is **not** the source: the residual is tie-break variance or turn-order.
- Run a 300-game NN mirror with `--per-game-out`, split by `kicking_first_half` and by
  `TieBreak::{Hash,Mover}`. ~8 h. Outcome is variance reduction on every benchmark.
- **One extra thread to pull, from plan 031 D6:** the corpus *labels* carry the same skew. In all
  three of gen05/06/07 the nn half is Home-frame Away-skewed (−1 mass 0.359-0.385 vs +1 mass
  0.327-0.365) while the heuristic hedge is near even. Same direction as the 56%. Since the search
  is now proven mirror-exact, that points at the *game*, not the search — turn order, kickoff, or
  the random-start generator — which narrows where to look.
- **Measured for free from the existing per-game logs (2026-09-07, `scripts/side_bias_pooled.py`).**
  No mirror match is needed for the seat term: every paired match plays each seed from both
  seats, so grouping the pooled games by *seat* instead of by candidate cancels the candidate
  effect and leaves the side bias. Pooling the five NN full-game logs that exist (`exp-data`
  s1-s4 + `exp-conv` b1, 600 games, five different nets, three seed bases):

  | games | Home points | z | TD Home:Away |
  |---|---|---|---|
  | all 600 NN | **0.438 ± 0.018** | **−3.46** | 1979:2156 |
  | Home kicked first | 0.418 ± 0.026 | −3.20 | 899:1028 |
  | Away kicked first | 0.456 ± 0.026 | −1.72 | 1080:1128 |
  | *receiving team, seat-agnostic* | 0.517 ± 0.018 | +0.96 | |
  | 340 heuristic/nn-value mirror games (`exp-search`) | 0.479 ± 0.025 | −0.83 | 757:819 |

  Every one of the five NN logs is below 0.50 (0.400-0.487). So: (1) the Away edge is real and
  the same size plan 027 saw (≈0.06); (2) it is **not** a kickoff/receive effect — receiving is
  worth +0.017 ± 0.018 and Away leads in both kick splits; (3) it is specific to a net being in
  the game (heuristic mirrors are within noise of 0.50, and plan 027's no-search games were 0.505).
  Since D9 proves the NN *search* is mirror-exact on decision states, the remaining suspects are
  the game phases the mirror tests never touch — coin toss / kick-receive choice
  (`scripted::coin_toss_pick` collapses it to a fixed pick), setup, kickoff placement — plus the
  chance model's fixed `Direction::up()` collapse, which is y-only and so seat-symmetric only if
  the pitch is. Stage 2/3 add 600 gen03-vs-gen03 games on one seed base; re-run the script on
  them before spending a dedicated mirror. Either way the paired design already removes this term
  from every point estimate in this plan; it costs only variance (≈+0.003 on SE at 120 games).

- **Found and fixed (2026-09-07): an engine bug, not a search or net effect.** Two no-search
  mirrors (`--candidate-bot scripted --rungs scripted`, 10,000 distinct games; `random` vs
  `random`, 40,000) showed the asymmetry is in the *game*: random-vs-random Home 0.495 ± 0.001
  (z −4.9), and **only when Home kicked first** (Home 0.490, z −7.1, TD 1858:2274 = Away +22%;
  Away-kicked-first exactly 0.500). A drive-level probe (`botbowl-engine/tests/seat_probe.rs`,
  `#[ignore]`d, 20k random games) localised it to the **line-ups**: in second-half drives after
  Home kicked first, Home fielded `B L L T` in 820 of 2159 drives and `B C L L` in 24, while Away
  fielded `B C L L` throughout; Home as receiver scored 657 TDs against Away's 834 in the mirror
  arm (first-half drives, where line-ups are still fresh, were even: 946 vs 949).

  Cause: `Setup::setup_line` fields players in **dugout-slot order** and stops at `team_size`,
  and `unfield_player` refills the **shared** dugout array first-free-slot. The receiving team
  sets up first in this engine; when that is Away, Away's players come off the pitch into Home's
  low slots, Home's benched Thrower (slot 4) now precedes Home's Catcher in iteration, and the
  formation fields T for C — permanently, a ratchet (once shifted, the order never recovers).
  Away's slots are never invaded, so Away's line-up never changes. Invisible on the full pitch
  (eleven of twelve fielded, roles identical either way); on 14x7/4 it swaps a MA8/Dodge Catcher
  for a MA6 Sure-Hands Thrower for one seat only. Fix: `setup_line` sorts its candidates by
  role rank (Lineman, Blitzer, Catcher, Thrower — the roster's own order, so drive 1 is
  unchanged) then id; regression test `lineups_do_not_drift_with_setup_order` (16x9/4, fails
  on the old code with exactly the T-for-C swap). After the fix the same 20k probe gives
  second-half receiver TDs 855 vs 849 and kicker TDs 194 vs 218; team totals Home 2265, Away
  2249.

  Why it reached the NN numbers: an NN game has ≈7 TDs, so the ratchet fires in almost every
  game (any Home TD → Away receives → Away sets up first), and a search bot uses the Catcher.
  It is the leading explanation for the 0.438 above, the plan-027 56% Away share, **and plan
  031 D6's Away-skewed value labels in the nn half of gen05-07** — the corpus was labelled
  under a one-seat handicap. (Hypothesis until the stage 2/3 NN games are pooled; the no-search
  evidence is direct, the NN link is inferred from the TD count.)
  Mirror-exactness (D9) could not see it: it checks decision states, never the setup procs.
  The bug also confounds every heuristic-vs-nn comparison slightly less (the heuristic bot
  scores ≈1 TD/game, so fewer re-setups). Stage 1 (`s1-d7-vs-gen03`) was launched at commit
  63d511e, before the fix; stages 2/3 pick up the rebuilt binary — paired designs are unaffected
  either way, only the seat term moves. **Consequence for the queue:** the self-play corpus
  from gen08 on is generated by a symmetric game; re-running `side_bias_pooled.py` on the stage
  2/3 gen03-vs-gen03 games (600) is the check that the NN seat term is gone.

  No-search mirrors before and after the fix (`runs/exp032/side/`, `side_bias_pooled.py`;
  deterministic mirrors count each seed once):

  | mirror | games | Home pts, old engine | Home pts, fixed | split, fixed |
  |---|---|---|---|---|
  | random vs random | 40,000 | 0.495 ± 0.001 (z −4.9); Home-kicked-first 0.490 (z −7.1) | **0.500 ± 0.001** (z +0.4), TD 4500:4498 | Home kicked first 0.499, Away 0.502 |
  | scripted vs scripted | 10,000 | 0.517 ± 0.004 (z +4.0), both splits | 0.490 ± 0.004 (z −2.2), TD 54755:55197 | Home kicked first 0.484 (z −2.5), Away 0.496 (z −0.7) |

  The random mirror — which exercises blocks, pushes, bounces, throw-ins and touchbacks with no
  preferences at all — is now even to ±0.001, so the *engine* is seat-symmetric for that action
  mix. The scripted bot flipped sign: under the old engine it was the one bot that *liked* the
  Thrower-for-Catcher swap (+0.017); with line-ups fixed it shows a residual −0.010 that sits in
  the Home-kicked-first split. Since the engine passes the random test, that residual is most
  likely in the bot's own board-coordinate tie-breaks (`max_by_key` over positions keeps the
  *last* maximum, and "forward" is +x for one seat and −x for the other) — a benchmark-rung
  nuisance of ≈0.01, not a training concern. Not pursued.

  A second, separate observation from the same probe, **not** a seat effect but worth a
  design note: the degenerate kickoff on 14x7/4 ends in a **touchback 59-61% of the time** (aim
  at `w/4` = 3.5 squares from the line, deviate d6 in eight directions, half the outcomes land
  out or on the kicker's side). Kick-offs are mostly "hand the ball to a receiver", so the whole
  receive-and-pick-up phase is under-represented in the corpus. Scaling the deviate to the board
  (or aiming deeper) is a rules-design choice for the tier, not a bug — logged in the open
  questions.

### Deprioritised, with the reason

- Ensembles of k short searches (plan 028 A-arms): a converging tree makes this a labels
  question; #7 is the cheaper version.
- Normalised Q: shown worse twice (plans 026, 028).
- Horizon 2 at equal iterations: lost 0.392 (plan 027 E3); retest only at equal wall clock after
  #2/#3 change the Q scale.
- `WINDOW_GENS` widening on its own: the trainer's view of "more data" is answered by #1; the
  streaming `prepare` already removes the memory ceiling.

## Open questions that need an artifact, not an experiment

**Answered by plan 031 (2026-09-07)** — kept here with the answer so the questions are not re-asked:

| question | answer |
|---|---|
| Which generation first played `--evaluator nn` (plans 028 and 029 disagree) | **gen05.** Plan 029's table is right; **plan 028**'s "Until gen05 … it now plays `--evaluator nn`" should read "up to and including gen04". Confirmed twice: `status.md` generate lines, and every shard's own `meta.home_bot` over all 40 shards. |
| How far do the gen06/gen07 fine-tunes move the weights | **Further than the reference, not less.** rel-L2 from gen03: gen06 **0.414**, gen07 **0.334**, against gen02→gen03's **0.245** and 1.26-1.94 for two unrelated nets. `WARM_LR` stays fixed (D5). |
| Root-Q calibration and its dependence on fan width | Optimistic by **+0.133-0.140** (mover frame), but slope moves the *wrong* way with fan width. The follow-up localises it: the bare leaf is at +0.031 and **the search adds +0.101** (D1). |
| Heuristic share of each pool by sample count | **33.2 / 32.9 / 33.5%** overall — but **29.1% of train and 47.5% of val** (D6). |
| Pair-correlation factor to apply to every quoted z | **There is none.** r = 1.003, CI [0.962, 1.060] over 813 games; every plan 027/029 z stands (D10). |
| Was any compared generation partly generated on tract after a sidecar fallback | **No.** Zero `NN_SERVER_FALLBACK` in all 24 gen05-07 shard logs, and per-shard `served` counts sum *exactly* to each server session's `samples=` (D6). |

**Still open:**

| question | where the answer is |
|---|---|
| Does the plan 028 convergence curve hold with tree reuse, which production uses and the probe does not | `runs/convergence/*.jsonl` from 2026-09-06 plus a `--tree-reuse` arm of `botbowl-ui convergence` |
| Does self-distillation *compound*? Plan 031 D2 caught the one-step shift (gen04→gen05, ρ 0.364→0.435) but **gen05/06/07 share one frozen generator net**, so the multi-generation test was vacuous | needs two consecutive generations with *different* generator nets — i.e. it cannot be answered until something passes the gate |
| Does a mean backup actually remove the +0.101 the minimax backup adds | `scripts/audit_value_head_bias.py` on a mean-backup corpus, before spending #2's 120 games |
| Is the mid-block-chain leaf-value gap real (a Skull leaf and a Pow leaf score identically, since block dice reach the tower only via the legal-action set) | needs an online measurement; invisible to a corpus whose value target is drive-constant (plan 031 D3) |
| Should the 14x7 kickoff deviate scale with the board? The tier's kick-off is a touchback 59-61% of the time (aim `w/4`, uncapped d6 deviate in 8 directions), so almost every drive starts as "hand the ball to a receiver" and the receive/pick-up phase is rare in the corpus (#11, `seat_probe.rs`) | a rules-design call for the tier — `Kickoff::step` (cap or scale the deviate), `get_best_kickoff_aim_for` (aim deeper); measure with the probe before and after, then regenerate |
