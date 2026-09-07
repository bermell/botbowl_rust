# Audit diagnostics: cheap checks before spending games

**Status:** Not started (written 2026-09-07 from the methodology audit). Every item here is under ~1 h
of machine time, most run offline on the shards already in `runs/loop14x7/`, and each one either
kills or promotes a specific entry in plan 032. Run in the order given; D1, D2 and D5 decide the
top three items of plan 032.

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

## D10 — Pair correlation on the per-game logs we already have

**Why.** Every z in plans 027/029 treats paired games as independent; plan 027 measured a 1.10×
SE inflation on one arm only.

**How.** `scripts/paired_summary.py runs/exp-data/*.games.jsonl runs/exp-search/*.games.jsonl`;
report the paired/unpaired SE ratio per arm and the pooled ratio.

**Read.** Use the pooled ratio to restate plan 029's z-scores and to size every match in plan 032.

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
