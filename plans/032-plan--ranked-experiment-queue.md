# Ranked experiment queue and open questions

**Status:** Live (written 2026-09-07 from the methodology audit). This is the single place the
open programme lives; completed plans point here. Ranking is by expected strength gain per hour
of machine time, not novelty. Re-rank after plan 031's diagnostics land — the "gate" column says
which diagnostic can demote an item before it costs any games.

Ground rules carried over from plans 027/029: paired Home/Away on a shared seed base, points
(W + D/2)/N, **120 games minimum** (SE ≈ 0.041 at a 20% draw rate; ~0.045 after pair correlation),
score with `scripts/paired_summary.py` as well as `eval_summary.py`, one variable per arm, commit
before launching, results decided by games not by `val_*`. Pre-commit the extension rule: a match
in [0.53, 0.58] gets 120 more games on the same pair, not a claim.

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
  both `.onnx` exist. (b) If (a) is positive: a warm start at `--lr 1e-3` (W3 at the from-scratch
  LR) vs gen03, to separate "warm start is the problem" from "2e-4 is the problem".
- **Gate.** None — runs first. Plan 031 D5's weight-distance check says how far 2e-4 moves gen03.
- **Cost.** (a) ~5 h; (b) 40 min + 5 h.
- **Decide.** D7 ≥ 0.55 vs gen03 → add a periodic from-scratch retrain (every k generations,
  `--eval-every`, `--select-on combined`) to the plan-030 rebuild. D7 ≤ 0.50 → neither data nor
  regime explains the plateau; #2/#3 move to the top.
- **Also adopt now, no experiment needed:** `--eval-every` in `train_loop.sh`. Plan 029 stage 3
  found gen05-07 all restored at epoch 0-1 of 10 under per-epoch validation, i.e. the loop has
  been shipping past-peak weights for lack of checkpoint resolution.

### 2. Mean backup instead of minimax with an NN leaf

- **Mechanism.** Player-node backup is max/min over child Q (`dynamics.rs`, `backprop_scores`);
  with a noisy learned leaf this is a max-over-noise estimator and it also sets FPU (= parent Q)
  to the most optimistic sibling. Averaging (visit-weighted child mean, AlphaZero-style) is the
  standard remedy. Chance nodes already average.
- **Change.** `BackupMode::{Minimax, Mean}` on `BloodBowlDynamics`, env/flag plumbed like
  `PuctMode`. Keep the known-outcome carve-out (exact ±1000 leaves). Note the mean changes the Q
  scale FPU and `PUCT_C` see, so run #3's `c` sweep on the winner.
- **Gate.** Plan 031 D1. Calibration on the diagonal at every fan width → demote below #3.
- **Cost.** ~half a day of code + tests (existing `backprop_*` unit tests pin minimax; add mean
  variants), 120 games ~5 h.
- **Expected.** +0.05 to +0.10 if D1 shows optimism; also expected to lift plan 028's flat spot,
  since budget is currently spent amplifying maxima.
- **Abandon.** Mean ≤ 0.50 vs minimax at 120 games *and* D1 calibration already good.

### 3. Re-tune `PUCT_C` under learned priors; add FPU reduction

- **Mechanism.** NN priors are softmax×len, so the exploration term's spread is ~3× the scripted
  one at the top and ~0 in the tail; `c=10` was tuned for scripted priors and plan 026's sweep
  used the heuristic evaluator. FPU has no reduction, so the first sweep is breadth-first by
  construction (plan 026 §successor experiment).
- **Arms.** `--puct-c 3 | 10 | 30` under `--evaluator nn`, 120 games each vs `c=10`; then
  `fpu = parent_Q − k·√(Σ visited priors)` at one `k`, 120 games. Flags exist for `c`; FPU
  reduction is ~30 lines in `puct_value`.
- **Gate.** Plan 031 D4. If root visit entropy under `nn` is not far below `nn-value`, drop to #6.
- **Cost.** ~15 h of games total.
- **Expected.** +0.03 to +0.06 (plan 026 saw c=30 at 0.593 under the heuristic, p=0.08).
- **Abandon.** All arms within 1 SE of each other.

### 4. Exploration in self-play: root Dirichlet noise + visit-temperature sampling for the first k decisions of a drive

- **Mechanism.** Generation is argmax-Q with no noise (`pick_best_action`); diversity comes only
  from random starts and dice. The value head never sees what non-greedy alternatives were worth
  and the policy target never sees a root that was pushed off its prior.
- **Change.** In `dataset` only (never `eval`): Dirichlet(α) mixed into root priors with
  α ≈ 10 / mean legal actions (plan 031 D4 gives the number), ε=0.25; sample the played action
  ∝ visits for the first k=2 decisions of a drive, argmax after. Apply noise only when
  `state == root_state` — a per-search constant like the horizon anchor, so recombination purity
  holds and the cached `prior_bits` in `BbAction` never differ between paths.
- **Gate.** Plan 031 D2: a high tie rate or falling π entropy promotes this above #3.
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

### 6. Heuristic hedge ablation

- **Mechanism.** One third of every corpus is heuristic-MCTS play with scripted-prior policy
  targets (plan 029 correction). It was a plan-020 hedge against a bad gen-0 net and has never
  been re-examined; it now teaches the policy head the scripted prior a third of the time.
- **Arms.** `HEUR_SHARDS=""` (all 8 nn) vs current, one generation each from the same champion,
  same seeds; train from the same init; 120 games. Also check corpus health (TDs/drive,
  scoreless %, class split) — the hedge may still be protecting value-target balance.
- **Gate.** Plan 031 D6's by-sample share. Below 20% of samples → skip.
- **Cost.** ~14 h. **Expected.** ±0.03. **Abandon.** Within 1 SE, or health metrics degrade.

### 7. Q-informed policy target (Gumbel / completed-Q style)

- **Mechanism.** At 1000 iterations over 30-100 children the visit target is mostly the FPU
  sweep plus prior. `π ∝ softmax(logit(prior) + σ(q_mover))` with completed Q for unvisited
  children extracts the search's *value* information into the target and is robust at low
  visit counts. `q` per child is already recorded, so this is a `targets.rs` change and a
  retrain on the existing corpus.
- **Gate.** Rerun plan 028 Stage 0 on 50 states (~1 h): top-1 agreement of the new target with
  the 16k reference must beat raw visits' 0.69. No gain → drop.
- **Cost.** ~2 h code + 1 h probe + 40 min train + 5 h games.
- **Expected.** +0.03 to +0.08 on the policy head; compounds with #4.

### 8. Encoder additions

- **Mechanism.** No plane for the active player's action type or blitz-block-thrown (partial
  observability, plan 031 D3); no coordinate planes (receptive field 27 covers 16 columns but
  not 28, so this matters from the 20x11 tier up); `kicking_first_half`/receiver flag absent
  (irrelevant in drive-bounded training, relevant in full-game eval).
- **Change.** +5 action-type planes on the active square, +2 normalised coordinate planes,
  +1 global "we receive this half". Schema bump, retrain D1 from the same init, `val_value`
  compare, then 120 games.
- **Gate.** Plan 031 D3 collision count. **Cost.** ~6 h. **Expected.** Small on 14x7.

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
  without. After plan 031 D9 (exact-mirror test under `Evaluator::Nn`) is green: a 300-game NN
  mirror with `--per-game-out`, split by `kicking_first_half` and by `TieBreak::{Hash,Mover}`.
  ~8 h. Outcome is variance reduction on every benchmark, or a real bug.

### Deprioritised, with the reason

- Ensembles of k short searches (plan 028 A-arms): a converging tree makes this a labels
  question; #7 is the cheaper version.
- Normalised Q: shown worse twice (plans 026, 028).
- Horizon 2 at equal iterations: lost 0.392 (plan 027 E3); retest only at equal wall clock after
  #2/#3 change the Q scale.
- `WINDOW_GENS` widening on its own: the trainer's view of "more data" is answered by #1; the
  streaming `prepare` already removes the memory ceiling.

## Open questions that need an artifact, not an experiment

| question | where the answer is |
|---|---|
| Which generation first played `--evaluator nn` (plans 028 and 029 disagree) | `runs/loop14x7/status.md`, gen05 generate line |
| How far do the gen06/gen07 fine-tunes move the weights (plan 029 stage 3 already shows they restore at the first or second checkpoint) | `gen0{6,7}/train.log`; `bbnet_14x7_gen0{3,6,7}.pt` (plan 031 D5) |
| Root-Q calibration and its dependence on fan width | `gen0{5,6,7}/shard*.jsonl` (plan 031 D1) |
| Heuristic share of each pool by sample count | same shards (plan 031 D6) |
| Does the plan 028 convergence curve hold with tree reuse, which production uses and the probe does not | `runs/convergence/*.jsonl` from 2026-09-06 plus a `--tree-reuse` arm of `botbowl-ui convergence` |
| Pair-correlation factor to apply to every quoted z | `runs/exp-data/*.games.jsonl`, `runs/exp-search/*.games.jsonl` (plan 031 D10) |
| Was any compared generation partly generated on tract after a sidecar fallback | `gen0*/shard*.log`, `nn_server.log` (plan 031 D6) |
