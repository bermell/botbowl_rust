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
  32 min → ~5.3 h (actual 284 min; the wide-fan probe shared the cores for 90 of them).
- **Result (2026-09-08 02:35): D7 = 0.396 ± 0.039 vs gen03 (W34 D27 L59, TD 384:419, z = −2.7).**
  Paired 0.396 ± 0.038 over 60 pairs, 20 pairs split 1-1. Both nets played both seats
  (Home share 44:49 — this match ran on the pre-#11-fix binary, commit 63d511e, so the seat
  handicap was present but symmetric). **The from-scratch retrain on the whole corpus does not
  beat the incremental champion; it loses clearly.** The ordering is gen03 ≫ D7 ≫ D1
  (0.662 vs D1, plan 029): the from-scratch data curve is real but rises from a floor well below
  the loop's net, and plan 029's "D7 is the strongest net we have" was never tested against the
  champion — `val_*` on a shifting held-out set cannot rank nets from different recipes.
- **Decision (pre-registered branch D7 ≤ 0.50).** No periodic from-scratch retrain in the plan-030
  rebuild. Neither "more data" nor "a bigger step" explains the gen03→gen07 plateau on its own;
  #2 (backup) and #7 (policy target), both in flight, stay at the top. Two things the result does
  *not* settle, and one cheap way to split them, recorded as **#1b** below:
  - *Recipe.* gen03's lineage is gen00 (10 epochs from scratch on **heuristic-MCTS** games) →
    three fine-tunes at 2e-4. D7's is `arm_init.pt` → 110k steps at 1e-3 on gen01-07 only. D7
    never saw the gen00 heuristic corpus. If the heuristic bootstrap is what carries the strength
    (a calibrated deterministic teacher, versus self-play labels from a weak net's search), the
    loop should keep heuristic shards in the window permanently, not just at gen00.
  - *Distillation ceiling.* gen04-07 were all generated by the frozen gen03; a net fitted to that
    pool is at best a student of gen03's search, and with drive-outcome value labels (noisy) and
    visit policy labels (D2: 0.22-0.25 top-1 agreement with the played move at wide fans) the
    student comes out weaker than the teacher. That is the same defect #7 attacks.
- **#1b — does the heuristic bootstrap corpus carry the strength? (new, ~6 h).** One arm:
  `d8h` = D7's exact recipe plus the gen00 shards in the pool (gen00-07, same shards, same init,
  same steps). Match `d8h vs gen03` on seed base 32000000. If d8h ≥ 0.50 where D7 was 0.40, the
  heuristic data is worth ≥ +0.10 and the loop's window must always include it; if d8h ≈ D7,
  the gap is the recipe/distillation and the from-scratch route is closed. Queue position: after
  #7's match, before #3's screens (`scripts/exp032_s1b_d8h.sh`, launched 2026-09-08 02:38; it
  holds `runs/exp032/s1b.pending`, which stage 3 blocks on). d8h validates on the loop's
  `gen07/prepared_val` like D7, so this is the one arm pair whose `val_*` are comparable.
- **Result (2026-09-08 12:45, 188 min): d8h = 0.425 ± 0.041 vs gen03** (W36 D30 L54, TD
  404:468; as Home W16 L29, as Away W20 L25; 28% of pairs split 1-1). On val the two are
  indistinguishable (d8h 1.829 vs D7 1.835 combined). **d8h ≈ D7** (0.425 vs 0.396, Δ = +0.03,
  inside one SE of the difference 0.057): the gen00 heuristic corpus is worth at most a few
  points, not the 0.10 that would have made it the missing ingredient. **The from-scratch route
  is closed**: the recipe/distillation gap is what separates D7-class nets from gen03, and #7's
  result (Q7 +0.11 over D7 from the label alone) says the label is the largest known piece of it.
  The loop keeps its heuristic hedge for the reasons in #6 (label balance, ties), not because it
  carries strength. Consequence for the queue: the natural next from-scratch test is **Q7 vs
  gen03** — if the cq label closes the 0.10 gap to the champion, a full retrain becomes viable
  again as the plan-030 rebuild's periodic step; it costs one 120-game match and no training,
  so it slots in after the τ decision (#7b).
- **#7c result (2026-09-09 16:31, 186 min, `scripts/exp032_s7c_q7_vs_champ.sh`): Q7 = 0.625 ±
  0.041 vs champion gen03** (W63 D24 L33, TD 503:425; Home 33-15, Away 30-18; z = +3.0). The
  same recipe that lost to gen03 at 0.396 with the visit label (D7) **beats it by the same margin
  with the completed-Q label** — a 0.23-point swing from the label alone, on the same 120 seeds.
  Three consequences:
  1. **The from-scratch route is reopened**, and it is now the strongest route we have: Q7 is
     the first net in the programme to beat gen03, after four incremental generations (gen04-07)
     failed to. It clears the loop's promotion gate (0.55) by 1.8 SE.
  2. **Q7 should be the loop's champion when it relaunches.** Otherwise gen08 generates from a
     net that Q7 beats 0.625, and fine-tunes gen03 onto data gen03 produced — the distillation
     ceiling #1 identified. The loop has no "install an external champion" step; the manual
     version is: copy `runs/exp032/q7.{onnx,pt}` to `models/bbnet_14x7_q7.{onnx,pt}`, write its
     path to `runs/loop14x7/champion.txt`, and let gen08 warm-start from it (`WARM_FROM=champion`)
     with the cq label already the default. Pending the user's go-ahead — it changes what the
     loop generates from.
  3. **The gen03 → gen07 plateau is explained**, not by data volume, backup, `c`, FPU, capacity
     or seed (all tested in this plan and flat), but by the policy label. The value head was
     never the problem (#2's calibration audit: slope 0.95 under minimax); the policy head was
     being taught an unconverged visit distribution, and under `EVALUATOR=nn` that head steers
     the search.

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
  minimax, 120 games, seed base 32000000.
- **Part A result (2026-09-08 03:25, fixed engine, commit 76bda95; `runs/exp032/audit_s2-corpus-*.txt`).**
  150 random-start self-play games per rule on the same seeds, gen03 both sides, 1000 iterations.

  | | minimax (5,436 rows) | mean (6,269 rows) |
  |---|---|---|
  | bare NN leaf gap vs drive outcome | +0.045 (SE 0.009) | **+0.120** (SE 0.009) |
  | 1000-iter search-root gap | +0.135 | +0.042 |
  | **added by the search** (paired) | **+0.090** (z = 24.8) | **−0.078** (z = −25.8) |
  | leaf calibration slope `b` | 0.95 | **0.59** |
  | leaf RMSE | 0.638 | 0.701 |
  | TDs / game, samples / game | 2.65, 36.2 | 2.47, 41.8 |

  Three readings. (a) **The mechanism is confirmed**: swapping the backup rule moves the
  search-added term by 0.17 of a drive outcome, from +0.09 optimistic to −0.08 pessimistic — the
  backup manufactures the bias, replicating D1 on the fixed engine and a fresh seed. (b) **Mean
  overshoots into pessimism by the same magnitude**, so the root is no better calibrated in
  absolute terms; it is calibrated at +0.04 only because the leaf on these rows is *worse*. The
  bias sign flips by fan width too: −0.16 at fans ≥31 (the visits there are mostly the FPU
  sweep, D2, so the visit-weighted mean averages in a crowd of never-refuted bad moves), ≈0 at
  11-30, −0.075 at ≤10. (c) The strongest signal is the leaf column: **on positions the
  mean-backup bot reaches, gen03's own value head decalibrates from slope 0.95 to 0.59** — in
  the top bin (pred +0.92) the drive is converted +0.50 under mean play vs +0.84 under minimax,
  and symmetrically lost positions (pred −0.89) end −0.54 vs −0.86. Same net, same seeds; only
  the play differs. Compression of outcomes toward zero from both ends, with 15% more decisions
  per drive and 7% fewer TDs, is the signature of a bot that fails to convert what it has —
  i.e. the mean-backup bot is the weaker player in its own self-play. Pre-committed prediction
  for Part B, made before the match: **mean loses**, and the follow-up is not "mean vs minimax"
  but a backup that keeps max-ness where the search has concentrated and averages only where it
  has not (visit-weighted mean over the top-k by visits, or a max/mean blend `λ·max + (1−λ)·mean`
  with λ rising in the child's visit share).
- **Part B result (2026-09-08 05:48, 143 min): mean = 0.454 ± 0.039 vs minimax (W38 D33 L49,
  TD 244:263, z = −1.2; paired 0.454 ± 0.041, 16/60 pairs split by side).** The pre-committed
  prediction held in direction; the abandon rule ("mean ≤ 0.50 at 120 games") is met.
  **Decision: pure mean backup is abandoned; minimax stays production, and stage 3's `c`/FPU
  sweep runs under minimax** (`pick_backup` chose it automatically). What the item taught:
  the backup rule is the *lever* on the search's value bias (0.17 of a drive outcome between
  the two rules, z ≈ 25 each way), but the bias itself is not the strength problem — the
  optimistic rule wins. The likely reason is Blood Bowl's turnover structure: most of a
  player-node's children are moves that hand over the turn cheaply, the search never refutes
  them at 1000 iterations (D2's FPU-sweep visits), and averaging them in makes every position
  look mediocre, so the bot stops distinguishing good plans from bad ones (the leaf-slope 0.59
  in Part A). Max is the right operator for "there exists a plan"; it is only the *noise* under
  max that hurts, and that is a leaf-quality problem more than a backup problem.
- **#2b — blended backup (new, deprioritised until #3 lands).** `λ·max + (1−λ)·mean` per player
  node with λ = visit share of the best child (→ max where the search has concentrated, → mean
  where it is a flat sweep), or mean over the top-k by visits. Interacts with `PUCT_C`/FPU
  (both change how concentrated visits are), so it is sized *after* stage 3 picks c and k:
  ~2 h code + one 120-game match. Skip if stage 3's FPU reduction alone removes the wide-fan
  sweep.

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
  a winner gets a sized match. Stage 2 gave mean 0.454, so all arms run **minimax**.
- **Results (2026-09-08, 120 games each, gen03 both sides, minimax).**

  | arm | points | W-D-L | TD | z |
  |---|---|---|---|---|
  | `c=3` vs `c=10` | **0.421 ± 0.039** | 36-29-55 | 450:500 | −2.0 |
  | `c=30` vs `c=10` | 0.521 ± 0.037 | 47-31-42 | 467:452 | +0.5 |
  | `k=100` vs `k=0` | 0.446 ± 0.043 | 38-31-51 | 412:427 | −1.3 |
  | `k=300` vs `k=0` | 0.500 ± 0.044 | 50-20-50 | 416:404 | 0.0 |

  `c=3` (15:40, 174 min): less exploration **loses** clearly, and it loses from both seats (Home
  20-25, Away 16-30). So under learned priors the search is not over-exploring at `c=10` — the
  gain, if any, is on the *more*-exploration side (`c=30`), consistent with plan 026's heuristic
  result and with D4's finding that the learned prior's top lift (5.5×) already concentrates the
  sweep. Note the paired SE is no tighter than unpaired again (1.04×): with a 1000-iteration
  search the two seats' games diverge early enough that pairing buys almost nothing on this tier.
  `c=30` (18:38, 177 min): **0.521, inside one SE of 0.50** — no measurable gain from tripling
  exploration either. Together the two `c` screens bracket `c=10`: the curve is flat-to-falling
  in both directions (0.421 / 0.50 / 0.521), so `c=10` is within ~0.02 of the optimum and the
  original +0.03-0.06 expectation for this item does not hold under the NN. Per the abandon
  rule `c` gets no sized match: strength is flat from c=10 to c=30 and only falls when
  exploration is cut. The FPU screens are the remaining hope for #3.
  `k=100` (21:29, 171 min): **0.446 — a mild loss** (z = −1.3), symmetric across seats (Home 19-26,
  Away 19-25). A 0.1-TD first-play penalty on unvisited children does not help; the plain FPU
  (`parent_Q`, which under minimax is the most optimistic sibling) is at worst no worse. This is
  the opposite of the Leela/KataGo experience and consistent with the direction the `c` screens
  gave: at 1000 iterations on this tier the search benefits from *breadth* at the root, and
  anything that narrows the first sweep (c=3, FPU reduction) costs points. `k=300` is running
  only because it is already queued; the expectation is now that it loses harder.
  `k=300` (2026-09-09 00:19, 169 min): **0.500 exactly** (W50 D20 L50, TD 416:404) — the
  prediction that it would lose harder was wrong; a 0.3-TD penalty is a wash, with fewer draws
  than k=100 (20 vs 31) but the wins and losses it converts are balanced. So FPU reduction is
  0.446 / 0.500 at k = 100 / 300: nowhere positive, and not monotone, which is what a
  null effect measured twice at SE 0.04 looks like.
- **Decision — #3 closed, nothing ships.** Four screens, no arm above one SE of 0.50 on the
  high side (0.421, 0.521, 0.446, 0.500); the abandon rule fires. `c=10`, plain FPU stay. The
  search-side knobs plan 026 identified are not where the strength is under the learned prior;
  the cheapest reading of the whole stage is that the 1000-iteration search on this tier is
  *breadth-limited at the root* (cutting exploration costs 0.08, adding it or narrowing the first
  sweep does nothing), which is an argument for #10 (budget) as a diagnostic and for the label
  work (#7) that changes what the prior points the breadth at. #2b (blended backup) drops below
  #10 — the one search-side lever that showed a real signal (#2) did so on bias, not on strength.

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
- **Run (launched 2026-09-08 05:54, `scripts/exp032_s59_seed_capacity.sh`).** Moved from the D1
  pool to the **D7 pool** (gen01-07, visit target) so the floor is measured on the recipe every
  from-scratch arm now shares, and one of the two nets is D7 itself (no new control to train).
  `d7s2` = D7's recipe from a second materialised init (`arm_init_s2.pt`, `--seed 20260907` for
  init, shuffle and augmentation), same steps, same held-out. Match `d7s2 vs d7`, **300 games**
  (SE ≈ 0.026), seed base 32000000, queued behind stage 3 and #9's match.
- **Training (done 07:11, 77 min on the GPU next to the eval chain).** The seeds are
  indistinguishable on the held-out set: `d7s2` best combined **1.8313** (step 90k: val_policy
  1.4461, val_value 0.3852, top-1 0.535) vs D7 **1.8352** (step 92.5k: 1.4477 / 0.3876 / 0.535);
  the same 0.004 as the run-to-run wobble of a single curve. Both peak at step 90-92.5k of 110k
  and drift up after. So whatever strength gap the 300 games find is *not* visible in `val_*` —
  which is the point: it bounds how much strength variance hides behind identical losses.
  The match moved to `scripts/exp032_s7b_tau50.sh` (order after stage 3: #9, #7b, then this).
- **Result (2026-09-09 13:23, 412 min, 300 games): d7s2 = 0.530 ± 0.028 vs D7** (W130 D58 L112,
  TD 1072:1019; Home 70-57, Away 60-55; paired SE 0.028 over 150 pairs, 25% split 1-1). So two
  nets that are **identical on val to 0.004** differ in strength by **+0.03 ± 0.03** — the seed
  floor is bounded at about ±0.03-0.06 (one-sided 95% bound on |Δ| ≈ 0.08). Readings:
  1. **Every 120-game verdict of ±0.04 in plans 029/032 sits on top of a ±0.03 seed term.** A
     single-arm result of |Δ| < 0.06 (c=30 0.521, k=300 0.500, d7w96 0.508, q50 0.471) is not
     distinguishable from a seed re-roll. The results that survive this floor are the ones with
     |Δ| ≥ 0.10: Q7 vs D7 (+0.11), c=3 (−0.08, borderline), mean backup (−0.05, not), D7/d8h vs
     gen03 (−0.10/−0.08), plan 029's D7 vs D1 (+0.16).
  2. Plan 029's +0.06 per data doubling is **about one seed floor per doubling** — real in
     aggregate across three doublings, but no single step of it was.
  3. **Practical rule going forward:** treat 120 games as a screen for |Δ| ≥ 0.10 only; anything
     that screens at 0.05-0.10 needs either a replicate from a second seed (the cleaner answer —
     it also averages the floor) or 300+ games *and* the awareness that 300 games still cannot
     separate a +0.03 effect from a lucky seed. The cheap version of a replicate is to train the
     candidate from `arm_init_s2.pt` as well and pool the two matches.

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
  not the norm, and the first decision after activation is often forced.

  **Result (2026-09-08, 93 min, `runs/exp032/s7-widefan-c10.jsonl`, `audit_s7-widefan-c10.txt`).**
  Two bookkeeping corrections first: the effective seed base was 32000000, not 91000000 (the
  lib exports `SEED` after the script sets it — harmless, the seeds are still disjoint from
  every match); and of 200 seeds **163 were skipped** for fan <30, leaving **37 states** (fans
  31-95, 24 pairs in the 31-60 bucket and 198 in >60). The bimodality from the smoke run held
  at scale: ≈18% of second decisions have a wide fan. Reference (16k) self-agreement across
  its own repeats is 0.79 on argmax visits and 1.00 on `chosen_action`, so the ceiling for any
  1000-budget target is ~0.8 on the visits column and 1.0 on the played column.

  | target | top-1 vs ref visits | top-1 vs ref played | >60 bucket, vs ref played (198 pairs) |
  |---|---|---|---|
  | visits (shipped) | 0.216 | 0.212 | 0.177 |
  | argmaxq | 0.270 | 0.347 | 0.308 |
  | **cq(25)** | **0.311** | **0.401** | **0.389** |
  | cq(50) | 0.311 | 0.365 | 0.348 |
  | cq(100) | 0.284 | 0.284 | 0.258 |
  | cq(200) | 0.239 | 0.239 | 0.207 |
  | cqv(25) | 0.288 | 0.288 | 0.263 |

  Reading. (a) **The shipped target is near-useless at wide fans**: 0.21 top-1 against either
  reference, versus 0.69 on the narrow dump — D2's 0.22/0.25 numbers were not an artefact of
  comparing against the played move, the 1000-visit distribution simply has not converged
  there. (b) **Completed-Q beats it in every cell of both probes.** τ=100 (the pre-registered
  value, best on the narrow dump) wins by +0.07 here; the sharper τ=25-50 wins by +0.10/+0.19
  at wide fans but was slightly *worse* than visits on the narrow dump (cq(50) 0.647 vs 0.687
  on ref visits). The trade-off is the expected one: at τ→0 the target collapses onto
  `argmaxq`, which is exactly what `pick_best_action` plays and so scores well on "ref played"
  while throwing away the distributional information a prior needs at narrow fans. (c) The
  `cqv` variants (ln visits instead of ln prior) track `visits` and lose — the prior, not the
  visit count, carries the useful ranking at wide fans, which is consistent with D2's finding
  that visits there are mostly the FPU sweep.

  **Gate passes.** Decision: train **Q7 at τ=100** as pre-registered (the only value that beats
  the shipped target on *all four* cells of the two probes: narrow/wide × ref visits/ref
  played); if Q7 beats D7, the follow-up is τ=50 as a second one-variable arm, and if that also
  wins, a fan-dependent τ. Both probe files stay under `runs/exp032/` for re-scoring new target
  ideas offline (`scripts/audit_q_target.py <jsonl>`) — no search needed.
- **Run (launched 2026-09-08 00:06, `scripts/exp032_s7_cq_target.sh`, commit 030745b).** D7 is
  the control (plan 029: from scratch on gen01-07, `arm_init.pt`, lr 1e-3, 110k steps, visit
  target). Q7 = the identical recipe with the pool *and* the held-out set re-prepared under
  `prepare --policy-target cq --tau 100` (so `--select-on combined` selects against the label
  the net is trained on). Caveat found on launch: D7's held-out was the loop's
  `gen07/prepared_val` (114k samples, shards 4+7 of the 3-generation window gen05-07), Q7's is
  gen07 shards 4+7 alone (36k) — so **no `val_*` number is comparable between the two arms**,
  including val_value (baselines 0.641 vs 0.667 on the untrained init). Only the match counts.
  Prepare + train overlapped stage 1 (prepare 1 min, train 51 min on the shared GPU; restored
  at step 75,000 of 110,000, val_value 0.377, val_top1 0.572 against its own label). The match
  `s7-q7-vs-d7` (120 games, seed base 32000000) waits for the stage-2 runner and precedes
  stage 3.
- **Result (2026-09-08 09:36, 227 min — slowed by two GPU trains sharing the cores).**
  **Q7 beats D7 0.613 ± 0.040 (z = +2.8)**: W62 D23 L35, TD 470:406; as Home W34 L17, as
  Away W28 L18 (it wins from both seats); the paired SE is no tighter than unpaired (0.99×) and 35% of pairs split 1-1, so
  the seeds carry little shared luck here. This is the **largest single-variable gain in the
  programme so far**, and it comes from the label alone: same data, same init, same seed, same
  steps, same search — only the policy target changed. It is also the first time an offline
  proxy (the two top-1 probes) has predicted a match result in the right direction *and* at
  roughly the right size (+0.06 on the played-move statistic → +0.11 in points, plausible
  since the policy improvement compounds through the search).
  Consequences:
  1. **Ship it.** The loop's `prepare` call should use `--policy-target cq --tau 100` from the
     next generation (`train_loop.sh` — not yet changed; deploy after the τ follow-up below so
     the shipped value is the tested one). Since the held-out set is re-prepared under the same
     target, `val_policy`/`val_top1` numbers from the loop will step-change and are not
     comparable across the switch; `val_value` is.
  2. **It re-ranks the queue.** D2's diagnosis (visit target ≈ FPU sweep at wide fans) is now a
     confirmed strength lever, which raises the value of the fan-dependent τ idea and of #4's
     class-balancing (which acts on the same label) and lowers the priority of the search-side
     items (#3, #2b) that were trying to fix the same symptom from the other end.
  3. **Follow-up launched (pre-registered): `q50` at τ=50**, `scripts/exp032_s7b_tau50.sh`,
     identical recipe, head-to-head **vs Q7** 120 games (the question is which target to ship,
     so the direct match is the cheapest discriminator; both arms' val sets are prepared under
     their own τ so again only the match counts). Queued after stage 3 and #9's match, ahead
     of #5's 300 games. If q50 wins, a fan-dependent τ (sharper where the fan is wide) is next;
     if Q7 holds, τ=100 ships.
     *Training (done 10:38, 58 min):* q50 restored at step 102.5k (val_policy 1.302, val_value
     **0.380**, val_top1 0.581 against its own sharper label) vs Q7 at 75k (1.362 / **0.377** /
     0.565). The held-out set is the same 36k rows for both and the value label does not depend
     on τ, so **val_value is comparable here**: 0.380 vs 0.377 — the sharper policy target costs
     the value head nothing measurable. Policy numbers are not comparable (different labels).
     *Result (2026-09-09 06:31, 190 min):* **q50 = 0.471 ± 0.042 vs Q7** (W47 D19 L54, TD
     461:470; Home 20-32, Away 27-22). The sharper target does **not** beat τ=100 — a mild,
     non-significant loss (z = −0.7), so the best reading is "τ=50 ≈ τ=100, if anything worse".
     That matches the offline probes' trade-off: τ=50 won at wide fans but lost at narrow ones,
     and narrow roots are ~85% of samples. **τ=100 ships** (already the loop default, ef3cf8e).
     Fan-dependent τ (sharper only where the fan is wide) remains the one untested variant with
     an offline case for it; it is a `prepare` change, and can be scored offline first against
     both probe files before any training. Not queued for now — the from-scratch-vs-champion
     question (Q7 vs gen03, #7c) comes first.

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
- **Run (launched 2026-09-08 05:54, same script as #5).** On the D7 pool against D7 rather than
  D3/D3 — same reasoning as #5, and D7 is the from-scratch net the loop would actually ship.
  `bbnn.train` gained `--width/--blocks`; every loader (`--init`, `nn_server.py`,
  `audit_value_head_bias.py`) now rebuilds `BBNet` from the state dict's shape
  (`BBNet.from_state_dict`), bit-identical for the 64x6 default. `d7w96` = width 96 / blocks 8
  (**1.39 M params, 2.9×**), `--seed 20260906` (D7's), same 110k steps and held-out; tract runs
  the wider ONNX fine. Match `d7w96 vs d7`, 120 games, seed base 32000000.
- **Training (done 08:51, 99 min vs D7's ~75 — the GPU is not the bottleneck, the Python batcher
  is).** `d7w96` best combined **1.8270** at step 95k (val_policy 1.4390, val_value 0.3880, top-1
  0.538) vs D7 1.8352 and d7s2 1.8313. The 2.9× net buys **−0.008 combined, all of it policy**
  (1.439 vs 1.446-1.448; value 0.388 is inside the seed spread 0.385-0.388). That is twice the
  seed gap on val but still tiny: at 2.2 M training samples the 64x6 net is not badly
  capacity-limited on this pool. Training loss at the end is 1.43/0.36 vs D7's 1.43/0.37 —
  barely lower, so the wide net is not memorising either; it is data-limited like the small one.
  Prediction for the match: within ±0.05 of 0.50 (val says 0.50-0.53).
- **Result (2026-09-09 03:21, 179 min): d7w96 = 0.508 ± 0.043 vs D7** (W49 D24 L47, TD 406:390;
  Home 23-26, Away 26-21). As predicted: **no gain from 2.9× the parameters at this data size.**
  The 0.008 val edge did not turn into strength (or did, at a size 120 games cannot see — the
  bound is < +0.09). **Abandon** per the pre-registered rule. Capacity is not the ceiling; data
  volume and label quality are (#7). The 96x8 net also costs ~2.5× the GPU time per forward, which
  on the sidecar's critical path (plan 033: the GPU kernels are the floor) would slow generation.
  Revisit only when the corpus is ≥ 5× larger or after the label change has been in the loop for
  a few generations and val stops improving on 64x6.

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

- **Confirmed on NN full games (2026-09-09): the NN seat term is gone.** `side_bias_pooled.py`
  over the eleven post-fix NN matches in `runs/exp032/` (stage 2/3, #1b, #5, #7/7b/7c, #9 —
  1,500 games, nine different nets, one seed base):

  | | games | Home points | z | TD Home:Away |
  |---|---|---|---|---|
  | all NN, fixed engine | **1500** | **0.498 ± 0.011** | **−0.17** | 5249:5280 |
  | Home kicked first | 770 | 0.475 ± 0.016 | −1.59 | 2652:2764 |
  | Away kicked first | 730 | 0.523 ± 0.016 | +1.37 | 2597:2516 |
  | receiving team, seat-agnostic | 1500 | 0.524 ± 0.011 | +2.10 | |

  Against the pre-fix pool (600 games, 0.438, z −3.46) the seat term has moved from −0.06 to
  0.00 ± 0.01. **The line-up ratchet was the whole NN Away edge**, and the hypothesis is
  closed. Two residuals, both small: (1) receiving first is now worth +0.024 ± 0.011 — expected
  in Blood Bowl, was masked before by the Away edge, and cancels in the paired design; (2) the
  two kick splits sit ±0.024 either side of 0.50 in opposite directions, which is what the
  receive advantage looks like when split by who kicked (Home kicks → Away receives → Away
  edge, and vice versa) — not a second seat effect. #11 is **closed**; the corpus from gen08 on
  is generated by a seat-symmetric game, and plan 031 D6's Away-skewed value labels in gen05-07
  are the ratchet's fingerprint in the data.

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
