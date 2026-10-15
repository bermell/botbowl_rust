# Experiment registry — question, setup, result, conclusion

Newest first. Each entry: the question, how to reproduce it (commit, script, settings), what came
out and what was decided. Full write-ups live in the plans; head-to-head numbers are also in
[MATCHES.md](MATCHES.md), per-net numbers in [NETS.md](NETS.md), corpora in [DATA.md](DATA.md).
**Keep this current:** add the entry when an experiment starts, the result and conclusion when it
ends (see the root CLAUDE.md).

## v9 rules (2026-10-06 on)

### <a id="exp073"></a>exp073: the 96x8 nets fine-tuned in the loop's regime, 2026-10-14/15

- **Question:** exp072's scratch 96x8 nets fit gen10-31's average target. Fine-tuned on exactly the
  window and recipe the loop's gen33 used (gen31-33, MC labels, cq τ 100, lr 5e-5, 3 epochs,
  `--freeze-bn`, value weight 0.36 per-drive), does a 96x8 start beat the 64x6 lineage?
- **Setup:** `scripts/exp073_bignet_finetune.sh` (commit in `runs/exp073/status.md`): arms A96x8 and
  B96x8h from `runs/exp072/<arm>/net.pt`; scored on gen33's held-out shards (absorption/test), the value
  bench paired vs gen33, and drives vs gen33 under C1 at 1000 descents, **300 fixed pairs per board**
  (no SPRT). The loop pauses after gen34's generate for the ~2 h of fine-tunes and is relaunched by the
  script; the drives share its hub.
- **Result, single-net (2026-10-15 00:38, 457120f):** fine-tunes took 39-40 min each (the 64x6 one ~27).

  | on gen33's held-out shards / the value bench | loop gen33 (64x6) | A96x8 fine-tuned | B96x8h fine-tuned |
  |---|---|---|---|
  | KL(target‖net) | **0.053** | 0.084 | 0.085 |
  | log P(played) / P(played) | **−0.695 / 0.681** | −0.715 / 0.674 | −0.716 / 0.673 |
  | value MSE | 0.0785 | 0.0760 | **0.0743** |
  | value bench RMS (paired vs gen33) | 0.193 | 0.189 (−2.2%) | **0.184 (−4.3%, ~3 SE)** |

  One fine-tune makes the 96x8 value heads clearly better than gen33's (B: −4.3% on the bench, −5.4%
  test MSE), but closes only part of the policy gap (KL 0.104 → 0.085 vs 0.053).
- **Drives vs gen33 (2026-10-15, C1 both sides, 1000 descents, 300 fixed pairs per board):** A96x8
  14x7 0.500 ± 0.012, 16x9 0.509 ± 0.014; B96x8h 14x7 0.507 ± 0.013, 16x9 0.515 ± 0.014 (+1.1 SE).
- **Conclusion:** after one fine-tune the 96x8 nets play even with gen33 (B slightly ahead on 16x9, not
  significant) at ~2x the inference cost: not adopted yet. The user (2026-10-15): the 96x8 + heads net's
  benches have not converged (gen33's had); keep fine-tuning it on the loop's data as a **shadow
  lineage** (train_loop.sh section 3b, `SHADOW_INIT`, models `shadow96h_genNN`) and pit the two again
  once its benches flatten.
- **Adoption rule:** an arm that beats gen33 on drives (both boards, or one with the other level) and is
  not worse on the value bench / test value MSE becomes the loop's net (the user's call on the ~2x
  generation slowdown).

### <a id="exp072"></a>exp072: a bigger net from scratch on the clean v9 corpora (plan 063), started 2026-10-13

- **Status:** started 2026-10-13 08:35 at e647d02 (`RELAUNCH=1 scripts/exp072_bigger_net.sh`, the loop stopped by `scripts/stop_after.sh "gen31 train done"`); the loop relaunches after the GPU phases and the drives share its hub. The user chose 96x8 and asked to test bigger heads (2026-10-13).
- **Result, training and single-net numbers (2026-10-14 09:38, e647d02):**

  | | 64x6 scratch (C) | 96x8 (A) | 96x8 + heads (B) | loop gen31 (64x6 fine-tune lineage) |
  |---|---|---|---|---|
  | train time | 204 min | 523 min | 556 min | — |
  | val (gen10-30 held-out) policy / value | 0.7717 / 0.0871 | 0.7614 / 0.0816 | **0.7607 / 0.0803** | — |
  | test (gen31 shards 4,7) KL(target‖net) | 0.117 | 0.105 | 0.104 | **0.055** |
  | test log P(played) / P(played) | −0.749 / 0.653 | −0.739 / 0.659 | −0.736 / 0.659 | **−0.701 / 0.680** |
  | test value MSE | 0.0899 | 0.0840 | 0.0830 | **0.0815** |
  | value bench RMS (paired vs gen31) | 0.219 (+14.1%) | 0.196 (+2.1%) | 0.196 (+2.2%) | **0.192** |
  | GPU µs per sample at batch 12 / 24 | 141 / 116 | 263 / 223 | 276 / 235 | 133 / 106 |

  At equal training (scratch on gen10-31) the 96x8 trunk beats 64x6 on every number (value MSE −6%,
  test KL −10%, value bench 0.219 → 0.196) and the bigger heads add a little (value −1.6%). But the
  loop's gen31 — 64x6 fine-tuned generation after generation on the most recent window — beats every
  scratch arm on gen31's own held-out shards (KL 0.055 vs 0.104): the scratch nets fit the average of
  gen10-31's targets, not the current ones. Inference: 96x8 costs ~2.0x per sample on the GPU (less
  than the 2.7x CPU estimate). Drives vs gen31 (C1, 1000 descents) follow on the loop's hub.
- **Drives vs gen31 (2026-10-14, C1 both sides, 1000 descents, SPRT 0.5:0.55 on the loop's hub):**
  C64x6 14x7 0.463 ± 0.029 (H0, 16 pairs), 16x9 0.502 ± 0.022 (H0, 113); A96x8 0.500 ± 0.032 (H0, 38),
  0.508 ± 0.018 (H0, 196); B96x8h 0.471 ± 0.021 (H0, 33), 0.508 ± 0.018 (H0, 188). Every scratch arm
  plays about even with gen31 despite its much worse fit to the current targets (the 14x7 SPRTs stopped
  far too early to read). **Conclusion:** at equal training 96x8 is the better net; the fair question is
  a 96x8 trained in the loop's regime → exp073.
- **Question:** is the 64x6 net the ceiling now? Does a 96x8 trunk, with or without bigger heads,
  trained from scratch on every clean v9 corpus beat the loop's gen31 in play — and does a 64x6
  trained the same way (the control) already do so?
- **Setup:** `scripts/exp072_bigger_net.sh` (commit in `runs/exp072/status.md`). Data: `runs/loopmix16x9v9`
  gen10-31 MC-labelled shards, cq τ=100, value blend 1.0; train = shards 0-3,5,6 of gen10-31 (~2.66 M),
  val = 4,7 of gen10-30 (~0.84 M, checkpoint selection), test = 4,7 of gen31 (~40 k). Arms `C64x6`
  (today's shape), `A96x8` (96x8, today's heads), `B96x8h` (96x8 + `--value-conv 64 --value-pool
  meanmax --value-hidden 128 --policy-conv 96`). Recipe: batch 256, Adam, lr 1e-3, 2000 warmup steps
  then cosine to 0, 10 epochs, val every 5000 steps, restore on val_policy + val_value, value weight
  0.36, per-drive value weight, BN live, 6 loader workers, seed 72. Scored by: test-set absorption vs
  gen31 (`absorb_probe.py`), the value bench (`v9_gen01_val.jsonl`) paired vs gen31, the sidecar's
  forward cost (`nn_server.py --bench`), and paired contested drives vs gen31 under C1 both sides
  (`cfgs/chance_all_partial.toml`, 1000 descents, both position sets, SPRT 0.5:0.55, cap 800/board).
- **Cost (CPU-measured, relative to 64x6):** forward 2.7× (A) / 2.8-2.9× (B), train step 2.5× / 2.7×,
  tract 2.5× / 2.6×; MACs 2.86× / 3.03×. GPU estimate: C ~2.7 h, A ~7.3 h, B ~7.8 h of training.
- **Decision rule (plan 063 §7):** H1 on both boards or H1 on one with the other level; no >1%
  regression on the value bench or test value MSE; A over B unless B is better by >1 SE; if C64x6
  passes too, adopt C first (no throughput cost) and require the 96x8 net to beat it. A 96x8 net
  makes a loop generation ~2.3× longer: adoption is the user's call.
- **Prior:** plan 032 #9 (2026-09-08, old rules): 96x8 from scratch on 2.2 M samples, 0.508 ± 0.043 vs
  its 64x6 twin in 120 games — no gain at that data size and that resolution.

### MC labels through the hub: equivalence and overhead (plan 062), 2026-10-10

- **Question:** does `botbowl-hub job label` (plan 062, protocol v19) write what
  `botbowl-ui mc-label` writes, and what does the hub cost?
- **Setup:** plan 062 at cf4e08c (merged to master as 44ed2f0), release build, default 28x17/11 board, fixture net
  `botbowl-nn/tests/fixtures/tiny.onnx` on tract, 2 threads either way. The corpus is 30
  trajectories, 2008 samples, two shards, one next-drive pair: the first 30 lines of `dataset
  --mode random-start --games 300 --seed 6000 --mcts-iters 4 --evaluator heuristic --next-drive
  --max-steps 2000`. 8 playouts, seed 7. The hub run is a test hub on :13339 with one
  `botbowl-worker --parallel-games 2` and `--chunk-samples 32` (80 items). The box was running the
  live loop (load 12–17).
- **Result:** local 218 s, hub 225 s (+3%, within noise). Both shards byte-identical. An item ships
  ~2% of its line's size on this board (616 KB → 14 KB zstd; ~7.5% on the loop's 16x9 corpus) and
  returns 4 bytes a sample.
- **Conclusion:** the hub adds no meaningful cost. Turning it on in the loop (`LABEL_VIA_HUB=1`)
  is a throughput question for the GPU box plus the laptop, to be read off the first generation's
  phase minutes against the local ~95.

### <a id="exp071"></a>exp071: the train step (rebalance, learning rate), run 2026-10-12

- **Question:** the loop's fine-tune barely moves its own validation loss (plan 058, gen07-22) and the
  policy absorbs ~+0.001 P(played) per generation. Does rebalancing the losses (the gradient probe:
  policy ~3x value at the trunk at τ=100, 4.2x at τ=50) or a larger step move the net more without
  costing the value head?
- **Setup:** `scripts/exp071_train_step.sh` (commit in the status file). Parent v9 gen26, the window the
  loop's gen27 trains on (gen25-27, all C1 corpora, MC labels), the loop's recipe with one change per
  arm: `tau50_vw36` (cq τ=50, value weight 0.36), `tau100_vw36` (value weight 0.36), `lr1e4` (lr 1e-4).
  Control: the loop's own gen27. The loop pauses for the arms (~2-2.5 h of GPU) and the script
  relaunches it. Scored by: the trainer's val movement, absorption on gen27's held-out shards (cq-100
  target for all), the value bench paired vs gen26.
- **Result (2026-10-12 08:52-10:47, commit 630c8d4):**

  | arm | best step | value bench vs gen26 | Δ log P(played) | Δ P(played) | Δ top-1 | Δ val MSE (gen27 held-out) |
  |---|---|---|---|---|---|---|
  | loop gen27 (control: τ 100, vw 0.25, lr 5e-5) | 10000 (ep 0) | −0.3% | −0.0026 | −0.0002 | −0.0003 | −0.0006 |
  | `tau50_vw36` | 7000 (ep 0) | **+1.2%** (worse) | +0.0022 | −0.0021 | −0.0001 | −0.0007 |
  | **`tau100_vw36`** | 29000 (ep 2) | **−1.5%** | −0.0026 | **+0.0011** | −0.0013 | −0.0008 |
  | `lr1e4` | 1000 (ep 0) | +0.6% | −0.0042 | −0.0016 | −0.0014 | +0.0011 |

  Within-run val still barely moves in every arm (val_policy step 250 → final: +0.0005 to +0.0011).
- **Conclusion:** **value weight 0.36 at τ=100 is better** (value bench −1.5% against the control's
  −0.3% on the same step, ~3 SE apart; the only arm with P(played) up; it kept improving into epoch 2).
  Adopted by the loop from gen28's training (`launch_plan058.sh`, `VALUE_WEIGHT=0.36`). **The rebalance
  does not rescue τ=50:** its value head still loses 1.2%, so τ=50's cost is the sharper target itself
  (it amplifies sub-0.1-TD Q noise), not gradient competition. **A larger step (lr 1e-4) is worse on
  every number:** the bottleneck is not step size. Next levers: the target's noise (sharpen only where
  the Q gap beats its uncertainty), net capacity.

### <a id="exp070"></a>exp070: the chance model (plan 061), run 2026-10-11, adopted

- **Status:** A1 + B1 started 2026-10-11 06:23 at the loop's plateau (gen21 drives not above gen18 on either board, value bench flat over gen19-21): `EXTERNAL_HUB=http://127.0.0.1:13337 NET=models/az_v7/bbnet_mix16x9v9_gen21.onnx ARMS="A1 B1" scripts/exp070_chance_model.sh`, on the loop's hub alongside gen22-23 (helper workers share it); gen21 both sides, cap 800 drives per board.
- **A1 result (2026-10-11 08:54):** catch-aware bounces under today's complete backup lose on both boards: 14x7 0.474 ± 0.012, 16x9 0.476 ± 0.012 (400 pairs each, ~2 SE below 0.5; the not-worse SPRT 0.47:0.5 stayed undecided at the cap). Not adopted alone. Consistent with the arm-0 telemetry: the wider bounce node eats budget in dice under `complete` (main line reaching the opponent's turn 73% → 43%). Revisit under the winning backup in C1.
- **B1 result (2026-10-11 09:21):** the partial chance backup wins 14x7, 0.552 ± 0.016 (H1, 121 pairs); 16x9 stopped at H0 after only 18 pairs (0.430 ± 0.045), too few to read. Follow-up B1b: 16x9 alone, 300 fixed pairs, no SPRT, seed 1: **0.517 ± 0.014** (not worse, +1.2 SE). Verdict: partial wins 14x7 and holds 16x9; it is the B winner for C1 (B2-B4 not run yet).
- **C1 vs the B winner (2026-10-11 11:30):** every roll model on top of the partial backup vs partial alone: 14x7 0.490 ± 0.015 (H0, 42 pairs), 16x9 0.506 ± 0.016 (H0, 151 pairs): the corrected dice are neutral once the backup no longer charges for width (they cost ~0.025 under `complete`, A1).
- **C1 vs today's control (2026-10-11 12:02):** every roll model + partial beats `gumbel16_f1000` on both boards: 14x7 0.567 ± 0.026 (H1, 60 pairs), 16x9 0.566 ± 0.027 (H1, 83 pairs). Remaining adoption check: `net_check.sh` under `cfgs/chance_all_partial.toml` (gen22 on gen23, paired with the loop's own control check on the same decisions), running.
- **Net check under C1 (2026-10-11 13:30):** `scripts/net_check.sh models/az_v7/bbnet_mix16x9v9_gen22.onnx runs/loopmix16x9v9/gen23 runs/exp070/net_check_c1/gen22_on_gen23 cfgs/chance_all_partial.toml` (same seeded decisions as the loop's control check): +0.001/+0.003/+0.006/+0.011 at 64/250/1000/4000, MONOTONE, slope 0.70, against the control's +0.001/+0.002/+0.006/+0.008, slope 0.61.
- **Conclusion: adopted.** The loop generates with `cfgs/chance_all_partial_gen.toml` and benchmarks (both sides) and net-checks with `cfgs/chance_all_partial.toml` from the relaunch after gen24's generate (first C1 corpus: gen25). Not run: B2-B4 (mass, sampled, widen), A3 alone, A2 (passes). Lesson: the corrected dice only pay once the backup stops charging for chance width; the partial backup is the win (14x7 0.552, 16x9 0.517 alone).
- **Question:** the search still scripts three rolls.
  - A bounce onto a player is dropped, although the engine makes a standing player try to catch
    it.
  - An inaccurate pass's scatter goes up x3, and a wildly inaccurate pass's deviate goes 1 up.
  - The throw-in takes the shortest in-bounds throw.

  Do exact (grouped) models of these help? And does any alternative to "a chance node has no value
  until every outcome is scored" (`partial`, `mass` 0.9, `sampled`, `widen`) beat the shipped
  `complete` backup?
- **Setup:** `scripts/exp070_chance_model.sh`; code at the plan 061 branch (3bbcfc8..1364275,
  hub protocol v18).
  - **Bots:** v9 gen13 (`models/az_v7/bbnet_mix16x9v9_gen13.onnx`) on both sides, 1000 descents.
    Each arm is a `cfgs/chance_*.toml` against `cfgs/gumbel16_f1000.toml`.
  - **Drives:** paired contested drives on `runs/loopmix16x9g/positions/contested_{14x7,16x9}_gen04g.json`,
    at most 800 per board.
  - **SPRT:** A1-A3 (roll models) non-inferiority `0.47:0.5`; B1-B4 (backup modes) `0.5:0.55`.
    C1 (every roll model under the B winner) runs `0.5:0.55` against the winner, then `0.47:0.5`
    against the control.
  - **CPU:** `scripts/perf_search_bench.sh` per arm.
- **Result (arm 0, telemetry only):**
  - **Real games.** `botbowl-ui roll-census` (16x9 build at the branch) over
    `runs/loopmix16x9v9/gen13/shard{0,1}.jsonl`: 1449 drives, 0 diverged. Per drive: 1.30 live
    bounces, 0.18 throw-ins, 0.45 kickoff deviates, **0.001 passes**. Live-bounce landing mass:
    71.4% empty, 17.4% standing player, 3.2% downed player, 8.1% out. 21% of it is what the shipped
    model drops.
  - **In search.** gen13 on tract, 1000 descents, 6 drives per arm (seed 61000, 14x7/4 + 16x9/6,
    one thread), per searched decision.
    - Shipped: 36 bounce nodes (5.8 outcomes) and 7 throw-ins; 23% of chance backups withheld;
      main line reaches the opponent's turn 73%.
    - All roll models, complete backup: bounce passes ×3, 34% withheld, main line 43%.
    - All roll models, partial backup: 0% withheld; own decisions per line 3.85 → 4.39, chance
      share 45% → 38%, exact-outcome leaves 0.37% → 1.85%.
    - Wall time ~2.6 s per decision in every arm (net-bound).
- **Conclusion:** pending the drives. A2 (passes) is deferred: the loop's bots do not pass.

### Gradient balance: policy vs value at the shared trunk (plan 059 follow-up), 2026-10-09

- **Question:** how do the policy and value losses split the gradient into the shared 64x6 trunk,
  how does the cq τ change it, and does a decision's number of legal actions change its weight?
- **Setup:** `scripts/grad_balance_probe.py models/az_v7/bbnet_mix16x9v9_gen12.pt <dirs>` (CPU,
  BN in eval mode = `--freeze-bn`), the loop's loss exactly (policy CE mean over samples + 0.25 ×
  per-drive-weighted value MSE). Data: gen13's held-out shard 4 (MC-labelled), prepared with
  `target/16x9/release/prepare` (built at 71827c9) at `--policy-target cq --tau 100|50|20
  --value-blend 1.0`: 20,285 samples.
- **Result:**

  | τ | KL(target‖net) | trunk grad norm, policy : 0.25·value | cosine |
  |---|---|---|---|
  | 100 | 0.070 | 0.455 : 0.155 = **2.9** | +0.02 |
  | 50 | 0.144 | 0.646 : 0.155 = **4.2** | +0.03 |
  | 20 | 0.367 | 1.023 : 0.155 = **6.6** | +0.03 |

  Policy CE is per decision (one cross-entropy over the legal set, averaged over samples), so a
  2-action decision weighs as much as a 100-action one; its gradient (|p − π| per sample) is
  smallest on 2-action decisions (22% of samples, 0.049 at τ=100) and largest at 3-20 actions
  (0.11). Optimiser: Adam (no weight decay, no clipping).
- **Conclusion:** the policy already supplies ~3/4 of the trunk gradient at τ=100; sharpening τ
  only grows it. The two gradients are near-orthogonal, so they do not cancel; they compete
  through Adam's per-parameter normaliser (the value term's effective step in the trunk shrinks
  ~28% at τ=50, ~54% at τ=20) and the trunk's capacity. Consistent with exp069 (τ 30 and Gumbel σ
  lose the value gain) and the τ=50 loop (gen11 value bench +1.1%, net-check gain fell).

### <a id="exp069"></a>exp069 — policy targets under the mean backup (plan 059), 2026-10-08

- **Question:** the loop's policy absorbs little per generation (Δ P(played) +0.001..+0.004). Does
  a sharper target (cq τ 50, 30, Gumbel σ) absorb more without hurting the value head?
- **Setup:** `scripts/exp069_targets_policy_only.sh` (script at 81b5842; `prepare` binary built at
  43a45c6, `botbowl-ui` at 3c25109; trainer `train/` unchanged across those). Parent v9 gen06; the
  window the loop's gen07 trained on (gen05-07, MC-labelled, train shards 0-3/5/6, val 4/7); the
  loop's recipe (lr 5e-5, 3 epochs, `--freeze-bn --eval-at 250,500`, select on combined,
  `--value-weight 0.25 --per-drive-value-weight`, `--value-blend 1.0`), only the policy target
  changed. Scored by: absorption on gen07's held-out shards prepared once at cq τ=100; the v9 value
  bench paired vs gen06; policy-only drives vs gen06, 600 pairs per board.
- **Result:**

  | arm | Δ log P(played) | Δ P(played) | Δ top-1 | value bench | policy-only mean |
  |---|---|---|---|---|---|
  | loop gen07 (τ 100) | +0.003 | +0.002 | +0.002 | −1.4% | 0.500 ± 0.005 |
  | τ 100 re-run | +0.002 | +0.001 | +0.001 | −1.7% | 0.502 ± 0.005 |
  | **τ 50** | **+0.016** | −0.001 | +0.002 | **−1.4%** | **0.509 ± 0.005** |
  | τ 30 | +0.028 | −0.010 | +0.001 | ±0% | 0.496 ± 0.007 |
  | Gumbel σ (c_scale 0.1, min_range 50) | −0.050 | −0.090 | −0.012 | +1.7% (worse) | 0.502 ± 0.007 |

- **Conclusion:** the τ=100 control reproduces to ~0.001, so the quick numbers are low-noise.
  Sharper τ raises log P(played) but costs mean P(played) and, through the shared trunk, the value
  head (τ 30 loses all of the value gain, Gumbel σ goes backwards). τ=50 keeps the value gain and
  is the only arm above its parent in policy-only play (+1.7 SE: a lead). **Adopted by the user:
  the loop trains on cq τ=50 from gen09.** Re-read via policy-only lineage at gen11-12.
- **Read-out (2026-10-09, plan 059 §7):** three τ=50 generations moved the bare policy by −0.004
  (gen11 vs gen08 0.496 ± 0.006) against +0.007 for the last three τ=100 steps (gen08 vs gen05
  0.507 ± 0.006): the one-step lead did not replicate. Recommendation: back to τ=100.
- **Side finding:** one-step policy-only drives are blunt (the argmax moves in ~0.2% of
  decisions); compare across several generations instead.

### Policy-only lineage (plan 059 §5), 2026-10-08

- **Question:** does the loop's bare policy improve over generations at all (τ=100)?
- **Setup:** `runs/exp069/po_lineage/run.sh`, botbowl-ui at 3c25109, policy-only both sides,
  600 pairs per board.
- **Result:** gen07 vs gen04 0.512 ± 0.006; gen07 vs the init net 0.518 ± 0.007 (MATCHES.md).
- **Conclusion:** yes, ~0.003-0.004 per generation at τ=100; invisible in a single step.

### 14x5 edge-bounce chains (plan 060 §6), 2026-10-08

- **Question:** why do 14x5 searches in gen09 average 63 plies (93% chance) when every other board
  averages ~9?
- **Setup:** `botbowl-ui/examples/deep_search_probe.rs` replays a corpus sample from its seed and
  dumps the deepest line (e.g. seed 19000196 drive 1 sample 7); 5 corpus positions at 1000
  descents, heuristic and gen08 (tract); a 24-drive 14x5 heuristic corpus (seed 777) before/after;
  `scripts/perf_search_bench.sh` on 16x9.
- **Result:** the search's scripted throw-in (`roll_outcomes::throw_in_outcome`, the shortest
  throw, 2D6 = 2) is `2 / scatter_divisor 3 = 0` squares on a 5-wide axis: the ball lands where it
  was thrown from, bounces out again, forever (not a DAG cycle: `bounce_squares` grows, so states
  never repeat). Fix: the shortest throw that leaves the origin square (f8aa794). 14x5: depth 49.7
  → 6.8 plies, chance 94% → 55%, turn-ending reach 51% → 83%, 98 → 37 ms per decision; the
  searches chose the same moves on the 5 probes. 16x9 games byte-identical. Also fixed: the debug
  teardown assertion, `Node::move_root` left a second root edge to the new root (526726a).
- **Conclusion:** a real search-model bug, live from the gen10 relaunch; gen01-09's 14x5 samples
  carry it (DATA.md). Only boards with a 3-5 square narrow axis play differently.

### Tree statistics, first live reading (plan 060), 2026-10-08

- **Question:** how deep does the search see, and do its lines reach the opponent's turn?
- **Setup:** `runs/plan060/budget_ladder.sh`: botbowl-ui built at 759e5b7 (`target/tree-16x9`),
  v9 gen06 on tract, `dataset --mode random-start --next-drive --games 12 --seed 60000
  --bot-config cfgs/gumbel16_f1000_gen.toml --parallel-games 3` + the loop's size flags, at
  `--mcts-iters` 250 / 1000 / 4000; `scripts/tree_stats.py` on each.
- **Result:** mean leaf depth 6.5 / 8.6 / 10.8 plies; share of descents reaching the opponent's
  turn 52% / 61% / 74% (87% at 1000 for the turn-ending decision); main line reaches it 58% / 73% /
  86%; ~half of every line is dice; ~3 own decisions per line at any budget.
- **Conclusion:** healthy on plan 060 §3's terms (depth and reach grow with budget; turn-ending
  decisions see the opponent's reply). Extra budget buys opponent-turn and dice depth, not deeper
  own planning. Open: separate a turnover-ended opponent turn from a played-out one.

### Generation throughput after the search CPU cuts (plan 058 §9), 2026-10-07 night

- **Question:** how many generation streams, and how many GPU sidecars, now that the search's CPU
  cost fell 76% (§7) and `GameState` 4x (§8)?
- **Setup:** `scripts/perf_gen_steady.sh OUT models/az_v7/bbnet_mix16x9v9_gen05.onnx "<streams>"
  1` (`SIDECARS=N` for N sidecars): the loop's generation with `--next-drive`, warm-up skipped,
  decisions counted in a 10 min (5076090) or 8 min (96fb083) window.
- **Result:** one sidecar saturates near 8k samples/s (GPU ~80%); two sidecars at 48 streams fill
  the GPU (8.55k/s, +9%); 64 streams add nothing; 48 streams use 4.4 GB on 96fb083 (6.5 GB before).
  In the loop gen06 made 1766 decisions/min, but ~40% of that was the laptop (10 streams on its
  own CPU): without it (gen08-09) the box makes ~1080/min, the benchmark's rate (plan 058 §9).
- **Conclusion:** generation is GPU-bound; the loop runs 48 streams over two sidecars and spends
  the speedup on data (400 games per shard).

### Forced moves in the corpus (plan 058 §6), 2026-10-07

- **Question:** how much search went into decisions with one action left after pruning?
- **Setup:** count of samples with < 2 children in v9 gen04 (the last corpus from the old bot).
- **Result:** 30,133 of 150,520 samples (20.0%); MoveAction 59% of them, Kickoff 100% forced.
- **Conclusion:** the bot plays them without search and `prepare` drops them (3dc4902, 2afd0db);
  gen04's window shrank 21%.

### Stream count before the CPU cuts (plan 058 §2), 2026-10-06

- **Setup:** `scripts/perf_gen_bench.sh`, 4 games per thread (the first pass, games ≈ threads, was
  wrong: tail effect).
- **Result:** CPU-bound from ~36-48 streams (17-20 games/min at 48); two sidecars +7%; mc-label
  per-sample work 2x (683fb34).
- **Conclusion:** 36 streams then (memory-capped), later superseded by §9.

## Pre-v9 rules (pointers; details in the plans)

| exp | plan | question | conclusion |
|---|---|---|---|
| exp068 | 056 | do MC-labelled values (arm F) play better? | F's search beats A's 0.533 ± 0.009 and its own policy 0.582 ± 0.011 at 1000 descents: **adopted** |
| exp067 | 056 | value labels: blend 0.5 / 1.0, TD(λ) 0.8 / 0.95, value weight 1.0, MC-averaged | MC-averaged (F): value bench RMS −15.5%, bias ~0; TD(λ) only removes bias |
| exp066 | 055 | does more search beat the bare policy, under the mean backup? | 0.525 / 0.542 / 0.590 at 250 / 1000 / 4000 descents: the criterion holds (mean backup hardcoded, ce4eda1) |
| exp065 | 055 | override audit: are the search's overrides of the prior right? | value head RMS 0.26-0.28 TD vs MC truth; overrides on Q gaps < 0.1 TD realise nothing |
| exp063 | 054 | train step: cq τ 50 vs 100 at lr 5e-5 | τ 100 at 5e-5 passed the gate (0.531), τ 50 0.510; search did not beat its policy (E5 0.517) |
| exp059 | 049/050 | policy targets (PUCT era) and WDL value head | no target beats cq τ 100; WDL step 1 fails |
| exp056 | 049 | τ and search budget on drives | see plan 049 |
| exp055 | 049 | why the vl0 loop regressed (τ, budget) | τ 100 recovers 14x7; aborted partial |
| exp054 | 049 | virtual loss 0 vs 30 | vl0 wins 0.686; the virtual-loss leak fixed (5e29d89) |
| exp048 | 048 | exploration A/B | see plan 048 |
