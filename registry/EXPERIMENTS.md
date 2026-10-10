# Experiment registry — question, setup, result, conclusion

Newest first. Each entry: the question, how to reproduce it (commit, script, settings), what came
out and what was decided. Full write-ups live in the plans; head-to-head numbers are also in
[MATCHES.md](MATCHES.md), per-net numbers in [NETS.md](NETS.md), corpora in [DATA.md](DATA.md).
**Keep this current:** add the entry when an experiment starts, the result and conclusion when it
ends (see the root CLAUDE.md).

## v9 rules (2026-10-06 on)

### MC labels through the hub: equivalence and overhead (plan 062), 2026-10-10

- **Question:** does `botbowl-hub job label` (plan 062, protocol v19) write what
  `botbowl-ui mc-label` writes, and what does the hub cost?
- **Setup:** branch head of plan 062, release build, default 28x17/11 board, fixture net
  `botbowl-nn/tests/fixtures/tiny.onnx` on tract, 2 threads either way. The corpus is 30
  trajectories, 2008 samples, two shards, one next-drive pair: the first 30 lines of `dataset
  --mode random-start --games 300 --seed 6000 --mcts-iters 4 --evaluator heuristic --next-drive
  --max-steps 2000`. 8 playouts, seed 7. The hub run is a test hub on :13339 with one
  `botbowl-worker --parallel-games 2` and `--chunk-samples 32` (80 items). The box was running the
  live loop (load 12–17).
- **Result:** local 218 s, hub 225 s (+3%, within noise). Both shards byte-identical. An item ships
  ~2% of its line's size (616 KB → 14 KB zstd) and returns 4 bytes a sample.
- **Conclusion:** the hub adds no meaningful cost. Turning it on in the loop (`LABEL_VIA_HUB=1`)
  is a throughput question for the GPU box plus the laptop, to be read off the first generation's
  phase minutes against the local ~95.

### <a id="exp070"></a>exp070: the chance model (plan 061), planned 2026-10-09

- **Status:** planned, not run.
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
