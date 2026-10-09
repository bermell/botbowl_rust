# Experiment registry — question, setup, result, conclusion

Newest first. Each entry: the question, how to reproduce it (commit, script, settings), what came
out and what was decided. Full write-ups live in the plans; head-to-head numbers are also in
[MATCHES.md](MATCHES.md), per-net numbers in [NETS.md](NETS.md), corpora in [DATA.md](DATA.md).
**Keep this current:** add the entry when an experiment starts, the result and conclusion when it
ends (see the root CLAUDE.md).

## v9 rules (2026-10-06 on)

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
