# First trained nets, the 14x7 bootstrap problem, and the road to gen-1

**Status:** In progress (started 2026-07-17). Learnings recorded through the first 14x7-native net; next steps at the bottom are being executed in order.

This plan captures everything learned from the first real data-generation + training session (2026-07-17/18): the pure-TD experiment on 8x3, the small-board bug harvest, the search-budget sweep on 14x7, and the failure of both nets (8x3-transfer and 14x7-native gen-0) as search evaluators on 14x7.

## What was built (all committed on master)

- `Evaluator::PureTd` (`botbowl-mcts`): leaf = `anchor.score_delta(state).clamp(-1,1) * 1000`, anchor-relative, no shaping. Scripted priors kept. `dataset --evaluator {heuristic|pure-td|nn}`, `--model PATH` for `nn`.
- Per-drive `outcome_value` backfill (`botbowl-data`), replacing the broadcast final-scoreline `z_home` (plan 017 caveat closed). `NN_SCHEMA_VERSION = 2`.
- Corpora (gitignored, commit-stamped):
  - `data/smallboard_8x3/` — 960 games, pure-td@1000 iters, 82,490 samples → 77,331 prepared. 3.24 TDs/game, targets −1/0/+1 = 33/31/36%.
  - `data/board_14x7/` — 960 games, heuristic@1000 iters, 104,819 samples → 104,696 prepared. 2.90 TDs/game, targets 30/25/45%.
- Nets (gitignored): `models/bbnet_8x3_gen0.{pt,onnx}` (value MSE 0.21, top-1 0.74 — training metrics), `models/bbnet_14x7_gen0.{pt,onnx}` (value MSE 0.167, top-1 0.574 — training metrics).

## Bug harvest — the tiny board as a fuzzer

Four latent bugs surfaced within the first hours of 8x3 generation, all fixed with regression tests (commits `508b6d2`, `89e8f82`, `4bf4bb7`):

1. **Crowd push picked the occupied straight square** instead of the OOB diagonal (`get_push_squares` popped the last candidate blindly) → `move_player` panic. Reachable on the full board via sideline sandwich.
2. **Scripted throw-in cycle**: the constant `ThrowIn{One, Two}` chance outcome lands OOB on a 3-row pitch; re-request states oscillate between two boundary squares → genuine search-DAG cycle. Fix: state-aware pick that lands in bounds (`ThrowIn::target_square` + `GameState::proc_stack_peek`).
3. **Touchback with zero receivers** (all injured — routine at 2v2) emitted an empty `NeedAction` → MCTS root terminal ("no move info"). Fix: ball drops at the receiving half's aim square and bounces.
4. **Recursive tree drop** in `recon_mcts::on_drop` overflowed the stack on deep DAGs. Fix: iterative worklist + `detached` flag — with the subtlety that only last-reference children may be parked on the worklist, or diamond recombinations skip state materialisation (`tests/deep_drop.rs` pins both).

**Learning:** every new board size / player count is a fuzzing campaign. Expect the 20x11 tier to surface its own crop; budget for it.

**Gotcha (memory-worthy, also in auto-memory):** MctsBot games are not reproducible from seeds — recon_mcts's std HashMaps randomize tie-break order per process. Reproduce rare crashes by looping batches, not replaying a seed.

## Experimental results

### 8x3 (playable 8x3, 2v2, MA 6): pure TD works

A TD is inside the search horizon (own turn + opponent reply, plan 014) from essentially every square (MA 6 + 2 GFI ≥ board length), so the unshaped ±1 signal is dense: **3.24 TDs/game**, 12% scoreless, near-uniform value targets. 84.8% of decisions have all-tied children Q, but many tie at +1000 ("all moves win") — correct, not blind.

### 14x7 (4v4, MA 6): the horizon is structural, search budget is not the answer

Random-start self-play, same seeds per arm, 1000 iters unless noted:

| arm | TDs/game | scoreless | Q discriminates | TD visible at root |
|---|---|---|---|---|
| pure-td @1k | 1.58 | 3/12 | 6.3% | 2.3% |
| pure-td @4k | 1.33 | 4/12 | 11.4% | 2.2% |
| pure-td @16k | 1.88 | 3/8 | 10.9% | 2.4% |
| heuristic @1k | **2.83** | 2/12 | 44.0% | ~80% |
| nn: 8x3 net @1k | 0.50 | 9/12 | 70.1% | 32.5% |
| nn: 14x7 gen-0 net @1k | 0.75 | 6/12 | 68.2% | — |

Key learnings:

1. **TD visibility is flat (≈2.3%) across a 16× iteration budget.** The horizon anchor makes anything past ~1.5 turns terminal; no budget extends sight. On 14 columns with 4 defenders, pure ±1 leaves are zero at ~98% of roots. More search re-explores a flat landscape. *Shaping or a learned value function is required at this tier and above.*
2. **The 8x3 net transfers mechanically (fully-conv) but not semantically.** It learned "possession ≈ certain score" (true on 8 columns, false on 14), discriminates confidently (70%) and stalls: 9/12 scoreless.
3. **The 14x7-native gen-0 net also fails as an evaluator (0.75 TDs/game)** — and the damning comparison is pure-td (1.58): the NN path keeps the exact known-outcome carve-out, so it has strictly more information than pure-td, yet scores half as often and plays the longest games. Its estimates actively steer away from scoring lines (over-valuing safe possession is the leading suspect), compounded by covariate shift once it stalls into states the heuristic teacher never produced.
4. **Confound:** `Evaluator::Nn` replaces *both* leaf value *and* scripted priors. Gen-0 policy targets contain a lot of near-uniform tied-root distributions, so learned priors may be flat/noisy vs. the hand-tuned scripted ones. Value vs. prior blame is unresolved → the hybrid experiment below.
5. **Training metrics are training-only.** The trainer has no validation split; MSE 0.167 / top-1 0.574 say nothing about generalization. Flying blind until fixed.
6. **960 games was arbitrary** (8 shards × 120, sized for wall-clock). Samples within a game are heavily correlated; the independent unit is the game. 16x9 4v4 state space dwarfs 10x5 2v2, so coverage per game collapsed exactly when it mattered. Data quantity is plausibly binding on 14x7. Generation is the cheap knob (~45 min / 960 games at 8-way parallelism).
7. **Costs:** heuristic ≈ 18 s/game, pure-td ≈ 15 s, NN ≈ 30–70 s (two tract forwards per expanded node). 16k iters ≈ 6 min/game — not worth it.
8. **Board-stat discrepancy vs plan 017's tier table:** the table prescribes MA 4 at 14x7 / MA 2 at 8x3; `generate_random_start` fields default MA 6 linemen. MA 6 is *why* pure-td worked at 8x3 (whole board in one activation). Lower MA would make the horizon even more binding. Treat MA as an open curriculum knob, not an accident.

## Next steps (in execution order)

1. **Hybrid evaluator experiment — isolate value vs. priors.** ✅ **Done (2026-07-18): the value head is to blame.** `Evaluator::NnValue` (NN leaf value + scripted priors) scored **0.83 TDs/game** — indistinguishable from full NN (0.75), far from the heuristic (2.83). Learned priors are not the bottleneck; the gen-0 value head actively steers the search away from scoring lines. Consequences: policy-target hygiene drops in priority; value-target quality and data volume rise.
2. **Validation split by shard, not by sample.** ✅ Done: trainer gained `--val-data` / `val_dir` (evaluated per epoch, no augmentation). Shard 7 (13.1k samples) held out; train subsets prepared at 2/4/7 shards. Sample-level splits leak (adjacent states are near-duplicates) — hold out whole shards.
3. **Data-scaling probe.** ✅ Done (2026-07-18). Two findings:
   - **Severe overfitting from ~epoch 5** in every run: best val value-MSE at epoch 3–6, degrading afterwards while train MSE keeps falling (0.15 train vs 0.50+ val at epoch 29). The gen-0 exports used in the eval arms were epoch-29 — deep in the memorization regime.
   - **Data helps with a shallow slope:** best val value-MSE 0.436 (27k samples) → 0.423 (52k) → 0.405 (91.6k), against target variance ≈ 0.73 (best R² ≈ 0.45). Data-limited, but 3.4× data bought only ~7% — label quality matters too. `val_top1` flat ≈ 0.55 at every scale.
4. **Early-stopped retrain (`bbnet_14x7_gen0b`, 6 epochs, val 0.40–0.41)** ✅ doubled the eval arms: nn 0.75 → **1.50** TDs/game, nn-value 0.83 → **1.67**. The net now ties pure-td instead of losing to it; still trails the heuristic teacher (2.83).

**Decision (probe + hybrid combined): gen-1 needs *more data* AND *better value targets* AND *early stopping as standard practice*.** Priors are fine (hybrid ≈ full-NN at both checkpoints). Concretely, in order:
   1. Scale 14x7 generation to ~5k games (heuristic evaluator, ≈4 h at 8-way parallelism), retrain early-stopped against the val shard.
   2. Solved-root exact value targets (plan 017's gold-standard note): use `root_value` where `root_solved`, drive outcome otherwise — cuts label noise where the search proved the answer.
   3. Light regularization (weight decay) to push the early-stopping point later; consider mixing the 8x3 corpus into training (multi-dims batches) against the possession-overvaluation bias.

### Result of the data scaling (`gen0c`, 2026-07-18): val metric moved, play strength did not

~4.8k games → 520,462 prepared samples (three shards truncated by an OOM kill — `spatial.npy` is 10 GB on a 16 GB machine; trainer now mmaps it and persists the best checkpoint to disk on every improvement). Best val value-MSE **0.359 @ epoch 5** (vs 0.405 at 91.6k samples). But the eval arms are flat vs gen0b: nn 1.25, nn-value 1.58 TDs/game (gen0b: 1.50 / 1.67; 12-game arms have ±~0.5 noise).

**Learning: val-MSE gains have decoupled from on-pitch strength.** The residual value error is probably not the kind data volume fixes:
   - *Label-noise floor*: drive outcomes under 1000-iter play are inherently stochastic; Bayes-optimal MSE on this target is well above 0. We may be approaching it.
   - *Calibration mismatch is baked into the labels*: "possession → usually scores" is TRUE under heuristic play (2.9 TDs/game) — the net learns the teacher's conditional, which breaks under its own (different) play. More heuristic data reinforces the same conditional. This is the covariate-shift argument for moving toward on-policy (mixed / self-play) data rather than more teacher data.

Next levers, reordered accordingly: **(a) head-to-head A-vs-B harness** (real strength measurement + bigger arms before any further conclusions), **(b) solved-root exact value targets**, **(c) mixed-corpus generation** (heuristic + nn games) to start closing the on-policy gap — see the gen-1 switch criteria below.

### Gen-1 switch criteria (agreed 2026-07-18)

Switch generation from heuristic to the net (or start shifting the mix) when, on the current tier:
1. **Head-to-head NN bot ≥ ~50% vs the heuristic bot** (alternating Home/Away, 50+ games) — build the harness first; and
2. **Self-play data health:** ≥ ~2 TDs/game and ≤ ~25% scoreless in NN-vs-NN games (keeps value-target classes balanced).

Parity suffices — don't wait for dominance: at parity, self-play data is no worse per game and strictly better in kind (on-policy, bias-free). Below parity, run mixed generations (e.g. 60/40 heuristic/nn, shifting per generation). Per-generation promotion gate once the loop runs: new net vs previous net ≥ 55%.

## Next-next steps (decision points after the above)

- **Better value targets:** plan 017 flags solved-root exact values as gold-standard. Blend: use `root_value` for solved roots, drive outcome otherwise. Reduces bootstrap noise where the search *proved* the answer.
- **Policy-target hygiene:** tied-root samples currently produce near-uniform π that may wash out the policy head. Options: drop all-tied samples from the policy loss (keep for value), or temperature-sharpen by Q.
- **Gen-1 loop on 14x7:** once an evaluator arm beats the heuristic teacher (or at least matches it), switch generation to `--evaluator nn` and iterate: generate → train → eval vs. previous gen. Define the promotion criterion *before* running (e.g. TDs/game ≥ teacher and head-to-head win rate > 55%).
- **Head-to-head eval harness:** TDs/game in self-play conflates offense and defense. A `dataset`-style A-vs-B mode (net A as Home, net B as Away, alternating) would measure real strength. Needed before any promotion decision.
- **Intermediate rung (12,7,4) = playable 10x5, 4 players** if 8x3→14x7 transfer keeps failing after data scaling: halves the semantic gap (same player count as 14x7, board closer to 8x3).
- **MA as curriculum knob:** revisit plan 017's MA 4/2 prescriptions. Lower MA lengthens drives (more multi-turn credit assignment pressure on the value head), higher MA densifies reward. Possibly: start MA 6, anneal down.
- **Known remaining engine risk:** the pass/handoff catch variant of the recurring-state pattern (see auto-memory `mcts-chance-cycle-hang`) is closed via `Catch::on_start`, but any *new* catch-like proc must go through `Catch` or replicate the hook.
- **Perf (deprioritized, note only):** NN generation is 2–4× slower per game; if gen-1 needs 10k games, either batch leaf evaluation or accept the wall-clock.

## Cross-references

- plan 017 (progressive board sizes) — tier table, architecture, solved-visits caveat, z_home caveat (closed).
- plan 018 (no intermediate leaf scoring) — chance-node semantics the evaluators plug into.
- plan 019 (random-start generation) — the state generator all corpora use.
- plan 014 (horizon bound) — the structural constraint that dominates everything above.
