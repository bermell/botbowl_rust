# Plan 070: a deterministic search, and the stale value in recon_mcts backprop

**Status:** in progress (2026-10-24), plan 069 work item 2. Experiment exp084. Branch
`worktree-agent-a0e0f48d44b5b99ff`.

## Goal

The same (position, seed, net) must give the same search, and therefore the same move, in any
process: across reruns, across the thousands of workers of the HPC allocation, and for two bots on
one net (exp079: gen48 agreed with itself on only 0.937 of searched decisions at 1000 descents,
0.959 with `tie_break = "mover"`). And the suspected stale value in recon_mcts's backprop must be
found or ruled out, with a test.

## Findings

1. **The equal-depth ordering hypothesis is wrong.** exp079 suspected that `backprop_scores`, which
   walks a heap ordered by depth only, recomputes a parent before a transposed child that sits at
   its depth, so the parent keeps a stale value, in an order set by the `parents` `HashSet`. On one
   thread that cannot happen: `set_min_depth` keeps every node one deeper than its deepest parent
   (raised on every new edge, before the backprop), so every edge goes strictly deeper, the walk
   pops a node only after every deeper node of the walk, and equal-depth nodes are never parent
   and child, so their order cannot change a value. Measured: `recon_mcts/tests/backprop_order.rs`
   (a transposing DAG with steps of different lengths, audited after every step, 20 seeds × 1500
   steps: 0 depth inversions, 0 stale values) and, on the real search, 1187 decisions of 20 drives
   of the 20x11 contested set (heuristic, C1 + `mover`, 300 descents) with the children maps
   fixed but the `parents` sets and registry still randomly keyed: **0 decisions differ in any root
   statistic** between two copies of the bot, and 0 depth inversions in 788k audited edges.
   Several threads can invert depths (races between `set_min_depth` and a walk); the loop's search
   runs on one thread (Gumbel root, `workers = 1`), so that source is accepted and documented.
2. **The nondeterminism is the children map's order.** A children `HashMap` uses `RandomState`, so
   its iteration order differs per tree. Three things read it:
   - `tie_break = "hash"` (shipped): every exact tie, at the Gumbel root and below.
   - `tie_break = "mover"` was not total for chance outcomes: `chance_key` gives every
     non-directional outcome (D6 faces, block dice, injuries, pass/fail) one key, so equally likely
     unscored outcomes were swept in map order.
   - The chance backup's f64 sum is sorted by `canonical_chance_key`, which folds `|dx|`, so
     mirrored directions tied and were summed in map order (last-bit differences).
   With `mover` made total and the sum ordered by value as well: **0 of 1200 decisions differ**
   between two copies under production (random) hashing.
3. **There is a stale value, and it is a twin splice.** An audit of the DAG after every search
   (`MctsBot::audit_dag`: recompute every expanded node's backup from its children) found 0.3% of
   nodes holding a value their children no longer give. Per-descent auditing traced two causes:
   - *A twin spliced under a parent* (`materialize_placeholder`'s `Twin` arm): the parent gains a
     scored child, the descent continues into the twin, and if its backprop is cut below the twin
     (an unscored chance node withholds), the parent never sees its new child.
     `recon_mcts/tests/twin_splice.rs` pins it (node 2 kept 200 next to a child worth 1000). This
     is a correctness bug, deterministic but wrong.
   - *Visit lag*: `select_node` bumps the chosen child's visits in place, and a descent whose
     backprop is cut below never re-weights that child's parents. This is the documented "visits
     lag cut-off descents"; it moves a visit-weighted mean by a point or a few, is healed by the
     next backup through the node, and is deterministic. Not changed here (it would change which
     lines cut descents revisit; a separate experiment if wanted).
   Every edge is mirrored in its child's parent set (0 unlisted parents in 262k edges).

## Design

- **recon_mcts:** a descent records the parent of every twin it splices in, and its backprop is
  seeded there too (`Node::backprop_scores_from`: one heap walk from all seeds, each walked past
  unconditionally). `GameDynamics::back_up_spliced_parents` (default `true`) turns it off for A/B
  runs. Nothing else in the walk changes: depth is a correct topological key on one thread.
- **botbowl-mcts:** `TieBreak::Mover` is total on chance outcomes (`mover_chance_order`:
  `chance_key`, a scatter's other two directions in the acting side's frame, then the canonical
  key; mirror-covariant). The chance backup sums in `chance_term_order` (canonical key, then score,
  probability, visits, proof). `MctsConfig.backprop = "fixed" | "legacy"` (default `fixed`); hub
  protocol v23.
- **Presets:** `chance_all_partial_det` (eval) and the new `chance_all_partial_det_gen`
  (generation) are the deterministic arms; `chance_all_partial_legacy` is the old search.
  The shipped `chance_all_partial{,_gen}` keep `tie_break = "hash"`: switching them is the user's
  call (see exp084).
- **Then FxHash** (separate commit): a fixed fast hasher for the children maps, `parents` sets
  and registry. Order-independence makes it free for `mover`; under `hash` it makes the order
  fixed (deterministic, but a different arbitrary order).

## Gates (exp084)

1. Repeatability: `botbowl-play/tests/search_repeat.rs` (two copies of a bot in lockstep, every
   root statistic compared, plus the DAG audit), heuristic and `--nn-backend null`; in-process
   gen48 at one stream with cuDNN deterministic via `drive-agree`.
2. `drive-agree` gen48 vs gen48 on `contested_20x11_gen48`, before/after, at 1 stream; then at 16
   streams in-process (batches vary) for the numeric residue.
3. Strength: new vs legacy search on gen52, paired contested drives, SPRT 0.5:0.55; must not lose.
   `scripts/net_check.sh` on gen52 must stay monotone.
4. CPU: instructions per descent before/after (`scripts/perf_search_bench.sh`, null net, 20x11).

Results: `registry/EXPERIMENTS.md#exp084`.
