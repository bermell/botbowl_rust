# Plan 070: a deterministic search, and the stale value in recon_mcts backprop

**Status:** built 2026-10-24 (plan 069 work item 2), review pending. Experiment exp084. Branch
`worktree-agent-a0e0f48d44b5b99ff` (178de78 … see the registry entry).

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
   of the 20x11 contested set (heuristic, 300 descents) with the children maps fixed but the
   `parents` sets and registry still randomly keyed: **0 decisions differ in any root statistic**
   between two copies of the bot, 0 depth inversions in 788k audited edges. Several threads can
   race a depth update against a walk; the loop searches on one thread (Gumbel root, `workers = 1`),
   so that source is accepted and documented.
2. **The nondeterminism was the children map's order.** Each children `HashMap` used
   `RandomState`, so its iteration order differed per map and per process. Three things read it:
   `tie_break = "hash"` (shipped: every exact tie, at the Gumbel root and below); `mover`, which
   was not total for chance outcomes (`chance_key` gave every non-directional outcome one key, so
   equally likely unscored outcomes were swept in map order); and the chance backup's f64 sum,
   sorted by a `|dx|`-folded key, so mirrored directions were summed in map order.
3. **A fixed order is a biased order.** Making `mover` total fixed the order among equally likely
   chance outcomes: the same one is swept first at every node, and under the partial backup the
   first scored outcomes are the chance node's value (for block dice, the skull came first). Played
   against the old search: **0.475 ± 0.022 over 100 pairs**. FxHash makes the map order fixed in
   the same way. What the old `hash` had right was a *different* random order at every node.
4. **There is a stale value, and it is a twin splice.** An audit of the DAG after every search
   (`MctsBot::audit_dag`: recompute every expanded node's backup from its children) found ~0.3% of
   nodes holding a value their children no longer give. Per-descent auditing traced two causes:
   - *A twin spliced under a parent* (`materialize_placeholder`'s `Twin` arm): the parent gains a
     possibly scored child, the descent continues into the twin, and if its backprop is cut below
     the twin (an unscored chance node withholds), the parent never sees its new child.
     `recon_mcts/tests/twin_splice.rs` pins it (node 2 kept 200 next to a child worth 1000). A
     correctness bug, deterministic but wrong.
   - *Visit lag*: `select_node` bumps the chosen child's visits in place, and a descent whose
     backprop is cut below never re-weights that child's parents. This is the documented "visits
     lag cut-off descents"; it moves a visit-weighted mean by a point or a few (up to ~50 when the
     children's values are far apart), is healed by the next backup through the node, and is
     deterministic. Not changed here: re-seeding the walk at every path node would also revert the
     bump, so a cut descent would leave no trace and the next descent would retrace it. A separate
     experiment if wanted.
   Every edge is mirrored in its child's parent set (0 unlisted parents in 262k edges).

## Design (built)

- **recon_mcts:** a descent records the parent of every twin it splices in, and its backprop is
  seeded there too (`Node::backprop_scores_from`: one heap walk from all seeds, each walked past
  unconditionally). `GameDynamics::back_up_spliced_parents` (default `true`) turns it off for A/B
  runs. Nothing else in the walk changes. Every hashed container (children maps, `parents` sets,
  registry, visited sets, `UniqueHeap`) uses a fixed FxHash (`fx_hash.rs`, exported); `Node::hash`
  stays SipHash. `deterministic_hash` is a no-op.
- **botbowl-mcts:** `tie_break = "hash"` orders every exact tie by `tie_hash(seed, action)`, the
  seed the node state's hash (`TieCtx`, computed only when a tie needs it; FxHash + splitmix64):
  the old per-node random order in distribution, a pure function of the state. The Gumbel root's
  children are ordered by every tie-break. `mover` is total and mirror-covariant (for mirror
  experiments). The chance backup sums in `chance_term_order`. `MctsConfig.backprop = "fixed" |
  "legacy"` (default `fixed`); hub protocol v23.
- **Presets:** none to switch. The shipped `chance_all_partial{,_gen}` (`hash`) are deterministic
  on the new binary and get the twin fix by default; `chance_all_partial_legacy` is the old search
  for A/B. `chance_all_partial_det` (`mover`) is superseded.

## Gates (exp084)

1. Repeatability: `botbowl-play/tests/search_repeat.rs` (two copies of a bot in lockstep, every
   root statistic compared, plus the DAG audit) under `hash` and `mover`; `drive-agree` gen48 vs
   gen48 on `contested_20x11_gen48`, null net and in-process gen48 (1 and 16 streams).
2. Strength: new vs legacy search on gen52, paired contested drives (100 pairs on 20x11, 200 on 16x9); must not lose.
   `scripts/net_check.sh` on gen52 must stay monotone.
3. CPU: instructions per descent before/after (`scripts/perf_search_bench.sh`, 20x11, null net and
   heuristic), paired where the games are identical.

Results: `registry/EXPERIMENTS.md#exp084`.
