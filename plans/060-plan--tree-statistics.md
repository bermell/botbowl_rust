# Plan 060 — tree statistics: how deep does the search see, and into whose turn?

**Status:** Instrumentation built 2026-10-08 (§5); reading it on the live loop's corpora is next.

## 1. Why

The search's horizon (`HorizonAnchor`, `horizon_turns = 1`) ends a line when the bot's next turn
begins: one own turn plus the opponent's turn. Nothing measures how much of that window a
1000-descent search actually covers. If the lines from a turn-ending decision (the last
activation, `EndTurn`) never reach the opponent's turn, protecting the ball carrier and screening rest
entirely on the value head; if lines often run *through* the opponent's turn, the opponent's turn
may be cut short (the class of bug plan 049 found: armour = casualty, pass = fumble). Neither is
visible today.

## 2. What to record

**Per search** (per root decision), collected per *descent*, not per node: a count over nodes is
dominated by thousands of shallow one-visit leaves, a count over descents is where the budget went.

1. **Leaf depth** of every descent: plies from the root (decision and chance edges), and own
   decisions only (edges where the root's mover chose). Mean, p90, max.
2. **Where the descent ended**, relative to the root's anchor:
   - `own_turn`: still in the root mover's current turn;
   - `opp_turn`: in the opponent's following turn;
   - `horizon`: past the end of the opponent's turn (the bot's next turn began; the horizon leaf);
   - `score`: a touchdown ended it (terminal);
   - `half_end` / `game_over`: the half or the game ended.
   Also how the leaf was valued (new leaf scored by the net, terminal, solved, horizon).
3. **The main line** after the search: from the root, follow the most-visited child (and the
   most-visited outcome at chance nodes) to the end. Its length in plies and own decisions, and the
   phase (as in 2) it reaches.
4. **`opp_turn_follows`** at the root: `false` when the opponent has no turn after this one (the end
   of the half), so those decisions can be filtered out.

**Where it lands:** a `tree` block on each corpus sample (optional field, absent in old corpora),
and per-search means in the `telemetry` block that `report.json`, `eval.games.jsonl` and the
corpus `meta.extra` already carry (plan 043).

**Analysis** (`scripts/tree_stats.py`): stratify by the number of the mover's decisions left in
this turn (known from the trajectory afterwards), by procedure and by board, with
`opp_turn_follows = false` filtered out.

## 3. Reading it

| metric | healthy | act on |
|---|---|---|
| share of descents reaching `opp_turn`, at turn-ending decisions | a solid share (guess 30%+) | the main line never enters the opponent's turn: more search at turn ends, or better value labels there |
| share reaching `horizon` | rare except at the very last decisions | common: the opponent's turn is being cut short (a bug, or pruning too hard) |
| leaf depth vs budget (250 / 1000 / 4000) | rises | flat: extra budget only widens the tree (fans too wide, chance nodes eat it); pairs with plan 055's budget check |
| chance plies' share of depth | | high: the depth is dice variety, not decisions |

Comparable published numbers: Leela Chess Zero reports depth and seldepth (mean and max line
depth), visit-vs-prior divergence and terminal hits; MuZero-style analyses add main-line length,
root visit entropy and Q spread.

## 4. Constraints

- Observational only: the search must play byte-identical games (`scripts/perf_search_bench.sh`'s
  trajectory hash), and its instruction count must not move by more than ~1%.
- Recombination purity and the `Hash`/`PartialEq` invariant are untouched (no state fields).
- `recon_mcts` stays game-agnostic: any hook it needs is generic (e.g. the descent's path length
  handed to the dynamics).

## 5. Results

### What was built (2026-10-08)

- **`recon_mcts`**: a generic hook, `GameDynamics::observe_descent(edges, leaf, end)`, called once
  per descent from `descend` just before it returns. `edges` are `(player of the node left,
  action)`, root first, twin swaps skipped (one entry per ply); `end` is `DescentEnd::Expanded |
  Terminal | Solved`. Default no-op (nim, `DynGD` unchanged). Lean inspection accessors
  `Node::get_children` (no state clones), `Node::mover`, `Node::with_state`.
  `recon_mcts/tests/observe_descent.rs`.
- **`botbowl-mcts/src/tree_stats.rs`**: `DescentLog` (on the bot, shared with every tree it builds,
  reset per search) folds each descent into exact depth histograms (plies, own decisions), chance
  plies, the end phase, `reached_opp_turn` and the valuation. `leaf_phase` classifies against the
  root's `HorizonAnchor` (captured even with the horizon off): game over → score → half end → the
  mover's counter advanced `turn_depth` times (horizon) → the opponent's counter ahead (opp turn)
  → own turn. `main_line` walks the most-visited child (most-visited outcome at chance nodes) to
  a node with no visited child. `opp_turn_follows` replays the engine `Half` procedure's turn
  order (receiver first when the counters are level, `TURNS_PER_HALF = 8`, now a named engine
  constant): `true` iff the next turn start inside the horizon is the opponent's. Pinned against
  random games in `botbowl-mcts/tests/tree_stats.rs`.
- **Where it lands**: `Sample.tree: Option<TreeStats>` (botbowl-data; `serde(default,
  skip_serializing_if = "Option::is_none")` — old corpora parse, unsearched samples write no key;
  `prepare` and the Python loaders ignore it); `SearchTelemetry.tree: TreeTelemetry` (summed
  counters, `serde(default)`; `summary()` prints the means on the `MCTS_TELEMETRY` line) → `report.json`
  and `eval.games.jsonl`; corpus `meta.extra` gets `tree_searches`, `tree_descents`, `tree_plies`,
  `tree_reached_opp_turn`, `tree_end_horizon`, `tree_main_plies`, `tree_main_reach_opp`,
  `tree_depth_mean`, `tree_reach_opp_share`. **Hub protocol v17**: `SearchTelemetry` rides inside
  `EvalGameLine` under postcard, so the frame changed and workers must update.
- **`scripts/tree_stats.py CORPUS...`**: per-decision `tree` blocks stratified by the mover's
  decisions left in its turn (from the trajectory: later samples of the same team, half and turn
  counter), by root procedure and by board, `opp_turn_follows = false` filtered out and counted.
- Each sample's `tree` block also names the root procedure (`proc`), so the script needs no
  procedure-stack parsing.

### Observational: same games, same cost

`scripts/perf_search_bench.sh` (3 random-start games, 16x9/6, heuristic, `gumbel16_f1000_gen`,
1000 descents; the hash step now drops each sample's `tree` block):

| build | instructions:u | trajectories |
|---|---|---|
| base 95d1852 | 27,040,513,251 (rerun 27,041,075,053) | a6a7edd89b18 |
| plan 060 (0e2fa45 code; a pre-commit run read 27,037,725,717) | 27,039,663,326 | a6a7edd89b18 |

Identical games; −0.003%, inside the base's own run-to-run spread.

### First reading (the bench corpus: heuristic evaluator, Gumbel m=16, 1000 descents, 3 games)

Not the live net — a smoke test of the tables, not a finding. 114 decisions, 91 searched, 3
filtered (no opponent turn).

| left | n | reach_opp | end_opp | horizon | depth mean/p90/max | own | chance | ml_opp |
|---|---|---|---|---|---|---|---|---|
| 0 | 13 | 84.8% | 61.4% | 20.9% | 7.08 / 9.8 / 13.4 | 1.37 | 60.9% | 100% |
| 1 | 17 | 65.4% | 48.5% | 16.3% | 6.82 / 10.4 / 13.9 | 2.04 | 61.3% | 94% |
| 2 | 12 | 53.5% | 42.7% | 10.8% | 6.51 / 10.2 / 12.6 | 2.19 | 58.3% | 83% |
| 5-9 | 26 | 37.7% | 33.6% | 4.2% | 6.19 / 9.2 / 12.5 | 2.61 | 53.8% | 65% |

Over half of every line's depth is dice (chance share 53-61%), and a line holds only ~2 own
decisions. Next: run `tree_stats.py` on a live-loop shard (net evaluator) and at 250 / 1000 / 4000
descents for §3's depth-vs-budget row.

### First live reading: gen06 net, 250 / 1000 / 4000 descents (2026-10-08)

`runs/plan060/budget_ladder.sh`: the loop's generation preset (`gumbel16_f1000_gen`), the gen06 net
on tract, 12 random-start drives with `--next-drive` on the same seeds at each budget; decisions
without an opponent turn inside the horizon filtered out. `reach_opp` = share of descents whose
line reached the opponent's turn (ending there or beyond); `horizon` = past the end of it;
`ml_opp` = the main line reaches the opponent's turn.

| budget | n | reach_opp (all / turn-ending / 10+ left) | horizon (all / turn-ending) | depth mean / p90 / max (plies) | own decisions per line | chance share | main line plies | ml_opp (all / 10+ left) |
|---|---|---|---|---|---|---|---|---|
| 250 | 299 | 52% / 77% / 26% | 5% / 5% | 6.5 / 9.9 / 12.5 | 2.7 | 45% | 8.2 | 58% / 31% |
| 1000 (the loop) | 310 | 61% / 87% / 37% | 13% / 17% | 8.6 / 12.5 / 16.5 | 3.2 | 48% | 12.3 | 73% / 49% |
| 4000 | 243 | 74% / 92% / 48% | 20% / 26% | 10.8 / 15.2 / 20.5 | 3.5 | 49% | 16.2 | 86% / 80% |

- **Healthy on §3's terms.** Depth grows with budget (+2.1 plies per 4x), and so does the reach into
  the opponent's turn. At the turn-ending decision the loop's search sends 87% of its lines into the
  opponent's turn, and its main line gets there 92% of the time.
- **Where the depth goes:** about half of every line is dice, and a line holds only ~3 of the
  mover's own decisions at any budget; the extra depth from more budget is mostly the opponent's
  turn and its dice.
- **Past the opponent's turn** (`horizon`): 17% of lines at the turn-ending decision at 1000. Many
  opponent turns end early on a turnover (a failed dodge, GFI or block), which is what this counts;
  the reading does not separate a turnover from a played-out turn yet.
- **Early in a turn** (10+ own decisions left) only 37% of lines reach the opponent at 1000, and
  half of the main lines stop inside the own turn: those decisions are valued by the value head
  about the end-of-turn position, as expected.

### Open: a debug-only assertion in node teardown (2026-10-08)

A debug build of the hub's generate path (the status-page work's manual test) hit
`recon_mcts/src/tree.rs`'s debug assertion "could not remove dropped node as child's parents".
Release builds skip the check, so the loop is unaffected; reported as pre-existing (not caused by
this plan's hook), not yet reproduced or bisected. It may point at a parent-set bookkeeping slip in
node teardown (`Node::on_drop` → `detach`). To do: reproduce in a debug `dataset` run, bisect
against `observe_descent` (75a6801) and the state-shrink commit (7841ef2).

### First full-corpus reading: v9 gen09 (2026-10-08)

`python3 scripts/tree_stats.py runs/loopmix16x9v9/gen09/shard[0-7].jsonl` →
`runs/plan060/gen09_tree_stats.txt`. 199,544 decisions: 139,780 searched and kept, 39,576
unsearched (forced, setup placements), 19,888 filtered (no opponent turn ahead).

| left in turn | n | reach_opp | horizon | depth mean / p90 | own per line | chance | ml_opp |
|---|---|---|---|---|---|---|---|
| all | 139,780 | 59% | 9% | 9.8 / 15.7 | 3.4 | 52% | 74% |
| 0 (turn-ending) | 9,137 | 85% | 16% | 12.7 / 20.3 | 1.7 | 66% | 89% |
| 10+ | 36,132 | 40% | 3% | 8.8 / 13.5 | 4.4 | 44% | 55% |

The same picture as the 12-drive ladder at 1000 descents. Larger boards reach the opponent less
(16x9/6 53%, 12x6/3 67%).

**Anomaly: 14x5/3.** Every board sits at ~9 plies mean and ~45% chance, except 14x5/3: mean 63
plies (p90 154, max 207 as per-search means), 93% chance. 931 of its 2,834 searched decisions have
a per-search mean over 40 plies; the worst ~2,000 (max ~3,100), nearly all chance, ~1-3 own
decisions, almost every descent ending inside the own turn (e.g. seed 19000196 drive 1 sample 7,
root `Catch`). Suspects: an engine chance loop on a 5-high pitch (throw-ins / scatter out of
bounds again and again) or a cycle in the search DAG through recombination. Under investigation
(subagent, own worktree); findings go in the next section.
