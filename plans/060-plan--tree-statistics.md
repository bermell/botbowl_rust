# Plan 060 — tree statistics: how deep does the search see, and into whose turn?

**Status:** Started 2026-10-08 (the user asked for it; implementation by a subagent in a worktree).

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

(pending)
