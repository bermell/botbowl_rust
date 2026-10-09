# Plan 061 — the chance model: the rolls the search still scripts, and the cost of width

**Status:** Built 2026-10-09 on a branch, every toggle off by default (the shipped search is
byte-identical: the goldens in `tests/virtual_loss_inert.rs`, `tests/lazy_mover_identity.rs` and
the mirror suites are unchanged). Telemetry measured (§2). Experiments planned, not run (§7):
drives only. Hub protocol v18.

## 1. Why

Plan 049 found three search-model bugs: armour always a casualty, every pass a fumble, a horizon
running past half time. They were rolls the search scripted to one outcome. Since then blocks,
armour/injury, fouls, the pass D6 and every pass/fail roll are enumerated at real odds. Four
families still are not (`botbowl-mcts/src/roll_outcomes.rs`):

1. **Bounce** (`bounce_outcomes`). When any neighbour is empty or out of bounds, the directions
   onto players are **dropped** and the rest renormalised. Its doc comment says the ball "bounces
   off and keeps going". The engine's `Bounce` (`ball_procs.rs`) does something else. A ball
   coming down on a **standing** player goes to `Catch`, a catch attempt that can change
   possession. A ball on a **downed** player bounces on from that square. So every loose ball next
   to a player was modelled as never reaching it. That is a modelling bug, not an approximation.
2. **Throw-in** (`throw_in_outcome`): one scripted child, the shortest axis-aligned throw that
   lands in bounds off its origin (plan 060 §6's fix for the 14x5 endless chains).
3. **Inaccurate pass** (`Scatter`, 3 D8 from the target): scripted to up, up, up, so the ball
   always lands 3 squares "up" from the target. **Wildly inaccurate pass** (`Deviate`, D6 x D8
   from the passer): scripted to one square "up". The engine walks each step and stops at the
   edge (a throw-in from the last square on the pitch). `DeflectOrResolve` then gives **any**
   player on the landing square a `Catch` (see §8), and an empty square a bounce.
4. Kickoff deviate, the kickoff table's 2D6, the weather's 2D6/D8 and the coin: left alone.
   They lie past every turn-horizon search.
   - A setup-decision search does reach the kickoff: `MctsConfig::setup = search` searches its own
     placements through the opponent's setup to the kickoff, and the receiver's first turn for the
     kicker. There these rolls stay scripted: deviate D6 One + up, which is 0 squares on boards
     with `scatter_divisor() = 2`; table 2D6 = 2 (the ref); weather D8 up.
   - So a setup search values its formation against **one** kick landing, the aim square, and
     never sees a deviated kick. This is worth knowing when reading setup decisions. It is not
     worth fixing before the in-turn rolls.

The other half of the problem is what makes width expensive. **A chance node has no value until
every outcome is scored.**
- `backprop_scores` withholds the expectation while the scored mass is below 0.999 (the
  completeness gate, plan 018).
- `select_node` sweeps unscored outcomes first to close that window, then keeps visit ratios at
  `p` with the deficit rule `p_i (N+1) - N_i`.
- Every descent ends at the first new leaf. So a chance node with k outcomes costs k descents
  before its parent sees anything, and nested chance nodes multiply.

That is why the rolls above were collapsed, and why fixing them naively could cost more than it
gains.

## 2. Inventory and telemetry

Two new counters (commit 3bbcfc8).

**In search:** `MCTS_CHANCE_STATS`, printed next to `MCTS_LEAF_STATS` (`leaf_stats = true` in a
preset, or `BLOOD_MCTS_LEAF_STATS=1` / `BLOOD_MCTS_STATS=1`). For each `RollKind` it gives
`created/outcomes/visited`:
- created: chance nodes expanded;
- outcomes: their outcome children;
- visited: descents through them.

It also prints the backup tally `backup_withheld/complete/partial`. `RollKind` splits a request by
the procedure asking: a D8 is a bounce, a kick bounce or the weather's; a deviate is a pass's or the
kickoff's; a 2D6 is the kickoff table's or the weather's.

**In real games:** `botbowl-ui roll-census --corpus SHARD...`. It replays random-start trajectories
from their seed under their own dice, using the new `GameState::step_observing_rolls`, which is
`step` plus a look at each roll; same dice, same order. It checks every recorded state and counts
the rolls the engine resolved, per kind, per drive and per decision. For live bounces it also
reports where the ball could land, as probability mass. For throw-ins it reports whether each was
the first of its chain.

### Real games: v9 gen13's corpus (2026-10-09)

**Setup.** `roll-census` over `runs/loopmix16x9v9/gen13/shard{0,1}.jsonl` (the live loop's
mixed-size corpus). 1449 drives (`--next-drive` follow-on drives included) and 48,670 decisions,
all replayed without divergence.

| kind | rolls | per drive | shipped search model |
|---|---|---|---|
| pass/fail (pickup, dodge, GFI, catch) | 10,412 | 7.19 | exact |
| block | 4,118 | 2.84 | exact (resolved outcomes) |
| armour (2D6 pass/fail) | 2,830 | 1.95 | exact |
| **bounce (live ball)** | **1,877** | **1.30** | **player squares dropped** |
| injury | 1,258 | 0.87 | exact |
| kickoff deviate | 649 | 0.45 | scripted (past the horizon) |
| kickoff bounce | 617 | 0.43 | as bounce |
| **throw-in** | **259** | **0.18** | **scripted** |
| foul armour / injury | 20 / 8 | 0.01 | exact |
| **pass roll / scatter / deviate** | **1 / 1 / 0** | **0.001** | exact / scripted / scripted |

- **Live bounces land on players a fifth of the time.** Their landing mass is 71.4% empty,
  17.4% a standing player, 3.2% a downed player and 8.1% out. So 21% of every real bounce's
  outcome mass, 0.27 bounce-onto-a-player per drive, is exactly what the shipped model drops.
- **Throw-ins:** 212 first throws, 18 late in a chain, 29 re-throws.
- **The loop's bots do not pass.** There was one pass in 1449 drives.
- **No kickoff table or weather** at 6 players a side (`kickoff_table_enabled` needs 7).

### In search: gen13 on tract, 1000 descents, `gumbel16_f1000` (2026-10-09)

**Setup.**
- `botbowl-ui dataset --mode random-start --board-sizes 14x7/4,16x9/6 --games 6 --seed 61000
  --mcts-iters 1000 --parallel-games 1`, under `gumbel16_f1000` plus `leaf_stats = true`. Same
  seeds per arm, but the games diverge once the bots play differently, so these are indicative,
  not paired.
- Per searched decision. Each cell is chance nodes created / outcomes per node / descents through
  them.

| per searched decision | shipped (160 searches) | all roll models, `complete` (225) | all roll models, `partial` (217) |
|---|---|---|---|
| bounce | 35.9 / 5.8 / 369 | 53.4 / 7.2 / **1090** | 51.4 / 7.5 / 317 |
| throw-in | 7.4 / 1.0 / 61 | 7.6 / 3.0 / 188 | 13.2 / 3.5 / 55 |
| pass/fail | 138 / 2 / 1675 | 118 / 2 / 1854 | 208 / 2 / 1781 |
| block | 28 / 4.0 / 583 | 28 / 4.0 / 644 | 37 / 4.0 / 608 |
| pass scatter + deviate + pass D6 | 0.14 | 0.08 | 0.06 |
| chance backups withheld | 23% | **34%** | 0% (23% emitted partial) |
| net forwards | 598 | 597 | 546 |
| wall time (one thread, loaded box) | 2.66 s | 2.56 s | 2.65 s |

`scripts/tree_stats.py` on the same runs:

| arm | reach_opp | horizon | depth mean / p90 | own decisions per line | chance share | main line plies | main line reaches opp. turn | exact (TD/half) leaves |
|---|---|---|---|---|---|---|---|---|
| shipped | 53.7% | 9.1% | 9.2 / 13.9 | 3.85 | 45.5% | 14.3 | 72.7% | 0.37% |
| all models, `complete` | 57.2% | 10.0% | 9.7 / 15.1 | 3.29 | 55.4% | 9.7 | **43.4%** | 0.40% |
| all models, `partial` | 43.4% | 2.4% | 9.7 / 14.8 | **4.39** | **38.3%** | **17.3** | 74.3% | **1.85%** |

**Reading.**
- **Wall time stays flat.** It is net-forward bound on tract, at ~600 forwards per decision in
  every arm.
- **Width under the complete gate goes to dice.** With the roll models on, descents through
  bounce nodes triple (369 → 1090), a third of chance backups are withheld, the chance share of a
  line rises to 55%, and the main line's reach into the opponent's turn falls from 73% to 43%.
- **The partial backup reverses it.** The bounce passes drop back to 317, the chance share falls
  below the shipped search's, lines carry more of the mover's own decisions, and searches reach
  known outcomes (touchdowns, half ends) five times as often. They see the opponent's turn
  somewhat less (43% of descents vs 54%), a trade the drives must judge (§7).
- **Passes do not matter today:** under 0.15 pass chance nodes per decision. **Throw-ins are
  rare but cheap.**

## 3. The fixes (a)-(c): exact distributions, grouped by consequence

All three live in `roll_outcomes.rs` behind `RollModel` (`enumerate_full`). Each child carries
real dice that make the engine produce its landing, every enumeration is a pure function of the
state, and every child reaches its own state. Recon_mcts cannot hold two chance edges into one
child, so `tests/chance_roll_models.rs` applies every child and compares the results.

### (a) `bounce_model = "catch"` (`bounce_catch_outcomes`)

| direction lands on | child | probability |
|---|---|---|
| empty square | settles there | 1/8 |
| standing player | `Catch` (enumerated next as pass/fail) | 1/8 |
| downed player | the ball bounces on from that square (`Bounce` again) | 1/8 |
| a player square already in `bounce_squares` | dropped, rest renormalised | — |
| out of bounds | one collapsed throw-in child (`oob_representative`) | n/8 |
| kickoff bounce: out, or the kicking half | one collapsed touchback child | n/8 |

- **Downed players.** We follow the engine and do not drop them: a downed player's square sends
  the ball on.
- **Termination.** The existing boxed-in rule (never return to a square in `bounce_squares`) now
  applies to every player square. That bounds the catch-fail-bounce chain: every lap must reach a
  new square, and an empty or out-of-bounds direction always ends it.
- **Touchbacks.** All touchback directions reach the same state: `Touchback` does not move the
  ball. So they must be one child.
- **Fan-out.** At most 8 children.

### (b) `pass_scatter_model = "grouped"` (`pass_landing_grouped`)

- **Exact distribution.** We walk all 512 scatter combinations from `Pass::target()`, or all 48
  deviate combinations from the passer. We apply the engine's rule: a step out of bounds stops the
  walk, and the ball is thrown in from the last square on the pitch. The result is the exact
  landing distribution.
- **Grouping** (`grouped_landing_children`):
  - one child per **occupied** landing square, at its exact probability. `DeflectOrResolve` gives
    any player there a `Catch`. We keep at most the 6 likeliest and renormalise the rest away;
    this is rare.
  - **one** "goes out" child carrying all the out-of-bounds mass. It is represented by its
    likeliest throw-in origin.
  - the **empty** mass, carried by the 3 likeliest empty squares in proportion.
- **Fan-out** is at most 10. The probabilities of catches and of going out are exact. Where an
  empty landing goes is approximated by its 3 likeliest squares.
- **Mirror safety.** Ties between squares, and the choice of representative dice among the
  combinations of a group, use the acting team's frame: forward direction, then dy, and distance
  from the attacked endzone. Mirroring swaps the teams and reflects x, so these keys map onto
  themselves (plan 023 H-c, `TieBreak::Mover`).
  `pass_landing_children_mirror_with_the_board` checks a scatter and a deviate against their
  mirrors.
- **Kickoff deviates.** A deviate under the kickoff, not under `Pass`, keeps the script.

### (c) `throw_in_model = "grouped"` (`throw_in_grouped`)

- **Enumeration.** The same grouping over 3 directions x 2D6, via the proc's own `target_square`,
  so `scatter_divisor` and `max_scatter` apply exactly as in the engine.
- **Classes:**
  - a standing player on the landing square is a catch;
  - an empty or downed player's square is a bounce;
  - out of bounds is a re-throw from where the ball went out (one child).
- **The 14x5 lesson: chains must end.** Only the *first* throw of a chain is enumerated. A
  **re-throw** keeps the scripted throw, and so does any throw-in whose ball already went through
  2+ squares this sequence (`bounce_squares.len() > 1`, i.e. after an earlier throw-in landed).
  - A re-throw is recognised because the ball's square is no longer the proc's origin: the engine
    moves the origin and leaves the ball.
  - The scripted throw lands in bounds off its origin, which plan 060 §6 showed ends a chain.
  - So at most two enumerated laps precede scripted ones.
- **Tests.** `every_throw_in_chain_ends_on_narrow_boards` walks the *whole* chance tree below edge
  throw-ins on 8x3, 12x5, 14x5 and 14x7, with every new model on, and bounds its size and depth.
  `a_rethrow_falls_back_to_the_scripted_throw` pins the fallback.

## 4. Chance backup: alternatives to "no value until complete"

`chance_backup` (`ChanceBackup`, `ChanceSearch`; `select_node`'s chance branch and
`backprop_scores`' chance branch). All modes keep the existing pieces:
- **Integers and ordering.** The `canonical_chance_key`-sorted summation (mirror-exact), and the
  `avg as i64` truncation.
- **Virtual loss.** No virtual loss on chance edges.
- **The mean backup at player nodes**, unchanged.
- **Proven only when complete.** A chance node is proven only when every outcome is scored and
  proven to the same result. A partial value is never proven, so the proven-win rule at player
  nodes cannot fire on a partly explored gamble.
- **Solvedness**, unchanged: recon_mcts needs every child solved. Under `widen`, a never-opened
  outcome keeps its parent unsolved, which is conservative.

| mode | selection | value available | literature |
|---|---|---|---|
| `complete` (shipped) | unscored first (likeliest first), then deficit `p_i(N+1) - N_i` | all outcomes scored (mass ≥ 0.999) | expectimax; plan 018 |
| `partial` | same | first scored outcome; `Σ_s p_i Q_i / Σ_s p_i` | sparse sampling (Kearns, Mansour, Ng 2002); *-minimax / MCMS (Lanctot et al. 2013) |
| `mass` | same | scored mass ≥ `chance_mass` (0.9), renormalised | Star1-style bounds (Ballard 1983): the unseen mass bounds the error |
| `sampled` | deficit from the first visit, no sweep | first scored outcome (as partial) | Stochastic MuZero chance nodes (Antonoglou et al. 2022) |
| `widen` | only the `k = ceil(c (N+1)^α)` likeliest outcomes open (c 1, α 0.5); unscored first inside, deficit over renormalised `p` | first scored outcome (as partial) | double progressive widening (Couëtoux et al. 2011; Auger, Couëtoux, Teytaud 2013) |

### Why each is defensible

- **`partial`, sparse sampling.** Kearns, Mansour & Ng's sparse sampling values a chance node from
  C sampled outcomes. Its error bound depends on C and the horizon, *not* on the number of
  outcomes. Monte Carlo *-Minimax Search applies the same idea to expectimax games: sampled chance
  outcomes, with bounds from the unsampled mass.
  - Our partial value is the probability-weighted mean over the outcomes seen so far. Selection
    still sweeps the likeliest unscored outcome first, so the first value is the likeliest
    outcome's, and it converges to the exact expectation as the sweep finishes.
  - **The risk is plan 018's over-valuation.** A 1-GFI touchdown whose fail branch is unseen reads
    as a sure 1000. The sweep closes that window one descent later, but in the meantime PUCT at the
    parent can prefer the optimistic child.
  - That is self-correcting: the preferred child gets the next descent, which goes to its unscored
    outcome. It is also optimism in the face of uncertainty, the usual MCTS stance.
  - At the root it can bias the final pick at small budgets, so §7 measures it, it is not argued.
- **`mass`, bounded error.** With values in [−1000, +1000] and scored mass m, the true expectation
  lies within `(1 − m) · 2000` of `m·Q_s + (1 − m)·[−1000, 1000]`. That is Ballard's Star1 bound.
  - At m = 0.9 the renormalised value is off by at most 200 points, and in practice far less.
  - It still drops the long tail: a grouped scatter's rare catches, a bounce's last 1/8.
  - It is the conservative middle ground between `complete` and `partial`.
- **`sampled`, Stochastic MuZero.** Stochastic MuZero's chance nodes select the outcome code by
  `argmax σ(c) / (N(c) + 1)`, a deterministic quasi-random sweep that matches the prior. They back
  up the plain mean with no completeness requirement.
  - Our deficit rule is the same family. Dropping the unscored-first tier means a 5/6 roll's
    likely branch is revisited until the unlikely one's share comes due (6 visits).
  - So the value tracks the visit-weighted expectation, and descents are not spent on
    low-probability outcomes early.
  - With an exact enumeration and exact probabilities, we use the probability-weighted mean over
    visited outcomes rather than the visit-weighted one. It has lower variance and the same limit,
    and the visit counts here double-count recombined subtrees.
- **`widen`, progressive widening.** Couëtoux et al.'s double progressive widening adds a new
  outcome at a chance node only while `#children < C·N^α`, and is consistent (Auger, Couëtoux,
  Teytaud 2013).
  - Our outcomes are enumerated with exact probabilities, so "add a new outcome" becomes "open the
    next likeliest" (the PW of Coulom 2007 / Chaslot et al. 2008, applied to chance nodes).
  - recon_mcts's `available_actions` is a fixed per-state set, and `select_node` sees every child.
    So widening lives in `select_node`: it filters to the top-k by probability, a pure function of
    the offered set.
  - With c = 1 and α = 0.5, 1, 2, 3 and 4 outcomes are open at N = 0, 3, 8 and 15.
  - It fits wide, skewed distributions such as a grouped scatter or a throw-in with one dominant
    landing. It is the natural partner of (b) and (c).

### Interaction with the rest of the search

- **Root pick and Gumbel.** Both read child Q. Under the partial modes a root child whose action
  leads into a roll gets a value after one descent instead of after its whole chance subtree. That
  matters most for passes, pickups and dodges with several nested rolls.
- **Visit counts.** A chance node's visits are its scored children's sum under every mode, so
  PUCT's `N` is unchanged in meaning.
- **Proven wins.** A proven touchdown for the side to move still short-circuits the player-node
  mean. But a partly scored chance node above it reads as unproven until complete, so a "certain"
  score behind an unresolved roll is averaged like any estimate. That is correct: it is not
  certain.

## 5. Toggles and preset keys

| key (`MctsConfig`) | values (default first) | env | what |
|---|---|---|---|
| `bounce_model` | `settle` \| `catch` | `BLOOD_MCTS_BOUNCE=catch` | §3 (a) |
| `pass_scatter_model` | `scripted` \| `grouped` | `BLOOD_MCTS_PASS_SCATTER=grouped` | §3 (b) |
| `throw_in_model` | `scripted` \| `grouped` | `BLOOD_MCTS_THROW_IN=grouped` | §3 (c) |
| `chance_backup` | `complete` \| `partial` \| `mass` \| `sampled` \| `widen` | `BLOOD_MCTS_CHANCE_BACKUP` | §4 |
| `chance_mass` | 0.9 | `BLOOD_MCTS_CHANCE_MASS` | `mass`'s threshold |
| `chance_widen_c`, `chance_widen_alpha` | 1.0, 0.5 | `BLOOD_MCTS_CHANCE_WIDEN_C`, `_ALPHA` | `widen`'s k |

Presets, each a `gumbel16_f1000` with one change, are in `cfgs/README.md`:
`chance_bounce`, `chance_pass`, `chance_throw_in`, `chance_partial`, `chance_mass90`,
`chance_sampled`, `chance_widen` and `chance_all_partial`.

`botbowl-play/tests/bot_config.rs` pins three things:
- every committed preset loads;
- each chance arm differs from `gumbel16_f1000` only in chance knobs;
- each survives the hub's postcard round trip.

**Hub protocol v18** (commit 1364275). The seven fields ride inside `SearchConfig.config`, and
postcard encodes positionally, so a v17 worker would misread every preset-carrying task. Every
worker must be rebuilt, the laptop included, before a job at this commit.

## 6. Cost expectations

### What it costs before any value arrives

Under `complete`, a chance node with k outcomes takes k descents through it before its parent
gets a value, and a chance child that is itself a chance node must complete first. Some examples,
counting descents before the first value:
- **A pass/fail roll:** 2.
- **A shipped bounce:** ~6.
- **A catch-model bounce:** up to 8, plus 2 per catch child (and up to 8 more behind each failed
  catch that bounces again). About 15-30.
- **A grouped scatter:** ≤ 10 children, each a catch (2, plus a bounce behind its fail) or a bounce
  (8). About 50-100 before the pass action shows any value.
- **A grouped throw-in:** ≤ 10 children, each a catch or a bounce, the "out" one a scripted
  re-throw. About 30-60.

At 1000 descents per decision this is a real share of the budget whenever such a line is
considered, and §2 measures it: 34% of chance backups withheld, and the main line's opponent-turn
reach falling from 73% to 43% with the roll models on under `complete`.

Under `partial`, `sampled` and `widen` the first value arrives after one descent, whatever the
width. Under `mass` 0.9 it arrives after enough of the likeliest outcomes:
- 2 for a 5/6 roll;
- 7 of 8 for a bounce;
- typically 4-6 for a grouped landing, whose tail is light.

### CPU

- **Enumeration.** A grouped scatter walks 512 dice combinations and a deviate 48, both pure
  geometry with no engine stepping, once per chance-node expansion. That is a few tens of µs,
  negligible next to a net forward.
- **Wall time.** It was unchanged within noise in §2: ~600 forwards per decision decide it.
- **Instructions.** `scripts/perf_search_bench.sh` gives the exact count per arm (§7). The
  heuristic evaluator makes enumeration a larger share there, so it is the upper bound.
- **Expectations:**
  - `bounce`: +0-5% instructions.
  - `throw_in` and `pass`: +0-2%. These rolls are rare.
  - the backup modes: ±5%. More early values mean more propagation up the DAG, but
    fewer withheld backups and fewer descents wasted completing chance nodes.

**Memory.** Wider chance nodes add placeholder children: +5-10% nodes per tree with every roll
model on. That is irrelevant at 1000 descents.

## 7. Experiments (drives only)

**Setup, common to every arm:**
- **Bots.** The same net on both sides, `models/az_v7/bbnet_mix16x9v9_gen13.onnx` (the newest
  `.onnx`; use gen14's once exported). 1000 descents.
- **Control.** `cfgs/gumbel16_f1000.toml`. Candidate: the arm's preset.
- **Positions.** Paired contested drives (plan 051) on both boards:
  `runs/loopmix16x9g/positions/contested_{14x7,16x9}_gen04g.json`.
- **Stopping.** SPRT on pentanomial pairs, capped at 800 drives per board.
- **No full games, no P1, no anchor matches.**

**Build.** Everything below assumes the branch is merged and built at its commit, at the loop's
capacity:

```sh
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9
cargo build --release -p botbowl-ui -p botbowl-hub -p botbowl-worker
M=models/az_v7; NET=$M/bbnet_mix16x9v9_gen13.onnx
POS=runs/loopmix16x9g/positions/contested_14x7_gen04g.json,runs/loopmix16x9g/positions/contested_16x9_gen04g.json
```

**One arm, through the hub.** Run it while the loop's hub serves; the loop's eval shares workers
the same way:

```sh
target/16x9/release/botbowl-hub job eval --hub http://127.0.0.1:13337 \
    --token-file ~/.config/botbowl/hub.token \
    --label "exp070 ARM vs gumbel16_f1000" --evaluator nn --model $NET \
    --vs-evaluator nn --vs-model $NET --mcts-iters 1000 --seed 0 --skip-fixed-rungs \
    --bot-config cfgs/ARM.toml --vs-config cfgs/gumbel16_f1000.toml \
    --positions $POS --sprt S0:S1 --vs-games 800 \
    --per-game-out runs/exp070/ARM/eval.games.jsonl --out runs/exp070/ARM/report.json --wait
```

**Or locally,** one process, with a GPU sidecar (`scripts/nn_server.py`, as `exp060c` does):

```sh
target/16x9/release/botbowl-ui eval --evaluator nn --model $NET --nn-server $SOCK \
    --vs-evaluator nn --vs-model $NET --mcts-iters 1000 --seed 0 --skip-fixed-rungs \
    --bot-config cfgs/ARM.toml --vs-config cfgs/gumbel16_f1000.toml \
    --positions $POS --sprt S0:S1 --vs-games 800 --parallel-games 8 \
    --per-game-out runs/exp070/ARM/eval.games.jsonl --out runs/exp070/ARM/report.json
```

`scripts/exp070_chance_model.sh` runs the whole ladder below in order. It starts its own hub and
sidecar like `exp060c` and refuses a dirty tree.

**CPU cost of each arm.** Pinned to one core, with the trajectory hash showing whether the games
changed:

```sh
CONFIG=cfgs/ARM.toml scripts/perf_search_bench.sh runs/exp070/perf ARM
CONFIG=cfgs/gumbel16_f1000_gen.toml scripts/perf_search_bench.sh runs/exp070/perf control
```

**Depth and opponent-turn reach.** Run `scripts/tree_stats.py` on a 12-game `dataset` run per arm
(plan 060's `budget_ladder` recipe at 1000), to see whether a backup mode buys depth or
opponent-turn reach:

```sh
target/16x9/release/botbowl-ui dataset --mode random-start --next-drive --games 12 --seed 60000 \
    --bot-config cfgs/ARM.toml --evaluator nn --model $NET --mcts-iters 1000 --parallel-games 3 \
    --board-sizes 14x7/4,16x9/6 --out runs/exp070/tree/ARM.jsonl --truncate
scripts/tree_stats.py runs/exp070/tree/ARM.jsonl
```

### Arms, in order

| # | arm (candidate preset) | question | SPRT | adopt into the loop if |
|---|---|---|---|---|
| 0 | telemetry: control with `leaf_stats = true`, 12 drives/board | how often each roll kind is met in search (§2) | — | — |
| A1 | `chance_bounce` | does seeing catches off bounces help? | `0.47:0.5` (non-inferiority) | H1 on both boards **and** CPU per decision ≤ +10% |
| A2 | `chance_pass` | exact pass landings | `0.47:0.5` | as A1 |
| A3 | `chance_throw_in` | exact throw-ins | `0.47:0.5` | as A1 |
| B1 | `chance_partial` | early chance values | `0.5:0.55` | H1 on both boards; then a net check (`scripts/net_check.sh` with the arm as CONFIG) shows the budget curve still monotone |
| B2 | `chance_mass90` | bounded early values | `0.5:0.55` | as B1 |
| B3 | `chance_sampled` | no sweep | `0.5:0.55` | as B1 |
| B4 | `chance_widen` | progressive widening | `0.5:0.55` | as B1 |
| C1 | `chance_all_partial`, re-pointed at the B winner (or `partial` if none) | do the roll models pay once width is cheap? | `0.5:0.55` vs the B winner alone, then `0.47:0.5` vs control | H1 vs control; adopt the combination |

**Why A uses non-inferiority.** A1-A3 fix the model, so the burden is "not worse, not much
slower". A mis-modelled bounce is a bug even if the drives cannot see it at 800. B1-B4 change the
search's behaviour, so they must win.

**Adopting.** A win is a preset change in `scripts/launch_plan058.sh`'s generator and eval config,
`EVAL_BOT_CONFIG` and the gen preset. The default stays off until the loop runs it. It must be
added to `cfgs/gumbel16_f1000_gen.toml`'s twin, because generation uses `_gen`. Generation and
eval must move together, or the eval's search no longer matches the data's.

**Order and stop rules.**
- **0 before everything.** If a roll kind barely appears in search, drop its arm. Done, §2:
  - **A2 (passes) is deferred.** Real games make 0.001 pass rolls per drive and the search under
    0.15 pass chance nodes per decision, so 800 drives could not see it. Rerun it only once a
    generation passes.
  - **A1 (bounce) first.** 1.3 per drive, 21% of the mass misrouted.
  - **A3 (throw-in)** is rare (0.18 per drive) and cheap.
- **A before B.** A1-A3 are cheap to decide.
- **B on the shipped roll models**, so a backup win is not confounded with the model change.
- **C last.**
- **If every B arm is H0,** test C1 under `complete` too (`chance_all_partial` with `chance_backup`
  removed). The roll models may simply be too expensive.
- **14x5.** Run 14x7 and 16x9 drives only (the loop's eval boards). `throw_in` is the arm that
  could regress the narrow boards. `every_throw_in_chain_ends_on_narrow_boards` guards that in
  CI, and a 14x5 `dataset` run with `tree_stats.py` (mean depth < 15 plies) should precede
  adopting A3 or C1.

## 8. Found along the way

Two engine bugs, both in `DeflectOrResolve` (`ball_procs.rs`). Neither is fixed here: a fix
changes the games the loop plays, so it needs its own TDD commit and a decision. The grouped pass
model follows the engine as it is.

1. **A downed player "catches" a pass.**
   - `DeflectOrResolve` hands *any* player on the landing square a `Catch`, with the
     accurate/inaccurate modifier. `Catch` does not check `can_catch()`.
   - So a prone or stunned player on the landing square makes a catch roll, and on a success
     carries the ball while down.
   - `Bounce` and `ThrowIn` both check `can_catch()` and bounce the ball on instead, which is the
     rule.
2. **A failed deflection bounces the ball from the passer's square.**
   - Only the no-interceptor branch puts the ball in the air at the landing square
     (`set_ball(InAir(self.to))`). The interceptor branch hands the failure proc to `Deflect`, and
     `Deflect::apply_failure` returns it without moving the ball. So the ball is still `Carried` by
     the passer.
   - When the landing square is empty, the following `Bounce` reads `get_ball_position()`, which
     is the passer's square.
   - Reproduced: an inaccurate pass from (5,5) scattered to (14,5), with an interceptor at (8,5)
     failing his roll. The ball ends `OnGround((6,5))`, next to the passer, and the turn goes on.
   - The deflecting team was also offered a team re-roll for its failed deflection during the
     opponent's turn. That looks wrong too.
   - A `Catch` failure proc is unaffected (its `on_start` moves the ball). So is a `ThrowIn` (it
     carries its own origin).

Also:

3. **Passes are absent in practice.** The census found **0 pass rolls in 400 real drives**. In
   search, 1 in ~1000 chance nodes was a pass scatter or deviate (§2). Arm A2 is therefore the
   least likely to matter: the policy has learned not to pass, or pruning and priors hide passes.
   That is worth its own look before spending drives on A2.
4. **Kickoff bounces under the shipped `settle` model** give each in-bounds touchback direction
   (onto the kicking half) its own child, although all of them reach the same state.
   - That is the duplicate-edge hazard `merge_coinciding_outcomes` exists for.
   - It is reachable only when a kickoff bounce starts next to the kicking half. The search's
     scripted kickoff lands on the aim square in the middle of the receiving half, so no searched
     line was seen to reach it. It is not reproduced; the `catch` model collapses those
     directions.
5. **Setup searches see one kick.** They reach the kickoff with the deviate and the table scripted
   (§1, item 4).

## 9. References

- B. W. Ballard, "The *-minimax search procedure for trees containing chance nodes", *Artificial
  Intelligence* 21 (1983).
- M. Kearns, Y. Mansour, A. Y. Ng, "A sparse sampling algorithm for near-optimal planning in large
  Markov decision processes", *Machine Learning* 49 (2002).
- R. Coulom, "Computing Elo ratings of move patterns in the game of Go", *ICGA Journal* (2007);
  G. Chaslot et al., "Progressive strategies for Monte-Carlo tree search", *New Mathematics and
  Natural Computation* (2008) — progressive widening / unpruning.
- A. Couëtoux, J.-B. Hoock, N. Sokolovska, O. Teytaud, N. Bonnard, "Continuous upper confidence
  trees", LION 5 (2011) — double progressive widening.
- D. Auger, A. Couëtoux, O. Teytaud, "Continuous upper confidence trees with polynomial
  exploration — consistency", ECML PKDD (2013).
- M. Lanctot, A. Saffidine, J. Veness, C. Archibald, M. Winands, "Monte Carlo *-Minimax Search",
  IJCAI (2013).
- I. Antonoglou, J. Schrittwieser, S. Ozair, T. Hubert, D. Silver, "Planning in stochastic
  environments with a learned model" (Stochastic MuZero), ICLR (2022).
