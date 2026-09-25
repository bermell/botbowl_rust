# Is 1000 iterations still the search budget on the larger boards?

**Status:** Planned 2026-09-24, not run. Written as an overnight runbook for a delegated agent;
the agent appends results under `## Results` and applies the pre-committed rules in `## Decision
rules` — nothing else in this file changes.

## Why now

`MCTS_ITERS=1000` was set in plan 020 and has only ever been measured on **14x7/4**:

- Plan 027 E2: 2000 vs 1000 = **0.508** (dead even); 1000 vs 500 = 0.575; 1000 vs 250 = **0.700**
  (p≈0.002). Verdict "keep 1000, don't raise it".
- Plan 028 Stage 0: the convergence curve is a **flat spot at 1000–2000, not a ceiling** — top-1
  agreement with a 16k reference goes 0.69 (1000) → 0.67 (2000) → 0.73 (4000) → 0.91 (16000).
  Prediction: 8000/16000 beat 1000 on strength; never tested (plan 032 #10, still open).

Since then, everything the search runs on has changed:

- **Boards.** The loop builds at 16x9/6 capacity and plays a centred mix of 12 boards from
  70 to 144 cells (plan 042), evaluated on 14x7/4, 16x9/6 and 12x9/4. One `MCTS_ITERS` serves
  every board for both generate and eval. A 16x9/6 root has ~1.5× the squares and 1.5× the
  players of 14x7/4, so at a fixed budget the search sees proportionally less of its fan.
  Plan 042 already lists "**search budget per size**" as an open question.
- **Search correctness.** Plan 044 took colliding states from 36.8% to 0% and wasted registry
  comparisons from 4.07 to 0.12 per probe, so recombination now actually recombines; each
  iteration builds a different (denser) DAG than the one plan 027/028 measured.
- **Engine.** Bounce/throw-in cycle fixes, ejection-turnover fix, formation bands for every
  non-full board (2026-09-18). Nothing before that commit is comparable at any size.

So both things could have moved: the knee at 14x7 itself, and how the knee scales with board.

## Hypotheses (pre-committed)

- **H1 (14x7 re-check).** 1000 still sits in the flat spot on 14x7/4: 2000 vs 1000 within
  0.50 ± 0.05.
- **H2 (size dependence).** 16x9/6 is more breadth-starved: 4000 vs 1000 gains **more** on 16x9
  than on 14x7 (difference in points ≥ 0.08).
- **H3 (convergence).** X\* (smallest budget whose signal ≤ the run-to-run floor) is larger on
  16x9 than on 14x7, roughly in proportion to the mean root fan.

## Design

Three parts, cheap → expensive. Every arm is *the same net at budget X* vs *the same net at
1000*, so the only variable is the budget. Points = (W + D/2)/N, paired Home/Away on a shared
seed base (plan 032 ground rules). **Budget is the candidate's `--mcts-iters`; the opponent is
always `--opponent-iters 1000`.** The `vs:` rung label does *not* show an iteration asymmetry
and `report.json` stamps only the candidate's budget — **the budget must be in the file name.**

### Fixed choices

| choice | value | why |
|---|---|---|
| net | the loop's current champion `models/az_v7/bbnet_mix16x9_gen<NN>.onnx` (latest with a `BENCHMARKED` verdict). Fallback if not on this box: `models/bb_best_14x7_gen23_v7.onnx` (E0's net; 0.938 vs scripted at 16x9). Record which. | the question is about the loop as it runs today |
| capacity build | `BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6`, `CARGO_TARGET_DIR=target/16x9` | the loop's build; one binary plays every board ≤ capacity |
| boards | `14x7/4` (the curriculum centre) and `16x9/6` (the capacity). `12x9/4` only if time is left. | the two ends of what the loop trains on |
| search config | loop defaults: `--backup minimax --puct-mode raw --horizon-turns 1 --fpu-reduction 0 --mcts-workers 1` | measure the budget the loop actually uses; workers must stay 1 because an `Iterations` budget is split across workers |
| seeds | `--seed 45000` for every arm | common situations across arms; far from the loop's eval seed 0 and corpus seeds ≥ 10 000 000 |
| games | 160 per (arm, board) for the primary arm, 120 for the rest, always even | 157 games detect 0.60 at 80% power (plan 032 power table); SE ≈ 0.04 |

### Part A — convergence curve per board (diagnostic, ~1–2 h CPU, no games)

The plan-025 method, shipped as `botbowl-ui convergence`: re-search the same mid-turn states at
a ladder of budgets, signal = TV to the 16k reference, floor = TV between independent 16k runs,
X\* = smallest budget with signal ≤ floor; also top-1 agreement per budget and the root fan
(`n_legal_actions`) per state. Run once per board by overriding `BOARD_SIZE_*` **at run time on
the already-built binary** (never `cargo run` — `build.rs` re-triggers on those env vars and
would rebuild at 14x7 capacity).

Deliverable: a two-row table (board × budget) of signal/floor ratio and top1, X\* per board, and
the mean fan per board. The fan ratio 16x9/14x7 is the "iters ∝ legal-move count" scale factor
plan 042 asked for.

### Part B — strength A/B per board (the experiment)

Arms in priority order; run them in this order so a short night still answers the primary
question:

| # | arm (candidate vs opponent) | boards | games/board | tests |
|---|---|---|---|---|
| B1 | **4000 vs 1000** | 14x7/4, 16x9/6 | 160 | H2 — the 4× span that gave 0.700 at 250→1000 |
| B2 | **2000 vs 1000** | 14x7/4, 16x9/6 | 120 | H1 on 14x7; whether a cheap 2× is worth anything on 16x9 |
| B3 | **500 vs 1000** | 16x9/6 | 120 | is the knee still above 500 on the big board (0.575 on 14x7 in plan 027) |
| B4 | **16000 vs 1000** | 14x7/4 | 60 | plan 032 #10, stretch only; ~17× the cost per game |
| B5 | B1 on 12x9/4 | 12x9/4 | 120 | held-out probe, only if everything above finished |

Cost per game relative to a 1000-vs-1000 game ≈ (X + 1000)/2000: B1 2.5×, B2 1.5×, B3 0.75×,
B4 8.5×. **Memory scales with iterations too:** the worker's governor models ~4.5 MB per board
cell per 1000 iterations, so on 16x9 (144 cells) one 4000-iteration tree is ~2.6 GB and a B1
game holds ~3.3 GB across its two trees; `botbowl-ui eval` has **no** memory governor (that is
in `botbowl-worker`), so `--parallel-games` must be sized by hand:

    P = floor(0.6 × RAM_GB / GB_per_game)   with GB_per_game ≈ 0.65 × (X + 1000)/1000 on 16x9

(B1 on a 32 GB box → P = 5; B4 → P = 1–2.) Verify with `free -g` during the pilot.

### Part C — cost of the alternative (only if B says raise or scale)

The loop's throughput at the winning budget: `scripts/measure_search_speedup.sh` with `ITERS=X`
(and `nn_throughput_probe.sh` if the sidecar is the bottleneck), ms/decision and s/game on the
same box, so the generation-cost side of the trade is a number and not "4×". Skip if B says
keep 1000.

## Runbook (for the overnight agent)

Everything below runs from the repo root on **one idle box**. Budget: 10 h wall.

**Or run Part B on the fleet.** Every arm is a plain ladder job, so `botbowl-hub job eval` takes
the same flags as `botbowl-ui eval` and both seats are fully described by the submission — as of
2026-09-25 the hub pins a complete `MctsConfig` into each task, so no helper box's `BLOOD_MCTS_*`
or `BOARD_SIZE_*` can reach an arm. Swap `$UI eval` for:

```sh
$HUB job eval --hub "$HUB_URL" --token-file "$HUB_TOKEN_FILE" \
    --evaluator nn --model $NET --mcts-iters $X \
    --vs-evaluator nn --vs-model $NET --opponent-iters 1000 \
    --skip-fixed-rungs --board-sizes $BOARDS --games $N --vs-games $N \
    --seed 45000 --mcts-workers 1 \
    --per-game-out $OUT/$tag.games.jsonl --out $OUT/$tag.report.json --wait
```

Three things change with the hub and nothing else does: `--parallel-games` is ignored (each worker
sizes itself, and its memory governor — which now scales its prediction by the iteration budget —
does the job of the `P` formula below per box, so the pilot is only needed for the time estimate);
`report.json` carries no `lectures`, which these arms skip anyway; and the boxes must all be on
the hub's commit, or on one named in its `hub-allowed-commits.toml`. Everything about the arms,
seeds, labels and output files is identical, so the `## Results` tables are filled the same way.
Do **not** mix: run a given arm entirely on the fleet or entirely locally.

### 0. Preflight (≈15 min)

```sh
cd <repo>
git status --short            # must be clean; note `git rev-parse --short HEAD`
pgrep -fl train_loop.sh       # must print nothing: the loop's generate/eval would share the CPU
                              # and the sidecar, and every timing below would be garbage.
                              # If it is running, stop here and report `needs input:`.
nproc; free -g                # cores → parallel_games ceiling; RAM → P per the formula above

export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6
export CARGO_TARGET_DIR=$PWD/target/16x9
cargo build --release -p botbowl-ui
UI=$CARGO_TARGET_DIR/release/botbowl-ui

# net: newest benchmarked mix champion, else E0's net
NET=$(ls -t models/az_v7/bbnet_mix16x9_gen*.onnx 2>/dev/null | head -1)
[ -n "$NET" ] || NET=models/bb_best_14x7_gen23_v7.onnx
OUT=runs/exp045; mkdir -p $OUT
printf 'commit %s\nnet %s\nbox %s\ncores %s\n' "$(git rev-parse --short HEAD)" "$NET" "$(hostname)" "$(nproc)" > $OUT/status.md
```

Sidecar (optional, faster): if a `.pt` sits beside `$NET` and the box has CUDA, start
`scripts/nn_server.py --socket /tmp/bbnn-exp045.sock --device cuda --model $NET` and pass
`--nn-server /tmp/bbnn-exp045.sock` to every `eval`/`convergence` below. Otherwise tract on CPU
is the default and is fine — it only changes how many games fit in the night. Record which in
`status.md`. Leave `BLOOD_MCTS_STATS` unset (it walks the whole DAG after every search and skews
timing).

### 1. Part A (≈1–2 h)

```sh
for B in 14x7/4 16x9/6; do
  W=${B%%x*}; rest=${B#*x}; H=${rest%%/*}; T=${B##*/}
  BOARD_SIZE_W=$W BOARD_SIZE_H=$H BOARD_PLAYERS=$T \
  $UI convergence --evaluator nn --model $NET --states 40 --repeats 3 --advance 1 \
      --budgets 250,500,1000,2000,4000,8000,16000 --seed 90000000 \
      --out $OUT/a_conv_${W}x${H}_${T}.jsonl 2> $OUT/a_conv_${W}x${H}_${T}.log
  scripts/convergence_summary.py $OUT/a_conv_${W}x${H}_${T}.jsonl | tee -a $OUT/status.md
done
```

Smoke first with `--states 1 --repeats 2 --budgets 100,200` per board and check the log's fan
numbers differ between boards (the 16x9 activation/move fans must be larger) — that is the proof
the runtime override took. Also record the mean of `n_legal_actions` per board from the JSONL.

If Part A alone exceeds 2 h (tract on a slow box), cut `--states` to 25; do not cut repeats.

### 2. Pilot (≈15 min) — size the night

```sh
t0=$(date +%s)
$UI eval --evaluator nn --model $NET --mcts-iters 4000 \
    --vs-evaluator nn --vs-model $NET --opponent-iters 1000 \
    --skip-lectures --skip-fixed-rungs --board-sizes 16x9/6 --vs-games 16 --games 16 \
    --seed 45000 --parallel-games $P --mcts-workers 1 \
    --per-game-out $OUT/pilot.games.jsonl --out $OUT/pilot.report.json 2> $OUT/pilot.log
echo "pilot 16 games 4000v1000@16x9: $(( $(date +%s) - t0 )) s" | tee -a $OUT/status.md
```

Let `s16 = wall seconds / 16` (s per B1 game on 16x9 at parallelism P). Estimate each arm's
wall time as `games × boards × s16 × cost_ratio(arm) / 2.5`, treating 14x7 as ~0.7× a 16x9
game. Fit the arms to the hours left in priority order; shrink an arm's games (even numbers,
floor 80 for B1/B2, 60 for B3) before dropping it; drop from the bottom. Write the plan of
record — arms, games, expected minutes — into `status.md` **before** launching.

### 3. Part B arms

One invocation per arm; `X` is the candidate budget, `BOARDS` the comma list, `N` the games:

```sh
run_arm() {  # X BOARDS N
  local X=$1 BOARDS=$2 N=$3 tag="b_${1}v1000_$(echo $2 | tr ',/' '_-')"
  local t0=$(date +%s)
  $UI eval --evaluator nn --model $NET --mcts-iters $X \
      --vs-evaluator nn --vs-model $NET --opponent-iters 1000 \
      --skip-lectures --skip-fixed-rungs --board-sizes $BOARDS --games $N --vs-games $N \
      --seed 45000 --parallel-games $P --mcts-workers 1 \
      --per-game-out $OUT/$tag.games.jsonl --out $OUT/$tag.report.json 2> $OUT/$tag.log
  echo "$tag: $(( $(date +%s) - t0 )) s wall, $N games/board" >> $OUT/status.md
  scripts/paired_summary.py $OUT/$tag.games.jsonl | tee -a $OUT/status.md
  scripts/eval_summary.py $OUT/$tag.report.json | tee -a $OUT/status.md
}
run_arm 4000  14x7/4,16x9/6 160    # B1 — recompute P for X=4000 first
run_arm 2000  14x7/4,16x9/6 120    # B2
run_arm 500   16x9/6        120    # B3 (P can go up)
run_arm 16000 14x7/4        60     # B4 stretch — P ≤ 2
run_arm 4000  12x9/4        120    # B5 stretch
```

`--games`/`--vs-games` are per rung and each board is its own rung, so `160` with two boards
plays 320 games. Rungs are named `vs:nn:<net> [...]@16x9/6`; the budget is only in `$tag`.
After **each** arm, append to `status.md`: points, W/D/L, paired SE and z vs 0.5, TD for/against,
s/game. A partial night must still leave a readable partial answer.

### 4. Morning write-up (the agent does this, ≈20 min)

- Fill `## Results` below: Part A table (X\*, top1 per budget, mean fan per board), Part B table
  (one row per arm × board: points, W/D/L, paired SE, z, TD/game for–against, s/game).
- Apply `## Decision rules`; write the one-line verdict under `**Verdict:**`.
- Cross-link: add a dated one-liner under plan 032 **#10** and under plan 042's **"Search budget
  per size"** open question pointing here. Do not edit anything else in those files.
- `runs/` is gitignored, so `status.md` will not be committed: paste the tables into this file
  and commit it (directly on `master`, per the repo's git workflow). Do not push.

## Decision rules (pre-committed, apply in order)

Read every arm as points with its paired SE; "significant" = z ≥ 2 against 0.5.

1. **Sanity.** If B1 on 14x7 comes out *below* 0.45 significantly, something is wrong (a
   higher budget of the same net should never lose): check `--mcts-workers 1`, the net path in
   both seats, and the rung labels before believing anything else.
2. **R-scale (H2 holds).** B1 ≥ 0.58 significant on 16x9 **and** B1 on 14x7 at least 0.08 points
   lower → the budget is binding on the large board specifically. Recommendation: a size-scaled
   budget, `iters(board) = 1000 × fan(board)/fan(14x7)` with the fan ratio from Part A (fallback:
   the cell ratio, 144/98 ≈ 1.47). This needs a knob the mixed sampler does not have yet (one
   `--mcts-iters` for all sizes) — file it as the implementation follow-up, do not build it
   overnight. Part C measures its generation cost.
3. **R-raise (both boards gain).** B1 ≥ 0.58 significant on both boards with a difference
   < 0.08 → the global budget is too low post-044, not a size effect. This is plan 032 #10's
   answer; if B2 also gains (≥ 0.55) 2000 is the cheap move, otherwise the trade is 4× generation
   cost per game vs 4× fewer games, which is a separate E2-style loop experiment. Recommend the
   experiment, not a blind change of `MCTS_ITERS`.
4. **R-keep.** B1 < 0.55 on both boards → 1000 is still fine on the larger boards; close plan
   042's open question with the numbers. If B3 is ≥ 0.47 on 16x9 too, note that the budget has
   slack even on the big board (do not act: capability, not speed, is the focus).
5. **Part A vs Part B disagreement.** If X\*(16x9) ≥ 2 × X\*(14x7) but Part B says keep, report
   "converges slower, doesn't play better" — the same shape plan 028 found at 14x7 (label
   agreement improves, strength does not). It informs label quality, not the loop's budget.
6. **B4 (16k vs 1000 on 14x7)**, if it ran: ≥ 0.60 confirms plan 028's prediction and upgrades
   032 #10 from "predicted" to "measured"; < 0.55 refutes it. Either way it only changes the
   ranking of #10, not `MCTS_ITERS`.

Whatever the outcome, the plan-025 caveat stands: the answer has a shelf life — it depends on the
net, the backup rule (plan 032 #2's mean backup is still not in the loop) and the board tier.
Re-run this file's Part B, not plan 027's, the next time any of those changes.

## Results

_(the agent fills these in)_

### Part A — convergence per board

| board | mean fan | X\* | top1@1000 | top1@2000 | top1@4000 | top1@8000 | signal/floor @1000 |
|---|---|---|---|---|---|---|---|
| 14x7/4 | | | | | | | |
| 16x9/6 | | | | | | | |

### Part B — strength

| arm | board | games | points | W/D/L | paired SE | z | TD/g for–against | s/game |
|---|---|---|---|---|---|---|---|---|
| B1 4000 v 1000 | 14x7/4 | | | | | | | |
| B1 4000 v 1000 | 16x9/6 | | | | | | | |
| B2 2000 v 1000 | 14x7/4 | | | | | | | |
| B2 2000 v 1000 | 16x9/6 | | | | | | | |
| B3 500 v 1000 | 16x9/6 | | | | | | | |
| B4 16000 v 1000 | 14x7/4 | | | | | | | |

**Verdict:** _pending_

## Cross-references

- `plans/completed/025-plan--search-budget-convergence--completed.md` — the method (signal vs
  floor, X\*), superseded findings, and the "board-tier-specific" caveat this plan acts on.
- `plans/completed/027-plan--search-strength-tuning--completed.md` E2 (budget ladder at 14x7) and E3 (horizon).
- `plans/completed/028-plan--search-noise-and-ensembles--completed.md` Stage 0 — the flat spot
  and the 16k prediction.
- `plans/032-plan--ranked-experiment-queue.md` — ground rules, power table, **#10**.
- `plans/042-plan--board-size-curriculum.md` — E0 (transfer across sizes at 1000), open question
  "search budget per size", E5 (capacity overhead, still unrun).
- `plans/044-plan--state-hash-discrimination.md` — why the DAG per iteration changed.
- `botbowl-mcts/src/dynamics.rs` `SearchBudget` — `Iterations` splits across workers,
  `Time` does not; `eval` and `convergence` are iteration-only, only `dataset` / `job generate`
  take `--mcts-time-ms`.
