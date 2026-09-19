# Plan 042 — Board-size curriculum: a centred size distribution with a uniform floor

**Status:** Code landed 2026-09-19 (this commit); **experiments not started.** Absorbs plan 039's
phases 1–2 (the per-game size sampler, schema v7's geometry features, the multi-dims trainer)
and keeps 039's phase 0 as experiment E0 below. Plan 017's tier table stays the reference for
the boards themselves. New results go into plan 032's ranked queue as usual; this file holds the
design, the protocol and the decision rules.

## Idea

Generate training data on a *distribution* of board sizes rather than one tier, and schedule
that distribution with a single scalar: a **centre** in playable area, smeared by a
**temperature**, mixed with a **uniform floor** so every size stays in every generation. The
sampler is a strict generalisation of the two earlier plans — temperature → ∞ is plan 039's
uniform sampling, temperature → 0 is plan 017's single tier — with one knob to move.

Three reasons this is the right next corpus change, in order of weight:

1. **The 14x7 loop is flat and plan 040 says the corpus has to change.** gen10–19 fit a
   constant 0.611 vs the anchor (χ² 8.39 on 9 df), and neither a wider window nor a from-scratch
   refresh moved val_policy on the same data. Mixed sizes are new *positions*, not more of the
   same ones, and they are cheaper than more 14x7 drives per generation.
2. **Size robustness is necessary for the real game.** The target is 26x15/11, where random
   play essentially never scores (plan 031 D8: the search rarely sees a TD even at 14x7 with 1000
   iterations). Small boards are the sparse-reward fix; a net that has only ever seen one size
   can read absolute position off its distance to the zero padding (plan 039's argument) and
   nothing it learned transfers. The curriculum is how learning on small boards reaches the
   big one.
3. **Honesty needs variety on both sides.** A fixed eval set of sizes, independent of the
   training centre, is what stops the number from following the curriculum.

What this does *not* address, so nobody reads it as the whole path to human strength: the value
target is drive outcome, not game result (plan 017's 2-1-grind trade-off); team size ≥ 7 turns on
the kickoff table, a new rule surface at 20x11; and human strength lives in the full skill and
roster ruleset, which board size does not touch.

## The sampler

`botbowl-play/src/board_sizes.rs`, pure and serialisable, shipped inside `GenerateConfig` so hub
and worker draw the same board for the same seed.

- **Legal grid.** Every playable `(w, h)` with `w` even ≥ 8, `h` ≥ 3, inside the compiled capacity,
  whose aspect `w/h` lies in a band (default 1.5–2.8; plan 017's tiers span 1.73–2.67, and 12x9 /
  10x7 fall outside as plan 039's deliberate off-distribution shapes). Team size is the density
  rule `clamp(round(w·h / 26), 2, capacity)` so the game is the same game at every size.
- **Weight.** `exp(−(ln(area/centre))² / 2T²)` per board, normalised, then
  `(1 − floor) · that + floor / n`. `T = 0` puts the centred mass on the nearest legal area.
- **Draw.** `sample(seed)` seeds its own ChaCha stream from the game's seed, so the corpus stays
  re-derivable and the choice never depends on which worker plays the game.
- **Provenance.** `TrajectoryMeta.board_dims` per trajectory (already), plus `extra.size_dist`
  with the distribution's label. The loop's `status.md` prints the resolved table.

A `--board-sizes 12x5,14x7:3,16x9/6` list form exists too (optional `/T` team, `:weight`), and is
what the eval ladder takes: each listed board is its own rung set.

## What landed (code map)

| where | what |
|---|---|
| `botbowl-engine` `BoardDims::try_new` | the board rules as a `Result`, so the grid can be enumerated without panics |
| `botbowl-play/src/board_sizes.rs` | `SizeDist` (list / centred), `legal_grid`, `team_size_for`, `parse_board`, `board_label` |
| `GenerateConfig.board_sizes` | per-game draw in self-play and random-start; curriculum mode ignores it |
| `play_ladder_game(.., board)` | ladder games on an explicit board; `EvalGameLine.board`, `LadderRow.board`, rung `opponent@14x7/4` |
| `botbowl-ui dataset` | `--board-sizes` \| `--size-centre --size-temperature --size-floor --size-aspect --size-max-area`, `--cells-per-player` |
| `botbowl-ui eval` | `--board-sizes` (fixed list, one rung set per board); `Report.board_env` lists them |
| `botbowl-hub job generate/eval`, proto v3 | same flags; `RungReq.board`, `Task::Eval.board`; `EvalGameLine`'s hand-written `Serialize` keeps the JSON line format byte-identical while the postcard wire always carries the field |
| `botbowl-nn` schema **v7** | planes `dist_to_us_endzone`, `dist_to_sideline` (C 59→61); globals `playable_w`, `playable_h`, `team_size` (F 15→18); fixtures and 14x7 goldens rebuilt |
| `train/bbnn/data.py` | `MultiDimsDataset` + `PerDimsBatchSampler` (a batch never mixes boards; batches per board ∝ samples), `open_prepared`, `make_loader` |
| `train/bbnn/train.py` | `--data`/`--val-data` accept a `prepared_*/` dir of `dims_*`; per-board val lines `val@dims_16x9: …` |
| `scripts/train_loop.sh` | `SIZE_MODE=fixed\|centred\|list`, `BUILD_*` capacity, `EVAL_BOARD_SIZES`, size baseline from gen00, `size_centre.txt` schedule; **the `head -1` that silently dropped every board but the first is gone** |
| `scripts/td_rate.py` | per-board breakdown (`per_board` in `--json`) |
| `scripts/size_curriculum.py` | the advance rule (below) |
| `scripts/eval_summary.py`, `anchor_curve.py` | `--board` filters a multi-size report |

Every default is the old behaviour: no size flag → the env board, bare rung names, unchanged line
bytes. `SIZE_MODE=fixed` is the 14x7 loop as it was, except that models are named by `$TIER`.

## The schedule

`size_curriculum.py` runs after every generate phase in `SIZE_MODE=centred`:

- pool the drives of every board with area ≥ 0.85 × centre ("at or above the centre");
- their TD/drive, divided by the same boards' TD/drive in `size_baseline.json` (the gen00
  heuristic corpus, generated on the same distribution) is the **relative** rate;
- if ≥ `SIZE_ADVANCE_TD` (0.75) over ≥ 200 drives, `centre ← min(centre × SIZE_STEP (1.25), SIZE_MAX_AREA)`;
- never moves down: the floor keeps every size present, and a regression is the per-size
  ladder's job to show. Advisory in plan 030's sense — `size_centre.txt` is a file the operator
  can overwrite at any phase boundary. History in `size_curve.tsv`.

Why the corpus TD rate and not the vs-scripted ladder: the ladder is the honest number but
costs games (a ±0.03 effect needs ~600, plan 032 D10), while the corpus's own conversion rate is
free, already tracked, and is exactly the sparse-reward signal small boards exist to provide. The
ladder confirms per size; the rule only decides where the *next* corpus is centred. TD rate is
confounded by size (bigger boards score less at any skill), which is what the per-board baseline
divides out.

## Experiments

Ground rules are plan 032's: paired Home/Away on a shared seed base, points `(W + D/2)/N`, SE ≈
0.041 at 120 games, 120-game screens resolve only |Δ| ≥ 0.10, commit before launching. "The
latest champion" below means whatever `champion.txt` names when the experiment is run — a run is
in flight and there will be a newer net; nothing here depends on which.

### E0 — zero-shot size degradation of the latest champion (no training, run first)

Plan 039's phase 0. Measures how much the size axis matters *before* any mixed corpus exists,
and is the baseline every mixed net must beat.

```sh
# one release build at 16x9/6 capacity
BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9 cargo build --release -p botbowl-ui -p botbowl-hub -p botbowl-worker
# the ladder, every rung on every board, 120 games each (SE 0.041)
target/16x9/release/botbowl-ui eval --evaluator nn --model $(cat runs/loop14x7/champion.txt) \
    --mcts-iters 1000 --games 120 --seed 0 --skip-lectures --rungs scripted \
    --board-sizes 12x5,14x7,16x9,12x9/4 --parallel-games 8 \
    --out runs/exp042/e0_report.json --per-game-out runs/exp042/e0_games.jsonl
train/.venv/bin/python scripts/eval_summary.py runs/exp042/e0_report.json
```

Note the champion is a schema-v6 net and this commit's encoder is v7: **E0 must run on the
parent commit's binaries** (v6 encoder, no size sampler needed — `--board-sizes` on eval is the
only 042 feature it uses, so cherry-pick the eval-side change or rebuild the ladder by hand from
`BOARD_SIZE_*` env per size). The cheapest honest route: one `eval` process per board with the
env board set, on the pre-042 binary, same games and seed. Record per board: points vs scripted,
TD/g for and against, draw rate.

Decision rule (plan 039's, unchanged): drop < 5 pp at 12x5 and 16x9 relative to 14x7 → the net
is already size-robust, run E2 with a lower prior and move the centre faster; ≥ 15 pp → E2 as
written; between → E2 with the schema change treated as the primary intervention.

### E1 — per-board baselines (cheap, no GPU)

Two reference rates the rest reads against.

1. **Corpus baseline** for the advance rule: the heuristic-MCTS random-start corpus on the full
   grid. `train_loop.sh` produces it as gen00 of any `SIZE_MODE=centred` run (and copies its
   `td_rate.json` to `size_baseline.json`); by hand:
   ```sh
   botbowl-ui dataset --mode random-start --games 300 --mcts-iters 1000 --evaluator heuristic \
       --size-centre 98 --size-temperature 100 --size-floor 1.0 --out runs/exp042/baseline.jsonl
   train/.venv/bin/python scripts/td_rate.py runs/exp042/baseline.jsonl --json runs/exp042/size_baseline.json
   ```
   (`--size-floor 1.0` is uniform over the grid; 300 games ≈ 25 per board on the 16x9 grid.)
2. **Ladder reference**: scripted vs scripted and heuristic-MCTS vs scripted at each eval board,
   120 games — the TD/g the eval rungs "should" show at each size, so a candidate's per-size TD/g
   can be read as a ratio the way the corpus rate is.

### E2 — the A/B/C: fixed tier vs centred curriculum vs uniform

Three loops, same everything except the size distribution. Same commit (this one — every arm on
schema v7 so the encoder is not a confound), same games per generation, same `MCTS_ITERS`,
same step budget per fine-tune, same `WINDOW_GENS`, same number of generations. Seed each from
the same random-init gen00 (`scripts/make_random_net.py`) via the bootstrap path, so no arm
inherits a 14x7-only head start.

| arm | `SIZE_MODE` | knobs | what it tests |
|---|---|---|---|
| **A** control | `fixed` at 14x7/4 but built at 16x9/6 capacity (`BUILD_W=16 BUILD_H=9 BUILD_PLAYERS=6 SIZE_MODE=list SIZE_LIST=14x7`) | one board | today's loop on the new encoder and capacity, so A vs B isolates the *distribution* |
| **B** curriculum | `centred` | centre 98, T 0.3, floor 0.2, advance 0.75 / step 1.25 / max 144 | the plan |
| **C** uniform | `centred` | T 100, floor 1.0 (≡ plan 039 phase 1) | does the schedule buy anything over just mixing |

All three eval on `EVAL_BOARD_SIZES=12x5,14x7,16x9` every generation: anchor rung (the *same*
frozen anchor for all arms — pick it before launch and never retrain it) and `EVAL_RUNGS=scripted`,
`ANCHOR_GAMES=120` per board so a single generation's point resolves 0.10 and the 3-gen rolling
mean 0.06. That triples the eval cost against today's 40-game 14x7-only anchor; it is the price
of a per-size curve and E0 will say whether 12x5 can be dropped from the set.

Held-out probes, run once on each arm's final net (not every generation):
- **12x9/4** — inside capacity, outside the aspect band: never trained on by any arm;
- **20x11/7** — the next tier; needs a `BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7` build
  for the ladder only, and turns the kickoff table on. Expect a bug harvest before the number
  means anything (plans 020/021: every new size has been one).

Sizing: 8 generations each. At ~7 h/generation (gen10–19 on one box plus a helper) that is
~56 h per arm, ~170 h total, or ~60 h with three boxes on the hub in parallel. Cut to 6
generations per arm if machines are short; below that the anchor curve is noise.

**Success** (pre-committed):
- B ≥ A + 0.10 on 16x9 and on 12x9 at 120 games (the transfer claim), and B within 0.10 of A on
  14x7 (mixed data did not cost in-tier strength);
- B ≥ C on 16x9 by the end, or B's per-size curve reaches the same point in fewer generations
  (the schedule claim). If C ≡ B, keep the sampler and drop the schedule — uniform is simpler;
- B's centre reached 144 (16x9) by the last generation, and each advance coincided with the
  vs-scripted rung at that size clearing 0.60 within a generation either way (E3 below).

**Failure** looks like: B ≤ A on 14x7 by more than 0.10 (dilution — the answer is more games
per generation or a higher floor on the centre board, not abandoning the idea, per plan 039),
or B's 16x9 curve flat while its 14x7 curve rises (the geometry features are not enough and the
net still reads position off the padding — inspect the per-board `val_value` lines first).

### E3 — does the advance rule move for the right reasons?

Free, read off B's logs. For every generation where `size_curve.tsv` says `advance`, compare the
per-size vs-scripted points at the boards ≥ the old centre: the rule is calibrated if those
boards already read ≥ 0.60 (a clear margin at 120 games) and mis-calibrated if it advanced while
they read ≤ 0.50 (too eager — raise `SIZE_ADVANCE_TD` or `--min-drives`) or held while they read
≥ 0.70 for two generations (too timid — lower it). Report the table in plan 032; retune once,
not per generation.

### E4 — the 20x11 tier

Only after E2 says B transfers. Rebuild at 20x11/7 capacity (`BUILD_W=20 BUILD_H=11
BUILD_PLAYERS=7`), `SIZE_MAX_AREA=220`, `EVAL_BOARD_SIZES=14x7,16x9,20x11`, warm-start gen01 from
B's last `.pt` (`INIT_CHAMPION`, `WARM_FROM=latest`). First a bug harvest on the kickoff table
(team size 7 enables it; `dataset` smoke at 20x11 for a few hundred games before the real run).
Then the zero-shot question plan 039 left open: the finished net on 26x15/11, one 120-game
ladder vs scripted on a default-capacity build. It will be bad; the number says how far off.

### E5 — one measurement, not a tuning campaign

`FullPitch` is sized by capacity, so a 16x9-capacity binary clones 18x11 arrays for a 12x5 game.
Measure games/hour of the 14x7 corpus once at 14x7 capacity and once at 16x9 capacity, same
seeds, same box, and write the ratio here. It sizes the trade of building at 20x11 capacity for
E4 (and whether the small boards should run on a separate small-capacity worker pool — the hub
already refuses a capacity mismatch, so that would be two hubs).

## Open questions

- **Search budget per size.** Fixed `--mcts-iters` gives the search less relative coverage on a
  bigger board. Start fixed (it is what 26x15 will face); if B's 16x9 TD rate lags its 12x5 rate
  by more than the baseline ratio predicts, try iters ∝ legal-move count as a plan-032 item.
- **Value target scale across sizes.** The drive-outcome label is ±1 everywhere but the base
  rate differs by size; the `playable_w/h` globals exist so the value head can absorb it. The
  per-board `val_value` lines are the check — a size whose val_value sits far above the others'
  is not being conditioned on.
- **Aspect band.** 1.5–2.8 keeps 12x9 and 10x7 out as probes. If E2 shows the band boards
  transfer to 12x9 anyway, widen it and lose the probe; if not, that is a finding too.
- **Eval cost.** Three boards × anchor + scripted at 120 games is ~720 games per generation.
  E0 decides whether 12x5 stays in the set; the anchor could also run on two boards and scripted
  on all three.
- **Formations on small boards.** `Formation::fits` already gates them; the LOS/wing bands
  changed for every non-full board on 2026-09-18, so nothing before that commit is comparable
  at any size — one more reason E2's arms all start from the same random gen00.
- **Web play.** `board_tag_of` reads `_WxH_` from a model name; `bbnet_mix16x9_genNN` has no
  such segment and is offered for every board, which is the right behaviour for a mixed net.
  A capability tag (`_anyboard_` or a `WxH-WxH` range, plan 039 phase 3) is still the cleaner
  spelling once such nets exist.

## Order

1. E0 on the pre-042 binaries (a day of eval time, no GPU).
2. E1's corpus baseline falls out of arm B's gen00; run the ladder reference alongside E0.
3. E2, three arms on the hub, 6–8 generations each.
4. E3 from B's logs; retune once.
5. E4 only if E2's transfer claim holds.

## Cross-references

- `plans/039-plan--mixed-board-size-training.md` — the uniform-sampling design this absorbs;
  its size table, the density argument, and the capacity-build note all stand.
- `plans/017-plan--neural-network-progressive-board-sizes.md` — the tier table; the
  "curriculum promotion criteria" it deferred are the advance rule here.
- `plans/040-plan--mechanism-1-window-and-refresh.md` — why the corpus, not the recipe, is
  the lever now.
- `plans/032-plan--ranked-experiment-queue.md` — where E0–E5's results go; D10 for the game
  counts every threshold above is sized against.
- `botbowl-play/CLAUDE.md`, `botbowl-nn/CLAUDE.md`, `botbowl-hub/CLAUDE.md` — the invariants
  the code map above touches.
