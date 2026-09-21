# Mixed-board-size training: make the net size-robust before the 26x15 jump

**Status:** Designed 2026-09-16. **Absorbed into `plans/042-plan--board-size-curriculum.md` on
2026-09-19**: phases 1–2 are implemented there (a centred size distribution with a uniform floor
generalises the uniform sampler; schema v7 carries the geometry features of §2a; the trainer reads
every `dims_*` dir), phase 0 is 042's E0, and the gate below is dropped — plan 040 found the
corpus is what has to change, and mixed sizes are that change. Kept for the size table, the
density argument and the capacity-build note, which all still hold.

## Idea

Generate training data on a *range* of board sizes instead of the single 14x7 tier, so the network
cannot memorise absolute positions and is forced to reason relative to the endzone planes and local
structure. The end goal is play on the full 26x15/11 board (plan 017 tiers); the hypothesis is that
a size-robust net transfers to the 20x11 and 26x15 tiers with far less data than a 14x7-only net.

Why we expect the shortcut to exist: the tower is 6 residual blocks of 3x3 convs (13 convs incl.
stem), receptive field ≈ 27 cells — it covers the entire 16x9 engine board. Zero-padding at the
tensor edge lets convs infer absolute x/y from distance to the border, so on one fixed size the net
can learn "x = 5 is good" rather than "3 squares from the endzone is good". Varying the size makes
the border-distance feature unreliable and the `us_td_zone`/`them_td_zone` planes the only stable
direction signal.

## What already works (no code needed)

- **Runtime board size within compiled capacity.** `BoardDims` (`botbowl-engine/src/core/model.rs`)
  is a `GameState` field; `GameStateBuilder::with_board_dims` overrides `BoardDims::from_env()`. One
  process mixes sizes freely — the web server does exactly this (plan 034 decision 8).
- **Fully-convolutional net.** `train/src/bbnn/model.py`: 1x1 policy head, global-average-pooled
  value head, the only `Linear` layers are on the 15-float global vector and after the pool. ONNX
  exports with dynamic `H`/`W`; `TractBackend` builds one runnable per `(h, w)` and caches it.
- **`prepare` already groups by dims** into `dims_{w}x{h}/` subdirs, so a mixed corpus prepares
  without changes and each dir is fixed-shape.
- **Corpus metadata carries the board:** `TrajectoryMeta.board_dims` per trajectory.

## What is missing

| gap | where | fix |
|---|---|---|
| Generator takes size from env vars per process | `botbowl-ui/src/dataset.rs` builds `GameStateBuilder::new()...build()`; `random_start.rs:144` `unwrap_or_else(BoardDims::from_env)` | `--board-sizes` sampler (phase 1) |
| Loader reads one dims dir | `train/src/bbnn/data.py::PreparedDataset(dims_dir)`; `train.py --data` single dir; `train_loop.sh:469` `ls -d dims_* \| head -1` | multi-dir dataset + per-dims batching (phase 2) |
| No board-size signal in the input | `encode.rs::global_feature_names()` has 15 features, none geometric; no coordinate / distance planes; plan 032 lines 1041/1062 already flag this for 20x11+ | schema v7 (phase 2) |
| Web server refuses a model whose `_WxH_` filename tag differs from the board | `botbowl-web/server/src/bots.rs::board_tag_of`, `session.rs:639` | new tag convention (phase 3) |
| Density not tied to area | `BOARD_PLAYERS` is one number | team size derived from area (phase 1) |

`botbowl-web/CLAUDE.md` says a model trained on another size "panics inside `NnEvaluator`". The tract
path is dynamic-shape, so the panic is not the tensor shape. **Phase 0 step 1 reproduces this and
finds the actual cause** before anything else — if it is real it also blocks the zero-shot eval.

## Engine constraints on the size range

`BoardDims::new` asserts **playable width even and ≥ 8, playable height odd and ≥ 3**, plus non-empty
LOS and wing ranges (`los_half = (H_engine − 4) / 4`, wings are the rows outside the LOS band). The
proposed 10–17 × 5–10 range therefore collapses to:

| playable W | playable H | engine tensor | area | team size @ ~26 cells/player |
|---|---|---|---|---|
| 10 | 5 | 12x7 | 50 | 2 |
| 10 | 7 | 12x9 | 70 | 3 |
| 12 | 5 | 14x7 | 60 | 2 |
| 12 | 7 | 14x9 | 84 | 3 |
| 12 | 9 | 14x11 | 108 | 4 |
| 14 | 5 | 16x7 | 70 | 3 |
| **14** | **7** | **16x9** | **98** | **4** (current tier) |
| 14 | 9 | 16x11 | 126 | 5 |
| 16 | 7 | 18x9 | 112 | 4 |
| 16 | 9 | 18x11 | 144 | 6 |

Plan 017's tiers all sit at 25–28 cells per player (14x7/4 = 24.5, 20x11/7 = 31, 26x15/11 = 35).
Fixing `team_size = clamp(round(w·h / 26), 2, capacity)` keeps the game recognisable across sizes;
sampling only (w, h) with a fixed 4 players would train on 5% to 16% density, which is a different
game at each end. Note `kickoff_table_enabled` flips at `team_size >= 7` — all rows above stay below
it, so the kickoff table is uniformly off in this experiment. It turns on at the 20x11/7 tier.

**Build capacity:** build at the *largest sampled* size, `BOARD_SIZE_W=16 BOARD_SIZE_H=9
BOARD_PLAYERS=6`, not at the 26x15/11 default. `FullPitch` is `[[T; HEIGHT]; WIDTH]`, so every
`GameState` clone pays for capacity, not for the active board; 28x17 arrays for a 12x5 game would
cost ~3x on the generator for nothing. The 16x9 capacity is the one the engine test suite is
already green on (one pre-existing failure, see engine CLAUDE.md).

## Phase 0 — zero-shot diagnostic (cheap, run first, no training)

Goal: measure how badly the *current* 14x7 net degrades on other sizes. This is the baseline any
mixed-size net must beat, and if the degradation is small the rest of the plan drops in priority.

1. Reproduce (or refute) the "panics inside `NnEvaluator`" note: build at 16x9/6 capacity, run
   `eval` with `--evaluator nn --model <14x7 net>` under `BOARD_SIZE_W=16 BOARD_SIZE_H=9
   BOARD_PLAYERS=6` at runtime. Fix or document whatever actually breaks.
2. Ladder the 14x7 net vs the scripted bot at each of: 12x5/2, 14x7/4 (control), 16x9/6, and one
   off-distribution shape (12x9/4, tall-narrow). Same `--mcts-iters`, same game count as the
   14x7 promotion gate in plan 030.
3. Record the table (win rate, TD diff, avg game length) in plan 031 under diagnostics. Decision rule:
   - drop of < 5 pp at 12x5 and 16x9 → net already size-robust; park phases 1–3, move on to the
     20x11 tier directly (plan 017 next-tier note).
   - drop of ≥ 15 pp → proceed to phase 1.
   - in between → proceed, but treat phase 2's schema change as the primary intervention.

Cost: one release build, ~4 ladder runs, no GPU.

## Phase 1 — per-game size sampling in the generator

1. `dataset` gets `--board-sizes 10x5,12x5,12x7,14x7,14x9,16x7,16x9` (playable dims, comma list;
   default unset = today's `from_env` behaviour so existing scripts are untouched) and
   `--cells-per-player 26` (team size derived per game, clamped to `[2, TEAM_SIZE]`). Parse via the
   existing `parse_size` in `cli.rs`. Validate every entry through `BoardDims::new` at startup so a
   bad list fails before the first game, not mid-run.
2. Sample one entry per game, uniform, from the generator's seeded RNG (reproducibility per seed
   is already broken by recon_mcts hash order — memory note — so don't over-engineer this; just
   keep it seeded).
3. Plumb through both modes: self-play (`dataset.rs:352-358`) via `with_board_dims`, and random-start
   via `RandomStartConfig.board_dims = Some(dims)` (`cli.rs:181` currently passes `None`).
4. `random_start.rs` already takes `dims` for placement; audit `own_side_feature` / wing logic at
   12x7 (engine) since that is narrower than anything it has run on. Expect a bug harvest — every
   new board size is a fuzzing campaign (plans 020/021). Run the `dataset` smoke at each size for a
   few hundred games before the real run.
5. Lectures stay out of scope: the battery is full-pitch-hardcoded and already skipped on 14x7
   (plan 021 note). Mixed-size corpus = self-play + random-start only.
6. Corpus stamping: `TrajectoryMeta.board_dims` already records it. Add the size list to
   `runs/*/status.md` so the run is self-describing.

## Phase 2 — training on a mixed corpus

### 2a. Schema v7: board-size features

Add to the spatial tensor (all cheap, all `u8`-exact, all derived from `state.board_dims`):

- `dist_to_us_endzone` — |x − endzone_x(us)| per cell, clamped to 255. Makes "3 squares from the
  endzone" a direct input instead of something to infer from the border.
- `dist_to_sideline` — min(y − 1, height − 2 − y). Same reasoning for the y axis.

Add to the global vector: `playable_w`, `playable_h`, `team_size` (scaled by 26, 15, 11 resp. so
the full board reads 1.0). The value head needs these: the probability of a TD in the remaining
turns depends on pitch length, and a pooled fully-conv net has no other way to know it.

Bump `NN_SCHEMA_VERSION` to 7, mirror `C` in `model.py`, rebuild `tiny.onnx` + parity fixtures and
the 14x7 goldens per `capacity_parity.rs`. This invalidates every checkpoint, so **land it together
with phase 1, not before**, and only once phase 0 has justified the experiment.

Rejected: coordinate planes with absolute x/y. They reintroduce exactly the memorisable feature we
are trying to remove; distance-to-endzone is the relative form and is what transfers.

### 2b. Multi-dims loader

- `PreparedDataset` grows a sibling `MultiDimsDataset(list_of_dirs)` that owns one
  `PreparedDataset` per dir and a batch sampler that draws each batch from a single dir (shapes must
  match within a batch; padding boards up to a common size would change the `oob` statistics and is
  rejected). Dir choice per batch is proportional to sample count so no size is over-weighted.
- `train.py --data` accepts a directory containing `dims_*` subdirs (auto-detect) as well as a
  single dims dir. Same for `--val-data`. `train_loop.sh` passes `prepared_train/` instead of
  `head -1`.
- Validation reports loss **per dims group**, not just pooled, so a size that lags is visible.
- `y-flip` augmentation is size-agnostic and stays. Consider adding x-mirror + mover-swap once the
  canonicalisation is confirmed symmetric; not part of this plan.

### 2c. Experiment protocol

Two nets, same architecture, same step budget, same MCTS generation budget per game:

- **A (control):** 14x7/4 only, today's pipeline.
- **B (mixed):** sizes from the phase 1 list **with 16x9 held out** of training.

Evaluate both vs the scripted bot on: 14x7/4 (in-distribution for both), 12x5/2 (in-dist for B
only), **16x9/6 (held out for B, out-of-dist for both)**, and 20x11/7 (the real target of the next
tier — needs a 20x11 capacity build for that ladder only). Also run **B vs A head-to-head** on 14x7
to check mixed training did not cost in-tier strength.

Success = B beats A on 16x9 and 20x11 by more than the noise band of the ladder (plan 030 game
counts), while staying within noise of A on 14x7. If B is worse on 14x7 by more than that, the
mixed data is diluting the in-tier signal and the answer is more data or a size-weighted sampler,
not abandoning the idea.

Write results to plan 032's ranked queue, not here.

## Phase 3 — plumbing for the web and for the next tier

- Replace the `_WxH_` filename convention with a capability tag (`_anyboard_` or a `WxH-WxH` range)
  and have `board_tag_of` accept a model whose range contains the requested board. Keep the strict
  match for legacy names.
- 20x11/7 tier: `kickoff_table_enabled` turns on. That is a new rule surface for the encoder (kickoff
  events) and the generator; budget a bug harvest before treating 20x11 numbers as signal.
- If B generalises, the from-scratch run for the 20x11 tier should be seeded from B's weights
  (`--init`), not from A's.

## Open questions

- **Uniform vs. curriculum over sizes.** Uniform sampling is the simplest and what this plan runs.
  A schedule (small early, large late) is plan 017's original idea; only worth it if uniform fails.
- **Search budget per size.** Fixed `--mcts-iters` gives the search less relative coverage on bigger
  boards (more legal moves). Scale iters by legal-move count, or accept it as part of the domain?
  Start fixed; it is what the 26x15 target will face anyway.
- **Value target scale.** The drive-outcome target is ±1 regardless of size, but the *rate* of
  scoring differs by size. The global size features (2a) are meant to let the value head absorb
  this; check the per-dims validation loss to confirm.
- **Does the 26x15 jump need an intermediate 20x11 corpus at all**, or does a size-robust B plus
  the 11-player density transfer directly? Cheapest answer: run B zero-shot on 26x15/11 once, with
  a 26x15 build. It will be bad, but the number tells us how far off we are.

## Cross-references

- `plans/017-plan--neural-network-progressive-board-sizes.md` — tier table, cross-tier weight
  transfer question (this plan is the experiment that answers it).
- `plans/032-plan--ranked-experiment-queue.md` lines 1041, 1062 — encoder gaps at 20x11+ that 2a
  closes; line 1306 — kickoff deviate scaling with board.
- `plans/completed/034-plan--web-play-ui.md` decision 8 — runtime `with_board_dims`.
- `plans/completed/021-plan--drive-bounded-corpus-and-gen1-loop--completed.md` line 70 — next-tier note.
- `botbowl-nn/CLAUDE.md` — schema layout, `td_zone` planes, `capacity_parity.rs`.
