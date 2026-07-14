# NN training pipeline + NN-backed MCTS bot (plan 017)

## Context

`botbowl-data` (commit `c3e204f`) defines the JSONL training-data schema (`Trajectory`/`Sample`/`ChildStat`), and
`botbowl-ui`'s `dataset` subcommand + `MctsBot::get_action_with_record` already generate it. This plan builds the rest
of plan 017's pipeline: a **feature encoder + prepare step (Rust)**, a **PyTorch trainer with ONNX export (Python)**,
and an **NN-backed `MctsBot`** (tract-onnx inference) to verify integration end-to-end with a throwaway smoke-test
model.

**Decided with user:** PyTorch → ONNX → tract-onnx (pure Rust CPU inference). All GameState→tensor encoding lives in
Rust — shared verbatim by the offline prepare step and the live evaluator, so Python never parses `GameState` and
there's no train/inference encoding skew. Real data-generation _strategy_ is deferred (user is still thinking); smoke
batch uses the existing `dataset` subcommand.

## Key verified facts

- Action space: 14 `PosAT` + 16 `SimpleAT` (`botbowl-engine/src/core/table.rs:5-40`) → policy head **A = 30** channels.
- Prior call site: `prior_for_engine_action` at `botbowl-mcts/src/dynamics.rs:276` (once per edge at expansion, cached
  in `BbAction::player(a, prior)`; un-normalized, BASE=1.0). Value call site: `leaf_score(state)` at `dynamics.rs:650`
  inside `score_leaf` (plan-018 gating above it stays untouched).
- `PUCT_C=10.0`, FPU, virtual-loss 30 are calibrated to `leaf_score`'s scale (TD=±1000) → NN value in [-1,1] must be
  **rescaled ×1000**; NN softmax priors rescaled ×`legal.len()` so mean prior ≈ 1.0.
- `BloodBowlDynamics` is `Copy` (`dynamics.rs:142`) — must drop `Copy` to carry an `Arc<NnEvaluator>`; only two
  construction sites (Default impl ~:153, literal in `run_search` ~:902).
- Board: compile-time capacity (default playable 26×15 → engine 28×17 with 2-cell OOB border) + runtime `BoardDims` ≤
  capacity per state. One binary serves all tiers; engine green down to engine 16×9 (14×7 tier).
- Purity invariant: frozen deterministic CPU NN is a pure fn of state → recombination-safe.
- Gotcha: `ScoreTdEasy::setup` hard-codes `carrier_y ∈ 3..=13` — small-tier lectures broken today. Smoke test runs on
  default board; small-tier path covered by a variable-dims encoder unit test. (Dims-relative lectures = separate
  follow-up.)

## New crate `botbowl-nn` (workspace member)

Deps: `botbowl-engine`, `botbowl-data`, `tract-onnx`, `serde{,_json}`, `clap` (bin). No dep on `botbowl-mcts` (mcts
depends on nn, not vice versa).

```
botbowl-nn/src/
  actions.rs       Action ↔ policy cell: exhaustive match pinning PosAT→0-13, SimpleAT→14-29
                   (enum-order; adding an engine variant = compile error → deliberate version bump).
                   action_cell(action, mover, dims) -> {channel, y, x, is_simple} + inverse.
  perspective.rs   Single authority: mover_for(state) = available_actions.team.unwrap_or(team_turn).
                   Canonical view = mover attacks toward x=1; Away → mirror x (no y-flip).
                   Value convention: network v is mover-centric; v_target = ±z_home by mover.
  encode.rs        encode(state) -> {spatial: C×H×W f32, global: F, h, w, mover}.
                   H/W = runtime board_dims incl. OOB border (tensor indexes Position directly;
                   border cells always masked for policy). C=37: 15 per side (present, standing,
                   stunned, used, movement_left, ST/MA/AG/AV normalized, 6 skill planes) +
                   ball_on_ground/in_air/carrier, active_player, us/them tackle-zone counts, oob.
                   F=15 non-spatial (half, us/them turn, scores+diff, rerolls+usable, blitz/pass/
                   handoff/foul available, turnover) — mover-perspective.
  targets.rs       Policy target per plan 017 §caveat: root solved → one-hot on argmax mover-Q;
                   partially solved → pseudo-counts (solved argmax-Q child gets ≥ max unsolved
                   sibling visits, other solved keep frozen count); else normalize visits.
                   Value target v1 = mover-signed outcome_value (z). Record choice in manifest.
  npy.rs           Minimal hand-rolled .npy v1 writer (+reader for tests) — ~120 lines, avoids
                   pulling the ndarray stack for serialization only.
  eval.rs          NnEvaluator: tract model_for_path + with_input_fact pinning concrete H/W per
                   BoardDims → into_optimized().into_runnable() (Send+Sync, run(&self)).
                   priors(state, &[Action]) -> Vec<f32>: ONE forward per expanded node, gather
                   per-legal-action logits (spatial: cell; simple: channel max), softmax in Rust,
                   rescale ×len. value_home_i64(state): (v.clamp(-1,1)*1000) as i64, sign-flipped
                   Home-centric via mover_for.
  bin/prepare.rs   prepare --in *.jsonl --out DIR [--solved-root onehot|skip] [--min-root-visits N]
                   Streams DatasetReader, groups by board_dims → one subdir per dims (fixed-shape
                   batches for free). Outputs: spatial/global/value/chosen .npy + ragged legal-
                   action CSR (actions.npy (M,4), policy.npy (M,), action_offsets.npy (N+1,)) +
                   manifest.json (versions, C/F/A, channel/feature names, value_target, git).
  tests/parity.rs  Committed tiny.onnx fixture + reference tensors; tract vs PyTorch outputs
                   max-abs-diff < 1e-4 at TWO board sizes (proves dynamic-axes concretization).
```

No mask/softmax/gather inside the ONNX graph — masking is "only gather legal actions", done in Rust/Python identically.

## Python trainer `train/` (uv project)

`pyproject.toml` deps: torch, numpy, onnx. `src/bbnn/`:

- `data.py` — np.load of one prepared dims-dir; collate pads ragged action lists to batch-K_max with pad mask.
- `model.py` — plan-017 tower: `Linear(F,16)+ReLU` → broadcast-expand to H×W → concat with spatial →
  `Conv3x3(C+16→64)+BN+ReLU` stem → 6 residual blocks → policy head `Conv1x1(64→30)` raw logits; value head
  `Conv1x1(64→32)+BN+ReLU → mean(dim=(2,3))` (ReduceMean, tract-safest) `→ MLP → tanh`.
- `train.py` — loss = masked policy cross-entropy (per-sample log-softmax over legal set, pad→-1e9; spatial logits
  gathered by flat index, simple = channel amax — mirrors Rust gather exactly) + value MSE. Adam lr 1e-3. Logs loss +
  chosen-action top-1 acc.
- `export.py` — ONNX opset 17, dynamic axes on batch/H/W, onnx.checker; dumps parity reference tensors at two (H,W).
- `fixture.py` — seeded tiny model (ch=16, 2 blocks) → `botbowl-nn/tests/fixtures/`.

## botbowl-mcts changes

1. `Evaluator` enum: `#[default] Heuristic | Nn(Arc<NnEvaluator>)`.
2. `BloodBowlDynamics`: drop `Copy`, add `evaluator: Evaluator` (fix 2 construction sites).
3. `available_actions` (~:276): Heuristic → unchanged per-action call; Nn → one `nn.priors(state, &filtered)` forward,
   zip into `BbAction::player`. NN **replaces** scripted priors (no blending — no principled scale between them).
   Pruning still applies first.
4. `score_leaf` (~:650): `Heuristic → leaf_score(state)`, `Nn → nn.value_home_i64(state)`. Gating untouched.
5. `MctsBot::with_evaluator(Arc<NnEvaluator>)` builder; default Heuristic → existing behavior byte-identical, all
   current tests/benchmarks untouched.
6. Document in `botbowl-mcts/CLAUDE.md`: frozen NN satisfies purity invariant; ×1000 scale bridge.

Two forwards per expanded node (priors at expansion + value at scoring) accepted — perf explicitly deprioritized.

## Step order & test points

1. **Scaffold + actions.rs** — `cargo test -p botbowl-nn`: 30-channel pin test, cell round-trip on real states' legal
   actions.
2. **perspective.rs + encode.rs** — golden cell test; mirror-consistency (Home encode == team-swapped x-mirrored Away
   encode); variable-dims test at `BoardDims::new(16,9,4)` → shape (37,9,16).
3. **targets.rs** — unit tests pinning every solved-correction branch + Away sign flips.
4. **npy.rs + prepare bin** — integration test (in-memory Trajectory → JSONL → prepare → re-read, assert
   shapes/offsets/π-sums). Then generate smoke batch:
   `cargo run --release -p botbowl-ui -- dataset --mode curriculum --lecture "Score TD" --difficulty easy --games 30 --mcts-time-ms 150 --out data/score_td.jsonl`
   → `cargo run -p botbowl-nn --bin prepare -- --in data/score_td.jsonl --out data/prepared/score_td`. Add `data/` to
   `.gitignore`.
5. **train/ + fixture** — `uv run pytest` (forward shapes at two sizes); overfit ≤100 samples → policy loss ~0 proves
   plumbing. Commit tiny.onnx + parity tensors.
6. **eval.rs + parity test** (not #[ignore]d — fixture committed). **De-risks tract op coverage (Expand/Shape broadcast,
   ReduceMean, BN fold, dynamic axes) before real training.** Fallback if broadcast fails under tract: export one
   fixed-shape ONNX per tier (same weights).
7. **MCTS integration** — `cargo test --workspace` green (default path unchanged); unit test with fixture evaluator
   asserting priors normalized-×K and value ∈ [-1000,1000].
8. **End-to-end smoke** — train ~20 epochs on step-4 data (loss decreases), export `models/score_td.onnx`; new
   `#[ignore]`d `botbowl-mcts/tests/nn_bot.rs`: reads `BLOOD_NN_MODEL` env (skip+message if unset),
   `MctsBot::new(Time(150ms)).with_workers(1).with_evaluator(...)` through `run_trials_cfg(ScoreTdEasy, 10 trials)` —
   asserts completion/legality, prints success rate vs heuristic bot for eyeballing (no rate assertion; smoke model
   isn't expected to be good).

Also: `botbowl-nn/CLAUDE.md` (crate convention) + note in `plans/017-...md` that implementation started.

**Untouched:** `botbowl-engine`, `botbowl-data` schema, `botbowl-curriculum`, `recon_mcts`, `botbowl-ui` (except nothing
— dataset cmd already exists), and within botbowl-mcts: `priors.rs`, `score.rs`, `pruning.rs`, `action.rs`, existing
tests.

## Verification

- `cargo test --workspace` — new unit/integration tests + all existing green.
- Parity test (step 6) — tract == PyTorch at two board sizes.
- Overfit run (step 5) — training plumbing works.
- Step 8 smoke — NN-backed bot plays legal Blood Bowl via the real search; side-by-side success rate print.

## Risks

- **tract op coverage** (broadcast Expand under dynamic H/W) — mitigated by concretizing H/W at load, ReduceMean over
  pooling ops, no graph-side masking, parity test early (step 6); fallback = per-tier fixed-shape export.
- **Perspective sign bugs** — single perspective.rs authority + mirror/sign tests at encoder, targets, evaluator.
- **Scale coupling** — ×1000 value, ×K priors documented as calibration bridges; PUCT_C retune is follow-up.
- **Leaf distribution shift** — net trained on decision states also scores terminal/past-horizon states; accepted v1
  (score-diff feature carries the signal).
