# CLAUDE.md — botbowl-nn

Neural-network plumbing for the MCTS bot (plan 017): the **shared feature encoder**, the **offline prepare step**, and the **tract-onnx inference evaluator**. Depends on `botbowl-engine` + `botbowl-data`; **no** dependency on `botbowl-mcts` (mcts depends on nn, not vice versa). The PyTorch trainer lives outside the workspace in `../train/` (a `uv` project).

## The one rule: encoding lives here, in Rust, once

`encode.rs` is the **single source of tensor layout**, used verbatim by both the offline `prepare` bin and the live `NnEvaluator`. Python never parses a `GameState` — it only consumes the `.npy` batches `prepare` writes. This is deliberate: there is **no train/inference encoding skew** possible. If you change a channel, you change it here and the manifest's `spatial_channel_names`/`global_feature_names` update automatically; bump `NN_SCHEMA_VERSION` in `bin/prepare.rs`.

Shapes: spatial `C=37` planes × `H` × `W`, global `F=15`, policy head `A=30`. `H`/`W` are the **runtime `board_dims` including the 2-cell OOB border** — a `Position` indexes the tensor directly, and border cells are flagged by the `oob` plane and masked out of the policy (the network never emits masking; masking = "only gather legal actions", done identically in `eval.rs` and `train/model.py`).

## Perspective is a single authority (`perspective.rs`)

`mover_for(state)` = `available_actions.team.unwrap_or(team_turn)`. Everything perspective-dependent — the encoder, the action↔cell map, the value sign — routes through here. **Canonical view: the mover attacks toward `x=1`.** `Home` already does (`endzone_x(Home)==1`) → encoded verbatim; `Away` is **x-mirrored** (`x → (width-1)-x`, no y-flip; involutive). The network's value `v` is **mover-centric** (`+1` = team-to-move winning); Home-centric is recovered by flipping sign for an `Away` mover.

## Action ↔ policy cell (`actions.rs`)

Exhaustive matches pin `PosAT → 0..14`, `SimpleAT → 14..30`. Adding an engine action variant is a **compile error here** — a deliberate trip-wire forcing a schema bump. Positional actions target one canonicalised cell `(channel, y, x)`; simple actions have no cell and their logit is the channel-wide spatial **max**.

## Targets (`targets.rs`) — never train π on raw visit counts

`recon_mcts` freezes a solved child's visits, and the fastest-solving child is often the *best* move, so `π ∝ visits` is wrong (plan 017 §caveat). `policy_target`: root solved → one-hot on argmax mover-`Q` (or `Skip`); partially solved → visits, but the argmax-`Q` solved child is floored at the max unsolved-sibling visit count; else normalise visits. All `Q` comparisons are in the **mover's** frame (`Away` negates the Home-centric stored `Q`). `value_target` = mover-signed outcome `z`.

## prepare (`bin/prepare.rs`)

`prepare --in *.jsonl --out DIR [--solved-root onehot|skip] [--min-root-visits N]`. Streams `DatasetReader`, groups samples **by board dims** (one `dims_{w}x{h}` subdir each → fixed-shape batches for free), writes `spatial/global/value/chosen` `.npy` + a **CSR** ragged legal-action encoding (`actions.npy (M,4)`, `policy.npy (M,)`, `action_offsets.npy (N+1,)`) + `manifest.json`. `npy.rs` is a hand-rolled `.npy` v1.0 writer/reader (avoids the ndarray stack).

## eval (`eval.rs`) — frozen NN, pure function of state

`NnEvaluator::from_path` loads ONNX; the model has dynamic `H`/`W`, so we build one tract runnable **per `(H,W)`** on first use and cache it (board dims are constant per game → one entry). `priors(state, &actions)` = one forward, gather per-legal-action logits (positional cell / simple channel-max), softmax in Rust, **rescale ×`len`** so mean prior ≈ 1.0. `value_home_i64(state)` = mover-centric `v`, clamped to `[-1,1]`, sign-flipped Home-centric, **×1000** (matches `leaf_score`'s TD = ±1000). A frozen deterministic CPU net is a **pure function of state** → recombination-safe. The ×1000 and ×K rescales are calibration bridges coupled to `PUCT_C`/`leaf_score` in `botbowl-mcts`.

## Parity fixture

`tests/fixtures/tiny.onnx` + `parity_{h}x{w}_*.npy` are **committed** (built by `train/src/bbnn/fixture.py`). `tests/parity.rs` (not `#[ignore]`d) asserts tract == PyTorch to `< 1e-4` at two board sizes — this is what de-risks tract op coverage (Expand/Shape broadcast, ReduceMean, BN fold, dynamic axes). Rebuild the fixture after any `model.py` architecture change: `cd train && uv run python -m bbnn.fixture`.

```sh
cargo test -p botbowl-nn                 # unit + prepare integration + parity
cargo run -p botbowl-nn --bin prepare -- --in data/*.jsonl --out data/prepared/x
```
