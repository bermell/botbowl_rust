# Encoder schema v6: every skill, factored sides, endzones, path probabilities

**Status:** Landed (commits `b71a656`, `2dd29a7`, `4ba7141`, `92555aa`, 2026-09-16), with two
follow-ups open — see *Open items*. Written after the fact so the reasoning behind a schema that
invalidated every checkpoint is recoverable; the per-decision detail lives in
`botbowl-nn/CLAUDE.md`, this is the narrative and the arithmetic.

The spatial tensor went **C = 37 → 59** and the corpus dtype `f32 → u8`. Net effect on size:
**21,312 → 8,496 bytes/sample**, i.e. smaller than before while carrying 39 skills instead of 6,
plus endzones and path probabilities.

| step | C | B/sample | schema |
|---|---|---|---|
| before | 37 | 21,312 | v2 |
| a plane per skill | 103 | 59,328 | v3 |
| u8 raw counts | 103 | 14,832 | v4 |
| unpaired per-player planes + endzones | 58 | 8,352 | v5 |
| path probabilities | 59 | 8,496 | v6 |

## Why each step

### 1. A plane per skill (`b71a656`)

The encoder carried a hand-picked six skills, frozen to keep older checkpoints loadable. Once the
curriculum could field players with arbitrary skills that subset was actively wrong — a Guard and a
plain lineman encoded identically. `Skill::ALL` / `COUNT` / `index()` now live in
`botbowl-engine/src/core/table.rs`; the `index()` match is exhaustive, so adding a variant is a
compile error there, next to the schema bump it forces.

**Rejected:** gating the planes on skills the engine actually implements (~12 of 39). It looks like
the obvious saving, but every skill subsequently implemented would re-invalidate every checkpoint.
Encoding all 39 buys schema stability for the rest of the roadmap.

### 2. u8 raw counts (`2dd29a7`)

Every spatial channel is an integer quantity — a flag, a characteristic, a tackle-zone count — so
`u8` is **exact, not a quantisation**. `encode_raw` became the source of truth and `encode` is
defined as `encode_raw / spatial_channel_scales()`, so the two views cannot drift; the identity is
pinned bit-for-bit. Scales ride in `manifest.json` and `train/src/bbnn/data.py` divides by them —
never hardcoded in Python, and a manifest without them is a hard error rather than a default,
because that is the one remaining place train/inference skew could enter.

The live path (`eval.rs`, `remote.rs`) is untouched and still `f32`: the win is disk and dataloader,
not inference.

**Rejected:** quantising the socket payload too. `movement_left / MOVE_NORM` is not
u8-representable, so it would break the `< 1e-3` tract-parity canary.

### 3. Unpaired per-player planes (`4ba7141`)

A square holds at most one player (`add_new_player_to_field` rejects an occupied cell), so
`us_present` / `them_present` already identify the owner. Duplicating the other 47 per-player planes
per side was pure redundancy: **103 → 56**.

Lossless, and recoverable in the *stem* rather than deep in the tower. For binary planes `a` with
`b = us_present`:

- "us has `a`" = `ReLU(a + b − 1.5)`
- "them has `a`" = `ReLU(a − b − 0.5)`

For a characteristic `v`: "the mover's `v`" = `ReLU(v + b − 1)`. Spending 39 channels to save the
network one ReLU was a fine trade at 6 skills and a bad one at 39.

**The `v ≤ 1` constraint is why the characteristics could not simply come along.** Four of the five
already held; `movement_left` did not — `total_movement_left()` is `ma + 2`, so MA 9 over a divisor
of 10 encoded to **1.1**, and an opponent's value would have leaked through as `v − 1`. Fixed at the
root rather than by patching the divisor: `PlayerStats::MAX_ST/MA/AG/AV/MOVEMENT` are now the
engine's characteristic caps, the encoder divides by them, and the curriculum's sampling ranges are
asserted to stay under them (`sampling_ranges_stay_under_the_engine_caps`). Raise a cap and both
sides follow.

Tackle zones stay paired: they are neighbour counts, not attached to the player in the cell, and
both teams can cover one square at once.

### 4. Endzone planes (`4ba7141`)

The tower is translation-equivariant and `oob` is symmetric in `x` (`is_out` is
`x <= 0 || x >= width - 1`), so **nothing in the tensor distinguished the scoring end from the
conceding one**. Canonicalisation fixes the convention but cannot communicate it; the policy head
had no basis to prefer a move toward `x = 1` over the identical move toward `x = w − 2`.

`us_td_zone` / `them_td_zone` are read from `dims.endzone_x` rather than hardcoded to the canonical
column, so they cannot drift from the rules. `td_zones_are_canonical_for_either_mover` asserts they
land on `x = 1` / `x = w − 2` under **both** movers, which doubles as an independent check on the
`Away` x-mirror.

### 5. Path probabilities (`92555aa`)

One plane: the pathfinder's per-square probability that the **active player** arrives — the move's
cumulative success chance across every dodge, GFI and pickup on the way. Already computed by the
engine, and not something a policy head should have to rediscover from geometry.

**It cannot be read off `state.path_buffer`**, which is the obvious implementation and is wrong
twice over:

```rust
#[serde(skip, default)]
#[derivative(PartialEq = "ignore")]
pub(crate) path_buffer: ...
```

- **`serde(skip)`.** Measured, not inferred: a state with 167 reachable squares comes back from a
  serde round-trip with `has_paths == true` and **zero**, and the two states still compare equal.
  `prepare` re-encodes from deserialised JSONL, so the plane would be all zeros in training and
  populated at inference — silent skew, with the loss converging happily throughout.
- **`PartialEq = "ignore"`.** Two states that recombine to one DAG node may hold different buffers,
  so a prior depending on it would be impure — the recombination invariant in the root `CLAUDE.md`.

So the encoder calls `PathFinder::player_paths`, a pure function of `(state, id)`.
`recomputed_paths_match_the_engines_own_buffer` pins that it agrees with the engine's own buffer
square for square; `the_path_plane_survives_the_corpus_round_trip` pins the property that motivated
the whole design. The `has_paths` gate *is* serialised and *is* in `PartialEq`, so gating on it
stays pure.

This is the **first lossy plane**: the `f32` is quantised to 1/255, floored at 1 so an unlikely path
never reads as unreachable (`0` means exactly "no path"). `encode_raw` stores the quantised byte and
`encode` divides by 255, so the raw/normalised identity still holds bit-for-bit — the loss is at the
input, not between the two views.

## The encode cost, and where it is paid

`encode` now runs the pathfinder. Measured (release, 2,000 iterations, 16x9 board, 5 players a
side):

```
no paths offered     1.8 µs/encode
paths offered       77.1 µs/encode
```

**`prepare` pays it once per sample and it does not matter.** 99,059 samples × 77 µs ≈ 7.6 s against
a job whose JSONL parse alone is ~27 s.

**Inference pays it per NN evaluation, and the memo does not protect it.** This is the part that is
easy to get wrong by reading:

```rust
pub fn priors(&self, state: &GameState, actions: &[EngineAction]) -> Vec<f32> {
    let enc = encode(state);                          // pathfinding happens here
    let (policy, _v) = self.forward_memo(&enc, true); // memo checked after
```

`forward_memo` is keyed by **comparing the encoded tensors** (`c.spatial != enc.spatial`), so the
encode must run just to build the memo key. The memo saves the *forward*, never the *encode*.

Worse, the search encodes each state **twice per node**: `score_leaf` calls
`value_home_i64_prefetch_policy` (encode #1), then `available_actions` calls `priors` (encode #2).
Per the doc comment in `eval.rs`, a 300-iteration `Evaluator::Nn` search hits those call sites 482
times; on tract the memo collapses that to 242 *forwards*, but it is still 482 *encodes*.

Derived from the measured 77.1 µs — **arithmetic, not an end-to-end measurement**:

| | per 300-iteration search |
|---|---|
| before | 482 × 1.8 µs ≈ 0.9 ms |
| now | 482 × 77.1 µs ≈ 37 ms |

Linear in budget. Against the remote sidecar's ~0.91 ms/forward × 482 ≈ 440 ms that is ~8% on the NN
path; against tract it is a smaller share, since tract's forwards are slower.

## Open items

### A. Cache the `Encoded`, not just the forward result — no correctness question

`priors` re-encodes a state that `value_home_i64_prefetch_policy` encoded microseconds earlier, on
the same thread. `LAST_FORWARD` is **already** a thread-local that assumes exactly that locality; it
just caches the forward *result* instead of the encoding that keyed it. Caching the `Encoded`
alongside it halves the new cost outright (~37 ms → ~18 ms per 300-iteration search) and, as a
bonus, replaces the 8,496-element `Vec<f32>` comparison on every memo probe with a cheap identity
check. No new assumption.

### B. Reuse the cached `path_buffer` when present — needs a proof first

The engine has already computed the paths at inference time; only the corpus path cannot see them.
Using the cache when `get_paths()` returns `Some` and recomputing otherwise would remove the
pathfinding bill entirely at inference.

**Do not add this casually.** `take_path` drains the buffer as a move is committed, so there may be
decision points where the cached buffer is a strict subset of a fresh recompute. If there are, this
reintroduces exactly the train/inference skew the recompute was adopted to avoid — and it would be
silent. The prerequisite is a test over many real search states asserting cache ≡ recompute at every
decision point with `has_paths`, not the single-fixture check that exists today. A `debug_assert`
comparing the two on every encode would surface a divergence in the test suite cheaply.

### C. End-to-end timing

The 37 ms above is arithmetic from a microbenchmark. A fixed-budget `Evaluator::Nn` search timed
before/after would confirm it. Worth doing before deciding whether B is needed at all.

### D. `lazy_mover_goldens_full.txt` is stale

Its NN arms were blessed at C=103 and the encoder has moved three times since. The `#[ignore]`d full
matrix will fail until re-blessed (~14 min); the default-on heuristic arm is unaffected and passes,
since it never touches the encoder. Deferred deliberately while the schema is still moving — flagged
in the test's own module docs.

## Consequences to remember

- **Every checkpoint in `models/` is unloadable** (wrong stem shape), and `data/prepared/` needs a
  fresh `prepare` run before the next training job.
- Adding a channel means: `encode.rs` (constant, name, write, scale if not a flag) →
  `NN_SCHEMA_VERSION` in `prepare.rs` → `SPATIAL_CHANNELS` in `train/src/bbnn/model.py` → both cheap
  fixture regens (`uv run python -m bbnn.fixture`, and `BLOOD_WRITE_GOLDEN=1` per
  `tests/capacity_parity.rs`). `name_lengths_match_channel_counts` fails until the pinned literal is
  updated — that is the intended trip-wire.
