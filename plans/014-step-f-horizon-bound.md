# Plan 014 — Step F: horizon-bounded MCTS

## Context

Plan 013 showed that the post-refactor MCTS DAG grows deep enough (max-depth ~54 on `ScoreTdEasy` at 1000 iters) that
the `Arc<Node>` drop chain and `Node::get_state` recursion overflow the default 2 MB worker stack. Plan 012 already
flagged Step F as a structural fix: bound the MCTS horizon at the boundary the leaf-scoring heuristic already treats as
terminal — the start of the bot's _next_ turn, i.e. after the bot and the opponent have each played one turn.

## Change

`botbowl-mcts/src/dynamics.rs`:

- New `HorizonAnchor { agent_team, home_turn, away_turn, home_score, away_score }` captured from the root state in
  `MctsBot::get_action`.
- `BloodBowlDynamics::horizon: Option<HorizonAnchor>` plumbs the anchor down to `available_actions`, which returns
  `None` (terminal) once `anchor.diverged(state)` is true. Divergence triggers when:
  - `state.info.game_over`,
  - either team's `score` has changed from the anchor,
  - or the agent's turn counter has incremented (the agent's _next_ turn has begun — meaning the bot played, the
    opponent played, and we're back to the bot's clock).
- `Default for BloodBowlDynamics` keeps `horizon: None`, so callers that build `Tree` directly (microbenches,
  `expand_bench`'s `CountingDynamics` wrapper) see the unbounded form unchanged.
- `BLOOD_MCTS_HORIZON=off` disables horizon in `MctsBot::get_action` for A/B comparison against the unbounded baseline.

The horizon is captured **once per `get_action` call** and held constant for the whole search, so it remains a pure
function of `(state, anchor)` — recombination invariants stay intact (`CLAUDE.md` requirement).

## Results

### `ScoreTdEasy` — fixed and improved

| Build                  | StoreState success rate |  Wall | Status         |
| ---------------------- | ----------------------: | ----: | -------------- |
| pre-Step F (plan 013)  |                     n/a |  15 s | STACK OVERFLOW |
| **Step F (this plan)** |              **0.9600** | 105 s | **PASS**       |

Single-call DAG-shape probe on the same seed, `StoreState`, 1000 iters:

| Build      |   reg_len | max_depth |      reuse |       wall |
| ---------- | --------: | --------: | ---------: | ---------: |
| no horizon |     4 123 |        54 |     0.6692 |     678 ms |
| **Step F** | **3 364** |    **19** | **0.7102** | **532 ms** |

Tree depth dropped from 54 to 19 (≈ 2.8×); per-call wall dropped by ~20 % even at the same iter count. At 10 000 iters
the max-depth still caps at 19 — the bound is the natural turn boundary, not a budget cap.

### `GetTheBallEasy` — known pre-existing regression

GTB completes with horizon (no stack overflow) but the success rate is **0.1200** (well below the 0.70 threshold the
test asserts on).

`BLOOD_MCTS_HORIZON=off` on the same binary still times out at 300 s, so the low rate is **not** caused by Step F — it's
the same broken trajectory selection plan 012 flagged. With horizon enabled the test at least _runs_ and exposes the
underlying mis-selection. The first-move choice on the lecture's canonical seed (`0xF00D_9012`) is `StartBlitz` or
`StartHandoff`, neither of which advances the ball acquisition task. Single-call depth on GTB caps at 14, so the bot has
tree to work with — the issue is upstream (priors / pruning / leaf-score tuning), and needs its own investigation. Out
of scope for this plan.

### `HashOnly` — still corrupted, as expected

Horizon doesn't change `HashOnly`'s collision-driven false-merge behaviour. Re-run of `score_td_easy/HashOnly` panics in
the engine at `is_legal_action` / `pending_roll.is_none()` assertions just like plan 013 reported. The next task (#4 in
the queue) retires `HashOnly` as the default.

## Verification

```sh
cd botbowl_rust
cargo test --release -p botbowl-mcts --no-run
STE=$(ls -t target/release/deps/score_td_easy-* | grep -v '\.d$' | head -1)
BLOOD_MCTS_MEMORY=store $STE --ignored --nocapture --test-threads=1
# → rate=0.96, completes in ~105 s

TS=$(ls -t target/release/deps/tree_shape-* | grep -v '\.d$' | head -1)
BLOOD_MCTS_MEMORY=store $TS --ignored --nocapture iters_10000_single \
    --test-threads=1
# → max_depth=19, ~1.4 s
```

## Follow-ups

- **GetTheBallEasy regression** — separate plan; the 0.12 rate on a test that used to pass at ~1.00 needs root-causing.
  Likely in priors or pruning; horizon merely surfaces it.
- **Switch default off `HashOnly`** — task #4. Now that `StoreState` is viable, `MctsBot::new` should default to it.
- **Step A** (set_min_depth pre-check) — likely still worthwhile but now perf-only; the hang it targets doesn't reappear
  under horizon-bounded `StoreState`. Profile first.
