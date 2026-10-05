# CLAUDE.md — botbowl-ui

The terminal frontend and the headless single-process shells: `dataset` / `eval` over
`botbowl-play`, plus read-only probes (`convergence`, `positions`, `override-audit`). Subcommands
are declared in `src/cli.rs` and dispatched in `src/main.rs`; each lives in its own module.

## `override-audit` (plan 055 §3 phase 2, `src/override_audit.rs`)

Does the search's overruling of its own policy win real drives? For sampled corpus decisions it
re-runs the configured search fresh, compares its move `a_s` with the policy's argmax `a_p`, and for
overrides (plus a `--control-frac` share of agreements, flagged, with `keep_prob` per row) plays the
drive out `--playouts` times after each move with policy-only on both sides, paired dice. Rows
carry MC mean/SE per move, the paired difference and its SE, the search's Q/visits, priors, the
net's V after each move and V(s), the root value, decision kind, fan and phase.
`scripts/override_audit_summary.py` turns them into the override ledger and the H1/H2/H5 tables.

```sh
BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9 \
cargo run --release -p botbowl-ui -- override-audit \
    --corpus runs/loopmix16x9g/gen06/shard4.jsonl runs/loopmix16x9g/gen06/shard7.jsonl \
    --model models/az_v7/bbnet_mix16x9g_gen05.onnx --nn-server /tmp/bbnn.sock \
    --search-config cfgs/gumbel16_f1000.toml --search-iters 1000 \
    --decisions 1000 --playouts 64 --parallel 8 --out audit.jsonl   # [--board 14x7,16x9] [--all]
scripts/override_audit_summary.py audit.jsonl
```

Things that are easy to get wrong:

- **Corpus states are replayed, never used as read.** A deserialised `GameState` has no path
  buffer (`#[serde(skip)]`), so at a mid-activation decision every path action is missing from
  `get_all_actions()` and stepping one panics. `replay_to` regenerates the trajectory's start with
  `drives::position_state` from `meta.seed` and the placement in `meta.extra`, replays the recorded
  moves under the state's own seeded dice, and checks every replayed state against the recorded
  one; a divergent trajectory is skipped with a message. The binary's capacity must match the
  corpus's (`meta.board_capacity`), which `index_corpus` checks up front.
- **`PolicyBot` is `cfgs/policy_only.toml` without the search:** argmax of `NnEvaluator::priors`
  over the search root's legal set (`should_prune`, with its empty-set fallback), first index on a
  tie, one forward per decision. `the_policy_bot_plays_what_the_policy_only_preset_plays` pins it
  against the preset along whole drives. Keep the two in step if pruning or the Gumbel root changes.
- **The drive ends on `drives::DriveStart`**, the same rule as the corpus generator and the drive
  benchmark, and scores `+1 / -1 / 0` in the mover's frame: the value target's units.
- **`policy.mc_*` is also MC(s)** (policy-only from `s` plays `a_p`), which is what the H1
  calibration uses. A control row plays only `a_p`; its `search` arm is a copy.
- **The search is fresh**, with no tree inherited from earlier decisions of the turn, unlike real
  play. Use a deterministic preset (`gumbel_scale = 0`); a noisy one is warned about.
