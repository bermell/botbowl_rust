# botbowl-play

"Play one game, return its record." Extracted from `botbowl-ui`'s `dataset.rs`/`eval.rs` in
plan 040 phase 0 so the single-box CLI and the distributed worker run the *same* code path.

## Contract

- **No process concerns.** Nothing here opens files, spawns threads, prints progress or parses
  flags. A caller hands in a config + seed and gets back a value. `botbowl-ui` is the
  single-process shell (parallel workers, JSONL writer, `NN_PROFILE`/`NN_SERVER` lines the loop
  greps); the plan-040 worker is the other.
- **Byte-compatible records.** `eval::EvalGameLine`'s field order *is* the `--per-game-out` file
  format read by `scripts/paired_summary.py` and friends; `generate::budget_label` is the
  provenance string stamped into every corpus. Both are pinned by tests — change them only with
  a schema bump.
- **`SearchConfig` knobs are `Option`: `None` means "leave `MctsBot`'s (env-driven) default".**
  `dataset` has always left them unset; `eval` has always set all four. Keep that split or the
  two paths' bots silently diverge from their pre-extraction behaviour.
- **Fold logic lives next to the record.** `LadderRow::record` is how a rung's per-game lines
  become the report row, in any order from any number of producers. The hub rebuilds
  `report.json` from workers' lines with exactly this.
- **Clap-free.** CLI enums live in `botbowl-ui/src/cli.rs` with `From` impls onto the types here.

## Verifying a change is behaviour-neutral

`cargo test -p botbowl-play` for the record-format pins, then at the 14x7 tier
(`BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4`, separate `CARGO_TARGET_DIR`)
`cargo test -p botbowl-ui --test parallel_games --test parallel_rungs`. Search output is not
reproducible across processes (recon_mcts HashMap order), so compare *metadata* between binaries
for MCTS runs and full output only for search-free ones
(`eval --candidate-bot scripted --rungs random,scripted`).
