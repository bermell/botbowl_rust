# Plan 047 — Per-player kickoff setup: search it, learn it

**Status:** Built 2026-09-28, not yet run. Engine, NN schema v8, MCTS, generation, web UI and
the training loop are wired; the experiments below are what comes next. Branch
`worktree-per-player-setup`.

## Why

The setup used to be four whole-team formation actions (`SetupLine | Spread | Wedge | Zone`) and
an `EndSetup`. Neither a human nor the bot could customise a setup to the team it actually had,
the four actions shared one policy channel so a net could not even prefer one, and
`is_setup_legal` was never enforced during play. Plan 034 recorded "manual setup is not possible"
as an engine limitation.

The design (the user's, 2026-09-27): one decision per player, so a setup is a tree of depth
"players available" that the existing spatial policy head can drive, with no bench
representation and no new architecture.

## What was built

### Engine — one `PlacePlayer` / `BenchPlayer` decision per player

`Setup` (`botbowl-engine/src/core/procedures/kickoff_procs.rs`):

- **Staging.** Every reserve is fielded onto the team's own endzone column (spilling forward if
  the roster outgrows it), in placement order, flagged `used = true` until placed. That
  deliberately puts more players on the pitch than allowed and never reaches a kickoff that way.
  The net sees who is still to come through the existing per-player planes plus `used`, and whom
  it is placing through `info.active_player`. No new input plane.
- **Fixed queue**: role rank (L, B, C, T) then id. No "whom next" choice, so no order permutation
  to recombine — and with first-free-slot ids two orders of the same placements would never have
  recombined anyway.
- **The mask makes the setup legal.** Own half only; a *placed* teammate's square is off, a
  *waiting* teammate's square swaps; a wing closes at its cap (the LOS column in a wing row is a
  wing square — the bands partition the rows, the tabletop "counts as both" rule is deliberately
  not modelled); when the line still needs as many players as there are placements left, only LOS
  squares; `BenchPlayer` only while `min(team_size, available)` and the line can still be met
  without the player (one bench per drive on the stock roster, none once injuries thin it). Setup
  ends by itself at `team_size` placed (the rest benched) or an empty queue; no `EndSetup`.
  `is_setup_legal` is `debug_assert`ed at the end and
  `any_sequence_of_offered_actions_ends_in_a_legal_setup` walks random offered actions on every
  test board down to 10x5/2.
- **Formations are planners now.** `Formation::plan` / `next_action` answer the prompts one at a
  time from a plan that is stable across placements; `auto_setup` plays one out. Used by
  `ScriptedBot` (Line), `GameStateBuilder`, the MCTS opponent model and the web UI's auto button.
  `test_setup_preconfigured_formations` still pins the exact Line squares by role.
- `GameOver` no longer offers the vanished `EndSetup` (only `DontUseReroll`).

### NN — schema v8

`PosAT::PlacePlayer` at channel 14, the simple block shifted to 15.., `SimpleAT::BenchPlayer` at
29, the five formation actions gone. Same shapes as v7 (C 61, F 18, A 30), so a checkpoint now
carries a `schema_version` buffer (`train/src/bbnn/model.py`); `bbnn.migrate` v7 → v8 permutes
the policy head (`POLICY_MAP_7_TO_8`, pinned from the Rust side by
`v8_channel_layout_matches_the_migration`), retained channels keep their weights, the two new
channels start at zero, and the verifier compares mapped channels. `BBNet.from_state_dict`
refuses an unmarked checkpoint with a pointer to `migrate`.

### MCTS

- `HorizonAnchor` carries the half: `Half` zeroes the turn counters at half-time, so a turn-8 root
  used to search through the half-time setups until a score.
- `MctsConfig`: `setup` (`auto | search | formation`; auto = search with a net, formation
  otherwise), `opponent_setup` (auto = formation), `setup_formation` (`line | … | random`, drawn
  per drive from the bot's seeded RNG), `setup_budget_scale`, `setup_horizon_turns`. Env spellings
  `BLOOD_MCTS_SETUP*`; preset keys in `cfgs/README.md`.
- **Opponent model in the tree**: `BloodBowlDynamics::opponent_setup_pick` answers the other
  team's placements with `Formation::Line` in the quiescent loop, so a kicker's search reaches the
  kickoff instead of stopping at half-built opposing setups the value net has never seen. Pure in
  state for a fixed config.
- **Unsearched setups.** Under `formation`, or when a setup root has a single legal action (the
  three-player boards put everyone on the line), the bot answers without a search and
  `get_action_with_record` emits a *scripted* sample: every legal action as a child, one visit on
  the one played, `Sample::scripted = true`. `targets.rs` makes it one-hot whatever the target kind
  and `prepare` keeps it despite `root_visits = 1`. That is the gen-0 teacher shard
  (`cfgs/setup-teacher.toml`: `setup = "formation"`, `setup_formation = "random"`).

### Generation — `--next-drive`

`random-start` still writes the drive-bounded record it always did. With `--next-drive`
(`botbowl-ui dataset`, `botbowl-hub job generate`, `NEXT_DRIVE=1` in `train_loop.sh`, on by
default there), a drive that *scores* is followed through the kickoff that comes next — both
per-player setups, the kick, the turns — until that drive resolves, and the second drive is its
own record (`meta.extra.drive = 2`, `start = setup`, its own `start_*` and telemetry). Two
records, not one, so `prepare`'s per-drive weight, `td_rate.py` and the hub's TD count keep their
meaning, and the setup samples are labelled by the drive they set up. The hub accepts several
JSON lines per game (`PROTOCOL_VERSION` 7). No fraction knob: setups arrive at the corpus TD
rate, which rises as the bot improves, and the roster they set up with is whatever the first
drive left.

### Web

`SetupView` in the view (team, placed, waiting, team size, fitting formations), `PlacePlayer`
squares and the `BenchPlayer` button as ordinary actions, one "auto" button per formation
(`ClientMsg::AutoSetup`, applied as one undoable decision). Humans place in the fixed queue
order; a reorder action could be added later and pruned out of the search by a pure rule.

## What to run next

1. **Migrate the champion**: `python -m bbnn.migrate champ.pt --out champ_v8.pt --onnx champ_v8.onnx`.
   The migrated net plays exactly as before on every retained action and starts with uniform
   priors over placements.
2. **Gen-0 teacher shard** at the small end of the size curriculum (`SIZE_MODE=centred`, centre on
   the 3–4-player boards where the setup is nearly forced): `--next-drive --bot-config
   cfgs/setup-teacher.toml`. Every scored drive contributes one setup per team as one-hot
   samples, and the corpus gets post-kickoff states for the value net.
3. **Search setups from gen 1** (`setup = "auto"` with the net). Watch, per generation:
   `iterations / searches` at setup roots (the `setup_budget_scale` knob), the corpus share of
   `drive = 2` records, and whether the receiver's post-kickoff positions improve.
4. **Diagnostic to add**: a histogram of the procedure a setup-rooted search's leaves end in
   (`LeafStats` has the machinery). Leaves inside `Setup` mean the search is not getting through;
   leaves in the kickoff or the first turn mean it is.
5. **Open knobs**: `setup_horizon_turns = 2` for the receiver (searches its own first turn instead
   of trusting the value net at the kickoff); `opponent_setup = search` once the net's own setups
   beat the line; a heuristic prior blend for placements if gen-1 setups still look random.

## Caveats

- Corpora before this change hold no setup samples and no post-kickoff states; the first
  generations of setup search lean on the teacher shard and the `--next-drive` records.
- The ladder's scripted and heuristic rungs keep formation setups (`setup = auto` resolves to
  `formation` for them), so eval rows stay comparable; only net-driven bots search setups.
- `test_canvas_serves_every_board_its_own_answer_from_shared_batches` in `train/tests` fails
  under this worktree's long path (`AF_UNIX path too long`), unrelated to the change.
