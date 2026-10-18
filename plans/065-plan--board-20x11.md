# Plan 065 — the loop on 20x11/7

**Status:** ready to switch (2026-10-18). Code, tests, launcher and procedure are in this commit;
the loop still runs at 16x9/6 until the parent session switches it at a generation boundary (§7).
Registry: [exp076](../registry/EXPERIMENTS.md#exp076) (planned).

**The decision (the user, 2026-10-18):** the loop moves from boards up to 16x9/6 to a 20x11/7
capacity (playable 20x11, 7 players a side) — not the full 26x15/11 yet. Evaluation, 16x9-only
today, becomes 20x11 with 16x9 kept as a second set. Bigger boards are the direction.

This is plan 042's E4 ("the 20x11 tier"), minus E2's A/B/C: the mixed-size loop has run since
2026-09-23 and its net plays every board up to 16x9; E0 found that transfer *upward* in board size
is nearly free (−2.0 pp from 14x7 to 16x9 vs scripted), which is the bet this switch makes again.

## 1. Build and test at 20x11/7

`BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 CARGO_TARGET_DIR=target/20x11 cargo test --workspace --no-fail-fast`:

| capacity | tree | passed | failed | ignored |
|---|---|---|---|---|
| 20x11/7 | master `c58803f` | 788 | 17 | 44 |
| 20x11/7 | this branch | 794 | 17 | 44 |
| 16x9/6 (today's loop) | master `c58803f` | 754 | 51 | 44 |
| 16x9/6 | this branch | 760 | 51 | 44 |

**No new failure at either capacity**: each branch run fails exactly the tests master fails at that
capacity (the +6 passes are the new tests). At 20x11 the 17 are hard-coded-coordinate or
board-shape assumptions, none a rule bug; 15 of them also fail at 16x9 (51 there, the family the
reviewers counted as 43), and two fail only at 20x11:
- engine (7): `ma_cap_always_falls_short_of_the_endzone`, `scatter_divisor_is_a_noop_on_full_pitch_and_shrinks_narrow_boards`,
  `crowd_surf_ball_carrier`, `knockdown_both_and_bounce_ball_around`, `throw_in_onto_catcher_bounces_from_catcher_square`,
  `placing_onto_a_waiting_teammates_square_swaps_them`, `pass_wildly_inaccurate_out_of_bounds`;
- curriculum (5): `get_the_ball` easy/medium/hard setups and `runner_caps` ×2 — `FullPitch index out of bounds: (25, 8)`,
  lectures written for the 26x15 pitch;
- mcts (1): `chance_roll_models::kickoff_bounce_touchbacks_are_one_child` — it searches for a kick that lands next
  to the kicking half; at 20x11 the deviate roll is halved (`scatter_divisor` 2) and no kick reaches;
- play (1): `cap_ma_to_board_shrinks_ma_on_a_narrow_board_and_is_a_noop_on_the_full_pitch` — treats the capacity as
  "the full pitch";
- ui (3): `snapshots` seed_42 step 0/50/500 — rendered at the capacity's size.

The two 20x11-only ones, checked because both touch the kickoff: `kickoff_bounce_touchbacks_are_one_child`
is the deviate-scaling case above; `placing_onto_a_waiting_teammates_square_swaps_them` assumes the next
waiting player can always "stay put", but on the 22x13 capacity board the wings are two rows with a cap
of one player (`max_players_per_wing` = (rows + 1) / 2), the first placement filled the north wing,
and the next player's staged square is in it. The mask is right; the test's board is the wrong one.

### What 7 players switch on

`BoardDims::kickoff_table_enabled` is `team_size >= 7`. No board the loop has played had 7, so every
kickoff so far was the degenerate "the ball just lands". At 20x11/7 capacity the density rule
(`round(area / 26)`, capped at the capacity) gives 7 players to 18x10, 20x9, 18x11, 20x10 and 20x11
— half the planned mix (§3). Everything else that depends on the team size was already exercised:

- **The kickoff table** (`KickoffTable`, 2D6). Implemented: 2 Get the Ref (a bribe each; bribes are
  never used), 3 Time-out (both turn markers move), 8 Changing Weather (new weather roll; on a nice
  day a gust scatters the ball once more). **Every other result is a no-op** (Solid Defence, High
  Kick, Cheering Fans, Brilliant Coaching, Quick Snap, Blitz, Officious Ref, Pitch Invasion) — a
  rules gap, not a bug: the game at 7 differs from 6 only by Time-out and weather.
- **Weather** becomes reachable for the first time (only the table changes it). Rain (−1 catch and
  pick-up) and Blizzard (GFI) are read by the pathfinder, so the `path_prob` plane carries them;
  **the encoder has no weather input** otherwise. Non-nice weather follows ~2.3% of kickoffs
  (P(8) × P(2, 3, 11, 12) = 5/36 × 6/36) and lasts the rest of the drive record. Small; noted in §8.
- **Roster** 8 per team (`team_size + 1`): 5 linemen + one blitzer, catcher, thrower — the same
  positional split as 6 players; one bench per drive in the per-player setup.
- **Setup** with 7: wings capped at 2, LOS at least 3, as on every board; the four 7-player boards
  other than the capacity now run in `any_sequence_of_offered_actions_ends_in_a_legal_setup`.

### Bug found and fixed (test first)

**Time-out read Home's turn marker, not the kicking team's** (`kickoff_procs.rs`, `Sum2D6::Three`).
BB2020: if the *kicking* team's marker is on 6, 7 or 8, both markers move back; otherwise forward.
When Away kicked on its turn 6 with Home's marker on 5 (Away received first in the half and scored
on its turn 6), the engine moved both markers *forward* — the half lost a turn instead of gaining
one. Reproduction: `botbowl-engine/tests/kickoff_table.rs::time_out_reads_the_kicking_teams_marker_not_homes`
(red on master: `(6, 7)` where `(4, 5)` is right). Rare: it needs that exact marker split at a
kickoff and a 3 on the table (1/18).

New tests: `tests/kickoff_table.rs` plays 30 full games under `RegisterRolls` on the five 7-player
boards, forcing the table through all 11 results and the weather roll through all 11, checking
Time-out's rule at every occurrence and that no turn counter passes 8.

## 2. Bug harvest

All at 20x11/7 capacity, `RUST_BACKTRACE=1`, `--mode random-start --next-drive`, the generation
preset `cfgs/chance_all_partial_gen.toml`:

| batch | evaluator | board | games | budget | result |
|---|---|---|---|---|---|
| H1 | heuristic | 20x11/7 | 300 (417 drive records) | 200 | no panic; roll census: 117 kickoff-table rolls, 19 weather rolls, 0 replay divergences |
| H2 | gen36 64x6, tract | 16x9/6 | 12 | 1000 | no panic (throughput reference) |
| H3 | gen36, tract | 20x11/7 | 12 | 1000 | no panic |
| H4 | gen36, tract | the old mix (centre 144) | 36 | 1000 | no panic |
| H5 | gen36, tract | the new mix (centre 220; 16 of its 22 boards drawn) | 36 (66 records) | 1000 | no panic |
| H6 | gen36 via `nn_server.py --device cpu --canvas 13x22` (priors and values remote) | the new mix | 12 (1305 samples) | 250 | no panic; 154,196 forwards served, `fell_back_to_tract=0`, canary handshake passed |
| H7 | heuristic, **debug build** (debug assertions) | the new mix (15 boards drawn) | 60 (84 records, 4308 decisions) | 100 | no panic, no debug assertion; 13 kickoff-table and 3 weather rolls, 0 replay divergences |
| H7c | the same, control | today's mix (centre 144) | 60 (3983 decisions) | 100 | no panic |

**No panic and no invariant violation anywhere.** The one engine bug of the step is the Time-out
rule above, found by reading the table's code against the rules when writing its test, not by a
crash.

One debug-only diagnostic grows with the board: **"pruning emptied the action list … falling back
to N unfiltered actions"** (`botbowl-mcts/src/dynamics.rs`, the designed safety net of
`pruning::search_actions`). 913 times over H7's 4308 decisions against 251 over H7c's 3983 — 3.4× as
often per decision at 20x11 (by activation: Blitz 374, Handoff 260, Pass 243, Move 27, Foul 9). It
fires inside the search when an activated player has nowhere useful to go: a Blitz with no square
next to a standing opponent reachable in one path and its `EndPlayerTurn` pruned (no move made yet),
a Pass/Handoff by a non-carrier who cannot reach the ball. The upstream rules (`StartBlitz` needs a
standing opponent *anywhere*, `StartPass` a teammate anywhere) were written for 14x7, where nearly
everything is in reach; on a 20-long board they let through activations that can only waste
budget. Not a correctness problem — the fallback searches the raw list — but a cheap search
improvement for the bigger boards: prune `StartBlitz` when no opponent-adjacent square is in the
player's reach this turn (pure in `(state, action)`, so recombination-safe). Not done here: it
changes the search on every board, so it is its own drive-judged experiment.

**The net at the new sizes.** The encoder and the policy gather run at every board of the mix
(tract builds one runnable per `(H, W)`; H3–H5). The sidecar's masked canvas forward is exact on
every board of the 20x11 grid on the 13x22 canvas (`train/tests/test_model.py`, now parametrised over
both canvases, ≤ 1e-5), and a live CPU sidecar on the 13x22 canvas agrees with tract to the value's
reported precision (1e-3) on 289 states of H5 across 16 boards (`value-bench` both ways); H6 ran the
priors through it. (`value-bench` skips `--next-drive` follow-on records — "replay diverged at sample
0" — as it always has; the frozen benches hold drive-1 states only.)

## 3. The size mix

The same knobs as today, re-centred: `--size-centre 220 --size-temperature 0.3 --size-floor 0.2
--size-aspect 1.5-2.8 --size-max-area 220 --size-min-area 70 --cells-per-player 26` at 20x11/7
capacity. Resolved by the Rust sampler (`dataset` and `job generate` print this table at the start):

| board | team | area | share | | board | team | area | share |
|---|---|---|---|---|---|---|---|---|
| 20x11 | 7 | 220 | 11.2% | | 16x8 | 5 | 128 | 2.9% |
| 20x10 | 7 | 200 | 10.7% | | 14x9 | 5 | 126 | 2.7% |
| 18x11 | 7 | 198 | 10.6% | | 18x7 | 5 | 126 | 2.7% |
| 18x10 | 7 | 180 | 9.2% | | 14x8 | 4 | 112 | 1.7% |
| 20x9 | 7 | 180 | 9.2% | | 16x7 | 4 | 112 | 1.7% |
| 18x9 | 6 | 162 | 7.0% | | 14x7 | 4 | 98 | 1.2% |
| 16x10 | 6 | 160 | 6.8% | | 12x8 | 4 | 96 | 1.1% |
| 20x8 | 6 | 160 | 6.8% | | 16x6 | 4 | 96 | 1.1% |
| 16x9 | 6 | 144 | 4.7% | | 12x7 | 3 | 84 | 1.0% |
| 18x8 | 6 | 144 | 4.7% | | 14x6 | 3 | 84 | 1.0% |
| | | | | | 12x6 | 3 | 72 | 0.9% |
| | | | | | 14x5 | 3 | 70 | 0.9% |

By team size: 7 players 50.9% (the kickoff table), 6 30.0%, 5 8.4%, 4 6.9%, 3 3.8%. Area > 144:
71.5%; mean area 168 (today 113). 22 boards. **20x11 gets 7 players**: the density rule asks for
`round(220 / 26) = 8` and the capacity caps it at 7 (31 cells a player, plan 039's figure for the
tier). Every board of today's grid keeps its team size (no area ≤ 144 rounds above 6), so the old
corpora and the new ones agree on what a 16x9 game is.

Why these knobs: they are today's, re-centred, so the shape of the curriculum is unchanged and the
floor keeps 16x9-sized and smaller boards at 20% of the mass (16x9/6 itself drops from 15.0% to 4.7%,
the team-6 boards together hold 30%). `--size-floor 0.25` (16x9 still 4.7%, 7 players 49%) and
`--size-temperature 0.25` (7 players 58%) were the alternatives; neither changes the cost much.
`size_curriculum.py` keeps running and is capped at 220, so the centre stays put. Note that at this
capacity even today's settings (max area 144) would draw 18x8/6 and 18x7/5, which the 16x9 build
could not.

## 4. Evaluation at 20x11

### Drives: a 20x11 contested set next to the 16x9 one

The 16x9 and 14x7 sets regenerate **identically** in a 20x11-capacity binary: a scripted-vs-scripted
drive rung over both sets gives the same 120 per-game lines at 16x9 and at 20x11 capacity. So the
16x9 curve continues across the switch, same set, same reference (`bbnet_mix16x9d1k_gen04_v9`).

The 20x11 set is built the way plan 051 built the others: candidates, a reference self-play drive
rung over them (4 playouts each), `positions_screen.py` keeping attacker TD rates in 0.25–0.75.
**On CPU it does not fit:** a 20x11 drive at 1000 descents on tract costs ~4 core-minutes
(12 games, 20 drive records, took 25 min on 3 cores), so 2000 screening drives would be ~5
core-days (42 h on this session's 3 cores). It runs on the GPU through the 20x11 hub after the switch, as one more eval job alongside generation:

```sh
# 1. candidates (seconds, deterministic; the same seeds at any commit with this position_state):
BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 target/20x11/release/botbowl-ui positions \
    --board 20x11/7 --count 300 --seed-base 72000000 --min-turns-left 4 --name cand_20x11 \
    --out runs/loopmix16x9v9/positions/cand_20x11.json
# 2. the screen: the champion at the switch against itself, the eval preset, 4 playouts a position,
#    on the loop's hub (its sidecar already serves the champion, so no second model is loaded):
REF=$(cat runs/loopmix16x9v9/champion.txt)
mkdir -p runs/loopmix16x9v9/positions/screen20
BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 target/20x11/release/botbowl-hub job eval \
    --hub http://127.0.0.1:13337 --token-file ~/.config/botbowl/hub.token --label "plan065 screen 20x11" \
    --evaluator nn --model "$REF" --vs-evaluator nn --vs-model "$REF" \
    --bot-config cfgs/chance_all_partial.toml --vs-config cfgs/chance_all_partial.toml --mcts-iters 1000 \
    --positions runs/loopmix16x9v9/positions/cand_20x11.json --vs-games 1200 --skip-fixed-rungs --seed 0 \
    --per-game-out runs/loopmix16x9v9/positions/screen20/eval.games.jsonl \
    --out runs/loopmix16x9v9/positions/screen20/report.json --wait \
    > runs/loopmix16x9v9/positions/screen20/eval.log 2>&1
# 3. keep the contested ones; name the set after its reference generation:
train/.venv/bin/python scripts/positions_screen.py --set runs/loopmix16x9v9/positions/cand_20x11.json \
    --games runs/loopmix16x9v9/positions/screen20/eval.games.jsonl --reference "$REF" --band 0.25:0.75 \
    --name contested_20x11_genNN --out runs/loopmix16x9v9/positions/contested_20x11_genNN.json
```

300 candidates × 4 playouts = 1200 drives — about 45% of today's generate phase in GPU forwards
(1200 × ~31k forwards on the bigger canvas, against 3200 × ~26k); the 16x9 screen kept 237 of 500 (47%), so expect ~140 contested positions, enough for `DRIVE_CAP=800`
(positions repeat with new dice past 400 pairs). The pipeline was smoke-tested on CPU at 50 descents
(8 candidates × 4 playouts at 50 descents, gen36 on tract → `positions_screen.py` kept 3; the
screened set then played as a drive rung next to the 16x9 set in one `eval`). `launch_plan065.sh`
picks up the newest `contested_20x11_*.json` in `runs/loopmix16x9v9/positions/` at launch; until one exists it runs drives on 16x9 only and says so in
`launch.log`. Relaunch once at the boundary before the first drive generation after the set exists
(§7). Using the champion of the switch as the screen reference, not `d1k gen04`, follows plan 051
§Step 5's lesson: screen with the current reference.

**The drive reference on 20x11 is weak.** `DRIVE_REF` (`d1k gen04`, a 16x9-era net) has never been
trained on a board wider than 16. Expect fast H1 on the 20x11 set at first: informative only once
the candidate's own 20x11 play is the question. Consider freezing the switch champion as a second,
20x11 reference after ~3 generations (a one-line change to `DRIVE_REF` per set is not supported
today — the loop has one reference; an `ORIGIN_REF` match is the existing second slot).

### Value benchmark

`runs/value_bench/v9_gen01_val.jsonl` (3023 states, 14x7/16x9, from gen01's held-out shards) keeps
running unchanged: from this commit a 20x11-capacity `value-bench` replays its 16x9-capacity
corpora (verified: identical values on 303 of its states, tract, gen36). It stays the series.

A 20x11 benchmark **is needed**: 51% of the new corpus is on boards the v9 bench has no state from,
and the value head's error there is the thing the switch changes. Build it from the first 20x11
generation's held-out shards (4, 7), exactly as v9's was built, with the generalised script:

```sh
GEN=gen45 BOARDS=20x11,20x10,18x11 BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 \
    scripts/value_bench_build_v9.sh runs/loopmix16x9v9 "$(cat runs/loopmix16x9v9/champion.txt)" \
    runs/value_bench/v9_20x11_val.jsonl
```

(GPU, the generator of gen45 as the MC policy, `override-audit --all --decisions 3000 --playouts 48`;
the three biggest 7-player boards so 3000 states come from ~one generation.) `launch_plan065.sh`
already points `VALUE_BENCH_2` at `runs/value_bench/v9_20x11_val.jsonl`; the loop scores it next to
the v9 bench (`value bench 2:` status line, `genNN/value_bench_2/`) from the first generation after the
file exists, no relaunch needed (the path is checked every generation).

### Net check

`net_check.sh` now takes the capacity from the caller (it hard-coded 16x9 and would have refused
the 20x11 corpus) and `NET_CHECK_BOARDS` (launcher: `16x9,20x11`; today `14x7,16x9`). The series
changes boards at the switch: compare a generation's net check with its neighbours on the same side
of the switch.

## 5. Throughput

Measured on CPU with the same net and binary (gen36 64x6 on tract, 20x11 build, the generation
preset at 1000 descents, `--next-drive`, 3 games at a time), today's mix (centre 144) against the new
one (centre 220), 36 games each, plus 12 games each on 16x9/6 and 20x11/7 alone:

| run | games | drive records | searched decisions / game | samples / game | forwards / game | forwards / decision | MC-label decisions / game | longest record |
|---|---|---|---|---|---|---|---|---|
| 16x9/6 | 12 | 21 | 43.4 | 53.3 | 23,972 | 450 | 1,145 | 77 |
| 20x11/7 | 12 | 20 | 87.1 | 108.3 | 51,563 | 476 | 7,012 | 288 |
| mix, centre 144 | 36 | 65 | 49.2 | 62.6 | 26,200 | 419 | 1,967 | 173 |
| **mix, centre 220** | 36 | 66 | 86.1 | 108.7 | 49,710 | 457 | 5,057 | 164 |

The centre-144 row agrees with the live loop's gen43 corpus (62.9 samples and 50.8 searched decisions
a game, 1,952 MC-label decisions; 96x8 net), so the CPU runs stand in for the loop. "MC-label
decisions" is Σ n(n+1)/2 over a game's records: `mc-label` plays every sample's drive out from that
sample, so its cost is quadratic in the drive length.

**Per game, the new mix costs 1.90× the forwards, yields 1.74× the samples and 2.57× the MC-label
work.** On the GPU sidecar every forward is padded to the canvas, 13x22 instead of 11x18 — 1.44× the
cells, which is the cost ratio once the GPU is arithmetic-bound (it is: plan 058 §9, two sidecars at
48 streams fill it). So per game: **generation 2.7×, MC labels 3.7×**; per sample 1.6× and 2.1×.
36-game samples with heavy-tailed drive lengths: read ±20%.

| `GAMES_PER_SHARD` | generate vs today | MC labels vs today | samples vs today |
|---|---|---|---|
| 400 (today) | 2.74× | 3.71× | 1.74× |
| 250 | 1.71× | 2.32× | 1.09× |
| 200 | 1.37× | 1.86× | 0.87× |
| **150** | **1.03×** | **1.39×** | **0.65×** |
| 125 | 0.86× | 1.16× | 0.54× |

**Recommendation: `GAMES_PER_SHARD=150`** (1200 games, ~130k samples a generation). The generate
phase stays at today's length (96x8 lineage: 230 min without an overlapping eval, 350–390 with one;
the 64x6 lineage would be 160–210), MC labelling grows ~40% (130–240 → 180–330 min) and the train step
stays about where it is (0.65× the samples at ~1.5× the cells each: mean area 168 vs 113): a
generation ~10–15% longer than today in all. The window of 3 generations still holds ~400k samples,
and they are bigger positions. Raise to 200 if a generation can be ~35% longer. The bigger net costs
the same factor at any board (exp074: ~2.1× the 64x6's GPU per sample), so the ratios hold for either
lineage. The launcher sets 150.

CPU side: the search's own (non-inference) CPU per forward rises ~30% (0.22 → 0.29 ms, tract runs);
peak RSS of 3 streams 1.09 GB on 20x11 vs 0.71 GB on 16x9, 1.12 GB on the new mix vs 0.62 GB on the
old one (+80%) — watch the 48-stream worker's `mem_governor` (§8).

**MC labels are the phase that grows most.** Options if it binds, none needed for the switch: fewer
playouts on long records (the label SE falls as 1/√n either way), or label a fixed share of each
long record's samples.

**Drives.** A 20x11 drive record costs ~2.3× the forwards of a 16x9 one (30.9k vs 13.7k, gen36) and
1.44× per forward: ~3.3× the GPU. A 20x11 SPRT of the same length as today's 16x9 one costs ~3× the
whole 16x9 benchmark; with drives every 3 generations, overlapped with generation, that generation's
generate phase stretches (gen40/43 already took 350/620 min against 230). If that hurts, run the 20x11
set at a lower `DRIVE_CAP` — the loop has one cap for all sets today.

## 6. The launcher, and which run

`scripts/launch_plan065.sh`: `launch_plan058.sh` with
- capacity `BUILD_W/H/PLAYERS=20/11/7`, `BOARD_SIZE_*` the same, `CARGO_TARGET_DIR=target/20x11`,
  `NN_CANVAS=13x22` (the loop's default for that capacity, spelled out);
- the size flags of §3, and `size_centre.txt` set to 220 on every launch (the loop only seeds it
  when empty; the centre is capped there and never moves down, so this is idempotent);
- `DRIVE_POSITIONS` = the 16x9 set + the newest 20x11 set if one exists;
- `VALUE_BENCH` unchanged, `VALUE_BENCH_2=runs/value_bench/v9_20x11_val.jsonl` (new in
  `train_loop.sh`: a second benchmark, same treatment, off when unset or missing),
  `NET_CHECK_BOARDS=16x9,20x11`, `EVAL_BOARD_SIZES=16x9,20x11` (full games are off; harmless);
- `GAMES_PER_SHARD` per §5 (env-overridable);
- a refusal to start while anything listens on port 13337: `train_loop.sh` reuses a running hub, and
  a leftover 16x9 hub would refuse every 20x11 worker, the local one included;
- everything else exactly as `launch_plan058.sh`: C1 presets `cfgs/chance_all_partial{,_gen}.toml`,
  `LABEL_VIA_HUB=1`, value weight 0.36, cq τ 100, lr 5e-5, `--freeze-bn`, MC labels ×8,
  `--next-drive`, drives every 3 generations vs `d1k gen04` with SPRT 0.5:0.55 cap 800, 48 local
  streams over 2 sidecars, `MAX_GENS=60`, `TIER_OVERRIDE=mix16x9v9`.

**Continue `runs/loopmix16x9v9`, don't start a new run dir.** The question was whether the
3-generation window can mix 16x9-capacity corpora with 20x11 ones. Before this commit it could not:
a stored `GameState` serialises its capacity-sized arrays (`FullPitch` 18x11 vs 22x13, the roster
arrays 14 vs 16 slots), and a 20x11 `prepare` panicked on the first old line ("invalid length 14,
expected an array of length 16"). Now `FullPitch` and the roster arrays deserialise at any capacity
the board fits (the cells past a board are always empty, so nothing is lost or invented), and the
replay tools check `BoardCapacity::holds(board)` instead of capacity equality. Verified with two
binaries of this commit, 16x9 and 20x11:
- `prepare` on a gen43 sample: **byte-identical** `.npy` and manifests (8 board dims, 250 samples);
- `value-bench` on 303 v9 bench states: identical values;
- `mc-label` (2 playouts) on 6 gen43 trajectories: identical labels (the rewritten states carry
  this binary's padding; `meta.board_capacity` keeps the writer's);
- the 14x7 and 16x9 contested sets: identical scripted drives (§4).

The trainer is already per-board (`MultiDimsDataset`, one `dims_WxH` per board, batches never mix
boards), so the new boards just add groups. Continuing keeps the window full on the first 20x11
generation (gen43 + gen44 at 16x9, gen45 at 20x11: a gentle turnover over three generations), the
drive curve, the champion pointer and the value-bench series. A new run dir would start its first
generation on one generation of data. Cosmetic cost: the models keep the `mix16x9v9` tag.

**Helper workers.** The hub refuses a worker of another capacity *and* of another env board
(`Hello` carries both). The laptop must rebuild at this commit **and run** with the env set:

```sh
BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 CARGO_TARGET_DIR=target/20x11 cargo build --release -p botbowl-worker
BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 target/20x11/release/botbowl-worker --hub ws://<hub>:13337/ws
```

## 7. The switch (for the parent session)

1. **Do not merge into the main checkout while the loop runs.** `train_loop.sh` changed (the
   `VALUE_BENCH_2` block), and bash re-reads a running script by byte offset (memory:
   "Editing a running train_loop"). Merge after the loop has exited.
2. **Stop at the end of a generation's train step.** The loop is in gen44's generate (16x9) since
   2026-10-18 15:02. Place the stop *before* the line, with a pattern that appears long before the
   checkpoint: `nohup scripts/stop_after.sh "gen44 train:" runs/loopmix16x9v9 > /dev/null 2>&1 &`.
   Training takes 40–60 min, so the 60 s poll cannot miss it (exp074's STOP landed seconds after the
   check it aimed at). The next checkpoint is `before gen44 eval`, after the train step, the
   absorption probe and the value bench; gen44 has no drives (`DRIVE_EVAL_EVERY=3`). Wait for
   `STOP file present — exiting cleanly (before gen44 eval)`.
3. Check nothing is left: no `botbowl-hub serve` on 13337 (`ss -ltnp | grep 13337`), no worker or
   `nn_server.py` of the loop. `champion.txt` / `latest_net.txt` name gen44 — the 20x11 loop starts
   from it; nothing to repoint (exp074's swap is the model if a different net should generate).
4. `rm runs/loopmix16x9v9/STOP`, merge this branch into master, commit (clean tree: the launcher
   refuses a dirty one).
5. `nohup scripts/launch_plan065.sh > /dev/null 2>&1 &`. It builds `target/20x11`, starts the 20x11
   hub and worker, writes `size_centre.txt` = 220, and generates gen45 on the 20x11 mix with
   `GAMES_PER_SHARD` per §5. First lines to check in `status.md`: `loop start: … board capacity
   20x11/7, sizes centred (--size-centre 220 …)`, the `board sizes:` table in `gen45/generate.log`,
   `nn_server up … --canvas 13x22` in `nn_server.log`, and no `NN_SERVER_FALLBACK`.
6. During gen45's generate, run the 20x11 screen of §4 on the hub (steps 1–3). When the set exists,
   stop before gen45's eval (`stop_after.sh "gen45 train:"`), relaunch the same way: the launcher
   adds the set, and gen45 is the first drive generation with 20x11 in it.
7. After gen45's mc-label, build `v9_20x11_val.jsonl` (§4); the loop picks it up by itself.
8. The laptop: rebuild and rerun as in §6 whenever it should help.

## 8. Risks and open questions

- **Cost** (§5): generation, MC labels and drives all grow faster than the board area — longer
  drives (quadratic in MC labelling) and a 1.44× canvas. At 150 games a shard a generation is ~10–15%
  longer than today and holds 0.65× the samples; MC labelling is the phase to watch.
- **Weather is invisible to the net** except through `path_prob`; affects ~2.3% of kickoffs.
  Adding `weather` globals is a schema bump (v10) with a `bbnn.migrate` step; not worth it until the
  table's other results exist.
- **The kickoff table is mostly no-ops.** The 7-player game is BB2020 minus eight kickoff events.
  Implementing them (Blitz, Quick Snap, High Kick, Solid Defence are the ones with decisions) is a
  rules task with its own schema/search consequences; out of scope here.
- **Memory.** A 20x11 search holds bigger states (22x13 arrays) and longer drives: the measured peak
  RSS of 3 streams was 1.09 GB on 20x11 vs 0.71 GB on 16x9 (gen36, tract, 1000 descents). At 48
  streams the worker's `mem_governor` is the backstop (`WORKER_MEM_FLOOR_MB=1536` kept); watch
  `generate.worker.log` for admission throttling in gen45.
- **Corpus and prepared size.** Bytes per sample +15% (10.0 kB vs 8.7 kB JSON), `spatial.npy`
  +45% per sample at 22x13; fewer games per generation (§5) roughly cancels it per generation.
- **Drive reference** too weak on 20x11 (§4).
- **The window's mixed capacities** rely on the capacity-portable load; a corpus written by a
  *bigger* capacity whose board does not fit still fails loudly, never silently.
- **Net-check and value-bench series** change boards at the switch (§4).
- **The 96x8 vs 64x6 question (exp074)** is independent: whichever lineage generates at the switch
  continues; §5 gives both costs.
- **26x15/11** (the full board) stays out: its capacity breaks `ROSTER_PER_TEAM`-sized assumptions
  nowhere we know of, but 11 players is another density and the remaining kickoff events matter more
  there. Plan 042 E4's zero-shot probe on 26x15 can run any time from a default-capacity build.
