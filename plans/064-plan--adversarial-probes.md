# Plan 064 — adversarial probes of the value/policy net and the search

**Status:** tool built and first results in (exp075, 2026-10-16), gen36 (64x6) vs gen39 (96x8 + heads).
Follow-ups proposed in §8; none adopted yet.

Averaged benchmarks (value bench RMS, drives, net check) say how good a net is *on average*. They
cannot say where it is wrong in a way a human would call a bug: a skill that makes the net think its
own side is worse off, a mirror image valued differently, a player whose best square depends on where
he happens to start. This plan builds a probe tool that applies **controlled edits whose effect is
known** to real positions and counts how often, and how badly, the net and its search get them wrong —
then keeps the worst cases for inspection.

## 1. The tool: `botbowl-ui probe <suite>`

`botbowl-ui/src/probe/` (`mod.rs` dispatch, `common.rs`, `edits.rs`, one module per suite):

```sh
UI=$(scripts/probe_build.sh)          # 16x9/6 capacity (the loop's nets need it), niced, -j 2
taskset -c 6,7 nice -n 19 $UI probe skills --model models/az_v7/bbnet_mix16x9v9_gen39.onnx \
    --corpus runs/loopmix16x9v9/gen39/shard4.jsonl \
    --positions runs/loopmix16x9g/positions/contested_16x9_gen04g.json \
    --board 16x9 --states 150 --parallel 2 --seed 64000 --out-dir runs/exp075
scripts/exp075_probes.sh OUT_DIR {skills|symmetry|dominance|lone-mover|stability|hillclimb|all} NET.onnx...
scripts/probe_headers.py OUT_DIR       # one line per summary: suite, net, commit, dirty, rows, time
```

- **CPU only.** The net always runs on tract (`NnEvaluator::from_path`; there is no `--nn-server`, and
  `BLOOD_NN_SERVER` is ignored with a warning), on the **raw** value head output (unclamped, unquantised,
  mover-centric). `exp075_probes.sh` pins two cores (`CPUS`, default `6,7`) at nice 19.
- **Base states.** `--corpus`: decisions sampled uniformly (seeded) from random-start trajectories on the
  wanted boards and **rebuilt by replay** (`override_audit::replay_to`; a `--next-drive` record is
  skipped). `--positions`: frozen drive position sets regenerated from their seeds
  (`drives::position_state`), the screened contested ones first. Each is labelled `turn_start`,
  `turn_mid`, `activation` (a player activated, path offerings live) or `other` (block dice, push,
  skill questions, ...).
- **Deterministic.** Seeded sampling, deterministic edits, results in input order whatever the thread
  count. Policy-only results are byte-identical across runs; searches are not (`recon_mcts` HashMap
  order), which the stability suite measures.
- **Output** per suite and net tag: `{suite}.{tag}.jsonl` (a row per measurement),
  `.summary.json` (header with model, commit, dirty flag, seed, sizes; the tables' numbers),
  `.table.txt` (also on stdout), `.worst.json` (the worst cases with their rows and **ASCII boards** —
  `H`/`A` standing, `h`/`a` down, `*` the ball, `!` the edited player) and `worst/*.rec.json`, a
  before/after `Recording` that `botbowl-ui replay F` and `snapshot --replay F --step 1` show.

### Edit operators (`edits.rs`) and the two engine helpers

Every edit returns a new state differing in exactly the stated way, or refuses (`add_skill`,
`remove_skill`, `bump_stat` inside the training range, `translate`, `set_status`, `set_moves`,
`remove_player`, `add_lineman`, `add_rerolls`, `rewind_clock`). The tests pin "exactly" through the
encoder: adding a skill changes **one input cell** (that skill's plane at that player's square), a stat
bump changes that stat's plane (plus `movement_left` for MA) at one square, a translation keeps every
other player, the ball and the game info, and every edit undone gives back the identical state.

Two analysis helpers were added to the engine, TDD-first, and nothing in the play path calls them:

- **`GameState::y_flipped()`** — the long-axis mirror (`y -> h-1-y`): players, board index, ball,
  bounce squares, positional offerings. Tests: twice is the identity; it moves players and ball only;
  **at a `Turn` decision it is a legal state** (it offers exactly the flipped actions and plays on), and
  the encoder maps it to **exactly the reflected tensor, path plane included** (`edits.rs` test) — so
  any value difference under the mirror is the net's, not the encoder's or the pathfinder's.
- **`GameState::reask_turn_decision()`** — regenerate a `Turn` decision's offerings after an edit
  (otherwise they still name the old squares). Idempotent on unedited states, follows a moved player,
  refuses anything but a `Turn` decision (other procedures advance on a re-ask; `MoveAction` panics).

Value-only suites edit any decision state: the encoder reads nothing the edit leaves stale (it
recomputes the path plane). Suites that play on (lone mover, hill-climb) edit `Turn` decisions only and
re-ask.

### Noise references

Two references, both measured on the same base states:

- **Level noise**: `|V(y_flipped s) - V(s)|`. True difference 0; it measures how far the net is from
  the symmetry it was trained with (`train/src/bbnn/data.py` random y-flip).
- **Placebo floor** (`tau`): the net's `|ΔV|` for adding a skill **no engine rule reads** (Leap, Pass,
  Kick, ... — §2's `inert` class) — the same one-cell edit with true effect 0. **`tau` = its 99th
  percentile.** A violation larger than `tau` is one a meaningless edit almost never produces. This is
  the right yardstick for differential effects; the level noise is 5-15x larger (§7) and would hide
  everything.

## 2. Skill classification (`probe/skills.rs::class_of`)

From the engine's rules (`botbowl-engine/src/core/procedures/*`) and what the training distribution
contains: random-start players are linemen with stats nudged off the baseline and skills drawn only
from `Skill::good_skills()`, so a skill outside that pool has a plane that has only ever been zero.

| class | skills | rule / reason |
|---|---|---|
| **beneficial** (in the pool) | Block, Dodge, Catch, Sure Hands, Jump Up, Sidestep*, Guard, Mighty Blow, Tackle*, Wrestle*, Stand Firm* | never hurt the owner; * = a use-or-not choice the owner controls (`UseSkill`/`DontUseSkill`) |
| beneficial, **never in training** (`beneficial_ood`) | Sure Feet, Dauntless, Fend, Claws, Arm Bar, Brawler*, Grab, Juggernaut* | implemented and never harmful, but no training position ever had them |
| **unencoded** | Strip Ball | implemented, but `ENCODED_SKILLS = 39` stops before it: the net cannot see it (ΔV ≡ 0, tested) |
| **double-edged** | Frenzy | forced follow-up and second block; excluded from the violation count |
| **inert** (no rule) | Leap, Sprint, Dirty Player, Pile Driver, Quick Foul, Shadowing, Sneaky Git, Kick, Accurate, Nerves of Steel, On the Ball, Pass, Break Tackle, Bone Head, Loner, Stunty, Throw, Wild Animal, Kick-Off Return | nothing reads them; the placebo |
| characteristics | ST+1, MA+1, AG+1, AV+1 (inside the generator's ranges) | better is better; PA is not encoded, so not probed |

## 3. The suites and their metrics

1. **`skills` — skill monotonicity.** Every player of every base state, every skill not held and every
   characteristic +1. `gain = ΔV` for a mover's player, `-ΔV` for an opponent's (mover's frame); a
   beneficial edit with `gain < 0` is a violation. Per edit: mean/median gain, violation rate,
   rate beyond `tau`, own/opp split, worst, and (since 8aa7c25) the split on prone/stunned players.
2. **`lone-mover` — the user's design.** From a `Turn` decision with the ball in the opponent's half,
   take the mover's free player (standing, unused, not carrying, unmarked; furthest from the ball) and
   put him on each square of the **back field** (the 4 columns in front of his own endzone) that is 3+
   squares (Chebyshev) from every other player and from the ball; re-ask; activate him (`StartMove`).
   *Representation:* a `Move` names a destination and the engine walks the path; the search's pruning
   (P1 + P8) makes a move activation exactly one `Move` then `EndPlayerTurn`, so the destination is the
   single `Move` target (the policy's argmax over the search's legal set, or the search's pick; the
   activation is played to its end under one fixed dice seed per scenario). Metrics per agent (policy,
   search@250, search@1000; `cfgs/chance_all_partial.toml`, the loop's eval preset): the **modal
   destination**, **agree** (share of the starts from which the mode is reachable that went there),
   spread (mean Chebyshev distance from the mode), distinct destinations; and the value side — every
   reachable square X gets `V_X` (the base with the player on X, marked used): **follow-V** (share of
   starts that went to the value's best reachable square), **regret** (`V_best - V_X(chosen)`) and
   **v_range** (`max - min V_X` over the agent's destinations; `> tau` means the value thinks the
   destinations differ, so the policy is not just breaking value ties).
3. **`symmetry`.** y-flip: `|ΔV|`, the policy's total variation between `π(s)` and the mirrored
   `π(flip s)` over the search's legal set, and argmax agreement. Colour swap (`GameState::mirrored`,
   x-mirror + sides): must be exactly 0 (the encoder canonicalises it) — a canary for the perspective
   code. **Used player's movement**: set a used player's `moves` to 0 or MA+2 — nothing reads it until
   his next turn resets it, but the encoder shows it (`movement_left`).
4. **`dominance`.** Edits that (nearly) dominate for one side, `gain` signed to be `>= 0`: a re-roll
   for the mover / the opponent (strict: an option is never a cost), remove an opponent / an own player
   (non-carrier), add an own lineman on a square 2+ from everyone, stand an own prone player up, knock a
   standing opponent down, move the mover's carrier one square toward his scoring end (both squares
   unmarked), and a soft one, one more turn on the clock for both sides when the mover has the ball.
5. **`stability`.** For each base decision with two or more moves: a fresh search at 250, 1000 and
   4000 descents and a second fresh search at 1000. Flip rates between budgets, the repeat flip rate
   (the preset is deterministic, `gumbel_scale = 0`, so a repeat differs only through HashMap order),
   the Q gap at 4000 between two budgets' picks (near tie < 0.01), agreement with the policy argmax.
6. **`hillclimb` — adversarial search for violations.** From a `Turn` decision take the worst of 8
   (own player, pooled beneficial skill) pairs, then propose 80 random mutations of *the rest* of the
   position (move any player one square, toggle a pooled skill or nudge a characteristic on another
   player, stand one up / knock one down), each kept only if it makes `f = V(s) - V(s + skill)` larger.
   Mutations stay inside the training ranges and the state is re-asked, so the end point is a legal,
   plausible position.

## 4. Runs (exp075)

`scripts/exp075_probes.sh runs/exp075 SUITE models/az_v7/bbnet_mix16x9v9_gen{36,39}.onnx` (outputs
in the worktree's `runs/exp075/`, not committed). Board 16x9/6 unless noted, seed 64000, 2 threads on
cores 6-7 at nice 19, tract. Commits (each summary header carries its own; `scripts/probe_headers.py`):
symmetry e6a198d; lone mover and stability 8aa7c25 (the binary they ran was built at e6a198d — the
suites' code did not change in between); skills, dominance, hill-climb 880fd51 (rerun after the
characteristic-range fix, see §6 "a bug the probes found in the probes"); 14x7/4 symmetry and
dominance b4be123, 14x7 skills 880fd51.

| suite | base states (per net) | size |
|---|---|---|
| skills | 150 gen39 val-shard decisions (`runs/loopmix16x9v9/gen39/shard4.jsonl`, held out from gen39's training, unseen by gen36) + 150 contested positions (`runs/loopmix16x9g/positions/contested_16x9_gen04g.json`) | 128,286 edits |
| symmetry | 300 + 300 | 600 states |
| dominance | 300 + 300 | 9,016 edits |
| hill-climb | 60 contested positions | 60 climbs × 80 proposals |
| lone mover | the first 12 contested positions that qualify | 140 starts × 3 agents |
| stability | 20 + 20 | 40 decisions × 4 searches (250, 1000, 4000, 1000 again) |
| 14x7/4 | 100 + 100 (`contested_14x7_gen04g`) | symmetry, skills (54,359 edits), dominance |
| value bench | `runs/value_bench/v9_gen01_val.jsonl` (3023 states) | plain and `--symmetrise`, `scripts/exp075_vbench.sh` |

Wall time on two cores: gen36 ~15 min for every suite together, gen39 ~45 min (a 96x8 forward is ~5.6
ms on two tract threads; the 64x6 ~2 ms).

## 5. Results: gen36 (64x6) vs gen39 (96x8 + heads)

### 5.1 Noise references

| | gen36 | gen39 |
|---|---|---|
| level noise `\|V(flip) - V\|`, p50 / p90 / p99 (600 states) | 0.019 / 0.070 / 0.127 | 0.027 / 0.103 / 0.194 |
| placebo `\|ΔV\|` (inert skills), p50 / p99 = `tau` (56,772 edits) | 0.0014 / **0.0112** | 0.0011 / **0.0094** |
| placebo violations beyond `tau` (the false-alarm rate by construction) | 0.5% | 0.6% |

### 5.2 Skill monotonicity (suite 1)

By class: share of edits where the owner lost value at all / by more than `tau`.

| class | gen36 16x9 | gen39 16x9 | gen36 14x7 | gen39 14x7 |
|---|---|---|---|---|
| beneficial (pooled skills + characteristics) | 23.2% / **2.0%** | 19.8% / **1.7%** | 21.5% / 1.8% | 17.3% / 1.1% |
| beneficial, never in training | 48.1% / 0.5% | 54.0% / 0.6% | 49.0% / 0.7% | 55.7% / 0.8% |
| inert (placebo) | 47.9% / 0.5% | 54.1% / 0.6% | 47.0% / 0.5% | 55.1% / 0.7% |
| Frenzy (double-edged) | 25.3% / 2.2% | 31.2% / 3.9% | 22.9% / 1.8% | 28.2% / 2.5% |
| Strip Ball (unencoded) | 0 (ΔV ≡ 0) | 0 | 0 | 0 |

Per edit, 16x9 (mean gain; violations; beyond `tau`; own / opp violation; the prone-player subset):

| edit | gen36 mean, viol, >tau | gen39 mean, viol, >tau | note |
|---|---|---|---|
| **Wrestle** | +0.0016, 43%, **7.8%** | +0.0024, 41%, **8.3%** | opp gets it → *our* value up 57% / 46% of the time |
| **Jump Up** | −0.0004, 50%, **11.8%** | +0.0026, 35%, **4.8%** | on a prone player (where it matters) only +0.0018 / +0.0044 |
| Guard | +0.0097, 19%, 1.9% | +0.0133, 14%, 2.5% | |
| Catch | +0.0026, 32%, 0.9% | +0.0023, 33%, 2.3% | gen39 worse |
| Block | +0.0276, 8%, 1.3% | +0.0262, 9%, 2.0% | the single worst edit: −0.11 (gen39) |
| Mighty Blow | +0.0018, 35%, 1.1% | +0.0032, 22%, 0.4% | barely valued |
| Tackle | +0.0024, 32%, 0.7% | +0.0045, 14%, 0.1% | |
| **AV+1** | **−0.0007**, 65%, 0.3% | **−0.0006**, 67%, 0.3% | **own player 85% / 86% of the time worse off** — a consistent sign error, small |
| ST+1 / AG+1 / Dodge | +0.034 / +0.015 / +0.030, ≤ 6% viol | +0.041 / +0.016 / +0.033 | well learned |
| MA+1 | +0.0073, 9%, 0.6% | +0.0072, 13%, 0.6% | worst −0.10 on an *active* player (gen39) |

Out-of-training beneficial skills (Claws, Juggernaut, Grab, Fend, ...): mean `|ΔV|` 0.002, the
placebo's — **the net is blind to them** (expected: their planes were always zero in training).

### 5.3 Symmetry

| | gen36 16x9 | gen39 16x9 | gen36 14x7 | gen39 14x7 |
|---|---|---|---|---|
| `\|ΔV\|` y-flip, mean / p90 / max | 0.029 / 0.070 / 0.20 | **0.043 / 0.103 / 0.35** | 0.033 / 0.077 / 0.14 | 0.048 / 0.096 / 0.27 |
| policy TV y-flip, mean / p90 | 0.073 / 0.165 | 0.064 / 0.157 | 0.071 / 0.162 | 0.056 / 0.140 |
| **policy argmax is not the mirrored move** | **10.2%** | **8.8%** | 7.8% | 6.7% |
| colour swap `\|ΔV\|` (canary) | 0 exactly | 0 exactly | 0 | 0 |
| used player's movement `\|ΔV\|`, mean / max | 0.027 / 0.23 | 0.022 / 0.19 | 0.027 / 0.21 | 0.025 / 0.13 |

**Test-time y-flip averaging** (`value-bench --symmetrise`, `scripts/exp075_vbench.sh`, paired on the
same 3023 states): gen36 RMS 0.196 → **0.194 (−1.2%, dMSE −0.0009 ± 0.0002)**, gen39 0.182 → **0.179
(−1.9%, dMSE −0.0013 ± 0.0003)**. For gen39 that is the size of the +1.8% its own value bench lost
against its generator.

### 5.4 Dominance (16x9; violation / beyond `tau` / worst)

| edit | gen36 | gen39 |
|---|---|---|
| reroll for the mover / for the opponent (strict) | 1.3% / 0 / −0.001; 3.9% / 1.1% / −0.043 | 2.6% / 0 / −0.002; 6.1% / 0.6% / −0.014 |
| add an own lineman (2+ from everyone) | 6.1% / 3.8% / −0.16 | 2.2% / 1.1% / −0.03 |
| remove an opponent / an own player | 3.7% / 1.4% / −0.11; 1.9% / 0.8% / −0.16 | 2.6% / 1.0% / −0.25; 2.4% / 1.2% / −0.19 |
| knock a standing opponent down | 9.8% / **4.7%** / −0.21 | 7.8% / **4.0%** / **−0.30** |
| stand an own prone player up (n = 39) | 15% / 10% / −0.03 | 15% / 5% / −0.03 |
| **carrier one square toward the endzone, open field** (n = 86) | **22% / 12%** / −0.06 | **23% / 9%** / −0.13 |
| one more turn for both, mover has the ball (soft) | 80% / 2.3% (mean +0.006) | 60% / 3.5% |

Violations concentrate where a player is activated: gen39's `remove_opp` beyond-`tau` rate is 2.3%
at activation decisions vs 0.8% at turn starts, `knock_down_opp` 5.4% vs 4.2%, `add_own` 2.5% vs 0.6%
(`scripts/probe_breakdown.py`). The worst case of all (gen39, `gen39/shard4.jsonl:148:7`, dumped as
`dominance...gen39.00`): a Home player mid-move (`MoveAction`, a plain move) is marked by one Away
player; knocking that marker down costs Home **0.30**, removing him 0.25 — both should free the mover.

### 5.5 Hill-climb (60 climbs × 80 proposals)

| | gen36 | gen39 |
|---|---|---|
| violation `V(s) - V(s + skill)` at the base, mean / max | 0.006 / 0.029 | 0.006 / 0.044 |
| after the climb, mean / max | 0.020 / 0.071 | 0.023 / **0.109** |
| climbs ending beyond `tau` | 48% | 60% |
| mutations kept, mean | 16 | 15 |

The worst found: Wrestle on an own player (gen39, 0.109; gen36 0.070) and Jump Up (gen36 0.071) — the
same two skills the random sample flags, now at ~10x `tau`, in legal positions ~15 mutations away
from a real one.

### 5.6 Lone mover (12 scenarios, 140 starts)

| agent | agree | all starts agree | 2+ squares off the mode | distinct | follow-V | regret | `v_range` > `tau` |
|---|---|---|---|---|---|---|---|
| gen36 policy | 62.5% | 2 / 12 | 20.7% | 3.5 | 4.2% | 0.073 | 75% |
| gen36 search@250 | 54.2% | 1 / 12 | 23.6% | 4.3 | 3.5% | 0.070 | 83% |
| gen36 search@1000 | 53.5% | 1 / 12 | 22.9% | 4.7 | 3.5% | 0.076 | 83% |
| gen39 policy | 66.0% | 2 / 12 | 22.1% | 3.3 | 0.7% | 0.076 | 67% |
| gen39 search@250 | 57.6% | 2 / 12 | 24.3% | 4.0 | 0.7% | 0.077 | 75% |
| gen39 search@1000 | 51.4% | 1 / 12 | 30.0% | 4.3 | 0.7% | 0.080 | 92% |

Same destination from the same start, policy vs search@1000: 66% (gen36), 73% (gen39).

**The frontier confound** (`scripts/lone_mover_frontier.py`). The agents move the full MA (6) in
81-83% of starts, and the destination is roughly "the start plus six squares toward the ball": the
square they want lies beyond reach, so the consistent move is the closest reachable one, and that
*does* depend on the start. The value agrees — its best reachable square is ~7 squares out on average
and is the same for every start in only 4 / 12 (gen36) and 3 / 12 (gen39) scenarios. Taking the
frontier out (starts from which the agent's modal destination is within 5 squares): **at the mode 51%
(policy) / 54% (search@1000) for gen36, 53% / 42% for gen39** (n ≈ 40 each) — the inconsistency
remains, on a small sample. And in these quiet positions the agents **GFI** (move 7+ squares) in
22-25% (policy) and 16-18% (search@1000) of starts — 8-20% even when the modal square was within 5.

### 5.7 Search stability (40 decisions)

| | gen36 | gen39 |
|---|---|---|
| move differs 250 vs 1000 / 1000 vs 4000 / 250 vs 4000 | 7.5% / **40%** / 45% | 20% / 20% / 35% |
| median Q gap (at 4000) between the two moves of a flip | 0.06 | 0.09 |
| a second search at 1000 picks another move | 2.5% | **10%** |
| plays the policy's argmax at 250 / 1000 / 4000 | 88% / 85% / 48% | 88% / 73% / 58% |
| turn-start decisions, 250 vs 4000 differ | 58% (n 26) | 50% (n 26) |

## 6. Reading

- **Monotonicity mostly holds, and where it fails the failures are skill-specific, not noise.** The
  pooled beneficial class violates beyond the placebo floor 3-4x as often as the placebo (2.0% /
  1.7% vs 0.5%), and almost all of the excess is a handful of edits: **Wrestle, Jump Up, Guard, Catch,
  Block**, and the AV sign. These are rare-event skills — Wrestle only acts on a Both Down, Jump Up only
  on a prone player, Catch only on a pass or a bounce, AV only on a broken armour — so the value target
  (one drive outcome, or 8 policy playouts) carries almost no signal about them, and a random-start
  player holds each with probability ~4%. That is a **training-signal problem** (credit too thin and too
  rare), not capacity: the 96x8 fixes Jump Up partly and Tackle/Mighty Blow, but not Wrestle, and gets
  Catch and Guard slightly worse. The AV sign error (−0.0007, 85% of own players) is the cleanest
  example: tiny, consistent, the wrong way, and on a characteristic that only matters through the
  injury roll — a correlation the net picked up, not a rule it learned.
- **Skills outside the pool are invisible**, Strip Ball by construction (no plane) and the eight
  never-fielded ones by training. Harmless for the loop today; it matters the day a roster or the web
  UI fields them (custom teams can), and it is a silent hole: the net will play a Claws or Juggernaut
  player as a lineman.
- **The mirror is not learned, and the bigger net is worse at it.** Despite random y-flip augmentation,
  a mirrored position's value differs by 0.03-0.04 on average and up to 0.35, and the policy's top move
  is not the mirrored one in ~9-10% of decisions. gen39's value is *less* symmetric than gen36's
  (mean 0.043 vs 0.029), its policy slightly more. Augmentation makes the expected target symmetric,
  not the function: with ~1 epoch of new data per generation the net sees each position in one
  orientation per epoch, and the larger net fits more orientation-specific noise. Averaging over the
  mirror at inference is a measured −1.2% / −1.9% value RMS for free (two forwards, or one batch of 2).
- **The net reacts to inputs that should not matter.** A used player's `movement_left` changes the
  value as much as the mirror does (mean 0.02-0.03, max 0.23). Nothing reads it; the encoder shows it;
  the net uses it as a proxy ("this player has moved, so the position is mid-turn ..."). Same family:
  the worst dominance cases sit at activation decisions, where the encoder shows an active player and
  the path plane but **not the decision's kind** (`player_action_type`, the procedure: move, blitz,
  block die, push) — so the value has to guess the context from the board, and removing a marker or a
  block target changes the guess.
- **The search does not make the lone mover more consistent; it makes it less.** At 250/1000 descents
  the Gumbel root mostly plays the policy (73-88% argmax agreement), and where it overrides in this
  quiet position it spreads the destinations further (agree 51-58% vs the policy's 62-66%): the
  move fan is ~96 squares, 16 are considered, and sequential halving picks among near-equal Qs that
  move with the budget. The stability suite says the same from the other side: the move changes
  between 1000 and 4000 descents in 20-40% of decisions with real Q gaps (0.06-0.09), and a repeat at
  1000 differs 2.5-10% — **the search has not converged at the loop's 1000-descent budget**, consistent
  with the net check's gain still rising from 1000 to 4000 (plan 055).
- **The user's hypothesis, refined.** The probe confirms the start-dependence but most of it is the
  frontier effect: the target is out of reach, so "as close as possible" legitimately depends on the
  start. What remains — ~50% agreement when the mode is 5 or fewer squares away, and GFIs in a quiet
  position 8-25% of the time (a failed GFI is a turnover) — is the real weakness: a policy that
  advances by habit (full MA, often +1) rather than to a square. The `regret` / `follow-V` numbers are
  upper bounds: `V_X` places the player on X without paying for the GFI or dodge it takes to get
  there, which biases the value's best square forward.
- **A bug the probes found in the probes.** The first skills run let AG+1 reach 6, which the engine
  cannot play (`ag_target` = 7 − ag has no 1+; it panicked on 14x7). The random-start generator's
  `AG_RANGE` upper bound is 6 too — unreachable today only because nudges are ±2 from AG 3. The edit
  ranges are now baseline ±2 (880fd51); worth a guard in the generator (`MAX_AG` should be 5, or
  `ag_target` should accept 1+). **The web app reaches it today:** the team editor clamps AG to
  `botbowl-web/proto` `MAX_AG = 6` (`TeamDef::sanitised`, whose doc says a saved team "cannot panic a
  session"), and an AG 6 player panics the first time the pathfinder builds his dodge target
  (`pathing.rs` `GameInfo::new` → `ag_target`). `PlayerStats::MAX_AG`'s own doc ("AG is a D6 target
  number, so 6 is the worst") reads AG the BB2020 way, the opposite of `ag_target`'s LRB6 one. Lowering
  `PlayerStats::MAX_AG` is **not** free: the encoder divides the AG plane by it (`AG_NORM`), so it is a
  schema change. The guards that change no current game: `ag_target` saturating at 2+ (a natural 1
  always fails, so AG 6 plays as 2+), or the web proto's `MAX_AG` at 5.
- **Caveats on the yardsticks (review).** The placebo skills' planes, like the never-fielded skills',
  were zero throughout training, so `tau` is the response of an *untrained* input channel; a trained
  plane's spurious response can be larger, so "beyond `tau`" is a floor-based screen, not a
  significance test — the evidence is the per-skill concentration (Wrestle, Jump Up, AV), not the
  pooled rate. The hill-climb keeps the best of 80 proposals of a noisy function and has no placebo
  climb as a control, so "climbs ending beyond `tau`" is inflated by selection; which skills top the
  climbs is the informative part.

## 7. Suggested fixes and follow-ups

1. **Symmetrised inference** (cheap, measured): average V (and the policy's logits, mirrored) over the
   y-flip in `NnEvaluator`, behind a flag; on the sidecar it is one batch of 2. Judge on drives
   (policy-only first: minutes on tract). Expected: the −1.2-1.9% value RMS and fewer
   mirror-dependent argmaxes.
2. **Invariance as a training loss**: add `λ · (V(s) − V(flip s))²` (and a KL between the mirrored
   policies) to the trainer instead of — or with — random flips; the flip pair costs one more forward
   per sample. Also zero the `movement_left` plane of used players in the encoder (a schema-neutral
   change? no — it changes inputs the nets were trained on; do it with the next schema bump), or add it
   to the invariance loss.
3. **Rare-skill credit**: the monotone skills are the ones whose effect is concentrated in rare
   events. Options, in order of cost: (a) raise `ADD_SKILL_PROB` or sample positions *around* the
   events (prone players with Jump Up, a pending Both Down with Wrestle); (b) a **paired monotonicity
   regulariser** — for a sample, add a random pooled skill to a random player and penalise
   `max(0, −sign·ΔV)` (the probe's gain), one extra forward; (c) MC-labelled *pairs* (with and without
   the skill, same dice) so the target itself carries the skill's effect.
4. **Encode the decision context**: `player_action_type` and the procedure kind (move / blitz / block
   die / push / skill question) as global one-hots — the value and the policy currently guess them.
   Schema bump; measure with the dominance suite's activation breakdown and the value bench's
   `mid_activation` RMS (the worst phase, 0.195 vs 0.137 at turn start).
5. **Lone mover v2**: write each start's fan and path probabilities into the rows, value `V_X`
   weighted by the probability of getting there, and fit one target square per scenario (is each
   destination the closest reachable square to a single T?). That separates "consistent toward a far
   target" from "inconsistent" properly; and add the GFI-in-a-quiet-position rate as its own metric.
6. **A standing per-net check** (`scripts/probe_check.sh`, like `net_check.sh`, on CPU, ~15 min at
   64x6 / ~45 at 96x8, or minutes on the sidecar): skills (beneficial >tau rate, the Wrestle / Jump Up /
   AV rows), symmetry (mean `|ΔV|`, argmax disagreement), dominance (carrier_forward, knock_down_opp),
   one line in `registry/NETS.md` per net. A regression in these is cheap to see and invisible to the
   averaged benches.
7. **Search**: the stability suite says 1000 descents have not converged; it is the same conclusion
   as the net check, from a different angle — if a budget change is ever on the table, this is the
   cheap per-decision view of it. Track the repeat-flip rate (HashMap order): 10% for gen39 at 1000.
8. **Engine guard**: `MAX_AG` / `AG_RANGE` vs `ag_target` (§6) — reachable from the web team editor
   today; a placebo-skill hill-climb as the suite's control.

