# Plan 051 — Fast bot ranking: SPRT on paired games, contested paired drives, and a validation harness

**Status:** Steps 1-5 done 2026-10-02 (the disagreement filter is not built). P1 (SPRT on full games) is
adopted for A/B decisions. P2 (contested drives) fails acceptance (a) on one pair but is, by the
user's decision, the main metric for this phase, with guards. See step 5.

## Problem

Resolving a small change between two bot configurations costs a day. The numbers behind
that, from plan 030 and `scripts/anchor_curve.py`:

| | |
|---|---|
| one full nn-vs-nn game at 1000 iters | 2.48 min (plan 030, B1 pace) |
| anchor rung per generation | 40 games, SE ≈ 0.07 |
| 3-generation rolling window | 120 games, SE ≈ 0.04 |
| games to resolve 0.55 vs 0.50 at 5%/5% error, fixed N | ≈ 780 (σ_pair ≈ 0.3) |

A "genuinely better" net scores 0.54–0.55 against its parent (plan 030). Nothing we run
today can see that in one sitting, so experiments either run overnight or are called on
noise. Plan 025 adds that the search itself is not seed-reproducible, so repeats are
unavoidable and no proxy can be made deterministic.

Head-to-head full games stay the ground truth. The goal here is a cheaper measurement that
orders bots the same way, validated against that ground truth before anything trusts it.

## Decision

Three things, in this order, each cheap on its own and each validated before use:

1. **SPRT with pentanomial pair scoring and a score margin** on the ladder we already run.
   Stops a head-to-head as soon as it is decided. Pure statistics, no change to play.
2. **A validation harness**: a frozen set of net pairs with gold results from long matches,
   and a script that scores any proxy on sign agreement, false positives on zero-gap pairs,
   and wall-clock to decision. Built once; every later proxy goes through it.
3. **Paired contested drives** as a new ladder rung kind: random-start drives from a frozen
   position set screened to be contested, each played with the sides swapped, optionally
   filtered to positions where the two bots' first moves disagree.

Search budget for ranking is left alone; it is a knob to tune once the harness exists.
Deterministic dice (`DicePolicy::SucceedAtOrEasier`) stays a lecture/diagnostic tool and is
**not** a ranking method: the bot plans under true probabilities while the environment
resolves them asymmetrically, so it penalises exactly the risk management the game is
about. Search-only probes against a deep reference (plan 025 machinery) are a regression
detector, not a ranking, and are deferred.

## Why these

**Pairing is the big free win and we already half-do it.** `ladder_assignment`
(`botbowl-play/src/eval.rs:106`) gives games `g` and `g+1` the same seed with sides swapped,
but `LadderRow::record` folds them as independent games. Two games from one seed are not
independent: a one-sided seed yields a win and a loss that cancel and say nothing about the
bots, while a trinomial count treats them as two informative samples and overstates
confidence. Scoring the **pair** as one unit with five outcomes (0, ½, 1, 1½, 2 points)
estimates the variance correctly and is what fishtest moved to for the same reason.

**SPRT answers the question plan 030 could not.** A fixed-N threshold test on a point
estimate gets *worse* with more games when the threshold sits above the true effect (plan
030: 41% at n=100, 33% at n=400). The sequential test asks "is the evidence for s₁ over s₀
strong enough" and stops at a bound; more games can only sharpen it. Expected cost, normal
approximation with σ_pair ≈ 0.3 and α = β = 0.05, s₀ = 0.50, s₁ = 0.55:

| true score | expected pairs | games |
|---|---|---|
| 0.50 (H0 true) | ≈ 190 | ≈ 380 |
| 0.55 (H1 true) | ≈ 200 | ≈ 400 |
| 0.60 | ≈ 70 | ≈ 140 |
| 0.70 | ≈ 30 | ≈ 60 |

So SPRT roughly halves the cost at the boundary and collapses it for clear results. It does
**not** make a 0.55 result cheap on full games — that is what step 4 is for.

**Drives are a fifth of a game and the training objective is drive-bounded anyway.** Plan
049 §8 already notes the mismatch: training covers single drives, eval plays full games. A
drive-level proxy is closer to what the net is trained on, which makes it more sensitive to
net changes and blind to full-game skills (kickoff setup, clock, the 2–1 grind). That is
acceptable for *ranking* if the harness shows the ordering agrees; it is the reason full
games stay the ground truth.

**Contested positions carry the information.** A position both bots score from, or both
fail at, is a sample with zero variance between bots. Screening the frozen set to positions
where the reference attacker scores 25–75% of the time concentrates every drive on states
that can separate the bots, the same logic as chess testing's balanced opening books.

**The disagreement filter is cheap and self-cancelling if useless.** Two searches per
position (≈ 1.5 s each at 1000 iters) decide whether the bots would even play differently.
Plan 025's noise floor says independent searches disagree often, so the filter may pass
nearly everything — then it costs two searches and changes nothing. It is optional and
measured.

## Where things are today (for whoever implements this)

- **Ladder game:** `botbowl-play/src/eval.rs::play_ladder_game` (:128) builds from
  `BuilderState::CoinToss`, `set_seed(seed)`, `DiceMode::RollDice`, loops to `game_over` or
  `max_steps`, returns an `EvalGameLine` (:38: `rung, game, seed, candidate_team, home_score,
  away_score, kicking_first_half, finished, board, telemetry`). `candidate_scores()` (:93)
  gives (candidate, opponent) TDs.
- **Fold:** `LadderRow::record` (:291) decides W/D/L by `cand.cmp(&opp)`; `finish` (:329) sets
  `win_rate = wins / games`, so a draw counts as a loss there. `scripts/eval_summary.py`
  already reports points `(W + D/2)/N` instead; `anchor_curve.py::rolling` computes points
  with a pooled, unpaired SE. Every `LadderRow` field is a commutative counter so the hub can
  fold lines arriving in any order.
- **Shells:** `botbowl-ui/src/eval.rs::run` (:262) builds rungs, `run_rung_games` (:217) pulls
  game indices from a shared counter and calls `ladder_assignment` then `play_ladder_game`.
  The hub mirrors it: `botbowl-hub/src/state.rs` creates one `Rung { name, total, opponent,
  board, row, done }` per rung (:318), folds in `eval_game_done` (:647) deduplicating by game
  index, and `finish` (:746) writes `report.json`. The worker calls the same two functions
  (`botbowl-worker/src/lib.rs:386`). **There is no cancel or early-stop path anywhere**: a job
  ends only via `all_done()` or `fail`.
- **Random start:** `botbowl_curriculum::generate_random_start(cfg: &RandomStartConfig, rng:
  &mut ChaCha8Rng) -> GameState` (`random_start.rs:143`); `RandomStartConfig` (:63) has the
  placement biases, `temperature`, `carried_prob`, `board_dims`. Deterministic given
  (config, seed) at a fixed commit. `botbowl-play/src/generate.rs::random_start_trajectory`
  (:415) ends a drive with the stop closure `score changed || half changed`, plus
  `game_over` and `max_steps`.
- **Reading a search:** `MctsBot::get_action_with_record` (`dynamics.rs:2923`) returns the
  action and a `Sample` with `children: Vec<ChildStat{action, visits, q, prior}>`,
  `root_value: Option<i64>` (mover-centric, `Q_SCALE` = 1000). `last_search()` (:2645) gives
  the `SearchSummary` with `chosen`, `root`, `children` sorted by visits. `evaluate_home`
  (:2615) is one NN forward pass, Home-centric, no search — cheap but plan 049 §4 says the
  value head is optimistic and self-referential, so it is a prefilter at most.
- **History:** `runs/` lives on the training box only (gitignored). Per generation:
  `report.json`, `eval.games.jsonl`, sometimes `anchor.json`; weights in
  `models/bbnet_<tier>_genNN.onnx`. Old results were played by older binaries (virtual-loss
  fix, hash fix), so the harness's gold must be **re-played with the current binary**, not
  read from history. History is only used to *pick* the pairs.
- **No dice log exists** (grep: nothing). Luck covariate adjustment would need a per-roll
  record through `RegisterRolls` or a counter in `resolve_with_rng`; parked, see "Not now".

## Step 1 — pentanomial pairs, SPRT, score margin (Rust, no box time)

New module `botbowl-play/src/stats.rs`, shared by both shells:

```rust
pub struct Pentanomial { pub counts: [u32; 5] }         // pair points 0, ½, 1, 1½, 2
impl Pentanomial { fn pairs(&self) -> u32; fn mean(&self) -> f64 /* per game, in [0,1] */;
                   fn var(&self) -> f64 /* of the per-game pair mean */ ; fn se(&self) -> f64 }

pub struct Sprt { pub s0: f64, pub s1: f64, pub alpha: f64, pub beta: f64 }
pub enum Verdict { H0, H1, Undecided }
impl Sprt {
    /// Normal-approximation LLR on the pair mean (fishtest's "LLR_normalized"):
    ///   LLR = N (s1 − s0) (2 ȳ − s0 − s1) / (2 σ²),  N = pairs, ȳ = mean, σ² = var
    /// Guard: Undecided until N ≥ 8 and σ² > 0 (fall back to s(1−s) if the sample variance is 0).
    pub fn llr(&self, p: &Pentanomial) -> f64;
    pub fn bounds(&self) -> (f64, f64);   // (ln(β/(1−α)), ln((1−β)/α)) = ±2.944 at 5%/5%
    pub fn verdict(&self, p: &Pentanomial) -> Verdict;
}
```

`LadderRow` gains, all commutative:

- `points: f64` = `(W + D/2) / games` next to `win_rate` (which stays, for old readers).
- `margin_sum: i64`, `margin_sq_sum: i64` (candidate TDs − opponent TDs per game) → mean
  margin ± SE in the report. The margin is reported, not decided on, until the harness says
  it discriminates better than points.
- `pairs: Pentanomial` and `sprt: Option<SprtStatus { s0, s1, llr, pairs, verdict }>`.

Lines arrive one at a time and in any order, so the pair fold needs a half-pair stash keyed
by `game / 2`, held by the fold owner (ui rung runner, hub `Rung`) and `#[serde(skip)]`-ed.
A pair is folded when both halves are in; an unfinished pair at the end is dropped from
`pairs` but its games stay in W/D/L.

Early stop:

- `botbowl-ui eval --sprt S0:S1[:ALPHA:BETA]` — the shared game counter in `run_rung_games`
  stops handing out indices once the rung's verdict is not `Undecided`; in-flight games finish
  and are recorded (overshoot does not bias an SPRT).
- Hub: same flag on `job eval`; `Rung` gets the rule and `eval_game_done` drops the rung's
  remaining `pending` entries on a verdict; `all_done()` treats a decided rung as done. This
  is the first job-level stop the hub has; keep it rung-scoped, no new routes. **Protocol
  bump**: `EvalGameLine` is unchanged on the wire, but the job request gains the rule.
- `--games` stays as the cap. Default when `--sprt` is absent: exactly today's behaviour.
- `scripts/eval_summary.py` and `anchor_curve.py` read `points` and the pair SE when present.
  The per-generation anchor rung keeps a fixed N: it is a curve point, not a verdict, and a
  variable N would make the rolling window heteroscedastic. SPRT is for A/B decisions.

Tests: known pentanomial → known LLR (hand-computed), symmetric verdicts under swapping
candidate/opponent, guard behaviour, and a `LadderRow` fold that gets the same `pairs` for
any arrival order.

**Done 2026-10-01 (local half; the hub half is order-of-work item 2).** `botbowl-play/src/stats.rs`
holds `Pentanomial` and `Sprt` (`parse`, `bounds`, `llr`, `verdict`, `status`). `LadderRow` gains
`points`, `margin_{sum,sq_sum,mean,se}`, `pairs`, `points_se` and `sprt: Option<SprtStatus>`, all
serde-defaulted so old reports parse. The pair fold lives *in* `LadderRow::record`, with the
half-pair stash a `#[serde(skip)]` field of the row rather than of each owner, so the hub pairs
identically for free. `botbowl-ui eval --sprt S0:S1[:A:B]` applies to **every** ladder rung, not
only the vs rung: each rung (and each board) is its own test, and a deterministic
scripted-vs-random rung is what lets `botbowl-ui/tests/sprt_stop.rs` pin the stop (H1 at about
31 pairs against a 400 cap). The zero-variance guard falls back to `s(1−s)` at the midpoint of
`s0`/`s1`. The printed ladder line gains `pts ± SE (pairs)`, the margin, and the SPRT verdict and
LLR, appended at the end so existing greps keep matching.

**Hub half, done 2026-10-01.** `botbowl-hub job eval --sprt` with the same syntax. The rule
travels in `EvalJobRequest` (the hub's local JSON API, serde-defaulted). Workers never see it and
`EvalGameLine` is unchanged, so **`PROTOCOL_VERSION` was not bumped**, contrary to the plan:
nothing on the worker wire changed, and a bump would have locked out remote workers for nothing.
A decided rung drops its queued games, `requeue` skips it, and `all_done` counts it as done.
In-flight games of a finished job retire without being recorded (no cancel frame exists).
`botbowl-hub/tests/eval_job.rs::a_decided_rung_stops_and_the_job_finishes_early` pins this.
`eval_summary.py` appends `[paired SE …, N pairs]` and `[SPRT … LLR …]` when present.
`anchor_curve.py` uses the pentanomial SE per generation, and pools it over the window when every
generation in it has pairs. Older reports read exactly as before. Both printers share
`LadderRow::report_line()`.

## Step 2 — the validation harness (script + a frozen pair list)

`scripts/validate_proxy.py` takes a gold file and one or more proxy result dirs and prints
one table. For each net pair it has: gold points ± SE (from step 3), and per proxy run:
verdict, point estimate, games/drives played, wall-clock. It reports:

- **sign agreement** on gap pairs (proxy verdict vs gold sign);
- **false H1 rate** on zero-gap pairs;
- **rank correlation** of proxy point estimates against gold points across pairs;
- **mean wall-clock to decision**, and its ratio to gold at the same error rates.

The pair list (`cfgs/validation_pairs.toml`, frozen, stamped with the commit) mixes:

| kind | pair | why |
|---|---|---|
| zero gap | net X vs net X | any proxy must say 0.50; this is the type-I check for free |
| zero-ish gap | two training-seed siblings of one recipe (exp052, if the nets exist) | the floor every result is compared against (plan 032 #5) |
| small gap | gen21 vs gen13 of `loopmix16x9` | plan 049: 0.50 ± 0.03 on the anchor, the hardest honest case |
| medium gap | gen13 vs gen08 | mid-curve |
| large gap | gen13 vs gen03 (the old anchor) | sanity, must be trivial |

Pairs are between **nets**, under the same `cfgs/baseline.toml`, so the harness measures the
proxy's ability to order nets. A separate config-vs-config pair (e.g. `exact_visits` vs
`baseline` on one net) is added once plan 043's A/B has a gold result, because search
changes are the case most likely to rank differently on drives than on games.

**Acceptance, stated now:** a proxy is adopted for A/B decisions if, over R = 3 runs per
pair, (a) no gap pair with gold ≥ 0.55 gets the wrong sign, (b) at most one false H1 across
all zero-gap runs, and (c) mean wall-clock to decision is ≤ ⅓ of the SPRT-on-full-games
cost on the same pairs. Failing (c) but passing (a)–(b) is still useful as a pre-screen
that decides which experiments deserve the full match.

## Step 3 — gold results (box time, one-off)

For each pair in the list: 400 paired full games at the eval budget with the current
binary, both configs `baseline`, via `job eval` so remote workers help. At the plan-030 pace
that is ≈ 16.5 h of game time per pair before parallelism; with the hub's workers it is an
overnight block for the whole list. Store under `runs/validation/<pair>/eval.games.jsonl`
and never overwrite — a binary change that touches play (engine rules, search) invalidates
gold and the directory gets a commit-stamped sibling instead.

This also yields, as a by-product, the first measured σ_pair and draw rate under pairing,
which fixes the expected-cost table above.

Run step 1's SPRT over the gold lines offline (replay the JSONL in order) to get the SPRT
full-game cost per pair without playing anything extra.

## Step 4 — paired contested drives (new rung kind)

New module `botbowl-play/src/drives.rs`:

**Position set.** A file is a recipe, not states: `{ commit, board, random_start: RandomStartConfig,
seeds: [u64], screen: { reference, playouts, kept: [{seed, attacker_td_rate}] } }`. Workers
regenerate positions with `generate_random_start` from `(config, seed)`, which is
deterministic at a commit; `GameState` never needs serialising. Attacker = the team to move
in the generated state; defender = the other.

**Screen** (`botbowl-ui positions build --reference <net> --candidates 500 --playouts 4
--keep 0.25:0.75 --out cfgs/positions/<name>.json`): play each candidate position
`playouts` times with the reference bot on both sides, drive-bounded, and keep the seeds
whose attacker scored within the band. ≈ 500 × 4 × 0.5 min ≈ 17 h of drive time, once,
parallel across workers. The band is the definition of contested; `evaluate_home` may be
used as a prefilter to skip positions with |v| > 0.8 **only if** a check on 100 positions
shows it drops nothing the playouts would have kept. Drop positions with fewer than 4 turns
left in the half: a drive the clock ends is a sample with no information.

**Playing one drive.** `play_drive_game(candidate, opponent, position: GameState,
candidate_team, seed, max_steps, trace) -> EvalGameLine`: `set_seed(seed)`, `RollDice`, loop
with `random_start_trajectory`'s stop condition (score or half changed, game over, cap).
`EvalGameLine` gains `kind: Kind { Game, Drive }` and `attacker: Option<TeamType>`, both
omitted from JSON when absent (as `board` is) and always present on the wire — **hub protocol
bump**, hub and workers update together as in plan 041. `home_score`/`away_score` hold the
score change over the drive, so `candidate_scores()` and `LadderRow::record` fold a drive
exactly like a game: +1 / 0 / −1 → 1 / ½ / 0 points, and the mirrored pair → pentanomial.

**Assignment.** `drive_assignment(set, base_seed, g) -> (position_seed, candidate_team, dice_seed)`:
pair `g/2` is position `g/2 mod |set|`, game `g` even → candidate is the attacker, odd →
defender, same dice seed across the pair. Rung name `drives(<set>)@<opponent>`; `--games`
counts drives as it counts games, SPRT and `--board-sizes` apply unchanged.

**Disagreement filter** (`--disagreement-filter`): before a pair is handed out, both bots
search the position's first decision once at the eval budget; the pair is skipped unless
the candidate's visit mass on the opponent's chosen action is < 0.5. Two searches per
position. Report the pass rate; if it is > 0.9 the filter is doing nothing and should be
left off.

**CLI:** `botbowl-ui eval --rung-kind drives --positions cfgs/positions/<name>.json ...`
and the same on `job eval`. Everything else — bot construction, `--bot-config`/`--vs-config`,
telemetry, per-game JSONL — is reused.

Cost sketch per paired sample: 2 drives ≈ 1 min vs 2 games ≈ 5 min. Whether a contested
paired drive carries as much information as a paired game is exactly what step 5 measures;
the hope is more, the floor is a fifth of the price for less.

**Done 2026-10-01, except the disagreement filter.** As built, with the deviations from the
design above:
- `botbowl-play/src/drives.rs`: `PositionSet` / `Screen` / `DriveRung`, `position_state`,
  `drive_assignment`, `play_drive_game`, `drive_rung_name`.
- `EvalGameLine` gains one trailing field, `attacker: Option<TeamType>`, rather than `kind` plus
  `attacker`: `Some` means a drive. A drive line's `seed` is the *position* seed (the dice seed is
  `base + g/2`). Hub protocol is **v10** (`Task::Eval.drives` + the new field).
- **The screen is an ordinary eval job**, not a new command. `botbowl-ui positions --board 14x7
  --count 500 --out cand.json` writes the unscreened recipe, skipping positions where the side to
  move has fewer than 4 turns left. A drive rung of the reference against itself over it, for
  example `job eval --positions cand.json --vs-games 2000` (4 playouts × 500 positions), runs on
  the hub's workers. `scripts/positions_screen.py` then keeps the positions whose attacker scored
  within the band. The same file, now carrying `screen`, is what `--positions` plays.
- `--positions A.json,B.json` on both `botbowl-ui eval` and `botbowl-hub job eval` turns every
  rung (fixed and vs) into drives, one rung per set on its own board. `--board-sizes` is ignored,
  and `--games` / `--vs-games` count drives. SPRT applies unchanged.
- Random-start positions keep their sampled MA, unlike `play_ladder_game`, which caps MA on narrow
  boards. That is the training distribution, which is the point of drives.
- Tests: `drives::tests` (assignment, regeneration, pair fold, screened subset), `eval::tests`
  (the attacker field over JSON and postcard), and
  `botbowl-hub/tests/eval_job.rs::drive_rungs_reproduce_the_single_process_drives` (workers' lines
  equal `play_drive_game` in-process, line for line).
- **Not built yet:** the disagreement filter. It is optional, and the plan says it comes last.
- Smoke result (12x5, scripted reference): 40 candidates, 42% of positions at attacker rate 0 and
  42% at rate 1, 6 kept at 0.25-0.75. A deterministic reference is bimodal, as expected. The real
  screen uses a net.

## Step 5 — validate and decide

Run on the validation pairs, R = 3 each:

- P1: SPRT on paired full games (step 1 alone).
- P2: SPRT on paired contested drives, filter off.
- P3: P2 with the disagreement filter.

Table from `validate_proxy.py`, judged by the acceptance rule in step 2. Outcomes:

| result | action |
|---|---|
| P2/P3 pass (a)–(c) | drives become the A/B default; full games on the final candidate of a line of experiments only |
| P2/P3 pass (a)–(b), fail (c) | drives pre-screen; anything that passes gets the SPRT full-game match |
| P2/P3 fail (a) on the config pair only | drives rank nets, not search changes; say so in `cfgs/README.md` |
| P2/P3 fail (a) on net pairs | the drive frame is the wrong proxy; keep P1, revisit with full-game openings (kickoff positions) |
| P1 itself needs > 400 games on the small-gap pair | the gap is below what any method resolves cheaply; that is a finding about the experiment, not the method |

Only after this: tune the ranking search budget (does 250 iters keep the P-ordering on the
pairs?) with the same harness, and consider the margin as the SPRT statistic if its SE/effect
ratio beat points in the gold data.

### Final table (2026-10-02, 3 reps; `runs/plan051_proxy/validation.txt`)

Gold is 400 full games per pair (200 per board). P1 is replayed offline from the gold lines. Cost
columns: P1 as a fraction of the full match, and P2 as a fraction of P1 (candidate MCTS
iterations).

| pair | board | gold | P1 | P1 cost | P2 verdicts | P2 mean n | P2 cost |
|---|---|---|---|---|---|---|---|
| large | 14x7 | 0.608 ± 0.029 | H1 @130 | 0.65 | **H0, H1, H0** | 263 | 0.23 |
| large | 16x9 | 0.690 ± 0.029 | H1 @42 | 0.21 | H1, H1, H1 | 117 | 0.35 |
| medium | 14x7 | 0.590 ± 0.030 | H1 @62 | 0.32 | H1, H1, H1 | 463 | 0.82 |
| medium | 16x9 | 0.618 ± 0.032 | H1 @78 | 0.39 | H1, H1, H1 | 225 | 0.39 |
| small | 14x7 | 0.515 ± 0.033 | undecided @200 | 1.00 | H0, H0, H0 | 460 | 0.27 |
| small | 16x9 | 0.438 ± 0.034 | H0 @86 | 0.42 | H0, H0, H0 | 179 | 0.32 |
| zero | 14x7 | 0.512 ± 0.029 | undecided @200 | 1.00 | H0, H0, H0 | 177 | 0.11 |
| zero | 16x9 | 0.465 ± 0.030 | H0 @90 | 0.45 | H0, H0, H0 | 76 | 0.12 |
| zero-ish | 14x7 | 0.537 ± 0.034 | undecided @200 | 1.00 | H0, H0, H0 | 181 | 0.10 |
| zero-ish | 16x9 | 0.455 ± 0.032 | H0 @32 | 0.16 | H0, H0, H0 | 310 | 1.46 |

- **P1:** 0 wrong signs, 0/4 false H1, rank correlation 0.98, at 0.52 of a fixed match.
- **P2:** 2 wrong signs (both on large/14x7), 0/12 false H1, rank correlation 0.71, at 0.29 of P1.
  That is about 7x cheaper than a fixed match.

Rep 1's large and medium runs were replayed after the SPRT-reversion hang (fixed at `b802cc4`);
the hung partials are in `hung_r1/`. The 16x9 near-even pairs read 0.44-0.47 at full gold N; the
earlier seat-bias worry faded to about 1.2 SE on the identical pair.

### Interim result and how we use it (2026-10-02, reps 1-2 of 3)

P1 (SPRT on paired full games): 0 wrong signs, 0/4 false H1, rank correlation 0.98 with gold, at
0.52 of a fixed match's cost. **Adopted for A/B decisions.**

P2 (SPRT on paired contested drives): 0/8 false H1, cost 0.27 of P1's (about 7x cheaper than a
fixed match). But one wrong sign: on **14x7, gen13 vs gen03** (gold 0.608) the drives said H0 in 2
of 3 runs, counting the run that hung. On 16x9 the same pair is H1 every time, and on 14x7 the
*medium* pair (gen13 vs gen08) is H1 every time. By the pre-committed rule, that fails (a).

**Decision (the user's, 2026-10-02): drives are the main metric for this phase.** The net is
trained on single drives, so a drive measurement measures what training optimises. Full games add
phases (kickoff returns, attrition, clock) that drive training cannot reach, and those mostly add
noise to the comparisons that matter now. Multi-drive tactics (the 2-1 grind, injuring strong
players early) are a later phase that needs game-level value targets, a training change. Three
guards go with it:
1. **Screen with the current reference, and refresh it.** A set's positions are contested *for the
   bot that screened it*, which is the point: it measures improvement exactly where the reference
   is unsure. For loop work, re-screen with the parent (each generation, or every few), or the
   positions drift easy as the net improves and the set stops discriminating. The 14x7 miss may
   partly be this: gen13 vs gen03 was judged on positions contested for gen21.
2. ~~An occasional P1 full-game check.~~ **Withdrawn by the user, 2026-10-03.**
3. ~~Confirm drive H1s and sweep winners with P1.~~ **Withdrawn by the user, 2026-10-03.**

**Drives only, until the user says otherwise (2026-10-03).** Guards 2 and 3 were never the user's;
they kept dragging the work back to full games. In this phase, experiments and the loop's
benchmark are judged on drives alone: no P1, no anchor match, no "confirm on full games" step,
and none proposed. The user decides when full games are needed. `train_loop.sh` keeps both as
opt-in knobs (`P1_GAMES`, `ANCHOR_EVERY`), off by default. A drive false H1 at alpha 5% is an
accepted cost for now.

**Open: investigate the 14x7 blind spot.** The user's intuition is that mid-drive play is what makes
a bot strong. There is no setup logic beyond defaults, so kickoff formations should not separate
gen13 from gen03. Candidates for what full games see and drives do not:
- **Attrition.** Injuries and KOs carry over between drives in a game; every drive starts fresh.
- **Clock and half management** across drives.
- **Kickoff returns** (the first turns after a kickoff are not in the random-start distribution).
- **The screen.** It used gen21 as reference and kept positions contested *for gen21*, which may be
  exactly the positions where gen03 is not worse.

The user's context, 2026-10-02: attrition and clock management cannot be *learned* from
drive-level training. A full-game win through them is a side effect of style, not something the
training can target. Setups are defaults, so formations should not separate the nets. **Kickoff
returns are the candidate that matters**, because random-start positions never include receiving
a kick, so training and drive evaluation are both blind to it. If that is where gen03 loses, the
fix is kickoff-start positions in the corpus (and an eval set), not a different eval. The user has
a setup-training phase prepared, to follow the tau and search-budget work.

Cheapest first checks: split the gold games' points by drive (the per-game lines do not hold
drives, so this needs a replay or a log), compare casualty counts between the sides in the gold
games, and re-screen with gen03 or a mixed reference.

## Not now

- **Luck covariate (CUPED-style adjustment).** Sum over rolls of actual − expected success
  per side, regress outcome on it. Reduces dice variance without changing play, but needs a
  per-roll record the engine does not keep. Add if step 5 shows dice variance, not position
  variance, dominates the residual — the gold data can tell (variance of the pair sum vs the
  pair difference).
- **Search-only probes** (agreement with a 32k-iter reference on the position set): plan 025
  machinery; a regression detector, cannot see a candidate that beats the reference.
- **Defensive puzzles** (one candidate turn from a threat position, graded by repeated
  reference replies): the drive rung with a 0–25% attacker band is the same measurement
  without new code; try that first if denial strength is the question.
- **Full-game openings** (frozen kickoff positions instead of mid-drive ones): the fallback
  if drives fail (a). Costs full-game time, keeps the pairing and screening gains.

## Order of work

1. `stats.rs` + `LadderRow` fields + `--sprt` in `botbowl-ui eval` (local only), tests.
2. Hub: rule in the job request, rung-scoped stop, protocol bump; `eval_summary.py`,
   `anchor_curve.py` read `points`/pair SE.
3. Freeze `cfgs/validation_pairs.toml`; launch the gold block on the hub (runs while 4 is built).
4. `drives.rs`: position recipe + screen, `play_drive_game`, assignment, rung kind, CLI, hub,
   disagreement filter last.
5. `validate_proxy.py`; run P1–P3; write the table and the decision under step 5 here.

Commit before each box-time step (the generator stamps the commit into every file).

## Open questions

- Is the random-start "attacker" well defined when the ball is loose (`carried_prob`
  < 1)? If not, the screen should define attacker as the side closer to the ball, or the
  position set should use `carried_prob = 1`.
- Drive length on 16x9: the 0.5 min figure is 14x7 at 1000 iters; measure once in step 4.
- Does the frozen position set need one file per board size? Yes if `--board-sizes` stays
  in the eval, since `generate_random_start` takes `board_dims`; the recipe carries it.
- Whether the hub should expose a generic job stop. Not for this plan; a rung-scoped
  stop is enough and keeps the API surface where it is.
