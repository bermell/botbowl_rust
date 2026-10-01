# Plan 051 — Fast bot ranking: SPRT on paired games, contested paired drives, and a validation harness

**Status:** Written 2026-10-01 after discussion. Not started. Steps 1–2 are pure Rust plus a
hub change and need no box time; step 3 needs one block of ground-truth games; step 4 builds
the drive rung; step 5 validates both proxies against step 3 and decides. Results go under
each step.

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
