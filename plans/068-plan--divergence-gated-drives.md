# Plan 068 — Divergence-gated drives: only score the pairs where the bots play differently

**Status:** built and measured 2026-10-21 (exp079); behind `--drive-gate`, off by default. Adoption by
the loop is the user's call (§8).

**Headline.** The gate works and is free: a move fingerprint on each drive line marks the pairs that
were one game played twice, which score exactly ½. In a null control (gen48 vs gen48) every such
pair did score ½, and the rest scored 0.485 ± 0.061. But it buys no compute. At 1000 descents the
search disagrees with *itself* at 4-6% of decisions, so 87-90% of a net's 20x11 pairs against itself
diverge (65% on the shorter 16x9 drives). A one-generation step (gen47 vs gen48) diverges exactly as
often. No cheap prescreen (bare policy, 64 descents) predicts the rest. The SPRT on divergent pairs
is the ordinary SPRT with `s1` scaled by `p_div`, with no efficiency gain. On gen45 vs gen48 (16x9)
both tests said H0 (at 36 and 39 pairs). Fixed on the way: Gumbel-root ties now follow a
deterministic `tie_break` (the null floor drops from 6.3% to 4.1% per decision), and the forward memo
could serve one net's evaluation to another on a shared thread. What would make the gate pay is an
order-independent recon_mcts backprop (§3, §4).

The user (2026-10-21): "that feature for the drive benchmark that checks if the networks take
different moves at all for each other and only play out the drives where it does". The loop's
reference is about to move to a much closer net (v9 gen48), so drive scores will sit near 0.5, and the
8-vs-16-playout A/B (exp078, plan 067) compares two near-identical one-step nets.

## 1. What a drive pair is today

`drive_assignment` (plan 051) gives games `2k` and `2k+1` one position and **one dice seed**, the
candidate attacking in the first and defending in the second. The dice are the engine's own stream
(`state.set_seed`). Bots search on clones, so they never draw from it. Two games of a pair whose bots
choose the same move at every decision are therefore **the same game**, with the sides' labels
swapped: the pair scores exactly ½ (1 + 0, ½ + ½ or 0 + 1 half points). The pairing is already
common random numbers for the dice. Nothing about that needs building.

Bots are seeded per *bot* (`dice ^ CANDIDATE_SEED_MIX`, `^ OPPONENT_SEED_MIX`), so the attacker of
game `2k` and the attacker of game `2k+1` draw different search noise. Under the eval preset
(`chance_all_partial`, Gumbel m = 16, `gumbel_scale = 0`) the seed reaches nothing (only root noise
and random setup formations read it, and drives contain no setup), so this matters only for noisy
presets.

**A lockstep run is the normal pair, interleaved.** Ask both bots of the side to move at every
decision, play the move while they agree, and split at the first disagreement into the two games.
With four bot instances, one per (bot, role), each instance sees exactly the states its counterpart
sees in the normal pair, so it builds the same trees and makes the same moves. It costs the same two
searches per decision as the normal pair. **So lockstep buys no compute.** It identifies the
exactly-½ pairs and where each pair diverged. The first is free without it: fingerprint each game's
moves, and equal fingerprints within a pair mean one game played twice.

## 2. Measurements (exp079)

`botbowl-ui drive-agree` (new) runs the lockstep per position and stops at the first disagreement.
It records every decision on the shared prefix: fan, search agreement, bare-policy argmax agreement,
and whether each search played its own argmax. Before the full run it also runs a policy-only and a
64-descent lockstep from the same position and dice (the prescreens). Positions are the first 40 of
`contested_20x11_gen43`; the dice are pair `i`'s at `--seed 0`, as in the loop's rung. Search is
1000 descents, in-process, 16 streams, nice 19, beside the live loop (mc-label phase). The opponent
is always gen48. `scripts/exp079_measure.sh`, `scripts/drive_agree_summary.py`, outputs in
`runs/exp079/agree_*.jsonl`, commit e7407b0. The 16x9 set was dropped for box time: the loop's
generate phase was due and the box has 16 GB.

"Decisions" are searched decisions (fan > 1) on the shared prefix, the diverging one included, so
`1 − agreement` is the per-decision hazard of a divergence.

| pair (preset) | never diverges | agreement per decision | first decision differs | median first div. | policy argmax agrees | P(search differs \| policy agrees) | P(search differs \| policy differs) |
|---|---|---|---|---|---|---|---|
| gen48 vs gen48 (shipped) | 0.100 ± 0.047 (4/40) | 0.937 ± 0.010 (575) | 3/40 | step 10 | 1.000 | 0.063 | — |
| gen48 vs gen48 (det) | 0.125 ± 0.052 (5/40) | 0.959 ± 0.007 (855) | 0/40 | step 21 | 1.000 | 0.041 | — |
| gen47 vs gen48 (det) | 0.125 ± 0.052 (5/40) | 0.933 ± 0.011 (522) | 5/40 | step 6 | 0.956 ± 0.009 | 0.056 | 0.30 (7/23) |
| gen45 vs gen48 (det) | 0.025 ± 0.025 (1/40) | 0.905 ± 0.015 (409) | 6/40 | step 9 | 0.956 ± 0.010 | 0.084 | 0.33 (6/18) |

Each search plays its own policy argmax at 84-88% of decisions under every pair.

Prescreens (the same lockstep with cheap bots, from the same position and dice). A prescreen is
useful only if its "never diverges" picks out the positions where the full search never diverges:

| pair | policy prescreen: never / of which full never | 64 descents: never / of which full never | full never |
|---|---|---|---|
| gen48 vs gen48 (shipped) | 40 / 4 | 17 / 3 | 4 |
| gen48 vs gen48 (det) | 40 / 5 | 26 / 3 | 5 |
| gen47 vs gen48 (det) | 3 / 0 | 3 / 1 | 5 |
| gen45 vs gen48 (det) | 4 / 0 | 3 / 1 | 1 |

The drives that never diverge are mostly short ones (8-25 steps, a quick score). A few long ones
(58-73 steps) do too: one 66-step drive never diverged in the null runs or in gen47's. Cost: about
12 s per search per stream at 16 streams beside the loop (24 s per lockstep decision). The policy
prescreen takes 1-4 s per position and the 64-descent one 23-101 s, against 244-511 s for the full
run to its first divergence.

**Reading.** A one-generation step (gen47) diverges in exactly as many drives as a net against
itself (35/40). Its per-decision hazard is 6.7% against the floor's 4.1%. Three generations
(gen45) diverge in 39/40. Where the two policies agree, the searches still differ 4-8% of the
time. Where they differ, about a third of the time. So the nets' policy difference shows up, but
on top of search noise of the same size.

## 3. Determinism: a net disagrees with itself

"Would the two nets play the same move" means something only if a bot's move is a function of the
state. It is not.

- **Exact ties resolve in `HashMap` order.** The shipped `tie_break = "hash"` breaks exact ties in
  each tree's own child-map order, random per map. The Gumbel root (`Halving`) sorts stably by
  `g + ln prior`, so among equal logits its considered set, the order its survivors are searched
  in, and its pick all follow that order. **Fixed for any deterministic tie-break:** with
  `tie_break != hash` the root's children are ordered by the tie-break first. `hash` is unchanged
  (`a_deterministic_tie_break_makes_two_gumbel_bots_agree`). `cfgs/chance_all_partial_det.toml` is
  the eval preset with `tie_break = "mover"`.
- **recon_mcts backprop is order-dependent (not fixed).** `backprop_scores` walks a heap ordered by
  depth only. Equal-depth nodes come out in insertion order, which is each node's `parents`
  `HashSet` order. In a recombined DAG a transposed child can sit at its parent's depth. The
  visited set then keeps the parent from being revisited, so it can keep a stale value. A single
  thread with batch-of-1 inference (bit-identical numerics) and the mover tie-break still split:
  position 0 of the 20x11 set diverged at step 4, then at step 7 on a rerun. Position 2 split at
  step 59, then never in 112 steps.
- **In-process inference** is bit-identical within a bucket and ≤1e-5 across buckets (plan 066).
  At 16 streams batch sizes vary. This is not separated from the above; both are inside the floor.
- **A memo bug, fixed on the way:** `NnEvaluator`'s `LAST_FORWARD` was keyed by state only. On a
  thread shared by two nets, the second could be served the first one's forward. That happens on
  every eval drive thread (candidate and opponent alternate there), whenever the opponent's root
  forward is the state the candidate's search evaluated last. It is now keyed by evaluator too
  (`tests/memo.rs`).

The floor, gen48 against itself, 20x11, 40 positions:

| preset | never diverges | agreement per searched decision | first decision differs | median first divergence |
|---|---|---|---|---|
| `chance_all_partial` (shipped) | 0.100 ± 0.047 | 0.937 ± 0.010 (575) | 3/40 | step 10 |
| `chance_all_partial_det` | 0.125 ± 0.052 | 0.959 ± 0.007 (855) | 0/40 | step 21 |

The searches overrule their own policy argmax at 15% of decisions. Those are decisions with a near
tie between moves, and about 4% of all decisions (6% shipped) flip under nothing more than search
order. A drive has 15-60 searched decisions, so **87-90% of a net's pairs against itself diverge.**

## 4. Design

Built:

1. **Common dice: already there.** Nothing to change.
2. **The gate is a fingerprint, not a lockstep.** `play_drive_game_gated(.., gate = true)` hashes
   every move it plays (`MovePath`: FNV-1a over `(kind, type, x, y)`, no std hasher, so two workers
   on any platform agree) into `EvalGameLine.path`. The pair fold (`LadderRow::record`, the same
   half-pair stash on the ui and on the hub) calls a pair **identical** when both halves carry the
   same path. Cost: one hash per move. The games are played exactly as today, by today's code path,
   on any worker. A lockstep player would cost the same two searches per decision and needs both
   games on one thread, so it stays a measurement tool (`lockstep_drive`, `drive-agree`).
3. **Bots seeded by role under the gate.** The attacker of both games draws `dice ^
   ATTACKER_SEED_MIX` and the defender `dice ^ DEFENDER_SEED_MIX`: common random numbers for search
   noise too. Under the eval preset (`gumbel_scale = 0`) the seed reaches nothing in a drive, so
   this changes no move. It is there so a noisy preset is not divergent by construction.
4. **A deterministic preset to go with it:** `cfgs/chance_all_partial_det.toml`, plus the Gumbel
   root fix (§3). It takes the null floor from 6.3% to 4.1% disagreement per decision and moves the
   median first divergence from step 10 to step 21.
5. **Two modes.** `--drive-gate report` adds the split to the report and leaves the rung's SPRT
   on every pair, so decisions are exactly today's. `--drive-gate sprt` runs `--sprt` on the
   divergent pairs and reports the all-pairs test beside it.

Not built:

- **A prescreen.** No cheap test predicts which drives never diverge (§2). The bare policy never
  diverges for a net against itself, where the search diverges in 35 of 40 positions. For gen47
  against gen48 it diverges in 37 of 40, and none of its 3 "never" positions is one of the 5 where
  the search never diverges. A 64-descent lockstep fares no better (gen47: 1 of its 3 "never"
  positions is a real one). The positions that never diverge are mostly short drives that end in a
  quick score, and nothing cheap tells which. Skipping by prescreen would throw away divergent,
  informative pairs and keep none of the identical ones.
- **A first-decision filter** (plan 051's "disagreement filter"). On gen47 against gen48 the first
  searched decision differs in 5 of 40 positions (0 of 40 for gen48 against itself), so it does
  find positions where the nets differ. But the other 35 positions diverge later in 30 cases, and
  dropping those discards information. Unlike the identical-pair gate, it can change the population
  the score is measured on, so it is not sign-safe.

**What the gate buys, then:** exact accounting (which pairs carried information, how often the
bots differ, how big the effect is where they do) and a hypothesis rescaled by `1 / p_div`. It
buys **no compute**: every pair is played in full, and the identical ones can only be known after
both games are played. With `p_div ≈ 0.875` the rescale is 1.14, so `sprt` mode on divergent pairs
at 0.5:0.55 is roughly the all-pairs test at 0.5:0.544. The user's premise, that close nets play the
same moves through most drives, does not hold for this search at 1000 descents. Near-ties are
everywhere, and search noise alone flips 4% of decisions.

**The lever that remains is search determinism.** With backprop made order-independent (a
deterministic parent order in recon_mcts, §3), the floor would drop toward the net-induced rate.
gen47's hazard is 6.7% against the floor's 4.1%, so the net-induced rate lies between about 2.6%
and 6.7% per decision. At the low end a 20-decision drive would stay identical about 60% of the time.
That is a recon_mcts change on the loop's search, with its own byte-identity gate. It is not done
here.

## 5. Statistics

- **No bias in the sign.** An identical pair scores exactly ½, so over every pair
  `s_all − ½ = p_div (s_div − ½)`. Dropping identical pairs rescales the effect by `1/p_div` and
  cannot flip it. The implied score `½ + p_div (s_div − ½)` *is* the all-pairs mean: the gate loses
  nothing that today's rung reports.
- **The filter never looks at an outcome.** "Identical" is a property of the two move sequences.
  It is correlated with the drive (a quick touchdown leaves fewer decisions to differ on), and that
  is fine: the dropped pairs are exactly ½ whatever they are.
- **Symmetric in A and B.** Swapping the bots swaps the two games of every pair, and the event "the
  two games are one game" is unchanged. Role seeding (§4) keeps that true for noisy presets: the
  attacker's noise depends on the role, not the bot.
- **Valid with noisy bots.** The gate looks at the moves actually played, so a bot that disagrees
  with itself only makes more pairs divergent (a larger `p_div`, less to gain), never a wrong
  answer. Determinism is about efficiency, not correctness.
- **What the SPRT means.** On divergent pairs, `--sprt 0.5:0.55` tests "when the two bots play
  differently, the candidate scores 0.50" against "0.55". With `p_div` below 1 that is a smaller
  unconditional effect, `0.05 p_div`: the gate turns "is it better by 0.05 per drive" into
  "is it better by 0.05 when it plays differently". H0 still means "no better", because `s_div = ½`
  iff `s_all = ½`. Divergent pairs arrive i.i.d. (positions are cycled as today), so the sequential
  test is valid on the subsequence. The pair count it reports is divergent pairs; `own_decided_at`
  counts every pair played.
- **The gated SPRT is the ordinary SPRT with `s1` scaled by `p_div`, at the same cost.** Write
  `p = p_div`, `δ = s_div − ½` and `σ_d²` for the variance of a divergent pair's score. The
  all-pairs mean is `½ + pδ` and its variance about `p σ_d²` (identical pairs sit at exactly ½).
  The normal-approximation LLR drift per pair played is then the same for both tests:
  - the all-pairs test at `s0 : s0 + p (s1 − s0)`;
  - the divergent-pairs test at `s0 : s1`, counted per pair of every kind:
    `p (s1 − s0)(2δ − (s1 − s0)) / (2σ_d²)`.

  So `--drive-gate sprt` at 0.5:0.55 *is* today's test at 0.5:(0.5 + 0.05 p_div), with `p_div`
  estimated as it goes instead of fixed in advance. There is no efficiency to gain: identical
  pairs carry no information about *which* bot is better, but they are evidence that the
  unconditional effect is small, and the all-pairs test already uses that.
- **Hence on a true null the gated test is slower, by about `1/p²`.** At the same thresholds it
  needs about `1/p²` times as many pairs as the all-pairs test to reach H0. On the 16x9 null arm (p = 0.65) Wald's
  projection at the observed moments is 69 pairs all-pairs against 141 gated. On a pair whose
  conditional effect is `s1`, the gated test reaches H1 where the all-pairs test, at `p < ½`,
  drifts to H0. That is the hypothesis change, not a speedup. The report gives both tests (`other`)
  so they can be compared on every run.

## 6. Flag and output

`botbowl-ui eval` and `botbowl-hub job eval`: `--drive-gate report|sprt` (requires `--positions`).
Unset, a drive line and a report are byte-identical to today's (`path` and `gate` are omitted from
JSON when absent). On the hub wire the gate needs **protocol v21** (`DriveRung.gate`,
`EvalGameLine.path`); the mode rides only in the hub's `EvalJobRequest.drive_gate`.

The rung's report line gains a trailing segment, after the SPRT, so existing greps still match.
Real output from the CPU smoke (`scripts/exp079_gate_smoke.sh`, scripted against random and scripted
on the 16x9 set):

```
  ladder  vs random drives(contested_16x9_gen04g)@16x9/6 ... pts 0.917 ± 0.048 (6 pairs)  margin +0.83 ± 0.11  SPRT(0.5:0.55) Undecided LLR 8.46  gate report p_div 1.000 ± 0.000 (6/6) s_div 0.917 ± 0.048 implied 0.917 ± 0.048 other Undecided LLR 8.46
  ladder  vs scripted drives(contested_16x9_gen04g)@16x9/6 ... pts 0.500 ± 0.000 (6 pairs)  margin +0.00 ± 0.25  SPRT(0.5:0.55) Undecided LLR -0.03  gate report p_div 0.000 ± 0.000 (0/6) s_div 0.500 ± 0.000 implied 0.500 ± 0.000 other Undecided LLR 0.00
```

(Undecided at LLR 8.46: the 8-pair guard.) Under `sprt` the segment ends with `decided@N` once the
rung's own test has crossed, and `[k mismatched, j unknown]` appears if either count is nonzero.

`report.json`'s row gains `gate`: `mode`, `identical`, `divergent` (pentanomial), `mismatched`
(identical pairs that did not score ½; must be 0), `unknown` (a half without a fingerprint),
`p_div`/`p_div_se`, `s_div`/`s_div_se`, `implied`/`implied_se`, `other_llr`,
`other_verdict_now`, `other_decided_at` ([verdict, pairs]) and `own_decided_at` (pairs of every
kind when the rung's own test decided). Under `sprt`, `sprt.pairs` counts divergent pairs. The
`gate` block has no key named `verdict`, so `train_loop.sh`'s `grep '"verdict": *"H1"'` sees only
the rung's own test. `scripts/eval_summary.py` (the loop's status line) appends
`[gate MODE p_div … s_div … implied … other …]`, and `scripts/drive_curve.py` appends
`(p_div … s_div …)` to each point. `scripts/gate_replay.py eval.games.jsonl` replays both
sequential tests in pair order from the per-game lines, per rung.

`train_loop.sh`: `DRIVE_GATE=report|sprt` (default empty = off) passes `--drive-gate` to the drive
rung only (not the origin match). It wants `EVAL_BOT_CONFIG=cfgs/chance_all_partial_det.toml`.

## 7. Validation

Real drive rungs through `botbowl-ui eval`, the rung's own code path, gated `report`: the first
positions of `contested_16x9_gen04g`, `--seed 0`, `chance_all_partial_det`, 1000 descents,
in-process, beside the live loop (`scripts/exp079_validate.sh`, `exp079_phase2.sh`; outputs in
`runs/exp079/val_*`). The arms ran at a fixed size (no SPRT stop) and were stopped by box time, not
by a verdict. `scripts/gate_replay.py` replayed both sequential tests from the per-game lines in
pair order. Throughput beside the loop's mc-label phase was about 1 drive per minute at 16 streams
(the null arm) and 0.6 at 8 (the gated gen45 arm, which ran alone: the ungated arm beside it failed
to load with a CUDA out-of-memory on the shared 6 GB card).

| arm | pairs | identical | p_div | s_div | all-pairs (= implied) | mismatched | SPRT 0.5:0.55 all pairs | SPRT 0.5:0.55 divergent |
|---|---|---|---|---|---|---|---|---|
| null: gen48 vs gen48 | 26 | 9 | 0.654 ± 0.093 | 0.485 ± 0.061 | 0.490 ± 0.040 | 0 | undecided (LLR −1.10; Wald ~69 pairs to H0) | undecided (LLR −0.54; Wald ~141) |
| gen45 vs gen48 | 56 | 7 | 0.875 ± 0.044 | 0.439 ± 0.033 | 0.446 ± 0.029 | 0 | **H0 at 36 pairs** | **H0 at 39 pairs** |

- **Null control passes.** s_div is 0.485 ± 0.061, consistent with ½. Every identical pair scored
  exactly ½ (0 mismatched), which is the gate's soundness check.
- **The real pair agrees with the loop's curve.** gen45 is worse than gen48 here (0.446 ± 0.029).
  Against the d1k reference on this set the loop measured gen45 at 0.565 and gen48 at 0.593. Both
  tests say H0, and the gated one needed 3 more pairs (39 against 36). At the final moments Wald
  projects 41 against 35. That is the direction of §5's H0 cost, smaller than its `1/p² = 1.3` at
  p = 0.875, because here the effect is not zero: s_div is below ½ and pushes both tests toward H0.
- **16x9 drives are shorter**, so more pairs are identical there than on 20x11. The null control
  gives 35% identical on 16x9 against 12.5% in the 20x11 lockstep. gen45 against gen48 is 12.5%
  identical on 16x9 (7/56), against 2.5% (1/40) in the 20x11 lockstep.
- **Compute per pair is unchanged by the gate:** 46-52 candidate searches per pair here, about
  100 searches per pair for both bots (the loop's gen48 rung: 28.8 candidate searches per 16x9
  drive and 33.6 per 20x11 drive, i.e. ~115 and ~134 per pair). The fingerprint is one hash per
  move.
- **Ungated arm: not run.** It failed to load beside the gated arm (CUDA out of memory on the
  6 GB card the loop shares). The loop's train and generate phases followed, and I would not leave
  a 1.6 GB GPU process beside the trainer, nor unattended beside a generate phase that needs ~10 GB
  of host memory. What it would have checked is close to tautological. An ungated drive *is* the
  gated one minus the hash (`a_gated_drive_fingerprints_its_moves_and_an_ungated_one_is_unchanged`
  pins the ungated line byte for byte), and role seeding reaches no move under this preset. The
  gated arm's all-pairs score, 0.446 ± 0.029, is therefore today's rung's measurement of this pair.
  An independent replicate costs about an hour per 30 pairs on a quiet GPU:
  `ARM=g45plain CAND=gen45 REF=gen48 GATE= SPRT= CAP=120 PARALLEL=16 scripts/exp079_validate.sh`,
  then `scripts/gate_replay.py runs/exp079/val_g45plain/eval.games.jsonl`.

## 8. What the loop would need

- **Merge, rebuild, relaunch, allowlist.** Protocol v21 locks out workers on an older commit, so
  helper workers update with the hub (as at every bump). Any commit made while the loop runs must
  be appended to `hub-allowed-commits.toml`.
- **`DRIVE_GATE=report` costs nothing and changes no decision.** It adds `p_div`/`s_div` to every
  drive verdict. It is the safe first step.
- **`DRIVE_GATE=sprt` changes what H1 means** ("better by 0.05 *when they differ*"). At
  `p_div ≈ 0.88` that is close to today's test, so the gain is small. It is the user's call.
- **`EVAL_BOT_CONFIG=cfgs/chance_all_partial_det.toml`** changes the benchmark's bot (exact ties
  only), so drive points before and after the switch are not strictly comparable. Expect the same
  strength. That has not been measured head to head.
- **The memo fix and the Gumbel tie-break fix ship with the merge either way.** The memo fix
  changes only threads that host two nets: eval rungs, never generation or mc-label. The Gumbel change is inert under the shipped `tie_break = "hash"`.

## 9. Risks

- **A gated pair that is not one game twice.** If the two games did not share dice (a future
  change to `drive_assignment`, or a bot that draws from the state's RNG), equal fingerprints
  would not imply equal outcomes. `gate.mismatched` counts identical pairs that did not score ½. It
  is 0 in every run here, and a nonzero count should stop the rung being read.
- **A fingerprint collision** (two different 64-bit FNV paths) would call a divergent pair
  identical. At these pair counts the chance is about 1e-15.
- **`sprt` mode on a near-zero `p_div`** (a fully deterministic search on two nearly equal nets)
  would wait a long time for divergent pairs. The rung's cap still ends it, and the report then
  says how few pairs differed.
- **The deterministic preset is not deterministic.** The backprop order (§3) remains, so the null
  control's `p_div` is about 0.88, not 0. The gate stays correct with noisy bots; it just drops
  less.
