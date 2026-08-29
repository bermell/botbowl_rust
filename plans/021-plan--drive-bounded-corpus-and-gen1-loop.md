# Drive-bounded corpus, the eval report card, and the gen-1 self-play loop

**Status:** In progress (started 2026-07-19). Continues `plans/020-plan--first-nets-and-14x7-bootstrap.md` — read that first for the value-head diagnosis (overfitting + teacher-conditional labels) this plan acts on.

## Headline result

The first net trained on the **drive-bounded** corpus beats its heuristic-MCTS teacher **28–2 (93% win rate, 97:22 TDs) in paired ladder games** as `nn-value` (NN leaf value + scripted priors) at equal search budget. The plan-020 gen-1 gate ("head-to-head ≥ ~50% vs the heuristic bot") is met with a wide margin, and the games are TD-rich (~4/game combined), satisfying the data-health criterion. **The gen-1 self-play loop is unblocked** (one engine bug to clear first — below).

## What was built (commit `38ebe95`)

1. **Drive-bounded random starts.** Random-start trajectories end at the first score, half change, or game over. Rationale: everything past the first drive resolution is downstream of self-play (correlated states, bot-chosen post-TD kickoff formations — the exact distribution plan 019's random placement exists to avoid). One random start = one drive = one label segment = one search horizon: generation, labels, and search are now in the same frame. `drive_bounded=true` in trajectory meta.
2. **Retuned placement biases** (defaults in both `RandomStartConfig` and `BiasArgs`): ball_distance 1.30, front_line 2.20, mark_teammate/opponent 1.50, own_side 1.50, carried 0.75, line_fraction 0.80, temperature **0.60 alternating with 1.5 per game** (`--temperature2`) so the corpus mixes sharp and flat placements.
3. **`botbowl-ui eval` report card** — the low-variance instrument replacing TDs/game in stochastic self-play (which had ±0.5 noise at 12 games):
   - *Lecture battery*: every lecture × difficulty, N trials, success rate. Cells whose hard-coded full-pitch coordinates don't fit the compiled board are skipped and marked (see open issues).
   - *Opponent ladder*: full games from kickoff vs RandomBot (floor), ScriptedBot, heuristic-MCTS (bar), alternating Home/Away on shared seeds — paired comparisons across candidates.
   - Prints a table, writes JSON for cross-generation tracking.

**Perspective question (answered, nothing built):** the NN already always sees the mover "as Home" — `perspective.rs` canonicalises to mover-attacks-toward-x=1 (Away states x-mirrored, involutive) at both prepare and inference; value is mover-centric, sign-flipped outside. Home/away variance never reaches the net.

## The drive-bounded corpus (`data/board_14x7_db/`, commit-stamped `38ebe95`)

~7.8k drive-bounded games (8 shards, heuristic@1000 iters, new biases): 152k samples (19.6/drive), 0.79 TDs/drive, 21% scoreless drives, value targets **31.4 / 33.5 / 35.1%** (−1/0/+1) — near-perfect class balance, temp split exactly even. Shard 7 held out as val.

Training (10 epochs, best-val restore): optimum at **epoch 2, val value-MSE 0.390** (val distribution differs from the old corpus — not directly comparable to plan 020's 0.359). Overfitting onset still visibly early → best-restore is doing real work every run.

## Report cards (30 paired ladder games per rung, 14x7, 1000 iters)

| candidate | vs random | vs scripted | vs mcts-heuristic |
|---|---|---|---|
| mcts(heuristic) — reference | 1.00 (TD 147:1) | 0.37 (TD 57:93) | 0.40 (mirror) |
| **mcts(nn-value: bbnet_14x7_db)** | 1.00 (TD 154:1) | **0.50** (TD 88:77) | **0.93** (W28 D0 L2, TD 97:22) |
| mcts(nn: bbnet_14x7_db) | 0.97 (TD 132:0) | 0.43 (TD 74:41) | **0.87** (W26 D3 L1, TD 73:11) |

(The `nn` card is from the re-run — the first attempt crashed; see open issues. Learned priors are now only a mild drag vs scripted (−6/−7 pts), consistent with "priors lag the value head by one generation"; nn-value remains the gen-1 generator.)

Attribution (all stacked in one generation, same architecture and search): drive-bounded labels (a sample's target is exactly its own drive, no cross-drive bleed, no post-TD formation states), tuned biases + temperature mixing (more realistic and more varied positions), best-val early stopping (plan 020's single biggest lever), and per-drive value backfill (plan 020, already in).

## Learnings

1. **Label frame purity beats label volume.** 152k drive-bounded samples produced a far stronger evaluator than 520k full-game samples (plan 020's gen0c barely moved). Data *shape* was the bottleneck, not data *scale*.
2. **The value head alone carries the win**: the 93% is `nn-value` — scripted priors, NN values. Consistent with plan 020's hybrid diagnosis (priors were never the problem).
3. **The eval harness pays for itself immediately** — the same net that looked mediocre under 12-game TDs/game noise is unambiguously strong under 30 paired ladder games; and the reference card exposed real anomalies (below) that free-play stats hid.
4. **ScriptedBot is the strongest fixed opponent** (beats heuristic-MCTS 18–11; the new net only reaches 0.50 vs it). It should be treated as the real ladder bar, and it's a candidate data source if we ever want a stronger teacher.
5. **Heuristic mirror match came out 0.40, not ~0.50** — ✅ **SETTLED: it is a genuine Home/Away asymmetry, not noise.** The plan-022 pre-flight ran the 100-game mirror (2026-08-28): the physical Home team took **0.645 of points (W57 D15 L28)**, p ≈ 0.003 clustered on the 50 seed-pairs. Two engine bugs verified by code reading — a kickoff-aim off-by-one (the only x-asymmetric expression in the engine) and an inverted post-touchdown kickoff that makes the *scorer* receive again. In-drive play is exonerated at n=4800. Full record, including what was ruled out and the deferred experiments: **`plans/023-idea--home-away-side-bias.md`**.

## Open issues

1. **Engine OOB panic under NN priors** — ✅ **FIXED (commit `d386dc0`)**. The looped crash hunt caught it: `Bounce::step` checked occupancy (`board[new_pos]`, raw physical index) *before* `is_out`, so a ball bouncing outward from a border-ring square (deviating kicks / scatter leave it there transiently) indexed one row past the array. Also fixed the follow-on: the throw-in origin is now walked back onto the pitch (ThrowIn's direction table panics on ring squares). Regression test: `bounce_off_border_square_throws_in_without_indexing_oob`. The bug existed on the full pitch too — third instance of "the small board is a fuzzer."
2. **Lectures are full-pitch-only** (hard-coded coordinates up to x=25/y=13) → the battery is skipped entirely on 14x7. Board-relative lecture setups would make the battery live on every tier.
3. **Shard shortfall**: ~7.8k of the planned 9.6k games materialized (some shards ended early; same OOM-adjacent machine pressure as plan 020's kills is suspected). Non-blocking — corpus is healthy — but shard logs deserve a look before the next big run.

## Next steps (proposed order)

1. **Fix the y=9 OOB panic** (backtrace → guard → regression test), then re-run the full-`nn` report card — completes the nn vs nn-value comparison on the new net.
2. **Mirror-match sanity run** (heuristic vs heuristic, 100 games) to settle the 0.40 anomaly before trusting fine-grained ladder deltas.
3. **Start the gen-1 loop on 14x7** with `nn-value` as the generator: → **automated as `plans/022-plan--weekend-training-loop.md`** (`scripts/train_loop.sh`, net-vs-net eval rung, promotion gate, mirror-match pre-flight).
   - Generate a drive-bounded corpus with `--evaluator nn-value --model bbnet_14x7_db.onnx` (mixed with ~30–40% heuristic games as hedge, per plan 020's mixed-transition design).
   - Train gen-1 net (best-val restore, val = fresh held-out shard from the *new* corpus).
   - Report card vs: heuristic reference, gen-0 net (`bbnet_14x7_db`), ScriptedBot. **Promotion gate: ≥55% vs gen-0 net.**
   - Iterate. Keep every generation's model + report JSON for a strength ladder over time.
4. **Once gen-1 promotes:** revisit NN priors (the fixed `nn` mode) — with a strong value head, learned priors may now add value; measure, don't assume.

## Next-next (unordered, from plan 020 + new)

- **Board-relative lecture setups** — makes the battery live on small tiers; also unlocks lecture-based *capability* tracking per generation.
- **Solved-root exact value targets** (still pending from plan 020; may matter less now that labels are drive-pure, but cheap to test).
- **Scripted-bot rung analysis**: why does it beat heuristic-MCTS? Its TD-attempt thresholding may encode play patterns worth stealing for priors.
- **Home/Away asymmetry audit** — mirror run confirmed the bias; audit done, fixes and follow-up experiments tracked in `plans/023-idea--home-away-side-bias.md`.
- **8x3 loop closure**: the small board never got its gen-1 (pure-td data → net → self-play); cheap to run end-to-end as a full-loop rehearsal.
- **Next tier (20x11, 7 players)** once the 14x7 loop shows compounding gains; expect a new bug harvest (every new board size is a fuzzing campaign).
- **MA as curriculum knob**; **weight decay** to push the early-stop point later; **NN generation cost** (~2–4× heuristic) if gen-2 wants 10k+ games.

## Cross-references

- plan 020 — value-head diagnosis, scaling probe, switch criteria (gen-1 gate now met).
- plan 019 — random-start generator (now drive-bounded, retuned defaults).
- plan 017 — architecture, perspective canonicalisation, progressive tiers.
