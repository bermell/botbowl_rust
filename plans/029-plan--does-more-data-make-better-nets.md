# Does more data make better nets?

**Status:** Not started. Design only — no arm has been run.

The loop has stopped compounding. The question this plan answers is whether the
corpus is the reason. Everything needed to answer it is already on disk: eight
generations, 64 shards, 38,400 games, 6.36 GB, all post-`e107f06`. Nothing has
to be generated. What has to be built is a *fair* comparison, and three separate
things make the obvious version of this experiment unable to answer the
question. Most of this document is about those three.

## What is actually known, and why it does not answer the question

The premise as usually stated — "four consecutive generations at parity with
champion gen03 while beating the fixed rungs better than gen03 did" — does not
survive being added up. Two corrections, both from `runs/loop14x7/status.md`:

**1. The four generations are not one sample.** gen04 and gen05 were trained
with `WARM_FROM=latest` off a drifting trajectory; the operator changed it to
`champion` on 2026-09-05 precisely because of that drift, and gen06/gen07 are
the only generations produced under the current configuration
(`WARM_FROM=champion`, `EVALUATOR=nn`, `SELECT_ON=combined`). Pooled against
champion gen03, 100 games each:

| group | W-D-L | n | points | SE¹ | z |
|---|---|---|---|---|---|
| gen04 + gen05 (`WARM_FROM=latest`) | 62-40-98 | 200 | **0.410** | 0.031 | **−2.90** |
| gen06 + gen07 (`WARM_FROM=champion`) | 75-42-83 | 200 | **0.480** | 0.031 | −0.64 |
| all four | 137-82-181 | 400 | 0.445 | 0.022 | −2.49 |

¹ From the realised W/D/L, not `0.5/sqrt(N)` — a 20% draw rate cuts the
per-game variance to ~0.196, so the SE is ~10% tighter than the naive one.

So the drifting pair really was *worse* than the champion (p≈0.004) and the
current pair really is at parity (p≈0.52). The right premise is the weaker one:
**two generations at parity, no evidence of decline, and no evidence of
improvement.** The loop reproduces its champion.

**2. The fixed rungs do not discriminate.** gen07 beat gen03 on the
mcts-heuristic rung (0.967 vs 0.900) and *lost* to it on the scripted rung
(0.867 vs 0.950), at 30 games per rung, i.e. SE ≈ 0.065 apiece. One rung up, one
rung down, neither resolvable. There is no rung evidence that recent generations
are stronger against fixed opposition, and the plan should not lean on any.

**What that leaves.** Under the current configuration the loop has produced two
nets that are neither better nor worse than the net that generated their data.
That is what a saturated learner looks like, and it is also what a data-starved
one looks like, and it is also what an underpowered promotion gate looks like.
Nothing on disk separates them.

**Why "data" is the first hypothesis worth spending on.** The window is 3
generations = ~350k samples = ~10,800 games. AlphaGo Zero sampled every batch
from its most recent 500,000 games. We are ~46x short on games and the entire
gap is testable for free, because seven generations are already written to disk
and only the three most recent are ever used.

**Related work in this repo.** Plan 028's closing section sketches the cheap
version of this test (`WINDOW_GENS=1` vs `WINDOW_GENS=3`, warm-started from
gen03, same epochs) and records the memory ceiling that blocks scaling it up.
This plan keeps that skeleton and fixes three things that would have made its
result uninterpretable. Plan 027 supplies the measurement method and the
standing warning that goes with it: **`val_value` has moved opposite to strength
four separate times in this project.** No arm here is decided on validation loss.

## The corpus, and what it is not

Eight generations, 8 shards x 600 games each, verified on disk 2026-09-06:

| gen | generator | shards | jsonl | in scope |
|---|---|---|---|---|
| gen00 | heuristic only, no network | 8 | 698 MB | **no** — see below |
| gen01-gen04 | `--evaluator nn-value` (NN leaf values, **scripted** priors) | 8 each | 811-820 MB each | yes |
| gen05-gen07 | `--evaluator nn` (learned priors) | 8 each | 805-812 MB each | yes |

It is not a homogeneous corpus, and a wide window necessarily mixes regimes.
Two decisions:

**gen00 is excluded from every arm.** Its policy targets come from a scripted
prior with no network in the loop at all — a different labelling function, not
merely an older net. `window_shards()` ages it out on its own once the window
passes it, so excluding it matches what production would do anyway. It stays
useful in one respect: it is the only generation guaranteed absent from every
arm, which makes it the natural fallback source for a shared warm start if the
from-scratch arms turn out too weak to measure (see "If the arms come out too
weak"). The widest arm is therefore **gen01..gen07 = 7 generations**.

**The `nn-value`/`nn` split is not removed — it is placed on an arm boundary and
read.** Any wide window mixes regimes; pretending otherwise would answer a
question production cannot ask. So the arms are chosen so that one contrast is
pure-regime and one crosses the boundary:

| arm | generations | regime | samples² | prepared size² |
|---|---|---|---|---|
| **D1** | gen07 | pure `nn` | 116,865 | 2.4 GiB |
| **D3** | gen05-gen07 | pure `nn` | **350,595** (measured) | **7.2 GiB** (on disk) |
| **D7** | gen01-gen07 | 4 `nn-value` + 3 `nn` | 818,055 | 16.8 GiB |
| **M1** *(control, conditional)* | first 86 games of each train shard of gen01-gen07 | same mix as D7 | ~117,255 | 2.4 GiB |

² D3's figures are read from `runs/loop14x7/gen07/prepared_train/dims_16x9/manifest.json`
(`num_samples: 350595`) and `du` (7.2 GiB, of which `spatial.npy` is
7,471,880,768 B = **21,312 B/sample**, matching plan 028). Everything else
scales from the measured 116,865 train samples per generation and is an
estimate.

- **D1 -> D3** is "3x the data, same regime, same recency". The clean read.
- **D3 -> D7** is "another 2.3x, but all of it off-regime and up to six
  generations stale". If D3 > D1 and D7 <= D3, the answer is "data helps, stale
  data does not", which is a different and more actionable finding than "data
  helps".
- **M1 vs D1** holds sample count fixed and swaps a single-generation pool for a
  seven-generation mixture. It separates *volume* from *diversity*, which the
  D-arms confound by construction. Only worth running if the D-curve rises.

M1's recipe is deliberately blunt: `head -n 86 shardK.jsonl` for each of the 6
train shards of each of the 7 generations, giving 3,612 games against D1's
3,600 — a corpus of the same size drawn from seven distributions instead of one.
A jsonl line is one game, so this subsamples at game granularity with no leakage
and no code.

## Trap 1 — identical initialisation

Every arm must start from the same weights, or the comparison measures the
draw rather than the data.

**The arms train from scratch, not from `gen03.pt`.** Warm-starting from the
champion contaminates the question directly: gen03 already encodes gen00-gen02,
so a `k=1` arm warm-started from it has not seen one generation of data, it has
seen three plus one. The whole quantity under test would be pre-loaded into the
initialisation of the arm that is supposed to lack it, biasing hard toward
"more data does nothing".

**The tradeoff, stated plainly.** From-scratch is the clean scientific answer to
"does more data make a better net". Warm-start is what production does, and a
from-scratch result therefore does not transfer to the loop one-for-one: it is
possible for data volume to matter from scratch and not matter on top of a
champion, because the champion has already extracted most of what the earliest
generations contain. If the from-scratch curve rises, the production-relevant
follow-up is one warm-started pair (`--init gen03.pt`, `--lr 2e-4`, D1 vs D3),
and it should be run before `WINDOW_GENS` is changed. That is one extra match,
not a redesign.

**How identity is guaranteed — do not rely on the seed alone.** Materialise the
initialisation once and `--init` it into every arm:

```sh
$PY -m bbnn.train --data $HOLD/dims_16x9 --epochs 0 --seed 20260906 \
    --out $X/arm_init.pt          # epochs=0: builds BBNet(), saves, trains nothing
```

then every arm runs `--init $X/arm_init.pt --lr 1e-3`. `--init` loads weights
only and `--lr` is independent of it (`train.py:141-145`), so this is bit-exact
from-scratch training, not a warm start. Two consequences worth having:

- It removes any dependence on the in-flight `--seed` flag being correct about
  weight init. `--seed` is still wanted, for data order and augmentation, but it
  is no longer load-bearing for the thing that matters most.
- `train.py:162-164` prints an `epoch -1 (warm-start baseline) | val_value` line
  whenever `--init` is given. **All arms must print the identical number.** That
  is a free assertion that the initialisation really was shared, and it should
  be checked before any games are played.

**Do not chase bit-reproducibility.** The requirement is identical *initial
weights*, not identical trajectories — the trajectories must differ, because the
data differs. `torch.use_deterministic_algorithms(True)` and
`cudnn.deterministic` are not needed and cost throughput.

## Trap 2 — the steps confound

At fixed `--epochs`, a 7-generation window takes 7x the gradient updates. "More
data wins" and "more updates wins" would be indistinguishable, and plan 028's
sketch (`same epochs, differing only in corpus size`) has exactly this flaw.

**Arms are equalised on total optimizer steps.** Fix `S = 110,000` steps at
`--batch-size 32` for every arm. That is production's current step count —
gen07 ran 10 epochs over 350,595 samples = 10,957 batches/epoch = 109,570 steps
in 27 min (`status.md`, 2026-09-06 01:09→01:36) — so the D3 arm is a near-replica
of what the loop actually does, and every other arm differs from it in the data
pool and nothing else. Passes over the pool then vary as intended:

| arm | samples | batches/epoch | passes at S=110,000 |
|---|---|---|---|
| D1 | 116,865 | 3,653 | 30.1 |
| D3 | 350,595 | 10,957 | 10.0 |
| D7 | 818,055 | 25,565 | 4.30 |
| M1 | ~117,255 | 3,665 | 30.0 |

An equal-steps design also makes wall-clock equal across arms (~40 min each),
which is a convenient side effect rather than the point.

### What has to change in `train.py`

The trainer is epoch-driven (`for epoch in range(epochs)`, `train.py:166`) and
validates once per epoch (`:183-206`). Three changes, all contained:

1. **`--max-steps N`.** Make the outer loop step-driven: keep a `step` counter,
   wrap the loader in an outer `while step < max_steps`, and `break` out of both
   loops when the counter is reached. Re-iterating the `DataLoader` reshuffles,
   so an arm that makes 30 passes sees 30 different orders. `--epochs` stays and
   remains the default; `--max-steps` overrides it when given, so
   `train_loop.sh` is untouched.

2. **`--eval-every N` (mandatory, not optional).** Per-epoch validation is fatal
   here: at S=110,000 the D1 arm would produce 30 validation points and the D7
   arm 4. Best-val restore would then be choosing from wildly different-sized
   candidate sets, which is itself a confound. Validate every **2,500 steps** —
   44 points for every arm regardless of pool size.

3. **`--seed N`** — `torch.manual_seed` before `BBNet()`, a seeded `generator=`
   on the `DataLoader`, and `random`/`numpy` for completeness. `num_workers`
   is 0, so the augmentation's `torch.rand(())` in
   `PreparedDataset.__getitem__` (`data.py:66`) draws from the same global RNG
   and is covered. (Being added in parallel; this plan only fixes what it must
   cover.)

Keep the exact string `restored best-val weights:` — `train_loop.sh:557` greps
it — but report the **step**, not the epoch: `restored best-val weights: step
17500 (val_policy+val_value 1.8489)`.

### Best-val restore reintroduces the confound through the back door

This is the subtle part and it must be reported, not designed away. Best-val
restore means each arm ships the weights from its *best* checkpoint, not its
110,000th step. The small-data arm overfits sooner and is therefore restored to
an earlier checkpoint — so at ship time the arms are back to unequal steps.

Two defensible responses:

- **(a) Take the final weights at step S.** Purest "fixed steps, varying unique
  samples". But it makes the D1 arm play a memorised net, and nobody would ship
  that.
- **(b) Keep best-val restore.** This measures "the best net obtainable from *k*
  generations at a step budget of S", which is the decision-relevant quantity —
  production always restores, on `SELECT_ON=combined`.

**Run (b), and record the restored step index for every arm.** The index
disambiguates the two readings for free after the fact: if the curve rises and
D1 restored at step ~4,000 while D7 restored at ~90,000, the honest statement of
the finding is *"more data lets you train longer before overfitting"* — which is
the same lever, but it should be said that way rather than as "more data teaches
the net more".

Note also that y-flip augmentation (`data.py:16-32`, fresh flip per access)
gives repeated passes some free diversity, so the small-data arm is treated
slightly better than a strict unique-samples reading would allow. That bias runs
**toward the null**, which is the safe direction: it makes a rising curve harder
to produce, not easier.

## Trap 3 — one common validation set

Today `VAL_SHARDS` are drawn from the same window as the training shards
(`train_loop.sh:499-500`), so each arm would validate against a *different*
held-out set and the curves would not be on the same axis at all.

**The holdout already exists and costs nothing:**

```
runs/loop14x7/gen07/prepared_val/dims_16x9    # 114,372 samples, 2.4 GiB
```

It is `VAL_SHARDS = 4 7` of gen05, gen06 and gen07 — one `nn` shard and one
heuristic-hedge shard per generation, prepared 2026-09-06 at commit `3ea5238`.

**Disjointness is structural, not incidental.** Every arm trains on
`TRAIN_SHARDS = 0 1 2 3 5 6` only; shards 4 and 7 appear in no arm's training
set for any generation. And shards are seed-disjoint by construction —
`train_loop.sh:157` gives gen *G* shard *K* the seed `10000000 + G*1e6 + K*1e5`
— so this is a whole-game holdout, which is the split `train.py:116-118`
requires ("samples within a game are consecutive states, so a sample-level split
leaks").

**Its regime is deliberately the target regime**: gen05-07 are the `nn`
generations, i.e. the distribution the current champion actually plays. The D7
arm therefore validates partly out-of-distribution relative to its own training
pool, which is correct — we want the net that is best on *current* play, not the
net that best fits a seven-generation average.

**Cost.** 114,372 samples = 3,574 forward-only batches per validation pass; at
44 passes that is an estimated ~13 min per arm on top of ~27 min of training.
If that bites, a three-line `--val-limit N` (first N samples of the holdout)
brings it down; the same holdout prefix must then be used by every arm.

## Trap 4 — strength, not validation loss

`val_value` has moved opposite to strength four times in this project. Plan 027
records the mechanism: the value head's optimum sits at epoch 0-2 while the
policy head is still improving at epoch 9, and gen02 scored 0.450 against gen01
*with a better* `val_value`. Validation numbers here are diagnostics — they
choose the checkpoint within an arm, and they are reported — but **no arm is
decided on them.**

### The match design

Every arm plays the **same fixed reference opponent, D1**, at production
settings (`--evaluator nn`, `--mcts-iters 1000`), paired Home/Away, from the
**same `--seed` base** so all arms face identical situations. Scored with
`scripts/paired_summary.py` (per-pair) alongside the pooled
`(W + D/2)/N` that `scripts/eval_summary.py` reports.

Using D1 rather than gen03 as the reference is the cheapest interpretable
design: it needs *n−1* matches instead of *n*, it anchors the curve at 0.500 by
construction, and it puts every arm on a scale that reads directly as "what does
adding data to one generation buy". Its weakness is that D1 is itself one draw,
so a bad D1 shifts every point together — which affects the *level* of the curve
but not its *shape*, and shape is the question. One external calibration match
(the best D-arm vs champion gen03) anchors the family to production and is
budgeted below.

### Game counts and what they can resolve

At a 20% draw rate — measured, from the gen05/gen06/gen07 vs-champion rungs
(20, 24, 18 draws per 100) — per-game variance is ~0.196 and the SE is ~10%
below the naive `0.5/sqrt(N)`:

| games | naive SE | SE at 20% draws | resolvable at 95% |
|---|---|---|---|
| 60 | 0.065 | 0.057 | 0.113 |
| **120** | **0.046** | **0.041** | **0.080** |
| 240 | 0.032 | 0.029 | 0.057 |

Plan 027 established that 60 games could not resolve a single iteration doubling
(E2b 0.508, E2d 0.517, both indistinguishable from zero) and that a 4x span at
0.700 was needed before anything cleared significance. **120 games is the floor
here**, and it still only resolves ~0.08.

**Pre-commit to the extension rule, before seeing the data.** If a match lands
in [0.53, 0.58] — real-looking, unresolvable — the response is *more games on
that same pair*, extending to 240 with a fresh seed block, not a new arm and not
a claim. If it lands below 0.53 the arm is called flat. Deciding this afterwards
is how 0.55 becomes "a win".

Sharing a seed base across matches costs nothing and correlates the arms through
their common openings, so the D3−D7 difference is slightly better resolved than
two independent samples. Plan 027 measured the analogous within-pair gain at only
**1.07x**, so budget as if independent.

## Cost, disk, and sequencing

Provenance for each rate: prepare from `status.md` (gen07's 18 train + 6 val
shards in 24 s wall); train from gen07's 27 min for 109,570 steps; games from
gen07's eval, 190 games in 407 min at `--parallel-games 4` with the sidecar =
**2.14 min/game** (conservatively 2.4 min/game here, since 30 of those 190 were
against `random` and near-instant). Everything below marked *(est)* is derived.

| step | wall | disk | RAM |
|---|---|---|---|
| prune stale `prepared_*` from gen00, gen01 | seconds | **frees 6.1 GiB** (measured) | — |
| prepare D1 (6 shards) | ~10 s *(est)* | +2.4 GiB *(est)* | ~2.5 GB *(est)* |
| prepare D3 | **0** — reuse `gen07/prepared_train` | 0 | 0 |
| prepare D7 (42 shards) | ~60 s *(est)* | +16.8 GiB *(est)* | **~17.4 GB — blocked, see below** |
| prepare M1 (42 truncated shards) | ~10 s *(est)* | +2.4 GiB *(est)* | ~2.5 GB *(est)* |
| train one arm, S=110,000 + 44 val passes | ~40 min *(est)* | +4 MB | mmap'd, ~2 GB |
| one 120-game match at 1000 iters, x4 | ~4.8 h *(est)* | ~30 MB of logs | ~8 GB (4 workers x 2 trees) |

**Staged, so a null result stops early:**

| stage | contents | wall *(est)* |
|---|---|---|
| **1 — the screen** | prune; prepare D1; train D1 and D3; **D3 vs D1, 120 games** | **~6.2 h** |
| **2 — the wide arm** | prepare D7; train D7; **D7 vs D1, 120 games** | **~5.6 h** |
| 3 — conditional, only if 1 or 2 rises | prepare/train M1; **M1 vs D1**; **best D-arm vs gen03** | ~10.4 h |

**Stages 1+2 total ~12 h.** For scale: one generation of the loop is 14.0 h
(gen07: 407 min generate + 27 min train + 407 min eval). *The whole experiment
costs about what one generation costs, and it decides whether the next ten are
worth running.*

**Disk: not actually tight, and I am not going to pretend it is.** Free space is
74 GiB (measured, `df`). Even with every prepared dir coexisting — D1 2.4 + D3
7.2 + D7 16.8 + M1 2.4 + holdout 2.4 = 31.2 GiB — it fits, and pruning gen00's
and gen01's stale prepared dirs frees another 6.1 GiB. So prepare-train-delete
sequencing is a **precaution, not a constraint**: delete D7's prepared dir after
its arm trains, so that (a) gen08 can resume with headroom and (b) adding a k=5
or k=6 point later does not require a cleanup first. Prepared dirs are fully
regenerable from the shards, which are what must not be deleted.

**RAM is the real constraint, and it blocks D7 today.** `prepare` still buffers
the whole corpus: 21,312 B/sample x 818,055 samples ≈ **17.4 GB** against 13 GiB
available on a 15 GiB box. gen06's prepare was already OOM-killed at 14.08 GB
(`status.md`, 2026-09-05: `anon-rss:14077600kB`), and `67ff00a` streamed only
the `.npy` *writes*, not the accumulation. Without the genuinely-streaming
prepare being built in parallel, the widest feasible arm is **k=4** (467,460
samples ≈ 10.0 GB), matching plan 028's table.

- The trainer is **not** affected: `data.py:45` mmaps `spatial.npy`, so a 16.8
  GiB pool trains fine. This is a prepare-only ceiling.
- **Fallback if streaming prepare is not ready:** run Stage 1 as written and
  substitute **D4 (gen04-gen07, 467,460 samples, ~9.9 GiB)** for D7 in Stage 2.
  D4 still crosses the `nn-value` boundary (gen04 is the first `nn-value`
  generation reached going back), so the regime contrast survives; only the top
  of the volume range is lost. Stage 1 alone is plan 028's cheap version done
  properly and is worth running on its own.

**Scheduling and hygiene.** `runs/loop14x7/STOP` is present and the loop exited
cleanly before gen08 generate (`status.md`, 2026-09-06 08:24), so the box is
free now; leave the STOP file in place for the duration. Per CLAUDE.md, **commit
before running** — `prepare` stamps the git commit into every manifest and a
dirty tree produces `<hash>-dirty` stamps that cannot be resolved back to the
code. Arms write to `runs/exp-data/`, never into `runs/loop14x7/`, so a resumed
loop cannot collide with them. Each arm must export both `.pt` and `.onnx`: the
sidecar resolves a client's `--model` path to the `.pt` beside it
(`nn_server.py:181-200`) and its registry holds `--max-models 4`, which is
enough for two arms plus the champion.

## What each outcome means

**Flat — D3 ≈ D1 and D7 ≈ D1, both inside 1 SE.** Data volume is not the binding
constraint at this scale, and the premise of this plan is refuted. Three
redirects, cheapest first:

1. **The gate, not the nets.** At 100 games the vs-champion rung has SE ≈ 0.041,
   so the 0.55 threshold sits 1.2 SE above parity: a genuinely +0.03 net fails it
   most of the time. "Four rejections" may be a measurement property rather than
   a plateau. Free to check — re-score the existing per-game logs with
   `paired_summary.py` and compute the gate's power directly.
2. **Label quality, not label quantity.** Plan 028's convergence curve puts
   top-1 agreement with a 16,000-iteration reference at **0.69 at the production
   budget of 1000** — roughly 31% of policy targets disagree with what a deep
   search would choose. Fewer, better-labelled games is the opposite lever, and
   plan 028's C1/C2 arms (8000 and 16000 iterations) already test the strength
   side of it.
3. **Capacity.** BBNet is 0.48 M params. If 7x the data changes nothing, a width
   or depth sweep at fixed data is the next cheap screen, and it needs no new
   games at all — the same three matches against D1 work unchanged.

**Rising — D1 < D3 < D7.** Data is a live lever, and the next question is *which*
lever, because from the trainer's point of view window width and
games-per-generation are the same thing (both only increase unique samples in the
pool) while their costs differ by orders of magnitude:

- **Widening `WINDOW_GENS` is free.** The shards are already written; the only
  costs are prepared-dir disk and the streaming-prepare fix.
- **Doubling games/generation costs ~7 h of generate per generation** (measured:
  407-425 min for 4800 games) and hits the identical prepare ceiling, since only
  sample count matters.

So a rising curve says **widen the window first, and only buy more games if the
curve is still rising at the widest window the disk allows.** Given the 46x gap
to AGZ's replay buffer, still-rising at k=7 is the expected outcome if data
matters at all — which is exactly why the k=7 point, not the k=3 point, is the
one worth the streaming-prepare prerequisite.

**Rising to D3 then flat or falling at D7.** Volume helps but staleness cancels
it. Keep `WINDOW_GENS` at 3-4, and the only remaining data lever is more games
per generation — the expensive one. This is also the outcome that would justify
a *weighted* window (recent generations sampled more often), which is a small
change to the sampler and not tested here.

**M1 ≈ D7 > D1.** The gain is diversity, not volume: a fixed-size corpus drawn
from seven distributions is as good as seven times the data from one. Then the
cheap intervention is corpus *variety* — keep or widen the heuristic hedge, vary
the generator across shards — rather than growing anything.

**M1 ≈ D1 < D7.** Volume, straightforwardly. The simplest and most useful
version of the result.

## What this plan does not settle

- **Whether it transfers to production.** Every arm trains from scratch; the loop
  warm-starts. The follow-up is one warm-started pair (`--init gen03.pt --lr
  2e-4`, D1 vs D3) and it must run before `WINDOW_GENS` changes.
- **Whether the step budget is right.** S=110,000 was chosen to match production,
  not because it is optimal. A wide-data arm might want more steps than a
  narrow one; equalising them is what makes the comparison clean, and it is also
  what stops it answering "what is the best net you can train on 7 generations".
- **Absolute strength.** Everything except the one calibration match is measured
  against another arm.
