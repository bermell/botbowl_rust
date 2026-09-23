# Plan 044 — Make `GameState`'s hash discriminate

**Status: implemented.** Follow-up to plan 043, which built the instrument that found this.

## Why

`GameState::hash` hashed a hand-picked subset of what `PartialEq` compares. For the procedure
stack — the thing that says what the engine is *about to do* — it contributed only the stack's
length and the **name** of its top frame, with a comment arguing that "collisions are corrected by
PartialEq". They are corrected: the search's answers were never wrong. Nobody had priced the
correction.

Plan 043's telemetry priced it. In a real search DAG:

- **36.8% of distinct states shared a hash** (6,607 of 17,975)
- the worst bucket held **84 distinct states on one hash**
- the search paid **4.07 full two-state comparisons per registry probe**, and **99.1% of them were
  rejected**

Under `MemoryMode::StoreState` a comparison goes through `Node::get_state()`, which **clones** the
state — so a rejected comparison is two whole `GameState` clones plus a deep compare, for nothing.

## The rule this rests on

> **Hash a field if and only if `PartialEq` compares it.**

Hashing *less* than equality compares is merely slow: the extra candidates arrive at the comparison
and get rejected. Hashing *more* is a **correctness bug** — two equal states would land in
different registry buckets and one DAG node would silently split into several, which is the
recombination invariant the whole search rests on. So the safe direction is always "skip it", and
that is what made the ablation below cheap to act on.

## Method

Guessing which fields to add would have been easy and wrong (see "What did not work"). Instead,
`botbowl-mcts/tests/hash_quality.rs` builds a corpus from the **registry of a finished search** —
by construction, exactly the population a state hash has to separate — groups it by hash, and for
each colliding group reports which publicly visible field differs.

That named the culprits directly:

| colliding states | groups | what differs |
|---|---|---|
| 4,816 (73%) | 860 | procedure-stack payload |
| 1,766 (27%) | 380 | `info.player_action_type` |
| 10 | 5 | `pending_roll` payload |
| 8 | 2 | `available_actions` |
| 7 | 3 | `info.*_available` |

## What changed

`AnyProc` and every procedure now derive `Hash` next to their existing `Eq` — which they all
already had, so no floats could be hiding — and `GameState::hash` walks:

- `info`, `home`, `away` **whole** (rather than a hand-picked field list; `player_action_type`
  alone was 27% of the collisions, and a maintained-by-hand subset is exactly how that drift
  happened — the team states also carry `rerolls`/`reroll_used`, which were missing)
- each fielded player's `id`, `position`, `status`, `used`, `moves` and `used_skills`
- `ball`, `bounce_squares`
- **`proc_stack` in full** — the 73%
- `pending_roll` in full, not just its discriminant
- `dice_mode` by discriminant only

`HashSet` fields (`used_skills`, `PlayerStats::skills`, `AvailableActions::simple`) hash through
`hash_set_unordered`, which sums per-element hashes: set *equality* is order-independent, so the
hash has to be too, or two equal states would hash differently.

**Deliberately skipped**, all safe because skipping always is: `available_actions` (derived from
the proc stack, 8 states, and it holds a `FullPitch`), `board` (a position index derived from
`fielded_players`), `board_dims` (constant per game), `dugout_players` (never separated a pair),
and the fixed per-player stat block (constant per game, implied by `id`, and `skills` is a
`HashSet` — the most expensive field available).

## Results

All measured on the same machine, alternating arms so drift hits both equally. Baseline is
`567e3de`, in a separate worktree with its own target dir.

**Discrimination** — `hash_quality.rs`, 17,975 distinct states from 12 real search DAGs:

| | before | after |
|---|---|---|
| distinct hashes | 12,618 | **17,975** |
| colliding states | 6,607 (36.8%) | **0 (0.0%)** |
| worst bucket | 84 | **1** |
| mean distinct states per bucket | 1.425 | **1.000** |

**Comparison work** — `hash_bench.rs`, 5 seeds x 8 positions x 3 decisions:

| | before | after |
|---|---|---|
| eq comparisons per probe | 4.07 ± 0.96 | **0.12 ± 0.005** (−97%) |
| eq reject rate | 99.1% | 72% |
| recombination hit rate | 0.0333 ± 0.0097 | 0.0333 ± 0.0097 |

**Wall clock** — paired on the seed, 9 rounds per arm. Pairing matters: both arms run the
*identical* search, so between-position variance (several times larger than the effect) cancels.

| seed | nodes | best-of-9 | median-of-9 |
|---|---|---|---|
| 43201 | 47,258 | 1.112x | 1.113x |
| 43202 | 47,651 | 1.084x | 1.065x |
| 43203 | 47,754 | 1.025x | 1.097x |
| 43204 | 47,487 | 1.035x | 1.084x |
| 43205 | 47,809 | 1.027x | 1.087x |

Best-of-9: mean **1.057x**, sd 0.039. Median-of-9: mean **1.089x**, sd 0.018. Pooled per-run:
9.637 ± 0.899 s before, 8.966 ± 0.961 s after (n=45 each). **5 of 5 seeds improve under both
estimators**, so call it **6–9% faster** and not more precise than that.

**Whole games** — `scripts/hash_ab.sh`, 2 rounds x 10 ladder games, heuristic, 600 iterations:

| | before | after |
|---|---|---|
| eq comparisons per probe | 1.076 | **0.248** (−77%) |
| rejected comparisons per probe | 0.983 | **0.163** (−83%) |
| recombination hit rate | 0.093 | 0.085 |

Note the regime difference: full games run many short searches with frequent rebuilds and sit at
~1.1 comparisons per probe, where the microbenchmark's three deep consecutive decisions sit at
~4.1. The improvement is large in both, but quote the right one for the question being asked.

**The wall clock from this A/B is not usable and is deliberately not quoted.** The two arms play
*different games* — 1653 vs 1335 searches in one round — because full games are not reproducible
across processes (recon_mcts registry iteration order; see plan 032). Different work cannot be
compared by total time. The paired microbenchmark above is the valid timing instrument precisely
because both arms run the identical search.

**Search output is unchanged**, which is the point of the Eq/Hash rule: node counts are identical
per seed across arms and rounds, the recombination hit rate is identical to four decimals, and
`lazy_mover_identity`'s default-on golden plus `mirror_search_exact` pass untouched.

## What did not work

The first attempt hashed *everything* `PartialEq` compares, including `available_actions` and
`dugout_players`. It removed exactly the same collisions — and was **no faster than the baseline**
(5275 ± 539 vs 5295 ± 381 nodes/s). `AvailableActions::positional` is a `FullPitch`, so hashing it
walks the whole board on every node insert; it was resolving 8 of 6,607 colliding states. Trimming
it and the other derived/constant fields is what turned a wash into a win.

The lesson is the ablation, not the field list: the collision table above says where the
discrimination is, and everything else is cost.

## Follow-ups

- The 72% residual eq-reject rate is **not** hash collisions any more — with zero colliding states
  it is hashbrown's 7-bit tag brushing, which is inherent to the table and costs one comparison
  each. `eq_hash_equal` in the telemetry separates the two if it ever needs re-checking.
- **Recombination still only hits 3–10% of probes.** That is the open question plan 043 raised and
  this plan does not answer: whether the DAG's recombination earns the registry at all. It is now
  much cheaper, which weakens the case for removing it.
- `Block` decisions never reuse the search tree (plan 043's other finding) — untouched here.
