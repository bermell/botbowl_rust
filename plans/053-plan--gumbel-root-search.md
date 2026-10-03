# Plan 053 — Gumbel root search: sequential halving over the best few root moves

**Status:** Started 2026-10-03. Step 1 (implementation) and step 2 (measurements) below; results
go under each step. Drives only (plan 051): no full-game checks unless the user asks.

## Why

- **exp057:** on 16x9 mid-turn roots (about 98 legal moves) the root decision does not converge
  below 4000 descents. Top-1 agreement with a 16000-descent reference is about 0.45 and flat from
  125 to 2000, rising only at 4000.
- **exp058:** gen21 at 4000 descents beats itself at the loop's budget on 16x9 drives, 0.576 H1.
- **The cost:** 4000 descents is about 15 h per generation on the training box, so the loop runs
  1000 everywhere (plan 049).
- **exp059:** the policy target and the value head are not what holds the loop at gen21's level.
  The search is the next candidate.

PUCT spends a small budget thinly over a wide fan. With NN priors (softmax × n) the tail gets a
single FPU visit and is never revisited, while the head of the prior soaks up the rest. The
search rarely compares the top few candidates against each other with enough descents to tell
them apart. Sequential halving (Danihelka et al. 2022, "Policy improvement by planning with
Gumbel", the root half of Gumbel MuZero) does exactly that with a fixed budget:
1. Take the top m actions by `g + logit`, where g is Gumbel noise (0 when playing
   deterministically).
2. Split the budget evenly over them in log2(m) phases.
3. After each phase, keep the better half by `g + logit + σ(q̂)`.
4. Play the survivor with the highest score.

Our cq policy target is already the other half of that paper, its completed-Q policy target, so
the training side needs nothing new.

## Design (step 1)

- **Root only.** Below the root the search stays PUCT, exactly as shipped. The paper also
  replaces non-root selection with a deterministic rule; that is a possible later step.
- **The search loop runs the halving.** `MctsBot::run_search` drives `tree.step()` one descent at
  a time and, before each step, names the root child that descent must take. `BloodBowlDynamics`
  carries a `ForcedRoot` slot that `select_node` takes on the first player-node selection of a
  descent. That selection is always the root, since a descent starts there. A solved or missing
  child falls back to PUCT. Single-threaded by construction: Gumbel mode ignores `workers`.
- **Knobs** (`MctsConfig`, presets, `BLOOD_MCTS_GUMBEL_*`):
  - `gumbel_m`: the number of root moves considered. 0 means off, which is the shipped search.
  - `gumbel_scale`: the Gumbel noise scale. 0 is deterministic play: top-m by prior, then
    halving. 1 is the paper's sampling, for self-play.
- **Budget:** the budget counts new descents (`budget_mode = "iterations"`). A visits budget is
  read as descents in Gumbel mode, because halving needs to know how many descents it has to
  split.
- **σ(q̂) = (c_visit + max_b N(b)) · c_scale · q̂**, with c_visit = 50 and c_scale = 0.1 (the mctx
  defaults, the same constants as `prepare --policy-target gumbel`).
  - q̂ is each survivor's Q in the mover's frame, min-max normalised over the survivors.
  - An unscored child (a chance subtree still withholding its value) takes the root's Q, the
    same fill FPU uses.
- **Schedule:** m considered, P = ceil(log2 m) phases. Each phase gives every survivor
  `max(1, floor(n / (P · survivors)))` descents, then halves while more than 2 remain. Whatever
  budget is left goes round-robin to the final survivors.
- **The move played** is the best survivor by `g + logit + σ(q̂)`, not the best-Q child of the
  whole root. The training record is unchanged: every root child's visits, Q and prior, so the cq
  target sees the whole fan.
- **Exploration:** in Gumbel mode, the Gumbel noise is the self-play exploration. Dirichlet root
  noise and visit-sampled moves should be off. Wiring that into generation is step 3, only if
  step 2 wins.
- **Wire:** `MctsConfig` crosses the hub inside `SearchConfig`, so the new fields bump
  `PROTOCOL_VERSION` to 11. Remote workers must rebuild.

## Step 2 — does it buy the 16x9 budget back? (drives only)

1. **Convergence (no games).** `botbowl-ui convergence` with the Gumbel preset at 250, 500, 1000
   and 2000 descents. It uses exp057's states and seeds, scored against exp057's 16000-descent
   PUCT reference rows (top-1, in-top-3, regret). Pass: on 16x9 wide roots, Gumbel at 1000 is
   at least as close to the reference as PUCT at 4000.
2. **Drives, same budget.** gen04 with Gumbel m=16 at 1000 descents against gen04 with PUCT at
   1000 descents, on the gen04-screened contested sets (`runs/exp059/positions`), SPRT 0.5:0.55
   on both boards.
3. **Drives, the budget question.** On 16x9, Gumbel at 1000 against PUCT at 4000. Level or better
   means the 16x9 budget problem is solved at a quarter of the cost.

## Step 3 — generation (only if step 2 wins)

Generate with Gumbel (`gumbel_scale = 1`, no Dirichlet noise, no sampled moves) and relaunch the
loop. Judge it by the loop's own drive benchmark against gen21.

**Results:** _(pending)_
