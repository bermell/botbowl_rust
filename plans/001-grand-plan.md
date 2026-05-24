# The grand plan to beat humans in the game of Blood Bowl

## Introduction

Blood bowl is a turn-based strategy game that combines elements of American football and fantasy. The game is played on a grid-based board, where two teams of players compete to score touchdowns while also trying to injure or eliminate their opponents. The game is known for its complexity and strategic depth, making it a challenging game for both human players and AI.

## Grand plan

Apply alpha-zero to blood bowl, pour in domain knowledge into action selection, chance node outcomes, heuristics, etc...
Script as much as possible to reduce tree size.

Blood bowl as a game can be simplified to test the approach:

- reduce the board size and number of players on the board.
- reduce the skills of the players.

To ensure a diverse set of training data for a neural network we'll build a curriculum learning suite that starts with a simplified version of the game and gradually increases in complexity. This will allow the AI to learn basic strategies and tactics before moving on to more complex scenarios. As well as a self-play training loop to generate data for the neural network. And a way to evaluate the performance of the AI without having a opponent to play against - because human-in-the-loop evaluation isn't scalable and even AI in the loop is noisy because of the stochastic nature of the game.

## What has been built?

- We have an implementation of the game engine. Though not yet optimized for the memory footprint of the search tree, it is very easy to control for the stochastic nature of the game.
- We have an MCTS implementation with a test implementation working for the game 2048. The tree searcher also has a recombination mechanism to reduce the size of the search tree by merging nodes that represent the same game state. The implementation supports multiple threads operating on the same tree.
- We have a low quality terminal UI implementation to visualize the game state and see agents play.
- We have a first-iteration scripted Rust bot (`botbowl-engine/src/scripted_bot.rs`) translated from the Python reference. It handles coin toss, setup, kickoff, block-die selection, reroll decisions, ball-carrier pathing toward the endzone, safe blocks, and pickup attempts. The ladder is the natural extension point — more capability per step before falling through to "end turn".
- We have the start of a curriculum-learning suite (`botbowl-curriculum/`) with a `Lecture` trait, a trial runner, and five lectures: "Score TD — Easy/Medium" and "Get the ball — Easy/Medium/Hard". The scripted bot scores ≥97% on all five; random ≤1% on four of them. The runner is what later steps (MCTS heuristic / rollout / NN) will measure against.
- We have a `DiceMode` enum on `GameState` (engine side, `core/gamestate.rs`) that makes dice resolution explicit: `RollDice` (RNG only, production play), `FixedDice(queue)` (tests/builders, FIFO pop, panic on empty), `RegisterRolls` (MCTS — engine pauses on every roll, caller resumes via `state.step_with_roll(result)`), and `DicePolicy(policy)` (lectures). Lecture policies are *total*: built-ins like `SucceedAtOrEasier { d6, sum2d6, block_dice }` pin pickup/dodge/block outcomes and delegate scatter/bounce/etc. to RNG internally. `SucceedAtOrEasier` covers the grand plan's "3+ succeeds, 4+ fails" pass/fail semantics plus a `KnockdownAtAdvantage` block-die policy that turns 2+ attacker-dice rolls into all-Pow.
- We have a hand-rolled `Hash`/`Eq` on `GameState` ignoring `log`/`rng`/etc., so MCTS-style transposition tables can recombine equivalent positions reached via different histories. MCTS clones the search root, calls `set_dice_mode(DiceMode::RegisterRolls)`, then drives the tree via `micro_step` (paused-on-roll observations land in `state.pending_roll`) and `step_with_roll(result)` (deterministic resumption per chance outcome).
- We have an MCTS bootstrap crate (`botbowl-mcts/`) with `GameDynamics` for the engine, explicit chance nodes (`BbAction::Chance` with probability weights, mirroring the 2048 reference), and an `MctsBot` adapter implementing the `Bot` trait. With *pure UCT* action selection (no scripted-bot warm start) and a tier-1+2+4 leaf-score ladder, MCTS lifts Score TD Easy from the random baseline of ~9% to **62%** at 1000 search iterations per move — a clear demonstration that the tree search is doing real navigation, attributable to the search itself rather than to an inner heuristic.

## Order of business

1. Improve the UI
2. Scripted agent
3. Curriculum learning suite
4. MCTS agent with heuristic
5. MCTS agent with roll out
6. MCTS agent with neural network
7. Self-play training loop

### 1. Improve the UI

We should have a terminal UI in rust that is nice to look at and allows us to easily visualize the game state and see agents play. This will be important for debugging and for evaluating the performance of our agents. And it should be easy for LLMs like claude code to extract a "screenshot" from a game state to see how the UI is rendering to be able to evaluate UI.

It should also be easy for human to play a game against an agent.

### 2. Scripted agent

A first-iteration scripted agent now lives at `botbowl-engine/src/scripted_bot.rs`. It is a small heuristic ladder: coin toss → setup → kickoff aim → block-die selection → reroll decision → ball-carrier path toward the endzone → safe blocks (≥2DB) → **blitz the enemy ball carrier when a 2-dice blitz is achievable** → pickups → end-turn. It handles all mechanics needed to verify curriculum lectures end-to-end (including "Get the ball — Hard", which requires moving an assist into position before the block lands), but is intentionally far from a "strong" baseline — the ladder is the natural extension point.

The next iterations of this agent should: (a) own a "must score this turn" signal so the TD-attempt threshold can be configured per situation rather than a global constant; (b) handle blitz declarations to clear a path; (c) make smarter push-direction / follow-up choices; (d) consider speed/capability trade-offs once it's used as the rollout policy in step 5. It needs to be fast — random rollouts aren't informative in a game this complex.

### 3. Curriculum learning suite

The curriculum learning suite is built up by a set of scenarios, each available in multiple difficulties; a scenario at a given difficulty is called a "lecture". The original aim was "random agent solves the Easy lecture ~1% of the time" — in practice this is a ballpark guideline. The minimal "free path" Easy setup yields ~8% random success because a single Move action can land directly on an endzone square; pushing the random rate to ≤1% requires either obstacles or contrived setups that break the "free path" spec. The real evaluation metric is the *gap*: a meaningful lecture is one where the scripted/MCTS/NN agent dramatically outperforms random. The hardest difficulty should still be delicate enough that the scripted agent struggles.

Initial implementation lives in `botbowl_rust/botbowl-curriculum/`: a `Lecture` trait, a deterministic trial runner, and four lectures (`ScoreTdEasy/Medium`, `GetTheBallEasy/Medium`). The target-aware `DicePolicy` hook is shipped on the engine side (`core/dices.rs`) and the Get-the-ball lectures install `SucceedAtOrEasier { d6: ThreePlus, sum2d6: SevenPlus }` to pin pickup outcomes — unmarked pickups succeed automatically, marked pickups (target 4+) auto-fail. Block-die and three-outcome rolls aren't policy-controlled yet; extend the enum when needed.

Because we might end up training neural networks in the CL suite it's important that we can generate a lecture in different ways while retaining the same underlying structure. For example, the simplest lecture is the agent starting with the ball and being very close to the end zone, the goal is simply to move into the end zone to score a touchdown. We can randomize the exact position on the board and the player skills. As well as the position of all other players and their states.

Another consideration is controlling the stochastic nature of the game in lecture. We don't want the evaluation to fail a sound strategy just because of a bad dice roll. So a given lecture needs to be able to say that 3+ dodges succeed but 4+ dodges fail and 2 dice blocks are knockdowns but 1 dice block are pushes or even skulls.

The less complex scenarios will likely play out on the last turn of the game so we never need an opponent to play against. The more complex scenarios, like defending, will require an opponent. I'm thinking that for these we'll configure the scenario to say that the opponent needs to have succeeded in some other scenarios at given levels. For example, the opponent needs to have succeeded in the "score TD" lecture at difficulty 3. This way we can ensure that the opponent is good enough to provide a challenge but we don't need to have a human in the loop to evaluate the performance of our agent.

Here's a list of lectures we can start with:

- "Score TD": Last turn. Agent needs to score touchdown. Intended learning: how to move the ball carrier, how to dodge, how to block.
  - Easy: start with ball. Free path to end zone.
  - Medium: start with ball. Some opponents on the board that need to be blocked or blitzed out of the way.
  - Hard: boll on the ground. Opponents in the way.
- "Pass/Hand-off TD": Same as the score but the agent needs to pass or hand-off.
- "Defend TD": Last turn. Opponent has the ball and is close to the end zone. Agent needs to prevent opponent from scoring. Intended learning: tackle zones, how to block and foul. Need an opponent to play against. Opponent gets to play and lecture is complete if opponent fails to score in one turn. Control stochasticity.
  - Easy: Blitzed already used, all players but one are used, need to move last player to one of a couple correct squares.
  - Medium: Blocks and blitz are available but need to be used correctly. Still few players to move. "kind of obvious" solution.
  - Hard: All players available, need to use blocks and blitzes correctly. More players on the board.
- "get the ball": Accurire the ball from ground or opponent. Intended learning: ball is harder to pickup when marked by opponent, how to clear opponent from ball. In this scenario the opponent gets to play and lecture is complete if successful if agent still has ball after opponent's turn. Control stochasticity.
  - Easy: Ball is on the ground, not marked by opponent.
  - Medium: Ball is on the ground, marked by opponent.
  - Hard: Opponent has the ball, need to tackle and pick up.

### 4. MCTS agent with heuristic

We'll adapt the recombining MCTS implementation to work with the blood bowl engine. We'll build a scripted heuristic for the first action selection policy but as a node gets more visit the instal heuristic will be replaced by normal UCT to balace exploration and exploitation. The scripted action selection will look very much like the scripted agent.

To score a leaf node we will build a another heuristic that has a scoring ladder.

- The game score is most important - to encourage scoring.
- ball control in order:
  - holding the ball without being marked,
  - holding ball but being marked,
  - ball on floor but agent has marked it,
  - ball on floor and not marked by anyone or marked by both teams,
  - ball on floor and opponent has marked it.
  - opponent has ball but is marked by agent,
  - opponent has ball and is not marked by agent.
- weighted player value based on the player's team value but multiplied by a health factor to encourage keeping players alive and hurting the other team. For example 1.0 for standing, 0.75 for prone, 0.5 for stunned.
- distance to end zone for the ball carrier, to encourage moving towards the end zone.

There's be many tunable parameters in the heuristics. We'll see if the parameters translates from smaller to bigger pitch sizes.

### 5. MCTS agent with roll out

We'll replace the leaf scoring heuristic with a roll out policy. The roll out policy will be the scripted agent we built in step 2. This will allow us to have a more accurate evaluation of the leaf nodes, at the cost of increased computation time. We'll run the roll out until the end of the opponent's next turn and use the same scoring ladder as in step 4 to evaluate the outcome of the roll out.

There are many considerations for the roll out related to the stochastic nature of the game. Just using random dice rolls will likely be too noisy. Even a fixed seed will create noise as a small change in the action will give different random rolls. So we like need a "optimistic" roll out where rolls with 30% chance of success are always successful. And conversely a "pessimistic" roll out. And all things in between.

In this step we might defer the roll outs to separate machines and let the main machine's threads focus on the tree search.

### 6. MCTS agent with neural network

In this step we'll investigate if a neural network can approximate

- the action selection policy by having it estimate the number of times each action was selected in the roll out MCTS agent just like AlphaZero does.
- the leaf scoring by having it estimate the outcome of the roll out policy. This is more similar to how MuZero does it.

To make the experiment fast we'll build a dataset from the MCTS agent playing against itself as well as playing all the curriculum learning lectures. We can the use immitation learning to train the network. It's critical to not have a self-play loop at this stage so we can iterate on the network architecture and other parameters.

### 7. Self-play training loop

Finally we'll build a self-play training loop just like AlphaZero.
