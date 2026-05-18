# MCTS action registery

This idea builds on the fact that the tree searcher runs to the end of the
agent's turn (and sometimes to the end of the opponent's turn). And we have
up to 11 players that can move. Most of the actions will be spatial, meaning
they contain coordinates. And it also builds on the fact that action on one
part of the board are often independent of actions on another part of the board.

The idea is to keep a registery of all actions taken during the tree search
and track their score. We update the score on each back-propagation.
We can then use this registery to bias the tree search towards actions that have
been successful in the past. This is similar to how we use the UCB score to bias
the tree search towards actions that have been explored less.
