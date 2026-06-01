# Biased action selection policy

To make the first few iterations of the tree searcher more efficient we can bias the action selection policy towards
actions that are usually good.

We can return a probability distribution over the action just like alpha zero does with prior probabilities but instead
of using a neural network we just use scripted domain knowledge.

We start by giving all actions a base probability of 1. Then we can apply some rules to adjust the probabilities.

- picking up the ball: \*10
- blitzing ball carrier: \*10
- moving to mark the ball carrier: \*5
- moving to mark a player with the ball: \*5
- moving ball carrier towards the endzone: \*5
- moving to mark opponent player: \*2
- moving towards ball or ball carrier: \*2

Basically we want don't want the tree searcher to waste time exploring moving players to empty squares which there are
many off. We likely want to tune these probabilities.

I'm not entirely sure how to make this probability be reduced once the nodes has a few visits. I know Alpha zero uses
the visit count to reduce the influence of the prior probability and thus use the back-propagated value instead.
