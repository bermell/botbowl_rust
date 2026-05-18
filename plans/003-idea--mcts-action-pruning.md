# Idea: MCTS action pruning with domain knowledge

The purpose of this idea is to improve the efficiency of Monte Carlo Tree Search (MCTS) by pruning actions based on domain knowledge. This can help reduce the search space and focus on actions that aren't creating duplicate states. For example when moving a player on the board the action is first to select the player to make it active, then select the square to move to. If there are movements left on the player the engine gives the option of selecting another square as well as ending the movement. This makes sense for a human playing the game but for a tree searcher it creates a lot of duplicate states. So after the first move we can queue up a "end player turn" action which marks the player as "used" and can no longer be moved.

However, there are edge cases when you might not want to do that. If the player has the ball and the action is pass action then we should still allow the selecting another play to pass to, however we should not allow further movement.

## Pathfinding probability threshold

There's also a case to be made for limiting the pathfinding to stop when a path has a low probability of success. The higher that threshold is the faster it will run and the less actions we have to explore.

## Force pass and handoff

If a player is activated with the "start pass" or "start handoff" action when not holding the ball. Then the only allowed path should be to the ball. And if the player can't even reach the ball then the action should be pruned entirely - though that can be hard to know and it's not nice to prune the action after the node has been added to the tree.. :thinking-face:

If a player is activated with "start handoff" we should only allow paths that end up on a teammate - because I think the pathfinding tracks the handoff probabilities.

If a player is activated with "start pass" the it'd be nice if we did the same thing as for "start handoff" but I don't think the pathfinding tracks pass probabilities because they are a bit weirder.

It's an open question if this should be implemented in the game engine as an option to activate or if the pruning should only happen in the mcts. Likely better computationally to do it in the engine's pathfinding but we want to keep the logic simple too...
