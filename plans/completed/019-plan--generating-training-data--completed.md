# Generating training data for mcts neural network

**Status:** Implemented 2026-07-15.

- Generator: `botbowl_curriculum::random_start::generate_random_start(&RandomStartConfig, &mut ChaCha8Rng)`.
  Each team is split into roles by `line_fraction`/`pocket_fraction`: **line** players decay toward the team's
  front column (`front_line^-|dx|`, engagement line sampled 2–4 squares ahead of the ball toward the attacker's
  endzone, defenders one square beyond) and toward the ball's y; **pocket** players decay toward the ball
  (`ball_distance^-d`, per-square, so it beats far-area growth); **wide** players place freely. All roles get the
  `mark_teammate`/`mark_opponent`/`own_side` multipliers, and the final weight is sharpened by
  `^(1/temperature)`. (The original `bias^(1-d/max_dim)` kernel was too weak at any knob setting — ring-area
  growth dominated it and clusters seeded far from the ball.)
- Resolved decisions: new dataset mode (`botbowl-ui dataset --mode random-start`, bias CLI flags, kickoff self-play kept);
  game context randomized (half, turn, score capped by elapsed turns, active team) — the NN encoder already observes all of these;
  ball carried with probability `--carried-prob` (default 0.75), never in an endzone column; per-team player count 7–11 skewed toward 11.
- Engine addition: `GameState::set_half(half)` relabels the in-progress half and drops the pending one so half-2 states end the game correctly.
- Visualizer: `cargo run -p botbowl-ui -- placement` — space regenerates, 1-5 select a bias variable, up/down adjust (re-sampling the same seed for like-for-like comparison), q quits.

In the plan we outline the way generate the starting positions for the mcts bot self-play training. Instead of starting
the game from the initial state at kickoff, allowing the bot to select the formations, we will place the players on
random positions on the field, and randomly place the ball.

The purpose of this is to generate a more diverse set of training data as we have seen that untrained bots tend to not
move all players.

The challenge is to generate positions that make can plausibly exist in a real game. Just randomly placing players makes
as much sense in blood bowl as it does in chess.

## A realistic blood bowl state

Blood bowl is a positional game. Usually the attacking team has the ball on a player a few squares behind the rest of
the team which is spread out in a line so that the defending team can't easily reach the ball carrier. The defending
team is spread out on a line to prevent the attacking team from moving forward in the middle of the pitch. They may
allow the attacking to to move forward on a flank to lock the down on a side line.

The attacking team may send a player around the defending team to act as a scoring threat that can receive a
pass/handoff and score a touchdown. Similarly the defending team may send a player or two around the attacking team to
threaten the ball carrier forcing the attacking team to protect the ball carrier even more.

But sometimes the positional brawl breaks up and players scatter and chase each other and the ball around the pitch.

## The approach

We place the ball randomly and then start placing players with some bias. Each square gets a probability which is
multiplied with the bias variables below (all defaulting to 1.0):

- `ball_distance`: increases the prob of squares closer to the ball.
- `mark_teammate`: increases the prob of squares next to teammates.
- `mark_opponent`: increases the prob of squares next to opponents.
- `own_side`: increases the prop of squares between the team's own endzone and the closest opponent.

After we have decided these biases we place players each team taking turns.

## Debugging

A visualizer should be built that choose the values of the bias variables and generates a random state. Use the existing
visualizer. The space bar should generate a new random state. number keys to select a bias variable, and the up/down
arrows to increase/decrease the value of the selected bias variable. q to exit.
