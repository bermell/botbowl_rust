# botbowl_rust

Attempt to implement [botbowl](https://github.com/njustesen/botbowl) in rust.
With purpose of speeding up search algorithms and machine learning. But mostly
because rust.

![Better rewrite in rust](https://i.redd.it/xx367w6kroz41.jpg)

## TODO

List of things to implement and write tests for in order to use as engine for
forward model in a botbowl competition:

- Botbowl 2020 rules
- Two Human team, only starting skills without Ogre

### Rules left to implement

- (1/11) Kickoff table
- Useful setup
- (Pathfinding) with leaping over prone players

### Tests to add

- handoff turnover if possession lost, needs test and implementation
- score on opponent's half needs test

### Other things (in order of priority)

- Record a game (rewind and forward state based on a diff)
- Watch a recording in terminal
- MCTS example bot (includes a gamestate hash)
- Play in terminal
- Bot benchmarking suite:
  - scenarios with rigged dice
  - Scenarios with expanded search tree for crazy thorough evaluation
  - Evaluate against scripted and random bot.
- Gym Env
- FFI to python

## Development

- The engine uses TDD. If code can be removed without breaking tests,
  it should be. With exception for error handling.

### Code coverage

Install tarpaulin into your machine `cargo install cargo-tarpaulin`
Then run `cargo tarpaulin --out Html`

Big thing:
