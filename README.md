# botbowl_rust

Attempt to implement the Blood Bowl 2020 rules in a blazingly fast engine
with the purpose of eventually creating an AI that is stronger than any human.
Heavily inspired by [botbowl](https://github.com/njustesen/botbowl) but re-written
in Rust to improve the execution speed for tree searching and machine learning.
But mostly because rust.

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

The engine is developed with TDD (test driven development).
If code can be removed without breaking tests, it should be.
With exception for handling of weird errors. This should
make it easy to get started with contributing and refactoring! :)

### Code coverage

Install tarpaulin into your machine `cargo install cargo-tarpaulin`
Then run `cargo tarpaulin --out Html` and browse to the newly crated html file.
Finally use `git blame` to see who added code without covering it with a test!

### Optimizations

> Premature optimizations is the root of all evil!

- **Value:** smaller recordings
  - **Solution:** make game recording json use `json_patch` instead
    of storing the entire state for each step!
  - **Solution:** store only action and roll outcomes.
