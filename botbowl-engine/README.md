# botbowl_rust

Attempt to implement the Blood Bowl 2020 rules in a blazingly fast engine
with the purpose of eventually creating an AI that is stronger than any human.
Heavily inspired by [botbowl](https://github.com/njustesen/botbowl) but re-written
in Rust to improve the execution speed for tree searching and machine learning.
But mostly because rust.

## Engine summary

This engine is a stack-driven rules machine built around a single `GameState`.
Each rule is a `Procedure` that consumes input, mutates the state, and returns
a `ProcState` that tells the engine whether it needs more input, a roll, or
should push more procedures onto the stack. Available actions are generated
per procedure and exposed to bots or UI clients. Movement uses a pathing
module that precomputes probabilistic paths, including dodge/GFI/pickup/pass
events, which the movement procedure replays as the player advances. Bots
plug in through a simple trait and the game runner can record full state
timelines for replay.


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
