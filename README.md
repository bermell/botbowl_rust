# botbowl_rust

Attempt to implement the Blood Bowl 2020 rules in a blazingly fast engine with the purpose of eventually creating an AI
that is stronger than any human. Heavily inspired by [botbowl](https://github.com/njustesen/botbowl) but re-written in
Rust to improve the execution speed for tree searching and machine learning. But mostly because rust.

![Better rewrite in rust](https://i.redd.it/xx367w6kroz41.jpg)

## Workspace layout

Thirteen member crates sharing one `Cargo.lock` and one `target/`, plus the nested `recon_mcts/` library (its own
workspace):

- `botbowl-engine/` — pure rules library (procedure-stack state machine, `DiceMode`, pathing, scripted bot).
- `botbowl-curriculum/` — training scenarios (`Lecture` trait, deterministic trial runner).
- `botbowl-mcts/` — MCTS adapter (`BloodBowlDynamics` + `MctsBot`) on top of `recon_mcts/`.
- `botbowl-nn/` — board encoding and ONNX inference (tract, or a remote Python sidecar); `train/` holds the PyTorch trainer.
- `botbowl-data/` — trajectory schema, JSONL persistence and provenance stamps.
- `botbowl-play/` — "play one game, return its record": generation, eval ladders, bot construction.
- `botbowl-hub/`, `botbowl-worker/`, `botbowl-hub-proto/` — distributed generation and eval over websockets.
- `botbowl-ui/` — `ratatui` terminal frontend plus the headless `dataset` / `eval` / `mc-label` pipeline commands.
- `botbowl-web/{proto,server,client}/` — play against the bots in a browser.

[`CLAUDE.md`](CLAUDE.md) has the architectural overview, commands and invariants (most crates have their own
`CLAUDE.md`); `plans/001-grand-plan.md` has the long-term roadmap.

## TODO

List of things to implement and write tests for in order to use as engine for forward model in a botbowl competition:

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

- More curriculum lectures (Defend TD, Pass/Hand-off TD).
- Improve MCTS bot capability — more priors / pruning rules, better leaf scoring, smarter scripted picks.
- Gym Env
- FFI to python

## Development

The engine is developed with TDD (test driven development). If code can be removed without breaking tests, it should be.
With exception for handling of weird errors. This should make it easy to get started with contributing and refactoring!
:)

### Code coverage

Install tarpaulin into your machine `cargo install cargo-tarpaulin` Then run
`cargo tarpaulin -p botbowl-engine --out Html` and browse to the newly created html file. Finally use `git blame` to
see who added code without covering it with a test!

### Profiling

`PROFILING.md` lists the throughput benchmarks and the recipe for `samply`-based CPU profiles of `MctsBot`.
