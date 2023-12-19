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

- [x] Fouling (in pathfinding, ejection)
- [x] Blocking (incl crowd surf and chain pushing)
- [ ] Passing (in pathfinding)
- [x] Handing off (in pathfinding)
- [ ] (1/11) Kickoff table
- [x] kickoff touchback
- [x] Very basic setup
- [ ] Useful setup
- [x] Touchdown

### Tests to add

- handoff turnover if possession lost, needs test and implementation
- score on opponent's half needs test

### Other things (in order of priority)

- Record a game
- Watch a recording
- [ ] Play in terminal
- [ ] Gym Env
- [ ] FFI to python
- [ ] MCTS example bot
