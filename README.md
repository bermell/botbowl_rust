# botbowl_rust

Attempt to implement [botbowl](https://github.com/njustesen/botbowl) in rust. With purpose of speeding up search algorithms and machine learning. But mostly because rust. 

![Better rewrite in rust](https://i.redd.it/xx367w6kroz41.jpg)

## TODO
List of things to implement and write tests for in order to use as engine for forward model in a botbowl competition: 
- Botbowl 2016 rules 
- Two Human team, only starting skills without Ogre

**Refactor and optimizations** 
- [ ] Available action require less heap allocation
- [ ] Refactor away unnessary struct and enums: 
          - Pathing::Path (Node is fine, also only traverse back when necessary),
          - Pathing::PlayerActionType (it's fine to use PosAT::{StartMove, StartBlitz} etc.. )

**Rules left to implement**
- [x] Fouling (in pathfinding, ejection) 
- [x] Blocking (incl crowd surf and chain pushing)
- [ ] Passing (in pathfinding) 
- [x] Handing off (in pathfinding) 
- [ ] (1/11) Kickoff table 
- [x] Very basic setup 
- [ ] Useful setup
- [x] Touchdown

**Other things**
- [ ] Play in terminal
- [ ] Gym Env 
- [ ] FFI to python 
- [ ] MCTS example bot