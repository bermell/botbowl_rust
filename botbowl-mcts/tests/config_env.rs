//! The environment reaches a search only through `from_env` (plan 059 #6): `MctsConfig::default()`
//! and `MctsBot::new` are the shipped configuration whatever `BLOOD_MCTS_*` says, so an exported
//! knob cannot change test results. Its own test binary (one process, one test) because it sets
//! process environment variables.

use botbowl_mcts::{MctsBot, MctsConfig, SearchBudget};

#[test]
fn only_from_env_reads_the_environment() {
    std::env::set_var("BLOOD_MCTS_HORIZON_TURNS", "3");
    std::env::set_var("BLOOD_MCTS_FPU_REDUCTION", "250");
    std::env::set_var("BLOOD_MCTS_TREE_REUSE", "off");

    let shipped = MctsConfig::new();
    assert_eq!(MctsConfig::default(), shipped, "Default must not read the environment");
    assert_eq!(
        *MctsBot::new(SearchBudget::Iterations(10)).config(),
        shipped,
        "MctsBot::new must not read the environment"
    );

    let env = MctsConfig::from_env();
    assert_eq!(env.horizon_turns, 3);
    assert_eq!(env.fpu_reduction, 250.0);
    assert!(!env.tree_reuse);
    assert_eq!(*MctsBot::from_env(SearchBudget::Iterations(10)).config(), env);
}
