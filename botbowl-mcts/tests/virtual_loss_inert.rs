//! Virtual loss exists to push *concurrent* workers onto different paths. Each descent must take
//! back what it added when it ends, so at one worker it has no effect at all: the same search,
//! descent for descent, whatever its magnitude.
//!
//! It used to be cleared only when a backprop replaced the chosen child's score. A descent cut
//! off below the root (at a chance node still withholding its value) left its penalty behind; it
//! accumulated and buried the best children. On the same net, virtual loss 0 beat the leaking 30 by
//! 0.686 over 400 games (plan 049).

mod common;

use botbowl_mcts::{MctsBot, MctsConfig, SearchBudget};

fn descents(state: &botbowl_engine::core::gamestate::GameState, vl: i32) -> Vec<(String, u32)> {
    let mut cfg = MctsConfig::new();
    cfg.workers = 1;
    cfg.virtual_loss = vl;
    cfg.trace_root_descents = true;
    let mut bot = MctsBot::with_budget_and_config(SearchBudget::Iterations(400), cfg);
    botbowl_engine::bots::Bot::get_action(&mut bot, state);
    let mut d: Vec<(String, u32)> = bot
        .last_search()
        .unwrap()
        .root_descents
        .clone()
        .unwrap()
        .into_iter()
        .map(|(a, n)| (format!("{a:?}"), n))
        .collect();
    d.sort();
    d
}

#[test]
fn at_one_worker_virtual_loss_changes_nothing() {
    let roots: Vec<_> = common::states(60, 5)
        .into_iter()
        .filter(|s| s.proc_stack_top() == Some("Turn") && s.get_all_actions().len() >= 8)
        .take(6)
        .collect();
    assert!(!roots.is_empty());
    for (i, s) in roots.iter().enumerate() {
        assert_eq!(
            descents(s, 0),
            descents(s, 30),
            "root {i}: virtual loss 30 changed a 1-worker search"
        );
    }
}
