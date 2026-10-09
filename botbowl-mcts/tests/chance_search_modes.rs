//! Plan 061: whole searches under every chance-backup mode with every new roll model on.
//!
//! The unit tests pin each piece; this one plays short drives with real dice so the trees hold
//! bounces onto players, pass scatters, deviates and throw-in chains, under each `ChanceBackup`,
//! and checks that nothing breaks on the way: no duplicate chance edge into one child (recon_mcts
//! panics dropping such a tree), no cycle, no runaway chain, a legal move every time.

mod common;

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::DiceMode;
use botbowl_mcts::chance_stats::{RollKind, CHANCE_STATS};
use botbowl_mcts::dynamics::ChanceBackup;
use botbowl_mcts::roll_outcomes::{BounceModel, PassScatterModel, ThrowInModel};
use botbowl_mcts::{MctsBot, MctsConfig, SearchBudget};

fn config(backup: ChanceBackup) -> MctsConfig {
    let mut cfg = MctsConfig::new();
    cfg.workers = 1;
    cfg.bounce_model = BounceModel::Catch;
    cfg.pass_scatter_model = PassScatterModel::Grouped;
    cfg.throw_in_model = ThrowInModel::Grouped;
    cfg.chance_backup = backup;
    cfg.chance_mass = 0.8;
    cfg
}

#[test]
fn every_chance_mode_searches_real_drives_with_the_new_roll_models() {
    let modes = [
        ChanceBackup::Complete,
        ChanceBackup::Partial,
        ChanceBackup::Mass,
        ChanceBackup::Sampled,
        ChanceBackup::Widen,
    ];
    for (i, backup) in modes.into_iter().enumerate() {
        for mut state in common::states(2, 61_000 + i as u64) {
            state.set_dice_mode(DiceMode::RollDice);
            let mut home = MctsBot::with_budget_and_config(SearchBudget::Iterations(48), config(backup));
            let mut away = MctsBot::with_budget_and_config(SearchBudget::Iterations(48), config(backup));
            for _ in 0..24 {
                if state.info.game_over {
                    break;
                }
                let team = state.available_actions.team.expect("a decision");
                let bot = if team == botbowl_engine::core::model::TeamType::Home {
                    &mut home
                } else {
                    &mut away
                };
                let a = bot.get_action(&state);
                assert!(state.is_legal_action(&a), "{backup:?}: illegal {a:?}");
                state.step(a).expect("legal step");
                home.release_stale_tree(&state);
                away.release_stale_tree(&state);
            }
        }
    }
    // The sweep reached the rolls it is about (process-wide counters; other tests only add).
    let (created, outcomes, _) = CHANCE_STATS.get(RollKind::Bounce);
    assert!(created > 0, "no bounce chance node in any tree");
    assert!(outcomes >= created, "{outcomes} outcomes for {created} bounces");
}
