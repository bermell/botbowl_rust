//! Factory for the bots `botbowl-ui` knows how to instantiate from CLI
//! flags. Centralised so the same mapping is shared by `live`, `snapshot`,
//! and any future entry point.

use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::scripted_bot::ScriptedBot;
use botbowl_mcts::{MctsBot, SearchBudget};

use crate::cli::BotKind;

pub fn make_bot(kind: BotKind, mcts_iters: usize) -> Box<dyn Bot> {
    match kind {
        BotKind::Random => Box::new(RandomBot::new()),
        BotKind::Scripted => Box::new(ScriptedBot::new()),
        BotKind::Mcts => Box::new(MctsBot::new(SearchBudget::Iterations(mcts_iters))),
    }
}
