use crate::core::gamestate::GameState;
use crate::core::model::Action;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub trait Bot {
    fn get_action(&mut self, state: &GameState) -> Action;
}

pub struct RandomBot {
    rng: ChaCha8Rng,
}
impl RandomBot {
    pub fn new() -> RandomBot {
        Self {
            rng: ChaCha8Rng::from_entropy(),
        }
    }
    pub fn set_seed(&mut self, rng: ChaCha8Rng) {
        self.rng = rng;
    }
}

impl Default for RandomBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for RandomBot {
    fn get_action(&mut self, state: &GameState) -> Action {
        let action_list = state.available_actions.get_all();
        let l = action_list.len();
        debug_assert!(l > 0);
        let choice = self.rng.gen_range(0..l);
        action_list[choice]
    }
}

#[cfg(test)]
mod tests {
    use crate::bots::RandomBot;
    use crate::core::game_runner::BotGameRunnerBuilder;

    #[test]
    fn random_bot_plays_game() {
        color_backtrace::install();
        for _ in 0..10 {
            let mut bot_game = BotGameRunnerBuilder::new()
                .set_home_bot(Box::new(RandomBot::new()))
                .set_away_bot(Box::new(RandomBot::new()))
                .build();

            let result = bot_game.run();
            println!("{:?}", result);
        }
    }
}
