use crate::core::gamestate::GameState;
use crate::core::model::Action;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

pub trait Bot {
    fn get_action(&mut self, state: &GameState) -> Action;
    fn set_seed(&mut self, _rng: ChaCha8Rng) {}
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
    fn set_seed(&mut self, rng: ChaCha8Rng) {
        self.rng = rng;
    }
}

#[cfg(test)]
mod tests {
    use crate::bots::RandomBot;
    use crate::core::game_runner::BotGameRunnerBuilder;

    #[test]
    #[ignore = "wall-clock bench — run with --ignored --nocapture"]
    fn random_bot_full_game_bench() {
        use crate::core::game_runner::GameRunner;
        let trials = 300;
        let mut total = std::time::Duration::ZERO;
        let mut total_actions: u64 = 0;
        for _ in 0..trials {
            let mut bot_game = BotGameRunnerBuilder::new()
                .set_home_bot(Box::new(RandomBot::new()))
                .set_away_bot(Box::new(RandomBot::new()))
                .build();
            let t0 = std::time::Instant::now();
            let mut actions: u64 = 0;
            while !bot_game.game_over() {
                bot_game.step();
                actions += 1;
            }
            total += t0.elapsed();
            total_actions += actions;
            eprintln!(
                "RANDOM_GAME_BENCH trial_ms={:.2} actions={}",
                t0.elapsed().as_secs_f64() * 1e3,
                actions,
            );
        }
        eprintln!(
            "RANDOM_GAME_BENCH avg_ms={:.2} avg_actions={:.1} trials={}",
            total.as_secs_f64() * 1e3 / trials as f64,
            total_actions as f64 / trials as f64,
            trials,
        );
    }

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
