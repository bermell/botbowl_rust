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
        let action_list = state.get_all_actions();
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
        // Fixed seeds keep the bench reproducible: every trial replays the same
        // sequence of game states and actions, so wall-clock deltas reflect code
        // changes rather than RNG variance.
        const BASE_SEED: u64 = 0xB10D_B0F1_2026_06_02;
        // Default to a stable distribution sample. For profilers (e.g. samply),
        // override with `BOTBOWL_BENCH_TRIALS=50` so the recording captures a
        // handful of games rather than minutes of accumulated noise.
        let trials: usize = std::env::var("BOTBOWL_BENCH_TRIALS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600);
        let mut total = std::time::Duration::ZERO;
        let mut total_actions: u64 = 0;
        let mut per_trial_ns: Vec<u64> = Vec::with_capacity(trials);
        for trial in 0..trials {
            let seed = BASE_SEED ^ (trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut bot_game = BotGameRunnerBuilder::new()
                .set_home_bot(Box::new(RandomBot::new()))
                .set_away_bot(Box::new(RandomBot::new()))
                .set_seed(seed)
                .build();
            let t0 = std::time::Instant::now();
            let mut actions: u64 = 0;
            while !bot_game.game_over() {
                bot_game.step();
                actions += 1;
            }
            let elapsed = t0.elapsed();
            per_trial_ns.push(elapsed.as_nanos() as u64);
            total += elapsed;
            total_actions += actions;
        }
        // Report distribution stats: per-trial means are noisy on shared CPUs,
        // so include min/median to make small wins visible.
        per_trial_ns.sort_unstable();
        let median = per_trial_ns[per_trial_ns.len() / 2];
        let min = per_trial_ns[0];
        let p10 = per_trial_ns[per_trial_ns.len() / 10];
        let p90 = per_trial_ns[per_trial_ns.len() * 9 / 10];
        eprintln!(
            "RANDOM_GAME_BENCH avg_ms={:.3} median_ms={:.3} min_ms={:.3} p10_ms={:.3} p90_ms={:.3} avg_actions={:.1} trials={}",
            total.as_secs_f64() * 1e3 / trials as f64,
            median as f64 / 1e6,
            min as f64 / 1e6,
            p10 as f64 / 1e6,
            p90 as f64 / 1e6,
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
