extern crate botbowl_engine;
use botbowl_engine::bots::RandomBot;
use botbowl_engine::core::game_runner::BotGameRunner;

#[test]
fn random_bot_plays_game() {
    let away_bot = Box::new(RandomBot::new());
    let home_bot = Box::new(RandomBot::new());
    let mut bot_game = BotGameRunner { home_bot, away_bot };

    let result = bot_game.run();
    println!("{:?}", result);
}
