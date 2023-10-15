use crate::bots::Bot;

use super::{
    gamestate::{BuilderState, GameStateBuilder},
    model::TeamType,
};

pub struct BotGameRunner {
    pub home_bot: Box<dyn Bot>,
    pub away_bot: Box<dyn Bot>,
}
impl BotGameRunner {
    pub fn run(&mut self) -> String {
        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::CoinToss)
            .build();
        state.rng_enabled = true;
        while !state.info.game_over {
            let action = match state.available_actions.team {
                Some(TeamType::Home) => self.home_bot.get_action(&state),
                Some(TeamType::Away) => self.away_bot.get_action(&state),
                None => todo!(),
            };
            state.step(action).unwrap();
        }
        "hey".to_string()
    }
}
