use std::{fs, io::Write};

use serde::{Deserialize, Serialize};

use crate::bots::Bot;

use super::{
    gamestate::{BuilderState, GameState, GameStateBuilder},
    model::TeamType,
};

pub trait GameRunner {
    fn step(&mut self);
    fn game_over(&self) -> bool;
    fn get_state(&self) -> &GameState;
    fn get_state_json(&self) -> String;
}

pub struct BotGameRunner {
    pub home_bot: Box<dyn Bot>,
    pub away_bot: Box<dyn Bot>,
    state: GameState,
    save_file: Option<String>,
    steps: Vec<GameState>,
    // initial_state_json: String,
}
impl GameRunner for BotGameRunner {
    fn get_state_json(&self) -> String {
        serde_json::to_string(&self.state).unwrap()
    }
    fn step(&mut self) {
        // initial state
        if self.save_file.is_some() && self.steps.is_empty() {
            self.steps.push(self.get_state().clone());
        }

        self.step_state();

        if self.save_file.is_some() {
            self.steps.push(self.get_state().clone());
        }
    }
    fn game_over(&self) -> bool {
        self.state.info.game_over
    }
    fn get_state(&self) -> &GameState {
        &self.state
    }
}
impl BotGameRunner {
    pub fn run(&mut self) -> String {
        while !self.game_over() {
            self.step();
        }
        "hey".to_string()
    }
    fn step_state(&mut self) {
        let action = match self.state.available_actions.team {
            Some(TeamType::Home) => Some(self.home_bot.get_action(&self.state)),
            Some(TeamType::Away) => Some(self.away_bot.get_action(&self.state)),
            None => None,
        };
        self.state.micro_step(action).unwrap();
    }
    pub fn save_to_file(&self) {
        let Some(file) = &self.save_file else {
            return;
        };
        let recording = Recording::new(self.steps.clone());
        recording.to_file(file);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Recording {
    states: Vec<GameState>,
    current_state: usize,
}
impl Recording {
    pub fn new(states: Vec<GameState>) -> Self {
        Self {
            states,
            current_state: 0,
        }
    }
    pub fn to_file(&self, file: &str) {
        let json_str = serde_json::to_string(&self).unwrap();
        let mut file = fs::File::create(file).unwrap();
        file.write_all(json_str.as_bytes()).unwrap();
    }
    pub fn from_file(file: &str) -> Self {
        let json_str = std::fs::read_to_string(file).unwrap();
        serde_json::from_str(&json_str).unwrap()
    }
}
impl GameRunner for Recording {
    fn step(&mut self) {
        if self.current_state >= self.states.len() {
            return;
        }
        self.current_state += 1;
    }
    fn game_over(&self) -> bool {
        self.current_state >= self.states.len()
    }
    fn get_state(&self) -> &GameState {
        &self.states[self.current_state]
    }
    fn get_state_json(&self) -> String {
        serde_json::to_string(&self.states[self.current_state]).unwrap()
    }
}

#[derive(Default)]
pub struct BotGameRunnerBuilder {
    home_bot: Option<Box<dyn Bot>>,
    away_bot: Option<Box<dyn Bot>>,
    replay_file: Option<String>,
}
impl BotGameRunnerBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set_home_bot(mut self, bot: Box<dyn Bot>) -> Self {
        self.home_bot = Some(bot);
        self
    }
    pub fn set_away_bot(mut self, bot: Box<dyn Bot>) -> Self {
        self.away_bot = Some(bot);
        self
    }
    pub fn set_replay_file(mut self, file: &str) -> Self {
        self.replay_file = Some(file.to_string());
        self
    }
    pub fn build(self) -> BotGameRunner {
        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::CoinToss)
            .build();
        state.rng_enabled = true;
        BotGameRunner {
            home_bot: self
                .home_bot
                .unwrap_or(Box::new(crate::bots::RandomBot::new())),
            away_bot: self
                .away_bot
                .unwrap_or(Box::new(crate::bots::RandomBot::new())),
            state,
            save_file: self.replay_file,
            steps: vec![],
        }
    }
}

#[cfg(test)]
mod gamestate_tests {

    use crate::core::gamestate::GameState;

    use super::{BotGameRunnerBuilder, GameRunner, Recording};

    #[test]
    fn save_and_replay_game() {
        let mut runner = BotGameRunnerBuilder::new()
            .set_replay_file("test.json")
            .build();

        let mut intermediate_states: Vec<GameState> = vec![runner.get_state().clone()];
        while !runner.game_over() {
            runner.step();
            intermediate_states.push(runner.get_state().clone());
        }
        runner.save_to_file();

        let mut recording = Recording::from_file("test.json");
        for state in intermediate_states.iter() {
            let recorded_state = recording.get_state();
            assert_eq!(*state, *recorded_state);
            recording.step();
            if state.info.game_over {
                assert!(recording.game_over());
                break;
            }
        }
        // remove the file
        std::fs::remove_file("test.json").unwrap();
    }
}
