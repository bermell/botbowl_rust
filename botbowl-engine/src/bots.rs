use crate::core::gamestate::GameState;
use crate::core::model::{Action, FullPitch, Position, SmallVecPosAT};
use crate::core::pathing::Node;
use crate::core::table::SimpleAT;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::rc::Rc;

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
    fn sample_simple(&mut self, aa: &HashSet<SimpleAT>) -> Action {
        let vec_sim: Vec<&SimpleAT> = aa.into_iter().collect();
        let l = vec_sim.len();
        debug_assert!(l > 0);
        let choice = self.rng.gen_range(0..l) as usize;
        Action::Simple(*vec_sim[choice])
    }
    fn sample_positional(&mut self, aa: &FullPitch<SmallVecPosAT>) -> Action {
        let positions: Vec<(Position, &SmallVecPosAT)> =
            aa.iter_position().filter(|(_, sv)| sv.len() > 0).collect();
        let l = positions.len();
        debug_assert!(l > 0);
        let choice = self.rng.gen_range(0..l) as usize;
        let (pos, sv) = positions[choice];
        let l = sv.len();
        debug_assert!(l > 0);
        let choice = self.rng.gen_range(0..l) as usize;
        let action_type = sv[choice];

        Action::Positional(action_type, pos)
    }
    fn sample_path(&mut self, aa: &FullPitch<Option<Rc<Node>>>) -> Action {
        let paths: Vec<Rc<Node>> = aa.iter().filter_map(|x| x.clone()).collect();
        let l = paths.len();
        debug_assert!(l > 0);
        let choice = self.rng.gen_range(0..l) as usize;
        let path = &paths[choice];
        Action::Positional(path.get_action_type(), path.position)
    }
}

impl Bot for RandomBot {
    fn get_action(&mut self, state: &GameState) -> Action {
        let aa = &state.available_actions;
        let path_action: Option<Action> =
            aa.get_paths().as_ref().map(|paths| self.sample_path(paths));

        let pos_action: Option<Action> = aa
            .get_positional()
            .as_ref()
            .map(|pos_actions| self.sample_positional(pos_actions));

        let simple_action: Option<Action> = if aa.get_simple().is_empty() {
            None
        } else {
            Some(self.sample_simple(aa.get_simple()))
        };
        let action_list: Vec<Action> = [path_action, pos_action, simple_action]
            .iter()
            .filter_map(|a| a.clone())
            .collect();
        let l = action_list.len();
        debug_assert!(l > 0);
        let choice = self.rng.gen_range(0..l) as usize;
        action_list[choice]
    }
}
