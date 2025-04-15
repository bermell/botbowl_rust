use std::sync::atomic::{AtomicU32, Ordering};

use recon_mcts::GameDynamics;

use crate::game_2048::{Coord, Direction, Game2048, GameState};

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub enum ActionChance {
    Action(Direction),
    Chance(Coord, u32, f32),
}
// impl hash, ord and eq for ActionChance - doesn't need to use the f32 value
impl std::hash::Hash for ActionChance {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ActionChance::Action(direction) => {
                direction.hash(state);
            }
            ActionChance::Chance(coord, value, _) => {
                coord.hash(state);
                value.hash(state);
            }
        }
    }
}
impl Eq for ActionChance {}

#[derive(Debug)]
pub struct ScoreItem {
    visits: AtomicU32,
    score: i32,
    action_node: GameState,
}

impl GameDynamics for Game2048 {
    type Player = ();

    type State = Game2048;

    type Action = ActionChance;

    type Score = ScoreItem;
    type ActionIter = Vec<(Self::Player, Self::Action)>;

    fn available_actions(
        &self,
        _player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::ActionIter> {
        match state.state {
            GameState::Done => None,
            GameState::WaitingForAction => Some(
                state
                    .available_action()
                    .iter()
                    .map(|x| ((), ActionChance::Action(*x)))
                    .collect(),
            ),
            GameState::WaitingForRandom => Some(
                state
                    .available_chance()
                    .iter()
                    .map(|x| ((), ActionChance::Chance(x.0, x.1, x.2)))
                    .collect(),
            ),
        }
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        let mut new_state = state;
        match action {
            ActionChance::Action(direction) => {
                debug_assert!(new_state.state == GameState::WaitingForAction);
                new_state.step_action(*direction);
            }
            ActionChance::Chance(coord, value, _) => {
                debug_assert!(new_state.state == GameState::WaitingForRandom);
                new_state.step_random(*coord, *value);
            }
        }
        Some(new_state)
    }

    fn select_node<II, Q, A>(
        &self,
        parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        parent_node_state: &Self::State,
        _purpose: recon_mcts::SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        Q: std::ops::Deref<Target = Option<Self::Score>>,
        A: std::ops::Deref<Target = Self::Action>,
    {
        debug_assert_eq!(parent_score.unwrap().action_node, parent_node_state.state);

        if parent_node_state.state == GameState::WaitingForRandom {
            // pick the one with fewest visits. We know for sure here that all the scores are
            // Chance types. We need to get that score, increment the visits and return the action
            let min_ii_item = scores_and_actions
                .clone()
                .into_iter()
                .min_by(|a, b| {
                    let a = a.0.as_ref().unwrap();
                    let b = b.0.as_ref().unwrap();

                    a.visits
                        .load(Ordering::Relaxed)
                        .cmp(&b.visits.load(Ordering::Relaxed))
                })
                .unwrap();

            // increment the visits
            min_ii_item
                .0
                .as_ref()
                .unwrap()
                .visits
                .fetch_add(1, Ordering::Relaxed);

            return min_ii_item.1.to_owned();
        } else if parent_node_state.state == GameState::WaitingForAction {
            // pick the one with highest score. We know for sure here that all the scores are
            // Score types. We need to get that score, increment the visits and return the action
            let max_ii_item = scores_and_actions
                .clone()
                .into_iter()
                .max_by(|a, b| {
                    let a = a.0.as_ref().unwrap();
                    let b = b.0.as_ref().unwrap();
                    a.visits
                        .load(Ordering::Relaxed)
                        .cmp(&b.visits.load(Ordering::Relaxed))
                })
                .unwrap();

            // increment the visits
            max_ii_item
                .0
                .as_ref()
                .unwrap()
                .visits
                .fetch_add(1, Ordering::Relaxed);
            // return the action
            return max_ii_item.1.to_owned();
        }
        panic!("Invalid state");
    }

    fn backprop_scores<II, Q, A>(
        &self,
        _player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: std::ops::Deref<Target = Self::Action>,
        Q: std::ops::Deref<Target = Self::Score>,
    {
        // if score_current is waiting for action or None; we need to return the max score in
        // child_scores_and_actions
        // else we return the weighted average of the scores that has at least one visit
        if score_current.is_none()
            || score_current.unwrap().action_node == GameState::WaitingForAction
        {
            let max_ii_item = child_scores_and_actions
                .clone()
                .into_iter()
                .max_by_key(|(q, _)| q.score)
                .unwrap();

            // remember to copy over the visits correctly
            Some(ScoreItem {
                visits: AtomicU32::new(max_ii_item.0.visits.load(Ordering::Relaxed)),
                score: max_ii_item.0.score,
                action_node: GameState::WaitingForAction,
            })
        } else if score_current.unwrap().action_node == GameState::WaitingForRandom {
            // we need to return the weighted average of the scores that has at least one visit
            let mut total_visits = 0;
            let mut total_score = 0.0;
            let mut total_prob = 0.0;
            for (q, a) in child_scores_and_actions.clone().into_iter() {
                if q.visits.load(Ordering::Relaxed) > 0 {
                    if let ActionChance::Chance(_, _, prob) = a.deref() {
                        total_prob += *prob;
                        total_score += *prob * q.score as f32;
                    } else {
                        panic!("Invalid action type");
                    }
                    total_visits += q.visits.load(Ordering::Relaxed);
                }
            }
            debug_assert!(total_visits > 0);
            debug_assert!(total_prob > 0.0);
            debug_assert!(total_prob <= 1.1, "total_prob: {}", total_prob);

            // normalize the score if total_prob < 1.0
            if total_prob < 1.0 {
                total_score /= total_prob;
            }

            Some(ScoreItem {
                visits: AtomicU32::new(total_visits),
                score: total_score as i32,
                action_node: GameState::WaitingForRandom,
            })
        } else {
            panic!("Invalid state");
        }
    }

    fn score_leaf(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        Some(ScoreItem {
            visits: AtomicU32::new(1),
            score: state.score as i32,
            action_node: state.state,
        })
    }
}

#[cfg(test)]
mod test {

    use recon_mcts::{GetState, SearchTree, Tree};

    use crate::game_2048::{Coord, Game2048};
    #[test]
    fn test_tree() {
        let game = Game2048::new_game(Coord { row: 2, col: 1 }, 2);
        let tree = Tree::new(game, GetState, (), game);
        for _ in 0..1000 {
            tree.step();
        }
    }
}
