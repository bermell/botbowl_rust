//! The search graph must be a DAG (see `GameDynamics::State` docs). A
//! `GameDynamics` implementation that lets a state recur creates a true
//! cycle once recombination merges the recurring state with its own
//! ancestor — and a descent that enters the cycle would otherwise spin
//! forever inside a single `step()` (observed as a multi-hour hang in
//! botbowl). The tree must instead crash the search: dump the descent
//! path to a file and panic.

use std::ops::Deref;
use std::sync::Arc;

use recon_mcts::prelude::*;

/// Two states, one legal action each: 0 → 1 → 0. With recombination
/// (`StoreState` keys nodes by state equality) the second visit to
/// state 0 merges with the root and the graph is a 2-cycle.
struct CycleGame;

#[derive(Clone, Debug, Hash, PartialEq)]
struct P;

impl GameDynamics for CycleGame {
    type Player = P;
    type State = u8;
    type Action = u8;
    type Score = f64;
    type ActionIter = Vec<(Self::Player, Self::Action)>;

    fn available_actions(&self, _player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        Some(vec![(P, (state + 1) % 2)])
    }

    fn apply_action(&self, _state: Self::State, action: &Self::Action) -> Option<Self::State> {
        Some(*action)
    }

    fn select_node<II, Q, A>(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        _parent_node_state: &Self::State,
        _purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        II: IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<Self::Score>>,
        A: Deref<Target = Self::Action>,
    {
        *scores_and_actions.into_iter().next().unwrap().1
    }

    fn backprop_scores<II, Q, A>(
        &self,
        _player: &Self::Player,
        _score_current: Option<&Self::Score>,
        _child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = Self::Action>,
        Q: Deref<Target = Self::Score>,
    {
        None
    }

    fn score_leaf(
        &self,
        _parent_score: Option<&Self::Score>,
        _parent_player: &Self::Player,
        _state: &Self::State,
    ) -> Option<Self::Score> {
        Some(0.0)
    }
}

#[test]
fn cyclic_game_panics_with_dump_instead_of_hanging() {
    let tree = Arc::new(Tree::new(CycleGame, StoreState, P, 0u8));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Without the guard this loop never finishes its first few
        // steps (the descent cycles 0 → 1 → 0 forever). With the
        // guard, one of these steps panics.
        for _ in 0..100 {
            tree.step();
        }
    }));

    let err = result.expect_err("search on a cyclic graph must panic, not hang");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("panic payload should be a string");
    assert!(msg.contains("cycle"), "panic message should mention the cycle: {}", msg);

    // The message names the dump file; it must exist and describe the
    // repeating descent path.
    let path = msg
        .split_whitespace()
        .find(|w| w.contains("recon_mcts_cycle_"))
        .expect("panic message should contain the dump file path");
    let dump = std::fs::read_to_string(path).expect("cycle dump file should have been written");
    assert!(dump.contains("cycle"), "dump should describe the cycle: {}", dump);
    std::fs::remove_file(path).ok();
}
