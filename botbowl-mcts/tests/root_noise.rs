//! Plan 048: self-play exploration — root Dirichlet noise and visit-proportional move sampling.
//!
//! The properties that matter: the default step is the plain search, byte for byte; noise moves
//! the search but never the priors written into the training sample; noise only takes effect on
//! a fresh root (a reused tree expanded its root long before it became one); a sampled move is
//! always a visited root child.

mod common;

use botbowl_data::Sample;
use botbowl_engine::core::gamestate::GameState;
use botbowl_mcts::{ExploreStep, MctsBot, ReuseOutcome, RootNoiseSpec, SampleSpec, SearchBudget};

const N: usize = 200;

fn bot() -> MctsBot {
    MctsBot::new(SearchBudget::Iterations(N)).with_workers(1)
}

fn noise(seed: u64) -> ExploreStep {
    ExploreStep {
        noise: Some(RootNoiseSpec {
            epsilon: 0.5,
            alpha: 10.0,
            seed,
        }),
        sample: None,
    }
}

/// A random start whose root offers a real choice.
fn wide_state() -> GameState {
    common::states(40, 11)
        .into_iter()
        .find(|s| s.available_actions.team.is_some() && s.get_all_actions().len() >= 8)
        .expect("a random start with at least 8 actions")
}

fn stats(s: &Sample) -> Vec<(String, u32, Option<i64>, Option<u32>)> {
    let mut v: Vec<_> = s
        .children
        .iter()
        .map(|c| (format!("{:?}", c.action), c.visits, c.q, c.prior.map(f32::to_bits)))
        .collect();
    v.sort();
    v
}

#[test]
fn the_default_step_is_the_plain_record() {
    let state = wide_state();
    let (a, plain) = bot().get_action_with_record(&state);
    let (b, explored, outcome) = bot().get_action_explore(&state, ExploreStep::default());
    assert_eq!(a, b);
    assert_eq!(stats(&plain), stats(&explored));
    assert!(!outcome.noised && !outcome.sampled && !outcome.deviated);
}

#[test]
fn noise_moves_the_search_but_the_sample_keeps_the_clean_priors() {
    let priors = |s: &Sample| {
        let mut v: Vec<_> = s
            .children
            .iter()
            .map(|c| (format!("{:?}", c.action), c.prior.map(f32::to_bits)))
            .collect();
        v.sort();
        v
    };
    let visits = |s: &Sample| {
        let mut v: Vec<_> = s
            .children
            .iter()
            .map(|c| (format!("{:?}", c.action), c.visits))
            .collect();
        v.sort();
        v
    };
    let searched_priors = |b: &MctsBot| {
        let mut v: Vec<_> = b
            .last_search()
            .unwrap()
            .children
            .iter()
            .map(|e| (format!("{:?}", e.action), e.action.prior_f32().map(f32::to_bits)))
            .collect();
        v.sort();
        v
    };
    let mut moved = 0;
    let roots: Vec<_> = common::states(40, 11)
        .into_iter()
        .filter(|s| s.available_actions.team.is_some() && s.get_all_actions().len() >= 8)
        .take(8)
        .collect();
    for (i, state) in roots.iter().enumerate() {
        let mut greedy = bot();
        let (_, clean, _) = greedy.get_action_explore(state, ExploreStep::default());
        let mut explorer = bot();
        let (_, noisy, outcome) = explorer.get_action_explore(state, noise(3 + i as u64));
        assert!(outcome.noised, "root {i}: a fresh root must take the noise");
        assert_eq!(
            priors(&clean),
            priors(&noisy),
            "root {i}: the sample must keep the pre-noise priors"
        );
        assert_ne!(
            searched_priors(&greedy),
            searched_priors(&explorer),
            "root {i}: the search itself must run on the noisy priors"
        );
        moved += (visits(&clean) != visits(&noisy)) as usize;
    }
    // The heuristic search is Q-dominated at c = 10, so a clear best move keeps its visits
    // whatever the priors say; across eight roots the noise must still move some searches.
    assert!(moved > 0, "ε = 0.5 left every root's visit distribution unchanged");
}

#[test]
fn noise_is_deterministic_in_its_seed() {
    let state = wide_state();
    let (_, a, _) = bot().get_action_explore(&state, noise(5));
    let (_, b, _) = bot().get_action_explore(&state, noise(5));
    assert_eq!(stats(&a), stats(&b));
}

#[test]
fn noise_takes_effect_on_fresh_roots_only() {
    let mut state = wide_state();
    let mut bot = bot();
    let (mut fresh, mut reused) = (0, 0);
    for i in 0..12 {
        if state.info.game_over || state.available_actions.team.is_none() {
            break;
        }
        let (action, _, outcome) = bot.get_action_explore(&state, noise(100 + i));
        let reuse = bot.last_search().unwrap().reuse.outcome;
        if reuse == ReuseOutcome::Reused {
            reused += 1;
            assert!(
                !outcome.noised,
                "decision {i}: a reused root was expanded without noise"
            );
        } else {
            fresh += 1;
            assert!(
                outcome.noised,
                "decision {i}: fresh root ({reuse:?}) must take the noise"
            );
        }
        state.step(action).unwrap();
        bot.release_stale_tree(&state);
    }
    assert!(
        fresh > 0 && reused > 0,
        "fresh {fresh}, reused {reused}: the walk should see both"
    );
}

#[test]
fn a_sampled_move_is_a_visited_root_child() {
    let state = wide_state();
    for (i, u) in [0.0, 0.3, 0.7, 0.999].into_iter().enumerate() {
        let step = ExploreStep {
            noise: None,
            sample: Some(SampleSpec { temperature: 1.0, u }),
        };
        let (action, sample, outcome) = bot().get_action_explore(&state, step);
        assert!(outcome.sampled);
        assert_eq!(sample.chosen_action, action, "the sample records the move played");
        let child = sample
            .children
            .iter()
            .find(|c| c.action == action)
            .expect("a root child");
        assert!(child.visits > 0, "draw {i}: sampled an unvisited child");
    }
    // A near-zero temperature collapses onto the most-visited child.
    let step = ExploreStep {
        noise: None,
        sample: Some(SampleSpec {
            temperature: 0.01,
            u: 0.5,
        }),
    };
    let (action, sample, _) = bot().get_action_explore(&state, step);
    let top = sample.children.iter().max_by_key(|c| c.visits).unwrap();
    assert_eq!(action, top.action);
}
