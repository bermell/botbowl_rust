//! Plan 048: an exploring generator stamps what it did into each trajectory and crosses the hub
//! intact. `exploration: None` stays the greedy generator.
//!
//! Not tested here: whole-trajectory reproducibility. The greedy generator does not reproduce
//! either (recon_mcts HashMap order, see this crate's CLAUDE.md); the noise draw itself is pinned
//! deterministic in `botbowl-mcts/tests/root_noise.rs`.

use botbowl_curriculum::lecture::Difficulty;
use botbowl_engine::core::model::BoardDims;
use botbowl_play::board_sizes::SizeDist;
use botbowl_play::bots::{Evaluator, SearchConfig};
use botbowl_play::generate::{play_trajectory, Exploration, GenMode, GenerateConfig, RandomStartBias};

fn cfg(exploration: Option<Exploration>) -> GenerateConfig {
    GenerateConfig {
        mode: GenMode::RandomStart,
        search: SearchConfig::iterations(40),
        evaluator: Evaluator::Heuristic,
        model: None,
        max_steps: 400,
        lecture: None,
        difficulty: Difficulty::Easy,
        bias: RandomStartBias::default(),
        board_sizes: Some(SizeDist::single(BoardDims::new(16, 9, 4))),
        config_name: None,
        exploration,
    }
}

const EXPLORE: Exploration = Exploration {
    noise_epsilon: 0.25,
    noise_alpha: 10.0,
    sample_moves: 2,
    temperature: 1.0,
};

#[test]
fn an_exploring_trajectory_says_what_it_did() {
    let t = play_trajectory(&cfg(Some(EXPLORE)), None, 3).unwrap().unwrap();
    let extra = |k: &str| t.meta.extra.get(k).cloned().unwrap_or_default();
    assert_eq!(extra("explore"), EXPLORE.label());
    assert!(t.meta.home_bot.contains(&EXPLORE.label()), "label: {}", t.meta.home_bot);
    let n = |k: &str| extra(k).parse::<usize>().unwrap();
    assert!(n("explore_noised") > 0, "fresh roots should have taken the noise");
    assert!(n("explore_noised") <= t.samples.len());
    // Two sampled moves per side, fewer only if the drive ended first.
    assert!(n("explore_sampled") <= 4 && n("explore_sampled") <= t.samples.len());
    assert!(n("explore_deviated") <= t.samples.len());
}

#[test]
fn the_greedy_generator_carries_no_exploration_stamp() {
    let t = play_trajectory(&cfg(None), None, 3).unwrap().unwrap();
    assert!(!t.meta.extra.keys().any(|k| k.starts_with("explore")));
    assert!(!t.meta.home_bot.contains("explore"));
}

#[test]
fn the_flags_turn_exploration_on_only_when_asked() {
    assert_eq!(Exploration::from_flags(None, 10.0, 0, 1.0), None);
    assert_eq!(Exploration::from_flags(Some(0.0), 10.0, 0, 1.0), None);
    assert_eq!(Exploration::from_flags(Some(0.25), 10.0, 2, 1.0), Some(EXPLORE));
    assert_eq!(
        Exploration::from_flags(None, 10.0, 3, 1.0).map(|e| (e.noise_epsilon, e.sample_moves)),
        Some((0.0, 3))
    );
}

#[test]
fn exploration_survives_the_hub_wire_format() {
    let bytes = postcard::to_stdvec(&cfg(Some(EXPLORE))).unwrap();
    let back: GenerateConfig = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(back.exploration, Some(EXPLORE));
}
