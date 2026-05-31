//! Single-call DAG-shape probe used by plan 013.
//!
//! Runs `MctsBot::get_action` exactly once at a small iteration
//! budget and inspects the resulting tree's registry counters + depth
//! distribution. Each test invocation honours `BLOOD_MCTS_MEMORY` and
//! `BLOOD_MCTS_WORKERS`, so the same binary covers all three markers
//! and any worker count.
//!
//! `BLOOD_MCTS_STATS=1` is set unconditionally inside the test so the
//! caller doesn't have to remember it.
//!
//! Usage:
//!
//! ```sh
//! for MODE in get store; do
//!   for ITERS in 50 200 500 1000; do
//!     BLOOD_MCTS_MEMORY=$MODE \
//!         ./target/release/deps/tree_shape-* \
//!         --ignored --nocapture "iters_${ITERS}_single" \
//!         --test-threads=1
//!   done
//! done
//! ```

use botbowl_curriculum::lectures::get_the_ball::GetTheBallEasy;
use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::Lecture;
use botbowl_engine::bots::Bot;
use botbowl_mcts::MctsBot;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 0xCAFE_1234;

fn build_state() -> botbowl_engine::core::gamestate::GameState {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    ScoreTdEasy::new().setup(&mut rng)
}

fn build_gtb_state() -> botbowl_engine::core::gamestate::GameState {
    let mut rng = ChaCha8Rng::seed_from_u64(0xF00D_9012);
    GetTheBallEasy::new().setup(&mut rng)
}

fn run_once(iters: usize) {
    std::env::set_var("BLOOD_MCTS_STATS", "1");
    let mut agent = MctsBot::new(iters).with_workers(1);
    let state = build_state();
    let t0 = std::time::Instant::now();
    let action = agent.get_action(&state);
    let elapsed = t0.elapsed();
    eprintln!(
        "TREE_SHAPE iters={iters} elapsed_ms={:.1} chosen_action={action:?}",
        elapsed.as_secs_f64() * 1e3
    );
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn iters_50_single() {
    run_once(50);
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn iters_200_single() {
    run_once(200);
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn iters_500_single() {
    run_once(500);
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn iters_1000_single() {
    run_once(1000);
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn iters_2000_single() {
    run_once(2000);
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn iters_5000_single() {
    run_once(5000);
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn iters_10000_single() {
    run_once(10000);
}

fn run_once_gtb(iters: usize) {
    std::env::set_var("BLOOD_MCTS_STATS", "1");
    let mut agent = MctsBot::new(iters).with_workers(1);
    let state = build_gtb_state();
    let t0 = std::time::Instant::now();
    let action = agent.get_action(&state);
    let elapsed = t0.elapsed();
    eprintln!(
        "TREE_SHAPE_GTB iters={iters} elapsed_ms={:.1} chosen_action={action:?}",
        elapsed.as_secs_f64() * 1e3
    );
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn gtb_iters_1000() {
    run_once_gtb(1000);
}

#[test]
#[ignore = "manual DAG-shape probe — run with --ignored"]
fn gtb_iters_5000() {
    run_once_gtb(5000);
}
