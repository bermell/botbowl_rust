//! Plan 038 micro-benchmark: what does `encode` cost against the two things a
//! state-keyed memo would pay instead — a `GameState` clone and a `GameState`
//! comparison? Release build, 16x9 board, 4 players a side.
//!
//! `cargo run --release -p botbowl-nn --example encode_bench`

use std::time::Instant;

use botbowl_curriculum::random_start::{generate_random_start, RandomStartConfig};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action, BoardDims};
use botbowl_engine::core::table::PosAT;
use botbowl_nn::encode::encode;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn time<R>(label: &str, iters: usize, mut f: impl FnMut() -> R) {
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    let per = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!("{label:<40} {per:8.2} µs");
}

fn main() {
    let cfg = RandomStartConfig {
        board_dims: Some(BoardDims::new(16, 9, 4)),
        ..Default::default()
    };
    // A decision state without paths, and one with the pathfinder's offerings
    // (activate the first mover-side player that can move).
    let mut no_paths: Option<GameState> = None;
    let mut with_paths: Option<GameState> = None;
    for seed in 0..200u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut s = generate_random_start(&cfg, &mut rng);
        s.set_logging_state(false);
        if s.available_actions.has_paths() {
            with_paths.get_or_insert(s);
            continue;
        }
        let starts: Vec<_> = s
            .get_all_actions()
            .into_iter()
            .filter(|a| matches!(a, Action::Positional(PosAT::StartMove, _)))
            .collect();
        if no_paths.is_none() {
            no_paths = Some(s.clone());
        }
        if with_paths.is_none() {
            if let Some(a) = starts.first() {
                s.step(*a).unwrap();
                if s.available_actions.has_paths() {
                    with_paths = Some(s);
                }
            }
        }
        if no_paths.is_some() && with_paths.is_some() {
            break;
        }
    }
    let no_paths = no_paths.expect("a no-path decision state");
    let with_paths = with_paths.expect("a decision state with paths");
    let reachable = with_paths.get_paths().unwrap().iter().filter(|n| n.is_some()).count();
    println!("with_paths: {reachable} reachable squares\n");

    const N: usize = 5000;
    for (label, s) in [("no paths", &no_paths), ("paths offered", &with_paths)] {
        println!("--- {label} ---");
        time("encode", N, || encode(s));
        time("GameState::clone", N, || s.clone());
        let other = s.clone();
        time("GameState == (equal)", N, || *s == other);
        let enc = encode(s);
        let enc2 = encode(s);
        time("Encoded spatial+global == (equal)", N, || {
            enc.spatial == enc2.spatial && enc.global == enc2.global
        });
        println!();
    }
    time("GameState == (different)", N, || no_paths == with_paths);
}
