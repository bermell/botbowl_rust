//! Compare final scores: random baseline, heuristic, and MCTS at one or more iteration budgets.
//!
//! Games are evaluated **in parallel** (one seed per Rayon task) with Rayon. Set `RAYON_NUM_THREADS`
//! to cap threads (e.g. `RAYON_NUM_THREADS=4` for four games at a time).
//!
//! Usage: `compare_2048 [--rollout] [num_games] [base_seed] [iter1] [iter2] ...`
//!
//! - **`--rollout`**: MCTS leaf values use a **random rollout** to game over (`LeafScoreMode::RandomRollout`);
//!   default is current cumulative score only (`CurrentScore`).
//! - Defaults: **100** games, base seed **0**, MCTS iterations per move **500 1000 2000 4000 8000**.
//! - Game `i` uses seed `base_seed + i`. Each seed runs: random baseline, heuristic, and each MCTS
//!   budget independently (same RNG seed for initial tile and spawns within each run).

use rayon::prelude::*;

use recon_mcts_test_nim::play_2048::{self, DEFAULT_WARMUP_STEPS};
use recon_mcts_test_nim::test_mcts_2048::{Game2048Dynamics, LeafScoreMode};

fn default_mcts_iterations() -> Vec<usize> {
    vec![500, 1000, 2000, 4000, 8000]
}

struct ScoreSummary {
    mean: f64,
    std_dev: f64,
    median: f64,
    q25: f64,
    q75: f64,
    min: i32,
    max: i32,
}

impl ScoreSummary {
    fn from_scores(scores: &[i32]) -> Self {
        let n = scores.len();
        if n == 0 {
            return ScoreSummary {
                mean: 0.0,
                std_dev: 0.0,
                median: 0.0,
                q25: 0.0,
                q75: 0.0,
                min: 0,
                max: 0,
            };
        }
        let mut sorted = scores.to_vec();
        sorted.sort_unstable();
        let sum: i64 = scores.iter().map(|&x| x as i64).sum();
        let mean = sum as f64 / n as f64;
        let variance = if n > 1 {
            scores
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        ScoreSummary {
            mean,
            std_dev,
            median: median_from_sorted(&sorted),
            q25: quantile_linear(&sorted, 0.25),
            q75: quantile_linear(&sorted, 0.75),
            min: sorted[0],
            max: sorted[n - 1],
        }
    }
}

fn median_from_sorted(sorted: &[i32]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
    }
}

/// Linear interpolation between closest ranks (common for quartiles on small samples).
fn quantile_linear(sorted: &[i32], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0] as f64;
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo] as f64
    } else {
        let w = pos - lo as f64;
        sorted[lo] as f64 * (1.0 - w) + sorted[hi] as f64 * w
    }
}

fn print_block(label: &str, scores: &[i32]) {
    let s = ScoreSummary::from_scores(scores);
    println!("{}", label);
    println!("  mean:        {:.1}", s.mean);
    println!("  std dev:     {:.1}", s.std_dev);
    println!("  median:      {:.1}", s.median);
    println!(
        "  quartiles:   p25 {:.1}  p75 {:.1}  (IQR {:.1})",
        s.q25,
        s.q75,
        s.q75 - s.q25
    );
    println!("  min / max:   {} / {}", s.min, s.max);
}

fn parse_args() -> (usize, u64, Vec<usize>, bool) {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let leaf_rollout = args.iter().any(|a| a == "--rollout");
    args.retain(|a| a != "--rollout");
    match args.len() {
        0 => (100, 0, default_mcts_iterations(), leaf_rollout),
        1 => (
            args[0].parse().unwrap_or(100),
            0,
            default_mcts_iterations(),
            leaf_rollout,
        ),
        2 => (
            args[0].parse().unwrap_or(100),
            args[1].parse().unwrap_or(0),
            default_mcts_iterations(),
            leaf_rollout,
        ),
        _ => {
            let num_games = args[0].parse().unwrap_or(100);
            let base_seed = args[1].parse().unwrap_or(0);
            let iters: Vec<usize> = args[2..]
                .iter()
                .filter_map(|s| s.parse::<usize>().ok())
                .filter(|&n| n > 0)
                .collect();
            let iters = if iters.is_empty() {
                default_mcts_iterations()
            } else {
                iters
            };
            (num_games, base_seed, iters, leaf_rollout)
        }
    }
}

fn main() {
    let (num_games, base_seed, mcts_iters, leaf_rollout) = parse_args();
    let mcts_dynamics = Game2048Dynamics {
        leaf: if leaf_rollout {
            LeafScoreMode::RandomRollout
        } else {
            LeafScoreMode::CurrentScore
        },
    };

    println!(
        "2048 comparison: {} games (parallel), warmup {} steps/move before reading root; seeds {}..{}",
        num_games,
        DEFAULT_WARMUP_STEPS,
        base_seed,
        base_seed.saturating_add(num_games.saturating_sub(1) as u64)
    );
    println!("MCTS iteration budgets per action move: {:?}", mcts_iters);
    println!(
        "MCTS leaf scoring: {}",
        if leaf_rollout {
            "random rollout to terminal"
        } else {
            "current cumulative score (no rollout)"
        }
    );
    println!(
        "Rayon threads: {} (override with RAYON_NUM_THREADS)",
        rayon::current_num_threads()
    );
    println!();

    let rows: Vec<(i32, i32, Vec<i32>)> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let seed = base_seed.wrapping_add(i as u64);
            let heuristic = play_2048::run_heuristic_game(seed);
            let random = play_2048::run_random_baseline_game(seed);
            let mcts: Vec<i32> = mcts_iters
                .iter()
                .map(|&it| play_2048::run_mcts_game(seed, it, DEFAULT_WARMUP_STEPS, mcts_dynamics))
                .collect();
            (heuristic, random, mcts)
        })
        .collect();

    let mut heuristic_scores = vec![0i32; num_games];
    let mut random_scores = vec![0i32; num_games];
    let mut mcts_by_iters: Vec<Vec<i32>> =
        mcts_iters.iter().map(|_| vec![0i32; num_games]).collect();

    for (i, (h, r, m)) in rows.into_iter().enumerate() {
        heuristic_scores[i] = h;
        random_scores[i] = r;
        for (j, score) in m.iter().enumerate() {
            mcts_by_iters[j][i] = *score;
        }
    }

    const PER_GAME_ROWS_MAX: usize = 40;
    let print_each_game = num_games <= PER_GAME_ROWS_MAX;

    if print_each_game {
        for i in 0..num_games {
            print!(
                "game {:>3}: random {:>6}  heuristic {:>6}",
                i + 1,
                random_scores[i],
                heuristic_scores[i]
            );
            for (j, &iters) in mcts_iters.iter().enumerate() {
                print!("  mcts@{:>5}: {:>6}", iters, mcts_by_iters[j][i]);
            }
            println!();
        }
    } else {
        println!(
            "(Omitted {} per-game lines; only summary below. Use num_games <= {} for full table.)\n",
            num_games, PER_GAME_ROWS_MAX
        );
    }

    println!();
    print_block(
        "Random baseline (uniform legal move, sorted tie-break)",
        &random_scores,
    );
    println!();
    print_block(
        "Heuristic (one-step snake / empty / score)",
        &heuristic_scores,
    );
    for (j, &iters) in mcts_iters.iter().enumerate() {
        println!();
        print_block(
            &format!(
                "MCTS ({} iters/move, {} warmup; leaf: {})",
                iters,
                DEFAULT_WARMUP_STEPS,
                if leaf_rollout {
                    "random rollout"
                } else {
                    "current score"
                }
            ),
            &mcts_by_iters[j],
        );
    }
}
