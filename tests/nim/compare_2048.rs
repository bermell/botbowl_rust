//! Compare final scores: random baseline, heuristic, and MCTS at one or more move budgets.
//!
//! Games are evaluated **in parallel** (one seed per Rayon task) with Rayon. Set `RAYON_NUM_THREADS`
//! to cap threads (e.g. `RAYON_NUM_THREADS=4` for four games at a time).
//!
//! Usage:
//! - **Iteration budgets (default):** `compare_2048 [--rollout] [num_games] [base_seed] [iter1] [iter2] ...`
//! - **Wall-clock budgets:** `compare_2048 [--rollout] --time-ms <list> [num_games] [base_seed]`
//!
//! **`--time-ms`** must be followed immediately by **one** argument: comma-separated (`150,300,600`) or
//! whitespace-separated inside quotes (`"150 300 600"`). Use **`default`** for the built-in ms list
//! (`100 200 400 800 1600` ms). After that, only **`num_games`** and **`base_seed`** are read (defaults
//! 100 and 0 if omitted).
//!
//! - **`--rollout`**: MCTS leaf values use a **random rollout** to game over (`LeafScoreMode::RandomRollout`);
//!   default is current cumulative score only (`CurrentScore`).
//! - Without **`--time-ms`**, remaining numbers are **iteration** counts per action move.
//! - Defaults: **100** games, base seed **0**, iteration budgets **500 1000 2000 4000 8000**.
//! - Game `i` uses seed `base_seed + i`. Each seed runs: random baseline, heuristic, and each MCTS
//!   budget independently (same RNG seed for initial tile and spawns within each run).
//! - For each MCTS budget, the summary also reports **total `tree.step()` counts per game** (warmup
//!   at every ply plus the search budget on action plies), so wall-time runs can be compared by
//!   actual work done.

use std::time::Duration;

use rayon::prelude::*;

use recon_mcts_test_nim::play_2048::{self, MctsMoveBudget, DEFAULT_WARMUP_STEPS};
use recon_mcts_test_nim::test_mcts_2048::{Game2048Dynamics, LeafScoreMode};

fn default_mcts_iteration_budgets() -> Vec<MctsMoveBudget> {
    [500usize, 1000, 2000, 4000, 8000]
        .iter()
        .map(|&n| MctsMoveBudget::Iterations(n))
        .collect()
}

fn default_mcts_time_budgets() -> Vec<MctsMoveBudget> {
    [100u64, 200, 400, 800, 1600]
        .iter()
        .map(|&ms| MctsMoveBudget::WallTime(Duration::from_millis(ms)))
        .collect()
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

struct U64Summary {
    mean: f64,
    std_dev: f64,
    median: f64,
    q25: f64,
    q75: f64,
    min: u64,
    max: u64,
}

impl U64Summary {
    fn from_values(values: &[u64]) -> Self {
        let n = values.len();
        if n == 0 {
            return U64Summary {
                mean: 0.0,
                std_dev: 0.0,
                median: 0.0,
                q25: 0.0,
                q75: 0.0,
                min: 0,
                max: 0,
            };
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let sum: u128 = values.iter().map(|&x| x as u128).sum();
        let mean = sum as f64 / n as f64;
        let variance = if n > 1 {
            values
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
        U64Summary {
            mean,
            std_dev,
            median: median_from_sorted_u64(&sorted),
            q25: quantile_linear_u64(&sorted, 0.25),
            q75: quantile_linear_u64(&sorted, 0.75),
            min: sorted[0],
            max: sorted[n - 1],
        }
    }
}

fn median_from_sorted_u64(sorted: &[u64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
    }
}

fn quantile_linear_u64(sorted: &[u64], q: f64) -> f64 {
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

fn print_u64_block(label: &str, values: &[u64]) {
    let s = U64Summary::from_values(values);
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

fn mcts_block_title(budget: &MctsMoveBudget, leaf_rollout: bool) -> String {
    let leaf = if leaf_rollout {
        "random rollout"
    } else {
        "current score"
    };
    match budget {
        MctsMoveBudget::Iterations(n) => format!(
            "MCTS ({} iters/move, {} warmup; leaf: {})",
            n, DEFAULT_WARMUP_STEPS, leaf
        ),
        MctsMoveBudget::WallTime(d) => format!(
            "MCTS ({} ms wall time/move, {} warmup; leaf: {})",
            d.as_millis(),
            DEFAULT_WARMUP_STEPS,
            leaf
        ),
    }
}

fn parse_time_ms_list_argument(spec: &str) -> Result<Vec<MctsMoveBudget>, String> {
    let s = spec.trim();
    if s.eq_ignore_ascii_case("default") {
        return Ok(default_mcts_time_budgets());
    }
    let nums: Vec<u64> = if s.contains(',') {
        s.split(',')
            .map(|t| t.trim())
            .filter_map(|t| t.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .collect()
    } else {
        s.split_whitespace()
            .filter_map(|t| t.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .collect()
    };
    if nums.is_empty() {
        return Err(
            "--time-ms value must be \"default\", comma-separated ms (150,300,600), or one quoted token with spaces (\"150 300 600\")"
                .into(),
        );
    }
    Ok(nums
        .into_iter()
        .map(|ms| MctsMoveBudget::WallTime(Duration::from_millis(ms)))
        .collect())
}

fn parse_iteration_positionals(
    args: &[String],
) -> Result<(usize, u64, Vec<MctsMoveBudget>), String> {
    match args.len() {
        0 => Ok((100, 0, default_mcts_iteration_budgets())),
        1 => Ok((
            args[0]
                .parse()
                .map_err(|_| "num_games must be a number".to_string())?,
            0,
            default_mcts_iteration_budgets(),
        )),
        2 => Ok((
            args[0]
                .parse()
                .map_err(|_| "num_games must be a number".to_string())?,
            args[1]
                .parse()
                .map_err(|_| "base_seed must be a number".to_string())?,
            default_mcts_iteration_budgets(),
        )),
        _ => {
            let num_games = args[0]
                .parse()
                .map_err(|_| "num_games must be a number".to_string())?;
            let base_seed = args[1]
                .parse()
                .map_err(|_| "base_seed must be a number".to_string())?;
            let iters: Vec<usize> = args[2..]
                .iter()
                .filter_map(|s| s.parse().ok())
                .filter(|&n| n > 0)
                .collect();
            if iters.is_empty() {
                return Err(
                    "no valid iteration budgets (positive integers) after base_seed".into(),
                );
            }
            Ok((
                num_games,
                base_seed,
                iters.into_iter().map(MctsMoveBudget::Iterations).collect(),
            ))
        }
    }
}

/// Positionals after flags: only `num_games` and optionally `base_seed`.
fn parse_time_mode_positionals(args: &[String]) -> Result<(usize, u64), String> {
    match args.len() {
        0 => Ok((100, 0)),
        1 => Ok((
            args[0]
                .parse()
                .map_err(|_| "num_games must be a number".to_string())?,
            0,
        )),
        2 => Ok((
            args[0]
                .parse()
                .map_err(|_| "num_games must be a number".to_string())?,
            args[1]
                .parse()
                .map_err(|_| "base_seed must be a number".to_string())?,
        )),
        n => Err(format!(
            "with --time-ms, use at most num_games and base_seed after the ms list (got {} extra argument(s))",
            n - 2
        )),
    }
}

fn parse_args() -> Result<(usize, u64, Vec<MctsMoveBudget>, bool), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut rollout = false;
    let mut time_ms_spec: Option<String> = None;
    let mut positionals: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--rollout" => {
                rollout = true;
                i += 1;
            }
            "--time-ms" => {
                i += 1;
                let spec = args.get(i).ok_or_else(|| {
                    "--time-ms must be followed by a value (e.g. 150,300,600)".to_string()
                })?;
                if spec.starts_with('-') {
                    return Err("--time-ms must be followed by a ms list, not another flag".into());
                }
                time_ms_spec = Some(spec.clone());
                i += 1;
            }
            _ => {
                positionals.push(args[i].clone());
                i += 1;
            }
        }
    }

    if let Some(spec) = time_ms_spec {
        let budgets = parse_time_ms_list_argument(&spec)?;
        let (num_games, base_seed) = parse_time_mode_positionals(&positionals)?;
        Ok((num_games, base_seed, budgets, rollout))
    } else {
        let (num_games, base_seed, budgets) = parse_iteration_positionals(&positionals)?;
        Ok((num_games, base_seed, budgets, rollout))
    }
}

fn main() {
    let (num_games, base_seed, mcts_budgets, leaf_rollout) = match parse_args() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("compare_2048: {}", e);
            std::process::exit(2);
        }
    };
    let time_ms_mode = mcts_budgets
        .first()
        .map(|b| matches!(b, MctsMoveBudget::WallTime(_)))
        .unwrap_or(false);
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
    if time_ms_mode {
        println!(
            "MCTS wall-time budgets per action move (ms): {:?}",
            mcts_budgets
                .iter()
                .map(|b| match b {
                    MctsMoveBudget::WallTime(d) => d.as_millis(),
                    _ => 0,
                })
                .collect::<Vec<_>>()
        );
    } else {
        println!(
            "MCTS iteration budgets per action move: {:?}",
            mcts_budgets
                .iter()
                .map(|b| match b {
                    MctsMoveBudget::Iterations(n) => *n,
                    _ => 0,
                })
                .collect::<Vec<_>>()
        );
    }
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

    let rows: Vec<(i32, i32, Vec<(i32, u64)>)> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let seed = base_seed.wrapping_add(i as u64);
            let heuristic = play_2048::run_heuristic_game(seed);
            let random = play_2048::run_random_baseline_game(seed);
            let mcts: Vec<(i32, u64)> = mcts_budgets
                .iter()
                .map(|b| play_2048::run_mcts_game(seed, *b, DEFAULT_WARMUP_STEPS, mcts_dynamics))
                .collect();
            (heuristic, random, mcts)
        })
        .collect();

    let mut heuristic_scores = vec![0i32; num_games];
    let mut random_scores = vec![0i32; num_games];
    let mut mcts_by_budget: Vec<Vec<i32>> =
        mcts_budgets.iter().map(|_| vec![0i32; num_games]).collect();
    let mut mcts_steps_by_budget: Vec<Vec<u64>> =
        mcts_budgets.iter().map(|_| vec![0u64; num_games]).collect();

    for (i, (h, r, m)) in rows.into_iter().enumerate() {
        heuristic_scores[i] = h;
        random_scores[i] = r;
        for (j, (score, steps)) in m.iter().enumerate() {
            mcts_by_budget[j][i] = *score;
            mcts_steps_by_budget[j][i] = *steps;
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
            for (j, budget) in mcts_budgets.iter().enumerate() {
                print!(
                    "  mcts@{:>7}: {:>6}  steps {:>12}",
                    budget.label_short(),
                    mcts_by_budget[j][i],
                    mcts_steps_by_budget[j][i]
                );
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
    for (j, budget) in mcts_budgets.iter().enumerate() {
        println!();
        print_block(&mcts_block_title(budget, leaf_rollout), &mcts_by_budget[j]);
        println!();
        print_u64_block(
            &format!(
                "Tree step() total per game — {} (warmup + search; same column as above)",
                budget.label_short()
            ),
            &mcts_steps_by_budget[j],
        );
    }
}
