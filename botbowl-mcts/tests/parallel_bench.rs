use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::run_trials;
use botbowl_mcts::{MctsBot, SearchBudget};

#[test]
#[ignore = "manual wall-clock bench"]
fn bench_parallel_vs_serial() {
    let lecture = ScoreTdEasy::new();
    let n_par_default = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    eprintln!("available_parallelism = {}", n_par_default);

    let iters = std::env::var("PARALLEL_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20000);
    let trials = std::env::var("PARALLEL_TRIALS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(2);
    let n_par = std::env::var("PARALLEL_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(n_par_default);

    let t0 = std::time::Instant::now();
    let mut agent = MctsBot::new(SearchBudget::Iterations(iters)).with_workers(1);
    let _ = run_trials(&lecture, &mut agent, trials, 0xCAFE_1234, 400);
    let serial = t0.elapsed();
    eprintln!(
        "serial   (1 worker, {} iters): {} trials in {:?}",
        iters, trials, serial
    );

    let t0 = std::time::Instant::now();
    let mut agent = MctsBot::new(SearchBudget::Iterations(iters)).with_workers(n_par);
    let _ = run_trials(&lecture, &mut agent, trials, 0xCAFE_1234, 400);
    let parallel = t0.elapsed();
    eprintln!(
        "parallel ({} workers, {} iters): {} trials in {:?}",
        n_par, iters, trials, parallel
    );
    eprintln!("speedup: {:.2}x", serial.as_secs_f64() / parallel.as_secs_f64());
}
