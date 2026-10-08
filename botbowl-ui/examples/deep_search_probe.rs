//! Plan 060 follow-up: reproduce a pathologically deep search from a corpus sample and dump the
//! deepest materialised line (procedure, ball, bounce squares per ply).
//!
//!   deep_search_probe FILE SEED DRIVE SAMPLE CONFIG [MODEL.onnx] [--dump N]
use std::io::BufRead;
use std::path::Path;
use std::time::Instant;

use botbowl_data::Trajectory;
use botbowl_mcts::{BbAction, SearchBudget};
use botbowl_play::bots::{load_mcts_config, load_nn, make_mcts, Evaluator, SearchConfig};
use botbowl_play::drives::position_state;
use botbowl_play::generate::RandomStartBias;

fn bias_of(meta: &botbowl_data::TrajectoryMeta) -> RandomStartBias {
    let f = |k: &str| -> f32 { meta.extra[k].parse().unwrap() };
    let temperature = f("temperature");
    RandomStartBias {
        ball_distance: f("ball_distance"),
        front_line: f("front_line"),
        mark_teammate: f("mark_teammate"),
        mark_opponent: f("mark_opponent"),
        own_side: f("own_side"),
        temperature,
        temperature2: temperature,
        carried_prob: f("carried_prob"),
        line_fraction: f("line_fraction"),
        pocket_fraction: f("pocket_fraction"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let file = &args[1];
    let seed: u64 = args[2].parse().unwrap();
    let drive = &args[3];
    let idx: usize = args[4].parse().unwrap();
    let cfg = load_mcts_config(Path::new(&args[5])).unwrap();
    let model = args.get(6).filter(|s| s.ends_with(".onnx")).cloned();
    let dump: usize = args
        .iter()
        .position(|a| a == "--dump")
        .map(|i| args[i + 1].parse().unwrap())
        .unwrap_or(0);
    let iters: usize = std::env::var("ITERS").ok().map(|s| s.parse().unwrap()).unwrap_or(1000);

    let reader = std::io::BufReader::new(std::fs::File::open(file).unwrap());
    let trajs: Vec<Trajectory> = reader
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| l[..l.len().min(4000)].contains(&format!("\"seed\":{seed},")))
        .map(|l| serde_json::from_str::<Trajectory>(&l).unwrap())
        .collect();
    let drive_n = |t: &Trajectory| t.meta.extra.get("drive").cloned().unwrap_or_default();
    let traj = trajs.iter().find(|t| &drive_n(t) == drive).expect("trajectory");
    // Replay from the seed (a deserialised state has no path buffer): the first drive, then this.
    let mut state = position_state(&bias_of(&traj.meta), traj.meta.board_dims, seed);
    let mut chain: Vec<&Trajectory> = Vec::new();
    if drive != "1" {
        chain.push(trajs.iter().find(|t| drive_n(t) == "1").expect("first drive"));
    }
    for t in chain {
        for s in &t.samples {
            assert!(state == s.state, "first drive diverged");
            state.step(s.chosen_action).unwrap();
        }
    }
    for s in &traj.samples[..idx] {
        assert!(state == s.state, "replay diverged");
        state.step(s.chosen_action).unwrap();
    }
    assert!(state == traj.samples[idx].state, "replay diverged at the sample");
    println!(
        "root: proc {:?} ball {:?} bounce {:?} dims {:?}",
        state.proc_stack_iter().map(|p| format!("{p:?}")).collect::<Vec<_>>(),
        state.ball,
        state.bounce_squares,
        state.board_dims
    );

    let evaluator = if model.is_some() {
        Evaluator::Nn
    } else {
        Evaluator::Heuristic
    };
    let nn = load_nn(evaluator, model.as_deref(), "model", None).unwrap();
    let search = SearchConfig {
        budget: SearchBudget::Iterations(iters),
        workers: 1,
        puct: None,
        horizon_turns: None,
        fpu_reduction: None,
        config: Some(cfg.config),
    };
    let mut bot = make_mcts(&search, evaluator, nn.as_ref());
    let t0 = Instant::now();
    let (action, sample) = bot.get_action_with_record(&state);
    let secs = t0.elapsed().as_secs_f64();
    println!("chose {action:?} in {secs:.3}s");
    println!("tree {}", serde_json::to_string(&sample.tree).unwrap());

    if dump == 0 {
        return;
    }
    // Walk the deepest materialised line: at each node prefer an expanded child, then visits.
    let mut path: Vec<BbAction> = Vec::new();
    for ply in 0..dump {
        let Some(view) = bot.explore(&path, true) else { break };
        if let Some(st) = &view.state {
            let all: Vec<_> = st.proc_stack_iter().collect();
            let top: Vec<String> = all
                .iter()
                .rev()
                .take(3)
                .map(|p| format!("{p:?}").chars().take(70).collect())
                .collect();
            println!(
                "{ply:5} {:?} v{} ball {:?} bounce_len {} roll {:?} | {}",
                view.stats.player,
                view.stats.visits,
                st.ball,
                st.bounce_squares.len(),
                st.pending_roll,
                top.join(" < ")
            );
        }
        let mut best: Option<(bool, u32, BbAction)> = None;
        for e in &view.children {
            let mut p = path.clone();
            p.push(e.action.clone());
            let expanded = bot.explore(&p, false).map(|v| !v.children.is_empty()).unwrap_or(false);
            let key = (expanded, e.stats.visits);
            if best.as_ref().map(|b| (b.0, b.1) < key).unwrap_or(true) {
                best = Some((expanded, e.stats.visits, e.action.clone()));
            }
        }
        let Some((_, _, a)) = best else { break };
        if ply < 40 || ply % 50 == 0 {
            println!("      -> {a:?}");
        }
        path.push(a);
    }
}
