//! End-to-end check of the `prepare` binary: build in-memory
//! trajectories, write them as JSONL, run `prepare`, and re-read the
//! `.npy` output — asserting shapes, CSR offsets, and per-sample policy
//! sums.

use std::process::Command;

use botbowl_data::{ChildStat, DatasetWriter, Outcome, Sample, Team, Trajectory, TrajectoryMeta};
use botbowl_engine::core::gamestate::GameStateBuilder;
use botbowl_engine::core::model::{Action, Position};
use botbowl_engine::core::table::{PosAT, SimpleAT};
use botbowl_nn::actions::POLICY_CHANNELS;
use botbowl_nn::encode::{encode, spatial_channel_scales, GLOBAL_FEATURES, SPATIAL_CHANNELS};
use botbowl_nn::npy;

fn child(action: Action, visits: u32, q: i64, solved: bool) -> ChildStat {
    ChildStat {
        action,
        visits,
        q: Some(q),
        prior: Some(1.0),
        solved,
        terminal: solved,
    }
}

fn make_sample(children: Vec<ChildStat>, root_solved: bool) -> Sample {
    let chosen = children[0].action;
    Sample {
        state: GameStateBuilder::new_start_of_game(),
        to_move: Team::Home,
        chosen_action: chosen,
        children,
        root_value: Some(0),
        root_visits: 50,
        root_solved,
        outcome_value: None,
        scripted: false,
        tree: None,
    }
}

#[test]
fn prepare_round_trips_shapes_offsets_and_policy_sums() {
    let tmp = std::env::temp_dir().join(format!("botbowl_nn_prepare_it_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let jsonl = tmp.join("data.jsonl");
    let out = tmp.join("prepared");

    let dims = GameStateBuilder::new_start_of_game().board_dims;

    // Two samples: 3 legal actions (not solved), 2 legal actions (not solved).
    let s1 = make_sample(
        vec![
            child(Action::Positional(PosAT::Move, Position::new((5, 5))), 30, 100, false),
            child(Action::Positional(PosAT::Move, Position::new((6, 5))), 20, 50, false),
            child(Action::Simple(SimpleAT::EndTurn), 10, -10, false),
        ],
        false,
    );
    let s2 = make_sample(
        vec![
            child(Action::Positional(PosAT::Block, Position::new((3, 3))), 40, 200, false),
            child(Action::Simple(SimpleAT::EndTurn), 5, 0, false),
        ],
        false,
    );

    let traj = Trajectory::new(
        TrajectoryMeta::new("test", dims).with_bots("mcts", "mcts"),
        vec![s1, s2],
        Outcome {
            home_score: 1,
            away_score: 0,
            winner: Some(Team::Home),
            game_over: true,
            z_home: 1.0,
            lecture_status: None,
        },
    );

    {
        let mut w = DatasetWriter::create(&jsonl).unwrap();
        w.write(&traj).unwrap();
        w.flush().unwrap();
    }

    let status = Command::new(env!("CARGO_BIN_EXE_prepare"))
        .args(["--in", jsonl.to_str().unwrap(), "--out", out.to_str().unwrap()])
        .status()
        .expect("run prepare");
    assert!(status.success(), "prepare exited with failure");

    let subdir = out.join(format!("dims_{}x{}", dims.width, dims.height));
    assert!(subdir.exists(), "expected dims subdir {}", subdir.display());

    let spatial = npy::read(subdir.join("spatial.npy")).unwrap();
    assert_eq!(
        spatial.shape,
        vec![2, SPATIAL_CHANNELS, dims.height as usize, dims.width as usize]
    );
    assert_eq!(spatial.descr, npy::U1, "spatial must be stored as raw u8 counts");

    // The corpus is only correct if `u8 / manifest scale` is exactly the
    // tensor the live evaluator builds. Both samples encode the same state.
    let scales = spatial_channel_scales();
    let want = encode(&GameStateBuilder::new_start_of_game()).spatial;
    let plane = (dims.height as usize) * (dims.width as usize);
    let raw = spatial.as_u8();
    for sample in 0..2 {
        let row = &raw[sample * want.len()..(sample + 1) * want.len()];
        for (i, (&r, &f)) in row.iter().zip(&want).enumerate() {
            assert_eq!(
                r as f32 / scales[i / plane],
                f,
                "sample {sample} channel {} cell {}",
                i / plane,
                i % plane
            );
        }
    }

    let global = npy::read(subdir.join("global.npy")).unwrap();
    assert_eq!(global.shape, vec![2, GLOBAL_FEATURES]);

    let value = npy::read(subdir.join("value.npy")).unwrap();
    assert_eq!(value.shape, vec![2]);
    // Home mover, z_home = +1 → value target +1 for both samples.
    assert_eq!(value.as_f32(), vec![1.0, 1.0]);

    let chosen = npy::read(subdir.join("chosen.npy")).unwrap();
    assert_eq!(chosen.shape, vec![2]);
    assert_eq!(chosen.as_i64(), vec![0, 0]); // chosen == children[0] in both

    let offsets = npy::read(subdir.join("action_offsets.npy")).unwrap();
    assert_eq!(offsets.shape, vec![3]); // N+1
    let off = offsets.as_i64();
    assert_eq!(off, vec![0, 3, 5]); // 3 actions then 2 actions

    let actions = npy::read(subdir.join("actions.npy")).unwrap();
    assert_eq!(actions.shape, vec![5, 4]); // M=5 legal actions, 4 cols

    let policy = npy::read(subdir.join("policy.npy")).unwrap();
    assert_eq!(policy.shape, vec![5]);
    let probs = policy.as_f32();
    // Per-sample policy sums (CSR ranges) must each be ~1.
    for w in off.windows(2) {
        let (lo, hi) = (w[0] as usize, w[1] as usize);
        let sum: f32 = probs[lo..hi].iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "policy range [{lo},{hi}) sums to {sum}");
    }
    // s1 not solved → π ∝ visits: [30,20,10]/60.
    assert!((probs[0] - 0.5).abs() < 1e-5);
    assert!((probs[1] - 20.0 / 60.0).abs() < 1e-5);
    assert!((probs[2] - 10.0 / 60.0).abs() < 1e-5);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(subdir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["policy_channels"], POLICY_CHANNELS);
    assert_eq!(manifest["spatial_channels"], SPATIAL_CHANNELS);
    assert_eq!(manifest["spatial_dtype"], "u8");
    assert_eq!(
        manifest["spatial_scales"].as_array().unwrap().len(),
        SPATIAL_CHANNELS,
        "the trainer needs one scale per channel"
    );
    assert_eq!(manifest["num_samples"], 2);
    assert_eq!(manifest["num_actions"], 5);

    std::fs::remove_dir_all(&tmp).ok();
}

/// Run `prepare` on one trajectory in a fresh temp dir; returns the dims subdir and the summary line.
fn run_prepare(tag: &str, traj: &Trajectory, extra: &[&str]) -> (std::path::PathBuf, String) {
    let tmp = std::env::temp_dir().join(format!("botbowl_nn_prepare_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let jsonl = tmp.join("data.jsonl");
    let out = tmp.join("prepared");
    {
        let mut w = DatasetWriter::create(&jsonl).unwrap();
        w.write(traj).unwrap();
        w.flush().unwrap();
    }
    let output = Command::new(env!("CARGO_BIN_EXE_prepare"))
        .args(["--in", jsonl.to_str().unwrap(), "--out", out.to_str().unwrap()])
        .args(extra)
        .output()
        .expect("run prepare");
    assert!(output.status.success(), "prepare exited with failure");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let summary = stdout
        .lines()
        .find(|l| l.starts_with("prepare done"))
        .expect("summary line")
        .to_string();
    let dims = traj.meta.board_dims;
    (out.join(format!("dims_{}x{}", dims.width, dims.height)), summary)
}

fn read_manifest(subdir: &std::path::Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(subdir.join("manifest.json")).unwrap()).unwrap()
}

/// A decision with one action left after pruning teaches nothing: `prepare` skips any sample whose
/// root had fewer than two children — the unsearched forced record `MctsBot` writes (scripted, one
/// child) and an older searched one alike — counts them in its summary, and computes the per-drive
/// value weight over the rows it kept. `--keep-forced` keeps them.
#[test]
fn prepare_skips_samples_with_fewer_than_two_children() {
    let dims = GameStateBuilder::new_start_of_game().board_dims;
    let end_player_turn = Action::Simple(SimpleAT::EndPlayerTurn);
    // `MctsBot`'s forced record: unsearched, the one post-pruning action as the only child.
    let mut forced = make_sample(
        vec![ChildStat {
            action: end_player_turn,
            visits: 1,
            q: None,
            prior: None,
            solved: false,
            terminal: false,
        }],
        false,
    );
    forced.scripted = true;
    forced.root_visits = 1;
    forced.root_value = None;
    // A forced decision from before the shortcut: searched, one child.
    let searched_forced = make_sample(vec![child(end_player_turn, 50, 10, false)], false);
    let choice = make_sample(
        vec![
            child(Action::Positional(PosAT::Block, Position::new((3, 3))), 40, 200, false),
            child(Action::Simple(SimpleAT::EndTurn), 5, 0, false),
        ],
        false,
    );
    let traj = Trajectory::new(
        TrajectoryMeta::new("test", dims).with_bots("mcts", "mcts"),
        vec![forced, choice, searched_forced],
        Outcome {
            home_score: 1,
            away_score: 0,
            winner: Some(Team::Home),
            game_over: true,
            z_home: 1.0,
            lecture_status: None,
        },
    );

    let (subdir, summary) = run_prepare("forced", &traj, &[]);
    assert!(summary.contains("kept 1 "), "{summary}");
    assert!(summary.contains("2 with fewer than two actions"), "{summary}");
    let offsets = npy::read(subdir.join("action_offsets.npy")).unwrap().as_i64();
    assert_eq!(offsets, vec![0, 2], "only the two-action decision is kept");
    // The drive's one kept row carries the drive's whole value weight.
    assert_eq!(npy::read(subdir.join("weight.npy")).unwrap().as_f32(), vec![1.0]);
    let manifest = read_manifest(&subdir);
    assert_eq!(manifest["num_samples"], 1);
    assert_eq!(manifest["skip_forced"], true);
    std::fs::remove_dir_all(subdir.parent().unwrap().parent().unwrap()).ok();

    let (subdir, summary) = run_prepare("keep_forced", &traj, &["--keep-forced"]);
    assert!(summary.contains("kept 3 "), "{summary}");
    assert!(summary.contains("0 with fewer than two actions"), "{summary}");
    let offsets = npy::read(subdir.join("action_offsets.npy")).unwrap().as_i64();
    assert_eq!(offsets, vec![0, 1, 3, 4]);
    assert_eq!(
        npy::read(subdir.join("weight.npy")).unwrap().as_f32(),
        vec![1.0 / 3.0; 3]
    );
    assert_eq!(read_manifest(&subdir)["skip_forced"], false);
    std::fs::remove_dir_all(subdir.parent().unwrap().parent().unwrap()).ok();
}

/// Plan 071 R8: `--jobs N` (one part per input, concatenated in input order) writes the same bytes
/// as one serial pass, over several inputs, two board shapes, drives of different lengths and an
/// input whose every sample is filtered out; `--dedup` refuses `--jobs` > 1.
#[test]
fn prepare_jobs_output_is_byte_identical_to_serial() {
    use botbowl_engine::core::model::{BoardDims, HEIGHT, TEAM_SIZE, WIDTH};
    let tmp = std::env::temp_dir().join(format!("botbowl_nn_prepare_jobs_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let big = GameStateBuilder::new_start_of_game().board_dims;
    let small = if WIDTH as i8 >= 16 && HEIGHT as i8 >= 9 && TEAM_SIZE >= 4 {
        BoardDims::new(16, 9, 4)
    } else {
        big
    };
    let outcome = |home: u8| Outcome {
        home_score: home,
        away_score: 1 - home,
        winner: Some(if home == 1 { Team::Home } else { Team::Away }),
        game_over: true,
        z_home: if home == 1 { 1.0 } else { -1.0 },
        lecture_status: None,
    };
    let mut inputs = Vec::new();
    for k in 0..4u8 {
        let path = tmp.join(format!("shard{k}.jsonl"));
        let mut w = DatasetWriter::create(&path).unwrap();
        for t in 0..(k as usize + 1) {
            let dims = if (k as usize + t) % 2 == 0 { big } else { small };
            let mut samples = Vec::new();
            for s in 0..(2 + (t + k as usize) % 3) {
                let mut children = vec![
                    child(
                        Action::Positional(PosAT::Move, Position::new((5, 5))),
                        30 + s as u32,
                        100,
                        false,
                    ),
                    child(Action::Simple(SimpleAT::EndTurn), 10 + t as u32, -10, false),
                ];
                if k == 2 {
                    children.truncate(1); // every sample forced: this input keeps nothing
                }
                let mut sample = make_sample(children, false);
                sample.state = GameStateBuilder::new_start_of_game_with(dims);
                samples.push(sample);
            }
            let traj = Trajectory::new(
                TrajectoryMeta::new("test", dims).with_bots("mcts", "mcts"),
                samples,
                outcome((k + t as u8) % 2),
            );
            w.write(&traj).unwrap();
        }
        w.flush().unwrap();
        inputs.push(path.to_str().unwrap().to_string());
    }

    let run = |out: &std::path::Path, extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_prepare"))
            .arg("--in")
            .args(&inputs)
            .args(["--out", out.to_str().unwrap(), "--policy-target", "cq"])
            .args(extra)
            .output()
            .expect("run prepare")
    };
    let summary = |o: &std::process::Output| {
        String::from_utf8_lossy(&o.stdout)
            .lines()
            .find(|l| l.starts_with("prepare done"))
            .unwrap()
            .to_string()
    };
    let list = |d: &std::path::Path| {
        let mut v: Vec<_> = walk(d)
            .into_iter()
            .map(|p| p.strip_prefix(d).unwrap().to_path_buf())
            .collect();
        v.sort();
        v
    };
    let serial = tmp.join("serial");
    let s = run(&serial, &[]);
    assert!(s.status.success(), "{}", String::from_utf8_lossy(&s.stderr));
    let files = list(&serial);
    assert!(files.len() >= 9, "expected at least one dims dir: {files:?}");
    for jobs in ["2", "4", "8"] {
        let par = tmp.join(format!("jobs{jobs}"));
        let p = run(&par, &["--jobs", jobs]);
        assert!(p.status.success(), "{}", String::from_utf8_lossy(&p.stderr));
        assert_eq!(summary(&s), summary(&p));
        assert_eq!(
            files,
            list(&par),
            "--jobs {jobs} wrote a different file set (parts left behind?)"
        );
        for f in &files {
            assert_eq!(
                std::fs::read(serial.join(f)).unwrap(),
                std::fs::read(par.join(f)).unwrap(),
                "--jobs {jobs}: {} differs",
                f.display()
            );
        }
    }
    let d = run(&tmp.join("dedup"), &["--jobs", "2", "--dedup"]);
    assert!(!d.status.success(), "--dedup with --jobs 2 must be refused");
    std::fs::remove_dir_all(&tmp).ok();
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}
