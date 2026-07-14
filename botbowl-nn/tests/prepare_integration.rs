//! End-to-end check of the `prepare` binary: build in-memory
//! trajectories, write them as JSONL, run `prepare`, and re-read the
//! `.npy` output — asserting shapes, CSR offsets, and per-sample policy
//! sums.

use std::process::Command;

use botbowl_data::{ChildStat, DatasetWriter, Outcome, Sample, Team, Trajectory, TrajectoryMeta};
use botbowl_engine::core::gamestate::GameStateBuilder;
use botbowl_engine::core::model::{Action, Position};
use botbowl_engine::core::table::{PosAT, SimpleAT};
use botbowl_nn::encode::{GLOBAL_FEATURES, SPATIAL_CHANNELS};
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
    assert_eq!(manifest["policy_channels"], 30);
    assert_eq!(manifest["spatial_channels"], SPATIAL_CHANNELS);
    assert_eq!(manifest["num_samples"], 2);
    assert_eq!(manifest["num_actions"], 5);

    std::fs::remove_dir_all(&tmp).ok();
}
