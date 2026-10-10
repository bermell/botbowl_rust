//! Monte Carlo value labels (plan 056 arm F): replace every sample's drive outcome with the mean of
//! `--playouts` policy-only drive playouts from that sample's state.
//!
//! The corpus's `outcome_value` is one dice-driven realisation of the drive; the mean of K playouts
//! from the same state has 1/K of its noise, at the price of being the value under the *policy's*
//! continuation, not the search's (the same definition as `override-audit`'s MC(s) and the value
//! benchmark's truth). Each trajectory is replayed from its seed exactly as `override-audit` does (a
//! deserialised state has lost its path buffer, which the policy's legal set needs), and every
//! replayed state is checked against the recorded one; a trajectory that diverges is written back
//! unchanged and counted.
//!
//! The output is the same corpus, trajectory for trajectory, with `samples[i].outcome_value` (Home's
//! frame, as backfilled) set to the playout mean and `meta.extra.value_label` naming the labelling,
//! so `prepare --value-blend 1.0` trains on it with no other change. One output shard per input
//! shard, written to `<name>.partial` and renamed when complete; a rerun skips finished shards.
//!
//! This is the single-box shell: files, threads (parallel over samples) and progress lines. Every
//! piece that decides a label or the output bytes is `botbowl_play::mc_label`, which the hub's
//! `job label` (plan 062) runs on its workers, so the two write the same file.
//!
//! ```text
//! BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9 \
//! cargo run --release -p botbowl-ui -- mc-label --corpus runs/loopmix16x9g/gen07/shard0.jsonl \
//!     --model models/az_v7/bbnet_mix16x9g_gen05.onnx --nn-server /tmp/bbnn.sock \
//!     --playouts 8 --parallel 8 --out-dir runs/exp067/mc_gen07
//! ```

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use botbowl_data::{BoardCapacity, Trajectory};
use botbowl_engine::core::gamestate::GameState;
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::bots::{load_nn, Evaluator};
use botbowl_play::mc_label::{apply_labels, first_drives, label_tag, replay_all, rng_for, sample_label, summary_line};
use botbowl_play::policy::PolicyBot;
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::McLabelArgs;

/// One shard. The work is spread over samples, not trajectories: a trajectory is replayed once
/// (cheap), then every one of its samples is a separate work item, so one long drive does not leave
/// the other threads idle at the end of the shard (they did: 16 and 64 threads took the same time).
fn label_shard(nn: &Arc<NnEvaluator>, args: &McLabelArgs, input: &str, tag: &str) -> io::Result<()> {
    let name = Path::new(input)
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{input}: no file name")))?;
    let done = Path::new(&args.out_dir).join(name);
    if done.exists() {
        eprintln!("mc-label: {} exists, skipping", done.display());
        return Ok(());
    }
    let partial = done.with_extension("jsonl.partial");
    let started = Instant::now();
    let cfg = args.config();
    let invalid = |e: serde_json::Error| io::Error::new(io::ErrorKind::InvalidData, e);
    let mut trajs: Vec<Trajectory> = Vec::new();
    for line in BufReader::new(File::open(input)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let t: Trajectory = serde_json::from_str(&line).map_err(invalid)?;
        if t.meta.board_capacity != BoardCapacity::current() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{input}: corpus built at capacity {:?}, this binary is {:?}",
                    t.meta.board_capacity,
                    BoardCapacity::current()
                ),
            ));
        }
        trajs.push(t);
    }
    // Next-drive records replay through their seed's first drive.
    let firsts = first_drives(trajs.iter().map(|t| &t.meta));
    let replayed: Vec<Option<Vec<GameState>>> = trajs
        .iter()
        .zip(&firsts)
        .map(|(t, first)| {
            replay_all(t, first.map(|j| &trajs[j]))
                .map_err(|why| eprintln!("left unlabelled (seed {:?}): {why}", t.meta.seed))
                .ok()
        })
        .collect();
    let items: Vec<(usize, usize)> = replayed
        .iter()
        .enumerate()
        .filter_map(|(t, r)| r.as_ref().map(|states| (t, states.len())))
        .flat_map(|(t, n)| (0..n).map(move |k| (t, k)))
        .collect();
    // f32 labels by item, as bits.
    let labels: Vec<AtomicU32> = (0..items.len()).map(|_| AtomicU32::new(0)).collect();
    let next = AtomicUsize::new(0);
    let worker = || {
        let mut home = PolicyBot::new(Arc::clone(nn));
        let mut away = PolicyBot::new(Arc::clone(nn));
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(t, k)) = items.get(i) else { return };
            let traj = &trajs[t];
            let state = &replayed[t].as_ref().expect("items only cover replayed trajectories")[k];
            labels[i].store(
                sample_label(&mut home, &mut away, state, rng_for(&cfg, &traj.meta, k), &cfg).to_bits(),
                Ordering::Relaxed,
            );
            if (i + 1).is_multiple_of(2000) {
                eprintln!(
                    "  {input}: {} / {} samples, {:.0}s",
                    i + 1,
                    items.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
    };
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.parallel.max(1))
            .map(|i| {
                std::thread::Builder::new()
                    .name(format!("mclabel-{i}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(scope, worker)
                    .expect("spawn mc-label thread")
            })
            .collect();
        for h in handles {
            h.join().expect("mc-label thread panicked");
        }
    });
    // Items run trajectory by trajectory, each one's samples in order.
    let mut by_traj: Vec<Option<Vec<f32>>> = replayed.iter().map(|r| r.as_ref().map(|_| Vec::new())).collect();
    for (i, &(t, _)) in items.iter().enumerate() {
        by_traj[t]
            .as_mut()
            .expect("items only cover replayed trajectories")
            .push(f32::from_bits(labels[i].load(Ordering::Relaxed)));
    }
    let mut out = io::BufWriter::new(File::create(&partial)?);
    for (traj, labels) in trajs.iter_mut().zip(&by_traj) {
        apply_labels(traj, labels.as_deref(), tag);
        out.write_all(serde_json::to_string(traj)?.as_bytes())?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    drop(out);
    fs::rename(&partial, &done)?;
    eprintln!(
        "{}",
        summary_line(
            input,
            &done.display().to_string(),
            trajs.len(),
            replayed.iter().filter(|r| r.is_none()).count(),
            items.len(),
            args.playouts,
            started.elapsed().as_secs_f64()
        )
    );
    Ok(())
}

pub fn run(args: McLabelArgs) -> io::Result<()> {
    if args.playouts == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--playouts must be > 0"));
    }
    fs::create_dir_all(&args.out_dir)?;
    let server = crate::cli::nn_server_path(args.nn_server.as_deref());
    let nn = load_nn(
        Evaluator::Nn,
        Some(&args.model),
        "--model is required",
        server.as_deref(),
    )?
    .expect("Evaluator::Nn always loads a net");
    let tag = label_tag(&args.config(), &args.model);
    for input in &args.corpus {
        label_shard(&nn, &args, input, &tag)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::model::{BoardDims, TeamType};
    use botbowl_play::bots::SearchConfig;
    use botbowl_play::drives::DriveStart;
    use botbowl_play::generate::{play_trajectory, GenMode, GenerateConfig, RandomStartBias};
    use botbowl_play::mc_label::{drive_of, sample_rng};
    use botbowl_play::policy::play_out;
    use rand::RngCore;

    const TINY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../botbowl-nn/tests/fixtures/tiny.onnx");

    fn config(next_drive: bool, max_steps: u32) -> GenerateConfig {
        let board = BoardDims::try_new(16, 9, 4).unwrap_or_else(|_| BoardDims::from_env());
        GenerateConfig {
            mode: GenMode::RandomStart,
            search: SearchConfig::iterations(4),
            evaluator: Evaluator::Heuristic,
            model: None,
            max_steps,
            lecture: None,
            difficulty: botbowl_curriculum::Difficulty::Easy,
            bias: RandomStartBias::default(),
            board_sizes: Some(botbowl_play::board_sizes::SizeDist::single(board)),
            config_name: None,
            exploration: None,
            next_drive,
        }
    }

    fn trajectory(seed: u64) -> Trajectory {
        play_trajectory(&config(false, 40), None, seed).unwrap().remove(0)
    }

    /// Each label is the mean of that many policy-only playouts from the replayed state, in Home's
    /// frame, reproducible from the seed; everything but `outcome_value` and the provenance tag is
    /// the input, so `prepare` sees the same corpus with a different value label.
    #[test]
    fn labels_are_the_mean_of_seeded_policy_playouts_in_homes_frame() {
        let dir = std::env::temp_dir().join(format!("mc-label-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("shard0.jsonl");
        let trajs = [trajectory(5151), trajectory(5152)];
        let mut f = File::create(&input).unwrap();
        for t in &trajs {
            writeln!(f, "{}", serde_json::to_string(t).unwrap()).unwrap();
        }
        drop(f);
        let args = McLabelArgs {
            corpus: vec![input.to_str().unwrap().into()],
            model: TINY.into(),
            nn_server: None,
            playouts: 3,
            seed: 7,
            parallel: 2,
            max_steps: 100_000,
            out_dir: dir.join("out").to_str().unwrap().into(),
        };
        run(args.clone()).unwrap();
        let text = fs::read_to_string(dir.join("out/shard0.jsonl")).unwrap();
        let got: Vec<Trajectory> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(got.len(), trajs.len());
        assert!(!dir.join("out/shard0.jsonl.partial").exists());

        let nn = Arc::new(NnEvaluator::from_path(TINY).unwrap());
        let mut checked = 0;
        for g in &got {
            let orig = trajs
                .iter()
                .find(|t| t.meta.seed == g.meta.seed)
                .expect("same trajectories");
            assert_eq!(g.samples.len(), orig.samples.len());
            assert!(g.meta.extra["value_label"].starts_with("mc_policy(playouts=3,"));
            let states = replay_all(orig, None).unwrap();
            for (k, (s, state)) in g.samples.iter().zip(&states).enumerate().step_by(4) {
                assert_eq!(s.chosen_action, orig.samples[k].chosen_action);
                let mut rng = sample_rng(7, orig.meta.seed.unwrap(), 1, k);
                let (mut h, mut a) = (PolicyBot::new(Arc::clone(&nn)), PolicyBot::new(Arc::clone(&nn)));
                let want: f32 = (0..3)
                    .map(|_| {
                        let (p, end) = play_out(
                            state,
                            None,
                            TeamType::Home,
                            &mut h,
                            &mut a,
                            None,
                            rng.next_u64(),
                            100_000,
                        );
                        // Home's frame: the drive's score delta, as the backfill writes it.
                        let (hs, aw) = DriveStart::of(state).scored(&end);
                        assert_eq!(p.outcome, (hs as f32 - aw as f32).clamp(-1.0, 1.0));
                        p.outcome as f64
                    })
                    .sum::<f64>() as f32;
                assert_eq!(s.outcome_value, Some(((want as f64) / 3.0) as f32), "sample {k}");
                checked += 1;
            }
        }
        assert!(checked >= 4, "too few samples checked: {checked}");

        // A rerun skips the finished shard.
        run(args).unwrap();
        assert_eq!(fs::read_to_string(dir.join("out/shard0.jsonl")).unwrap(), text);
        fs::remove_dir_all(&dir).ok();
    }

    /// A `--next-drive` record (plan 047: the kickoff setups and the drive after a score) is
    /// replayed through its seed's first drive and labelled like any other, with its own dice.
    #[test]
    fn next_drive_records_replay_through_their_first_drive_and_get_labelled() {
        let cfg = config(true, 2000);
        let pair = (6000..6200u64)
            .map(|seed| play_trajectory(&cfg, None, seed).unwrap())
            .find(|v| v.len() == 2)
            .expect("no seed in 200 scored and produced a next-drive record");
        assert_eq!((drive_of(&pair[0].meta), drive_of(&pair[1].meta)), (1, 2));
        // Replay alone cannot rebuild drive 2; with its first drive it reproduces every state.
        assert!(replay_all(&pair[1], None).is_err());
        let states = replay_all(&pair[1], Some(&pair[0])).unwrap();
        assert_eq!(states.len(), pair[1].samples.len());

        let dir = std::env::temp_dir().join(format!("mc-label-next-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("shard0.jsonl");
        let mut f = File::create(&input).unwrap();
        // Drive 2 first: the first-drive index must not depend on line order.
        for t in [&pair[1], &pair[0]] {
            writeln!(f, "{}", serde_json::to_string(t).unwrap()).unwrap();
        }
        drop(f);
        run(McLabelArgs {
            corpus: vec![input.to_str().unwrap().into()],
            model: TINY.into(),
            nn_server: None,
            playouts: 2,
            seed: 9,
            parallel: 2,
            max_steps: 100_000,
            out_dir: dir.join("out").to_str().unwrap().into(),
        })
        .unwrap();
        let got: Vec<Trajectory> = fs::read_to_string(dir.join("out/shard0.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(got.len(), 2);
        assert!(
            got.iter().all(|t| t.meta.extra.contains_key("value_label")),
            "both drives labelled"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
