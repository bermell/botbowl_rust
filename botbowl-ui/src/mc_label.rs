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
//! ```text
//! BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR=target/16x9 \
//! cargo run --release -p botbowl-ui -- mc-label --corpus runs/loopmix16x9g/gen07/shard0.jsonl \
//!     --model models/az_v7/bbnet_mix16x9g_gen05.onnx --nn-server /tmp/bbnn.sock \
//!     --playouts 8 --parallel 8 --out-dir runs/exp067/mc_gen07
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use botbowl_data::{BoardCapacity, Trajectory, TrajectoryMeta};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::TeamType;
use botbowl_nn::eval::NnEvaluator;
use botbowl_play::bots::{load_nn, Evaluator};
use botbowl_play::drives::position_state;
use botbowl_play::GAME_STACK_SIZE;

use crate::cli::McLabelArgs;
use crate::override_audit::{bias_of, play_out, PolicyBot};

/// Which drive of its seed a record is: 1 for the random-start drive, 2 for the `--next-drive`
/// record that follows a score through the kickoff (plan 047, `meta.extra.drive`).
pub fn drive_of(meta: &TrajectoryMeta) -> u32 {
    meta.extra.get("drive").and_then(|d| d.parse().ok()).unwrap_or(1)
}

/// Every recorded state of `traj`, rebuilt by replay from its seed; `Err` at the first divergence.
/// A next-drive record (`drive_of` 2) starts where its seed's first drive ended, so `first` (that
/// record) is replayed through first, under the same engine and dice stream.
pub fn replay_all(traj: &Trajectory, first: Option<&Trajectory>) -> Result<Vec<GameState>, String> {
    let seed = traj.meta.seed.ok_or("trajectory has no seed")?;
    let mut state = position_state(&bias_of(&traj.meta)?, traj.meta.board_dims, seed);
    if drive_of(&traj.meta) > 1 {
        let first = first.ok_or("a next-drive record without its first drive in the shard")?;
        if first.meta.seed != traj.meta.seed || drive_of(&first.meta) != 1 {
            return Err("the first drive given is not this record's".into());
        }
        for (k, sample) in first.samples.iter().enumerate() {
            if state != sample.state {
                return Err(format!("replay of the first drive diverged at sample {k}"));
            }
            state
                .step(sample.chosen_action)
                .map_err(|e| format!("replaying the first drive's sample {k}: {e:?}"))?;
        }
    }
    let mut out = Vec::with_capacity(traj.samples.len());
    for (k, sample) in traj.samples.iter().enumerate() {
        if state != sample.state {
            return Err(format!("replay diverged from the recorded state at sample {k}"));
        }
        out.push(state.clone());
        state
            .step(sample.chosen_action)
            .map_err(|e| format!("replaying sample {k}: {e:?}"))?;
    }
    Ok(out)
}

/// The playouts' dice for sample `k` of drive `drive` of the trajectory with seed `traj_seed`.
fn sample_rng(base: u64, traj_seed: u64, drive: u32, k: usize) -> ChaCha8Rng {
    let k = k as u64 + (drive.saturating_sub(1) as u64) * 1_000_000;
    let id = traj_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    ChaCha8Rng::seed_from_u64(base ^ id)
}

/// Relabel one trajectory in place. `false` if it could not be replayed (left as it was).
fn label(
    nn: &Arc<NnEvaluator>,
    traj: &mut Trajectory,
    first: Option<&Trajectory>,
    args: &McLabelArgs,
    tag: &str,
) -> bool {
    let states = match replay_all(traj, first) {
        Ok(s) => s,
        Err(why) => {
            eprintln!("left unlabelled (seed {:?}): {why}", traj.meta.seed);
            return false;
        }
    };
    let traj_seed = traj.meta.seed.unwrap_or(0);
    let drive = drive_of(&traj.meta);
    let mut home = PolicyBot::new(Arc::clone(nn));
    let mut away = PolicyBot::new(Arc::clone(nn));
    for (k, (sample, state)) in traj.samples.iter_mut().zip(&states).enumerate() {
        let mut rng = sample_rng(args.seed, traj_seed, drive, k);
        let mut sum = 0.0f64;
        for _ in 0..args.playouts {
            // Home's frame, the frame `outcome_value` is backfilled in.
            let (p, _) = play_out(
                state,
                None,
                TeamType::Home,
                &mut home,
                &mut away,
                None,
                rng.next_u64(),
                args.max_steps,
            );
            sum += p.outcome as f64;
        }
        sample.outcome_value = Some((sum / args.playouts as f64) as f32);
    }
    traj.meta.extra.insert("value_label".into(), tag.to_string());
    true
}

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
    let lines: Vec<String> = BufReader::new(File::open(input)?)
        .lines()
        .collect::<io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|l| !l.trim().is_empty())
        .collect();
    // Next-drive records replay through their seed's first drive: index the first drives by seed.
    #[derive(serde::Deserialize)]
    struct MetaOnly {
        meta: TrajectoryMeta,
    }
    let mut first_drive: HashMap<u64, usize> = HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        let m: MetaOnly = serde_json::from_str(line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if let (Some(seed), 1) = (m.meta.seed, drive_of(&m.meta)) {
            first_drive.insert(seed, i);
        }
    }
    let out = Mutex::new(io::BufWriter::new(File::create(&partial)?));
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let samples = AtomicUsize::new(0);
    let worker = || -> io::Result<()> {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(line) = lines.get(i) else { return Ok(()) };
            let mut traj: Trajectory =
                serde_json::from_str(line).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            if traj.meta.board_capacity != BoardCapacity::current() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{input}: corpus built at capacity {:?}, this binary is {:?}",
                        traj.meta.board_capacity,
                        BoardCapacity::current()
                    ),
                ));
            }
            let first: Option<Trajectory> = match (drive_of(&traj.meta), traj.meta.seed) {
                (d, Some(seed)) if d > 1 => first_drive
                    .get(&seed)
                    .map(|&j| serde_json::from_str(&lines[j]))
                    .transpose()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
                _ => None,
            };
            if label(nn, &mut traj, first.as_ref(), args, tag) {
                samples.fetch_add(traj.samples.len(), Ordering::Relaxed);
            } else {
                failed.fetch_add(1, Ordering::Relaxed);
            }
            let text = serde_json::to_string(&traj)?;
            let mut f = out.lock().expect("output mutex");
            f.write_all(text.as_bytes())?;
            f.write_all(b"\n")?;
            if (i + 1).is_multiple_of(25) {
                eprintln!(
                    "  {input}: {} / {} trajectories, {:.0}s",
                    i + 1,
                    lines.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
    };
    std::thread::scope(|scope| -> io::Result<()> {
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
            h.join().expect("mc-label thread panicked")?;
        }
        Ok(())
    })?;
    out.into_inner().expect("output mutex").flush()?;
    fs::rename(&partial, &done)?;
    eprintln!(
        "mc-label: {} -> {}: {} trajectories ({} left unlabelled), {} samples x {} playouts in {:.0}s",
        input,
        done.display(),
        lines.len(),
        failed.load(Ordering::Relaxed),
        samples.load(Ordering::Relaxed),
        args.playouts,
        started.elapsed().as_secs_f64()
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
    let tag = format!(
        "mc_policy(playouts={},model={},seed={})",
        args.playouts, args.model, args.seed
    );
    for input in &args.corpus {
        label_shard(&nn, &args, input, &tag)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::model::BoardDims;
    use botbowl_play::bots::SearchConfig;
    use botbowl_play::drives::DriveStart;
    use botbowl_play::generate::{play_trajectory, GenMode, GenerateConfig, RandomStartBias};

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
