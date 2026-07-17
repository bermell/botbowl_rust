//! `prepare` — turn JSONL trajectory files into fixed-shape `.npy` batches
//! for the PyTorch trainer.
//!
//! Streams every input trajectory, encodes each decision sample with the
//! shared Rust encoder ([`botbowl_nn::encode`]), builds the solved-aware
//! policy/value targets ([`botbowl_nn::targets`]), and groups samples by
//! board dimensions — one subdirectory per `(w, h)`, so every batch in a
//! subdir has a single tensor shape (no ragged spatial padding needed).
//!
//! Per dims-subdir it writes:
//! - `spatial.npy`  `(N, C, H, W)` f32
//! - `global.npy`   `(N, F)` f32
//! - `value.npy`    `(N,)` f32           — mover-signed outcome target
//! - `chosen.npy`   `(N,)` i64           — local index of the played action
//! - `actions.npy`  `(M, 4)` i64         — `[channel, y, x, is_simple]` per legal action
//! - `policy.npy`   `(M,)` f32           — policy-target prob per legal action
//! - `action_offsets.npy` `(N+1,)` i64   — CSR offsets into the ragged action list
//! - `manifest.json`                     — versions, C/F/A, names, provenance
//!
//! The `actions`/`policy`/`offsets` triple is a CSR encoding of the
//! variable-length legal-action set per sample: sample `i`'s actions are
//! rows `offsets[i]..offsets[i+1]`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use botbowl_data::DatasetReader;
use botbowl_nn::actions::{action_cell, POLICY_CHANNELS};
use botbowl_nn::encode::{encode, global_feature_names, spatial_channel_names, GLOBAL_FEATURES, SPATIAL_CHANNELS};
use botbowl_nn::npy;
use botbowl_nn::perspective::mover_for;
use botbowl_nn::targets::{policy_target, value_target, SolvedRootPolicy};

/// Layout schema version — bump when the tensor layout / channel meaning
/// changes so a stale prepared dir can be rejected.
// v2: value target became the drive-relative outcome (per-sample backfill
// in `Trajectory::backfill_outcome_value`) instead of the broadcast
// final-scoreline z — batches prepared at v1 are not comparable.
const NN_SCHEMA_VERSION: u32 = 2;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SolvedRootArg {
    /// One-hot on the argmax mover-Q child.
    Onehot,
    /// Drop solved-root samples entirely.
    Skip,
}

impl From<SolvedRootArg> for SolvedRootPolicy {
    fn from(a: SolvedRootArg) -> Self {
        match a {
            SolvedRootArg::Onehot => SolvedRootPolicy::OneHot,
            SolvedRootArg::Skip => SolvedRootPolicy::Skip,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Encode JSONL trajectories into .npy training batches (plan 017).")]
struct Args {
    /// Input JSONL trajectory files.
    #[arg(long = "in", required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// Output directory (one subdir per board-dims group is created under it).
    #[arg(long)]
    out: PathBuf,
    /// How to treat fully-solved roots when building the policy target.
    #[arg(long = "solved-root", value_enum, default_value = "onehot")]
    solved_root: SolvedRootArg,
    /// Drop samples whose root received fewer than this many descents.
    #[arg(long = "min-root-visits", default_value_t = 0)]
    min_root_visits: u32,
}

/// Accumulates all samples that share one board shape.
#[derive(Default)]
struct DimsGroup {
    n: usize,
    spatial: Vec<f32>,
    global: Vec<f32>,
    value: Vec<f32>,
    chosen: Vec<i64>,
    // CSR ragged legal actions.
    action_rows: Vec<i64>, // flattened (M, 4)
    policy: Vec<f32>,      // (M,)
    offsets: Vec<i64>,     // (N+1,), starts with 0
}

fn main() {
    let args = Args::parse();
    let solved_root: SolvedRootPolicy = args.solved_root.into();

    // Keyed by (w, h) engine dims.
    let mut groups: BTreeMap<(usize, usize), DimsGroup> = BTreeMap::new();
    let mut total_read = 0usize;
    let mut total_kept = 0usize;
    let mut total_skipped_policy = 0usize;
    let mut total_skipped_value = 0usize;
    let mut total_below_min = 0usize;

    for input in &args.inputs {
        let reader =
            DatasetReader::open(input).unwrap_or_else(|e| panic!("cannot open input {}: {e}", input.display()));
        for traj in reader {
            let traj = traj.unwrap_or_else(|e| panic!("bad trajectory in {}: {e}", input.display()));
            for sample in &traj.samples {
                total_read += 1;
                if sample.root_visits < args.min_root_visits {
                    total_below_min += 1;
                    continue;
                }
                let policy = match policy_target(sample, solved_root) {
                    Some(p) => p,
                    None => {
                        total_skipped_policy += 1;
                        continue;
                    }
                };
                let value = match value_target(sample) {
                    Some(v) => v,
                    None => {
                        // Value target missing (outcome not backfilled) — the
                        // value head needs it, so drop the sample.
                        total_skipped_value += 1;
                        continue;
                    }
                };

                let enc = encode(&sample.state);
                let mover = mover_for(&sample.state);
                let dims = sample.state.board_dims;
                let group = groups.entry((enc.w, enc.h)).or_default();

                group.spatial.extend_from_slice(&enc.spatial);
                group.global.extend_from_slice(&enc.global);
                group.value.push(value);

                // Legal actions (CSR): one row per child, aligned to the
                // policy-target probabilities.
                let mut chosen_local: i64 = -1;
                for (j, cstat) in sample.children.iter().enumerate() {
                    let cell = action_cell(cstat.action, mover, dims);
                    group.action_rows.push(cell.channel as i64);
                    group.action_rows.push(cell.y as i64);
                    group.action_rows.push(cell.x as i64);
                    group.action_rows.push(cell.is_simple as i64);
                    group.policy.push(policy.probs[j]);
                    if cstat.action == sample.chosen_action {
                        chosen_local = j as i64;
                    }
                }
                // Fallback: if the played action isn't among the children
                // (shouldn't happen), point at the policy argmax.
                if chosen_local < 0 {
                    chosen_local = policy
                        .probs
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as i64)
                        .unwrap_or(0);
                }
                group.chosen.push(chosen_local);

                group.n += 1;
                group.offsets.push((group.policy.len()) as i64);
                total_kept += 1;
            }
        }
    }

    if groups.is_empty() {
        eprintln!("prepare: no samples survived filtering — nothing written");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&args.out).expect("create output dir");
    let channel_names = spatial_channel_names();
    let feature_names = global_feature_names();

    for ((w, h), group) in &groups {
        let subdir = args.out.join(format!("dims_{w}x{h}"));
        std::fs::create_dir_all(&subdir).expect("create dims subdir");
        let n = group.n;
        let m = group.policy.len();

        npy::write_f32(
            subdir.join("spatial.npy"),
            &group.spatial,
            &[n, SPATIAL_CHANNELS, *h, *w],
        )
        .unwrap();
        npy::write_f32(subdir.join("global.npy"), &group.global, &[n, GLOBAL_FEATURES]).unwrap();
        npy::write_f32(subdir.join("value.npy"), &group.value, &[n]).unwrap();
        npy::write_i64(subdir.join("chosen.npy"), &group.chosen, &[n]).unwrap();
        npy::write_i64(subdir.join("actions.npy"), &group.action_rows, &[m, 4]).unwrap();
        npy::write_f32(subdir.join("policy.npy"), &group.policy, &[m]).unwrap();
        // offsets is 0 followed by the running action-count after each sample.
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0i64);
        offsets.extend_from_slice(&group.offsets);
        npy::write_i64(subdir.join("action_offsets.npy"), &offsets, &[n + 1]).unwrap();

        let manifest = serde_json::json!({
            "nn_schema_version": NN_SCHEMA_VERSION,
            "data_format_version": botbowl_data::FORMAT_VERSION,
            "git_commit": botbowl_data::git_commit(),
            "git_dirty": botbowl_data::git_dirty(),
            "board_dims": { "width": w, "height": h },
            "spatial_channels": SPATIAL_CHANNELS,
            "global_features": GLOBAL_FEATURES,
            "policy_channels": POLICY_CHANNELS,
            "spatial_channel_names": channel_names,
            "global_feature_names": feature_names,
            "value_target": "mover_signed_drive_outcome",
            "solved_root_policy": format!("{solved_root:?}"),
            "min_root_visits": args.min_root_visits,
            "num_samples": n,
            "num_actions": m,
        });
        std::fs::write(
            subdir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        println!(
            "wrote {subdir}: N={n} M={m} shape=({SPATIAL_CHANNELS},{h},{w})",
            subdir = subdir.display()
        );
    }

    println!(
        "prepare done: read {total_read} samples, kept {total_kept} \
         (dropped {total_below_min} below min-root-visits, {total_skipped_policy} without a policy target, \
         {total_skipped_value} without a value target) \
         across {} board-dims group(s)",
        groups.len()
    );
}
