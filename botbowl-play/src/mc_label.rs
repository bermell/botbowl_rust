//! Monte Carlo value labels (plan 056 arm F), the process-agnostic core: replay a corpus
//! trajectory, and label each of its samples with the mean of `playouts` policy-only drive playouts
//! from that sample's state.
//!
//! Two shells run it: `botbowl-ui mc-label` (one box, files and threads, parallel over samples) and
//! the hub's `job label` (plan 062: the hub splits shards into sample ranges, workers label them,
//! the hub writes the shard). Everything that decides a label or the output bytes is here, so the
//! two produce the same file:
//!
//! - [`replay_all`] rebuilds every recorded state from the trajectory's seed (a deserialised state
//!   has lost its path buffer, which the policy's legal set needs), through the seed's first drive
//!   for a `--next-drive` record ([`first_drives`] finds it in the shard), and fails at the first
//!   divergence: such a trajectory is written back unchanged ("left unlabelled") and counted;
//! - [`sample_rng`] keys each sample's playout dice on `(seed, trajectory seed, drive, sample)`, so
//!   a label does not depend on which thread or machine computed it, nor on what else was labelled;
//! - [`sample_label`] draws one `u64` per playout from that rng, in order. Playouts `1..=n` of an
//!   `m > n` labelling are therefore exactly the `n` playouts of an `n` labelling, and since every
//!   playout scores -1, 0 or +1 the stored mean recovers the integer sum: a top-up from `n` to `m`
//!   is exact (plan 062 §5);
//! - [`apply_labels`] writes the labels and the provenance tag ([`label_tag`]) into the trajectory.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use botbowl_data::{Trajectory, TrajectoryMeta};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::TeamType;
use botbowl_nn::eval::NnEvaluator;

use crate::drives::position_state;
use crate::generate::RandomStartBias;
use crate::policy::{play_out, PolicyBot};

/// What a label is a function of, besides the net and the state: the playouts averaged, the base
/// seed of their dice, and the engine-step cap per playout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelConfig {
    pub playouts: u32,
    pub seed: u64,
    pub max_steps: u32,
}

/// The provenance stamped into `meta.extra.value_label` of every labelled trajectory. `model` is
/// the path as the user typed it.
pub fn label_tag(cfg: &LabelConfig, model: &str) -> String {
    format!("mc_policy(playouts={},model={},seed={})", cfg.playouts, model, cfg.seed)
}

/// The random-start placement a trajectory was drawn with, from its provenance. `extra.temperature`
/// is the one this seed used (already alternated), so it stands in for both temperatures.
pub fn bias_of(meta: &TrajectoryMeta) -> Result<RandomStartBias, String> {
    if meta.extra.get("mode").map(String::as_str) != Some("random-start") {
        return Err("not a random-start trajectory".into());
    }
    let f = |k: &str| -> Result<f32, String> {
        meta.extra
            .get(k)
            .ok_or_else(|| format!("meta.extra has no {k}"))?
            .parse::<f32>()
            .map_err(|e| format!("meta.extra.{k}: {e}"))
    };
    let temperature = f("temperature")?;
    Ok(RandomStartBias {
        ball_distance: f("ball_distance")?,
        front_line: f("front_line")?,
        mark_teammate: f("mark_teammate")?,
        mark_opponent: f("mark_opponent")?,
        own_side: f("own_side")?,
        temperature,
        temperature2: temperature,
        carried_prob: f("carried_prob")?,
        line_fraction: f("line_fraction")?,
        pocket_fraction: f("pocket_fraction")?,
    })
}

/// Which drive of its seed a record is: 1 for the random-start drive, 2 for the `--next-drive`
/// record that follows a score through the kickoff (plan 047, `meta.extra.drive`).
pub fn drive_of(meta: &TrajectoryMeta) -> u32 {
    meta.extra.get("drive").and_then(|d| d.parse().ok()).unwrap_or(1)
}

/// For each record of a shard, the index of its seed's first drive when it is a next-drive record
/// (`drive_of` > 1) and the shard holds one; `None` otherwise. Order-free: a first drive may come
/// after its follow-on. Should a seed's first drive appear twice, the later line wins.
pub fn first_drives<'a>(metas: impl IntoIterator<Item = &'a TrajectoryMeta>) -> Vec<Option<usize>> {
    let metas: Vec<&TrajectoryMeta> = metas.into_iter().collect();
    let first: HashMap<u64, usize> = metas
        .iter()
        .enumerate()
        .filter(|(_, m)| drive_of(m) == 1)
        .filter_map(|(i, m)| m.seed.map(|s| (s, i)))
        .collect();
    metas
        .iter()
        .map(|m| m.seed.filter(|_| drive_of(m) > 1).and_then(|s| first.get(&s).copied()))
        .collect()
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
pub fn sample_rng(base: u64, traj_seed: u64, drive: u32, k: usize) -> ChaCha8Rng {
    let k = k as u64 + (drive.saturating_sub(1) as u64) * 1_000_000;
    let id = traj_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    ChaCha8Rng::seed_from_u64(base ^ id)
}

/// Sample `k` of `traj`'s playout dice ([`sample_rng`] keyed on the trajectory's own seed/drive).
pub fn rng_for(cfg: &LabelConfig, traj: &TrajectoryMeta, k: usize) -> ChaCha8Rng {
    sample_rng(cfg.seed, traj.seed.unwrap_or(0), drive_of(traj), k)
}

/// One sample's label: the mean, in Home's frame (the frame `outcome_value` is backfilled in), of
/// `cfg.playouts` policy-only drive playouts from `state`, playout `i` on the `i`-th `u64` of `rng`.
pub fn sample_label(
    home: &mut PolicyBot,
    away: &mut PolicyBot,
    state: &GameState,
    mut rng: ChaCha8Rng,
    cfg: &LabelConfig,
) -> f32 {
    let mut sum = 0.0f64;
    for _ in 0..cfg.playouts {
        let (p, _) = play_out(
            state,
            None,
            TeamType::Home,
            home,
            away,
            None,
            rng.next_u64(),
            cfg.max_steps,
        );
        sum += p.outcome as f64;
    }
    (sum / cfg.playouts as f64) as f32
}

/// Labels for samples `range` of `traj` (clamped to its length), or why it cannot be labelled.
/// The whole trajectory is replayed whatever the range, so every range of one trajectory agrees on
/// whether it is labellable — exactly the local tool's per-trajectory verdict.
pub fn label_range(
    nn: &Arc<NnEvaluator>,
    traj: &Trajectory,
    first: Option<&Trajectory>,
    range: Range<usize>,
    cfg: &LabelConfig,
) -> Result<Vec<f32>, String> {
    let states = replay_all(traj, first)?;
    let range = range.start.min(states.len())..range.end.min(states.len());
    let mut home = PolicyBot::new(Arc::clone(nn));
    let mut away = PolicyBot::new(Arc::clone(nn));
    Ok(range
        .map(|k| sample_label(&mut home, &mut away, &states[k], rng_for(cfg, &traj.meta, k), cfg))
        .collect())
}

/// Write a trajectory's labels: every sample's `outcome_value` (Home's frame) and the provenance
/// tag, or nothing at all for a trajectory that could not be labelled (`None`).
pub fn apply_labels(traj: &mut Trajectory, labels: Option<&[f32]>, tag: &str) {
    let Some(labels) = labels else { return };
    assert_eq!(labels.len(), traj.samples.len(), "one label per sample");
    for (s, &v) in traj.samples.iter_mut().zip(labels) {
        s.outcome_value = Some(v);
    }
    traj.meta.extra.insert("value_label".into(), tag.to_string());
}

/// The line each shell prints when a shard is written; `train_loop.sh` sums the unlabelled counts
/// from it (`grep -o '([0-9]* left unlabelled)'`), so local and hub runs print it identically.
pub fn summary_line(
    input: &str,
    out: &str,
    trajectories: usize,
    unlabelled: usize,
    samples: usize,
    playouts: u32,
    secs: f64,
) -> String {
    format!(
        "mc-label: {input} -> {out}: {trajectories} trajectories ({unlabelled} left unlabelled), {samples} samples x {playouts} playouts in {secs:.0}s"
    )
}

/// A trajectory of `samples` samples as work items of at most `chunk` samples each, in order and
/// covering every sample once. A trajectory with no samples is still one (empty) item: whether it
/// replays decides whether it is stamped as labelled.
pub fn split_items(samples: usize, chunk: usize) -> Vec<Range<usize>> {
    let chunk = chunk.max(1);
    if samples == 0 {
        return vec![0..0];
    }
    // Even pieces rather than `chunk, chunk, .., remainder`: no straggler of one sample.
    let pieces = samples.div_ceil(chunk);
    let (base, extra) = (samples / pieces, samples % pieces);
    let mut out = Vec::with_capacity(pieces);
    let mut start = 0;
    for i in 0..pieces {
        let len = base + usize::from(i < extra);
        out.push(start..start + len);
        start += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_cover_every_sample_once_in_order() {
        for (n, chunk) in [
            (0, 4),
            (1, 4),
            (4, 4),
            (5, 4),
            (9, 4),
            (100, 32),
            (65, 32),
            (7, 1),
            (3, 0),
        ] {
            let items = split_items(n, chunk);
            assert!(!items.is_empty(), "n={n}");
            let flat: Vec<usize> = items.iter().flat_map(|r| r.clone()).collect();
            assert_eq!(flat, (0..n).collect::<Vec<_>>(), "n={n} chunk={chunk}");
            assert!(
                items.iter().all(|r| r.len() <= chunk.max(1)),
                "n={n} chunk={chunk}: {items:?}"
            );
            let (lo, hi) = (
                items.iter().map(|r| r.len()).min().unwrap(),
                items.iter().map(|r| r.len()).max().unwrap(),
            );
            assert!(hi - lo <= 1, "n={n} chunk={chunk}: uneven {items:?}");
        }
    }

    #[test]
    fn a_next_drive_record_finds_its_first_drive_in_any_order() {
        let meta = |seed: Option<u64>, drive: Option<&str>| {
            let mut m = TrajectoryMeta::new("random-start", botbowl_engine::core::model::BoardDims::from_env());
            m.seed = seed;
            if let Some(d) = drive {
                m.extra.insert("drive".into(), d.into());
            }
            m
        };
        let metas = [
            meta(Some(7), Some("2")), // its first drive comes later
            meta(Some(5), None),
            meta(Some(7), Some("1")),
            meta(Some(9), Some("2")), // no first drive in the shard
            meta(None, Some("2")),
            meta(Some(5), Some("2")),
        ];
        assert_eq!(first_drives(&metas), vec![Some(2), None, None, None, None, Some(1)]);
    }

    #[test]
    fn a_label_tag_names_playouts_model_and_seed() {
        let cfg = LabelConfig {
            playouts: 8,
            seed: 56_016,
            max_steps: 100_000,
        };
        assert_eq!(
            label_tag(&cfg, "m.onnx"),
            "mc_policy(playouts=8,model=m.onnx,seed=56016)"
        );
    }
}
