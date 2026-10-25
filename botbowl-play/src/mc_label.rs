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
//!
//! **Top-up** (plan 067): [`LabelConfig::skip`] starts a labelling at draw `skip` of each sample's
//! stream, so `k` more playouts on top of an `n`-playout labelling ([`topup_config`], with the
//! prior's own seed read back from its tag, [`LabelTag::parse`]) are draws `n..n+k` of the same
//! streams: disjoint from the prior's draws `0..n`, and exactly the extra playouts a fresh
//! `(n+k)`-playout labelling would play. [`apply_topup`] combines the two means ([`combine`]: the
//! integer sums, bit for bit the fresh label) and returns the per-sample prior and added means
//! ([`TopUpRecord`]): two independent estimates of one quantity, a direct measurement of the label
//! noise.

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
    /// Plan 067: draws of each sample's playout stream already used — the playouts of the labelling
    /// being topped up. This labelling plays draws `skip..skip + playouts`; 0 (a fresh labelling)
    /// is exactly the old behaviour.
    #[serde(default)]
    pub skip: u32,
}

/// The provenance stamped into `meta.extra.value_label` of every labelled trajectory. `model` is
/// the path as the user typed it.
/// For a top-up (`cfg.skip > 0`) it names the total, `skip + playouts`: the label it stamps is that
/// many playouts' mean, as a fresh labelling of that many would stamp it.
pub fn label_tag(cfg: &LabelConfig, model: &str) -> String {
    format!(
        "mc_policy(playouts={},model={},seed={})",
        cfg.skip + cfg.playouts,
        model,
        cfg.seed
    )
}

/// A parsed [`label_tag`]: what a top-up needs to continue a labelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelTag {
    pub playouts: u32,
    pub model: String,
    pub seed: u64,
}

impl LabelTag {
    /// `mc_policy(playouts=N,model=M,seed=S)`; `None` for anything else. The model is everything
    /// between `,model=` and the last `,seed=`, so a path with a comma in it still parses.
    pub fn parse(tag: &str) -> Option<LabelTag> {
        let body = tag.strip_prefix("mc_policy(playouts=")?.strip_suffix(')')?;
        let (playouts, rest) = body.split_once(",model=")?;
        let (model, seed) = rest.rsplit_once(",seed=")?;
        Some(LabelTag {
            playouts: playouts.parse().ok()?,
            model: model.to_string(),
            seed: seed.parse().ok()?,
        })
    }

    /// A trajectory's labelling, from `meta.extra.value_label`.
    pub fn of(meta: &TrajectoryMeta) -> Option<LabelTag> {
        meta.extra.get("value_label").and_then(|t| LabelTag::parse(t))
    }
}

/// The labelling a top-up continues: the tag every MC-labelled trajectory of the shard carries.
/// `Err` when the shard holds none (a raw shard, not its `mc/` copy) or two different ones (every
/// playout of a top-up is keyed on its tag's seed and count, so one shard must have one).
/// Trajectories with no tag (left unlabelled by the labelling) are left alone by the top-up too.
pub fn shard_prior<'a>(metas: impl IntoIterator<Item = &'a TrajectoryMeta>) -> Result<LabelTag, String> {
    let mut prior: Option<(&str, LabelTag)> = None;
    for m in metas {
        let Some(tag) = m.extra.get("value_label") else {
            continue;
        };
        let Some(parsed) = LabelTag::parse(tag) else {
            return Err(format!("value_label {tag:?} is not an mc_policy label"));
        };
        match &prior {
            None => prior = Some((tag.as_str(), parsed)),
            Some((first, _)) if *first == tag.as_str() => {}
            Some((first, _)) => return Err(format!("two labellings in one shard: {first} and {tag}")),
        }
    }
    prior
        .map(|(_, p)| p)
        .ok_or_else(|| "no MC-labelled trajectory in the shard (top up its mc/ copy, not the raw shard)".into())
}

/// A top-up must play under the net that made the prior labels, or the combined mean mixes two
/// policies' values: their file stems must agree (`gen43.onnx` and `gen43.ts` are one net).
pub fn check_topup_model(prior: &LabelTag, model: &str) -> Result<(), String> {
    let stem = |p: &str| std::path::Path::new(p).file_stem().map(|s| s.to_os_string());
    if stem(&prior.model) == stem(model) {
        Ok(())
    } else {
        Err(format!(
            "top-up under {model}, but the labels were made under {}: the combined mean would mix two policies",
            prior.model
        ))
    }
}

/// `added` more playouts on top of `prior`: the prior's seed, its playouts' draws skipped.
pub fn topup_config(prior: &LabelTag, added: u32, max_steps: u32) -> LabelConfig {
    LabelConfig {
        playouts: added,
        seed: prior.seed,
        max_steps,
        skip: prior.playouts,
    }
}

/// The integer sum `n` playouts scoring -1, 0 or +1 came to, from their mean as stored (an f32:
/// `round(n * mean)` is exact for any n below 2^23). `Err` when `mean` is not on the 1/n grid.
pub fn playout_sum(mean: f32, n: u32) -> Result<i64, String> {
    let x = mean as f64 * n as f64;
    let r = x.round();
    if n == 0 || (x - r).abs() > 1e-3 || r.abs() > n as f64 {
        return Err(format!("{mean} is not a mean of {n} playouts scoring -1, 0 or +1"));
    }
    Ok(r as i64)
}

/// Every label of a trajectory the top-up will extend is an `n`-playout mean ([`playout_sum`]), so
/// a bad shard is refused before any playout is spent on it.
pub fn check_prior_labels(labels: impl IntoIterator<Item = Option<f32>>, n: u32) -> Result<(), String> {
    for (k, v) in labels.into_iter().enumerate() {
        let v = v.ok_or_else(|| format!("sample {k}: labelled trajectory without an outcome_value"))?;
        playout_sum(v, n).map_err(|e| format!("sample {k}: {e}"))?;
    }
    Ok(())
}

/// The mean of `n + k` playouts from the means of the first `n` and the next `k`: the sums, added
/// and divided in f64 and rounded to f32 as [`sample_label`] rounds, so bit for bit the label a
/// fresh `(n+k)`-playout labelling writes.
pub fn combine(prior: f32, n: u32, added: f32, k: u32) -> Result<f32, String> {
    let sum = playout_sum(prior, n)? + playout_sum(added, k)?;
    Ok((sum as f64 / (n + k) as f64) as f32)
}

/// One topped-up trajectory's two independent estimates per sample (plan 067), one line of
/// `<out-dir>/topup/<shard>`: the prior `n`-playout means and the `k` added playouts' means. With
/// `n == k`, `Var(prior - added) / 2` over samples is the `n`-playout label noise
/// (`scripts/label_noise.py`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopUpRecord {
    pub seed: Option<u64>,
    pub drive: u32,
    /// `WxH/T`, playable.
    pub board: String,
    pub prior_playouts: u32,
    pub added_playouts: u32,
    pub prior: Vec<f32>,
    pub added: Vec<f32>,
}

/// Where a top-up's [`TopUpRecord`]s for the shard written to `out` go: `<out-dir>/topup/<name>`,
/// out of the way of the `mc/shard*.jsonl` globs that feed `prepare`.
pub fn topup_records_path(out: &std::path::Path) -> std::path::PathBuf {
    let dir = out.parent().unwrap_or_else(|| std::path::Path::new("."));
    dir.join("topup").join(out.file_name().unwrap_or_default())
}

/// Top up a trajectory: every sample's `outcome_value` becomes the mean of `prior.playouts` plus
/// the added playouts ([`combine`]), `value_label` becomes `tag` (the new total's), and
/// `value_label_history` gains the old tag (`|`-separated, oldest first), so a top-up can itself be
/// topped up and its parts read back. `added: None` (the trajectory does not replay, or carries no
/// label) leaves it exactly as it was and returns `None`.
pub fn apply_topup(
    traj: &mut Trajectory,
    added: Option<&[f32]>,
    prior: &LabelTag,
    tag: &str,
) -> Result<Option<TopUpRecord>, String> {
    let Some(added) = added else { return Ok(None) };
    if LabelTag::of(&traj.meta).as_ref() != Some(prior) {
        return Err(format!(
            "trajectory seed {:?} is not labelled as its shard is",
            traj.meta.seed
        ));
    }
    if added.len() != traj.samples.len() {
        return Err(format!(
            "{} added labels for {} samples",
            added.len(),
            traj.samples.len()
        ));
    }
    let total = LabelTag::parse(tag).ok_or_else(|| format!("{tag:?} is not an mc_policy tag"))?;
    let k = total
        .playouts
        .checked_sub(prior.playouts)
        .filter(|&k| k > 0)
        .ok_or_else(|| format!("{tag} adds no playouts to {}", prior.playouts))?;
    let mut prior_means = Vec::with_capacity(added.len());
    let mut combined = Vec::with_capacity(added.len());
    for (s, &a) in traj.samples.iter().zip(added) {
        let p = s.outcome_value.ok_or("a labelled sample without an outcome_value")?;
        prior_means.push(p);
        combined.push(combine(p, prior.playouts, a, k)?);
    }
    for (s, v) in traj.samples.iter_mut().zip(combined) {
        s.outcome_value = Some(v);
    }
    let old = traj
        .meta
        .extra
        .insert("value_label".into(), tag.to_string())
        .expect("checked above");
    let history = match traj.meta.extra.get("value_label_history") {
        Some(h) => format!("{h}|{old}"),
        None => old,
    };
    traj.meta.extra.insert("value_label_history".into(), history);
    Ok(Some(TopUpRecord {
        seed: traj.meta.seed,
        drive: drive_of(&traj.meta),
        board: crate::board_sizes::board_label(traj.meta.board_dims),
        prior_playouts: prior.playouts,
        added_playouts: k,
        prior: prior_means,
        added: added.to_vec(),
    }))
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
    // Path as the generator did, whatever this process's `BLOOD_JUMP_PATHS`; clones inherit it.
    state.set_jump_paths(traj.meta.jump_paths());
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
/// `cfg.playouts` policy-only drive playouts from `state`, playout `i` on the `(cfg.skip + i)`-th
/// `u64` of `rng` (a top-up's first `cfg.skip` draws belong to the labelling it continues).
pub fn sample_label(
    home: &mut PolicyBot,
    away: &mut PolicyBot,
    state: &GameState,
    rng: ChaCha8Rng,
    cfg: &LabelConfig,
) -> f32 {
    let mut sum = 0.0f64;
    for dice in playout_seeds(rng, cfg) {
        let (p, _) = play_out(state, None, TeamType::Home, home, away, None, dice, cfg.max_steps);
        sum += p.outcome as f64;
    }
    (sum / cfg.playouts as f64) as f32
}

/// The dice seeds of a sample's playouts under `cfg`: draws `cfg.skip..cfg.skip + cfg.playouts` of
/// its stream, one `u64` each, in order. Each playout seeds its own dice from its draw and draws
/// nothing else from the stream, so playout `i` of a labelling is a function of `i` alone: a
/// top-up's playouts are positions the labelling it continues never used.
pub fn playout_seeds(mut rng: ChaCha8Rng, cfg: &LabelConfig) -> impl Iterator<Item = u64> {
    for _ in 0..cfg.skip {
        rng.next_u64();
    }
    (0..cfg.playouts).map(move |_| rng.next_u64())
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
        return std::iter::once(0..0).collect();
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

    fn tiny() -> Arc<NnEvaluator> {
        Arc::new(
            NnEvaluator::from_path(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../botbowl-nn/tests/fixtures/tiny.onnx"
            ))
            .unwrap(),
        )
    }

    fn board() -> botbowl_engine::core::model::BoardDims {
        botbowl_engine::core::model::BoardDims::try_new(16, 9, 4)
            .unwrap_or_else(|_| botbowl_engine::core::model::BoardDims::from_env())
    }

    fn cfg(playouts: u32, skip: u32) -> LabelConfig {
        LabelConfig {
            playouts,
            seed: 5,
            max_steps: 300,
            skip,
        }
    }

    /// Plan 062 §5 / plan 067: a top-up is exact. An `n`-playout label is the first `n` playouts of
    /// an `m`-playout one (the rng is drawn in order, one `u64` per playout), every playout scores
    /// -1, 0 or +1, so `n * label_n` recovers the integer sum, and the `m - n` playouts a top-up
    /// plays (`skip = n`) are the `m`-playout labelling's playouts `n+1..=m`: combined, the
    /// `m`-playout label bit for bit.
    #[test]
    fn topping_up_n_playouts_to_m_gives_the_m_playout_label() {
        let nn = tiny();
        let (n, m) = (3u32, 7u32);
        let (mut home, mut away) = (PolicyBot::new(Arc::clone(&nn)), PolicyBot::new(Arc::clone(&nn)));
        let mut nonzero = 0;
        for seed in 0..4u64 {
            let state = position_state(&RandomStartBias::default(), board(), 900 + seed);
            let rng = || sample_rng(5, seed, 1, 0);
            let few = sample_label(&mut home, &mut away, &state, rng(), &cfg(n, 0));
            let many = sample_label(&mut home, &mut away, &state, rng(), &cfg(m, 0));
            // By hand: the rng advanced past the first n draws.
            let mut r = rng();
            for _ in 0..n {
                r.next_u64();
            }
            let extra: f64 = (n..m)
                .map(|_| {
                    let (p, _) = play_out(
                        &state,
                        None,
                        TeamType::Home,
                        &mut home,
                        &mut away,
                        None,
                        r.next_u64(),
                        300,
                    );
                    p.outcome as f64
                })
                .sum();
            let sum = (few as f64 * n as f64).round() + extra;
            assert_eq!(((sum / m as f64) as f32).to_bits(), many.to_bits(), "seed {seed}");
            // The same through the top-up path: `skip`, then `combine`.
            let prior = LabelTag {
                playouts: n,
                model: "tiny.onnx".into(),
                seed: 5,
            };
            let added = sample_label(&mut home, &mut away, &state, rng(), &topup_config(&prior, m - n, 300));
            assert_eq!(added, (extra / (m - n) as f64) as f32, "seed {seed}");
            let topped = combine(few, n, added, m - n).unwrap();
            assert_eq!(
                topped.to_bits(),
                many.to_bits(),
                "seed {seed}: {few} + {added} -> {topped} vs {many}"
            );
            nonzero += (many != 0.0) as u32;
        }
        // Not vacuous: some states' playouts must score.
        assert!(nonzero > 0, "every label was 0: the test checks nothing");
    }

    /// The arithmetic, exhaustively: for every playout count up to 32 and every sum two batches of
    /// playouts can come to, combining the stored means is the stored mean of the whole, bit for
    /// bit; and a value off the 1/n grid is refused rather than rounded into a wrong sum.
    #[test]
    fn combining_two_means_is_the_mean_of_all_playouts_bit_for_bit() {
        let mean = |s: i64, n: u32| (s as f64 / n as f64) as f32;
        for n in 1..=32u32 {
            for k in 1..=32u32 {
                for s1 in -(n as i64)..=n as i64 {
                    assert_eq!(playout_sum(mean(s1, n), n), Ok(s1));
                    for s2 in -(k as i64)..=k as i64 {
                        let got = combine(mean(s1, n), n, mean(s2, k), k).unwrap();
                        assert_eq!(got.to_bits(), mean(s1 + s2, n + k).to_bits(), "{s1}/{n} + {s2}/{k}");
                    }
                }
            }
        }
        // Not an 8-playout mean (a raw drive outcome blended with a root value, say), or out of range.
        assert!(playout_sum(0.3, 8).is_err());
        assert!(playout_sum(1.5, 8).is_err());
        assert!(playout_sum(0.0, 0).is_err());
        assert!(combine(0.3, 8, 0.0, 8).is_err());
    }

    /// Seed disjointness: a top-up's playouts are seeded from positions of each sample's stream the
    /// labelling it continues never drew (`n..n+k` against `0..n`), and those are exactly the
    /// positions a fresh `(n+k)` labelling gives its last `k` playouts. Across a few thousand
    /// samples of distinct trajectories, drives and indices, no playout seed of a top-up equals any
    /// playout seed of the prior labelling.
    #[test]
    fn a_topups_playout_seeds_are_disjoint_from_the_prior_and_continue_its_stream() {
        let (n, k) = (8u32, 8u32);
        let prior_cfg = LabelConfig {
            playouts: n,
            seed: 56_044,
            max_steps: 100_000,
            skip: 0,
        };
        let prior = LabelTag::parse(&label_tag(&prior_cfg, "m.onnx")).unwrap();
        let top = topup_config(&prior, k, 100_000);
        assert_eq!((top.seed, top.skip, top.playouts), (56_044, n, k));
        let fresh = LabelConfig {
            playouts: n + k,
            ..prior_cfg
        };
        let mut prior_seeds = std::collections::HashSet::new();
        let mut topup_seeds = Vec::new();
        for traj_seed in [1u64, 54_400_048, 54_400_049, u64::MAX] {
            for drive in [1u32, 2] {
                for i in 0..300usize {
                    let rng = || sample_rng(prior.seed, traj_seed, drive, i);
                    let a: Vec<u64> = playout_seeds(rng(), &prior_cfg).collect();
                    let b: Vec<u64> = playout_seeds(rng(), &top).collect();
                    let all: Vec<u64> = playout_seeds(rng(), &fresh).collect();
                    assert_eq!([a.clone(), b.clone()].concat(), all, "the top-up continues the stream");
                    prior_seeds.extend(a);
                    topup_seeds.extend(b);
                }
            }
        }
        assert_eq!(prior_seeds.len(), 4 * 2 * 300 * n as usize, "prior seeds repeat");
        assert!(
            topup_seeds.iter().all(|s| !prior_seeds.contains(s)),
            "a top-up playout reuses a prior playout's dice"
        );
    }

    /// A short heuristic trajectory cut to `labels.len()` samples, labelled `labels` under `tag`.
    fn labelled(seed: u64, labels: &[f32], tag: &str) -> Trajectory {
        let gen = crate::generate::GenerateConfig {
            mode: crate::generate::GenMode::RandomStart,
            search: crate::bots::SearchConfig::iterations(2),
            evaluator: crate::bots::Evaluator::Heuristic,
            model: None,
            max_steps: 60,
            lecture: None,
            difficulty: botbowl_curriculum::Difficulty::Easy,
            bias: RandomStartBias::default(),
            board_sizes: Some(crate::board_sizes::SizeDist::single(board())),
            config_name: None,
            exploration: None,
            next_drive: false,
        };
        let mut t = crate::generate::play_trajectory(&gen, None, seed).unwrap().remove(0);
        assert!(t.samples.len() >= labels.len(), "seed {seed}: too short a trajectory");
        t.samples.truncate(labels.len());
        for (s, &v) in t.samples.iter_mut().zip(labels) {
            s.outcome_value = Some(v);
        }
        t.meta.extra.insert("value_label".into(), tag.into());
        t
    }

    /// The bookkeeping: an `n`-playout trajectory topped up by `k` carries the `n+k` tag (the same
    /// string a fresh `n+k` labelling stamps), the old tag in its history, the combined labels, and
    /// a record of both halves; topped up again it chains. A trajectory of another labelling, a
    /// wrong label count or a top-up that adds nothing is refused; `None` leaves it untouched.
    #[test]
    fn a_topup_records_the_total_and_its_history() {
        let c8 = LabelConfig {
            playouts: 8,
            seed: 56_044,
            max_steps: 100_000,
            skip: 0,
        };
        let tag8 = label_tag(&c8, "/m/gen43.onnx");
        let prior = LabelTag::parse(&tag8).unwrap();
        assert_eq!(
            prior,
            LabelTag {
                playouts: 8,
                model: "/m/gen43.onnx".into(),
                seed: 56_044
            }
        );
        let top = topup_config(&prior, 8, 100_000);
        let tag16 = label_tag(&top, "/m/gen43.ts");
        assert_eq!(tag16, label_tag(&LabelConfig { playouts: 16, ..c8 }, "/m/gen43.ts"));

        let mut t = labelled(11, &[0.25, -1.0, 0.0], &tag8);
        let json = |t: &Trajectory| serde_json::to_string(t).unwrap();
        let before = json(&t);
        assert_eq!(apply_topup(&mut t, None, &prior, &tag16), Ok(None));
        assert_eq!(json(&t), before, "None leaves the trajectory as it was");
        assert!(apply_topup(&mut t, Some(&[0.0, 0.0]), &prior, &tag16).is_err());
        assert!(
            apply_topup(&mut t, Some(&[0.0; 3]), &prior, &tag8).is_err(),
            "adds nothing"
        );
        assert_eq!(json(&t), before);

        let rec = apply_topup(&mut t, Some(&[0.75, -0.5, 0.125]), &prior, &tag16)
            .unwrap()
            .unwrap();
        let got: Vec<f32> = t.samples.iter().map(|s| s.outcome_value.unwrap()).collect();
        assert_eq!(got, [0.5, -0.75, 0.0625]);
        assert_eq!(t.meta.extra["value_label"], tag16);
        assert_eq!(t.meta.extra["value_label_history"], tag8);
        assert_eq!(
            rec,
            TopUpRecord {
                seed: Some(11),
                drive: 1,
                board: crate::board_sizes::board_label(board()),
                prior_playouts: 8,
                added_playouts: 8,
                prior: vec![0.25, -1.0, 0.0],
                added: vec![0.75, -0.5, 0.125],
            }
        );
        // Again, 16 -> 20: the history chains, oldest first, and the draws continue at 16.
        let prior16 = LabelTag::of(&t.meta).unwrap();
        let top20 = topup_config(&prior16, 4, 100_000);
        assert_eq!((top20.skip, top20.playouts, top20.seed), (16, 4, 56_044));
        let tag20 = label_tag(&top20, "/m/gen43.ts");
        apply_topup(&mut t, Some(&[1.0, 1.0, -1.0]), &prior16, &tag20)
            .unwrap()
            .unwrap();
        assert_eq!(t.meta.extra["value_label_history"], format!("{tag8}|{tag16}"));
        assert_eq!(t.samples[0].outcome_value, Some((12.0 / 20.0f64) as f32));

        // A trajectory of another labelling is not this shard's.
        let mut other = labelled(12, &[0.0], &label_tag(&LabelConfig { seed: 1, ..c8 }, "/m/gen43.onnx"));
        assert!(apply_topup(&mut other, Some(&[0.0]), &prior, &tag16).is_err());
    }

    #[test]
    fn a_shard_has_one_prior_labelling_and_the_topup_plays_its_net() {
        let c = LabelConfig {
            playouts: 8,
            seed: 7,
            max_steps: 100_000,
            skip: 0,
        };
        let a = labelled(1, &[0.0], &label_tag(&c, "x/gen43.onnx")).meta;
        let mut unlabelled = a.clone();
        unlabelled.extra.remove("value_label");
        let b = labelled(2, &[0.0], &label_tag(&LabelConfig { seed: 8, ..c }, "x/gen43.onnx")).meta;
        let prior = shard_prior([&a, &unlabelled, &a]).unwrap();
        assert_eq!((prior.playouts, prior.seed), (8, 7));
        assert!(shard_prior([&a, &b]).is_err(), "two seeds in one shard");
        assert!(shard_prior([&unlabelled]).is_err(), "a raw shard");
        let mut odd = a.clone();
        odd.extra.insert("value_label".into(), "blend(0.5)".into());
        assert!(shard_prior([&odd]).is_err());
        assert!(check_topup_model(&prior, "/elsewhere/gen43.ts").is_ok());
        assert!(check_topup_model(&prior, "x/gen44.onnx").is_err());
        assert_eq!(
            LabelTag::parse("mc_policy(playouts=8,model=a,b.onnx,seed=3)").map(|t| t.model),
            Some("a,b.onnx".to_string())
        );
        assert_eq!(LabelTag::parse("mc_policy(playouts=x,model=a,seed=3)"), None);
        assert_eq!(
            topup_records_path(std::path::Path::new("/r/gen44/mc16/shard4.jsonl")),
            std::path::PathBuf::from("/r/gen44/mc16/topup/shard4.jsonl")
        );
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
            skip: 0,
        };
        assert_eq!(
            label_tag(&cfg, "m.onnx"),
            "mc_policy(playouts=8,model=m.onnx,seed=56016)"
        );
        // A top-up's tag names the total.
        assert_eq!(
            label_tag(&LabelConfig { skip: 8, ..cfg }, "m.onnx"),
            "mc_policy(playouts=16,model=m.onnx,seed=56016)"
        );
    }

    /// The wire and the API carry `skip`; a request written before it existed reads as a fresh
    /// labelling.
    #[test]
    fn a_label_config_without_skip_reads_as_a_fresh_labelling() {
        let c: LabelConfig = serde_json::from_str(r#"{"playouts":8,"seed":3,"max_steps":9}"#).unwrap();
        assert_eq!(c.skip, 0);
        let top = LabelConfig { skip: 8, ..c };
        let back: LabelConfig = postcard::from_bytes(&postcard::to_allocvec(&top).unwrap()).unwrap();
        assert_eq!(back, top);
    }
}
