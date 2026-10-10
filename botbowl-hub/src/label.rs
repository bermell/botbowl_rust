//! Plan 062: the hub's half of a label job — read the shards in, write them back out.
//!
//! The labelling itself is `botbowl_play::mc_label` on the workers. What lives here is everything
//! `botbowl-ui mc-label` does around it, done so the file the hub writes is the file the local tool
//! writes, byte for byte:
//!
//! - **reading** ([`load_inputs`], off the hub's lock): a shard's lines exactly as the local tool
//!   reads them (`BufRead::lines`, blank lines skipped), the capacity check, each line kept
//!   zstd-compressed (~7% of the JSON on a 16x9 corpus) for shipping and for the final write, the seed's
//!   first drive of every `--next-drive` record ([`first_drives`], the local rule), and each
//!   trajectory cut into work items of at most `chunk_samples` samples ([`split_items`]). A
//!   trajectory with no samples is still one empty item, because whether it replays decides
//!   whether it is stamped;
//! - **writing** ([`ShardWrite::run`], on its own thread once a shard's last item is in): every
//!   line parsed, labelled with [`apply_labels`] (or left alone when any item found it does not
//!   replay), re-serialised and written to `out.partial`, renamed to `out` when complete.
//!
//! A shard whose output exists at submit is skipped, as locally, so a rerun of a phase that died
//! half-way only labels what is missing.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use botbowl_data::{BoardCapacity, Trajectory, TrajectoryMeta};
use botbowl_play::mc_label::{apply_labels, first_drives, split_items};

use crate::api::{JobId, LabelJobRequest};

/// zstd level for the lines the hub keeps and ships, as for trajectories from workers.
const ZSTD_LEVEL: i32 = 3;

/// One sample range of one trajectory: `(trajectory index, start, end)`.
pub type Item = (usize, u32, u32);

/// A label job's shard as read at submit.
pub struct LabelInput {
    pub name: String,
    pub input: PathBuf,
    pub out: PathBuf,
    /// `out` existed: nothing to do, as `botbowl-ui mc-label` skips it.
    pub skipped: bool,
    /// Every trajectory line, zstd.
    pub lines: Arc<Vec<Vec<u8>>>,
    pub seeds: Vec<Option<u64>>,
    pub samples: Vec<usize>,
    /// The line of each next-drive record's first drive.
    pub firsts: Vec<Option<usize>>,
    /// The work items, trajectory by trajectory, each one's ranges in order.
    pub items: Vec<Item>,
}

/// What reading needs from a line, without building a `GameState` per sample.
#[derive(Deserialize)]
struct Lite {
    meta: TrajectoryMeta,
    samples: Vec<serde::de::IgnoredAny>,
}

fn invalid(what: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

/// Read one shard. Errors are the local tool's: an unreadable file, a line that is not a
/// trajectory, a corpus built at another capacity.
pub fn load_input(name: &str, input: &Path, out: &Path, chunk_samples: u32) -> io::Result<LabelInput> {
    let mut shard = LabelInput {
        name: name.to_string(),
        input: input.to_path_buf(),
        out: out.to_path_buf(),
        skipped: out.exists(),
        lines: Arc::new(Vec::new()),
        seeds: Vec::new(),
        samples: Vec::new(),
        firsts: Vec::new(),
        items: Vec::new(),
    };
    if shard.skipped {
        return Ok(shard);
    }
    let mut lines = Vec::new();
    let mut metas = Vec::new();
    let file = File::open(input).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", input.display())))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let lite: Lite = serde_json::from_str(&line).map_err(|e| invalid(format!("{}: {e}", input.display())))?;
        if lite.meta.board_capacity != BoardCapacity::current() {
            return Err(invalid(format!(
                "{}: corpus built at capacity {:?}, this binary is {:?}",
                input.display(),
                lite.meta.board_capacity,
                BoardCapacity::current()
            )));
        }
        shard.seeds.push(lite.meta.seed);
        shard.samples.push(lite.samples.len());
        metas.push(lite.meta);
        lines.push(zstd::encode_all(line.as_bytes(), ZSTD_LEVEL)?);
    }
    shard.firsts = first_drives(&metas);
    for (t, &n) in shard.samples.iter().enumerate() {
        for r in split_items(n, chunk_samples as usize) {
            shard.items.push((t, r.start as u32, r.end as u32));
        }
    }
    shard.lines = Arc::new(lines);
    Ok(shard)
}

/// Every shard of a request, read in parallel (a generation is ~1.5 GB of JSON).
pub fn load_inputs(req: &LabelJobRequest) -> io::Result<Vec<LabelInput>> {
    if req.shards.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "label job with no shards"));
    }
    if req.cfg.playouts == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--playouts must be > 0"));
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = req
            .shards
            .iter()
            .map(|s| scope.spawn(|| load_input(&s.name, &s.input, &s.out, req.chunk_samples)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(io::Error::other("reading a shard panicked")))
            })
            .collect()
    })
}

/// A shard whose every item is in, to be written off the hub's lock.
pub struct ShardWrite {
    pub job: JobId,
    pub unit: usize,
    pub out: PathBuf,
    pub lines: Arc<Vec<Vec<u8>>>,
    /// Per trajectory: its labels, or `None` when it does not replay.
    pub labels: Vec<Option<Vec<f32>>>,
    pub tag: String,
}

impl ShardWrite {
    /// Write `out.partial`, then rename it to `out`: the local tool's file, line for line.
    pub fn run(&self) -> io::Result<()> {
        if let Some(dir) = self.out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let partial = self.out.with_extension("jsonl.partial");
        let mut w = io::BufWriter::new(File::create(&partial)?);
        for (line, labels) in self.lines.iter().zip(&self.labels) {
            let json = zstd::decode_all(&line[..])?;
            let mut traj: Trajectory = serde_json::from_slice(&json).map_err(|e| invalid(e.to_string()))?;
            apply_labels(&mut traj, labels.as_deref(), &self.tag);
            w.write_all(serde_json::to_string(&traj)?.as_bytes())?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        drop(w);
        std::fs::rename(&partial, &self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_hub_proto::{BoardDims, Evaluator, SearchConfig};
    use botbowl_play::generate::{play_trajectory, GenMode, GenerateConfig, RandomStartBias};

    fn trajectory(seed: u64) -> Trajectory {
        let board = BoardDims::try_new(16, 9, 4).unwrap_or_else(|_| BoardDims::from_env());
        let cfg = GenerateConfig {
            mode: GenMode::RandomStart,
            search: SearchConfig::iterations(2),
            evaluator: Evaluator::Heuristic,
            model: None,
            max_steps: 30,
            lecture: None,
            difficulty: botbowl_curriculum::lecture::Difficulty::Easy,
            bias: RandomStartBias::default(),
            board_sizes: Some(botbowl_play::board_sizes::SizeDist::single(board)),
            config_name: None,
            exploration: None,
            next_drive: false,
        };
        play_trajectory(&cfg, None, seed).unwrap().remove(0)
    }

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hub-label-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Lines as the local tool reads them, items covering every sample once, and a follow-on
    /// record pointed at its first drive wherever it sits.
    #[test]
    fn a_shard_reads_into_items_and_first_drives() {
        let d = dir("read");
        let first = trajectory(41);
        // A stand-in follow-on: same seed, `drive` 2 (reading only looks at the metadata).
        let mut next = trajectory(42);
        next.meta.seed = first.meta.seed;
        next.meta.extra.insert("drive".into(), "2".into());
        let other = trajectory(43);
        let line = |t: &Trajectory| serde_json::to_string(t).unwrap();
        // A blank line and a CRLF ending, both of which `BufRead::lines` absorbs.
        let text = format!("{}\n\n{}\r\n{}\n", line(&next), line(&other), line(&first));
        let input = d.join("shard0.jsonl");
        std::fs::write(&input, text).unwrap();
        let shard = load_input("shard0", &input, &d.join("out/shard0.jsonl"), 7).unwrap();
        assert!(!shard.skipped);
        assert_eq!(shard.lines.len(), 3);
        assert_eq!(shard.firsts, vec![Some(2), None, None]);
        assert_eq!(shard.seeds[0], first.meta.seed);
        let lens = [next.samples.len(), other.samples.len(), first.samples.len()];
        assert_eq!(shard.samples, lens);
        for (t, &n) in lens.iter().enumerate() {
            let mine: Vec<(u32, u32)> = shard.items.iter().filter(|i| i.0 == t).map(|i| (i.1, i.2)).collect();
            let flat: Vec<u32> = mine.iter().flat_map(|&(a, b)| a..b).collect();
            assert_eq!(flat, (0..n as u32).collect::<Vec<_>>(), "trajectory {t}");
            assert!(mine.iter().all(|&(a, b)| b - a <= 7));
        }
        // Items run trajectory by trajectory.
        assert!(shard.items.windows(2).all(|w| w[0].0 <= w[1].0));
        // The shipped bytes are the line itself.
        let back = zstd::decode_all(&shard.lines[2][..]).unwrap();
        assert_eq!(back, line(&first).into_bytes());

        // Written back with no labels (all unlabellable), every line is the local tool's
        // re-serialisation of the input.
        let w = ShardWrite {
            job: 0,
            unit: 0,
            out: d.join("out/shard0.jsonl"),
            lines: Arc::clone(&shard.lines),
            labels: vec![None; 3],
            tag: "t".into(),
        };
        w.run().unwrap();
        let out = std::fs::read_to_string(d.join("out/shard0.jsonl")).unwrap();
        assert_eq!(out, format!("{}\n{}\n{}\n", line(&next), line(&other), line(&first)));
        // Once the output exists, a shard is skipped and not even read.
        let again = load_input("shard0", &input, &d.join("out/shard0.jsonl"), 7).unwrap();
        assert!(again.skipped && again.lines.is_empty());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_corpus_of_another_capacity_is_refused() {
        let d = dir("capacity");
        let mut t = trajectory(44);
        t.meta.board_capacity.width += 2;
        let input = d.join("shard0.jsonl");
        std::fs::write(&input, serde_json::to_string(&t).unwrap() + "\n").unwrap();
        let e = load_input("shard0", &input, &d.join("out.jsonl"), 32)
            .err()
            .expect("refused");
        assert!(e.to_string().contains("capacity"), "{e}");
        std::fs::remove_dir_all(&d).ok();
    }
}
