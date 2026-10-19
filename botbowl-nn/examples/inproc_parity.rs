//! Plan 066 parity harness, Rust side. Two modes:
//!
//! ```sh
//! # 1. Encode corpus states once (no GPU, no `torch` feature): u8 raw planes + f32 globals.
//! cargo run --release -p botbowl-nn --example inproc_parity -- dump --out states.bin --max 120000 shard4.jsonl ...
//! # 2. Run a batch plan (scripts/parity066.py plan) through the in-process graphs on one lane.
//! cargo run --release -p botbowl-nn --features torch --example inproc_parity -- \
//!     run --ts gen44.ts --states states.bin --plan plan.bin --out out.bin --lane 0 [--deterministic] [--graph g.txt]
//! ```
//!
//! `states.bin`: `b"BBST" | u32 version=1 | u32 C | u32 F | u32 N | f32 scales[C]`, then per state
//! `u16 h | u16 w | u8 raw[C·h·w] | f32 global[F]`. The f32 spatial input is `raw / scale` per
//! channel — exactly `encode` (checked for every state at dump time), and what numpy computes.
//!
//! `plan.bin`: `b"BBPL" | u32 batches`, then per batch `u32 n | u32 want_policy | u32 idx[n]`.
//! `out.bin`: per batch, `f32 value[n]`, then (if want_policy) `f32 policy[A·h·w]` per row, cropped.
//! `scripts/parity066.py` writes the plan, runs the same plan through the sidecar's own
//! `GraphRunner`, and compares the two `out.bin`s bit for bit.

use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;

use botbowl_data::DatasetReader;
use botbowl_nn::encode::{encode, encode_raw, spatial_channel_scales, GLOBAL_FEATURES, SPATIAL_CHANNELS};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("dump") => dump(&args[1..]),
        Some("run") => run(&args[1..]),
        _ => {
            eprintln!("usage: inproc_parity dump --out F [--max N] SHARD... | run --ts F --states F --plan F --out F [--lane L] [--deterministic] [--graph F]");
            std::process::exit(2);
        }
    }
}

fn dump(args: &[String]) {
    let out = PathBuf::from(arg(args, "--out").expect("--out"));
    let max: usize = arg(args, "--max").map_or(usize::MAX, |v| v.parse().unwrap());
    let mut shards = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with("--") {
            i += 2;
        } else {
            shards.push(args[i].clone());
            i += 1;
        }
    }
    let scales = spatial_channel_scales();
    // Streamed: the u8 planes of 100k+ states are gigabytes. `N` is patched in at the end.
    let mut f = BufWriter::new(std::fs::File::create(&out).expect("create out"));
    f.write_all(b"BBST").unwrap();
    for v in [1u32, SPATIAL_CHANNELS as u32, GLOBAL_FEATURES as u32, 0] {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    for s in &scales {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
    let body = &mut f;
    let mut n = 0u32;
    let mut dims = std::collections::BTreeMap::new();
    'outer: for s in &shards {
        for traj in DatasetReader::open(s).expect("open shard") {
            let traj = traj.expect("parse trajectory");
            for sample in &traj.samples {
                let raw = encode_raw(&sample.state);
                let enc = encode(&sample.state);
                let plane = raw.h * raw.w;
                for (k, (&r, &e)) in raw.spatial.iter().zip(&enc.spatial).enumerate() {
                    let f = r as f32 / scales[k / plane];
                    assert_eq!(f.to_bits(), e.to_bits(), "encode != raw / scale");
                }
                body.write_all(&(raw.h as u16).to_le_bytes()).unwrap();
                body.write_all(&(raw.w as u16).to_le_bytes()).unwrap();
                body.write_all(&raw.spatial).unwrap();
                for g in &raw.global {
                    body.write_all(&g.to_le_bytes()).unwrap();
                }
                *dims.entry((raw.w, raw.h)).or_insert(0usize) += 1;
                n += 1;
                if n as usize >= max {
                    break 'outer;
                }
            }
        }
    }
    let mut file = f.into_inner().unwrap();
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(16)).unwrap();
    file.write_all(&n.to_le_bytes()).unwrap();
    eprintln!("dumped {n} states to {} ({} boards):", out.display(), dims.len());
    for ((w, h), c) in dims {
        eprintln!("  {w}x{h}: {c}");
    }
}

/// A `states.bin`, held as its raw bytes (u8 planes are a quarter of f32) with an offset per state;
/// rows are decoded to f32 only when a batch needs them.
#[allow(dead_code)]
struct States {
    buf: Vec<u8>,
    scales: Vec<f32>,
    offsets: Vec<usize>,
}

#[allow(dead_code)]
impl States {
    fn open(path: &PathBuf) -> States {
        let mut buf = Vec::new();
        std::fs::File::open(path).unwrap().read_to_end(&mut buf).unwrap();
        assert_eq!(&buf[..4], b"BBST");
        let u32_at = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
        let (c, f, n) = (u32_at(8), u32_at(12), u32_at(16));
        assert_eq!((c, f), (SPATIAL_CHANNELS, GLOBAL_FEATURES));
        let scales: Vec<f32> = (0..c)
            .map(|k| f32::from_le_bytes(buf[20 + 4 * k..24 + 4 * k].try_into().unwrap()))
            .collect();
        let mut o = 20 + 4 * c;
        let mut offsets = Vec::with_capacity(n);
        for _ in 0..n {
            offsets.push(o);
            let h = u16::from_le_bytes([buf[o], buf[o + 1]]) as usize;
            let w = u16::from_le_bytes([buf[o + 2], buf[o + 3]]) as usize;
            o += 4 + c * h * w + 4 * f;
        }
        States { buf, scales, offsets }
    }

    /// `(spatial f32 C·h·w, global, h, w)` of state `i`.
    fn get(&self, i: usize) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let b = &self.buf;
        let o = self.offsets[i];
        let h = u16::from_le_bytes([b[o], b[o + 1]]) as usize;
        let w = u16::from_le_bytes([b[o + 2], b[o + 3]]) as usize;
        let plane = h * w;
        let raw = &b[o + 4..o + 4 + SPATIAL_CHANNELS * plane];
        let spatial: Vec<f32> = raw.iter().enumerate().map(|(k, &r)| r as f32 / self.scales[k / plane]).collect();
        let go = o + 4 + SPATIAL_CHANNELS * plane;
        let global: Vec<f32> = (0..GLOBAL_FEATURES)
            .map(|k| f32::from_le_bytes(b[go + 4 * k..go + 4 * k + 4].try_into().unwrap()))
            .collect();
        (spatial, global, h, w)
    }
}

#[cfg(not(feature = "torch"))]
fn run(_args: &[String]) {
    eprintln!("`run` needs the in-process backend: rebuild with --features torch");
    std::process::exit(2);
}

#[cfg(feature = "torch")]
fn run(args: &[String]) {
    use botbowl_nn::batch_queue::stage_canvas;
    use botbowl_nn::inproc::{set_cudnn, sidecar_validation_calls, CudnnMode, InProcConfig, TorchExecutor, TorchModel};

    let ts = PathBuf::from(arg(args, "--ts").expect("--ts"));
    let states = States::open(&PathBuf::from(arg(args, "--states").expect("--states")));
    let plan_path = PathBuf::from(arg(args, "--plan").expect("--plan"));
    let out = PathBuf::from(arg(args, "--out").expect("--out"));
    let lane: usize = arg(args, "--lane").map_or(0, |v| v.parse().unwrap());
    let device = arg(args, "--device").unwrap_or_else(|| "cuda".into());
    let mode = if args.iter().any(|a| a == "--deterministic") {
        CudnnMode::Deterministic
    } else {
        CudnnMode::Production
    };
    let optimize = !args.iter().any(|a| a == "--no-optimize");

    let mut cfg = InProcConfig::from_env(&device).unwrap();
    cfg.lanes = lane + 1;
    cfg.cudnn = mode;
    set_cudnn(mode).unwrap();
    let geo = botbowl_nn::batch_queue::Geometry {
        c: SPATIAL_CHANNELS,
        f: GLOBAL_FEATURES,
        a: botbowl_nn::actions::POLICY_CHANNELS,
        canvas_h: cfg.canvas.0,
        canvas_w: cfg.canvas.1,
    };
    let t0 = std::time::Instant::now();
    let model = TorchModel::load(&ts, &device, optimize).unwrap();
    sidecar_validation_calls(&model, &geo).unwrap();
    let exec = TorchExecutor::capture(model, geo, cfg.lanes).unwrap();
    eprintln!(
        "loaded + captured {} lanes on {device} ({mode:?}, canvas {}x{}) in {:.1}s",
        cfg.lanes,
        geo.canvas_h,
        geo.canvas_w,
        t0.elapsed().as_secs_f64()
    );
    if let Some(g) = arg(args, "--graph") {
        std::fs::write(g, exec.model.graph()).unwrap();
    }

    let mut buf = Vec::new();
    std::fs::File::open(&plan_path).unwrap().read_to_end(&mut buf).unwrap();
    assert_eq!(&buf[..4], b"BBPL");
    let u32_at = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as usize;
    let batches = u32_at(4);
    let mut o = 8;
    let mut f = BufWriter::new(std::fs::File::create(&out).unwrap());
    let t1 = std::time::Instant::now();
    let mut rows_done = 0usize;
    for _ in 0..batches {
        let n = u32_at(o);
        let want = u32_at(o + 4) != 0;
        o += 8;
        let idx: Vec<usize> = (0..n).map(|k| u32_at(o + 4 * k)).collect();
        o += 4 * n;
        let decoded: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> = idx
            .iter()
            .map(|&i| {
                let (s, g, h, w) = states.get(i);
                let (cs, m) = stage_canvas(&geo, &s, h, w);
                (cs, g, m, h, w)
            })
            .collect();
        let rows: Vec<(&[f32], &[f32], &[f32], usize, usize)> = decoded
            .iter()
            .map(|(s, g, m, h, w)| (s.as_slice(), g.as_slice(), m.as_slice(), *h, *w))
            .collect();
        let (values, policies) = exec.run_rows(lane, &rows, want).unwrap();
        for v in &values {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        for p in &policies {
            for x in p {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        rows_done += n;
    }
    f.flush().unwrap();
    eprintln!(
        "ran {batches} batches / {rows_done} rows on lane {lane} in {:.1}s -> {}",
        t1.elapsed().as_secs_f64(),
        out.display()
    );
}
