//! Plan 064: adversarial probes of a net and its search. Each suite applies controlled edits to
//! base positions — replayed corpus decisions (`--corpus`) and regenerated drive positions
//! (`--positions`) — whose effect is known (a skill can only help its owner, the top-bottom
//! mirror changes nothing, one destination is best for a lone player wherever he starts), and
//! counts how often the net or the search gets them wrong, how badly, and keeps the worst cases.
//!
//! Everything runs on the **tract CPU backend** (no sidecar, no GPU) and is seeded: base states
//! come from `--seed`, every edit is deterministic, and policy-only results are byte-identical
//! across runs. Search results are not (`recon_mcts` HashMap order), which the stability suite
//! measures.
//!
//! ```text
//! UI=$(scripts/probe_build.sh)       # 16x9/6 capacity, niced, two jobs
//! taskset -c 6,7 nice -n 19 $UI probe skills --model models/az_v7/bbnet_mix16x9v9_gen39.onnx \
//!     --corpus runs/loopmix16x9v9/gen39/shard4.jsonl --board 16x9 --states 200 --parallel 2 \
//!     --out-dir runs/exp075
//! ```
//!
//! Output, per suite and tag (default: the model's file stem): `{suite}.{tag}.jsonl` (one row per
//! measurement), `{suite}.{tag}.summary.json` (the header, the tables' numbers), the table on
//! stdout and in `{suite}.{tag}.table.txt`, and `{suite}.{tag}.worst.json` with each worst case's
//! row, ASCII boards and a two-state `Recording` under `worst/`.

mod common;
mod dominance;
mod edits;
mod hillclimb;
mod lone_mover;
mod skills;
mod stability;
mod symmetry;

use std::io::{self, Write};
use std::time::Instant;

use botbowl_engine::core::gamestate::GameState;

use crate::cli::{ProbeArgs, ProbeSuite};
use common::{corpus_bases, position_bases, Base, Net, TopK};

/// What a suite hands back to be written.
pub struct SuiteOutput {
    pub rows: Vec<serde_json::Value>,
    pub summary: serde_json::Value,
    pub table: String,
    pub worst: TopK,
}

/// Base states: up to `--states` replayed corpus decisions and up to `--states` positions, each
/// passing `accept`.
pub fn bases(args: &ProbeArgs, accept: &dyn Fn(&GameState) -> bool) -> io::Result<Vec<Base>> {
    let mut out = Vec::new();
    if !args.corpus.is_empty() {
        out.extend(corpus_bases(
            &args.corpus,
            &args.boards,
            args.states,
            args.seed,
            accept,
        )?);
    }
    if !args.positions.is_empty() {
        out.extend(position_bases(
            &args.positions,
            &args.boards,
            args.states,
            args.seed ^ 0x5eed,
            accept,
        )?);
    }
    if out.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no base states: give --corpus and/or --positions (and check --board)",
        ));
    }
    Ok(out)
}

pub fn suite_name(s: ProbeSuite) -> &'static str {
    match s {
        ProbeSuite::Skills => "skills",
        ProbeSuite::LoneMover => "lone_mover",
        ProbeSuite::Symmetry => "symmetry",
        ProbeSuite::Dominance => "dominance",
        ProbeSuite::Stability => "stability",
        ProbeSuite::Hillclimb => "hillclimb",
    }
}

pub fn run(args: ProbeArgs) -> io::Result<()> {
    if std::env::var_os("BLOOD_NN_SERVER").is_some() {
        eprintln!("probe: BLOOD_NN_SERVER is ignored, the probes run on tract (CPU) only");
    }
    let started = Instant::now();
    let net = Net::load(&args.model)?;
    std::fs::create_dir_all(&args.out_dir)?;
    let tag = args.tag.clone().unwrap_or_else(|| {
        std::path::Path::new(&args.model)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("net")
            .to_string()
    });
    let name = suite_name(args.suite);
    let out = match args.suite {
        ProbeSuite::Skills => skills_suite(&args, &net)?,
        ProbeSuite::Symmetry => symmetry::run(&args, &net)?,
        ProbeSuite::Dominance => dominance::run(&args, &net)?,
        ProbeSuite::LoneMover => lone_mover::run(&args, &net)?,
        ProbeSuite::Stability => stability::run(&args, &net)?,
        ProbeSuite::Hillclimb => hillclimb::run(&args, &net)?,
    };
    let stem = format!("{name}.{tag}");
    let dir = &args.out_dir;
    let mut f = io::BufWriter::new(std::fs::File::create(format!("{dir}/{stem}.jsonl"))?);
    for r in &out.rows {
        serde_json::to_writer(&mut f, r)?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    let (commit, dirty) = botbowl_data::git_provenance();
    let header = serde_json::json!({
        "suite": name,
        "model": args.model,
        "tag": tag,
        "commit": commit,
        "dirty": dirty,
        "seed": args.seed,
        "states": args.states,
        "boards": args.boards,
        "corpus": args.corpus,
        "positions": args.positions,
        "search_config": args.search_config,
        "budgets": args.budgets,
        "parallel": args.parallel,
        "rows": out.rows.len(),
        "elapsed_s": started.elapsed().as_secs_f64(),
    });
    let summary = serde_json::json!({ "header": header, "summary": out.summary });
    std::fs::write(
        format!("{dir}/{stem}.summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    let table = format!(
        "probe {name} — {} [{}], {} rows, {:.0}s\n{}",
        args.model,
        &commit[..commit.len().min(10)],
        out.rows.len(),
        started.elapsed().as_secs_f64(),
        out.table
    );
    std::fs::write(format!("{dir}/{stem}.table.txt"), &table)?;
    let worst = if out.worst.len() > 0 {
        out.worst.write(dir, &stem)?
    } else {
        "(none)".into()
    };
    println!("{table}worst cases: {worst}");
    Ok(())
}

/// Suite 1 (see `skills.rs`): every player of every base state, every skill and characteristic.
fn skills_suite(args: &ProbeArgs, net: &Net) -> io::Result<SuiteOutput> {
    let bases = bases(args, &|_| true)?;
    eprintln!("probe skills: {} base states", bases.len());
    let results = common::par_map(bases.len(), args.parallel, |i| {
        let r = skills::probe_state(net, &bases[i], args.top);
        if (i + 1) % 25 == 0 {
            eprintln!("  {}/{} states", i + 1, bases.len());
        }
        r
    });
    let noise: Vec<f32> = results.iter().map(|r| r.noise).collect();
    let noise_d = common::dist(&noise);
    let mut worst = TopK::new(args.top);
    let mut rows = Vec::new();
    for r in results {
        worst.extend(r.worst);
        rows.extend(r.rows);
    }
    // The placebo floor: |ΔV| for skills no rule reads, on the same players of the same states.
    let placebo = common::dist(
        &rows
            .iter()
            .filter(|r| r.class == skills::SkillClass::Inert)
            .map(|r| r.dv.abs())
            .collect::<Vec<_>>(),
    );
    let tau = placebo.p99;
    let (edits, classes) = skills::summarise(&rows, tau);
    let table = skills::table(&edits, tau, &noise_d, &placebo);
    Ok(SuiteOutput {
        summary: serde_json::json!({ "tau": tau, "tau_rule": "p99 |dV| of inert-skill edits",
            "placebo_abs_dv": placebo, "flip_noise": noise_d, "edits": edits, "classes": classes }),
        rows: rows.iter().map(|r| serde_json::to_value(r).expect("row")).collect(),
        table,
        worst,
    })
}
