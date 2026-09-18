use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};

use botbowl_curriculum::lecture::Difficulty;
use botbowl_hub::api::{
    BotReq, EvalJobRequest, GenerateJobRequest, HubStatus, JobKind, JobRequest, JobState, JobStatus, RungReq, ShardReq,
    Submitted,
};
use botbowl_hub::http::request;
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{Evaluator, GenerateConfig, SearchConfig};
use botbowl_play::bots::{candidate_label, evaluator_label, parse_backup, parse_puct, CandidateBot};
use botbowl_play::generate::{GenMode, RandomStartBias};

#[derive(Parser, Debug)]
#[command(name = "botbowl-hub", about = "Job queue for distributed generation/eval (plan 041)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the daemon: worker websocket + control API on one port.
    Serve(ServeArgs),
    /// Submit a job to a running daemon.
    Job {
        #[command(subcommand)]
        job: JobCommand,
    },
    /// Print the daemon's status.
    Status(ClientArgs),
}

#[derive(Args, Debug)]
struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0:7777")]
    bind: SocketAddr,
    /// Shared secret. Created (random) and printed if the file does not exist.
    #[arg(long, default_value = "hub.token")]
    token_file: PathBuf,
    /// Accept workers built from another commit (plan 041 decision 5).
    #[arg(long, default_value_t = false)]
    allow_commit_mismatch: bool,
}

#[derive(Args, Debug, Clone)]
struct ClientArgs {
    /// Daemon control URL.
    #[arg(long, default_value = "http://127.0.0.1:7777")]
    hub: String,
    #[arg(long, default_value = "hub.token")]
    token_file: PathBuf,
}

#[derive(Subcommand, Debug)]
enum JobCommand {
    /// Opponent-ladder eval of a candidate; same flags as `botbowl-ui eval`'s ladder.
    Eval(EvalJobArgs),
    /// Corpus shards; same flags as `botbowl-ui dataset`, plus which shards.
    Generate(GenerateJobArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum CliGenMode {
    #[default]
    SelfPlay,
    Curriculum,
    RandomStart,
}

impl From<CliGenMode> for GenMode {
    fn from(m: CliGenMode) -> Self {
        match m {
            CliGenMode::SelfPlay => GenMode::SelfPlay,
            CliGenMode::Curriculum => GenMode::Curriculum,
            CliGenMode::RandomStart => GenMode::RandomStart,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum CliDifficulty {
    #[default]
    Easy,
    Medium,
    Hard,
}

impl From<CliDifficulty> for Difficulty {
    fn from(d: CliDifficulty) -> Self {
        match d {
            CliDifficulty::Easy => Difficulty::Easy,
            CliDifficulty::Medium => Difficulty::Medium,
            CliDifficulty::Hard => Difficulty::Hard,
        }
    }
}

/// `botbowl-ui dataset` flag-for-flag, except that one job writes several
/// shards: `--out-dir D --shards "0 1 2"` writes `D/shard0.jsonl` .. with
/// shard `K` seeded at `seed_base + K * shard_seed_stride`, which is the
/// `SEED_BASE + G*1e6 + K*1e5` layout `train_loop.sh` has always used.
/// `--heuristic-shards` names shards that ignore `--evaluator/--model`
/// (the loop's heuristic hedge). `--out FILE` is the single-shard form.
#[derive(Args, Debug)]
struct GenerateJobArgs {
    #[command(flatten)]
    client: ClientArgs,
    #[arg(long, value_enum, default_value_t = CliGenMode::SelfPlay)]
    mode: CliGenMode,
    /// Directory for `shard<K>.jsonl`; required unless --out is given.
    #[arg(long, conflicts_with = "out")]
    out_dir: Option<PathBuf>,
    /// One shard, this file (as `botbowl-ui dataset --out`).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Shard indices, space- or comma-separated.
    #[arg(long, default_value = "0")]
    shards: String,
    /// Shards played with the heuristic evaluator regardless of --evaluator.
    #[arg(long, default_value = "")]
    heuristic_shards: String,
    /// Truncate shard files at submit instead of appending.
    #[arg(long, default_value_t = false)]
    truncate: bool,
    /// Games per shard.
    #[arg(long, default_value_t = 1)]
    games: u32,
    /// Shard K's first seed is `seed_base + K * shard_seed_stride`; game g adds g.
    #[arg(long, default_value_t = 0, alias = "seed")]
    seed_base: u64,
    #[arg(long, default_value_t = 100_000)]
    shard_seed_stride: u64,
    #[arg(long, default_value_t = 1000)]
    mcts_iters: usize,
    #[arg(long)]
    mcts_time_ms: Option<u64>,
    #[arg(long, default_value_t = 1)]
    mcts_workers: usize,
    #[arg(long, default_value_t = 100_000)]
    max_steps: u32,
    /// (curriculum mode) Lecture name.
    #[arg(long)]
    lecture: Option<String>,
    #[arg(long, value_enum, default_value_t = CliDifficulty::Easy)]
    difficulty: CliDifficulty,
    // Random-start placement biases; unset = `RandomStartBias::default()`,
    // the same numbers `botbowl-ui dataset` defaults to.
    #[arg(long)]
    ball_distance: Option<f32>,
    #[arg(long)]
    front_line: Option<f32>,
    #[arg(long)]
    mark_teammate: Option<f32>,
    #[arg(long)]
    mark_opponent: Option<f32>,
    #[arg(long)]
    own_side: Option<f32>,
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    temperature2: Option<f32>,
    #[arg(long)]
    carried_prob: Option<f32>,
    #[arg(long)]
    line_fraction: Option<f32>,
    #[arg(long)]
    pocket_fraction: Option<f32>,
    #[arg(long, value_enum, default_value_t = CliEvaluator::Heuristic)]
    evaluator: CliEvaluator,
    /// ONNX path; stamped into the corpus provenance exactly as written.
    #[arg(long)]
    model: Option<String>,
    /// Games per task handed to a worker.
    #[arg(long, default_value_t = 4)]
    batch: u16,
    /// Block until the job finishes; exit nonzero if it failed.
    #[arg(long, default_value_t = false)]
    wait: bool,
    /// Accepted and ignored (the hub sizes workers, not jobs).
    #[arg(long, hide = true)]
    parallel_games: Option<u32>,
    /// Accepted and ignored (workers own their sidecar).
    #[arg(long, hide = true)]
    nn_server: Option<String>,
}

fn parse_shards(s: &str) -> Result<Vec<u32>, String> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u32>().map_err(|e| format!("--shards: {t:?}: {e}")))
        .collect()
}

fn build_generate_request(a: &GenerateJobArgs) -> Result<GenerateJobRequest, String> {
    let budget = match a.mcts_time_ms {
        Some(ms) => botbowl_mcts::SearchBudget::Time(Duration::from_millis(ms)),
        None => botbowl_mcts::SearchBudget::Iterations(a.mcts_iters),
    };
    let d = RandomStartBias::default();
    let bias = RandomStartBias {
        ball_distance: a.ball_distance.unwrap_or(d.ball_distance),
        front_line: a.front_line.unwrap_or(d.front_line),
        mark_teammate: a.mark_teammate.unwrap_or(d.mark_teammate),
        mark_opponent: a.mark_opponent.unwrap_or(d.mark_opponent),
        own_side: a.own_side.unwrap_or(d.own_side),
        temperature: a.temperature.unwrap_or(d.temperature),
        temperature2: a.temperature2.unwrap_or(d.temperature2),
        carried_prob: a.carried_prob.unwrap_or(d.carried_prob),
        line_fraction: a.line_fraction.unwrap_or(d.line_fraction),
        pocket_fraction: a.pocket_fraction.unwrap_or(d.pocket_fraction),
    };
    let evaluator = Evaluator::from(a.evaluator);
    if evaluator.needs_model() && a.model.is_none() {
        return Err("--evaluator nn/nn-value requires --model PATH".into());
    }
    let base = GenerateConfig {
        mode: a.mode.into(),
        search: SearchConfig {
            budget,
            workers: a.mcts_workers,
            // `dataset` leaves these to the bot's env-driven defaults. The
            // backup rule is stamped into the provenance label, so resolve
            // it *here*, from the submitting environment, rather than on
            // whichever worker happens to play the game.
            puct: None,
            horizon_turns: None,
            backup: Some(botbowl_mcts::BackupMode::from_env()),
            fpu_reduction: None,
        },
        evaluator,
        model: a.model.clone(),
        max_steps: a.max_steps,
        lecture: a.lecture.clone(),
        difficulty: a.difficulty.into(),
        bias,
    };
    let heuristic = GenerateConfig {
        evaluator: Evaluator::Heuristic,
        model: None,
        ..base.clone()
    };
    let model_path = a.model.as_ref().map(|m| abs(&PathBuf::from(m)));
    let mut shards = Vec::new();
    if let Some(out) = &a.out {
        shards.push(ShardReq {
            name: out
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "out".into()),
            out: abs(out),
            seed: a.seed_base,
            games: a.games,
            cfg: base.clone(),
            model_path: model_path.clone(),
        });
    } else {
        let Some(dir) = &a.out_dir else {
            return Err("pass --out-dir DIR (with --shards) or --out FILE".into());
        };
        let heur = parse_shards(&a.heuristic_shards)?;
        let mut ks = parse_shards(&a.shards)?;
        ks.extend(heur.iter().copied());
        ks.sort_unstable();
        ks.dedup();
        if ks.is_empty() {
            return Err("--shards is empty".into());
        }
        for k in ks {
            let is_heur = heur.contains(&k);
            shards.push(ShardReq {
                name: format!("shard{k}"),
                out: abs(&dir.join(format!("shard{k}.jsonl"))),
                seed: a.seed_base + k as u64 * a.shard_seed_stride,
                games: a.games,
                cfg: if is_heur { heuristic.clone() } else { base.clone() },
                model_path: if is_heur { None } else { model_path.clone() },
            });
        }
    }
    Ok(GenerateJobRequest {
        shards,
        truncate: a.truncate,
        batch: a.batch,
    })
}

#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum CliEvaluator {
    #[default]
    Heuristic,
    PureTd,
    Nn,
    NnValue,
}

impl From<CliEvaluator> for Evaluator {
    fn from(e: CliEvaluator) -> Self {
        match e {
            CliEvaluator::Heuristic => Evaluator::Heuristic,
            CliEvaluator::PureTd => Evaluator::PureTd,
            CliEvaluator::Nn => Evaluator::Nn,
            CliEvaluator::NnValue => Evaluator::NnValue,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum CliCandidateBot {
    #[default]
    Mcts,
    Scripted,
    Random,
}

impl From<CliCandidateBot> for CandidateBot {
    fn from(c: CliCandidateBot) -> Self {
        match c {
            CliCandidateBot::Mcts => CandidateBot::Mcts,
            CliCandidateBot::Scripted => CandidateBot::Scripted,
            CliCandidateBot::Random => CandidateBot::Random,
        }
    }
}

/// Flag-for-flag the ladder half of `botbowl-ui eval`, so `train_loop.sh`
/// swaps the binary name and nothing else. Lecture flags are accepted and
/// ignored: the battery does not run on the hub.
#[derive(Args, Debug)]
struct EvalJobArgs {
    #[command(flatten)]
    client: ClientArgs,
    #[arg(long, value_enum, default_value_t = CliEvaluator::Heuristic)]
    evaluator: CliEvaluator,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 1000)]
    mcts_iters: usize,
    #[arg(long, default_value_t = 1)]
    mcts_workers: usize,
    #[arg(long, default_value_t = 50)]
    games: u32,
    #[arg(long)]
    vs_games: Option<u32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    max_steps: u32,
    #[arg(long)]
    opponent_iters: Option<usize>,
    /// Accepted for CLI compatibility; the hub never runs lectures.
    #[arg(long, default_value_t = true, hide = true)]
    skip_lectures: bool,
    #[arg(long, default_value_t = 100, hide = true)]
    trials: u32,
    #[arg(long, value_enum)]
    vs_evaluator: Option<CliEvaluator>,
    #[arg(long)]
    vs_model: Option<PathBuf>,
    #[arg(long, default_value = "raw")]
    puct_mode: String,
    #[arg(long)]
    puct_c: Option<f32>,
    #[arg(long)]
    vs_puct_mode: Option<String>,
    #[arg(long)]
    vs_puct_c: Option<f32>,
    #[arg(long, default_value_t = 1)]
    horizon_turns: u8,
    #[arg(long)]
    vs_horizon_turns: Option<u8>,
    #[arg(long, default_value = "minimax")]
    backup: String,
    #[arg(long)]
    vs_backup: Option<String>,
    #[arg(long, default_value_t = 0.0)]
    fpu_reduction: f32,
    #[arg(long)]
    vs_fpu_reduction: Option<f32>,
    #[arg(long, default_value_t = false)]
    skip_fixed_rungs: bool,
    #[arg(long, default_value = "random,scripted,mcts-heuristic")]
    rungs: String,
    #[arg(long, value_enum, default_value_t = CliCandidateBot::Mcts)]
    candidate_bot: CliCandidateBot,
    /// `report.json`.
    #[arg(long)]
    out: PathBuf,
    /// One JSON line per game.
    #[arg(long)]
    per_game_out: PathBuf,
    /// Games per task handed to a worker.
    #[arg(long, default_value_t = 4)]
    batch: u16,
    /// Block until the job finishes; exit nonzero if it failed.
    #[arg(long, default_value_t = false)]
    wait: bool,
    /// Accepted and ignored (the hub sizes workers, not jobs).
    #[arg(long, hide = true)]
    parallel_games: Option<u32>,
    /// Accepted and ignored (workers own their sidecar).
    #[arg(long, hide = true)]
    nn_server: Option<String>,
}

fn abs(p: &PathBuf) -> PathBuf {
    if p.is_absolute() {
        p.clone()
    } else {
        std::env::current_dir().expect("cwd").join(p)
    }
}

fn build_request(a: &EvalJobArgs) -> Result<EvalJobRequest, String> {
    let evaluator = Evaluator::from(a.evaluator);
    let cand = SearchConfig {
        budget: botbowl_mcts_budget(a.mcts_iters),
        workers: a.mcts_workers,
        puct: Some(parse_puct(&a.puct_mode, a.puct_c).map_err(|e| format!("--puct-mode: {e}"))?),
        horizon_turns: Some(a.horizon_turns),
        backup: Some(parse_backup(&a.backup).map_err(|e| format!("--backup: {e}"))?),
        fpu_reduction: Some(a.fpu_reduction),
    };
    let opp = SearchConfig {
        budget: botbowl_mcts_budget(a.opponent_iters.unwrap_or(a.mcts_iters)),
        workers: a.mcts_workers,
        puct: Some(
            parse_puct(
                a.vs_puct_mode.as_deref().unwrap_or(&a.puct_mode),
                a.vs_puct_c.or(a.puct_c),
            )
            .map_err(|e| format!("--vs-puct-mode: {e}"))?,
        ),
        horizon_turns: Some(a.vs_horizon_turns.unwrap_or(a.horizon_turns)),
        backup: Some(
            parse_backup(a.vs_backup.as_deref().unwrap_or(&a.backup)).map_err(|e| format!("--vs-backup: {e}"))?,
        ),
        fpu_reduction: Some(a.vs_fpu_reduction.unwrap_or(a.fpu_reduction)),
    };
    let model_str = a.model.as_ref().map(|p| p.to_string_lossy().into_owned());
    let candidate = match CandidateBot::from(a.candidate_bot) {
        CandidateBot::Mcts => BotReq::Mcts {
            search: cand,
            evaluator,
            model: a.model.as_ref().map(abs),
        },
        CandidateBot::Scripted => BotReq::Scripted,
        CandidateBot::Random => BotReq::Random,
    };
    let mut rungs = Vec::new();
    if !a.skip_fixed_rungs {
        for name in a.rungs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let opponent = match name {
                "random" => BotReq::Random,
                "scripted" => BotReq::Scripted,
                "mcts-heuristic" => BotReq::Mcts {
                    search: opp,
                    evaluator: Evaluator::Heuristic,
                    model: None,
                },
                other => {
                    return Err(format!(
                        "--rungs: expected `random`, `scripted` or `mcts-heuristic`, got `{other}`"
                    ))
                }
            };
            rungs.push(RungReq {
                name: name.to_string(),
                games: a.games,
                opponent,
            });
        }
    }
    if let Some(vs) = a.vs_evaluator {
        let vs = Evaluator::from(vs);
        let (opp_puct, opp_h, opp_b, opp_f) = (
            opp.puct.unwrap(),
            opp.horizon_turns.unwrap(),
            opp.backup.unwrap(),
            opp.fpu_reduction.unwrap(),
        );
        let (cand_h, cand_b, cand_f) = (
            cand.horizon_turns.unwrap(),
            cand.backup.unwrap(),
            cand.fpu_reduction.unwrap(),
        );
        // Same label as `botbowl-ui eval` builds, so downstream scripts
        // keep matching on it.
        let label = format!(
            "vs:{} [{}{}{}{}]",
            evaluator_label(vs, a.vs_model.as_ref().map(|p| p.to_string_lossy()).as_deref()),
            opp_puct.label(),
            if opp_h != cand_h {
                format!(" horizon={opp_h}v{cand_h}")
            } else {
                String::new()
            },
            if opp_b != cand_b {
                format!(" {}v{}", opp_b.label(), cand_b.label())
            } else {
                String::new()
            },
            if opp_f != cand_f {
                format!(" fpu_k={opp_f}v{cand_f}")
            } else {
                String::new()
            },
        );
        rungs.push(RungReq {
            name: label,
            games: a.vs_games.unwrap_or(a.games),
            opponent: BotReq::Mcts {
                search: opp,
                evaluator: vs,
                model: a.vs_model.as_ref().map(abs),
            },
        });
    }
    if rungs.is_empty() {
        return Err("no rungs: pass --rungs and/or --vs-evaluator".into());
    }
    Ok(EvalJobRequest {
        candidate,
        candidate_label: candidate_label(
            CandidateBot::from(a.candidate_bot),
            &cand,
            evaluator,
            model_str.as_deref(),
        ),
        mcts_iters: a.mcts_iters,
        rungs,
        seed: a.seed,
        max_steps: a.max_steps,
        per_game_out: abs(&a.per_game_out),
        report_out: abs(&a.out),
        batch: a.batch,
    })
}

fn print_generate_lines(s: &JobStatus) {
    let (mut games, mut samples) = (0u64, 0u64);
    for u in &s.units {
        println!(
            "  {:12} {:>5}/{:<5} games  {:>8} samples",
            u.name, u.done, u.total, u.samples
        );
        games += u.done as u64;
        samples += u.samples;
    }
    println!(
        "wrote {games} trajectories / {samples} samples in {} s (commit {}{})",
        s.elapsed_secs,
        botbowl_data::git_commit(),
        if botbowl_data::git_dirty() { "-dirty" } else { "" },
    );
}

fn botbowl_mcts_budget(iters: usize) -> botbowl_mcts::SearchBudget {
    botbowl_mcts::SearchBudget::Iterations(iters)
}

fn read_token(path: &PathBuf) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => {
            eprintln!("cannot read token file {}: {e}", path.display());
            std::process::exit(2)
        }
    }
}

fn or_create_token(path: &PathBuf) -> String {
    if let Ok(s) = std::fs::read_to_string(path) {
        return s.trim().to_string();
    }
    use rand::distributions::Alphanumeric;
    use rand::Rng;
    let token: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    std::fs::write(path, format!("{token}\n")).unwrap_or_else(|e| {
        eprintln!("cannot write token file {}: {e}", path.display());
        std::process::exit(2)
    });
    eprintln!("[hub] new token written to {}", path.display());
    token
}

fn print_report_lines(s: &JobStatus) {
    if let Some(r) = &s.report {
        println!("== report card: {} ==", r.candidate);
        for row in &r.ladder {
            println!(
                "  ladder  vs {:16} win_rate {:.2}  (W{} D{} L{})  [home {}-{} away {}-{}]  TD {}:{}  [side TD H{} A{}]{}",
                row.opponent,
                row.win_rate,
                row.wins,
                row.draws,
                row.losses,
                row.wins_as_home,
                row.losses_as_home,
                row.wins_as_away,
                row.losses_as_away,
                row.tds_for,
                row.tds_against,
                row.tds_by_home,
                row.tds_by_away,
                if row.unfinished > 0 { format!("  [{} unfinished]", row.unfinished) } else { String::new() },
            );
        }
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(a) => {
            let token = or_create_token(&a.token_file);
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async move {
                let (_hub, addr, task) = Hub::start(HubConfig {
                    bind: a.bind,
                    token,
                    allow_commit_mismatch: a.allow_commit_mismatch,
                })
                .await
                .unwrap_or_else(|e| {
                    eprintln!("[hub] cannot bind {}: {e}", a.bind);
                    std::process::exit(1)
                });
                eprintln!(
                    "[hub] listening on {addr} (commit {}{}); workers dial ws://<host>:{}/ws",
                    &botbowl_data::git_commit()[..12],
                    if botbowl_data::git_dirty() { "-dirty" } else { "" },
                    addr.port()
                );
                tokio::select! {
                    _ = task => {}
                    _ = tokio::signal::ctrl_c() => eprintln!("[hub] shutting down"),
                }
            });
        }
        Command::Status(c) => {
            let token = read_token(&c.token_file);
            match request("GET", &format!("{}/api/status", c.hub), &token, None) {
                Ok((200, body)) => {
                    let s: HubStatus = serde_json::from_str(&body).expect("status json");
                    println!("{}", serde_json::to_string_pretty(&s).unwrap());
                }
                Ok((code, body)) => {
                    eprintln!("hub returned {code}: {body}");
                    std::process::exit(1)
                }
                Err(e) => {
                    eprintln!("cannot reach hub at {}: {e}", c.hub);
                    std::process::exit(1)
                }
            }
        }
        Command::Job { job } => {
            let (client, wait, req, what) = match &job {
                JobCommand::Eval(a) => {
                    let req = build_request(a).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2)
                    });
                    let what = format!(
                        "eval job: {} rung(s), {} games",
                        req.rungs.len(),
                        req.rungs.iter().map(|r| r.games).sum::<u32>()
                    );
                    (a.client.clone(), a.wait, JobRequest::Eval(req), what)
                }
                JobCommand::Generate(a) => {
                    let req = build_generate_request(a).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2)
                    });
                    let what = format!(
                        "generate job: {} shard(s), {} games",
                        req.shards.len(),
                        req.shards.iter().map(|s| s.games).sum::<u32>()
                    );
                    (a.client.clone(), a.wait, JobRequest::Generate(req), what)
                }
            };
            let token = read_token(&client.token_file);
            let body = serde_json::to_string(&req).unwrap();
            let id = match request("POST", &format!("{}/api/jobs", client.hub), &token, Some(&body)) {
                Ok((200, body)) => serde_json::from_str::<Submitted>(&body).expect("submit json").id,
                Ok((code, body)) => {
                    eprintln!("hub refused the job ({code}): {body}");
                    std::process::exit(1)
                }
                Err(e) => {
                    eprintln!("cannot reach hub at {}: {e}", client.hub);
                    std::process::exit(1)
                }
            };
            eprintln!("[hub job] submitted {what} (job {id})");
            if !wait {
                println!("{id}");
                return;
            }
            let mut idle_polls: u64 = 0;
            loop {
                std::thread::sleep(Duration::from_secs(5));
                let (code, body) = match request("GET", &format!("{}/api/jobs/{id}", client.hub), &token, None) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[hub job] poll failed ({e}); retrying");
                        continue;
                    }
                };
                if code != 200 {
                    eprintln!("[hub job] hub returned {code}: {body}");
                    std::process::exit(1);
                }
                let s: JobStatus = serde_json::from_str(&body).expect("job json");
                match &s.state {
                    JobState::Running => {
                        // A job with nobody to run it never fails on its
                        // own; say so in the log rather than sit silent.
                        if s.workers_connected == 0 {
                            idle_polls += 1;
                            if idle_polls % 12 == 1 {
                                eprintln!(
                                    "[hub job] WARN: job {id} is running but no workers are connected ({}s idle so far)",
                                    idle_polls * 5
                                );
                            }
                        } else {
                            idle_polls = 0;
                        }
                    }
                    JobState::Done => {
                        match (&s.kind, &req) {
                            (JobKind::Eval, JobRequest::Eval(r)) => {
                                print_report_lines(&s);
                                println!("wrote {}", r.report_out.display());
                            }
                            _ => print_generate_lines(&s),
                        }
                        return;
                    }
                    JobState::Failed { error } => {
                        eprintln!("[hub job] job {id} failed: {error}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}
