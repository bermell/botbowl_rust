use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand, ValueEnum};

use botbowl_hub::api::{BotReq, EvalJobRequest, HubStatus, JobState, JobStatus, RungReq, Submitted};
use botbowl_hub::http::request;
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{Evaluator, SearchConfig};
use botbowl_play::bots::{candidate_label, evaluator_label, parse_backup, parse_puct, CandidateBot};

#[derive(Parser, Debug)]
#[command(name = "botbowl-hub", about = "Job queue for distributed generation/eval (plan 040)")]
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
    /// Accept workers built from another commit (plan 040 decision 5).
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
        Command::Job {
            job: JobCommand::Eval(a),
        } => {
            let token = read_token(&a.client.token_file);
            let req = build_request(&a).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(2)
            });
            let body = serde_json::to_string(&req).unwrap();
            let id = match request("POST", &format!("{}/api/jobs", a.client.hub), &token, Some(&body)) {
                Ok((200, body)) => serde_json::from_str::<Submitted>(&body).expect("submit json").id,
                Ok((code, body)) => {
                    eprintln!("hub refused the job ({code}): {body}");
                    std::process::exit(1)
                }
                Err(e) => {
                    eprintln!("cannot reach hub at {}: {e}", a.client.hub);
                    std::process::exit(1)
                }
            };
            eprintln!(
                "[hub job] submitted eval job {id}: {} rung(s), {} games",
                req.rungs.len(),
                req.rungs.iter().map(|r| r.games).sum::<u32>()
            );
            if !a.wait {
                println!("{id}");
                return;
            }
            loop {
                std::thread::sleep(Duration::from_secs(5));
                let (code, body) = match request("GET", &format!("{}/api/jobs/{id}", a.client.hub), &token, None) {
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
                    JobState::Running => {}
                    JobState::Done => {
                        print_report_lines(&s);
                        println!("wrote {}", req.report_out.display());
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
