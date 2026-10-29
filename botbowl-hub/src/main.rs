use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::parser::ValueSource;
use clap::{ArgMatches, Args, CommandFactory, FromArgMatches, Parser, Subcommand};

use botbowl_hub::api::{
    BotReq, EvalJobRequest, GenerateJobRequest, HubStatus, JobKind, JobRequest, JobState, JobStatus, LabelJobRequest,
    LabelShardReq, RungReq, ShardReq, StreamLabelRequest, Submitted, UnitStats,
};
use botbowl_hub::http::{request, request_with_timeout};
use botbowl_hub::{Hub, HubConfig};
use botbowl_hub_proto::{BoardDims, Evaluator, GenerateConfig, SearchConfig};
use botbowl_play::bots::{candidate_label, evaluator_label, load_mcts_config, CandidateBot};
use botbowl_play::cli_args::{vs_rung_label, DatasetArgs, EvalArgs, McLabelArgs};
use botbowl_play::drives::{drive_rung_name, DriveRung, PositionSet};
use botbowl_play::eval::rung_name;
use botbowl_play::generate::Exploration;

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
    Status(StatusArgs),
}

#[derive(Args, Debug)]
struct ServeArgs {
    #[arg(long, default_value = "0.0.0.0:7777")]
    bind: SocketAddr,
    /// Shared secret; default `~/.config/botbowl/hub.token`. Created (random) only if the file
    /// does not exist, so every hub on this machine serves the same token until someone replaces it.
    #[arg(long)]
    token_file: Option<PathBuf>,
    /// Accept workers built from *any* commit (plan 041 decision 5). The blunt instrument, for
    /// hacking on the worker itself; use `--allowed-commits` to run a programme.
    #[arg(long, default_value_t = false)]
    allow_commit_mismatch: bool,
    /// Named commits a worker may also connect on, as TOML keyed by the hub's own commit — see
    /// `botbowl_hub::allowlist`. Untracked, and stale the moment you commit again. Absent file =
    /// exact match only, which is the default.
    #[arg(long, default_value = botbowl_hub::allowlist::DEFAULT_PATH)]
    allowed_commits: PathBuf,
    /// Seconds of silence before a worker is dropped and its games requeued.
    /// Workers heartbeat every 30 s; this catches the machine that goes away
    /// without closing its socket (a slept laptop), which is otherwise
    /// indistinguishable from a healthy one and strands its games.
    #[arg(long, default_value_t = 120)]
    worker_timeout: u64,
    /// The training loop's run directory (`runs/<run>`). The status page then also shows the
    /// loop's latest status lines and, while the box trains a net, the trainer's progress.
    #[arg(long)]
    run_dir: Option<PathBuf>,
    /// The project registry served at `/registry/` (read on every request, so edits show
    /// without a restart). Default `<repo>/registry`.
    #[arg(long)]
    registry_dir: Option<PathBuf>,
    /// Seconds per interval of the status page's rate history, and between the
    /// `[hub] rate ...` lines in hub.log (written only for intervals in which something
    /// finished; aligned to the wall clock).
    #[arg(long, default_value_t = 300)]
    rate_interval_secs: u64,
    #[command(flatten)]
    play: PlayArgs,
    /// Directories hashed at startup so a worker's cached nets can be named on connect (repeat
    /// the flag for more). Default `<repo>/runs` and `<repo>/models`.
    #[arg(long = "model-index-dir")]
    model_index_dirs: Vec<PathBuf>,
}

/// The web play app at `/play/` (botbowl-web-server's router, nested). Every path defaults from
/// this crate's source location, not the working directory.
#[derive(Args, Debug)]
struct PlayArgs {
    /// Do not serve `/play/`.
    #[arg(long, default_value_t = false)]
    no_play: bool,
    /// Only these client addresses (comma-separated, an address or a CIDR network) get the
    /// browser-facing pages and the web play app; others get 403. Loopback always does, and the
    /// workers' `/ws` never checks (token-authenticated). Unset = everyone, as before.
    #[arg(long = "allow-from", value_delimiter = ',')]
    allow_from: Vec<botbowl_hub::AllowNet>,
    /// `trunk build --release` output of `botbowl-web/client`. Without one, `/play/` is off.
    /// This and the other `--play-*-dir`s override `~/.config/botbowl/web.toml`.
    #[arg(long)]
    play_dist_dir: Option<PathBuf>,
    /// The sprite directory (`assets_dir` in web.toml). Default
    /// `<repo>/../botbowl/botbowl/web/static/img` when it exists.
    #[arg(long)]
    play_assets_dir: Option<PathBuf>,
    /// Where the lobby finds `.onnx` nets first (then web.toml's `models_dirs` and the worker
    /// cache). Default `<repo>/models`.
    #[arg(long)]
    play_models_dir: Option<PathBuf>,
    /// Saved teams and uploaded pictures. Default `~/.config/botbowl/teams`.
    #[arg(long)]
    play_teams_dir: Option<PathBuf>,
    /// The most search threads one bot may use in a browser game, whatever the lobby asks —
    /// the hub listens on the network, and its box is also training.
    #[arg(long, default_value_t = 4)]
    play_max_workers: usize,
}

/// The repo root, from this crate's source path.
fn repo_path(relative: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join(relative);
    std::fs::canonicalize(&path).unwrap_or(path)
}

/// The play app's router, or `None` (with the reason on stderr) when it is off. Paths: the
/// `--play-*` flags, then `~/.config/botbowl/web.toml`, then the built-in defaults.
fn play_router(a: &PlayArgs) -> Option<axum::Router> {
    use botbowl_web_server::{compiled_capacity, config, teams, AppState, PlayOptions};
    if a.no_play {
        return None;
    }
    let paths = match config::resolve(config::Flags {
        dist_dir: a.play_dist_dir.clone(),
        models_dir: a.play_models_dir.clone(),
        assets_dir: a.play_assets_dir.clone(),
        teams_dir: a.play_teams_dir.clone(),
    }) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[hub] /play off: {e}");
            return None;
        }
    };
    if !paths.dist_dir.join("index.html").is_file() {
        eprintln!(
            "[hub] /play off: no client build at {} (cd botbowl-web/client && trunk build --release)",
            paths.dist_dir.display()
        );
        return None;
    }
    if paths.assets_dir.is_none() {
        eprintln!(
            "[hub] /play: no sprite directory (assets_dir in {}, or --play-assets-dir); the pitch draws without pictures",
            paths.config.display()
        );
    }
    let capacity = compiled_capacity();
    let app = std::sync::Arc::new(AppState {
        capacity,
        models_dir: paths.models_dir.clone(),
        recordings_dir: repo_path("data/web-games"),
        model_cache: Default::default(),
        server: format!(
            "botbowl-hub {} (capacity {}x{}/{})",
            &botbowl_data::git_commit()[..12],
            capacity.width,
            capacity.height,
            capacity.team_size
        ),
        opts: PlayOptions {
            max_workers: Some(a.play_max_workers.max(1)),
            teams: teams::TeamStore {
                dir: paths.teams_dir.clone(),
            },
            assets_dir: paths.assets_dir.clone(),
            allow_recording_paths: false,
            extra_model_dirs: paths.extra_model_dirs.clone(),
            worker_cache: paths.worker_cache.clone(),
            ..PlayOptions::default()
        },
    });
    eprintln!(
        "[hub] /play: config {}, capacity {}x{}/{}, {} model(s), teams in {}, at most {} search thread(s) per bot",
        paths.config.display(),
        capacity.width,
        capacity.height,
        capacity.team_size,
        app.list_models().len(),
        paths
            .teams_dir
            .as_deref()
            .map(|d| d.display().to_string())
            .unwrap_or_else(|| "-".into()),
        a.play_max_workers.max(1),
    );
    Some(botbowl_web_server::router(
        app,
        paths.assets_dir.as_deref(),
        Some(&paths.dist_dir),
    ))
}

#[derive(Args, Debug, Clone)]
struct ClientArgs {
    /// Daemon control URL.
    #[arg(long, default_value = "http://127.0.0.1:7777")]
    hub: String,
    /// Default `~/.config/botbowl/hub.token`, the file `serve` uses.
    #[arg(long)]
    token_file: Option<PathBuf>,
    /// What the job is called on the status page (`gen03 drives vs gen21`). Ignored by `status`.
    #[arg(long)]
    label: Option<String>,
    /// Plan 071 R5: `background` takes only worker streams no foreground job is waiting for (A/B
    /// arms, benchmarks, drives); `foreground` is the default and what every job was before.
    /// Ignored by `status`.
    #[arg(long, default_value = "foreground")]
    priority: botbowl_hub::api::Priority,
}

/// How `job ... --wait` behaves when the hub goes away or time runs out (plan 069 item 5).
#[derive(Args, Debug, Clone, Default)]
struct WaitArgs {
    /// Unix time this job must be off the hub by (a Slurm job's end). `--drain-secs` before it the
    /// job is drained: no new games start, the ones on workers finish and are written; then (or at
    /// the deadline) the client exits 75, and the same job submitted with `--resume` later plays
    /// the rest. Needs `--wait`.
    #[arg(long)]
    deadline: Option<u64>,
    /// With `--deadline`: how long before it to stop handing out games.
    #[arg(long, default_value_t = 300)]
    drain_secs: u64,
    /// Exit 75 once the hub has been unreachable this long (0: keep trying for ever). A resumable
    /// job (`--resume`, any label job) is submitted again when a hub answers, whatever this says.
    #[arg(long, default_value_t = 0)]
    hub_lost_secs: u64,
}

#[derive(Args, Debug)]
struct StatusArgs {
    #[command(flatten)]
    client: ClientArgs,
    /// The status page as text, instead of the JSON.
    #[arg(long, default_value_t = false)]
    text: bool,
    /// With `--text`: the loop's run directory, as `serve --run-dir`.
    #[arg(long)]
    run_dir: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum JobCommand {
    /// Opponent-ladder eval of a candidate; same flags as `botbowl-ui eval`'s ladder.
    Eval(EvalJobArgs),
    /// Corpus shards; same flags as `botbowl-ui dataset`, plus which shards.
    Generate(GenerateJobArgs),
    /// Monte Carlo value labels (plan 062); same flags as `botbowl-ui mc-label`.
    Label(LabelJobArgs),
}

/// `botbowl-ui mc-label` flag for flag (the same [`McLabelArgs`], flattened): `--in` (or
/// `--corpus`) shards, each written to `--out-dir` under its own name, byte for byte what the local
/// tool writes on the same backend. The hub cuts every trajectory into work items of at most
/// `--chunk-samples` samples, workers label them with the `--model` net (their sidecar if they have
/// one, tract otherwise) and the hub writes each shard once its last item is in. A shard whose
/// output exists is skipped, as locally. `--parallel` and `--nn-server` are refused: workers size
/// themselves and own their sidecar.
#[derive(Args, Debug)]
struct LabelJobArgs {
    #[command(flatten)]
    client: ClientArgs,
    #[command(flatten)]
    ml: McLabelArgs,
    /// Most samples per work item; a longer trajectory is split into even ranges. 32 samples x 8
    /// playouts is ~15 s on a GPU-backed stream and a few minutes on a laptop's tract.
    #[arg(long, default_value_t = 32)]
    chunk_samples: u32,
    /// Work items per task handed to a worker.
    #[arg(long, default_value_t = 1)]
    batch: u16,
    /// Block until the job finishes; exit nonzero if it failed.
    #[arg(long, default_value_t = false)]
    wait: bool,
    #[command(flatten)]
    w: WaitArgs,
}

fn build_label_request(a: &LabelJobArgs, m: &ArgMatches) -> Result<LabelJobRequest, String> {
    refuse_local_flags(
        m,
        &[
            ("parallel", "workers size themselves (botbowl-worker --parallel-games)"),
            ("nn_server", "each worker owns its sidecar (botbowl-worker --nn-server)"),
            (
                "nn_backend",
                "each worker picks its own backend (botbowl-worker --nn-backend)",
            ),
            (
                "nn_device",
                "each worker picks its own device (botbowl-worker --nn-device)",
            ),
        ],
    )?;
    let ml = &a.ml;
    if ml.playouts == 0 {
        return Err("--playouts must be > 0".into());
    }
    if ml.top_up == Some(0) {
        return Err("--top-up must be > 0".into());
    }
    let out_dir = PathBuf::from(&ml.out_dir);
    let shards = ml
        .corpus
        .iter()
        .map(|input| {
            let input = PathBuf::from(input);
            let file = input
                .file_name()
                .ok_or_else(|| format!("{}: no file name", input.display()))?;
            Ok(LabelShardReq {
                name: input
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                input: abs(&input),
                out: abs(&out_dir.join(file)),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(LabelJobRequest {
        shards,
        model_path: abs(&PathBuf::from(&ml.model)),
        // As typed: the provenance tag must read as the local tool's would.
        model: ml.model.clone(),
        cfg: ml.config(),
        chunk_samples: a.chunk_samples.max(1),
        batch: a.batch.max(1),
        label: a.client.label.clone(),
        top_up: ml.top_up.is_some(),
        priority: a.client.priority,
    })
}

/// `botbowl-ui mc-label`'s summary line for every shard of a finished label job, on stderr where
/// the local tool prints it (`train_loop.sh` sums the unlabelled counts from these lines).
fn print_label_lines(s: &JobStatus, playouts: u32) {
    for u in &s.units {
        let Some(UnitStats::Label(l)) = &u.stats else { continue };
        if l.skipped {
            eprintln!("mc-label: {} exists, skipping", l.out);
        } else {
            eprintln!(
                "{}",
                botbowl_play::mc_label::summary_line(
                    &l.input,
                    &l.out,
                    l.trajectories as usize,
                    l.unlabelled as usize,
                    l.samples as usize,
                    playouts,
                    l.secs
                )
            );
        }
    }
    println!(
        "labelled {} samples over {} shard(s) in {} s (commit {}{})",
        s.units.iter().map(|u| u.samples).sum::<u64>(),
        s.units.len(),
        s.elapsed_secs,
        botbowl_data::git_commit(),
        if botbowl_data::git_dirty() { "-dirty" } else { "" },
    );
}

/// `botbowl-ui dataset` flag-for-flag (the same [`DatasetArgs`], flattened), except that one job
/// writes several shards: `--out-dir D --shards "0 1 2"` writes `D/shard0.jsonl` .. with shard `K`
/// seeded at `--seed-base + K * --shard-seed-stride` (`--seed-base` is `--seed`'s hub spelling),
/// which is the `SEED_BASE + G*1e6 + K*1e5` layout `train_loop.sh` has always used.
/// `--heuristic-shards` names shards that ignore `--evaluator/--model` (the loop's heuristic
/// hedge). `--out FILE` is the single-shard form. `--parallel-games` and `--nn-server` are refused:
/// workers size themselves and own their sidecar.
#[derive(Args, Debug)]
struct GenerateJobArgs {
    #[command(flatten)]
    client: ClientArgs,
    /// Directory for `shard<K>.jsonl`; required unless --out is given.
    #[arg(long, conflicts_with = "out")]
    out_dir: Option<PathBuf>,
    /// Shard indices, space- or comma-separated.
    #[arg(long, default_value = "0")]
    shards: String,
    /// Shards played with the heuristic evaluator regardless of --evaluator.
    #[arg(long, default_value = "")]
    heuristic_shards: String,
    /// Shard K's first seed is `--seed-base + K * shard_seed_stride`; game g adds g.
    #[arg(long, default_value_t = 100_000)]
    shard_seed_stride: u64,
    #[command(flatten)]
    ds: DatasetArgs,
    /// Games per task handed to a worker.
    #[arg(long, default_value_t = 4)]
    batch: u16,
    /// Keep each shard's complete games and play only the missing seeds (a torn tail is cut off);
    /// a missing shard starts empty. Excludes `--truncate`. See `botbowl_hub::resume`.
    #[arg(long, default_value_t = false, conflicts_with = "truncate")]
    resume: bool,
    /// Block until the job finishes; exit nonzero if it failed.
    #[arg(long, default_value_t = false)]
    wait: bool,
    #[command(flatten)]
    mc: StreamLabelArgs,
    #[command(flatten)]
    w: WaitArgs,
}

/// Plan 069 item 4b: label the shards while they are generated. Each record is queued for MC
/// labelling as it is written, under the generator's `--model` (as typed, for the provenance tag),
/// and `<--mc-label-out-dir>/<shard>.jsonl` is written once the shard has all its games and every
/// label: the file `job label --in <shard> --model <model> --playouts N --seed S --out-dir D` writes
/// from the finished shard. `--wait` then also waits for the labels and prints `job label`'s
/// summary lines; streamed labels that fail do not fail the generation (the loop's mc-label phase
/// labels what is missing).
#[derive(Args, Debug)]
struct StreamLabelArgs {
    /// Playouts per sample; 0 (the default) = no streamed labels.
    #[arg(long, default_value_t = 0)]
    mc_label_playouts: u32,
    /// `mc-label --seed`.
    #[arg(long, default_value_t = 56_000)]
    mc_label_seed: u64,
    /// `mc-label --max-steps`.
    #[arg(long, default_value_t = 100_000)]
    mc_label_max_steps: u32,
    /// `mc-label --out-dir`: required with `--mc-label-playouts`.
    #[arg(long)]
    mc_label_out_dir: Option<PathBuf>,
    /// Shard indices to label (space- or comma-separated); default every shard of the job.
    #[arg(long)]
    mc_label_shards: Option<String>,
    /// `job label --chunk-samples`.
    #[arg(long, default_value_t = 32)]
    mc_label_chunk_samples: u32,
    /// `job label --batch`.
    #[arg(long, default_value_t = 1)]
    mc_label_batch: u16,
}

/// Whether `id` was typed on the command line (not a default). How the hub tells a process-local
/// `botbowl-ui` flag it cannot honour, or `--out` given vs defaulted, from the shared struct.
fn given(m: &ArgMatches, id: &str) -> bool {
    matches!(m.value_source(id), Some(ValueSource::CommandLine))
}

/// Refuse the shared flags a hub job cannot honour, naming where they belong.
fn refuse_local_flags(m: &ArgMatches, ids: &[(&str, &str)]) -> Result<(), String> {
    for (id, why) in ids {
        if given(m, id) {
            return Err(format!("--{} is not a hub job flag: {why}", id.replace('_', "-")));
        }
    }
    Ok(())
}

fn parse_shards(s: &str) -> Result<Vec<u32>, String> {
    s.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u32>().map_err(|e| format!("--shards: {t:?}: {e}")))
        .collect()
}

fn build_generate_request(a: &GenerateJobArgs, m: &ArgMatches) -> Result<GenerateJobRequest, String> {
    refuse_local_flags(
        m,
        &[
            (
                "parallel_games",
                "workers size themselves (botbowl-worker --parallel-games)",
            ),
            ("nn_server", "each worker owns its sidecar (botbowl-worker --nn-server)"),
            (
                "nn_backend",
                "each worker picks its own backend (botbowl-worker --nn-backend)",
            ),
            (
                "nn_device",
                "each worker picks its own device (botbowl-worker --nn-device)",
            ),
        ],
    )?;
    let ds = &a.ds;
    let budget = match ds.mcts_time_ms {
        Some(ms) => botbowl_mcts::SearchBudget::Time(Duration::from_millis(ms)),
        None => botbowl_mcts::SearchBudget::Iterations(ds.mcts_iters),
    };
    let bias = ds.bias.to_bias();
    let evaluator = Evaluator::from(ds.evaluator);
    if evaluator.needs_model() && ds.model.is_none() {
        return Err("--evaluator nn/nn-value requires --model PATH".into());
    }
    // Resolved on the submitter: a preset must describe the games, not the machine that happened
    // to play them.
    let preset = ds
        .bot_config
        .as_deref()
        .map(load_mcts_config)
        .transpose()
        .map_err(|e| e.to_string())?;
    let base = GenerateConfig {
        mode: ds.mode.into(),
        search: SearchConfig {
            budget,
            workers: ds.mcts_workers,
            // `dataset` leaves these `None`, meaning "the bot's env-driven default"; keep that,
            // because `candidate_label`/the provenance label read these fields and a `Some` here
            // would change every corpus label.
            //
            // `pinned_to_env` below then fills `config` with the *whole* resolved `MctsConfig`
            // from this same environment, so "env-driven default" means the hub's environment,
            // once, and not whichever worker happened to pick the shard up.
            puct: None,
            horizon_turns: None,
            fpu_reduction: None,
            config: preset.as_ref().map(|p| p.config),
        }
        .pinned_to_env(),
        config_name: preset.as_ref().map(|p| p.name.clone()),
        exploration: Exploration::from_flags(
            ds.explore_noise,
            ds.explore_alpha,
            ds.explore_sample_moves,
            ds.explore_temperature,
        ),
        evaluator,
        model: ds.model.clone(),
        max_steps: ds.max_steps,
        lecture: ds.lecture.clone(),
        difficulty: ds.difficulty.into(),
        bias,
        board_sizes: ds.sizes.to_dist()?,
        next_drive: ds.next_drive,
    };
    if let Some(d) = &base.board_sizes {
        eprintln!(
            "[hub job] board sizes: {} -> {}",
            d.label,
            d.table()
                .iter()
                .map(|(b, p)| format!("{b} {:.1}%", p * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let heuristic = GenerateConfig {
        evaluator: Evaluator::Heuristic,
        model: None,
        ..base.clone()
    };
    let model_path = ds.model.as_ref().map(|m| abs(&PathBuf::from(m)));
    let mut shards = Vec::new();
    // `--out` has `dataset`'s default (`dataset.jsonl`) in the shared struct; a job writes it only
    // when it was typed, as before.
    if a.out_dir.is_none() && given(m, "out") {
        let out = PathBuf::from(&ds.out);
        let out = &out;
        shards.push(ShardReq {
            name: out
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "out".into()),
            out: abs(out),
            seed: ds.seed,
            games: ds.games,
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
                seed: ds.seed + k as u64 * a.shard_seed_stride,
                games: ds.games,
                cfg: if is_heur { heuristic.clone() } else { base.clone() },
                model_path: if is_heur { None } else { model_path.clone() },
            });
        }
    }
    let mc_label = stream_label_request(
        &a.mc,
        &shards,
        ds.model.as_deref(),
        a.out_dir.is_some(),
        a.client.label.as_deref(),
    )?;
    Ok(GenerateJobRequest {
        shards,
        truncate: ds.truncate,
        batch: a.batch,
        label: a.client.label.clone(),
        mc_label,
        resume: a.resume,
        priority: a.client.priority,
    })
}

/// Plan 069 item 4b: `--mc-label-*` as the request's streamed labelling, or `None`.
fn stream_label_request(
    a: &StreamLabelArgs,
    shards: &[ShardReq],
    model: Option<&str>,
    by_index: bool,
    label: Option<&str>,
) -> Result<Option<StreamLabelRequest>, String> {
    if a.mc_label_playouts == 0 {
        return Ok(None);
    }
    let model = model.ok_or("--mc-label-playouts labels with the generator's net: pass --model")?;
    let dir = a
        .mc_label_out_dir
        .as_ref()
        .ok_or("--mc-label-playouts needs --mc-label-out-dir")?;
    let names: Vec<String> = match &a.mc_label_shards {
        Some(ks) if by_index => parse_shards(ks)?.into_iter().map(|k| format!("shard{k}")).collect(),
        Some(_) => return Err("--mc-label-shards needs --out-dir/--shards".into()),
        None => shards.iter().map(|s| s.name.clone()).collect(),
    };
    let mut pairs = Vec::new();
    for n in names {
        let s = shards
            .iter()
            .find(|s| s.name == n)
            .ok_or_else(|| format!("--mc-label-shards: {n} is not generated by this job"))?;
        let file = s
            .out
            .file_name()
            .ok_or_else(|| format!("{}: no file name", s.out.display()))?;
        pairs.push((n, abs(&dir.join(file))));
    }
    Ok(Some(StreamLabelRequest {
        shards: pairs,
        model_path: abs(&PathBuf::from(model)),
        model: model.to_string(),
        cfg: botbowl_hub_proto::LabelConfig {
            playouts: a.mc_label_playouts,
            seed: a.mc_label_seed,
            max_steps: a.mc_label_max_steps,
            skip: 0,
        },
        chunk_samples: a.mc_label_chunk_samples.max(1),
        batch: a.mc_label_batch.max(1),
        label: label.map(|l| format!("{l} · mc-label (streamed)")),
    }))
}

/// The ladder half of `botbowl-ui eval`, flag for flag (the same [`EvalArgs`], flattened), so
/// `train_loop.sh` swaps the binary name and nothing else. `--out` and `--per-game-out` are
/// required. The process-local flags (`--parallel-games`, `--nn-server`, `--trials`,
/// `--skip-ladder`, `--trace-reuse`) are refused; `--skip-lectures` is accepted, since the hub
/// never runs the lecture battery.
#[derive(Args, Debug)]
struct EvalJobArgs {
    #[command(flatten)]
    client: ClientArgs,
    #[command(flatten)]
    ev: EvalArgs,
    /// Games per task handed to a worker.
    #[arg(long, default_value_t = 4)]
    batch: u16,
    /// Keep the games already in `--per-game-out` (folded into the report in file order) and
    /// play only the others. Without it the file is appended to as it is.
    #[arg(long, default_value_t = false)]
    resume: bool,
    /// Block until the job finishes; exit nonzero if it failed.
    #[arg(long, default_value_t = false)]
    wait: bool,
    #[command(flatten)]
    w: WaitArgs,
}

fn abs(p: &PathBuf) -> PathBuf {
    if p.is_absolute() {
        p.clone()
    } else {
        std::env::current_dir().expect("cwd").join(p)
    }
}

fn build_request(job: &EvalJobArgs, m: &ArgMatches) -> Result<EvalJobRequest, String> {
    refuse_local_flags(
        m,
        &[
            (
                "parallel_games",
                "workers size themselves (botbowl-worker --parallel-games)",
            ),
            ("nn_server", "each worker owns its sidecar (botbowl-worker --nn-server)"),
            (
                "nn_backend",
                "each worker picks its own backend (botbowl-worker --nn-backend)",
            ),
            (
                "nn_device",
                "each worker picks its own device (botbowl-worker --nn-device)",
            ),
            ("trials", "the hub never runs the lecture battery"),
            ("skip_ladder", "a hub eval job is the ladder"),
            (
                "trace_reuse",
                "not collected from workers; run `botbowl-ui eval --trace-reuse`",
            ),
        ],
    )?;
    let a = &job.ev;
    let report_out = a.out.as_deref().ok_or("--out PATH (report.json) is required")?;
    let per_game_out = a.per_game_out.as_deref().ok_or("--per-game-out PATH is required")?;
    let evaluator = Evaluator::from(a.evaluator);
    // Same rule as `botbowl-ui eval`: a preset replaces every per-knob field, and clap keeps the
    // two from being mixed. Unset `--vs-config` inherits the candidate's.
    let cand_preset = a
        .bot_config
        .as_deref()
        .map(load_mcts_config)
        .transpose()
        .map_err(|e| e.to_string())?;
    let opp_preset = match a.vs_config.as_deref() {
        Some(p) => Some(load_mcts_config(p).map_err(|e| e.to_string())?),
        None => cand_preset.clone(),
    };
    // Everything a search depends on is resolved here, from the hub's environment, and shipped.
    // A helper box's `BLOOD_MCTS_*` never reaches a job.
    let cand = a.candidate_search(cand_preset.as_ref())?.pinned_to_env();
    let opp = a.opponent_search(opp_preset.as_ref())?.pinned_to_env();
    let model_str = a.model.clone();
    let candidate = match CandidateBot::from(a.candidate_bot) {
        CandidateBot::Mcts => BotReq::Mcts {
            search: cand,
            evaluator,
            model: a.model.as_ref().map(|p| abs(&PathBuf::from(p))),
        },
        CandidateBot::Scripted => BotReq::Scripted,
        CandidateBot::Random => BotReq::Random,
    };
    // Plan 042: one rung set per board; `[None]` is the env board.
    let boards: Vec<Option<BoardDims>> = a.sizes.boards()?;
    // Plan 051: `--positions` replaces the boards with one drive rung per position set.
    let venues: Vec<(Option<BoardDims>, Option<DriveRung>)> = match a.positions.as_deref() {
        Some(list) => list
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|path| {
                let set = PositionSet::load(path)?;
                let mut rung = set.rung()?;
                rung.gate = a.drive_gate.is_some();
                Ok((Some(set.board_dims()?), Some(rung)))
            })
            .collect::<Result<_, String>>()
            .map_err(|e| format!("--positions: {e}"))?,
        None => boards.into_iter().map(|b| (b, None)).collect(),
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
            for (board, drives) in &venues {
                rungs.push(RungReq {
                    name: venue_name(name, *board, drives.as_ref()),
                    games: a.games,
                    opponent: opponent.clone(),
                    board: *board,
                    drives: drives.clone(),
                });
            }
        }
    }
    if let Some(vs) = a.vs_evaluator {
        let vs = Evaluator::from(vs);
        // Same label as `botbowl-ui eval` builds, so downstream scripts keep matching on it —
        // including the preset branch, where the per-knob fields are deliberately `None` and the
        // configuration *name* is the difference worth printing. Unwrapping them here used to
        // panic the submitter for every `--bot-config` + `--vs-evaluator` job.
        let base = evaluator_label(vs, a.vs_model.as_deref());
        let label = vs_rung_label(&base, cand_preset.as_ref(), opp_preset.as_ref(), &cand, &opp);
        for (board, drives) in &venues {
            rungs.push(RungReq {
                name: venue_name(&label, *board, drives.as_ref()),
                games: a.vs_games.unwrap_or(a.games),
                opponent: BotReq::Mcts {
                    search: opp,
                    evaluator: vs,
                    model: a.vs_model.as_ref().map(|p| abs(&PathBuf::from(p))),
                },
                board: *board,
                drives: drives.clone(),
            });
        }
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
            cand_preset.as_ref().map(|p| p.name.as_str()),
        ),
        candidate_config: cand_preset.as_ref().map(|p| p.name.clone()),
        opponent_config: opp_preset.as_ref().map(|p| p.name.clone()),
        mcts_iters: a.mcts_iters,
        rungs,
        seed: a.seed,
        max_steps: a.max_steps,
        per_game_out: abs(&PathBuf::from(per_game_out)),
        report_out: abs(&PathBuf::from(report_out)),
        batch: job.batch,
        sprt: a.sprt,
        drive_gate: a.drive_gate,
        label: job.client.label.clone(),
        resume: job.resume,
        priority: job.client.priority,
    })
}

/// The rung label, as `botbowl-ui eval` spells it: `opponent@board` for games, `opponent
/// drives(set)@board` for drives.
fn venue_name(opponent: &str, board: Option<BoardDims>, drives: Option<&DriveRung>) -> String {
    match (drives, board) {
        (Some(d), Some(b)) => drive_rung_name(opponent, &d.set, b),
        _ => rung_name(opponent, board),
    }
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

fn token_path(p: &Option<PathBuf>) -> PathBuf {
    p.clone().unwrap_or_else(botbowl_hub_proto::default_token_path)
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
    let written = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| write_private(path, &format!("{token}\n")));
    if let Err(e) = written {
        eprintln!("cannot write token file {}: {e}", path.display());
        std::process::exit(2)
    }
    eprintln!("[hub] new token written to {}", path.display());
    token
}

fn write_private(path: &PathBuf, contents: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    opts.open(path)?.write_all(contents.as_bytes())
}

fn print_report_lines(s: &JobStatus) {
    if let Some(r) = &s.report {
        println!("== report card: {} ==", r.candidate);
        for row in &r.ladder {
            println!("{}", row.report_line());
        }
    }
}

fn main() {
    // Parsed via `ArgMatches` so a job can tell a typed flag from a shared struct's default.
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    let job_matches = matches
        .subcommand_matches("job")
        .and_then(|j| j.subcommand())
        .map(|(_, m)| m.clone());
    match cli.command {
        Command::Serve(a) => {
            let token = or_create_token(&token_path(&a.token_file));
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(async move {
                // Say at startup what the allowlist does, if there is one: a file that has gone
                // stale (the usual case — someone committed after writing it) is the thing an
                // operator most needs told *before* a worker is turned away for it.
                if let Some(list) =
                    botbowl_hub::allowlist::Allowlist::load(&a.allowed_commits, botbowl_data::git_commit())
                {
                    eprintln!("[hub] {}", list.describe());
                }
                let play = play_router(&a.play);
                let (hub, addr, task) = Hub::start_with(
                    HubConfig {
                        bind: a.bind,
                        token,
                        allow_commit_mismatch: a.allow_commit_mismatch,
                        allowed_commits: a.allowed_commits.clone(),
                        worker_timeout: std::time::Duration::from_secs(a.worker_timeout),
                        run_dir: a.run_dir.clone(),
                        allow_from: a.play.allow_from.clone(),
                        rate_interval: std::time::Duration::from_secs(a.rate_interval_secs),
                        registry_dir: Some(a.registry_dir.clone().unwrap_or_else(|| repo_path("registry"))),
                    },
                    play,
                )
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
                let index_dirs = if a.model_index_dirs.is_empty() {
                    vec![repo_path("runs"), repo_path("models")]
                } else {
                    a.model_index_dirs.clone()
                };
                let indexing = hub.index_models(index_dirs.clone());
                std::thread::spawn(move || {
                    if let Ok(n) = indexing.join() {
                        eprintln!(
                            "[hub] {n} net(s) indexed under {}; workers' caches are named from them",
                            index_dirs
                                .iter()
                                .map(|d| d.display().to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                });
                // SIGTERM is how Slurm (and train_loop.sh's cleanup) stop the hub: exit between two
                // writes, never in one, so every shard ends on a whole line.
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
                tokio::select! {
                    _ = task => {}
                    _ = tokio::signal::ctrl_c() => hub.shutdown("SIGINT"),
                    _ = term.recv() => hub.shutdown("SIGTERM"),
                }
            });
        }
        Command::Status(a) => {
            let c = &a.client;
            let token = read_token(&token_path(&c.token_file));
            match request("GET", &format!("{}/api/status", c.hub), &token, None) {
                Ok((200, body)) => {
                    let s: HubStatus = serde_json::from_str(&body).expect("status json");
                    if a.text {
                        let port = c.hub.rsplit(':').next().and_then(|p| p.parse().ok()).unwrap_or(0);
                        let page = botbowl_hub::page::gather(s, port, a.run_dir.as_deref());
                        print!("{}", botbowl_hub::page::render(&page, std::time::SystemTime::now()));
                    } else {
                        println!("{}", serde_json::to_string_pretty(&s).unwrap());
                    }
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
            let (client, wait, wait_args, req, what) = match &job {
                JobCommand::Eval(a) => {
                    let req = build_request(a, job_matches.as_ref().expect("job matches")).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2)
                    });
                    let what = format!(
                        "eval job: {} rung(s), {} games",
                        req.rungs.len(),
                        req.rungs.iter().map(|r| r.games).sum::<u32>()
                    );
                    (a.client.clone(), a.wait, a.w.clone(), JobRequest::Eval(req), what)
                }
                JobCommand::Generate(a) => {
                    let req =
                        build_generate_request(a, job_matches.as_ref().expect("job matches")).unwrap_or_else(|e| {
                            eprintln!("{e}");
                            std::process::exit(2)
                        });
                    let what = format!(
                        "generate job: {} shard(s), {} games",
                        req.shards.len(),
                        req.shards.iter().map(|s| s.games).sum::<u32>()
                    );
                    (a.client.clone(), a.wait, a.w.clone(), JobRequest::Generate(req), what)
                }
                JobCommand::Label(a) => {
                    let req = build_label_request(a, job_matches.as_ref().expect("job matches")).unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(2)
                    });
                    let what = format!(
                        "label job: {} shard(s) x {} playouts per sample{}",
                        req.shards.len(),
                        req.cfg.playouts,
                        if req.top_up { " (top-up)" } else { "" }
                    );
                    (a.client.clone(), a.wait, a.w.clone(), JobRequest::Label(req), what)
                }
            };
            let token = read_token(&token_path(&client.token_file));
            run_job(&client, &token, wait, &wait_args, &req, &what);
        }
    }
}

/// Exit code for "stopped early, resumable": a deadline, or a hub gone too long (`EX_TEMPFAIL`,
/// the code `train_loop.sh` already exits with at a deadline).
const EXIT_RESUME: i32 = 75;

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Submit; `Err` with whether the hub was unreachable (worth retrying) or refused the job.
fn submit(client: &ClientArgs, token: &str, req: &JobRequest) -> Result<Submitted, (bool, String)> {
    let body = serde_json::to_string(req).unwrap();
    // A label job reads its shards before it answers, a resumed generate job too: give it time.
    match request_with_timeout(
        "POST",
        &format!("{}/api/jobs", client.hub),
        token,
        Some(&body),
        Duration::from_secs(600),
    ) {
        Ok((200, body)) => Ok(serde_json::from_str::<Submitted>(&body).expect("submit json")),
        Ok((code, body)) => Err((false, format!("hub refused the job ({code}): {body}"))),
        Err(e) => Err((true, format!("cannot reach hub at {}: {e}", client.hub))),
    }
}

/// Submit `req` and, with `wait`, follow it to the end.
///
/// Plan 069 item 5: the hub may restart under a waiting client (a Slurm job cut, a redeploy).
/// Polls that fail are retried; a hub that answers but no longer knows the job (404, or another
/// [`Submitted::instance`]) gets the request again when it is resumable ([`JobRequest::resumable`]:
/// `--resume`, or a label job) — the new hub reads the files and continues — and is an error
/// otherwise, as before. A client whose parent has gone (an orphan of a loop that died) stops
/// instead of resubmitting into whatever runs next. `--deadline` drains the job and exits 75.
fn run_job(client: &ClientArgs, token: &str, wait: bool, w: &WaitArgs, req: &JobRequest, what: &str) {
    let first = match submit(client, token, req) {
        Ok(s) => s,
        Err((_, e)) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    };
    let (mut id, mut instance) = (first.id, first.instance);
    eprintln!("[hub job] submitted {what} (job {id})");
    if !wait {
        println!("{id}");
        return;
    }
    #[cfg(unix)]
    let parent = std::os::unix::process::parent_id();
    let mut idle_polls: u64 = 0;
    let mut lost_since: Option<std::time::Instant> = None;
    let mut drained = false;
    // Plan 069 item 4b: a generate job with streamed labels is followed by its label job, `id`
    // then names that job. Its playouts, for the summary lines.
    let mut labels_phase = false;
    let label_playouts = match req {
        JobRequest::Generate(r) => r.mc_label.as_ref().map_or(0, |m| m.cfg.playouts),
        _ => 0,
    };
    loop {
        std::thread::sleep(Duration::from_secs(5));
        #[cfg(unix)]
        if std::os::unix::process::parent_id() != parent {
            eprintln!("[hub job] the process that started this client is gone; stopping (job {id} stays on the hub)");
            std::process::exit(1);
        }
        let now = unix_now();
        if let Some(deadline) = w.deadline {
            if !drained && now + w.drain_secs >= deadline {
                drained = true;
                match request("POST", &format!("{}/api/jobs/{id}/drain", client.hub), token, None) {
                    Ok((200, _)) => eprintln!(
                        "[hub job] deadline in {} s: job {id} drained (no new games; the ones on workers finish)",
                        deadline.saturating_sub(now)
                    ),
                    other => eprintln!("[hub job] deadline: drain of job {id} failed ({other:?})"),
                }
            }
            if now >= deadline {
                eprintln!("[hub job] deadline reached: stopping; submit again with --resume to continue job {id}");
                std::process::exit(EXIT_RESUME);
            }
        }
        let gone = match request("GET", &format!("{}/api/jobs/{id}", client.hub), token, None) {
            Err(e) => {
                let since = *lost_since.get_or_insert_with(std::time::Instant::now);
                let secs = since.elapsed().as_secs();
                if secs < 5 || secs % 60 < 5 {
                    eprintln!("[hub job] poll failed ({e}); retrying ({secs} s so far)");
                }
                if w.hub_lost_secs > 0 && secs >= w.hub_lost_secs {
                    eprintln!("[hub job] the hub has been unreachable for {secs} s: stopping (resumable)");
                    std::process::exit(EXIT_RESUME);
                }
                continue;
            }
            Ok((404, _)) => true,
            Ok((200, body)) => {
                let s: JobStatus = serde_json::from_str(&body).expect("job json");
                if s.hub_instance != instance && instance != 0 && s.hub_instance != 0 {
                    true
                } else {
                    lost_since = None;
                    match &s.state {
                        JobState::Running => {
                            if drained && s.in_flight == 0 {
                                print_progress(&s);
                                eprintln!(
                                    "[hub job] deadline: job {id} drained; submit again with --resume to continue"
                                );
                                std::process::exit(EXIT_RESUME);
                            }
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
                        JobState::Done if labels_phase => {
                            print_label_lines(&s, label_playouts);
                            return;
                        }
                        JobState::Done => {
                            match (&s.kind, req) {
                                (JobKind::Eval, JobRequest::Eval(r)) => {
                                    print_report_lines(&s);
                                    println!("wrote {}", r.report_out.display());
                                }
                                (JobKind::Label, JobRequest::Label(r)) => print_label_lines(&s, r.cfg.playouts),
                                _ => print_generate_lines(&s),
                            }
                            // Plan 069 item 4b: the streamed labels finish shortly after the last game.
                            let Some(lid) = s.labels_job else { return };
                            eprintln!("[hub job] waiting for the streamed labels (job {lid})");
                            (id, labels_phase, idle_polls) = (lid, true, 0);
                        }
                        // The generation is on disk whatever happened to its labels: a shard without
                        // an output is labelled by `job label` / `mc-label` (they skip written ones).
                        JobState::Failed { error } if labels_phase => {
                            eprintln!(
                                "[hub job] WARN: streamed labels (job {id}) failed: {error}; label the shards without an output with `job label`"
                            );
                            return;
                        }
                        JobState::Failed { error } => {
                            eprintln!("[hub job] job {id} failed: {error}");
                            std::process::exit(1);
                        }
                    }
                    false
                }
            }
            Ok((code, body)) => {
                eprintln!("[hub job] hub returned {code}: {body}");
                std::process::exit(1);
            }
        };
        if !gone {
            continue;
        }
        // The hub restarted: job `id` is not there any more.
        lost_since = None;
        // Plan 069 item 4b: the labels' state died with the hub, the corpus did not. Resubmitting a
        // resumable generate request rebuilds it from the shards on disk: every game is there, so
        // the new job only labels (the shards whose labelled output is not written yet).
        if labels_phase && !req.resumable() {
            eprintln!(
                "[hub job] WARN: the hub restarted and the streamed labels (job {id}) are gone; label the shards without an output with `job label`"
            );
            return;
        }
        labels_phase = false;
        if !req.resumable() {
            eprintln!("[hub job] the hub restarted and job {id} is gone; it is not resumable (submit with --resume)");
            std::process::exit(1);
        }
        if drained {
            eprintln!("[hub job] the hub restarted during the drain before the deadline: stopping (resumable)");
            std::process::exit(EXIT_RESUME);
        }
        match submit(client, token, req) {
            Ok(s) => {
                eprintln!(
                    "[hub job] the hub restarted; submitted again with what is on disk (job {} was {id})",
                    s.id
                );
                (id, instance) = (s.id, s.instance);
            }
            Err((true, e)) => eprintln!("[hub job] the hub restarted; resubmit failed ({e}); retrying"),
            Err((false, e)) => {
                eprintln!("[hub job] the hub restarted; {e}");
                std::process::exit(1);
            }
        }
    }
}

/// Where a job stands, on stdout, when the client leaves it before the end (a deadline).
fn print_progress(s: &JobStatus) {
    for u in &s.units {
        println!("  {:24} {:>5}/{:<5} done", u.name, u.done, u.total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(name: &str) -> String {
        format!("{}/../cfgs/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    /// Parse a `job …` command line the way `main` does, keeping the sub-matches.
    fn job(args: &[&str]) -> Result<(JobCommand, ArgMatches), String> {
        let argv: Vec<&str> = ["botbowl-hub", "job"].iter().chain(args).copied().collect();
        let matches = Cli::command().try_get_matches_from(argv).map_err(|e| e.to_string())?;
        let cli = Cli::from_arg_matches(&matches).map_err(|e| e.to_string())?;
        let sub = matches
            .subcommand_matches("job")
            .unwrap()
            .subcommand()
            .unwrap()
            .1
            .clone();
        match cli.command {
            Command::Job { job } => Ok((job, sub)),
            _ => unreachable!(),
        }
    }

    fn generate(args: &[&str]) -> Result<GenerateJobRequest, String> {
        match job(args)? {
            (JobCommand::Generate(a), m) => build_generate_request(&a, &m),
            _ => unreachable!(),
        }
    }

    fn eval(args: &[&str]) -> Result<EvalJobRequest, String> {
        match job(args)? {
            (JobCommand::Eval(a), m) => build_request(&a, &m),
            _ => unreachable!(),
        }
    }

    /// `train_loop.sh`'s `generate_jobs`, flag for flag (plan 058 values).
    #[test]
    fn the_loops_generate_command_still_parses() {
        let gen_cfg = cfg("gumbel16_f1000_gen.toml");
        let req = generate(&[
            "generate",
            "--hub",
            "http://127.0.0.1:13337",
            "--token-file",
            "/tmp/t",
            "--mode",
            "random-start",
            "--games",
            "300",
            "--seed-base",
            "5000000",
            "--shard-seed-stride",
            "100000",
            "--mcts-iters",
            "1000",
            "--evaluator",
            "nn",
            "--model",
            "m.onnx",
            "--size-centre",
            "144",
            "--size-temperature",
            "0.3",
            "--size-floor",
            "0.2",
            "--size-max-area",
            "144",
            "--size-min-area",
            "70",
            "--next-drive",
            "--bot-config",
            &gen_cfg,
            "--shards",
            "0 1 2 3 4 5 6 7",
            "--heuristic-shards",
            "",
            "--label",
            "gen01 generate",
            "--truncate",
            "--out-dir",
            "/tmp/gen01",
            "--wait",
        ])
        .unwrap();
        assert_eq!(req.shards.len(), 8);
        assert_eq!(req.shards[3].seed, 5_000_000 + 3 * 100_000);
        assert_eq!(req.shards[0].games, 300);
        assert!(req.truncate);
        assert_eq!(
            req.shards[0].cfg.bias,
            botbowl_play::generate::RandomStartBias::default()
        );
        assert!(req.shards[0].cfg.next_drive);
        assert_eq!(req.shards[0].cfg.config_name.as_deref(), Some("gumbel16_f1000_gen"));
        assert!(req.mc_label.is_none(), "no streamed labels unless asked");
    }

    /// Plan 069 item 4b: `train_loop.sh`'s generate command under `LABEL_STREAMING=1`.
    #[test]
    fn the_loops_streamed_label_flags_parse() {
        let base = [
            "generate",
            "--mode",
            "random-start",
            "--games",
            "3",
            "--evaluator",
            "nn",
            "--model",
            "m.onnx",
            "--shards",
            "0 1 2 3 4 5 6 7",
            "--out-dir",
            "/tmp/gen03",
        ];
        let flags = [
            "--mc-label-playouts",
            "8",
            "--mc-label-seed",
            "56003",
            "--mc-label-out-dir",
            "/tmp/gen03/mc",
            "--mc-label-chunk-samples",
            "32",
            "--mc-label-shards",
            "0,1,2,3,4,5,6",
        ];
        let req = generate(&[&base[..], &flags[..]].concat()).unwrap();
        let m = req.mc_label.expect("streamed labels");
        assert_eq!(m.shards.len(), 7);
        assert_eq!(
            m.shards[4],
            ("shard4".to_string(), PathBuf::from("/tmp/gen03/mc/shard4.jsonl"))
        );
        assert_eq!(
            (m.cfg.playouts, m.cfg.seed, m.cfg.max_steps, m.cfg.skip),
            (8, 56003, 100_000, 0)
        );
        assert_eq!(m.model, "m.onnx");
        assert_eq!(m.chunk_samples, 32);
        // Every shard by default; a shard the job does not write, or no out dir, is refused.
        let all = generate(&[&base[..], &flags[..8]].concat()).unwrap();
        assert_eq!(all.mc_label.unwrap().shards.len(), 8);
        let bad = generate(&[&base[..], &flags[..8], &["--mc-label-shards", "9"]].concat());
        assert!(bad.is_err());
        assert!(generate(&[&base[..], &flags[..4]].concat()).is_err());
    }

    /// Plan 071 R5: `--priority` defaults to foreground, and a foreground request's JSON (its
    /// attach key) carries no priority field, so it is what it was before the flag existed.
    #[test]
    fn priority_defaults_to_foreground_and_stays_off_its_json() {
        use botbowl_hub::api::Priority;
        let fg = generate(&["generate", "--out-dir", "/tmp/g"]).unwrap();
        assert_eq!(fg.priority, Priority::Foreground);
        assert!(!serde_json::to_string(&fg).unwrap().contains("priority"));
        let bg = generate(&["generate", "--out-dir", "/tmp/g", "--priority", "background"]).unwrap();
        assert_eq!(bg.priority, Priority::Background);
        let json = serde_json::to_string(&bg).unwrap();
        assert!(json.contains(r#""priority":"background""#), "{json}");
        let back: botbowl_hub::api::GenerateJobRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.priority, Priority::Background);
        assert!(generate(&["generate", "--out-dir", "/tmp/g", "--priority", "urgent"]).is_err());
    }

    #[test]
    fn resume_excludes_truncate_and_the_deadline_flags_parse() {
        let req = generate(&[
            "generate",
            "--out-dir",
            "/tmp/g",
            "--resume",
            "--wait",
            "--deadline",
            "1800000000",
        ])
        .unwrap();
        assert!(req.resume && !req.truncate);
        assert!(JobRequest::Generate(req).resumable());
        let e = generate(&["generate", "--out-dir", "/tmp/g", "--resume", "--truncate"]).unwrap_err();
        assert!(e.contains("--truncate") || e.contains("cannot be used"), "{e}");
        let (j, _) = job(&[
            "generate",
            "--out-dir",
            "/tmp/g",
            "--deadline",
            "7",
            "--drain-secs",
            "60",
            "--hub-lost-secs",
            "600",
        ])
        .unwrap();
        let JobCommand::Generate(a) = j else { unreachable!() };
        assert_eq!((a.w.deadline, a.w.drain_secs, a.w.hub_lost_secs), (Some(7), 60, 600));
        let plain = generate(&["generate", "--out-dir", "/tmp/g", "--truncate"]).unwrap();
        assert!(
            !JobRequest::Generate(plain).resumable(),
            "a truncating job must never be resubmitted"
        );
    }

    #[test]
    fn a_generate_job_needs_an_out_dir_or_a_typed_out() {
        assert!(generate(&["generate"]).unwrap_err().contains("--out-dir"));
        let req = generate(&["generate", "--out", "/tmp/one.jsonl", "--seed", "7"]).unwrap();
        assert_eq!(req.shards.len(), 1);
        assert_eq!(req.shards[0].seed, 7);
    }

    #[test]
    fn process_local_flags_are_refused_not_ignored() {
        let e = generate(&["generate", "--out-dir", "/tmp/x", "--parallel-games", "8"]).unwrap_err();
        assert!(e.contains("--parallel-games"), "{e}");
        let e = generate(&["generate", "--out-dir", "/tmp/x", "--nn-server", "/tmp/s"]).unwrap_err();
        assert!(e.contains("--nn-server"), "{e}");
        for flag in [
            &["--trials", "5"][..],
            &["--parallel-games", "2"],
            &["--nn-server", "/s"],
            &["--skip-ladder"],
        ] {
            let mut args = vec!["eval", "--out", "/tmp/r.json", "--per-game-out", "/tmp/g.jsonl"];
            args.extend_from_slice(flag);
            let e = eval(&args).unwrap_err();
            assert!(e.contains(flag[0]), "{e}");
        }
        // The hub never runs lectures, so saying so is fine.
        eval(&[
            "eval",
            "--out",
            "/tmp/r.json",
            "--per-game-out",
            "/tmp/g.jsonl",
            "--skip-lectures",
        ])
        .unwrap();
    }

    fn label(args: &[&str]) -> Result<LabelJobRequest, String> {
        match job(args)? {
            (JobCommand::Label(a), m) => build_label_request(&a, &m),
            _ => unreachable!(),
        }
    }

    /// `train_loop.sh`'s mc-label phase under `LABEL_VIA_HUB=1`, flag for flag.
    #[test]
    fn the_loops_label_command_parses() {
        let req = label(&[
            "label",
            "--hub",
            "http://127.0.0.1:13337",
            "--token-file",
            "/tmp/t",
            "--label",
            "gen16 mc-label",
            "--corpus",
            "/r/gen16/shard0.jsonl",
            "/r/gen16/shard4.jsonl",
            "--model",
            "/r/models/bbnet_gen16.onnx",
            "--playouts",
            "8",
            "--seed",
            "56016",
            "--out-dir",
            "/r/gen16/mc",
            "--wait",
        ])
        .unwrap();
        assert_eq!(req.shards.len(), 2);
        assert_eq!(req.shards[1].name, "shard4");
        assert_eq!(req.shards[1].input, PathBuf::from("/r/gen16/shard4.jsonl"));
        assert_eq!(req.shards[1].out, PathBuf::from("/r/gen16/mc/shard4.jsonl"));
        assert_eq!(req.model, "/r/models/bbnet_gen16.onnx");
        assert_eq!((req.cfg.playouts, req.cfg.seed, req.cfg.max_steps), (8, 56016, 100_000));
        assert_eq!((req.chunk_samples, req.batch), (32, 1));
        assert_eq!(req.label.as_deref(), Some("gen16 mc-label"));
        // `--in` is `--corpus`'s other spelling; the process-local flags are refused.
        let req = label(&["label", "--in", "a.jsonl", "--model", "m.onnx", "--out-dir", "o"]).unwrap();
        assert_eq!(req.shards[0].name, "a");
        for flag in [&["--parallel", "16"][..], &["--nn-server", "/tmp/s"]] {
            let mut args = vec!["label", "--in", "a.jsonl", "--model", "m.onnx", "--out-dir", "o"];
            args.extend_from_slice(flag);
            let e = label(&args).unwrap_err();
            assert!(e.contains(flag[0]), "{e}");
        }
        let e = label(&[
            "label",
            "--in",
            "a.jsonl",
            "--model",
            "m",
            "--out-dir",
            "o",
            "--playouts",
            "0",
        ])
        .unwrap_err();
        assert!(e.contains("--playouts"), "{e}");
        assert!(!req.top_up);
        // Plan 067: `--top-up K` plays K more per sample; the seed and the prior count come from the
        // shards' labels, so `--seed` and `--playouts` are refused alongside it.
        let base = ["label", "--in", "mc/a.jsonl", "--model", "m.onnx", "--out-dir", "o"];
        let req = label(&[&base[..], &["--top-up", "8"]].concat()).unwrap();
        assert!(req.top_up);
        assert_eq!((req.cfg.playouts, req.cfg.skip), (8, 0));
        for flag in [&["--seed", "56044"][..], &["--playouts", "8"]] {
            assert!(
                job(&[&base[..], &["--top-up", "8"], flag].concat()).is_err(),
                "{flag:?}"
            );
        }
        assert!(label(&[&base[..], &["--top-up", "0"]].concat()).is_err());
    }

    /// `train_loop.sh`'s `eval_job` for the drives rung, minus `--positions` (it loads files).
    #[test]
    fn the_loops_eval_command_still_parses() {
        let eval_cfg = cfg("gumbel16_f1000.toml");
        let req = eval(&[
            "eval",
            "--hub",
            "http://127.0.0.1:13337",
            "--token-file",
            "/tmp/t",
            "--label",
            "gen01 drives",
            "--evaluator",
            "nn",
            "--mcts-iters",
            "1000",
            "--games",
            "30",
            "--bot-config",
            &eval_cfg,
            "--vs-config",
            &eval_cfg,
            "--model",
            "m.onnx",
            "--seed",
            "0",
            "--skip-fixed-rungs",
            "--sprt",
            "0.5:0.55",
            "--vs-games",
            "800",
            "--vs-evaluator",
            "nn",
            "--vs-model",
            "ref.onnx",
            "--per-game-out",
            "/tmp/g.jsonl",
            "--out",
            "/tmp/r.json",
            "--wait",
        ])
        .unwrap();
        assert_eq!(req.rungs.len(), 1);
        assert_eq!(req.rungs[0].name, "vs:mcts(nn:ref.onnx) [gumbel16_f1000]");
        assert_eq!(req.rungs[0].games, 800);
        assert!(eval(&["eval", "--per-game-out", "/tmp/g.jsonl"])
            .unwrap_err()
            .contains("--out"));
    }
}
