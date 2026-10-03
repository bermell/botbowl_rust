mod convergence;
mod curriculum;
mod dataset;
mod eval;
mod live;
mod placement;
mod positions;
mod replay;

use clap::Parser;
use std::io;

use botbowl_ui::cli;

/// As in `botbowl-worker`: `dataset` and `eval` play many MCTS games per process, and glibc
/// malloc keeps the freed search trees in per-thread arenas instead of returning them.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> io::Result<()> {
    let cli = cli::Cli::parse();
    match cli.command {
        cli::Command::Live(args) => live::run(args),
        cli::Command::Replay(args) => replay::run(args),
        cli::Command::Snapshot(args) => botbowl_ui::snapshot::run(args),
        cli::Command::Curriculum(args) => curriculum::run(args),
        cli::Command::Dataset(args) => dataset::run(args),
        cli::Command::Eval(args) => eval::run(args),
        cli::Command::Placement(args) => placement::run(args),
        cli::Command::Convergence(args) => convergence::run(args),
        cli::Command::Positions(args) => positions::run(args),
    }
}
