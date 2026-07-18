mod curriculum;
mod dataset;
mod eval;
mod live;
mod placement;
mod replay;

use clap::Parser;
use std::io;

use botbowl_ui::cli;

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
    }
}
