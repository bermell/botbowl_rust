//! `botbowl-ui positions`: write the candidate positions of a drive-rung set (plan 051 step 4).
//!
//! The file is a recipe: board, placement bias and seeds. Anyone at the same commit regenerates
//! each position with `botbowl_play::drives::position_state`. Screening for contested positions
//! is a separate step: a drive rung of the reference bot against itself over these positions,
//! then `scripts/positions_screen.py` on its per-game lines.

use std::io;

use botbowl_play::board_sizes::{board_label, parse_board};
use botbowl_play::drives::{position_state, turns_left, PositionSet};
use botbowl_play::generate::RandomStartBias;

use crate::cli::PositionsArgs;

pub fn run(args: PositionsArgs) -> io::Result<()> {
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidInput, e);
    let board = parse_board(&args.board, args.cells_per_player).map_err(invalid)?;
    let bias = RandomStartBias::default();
    let mut seeds = Vec::with_capacity(args.count as usize);
    let mut seed = args.seed_base;
    let mut skipped = 0u64;
    while seeds.len() < args.count as usize {
        if turns_left(&position_state(&bias, board, seed)) >= args.min_turns_left {
            seeds.push(seed);
        } else {
            skipped += 1;
        }
        seed += 1;
    }
    let name = args.name.clone().unwrap_or_else(|| {
        std::path::Path::new(&args.out)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "positions".into())
    });
    let set = PositionSet {
        name,
        commit: botbowl_data::git_commit().to_string(),
        board: board_label(board),
        bias,
        seeds,
        screen: None,
    };
    std::fs::write(&args.out, serde_json::to_string_pretty(&set)?)?;
    eprintln!(
        "wrote {} positions on {} to {} ({skipped} skipped with fewer than {} turns left)",
        set.seeds.len(),
        set.board,
        args.out,
        args.min_turns_left
    );
    Ok(())
}
