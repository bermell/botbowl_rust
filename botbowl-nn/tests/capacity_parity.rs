//! Plan 034 phase 0, decision 8: **one** web-server binary, built at the
//! default 26x15 capacity, must play a 14x7 game with byte-identical network
//! inputs to a purpose-built 14x7 binary — otherwise the lobby cannot offer a
//! board size per game and the trained 14x7 models are unusable from it.
//!
//! Everything capacity-dependent in the NN path is on this side of the ONNX
//! boundary: `encode` reads `state.board_dims` for `H`/`W` and indexes a
//! `Position` straight into the tensor, and `actions::action_cell` mirrors `x`
//! with `dims.width`. Downstream, `NnEvaluator` concretises its runnable from
//! the `(h, w)` `encode` reports and gathers logits at the cells
//! `action_cell` reports, so **encoder + action-cell parity implies forward
//! parity**. This test pins both against golden bytes captured under a
//! `BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4` build.
//!
//! Regenerate the goldens (only when the encoder schema deliberately changes,
//! which is a `NN_SCHEMA_VERSION` bump) with:
//!
//! ```sh
//! BLOOD_WRITE_GOLDEN=1 BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4 \
//!   CARGO_TARGET_DIR=target/14x7 cargo test -p botbowl-nn --test capacity_parity
//! ```
//!
//! and then verify at the default capacity with a plain
//! `cargo test -p botbowl-nn --test capacity_parity`.

use std::path::PathBuf;

use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action, BoardDims, HEIGHT, TEAM_SIZE, WIDTH};
use botbowl_engine::core::table::SimpleAT;
use botbowl_nn::actions::action_cell;
use botbowl_nn::encode::encode;
use botbowl_nn::perspective::mover_for;

/// The plan's first target board: 14x7 playable → 16x9 engine, 4 a side.
const DIMS: (i8, i8, usize) = (16, 9, 4);

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures")).join(name)
}

fn capacity_fits() -> bool {
    let (w, h, team_size) = DIMS;
    WIDTH as i8 >= w && HEIGHT as i8 >= h && TEAM_SIZE >= team_size
}

/// A hand-placed 14x7 position. Every player and the ball are placed
/// explicitly (`build()` clears the fast-forwarded kickoff lineup first), so
/// the encoded tensor is a function of the runtime dims alone.
fn small_state() -> GameState {
    let (w, h, team_size) = DIMS;
    let mut state = GameStateBuilder::new()
        .with_board_dims(BoardDims::new(w, h, team_size))
        // Home defends x = width-2 and attacks x = 1, so its own half is high x.
        .add_home_players(&[(8, 3), (8, 4), (9, 5), (10, 2)])
        .add_away_players(&[(7, 3), (7, 4), (6, 5), (5, 2)])
        .add_ball((8, 5))
        .build();
    state.set_logging_state(false);
    state
}

/// Same position with the *other* team to move, so the `Away` x-mirror in
/// `perspective::canonical_x` is covered too — a `WIDTH_` in place of
/// `dims.width` there would only show up on this half of the fixture.
fn small_state_away_to_move() -> GameState {
    let mut state = small_state();
    // FixedDice would panic on any roll the turn change happens to request;
    // a seeded RNG keeps it deterministic without pinning a dice script.
    state.set_dice_mode(DiceMode::RollDice);
    state.set_seed(7);
    state.step(Action::Simple(SimpleAT::EndTurn)).unwrap();
    state
}

/// `spatial` then `global`, little-endian f32 — the exact bytes a `.npy` batch
/// row would carry.
fn tensor_bytes(state: &GameState) -> Vec<u8> {
    let e = encode(state);
    let mut out = Vec::with_capacity((e.spatial.len() + e.global.len() + 2) * 4);
    out.extend_from_slice(&(e.h as u32).to_le_bytes());
    out.extend_from_slice(&(e.w as u32).to_le_bytes());
    for v in e.spatial.iter().chain(e.global.iter()) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// One line per legal action: the action and the policy cell it gathers from.
fn action_cell_table(state: &GameState) -> String {
    let mover = mover_for(state);
    let dims = state.board_dims;
    let mut lines: Vec<String> = state
        .get_all_actions()
        .into_iter()
        .map(|a| {
            let c = action_cell(a, mover, dims);
            format!("{a:?} -> c={} y={} x={} simple={}", c.channel, c.y, c.x, c.is_simple)
        })
        .collect();
    // `get_all_actions` is already sorted, but pin it so a HashSet iteration
    // order change cannot masquerade as a capacity difference.
    lines.sort();
    format!("mover={mover:?} dims={dims:?}\n{}\n", lines.join("\n"))
}

fn writing_goldens() -> bool {
    std::env::var("BLOOD_WRITE_GOLDEN").is_ok_and(|v| v == "1")
}

fn check_bytes(name: &str, actual: &[u8]) {
    let path = fixture(name);
    if writing_goldens() {
        std::fs::write(&path, actual).unwrap();
        eprintln!("wrote golden {} ({} bytes)", path.display(), actual.len());
        return;
    }
    let expected = std::fs::read(&path).unwrap_or_else(|e| panic!("missing golden {}: {e}", path.display()));
    assert_eq!(
        expected.len(),
        actual.len(),
        "{name}: tensor length changed ({} -> {}) — a capacity leak would change H/W, \
         a schema change would change C or F",
        expected.len() / 4,
        actual.len() / 4,
    );
    if let Some(i) = (0..expected.len()).find(|&i| expected[i] != actual[i]) {
        let word = i / 4;
        panic!(
            "{name}: first difference at f32 word {word} (byte {i}); \
             words 0..2 are H,W — expected H,W = {},{}, got {},{}",
            u32::from_le_bytes(expected[0..4].try_into().unwrap()),
            u32::from_le_bytes(expected[4..8].try_into().unwrap()),
            u32::from_le_bytes(actual[0..4].try_into().unwrap()),
            u32::from_le_bytes(actual[4..8].try_into().unwrap()),
        );
    }
}

fn check_text(name: &str, actual: &str) {
    let path = fixture(name);
    if writing_goldens() {
        std::fs::write(&path, actual).unwrap();
        eprintln!("wrote golden {}", path.display());
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("missing golden {}: {e}", path.display()));
    assert_eq!(expected, actual, "{name}: action → policy-cell map differs");
}

#[test]
fn encoding_a_14x7_game_is_independent_of_the_compiled_capacity() {
    if !capacity_fits() {
        eprintln!("skipped: compiled capacity {WIDTH}x{HEIGHT}/{TEAM_SIZE} is smaller than {DIMS:?}");
        return;
    }
    let home = small_state();
    let away = small_state_away_to_move();

    // Sanity: the fixture really is a 14x7 board and really covers both movers.
    let e = encode(&home);
    assert_eq!((e.h, e.w), (9, 16), "fixture board is not 16x9");
    assert_eq!(mover_for(&home), botbowl_engine::core::model::TeamType::Home);
    assert_eq!(mover_for(&away), botbowl_engine::core::model::TeamType::Away);

    check_bytes("capacity_encode_14x7_home.bin", &tensor_bytes(&home));
    check_bytes("capacity_encode_14x7_away.bin", &tensor_bytes(&away));
    check_text("capacity_action_cells_14x7_home.txt", &action_cell_table(&home));
    check_text("capacity_action_cells_14x7_away.txt", &action_cell_table(&away));
}

/// End-to-end confirmation with a real net. `models/` is gitignored so this
/// cannot carry a committed golden — it prints the numbers instead, and the
/// check is to run it under both capacities and diff the output:
///
/// ```sh
/// cargo test -p botbowl-nn --test capacity_parity -- --ignored --nocapture > /tmp/cap26.txt
/// BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4 CARGO_TARGET_DIR=target/14x7 \
///   cargo test -p botbowl-nn --test capacity_parity -- --ignored --nocapture > /tmp/cap14.txt
/// diff /tmp/cap26.txt /tmp/cap14.txt
/// ```
#[test]
#[ignore]
fn nn_forward_on_a_14x7_model_is_capacity_independent() {
    if !capacity_fits() {
        eprintln!("skipped: capacity too small");
        return;
    }
    let model = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/bbnet_14x7_gen0c.onnx"));
    if !model.exists() {
        eprintln!("skipped: {} not present", model.display());
        return;
    }
    let nn = botbowl_nn::eval::NnEvaluator::from_path(&model).unwrap();
    for (label, state) in [("home", small_state()), ("away", small_state_away_to_move())] {
        let actions = state.get_all_actions();
        let priors = nn.priors(&state, &actions);
        println!("{label}: value_home_i64 = {}", nn.value_home_i64(&state));
        for (a, p) in actions.iter().zip(priors.iter()) {
            println!("{label}: prior {a:?} = {p:.9}");
        }
    }
}
