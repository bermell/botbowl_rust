//! Plan 065: a stored `GameState` loads in a binary of another compiled capacity.
//!
//! `FullPitch` and the two roster arrays are sized by the build-time capacity (`WIDTH`/`HEIGHT`,
//! `ROSTER_PER_TEAM`), so a corpus written by a 16x9-capacity binary used to be unreadable by a
//! 20x11-capacity one ("invalid length 14, expected an array of length 16"): moving the training
//! loop to a bigger capacity would have cut every older corpus out of the window, the value
//! benchmark included. The active board sits in the low corner of those arrays and everything past
//! it is permanently empty, so a capacity change only adds or drops empty cells. In a
//! human-readable format (the JSONL corpora) the arrays now load at any length: a shorter one is
//! padded with empty cells, a longer one may only carry empty cells past the capacity. The output
//! of `Serialize` is unchanged, and the non-self-describing wire (postcard, hub <-> worker, which
//! refuses a capacity mismatch anyway) keeps the fixed-length layout.
//!
//! One binary cannot hold two capacities, so these tests rewrite the JSON a state serialises to:
//! cut the arrays down to the board and roster a smaller capacity would have had, or pad them out
//! to a bigger one.

use botbowl_engine::core::gamestate::{GameState, GameStateBuilder};
use botbowl_engine::core::model::BoardDims;
use serde_json::Value;

/// Engine dims (border included) of the small board every test plays on: 10x5 playable, 3 a side.
/// Fits every capacity the workspace is built at.
const W: usize = 12;
const H: usize = 7;
const PLAYERS: usize = 3;

fn small_state() -> GameState {
    GameStateBuilder::new()
        .with_board_dims(BoardDims::new(W as i8, H as i8, PLAYERS))
        .add_home_players(&[(3, 3), (5, 2)])
        .add_away_players(&[(6, 3), (7, 4)])
        .build()
}

fn json(state: &GameState) -> Value {
    serde_json::to_value(state).expect("a state serialises")
}

/// The `data` matrix of a serialised `FullPitch` (`{"data": [[cell; HEIGHT]; WIDTH]}`).
fn pitch_mut(v: &mut Value) -> &mut Vec<Value> {
    v.get_mut("data")
        .and_then(Value::as_array_mut)
        .expect("FullPitch serialises as {data: [[..]]}")
}

/// Cut a serialised pitch down to `w` columns of `h` cells, asserting that only empty cells go.
fn shrink_pitch(v: &mut Value, w: usize, h: usize, empty: &Value) {
    let cols = pitch_mut(v);
    for col in cols.iter().skip(w) {
        assert!(
            col.as_array().unwrap().iter().all(|c| c == empty),
            "a cell past the board is set"
        );
    }
    cols.truncate(w);
    for col in cols.iter_mut() {
        let col = col.as_array_mut().unwrap();
        assert!(col.iter().skip(h).all(|c| c == empty), "a cell past the board is set");
        col.truncate(h);
    }
}

/// Pad a serialised pitch with `dw` empty columns and `dh` empty cells per column.
fn grow_pitch(v: &mut Value, dw: usize, dh: usize, empty: &Value) {
    let cols = pitch_mut(v);
    for col in cols.iter_mut() {
        col.as_array_mut()
            .unwrap()
            .extend(std::iter::repeat(empty.clone()).take(dh));
    }
    let h = cols[0].as_array().unwrap().len();
    for _ in 0..dw {
        cols.push(Value::Array(vec![empty.clone(); h]));
    }
}

fn roster_mut<'a>(v: &'a mut Value, field: &str) -> &'a mut Vec<Value> {
    v.get_mut(field).and_then(Value::as_array_mut).expect("roster array")
}

/// The JSON a binary of capacity `W x H / PLAYERS` (the board itself) would have written.
fn as_smallest_capacity(state: &GameState) -> Value {
    let mut v = json(state);
    shrink_pitch(&mut v["board"], W, H, &Value::Null);
    let positional = &mut v["available_actions"]["positional"];
    assert!(!positional.is_null(), "the test state must offer positional actions");
    shrink_pitch(positional, W, H, &Value::Array(vec![]));
    let roster_slots = 2 * (PLAYERS + 1);
    for field in ["fielded_players", "dugout_players"] {
        let arr = roster_mut(&mut v, field);
        assert!(
            arr.iter().skip(roster_slots).all(Value::is_null),
            "{field}: a slot past the roster is set"
        );
        arr.truncate(roster_slots);
    }
    v
}

/// The JSON a binary of a capacity bigger than this one's would have written.
fn as_bigger_capacity(state: &GameState) -> Value {
    let mut v = json(state);
    grow_pitch(&mut v["board"], 4, 2, &Value::Null);
    grow_pitch(&mut v["available_actions"]["positional"], 4, 2, &Value::Array(vec![]));
    for field in ["fielded_players", "dugout_players"] {
        roster_mut(&mut v, field).extend(std::iter::repeat(Value::Null).take(6));
    }
    v
}

#[test]
fn a_state_written_at_a_smaller_capacity_loads_and_reserialises_at_this_one() {
    let state = small_state();
    let small = as_smallest_capacity(&state);
    let back: GameState = serde_json::from_value(small).expect("a smaller capacity's state loads");
    assert_eq!(back, state);
    assert_eq!(
        json(&back),
        json(&state),
        "the padded state serialises at this binary's capacity"
    );
}

#[test]
fn a_state_written_at_a_bigger_capacity_loads_when_the_extra_cells_are_empty() {
    let state = small_state();
    let back: GameState = serde_json::from_value(as_bigger_capacity(&state)).expect("a bigger capacity's state loads");
    assert_eq!(back, state);
    assert_eq!(json(&back), json(&state));
}

#[test]
fn a_player_past_this_capacity_is_an_error_not_a_silent_drop() {
    let state = small_state();
    // A board cell past the capacity holding a player.
    let mut v = as_bigger_capacity(&state);
    let cols = pitch_mut(&mut v["board"]);
    let last = cols.len() - 1;
    cols[last].as_array_mut().unwrap()[0] = Value::from(0);
    let err = serde_json::from_value::<GameState>(v).expect_err("a player past the capacity must not load");
    assert!(err.to_string().contains("capacity"), "{err}");

    // A roster slot past the capacity holding a player.
    let mut v = as_bigger_capacity(&state);
    let arr = roster_mut(&mut v, "fielded_players");
    let player = arr[0].clone();
    assert!(!player.is_null());
    *arr.last_mut().unwrap() = player;
    let err = serde_json::from_value::<GameState>(v).expect_err("a roster slot past the capacity must not load");
    assert!(err.to_string().contains("capacity"), "{err}");
}

#[test]
fn a_same_capacity_round_trip_is_unchanged() {
    let state = small_state();
    let s = serde_json::to_string(&state).unwrap();
    let back: GameState = serde_json::from_str(&s).unwrap();
    assert_eq!(back, state);
    assert_eq!(serde_json::to_string(&back).unwrap(), s);
}
