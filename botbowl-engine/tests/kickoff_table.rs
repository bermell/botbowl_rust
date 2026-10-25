//! Plan 065: the kickoff table on the 7-player boards a 20x11/7 loop draws.
//!
//! `BoardDims::kickoff_table_enabled` is `team_size >= 7`, so no board the loop played up to 16x9/6
//! ever rolled it: every kickoff there is the degenerate "the ball just lands". At 20x11/7 capacity
//! the density rule (26 cells a player) gives 7 players to 18x10, 20x9, 18x11, 20x10 and 20x11, and
//! every kickoff on them rolls 2D6 on the table, and every S3 result is implemented
//! (`procedures/kickoff_events.rs`), several with decisions for the coaches.
//!
//! Each test plays whole games through `DiceMode::RegisterRolls` with the caller rolling every die,
//! as `register_rolls_full_game.rs` does, but forces the table and the weather roll through every
//! value, so every result is resolved on every 7-player board, and checks Time-out's rule at each.

use botbowl_engine::core::dices::{resolve_with_rng, RequestedRoll, RollResult, Sum2D6};
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{
    BoardDims, MicroStepState, SomeProcInput, TeamType, Weather, HEIGHT, TEAM_SIZE, WIDTH,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const STEP_BUDGET: usize = 400_000;

/// Engine dims (playable + 2) of every board the density rule gives 7 players at 20x11/7
/// capacity: 18x10, 20x9, 18x11, 20x10, 20x11 playable.
const SEVEN_PLAYER_BOARDS: [(i8, i8); 5] = [(20, 12), (22, 11), (20, 13), (22, 12), (22, 13)];

fn fits(w: i8, h: i8, team_size: usize) -> bool {
    WIDTH as i8 >= w && HEIGHT as i8 >= h && TEAM_SIZE >= team_size
}

const TABLE: [Sum2D6; 11] = [
    Sum2D6::Two,
    Sum2D6::Three,
    Sum2D6::Four,
    Sum2D6::Five,
    Sum2D6::Six,
    Sum2D6::Seven,
    Sum2D6::Eight,
    Sum2D6::Nine,
    Sum2D6::Ten,
    Sum2D6::Eleven,
    Sum2D6::Twelve,
];

fn turn_of(state: &GameState, team: TeamType) -> u8 {
    match team {
        TeamType::Home => state.info.home_turn,
        TeamType::Away => state.info.away_turn,
    }
}

#[derive(Default)]
struct Seen {
    /// Table and weather rolls forced so far, across games: the next forced value's index.
    tables: usize,
    weathers: usize,
    table: [usize; 11],
    weather: [usize; 11],
    timeouts_back: usize,
    timeouts_forward: usize,
}

/// One full game on `dims`. Across games, the kickoff-table rolls alternate between the next value
/// of `TABLE` and Changing Weather (so the weather roll comes up often), and the weather rolls walk
/// `TABLE` in turn. Every other roll comes from the rng.
fn play(dims: BoardDims, seed: u64, seen: &mut Seen) {
    let mut state = GameStateBuilder::new()
        .with_board_dims(dims)
        .set_state(BuilderState::CoinToss)
        .build();
    state.set_logging_state(false);
    state.set_dice_mode(DiceMode::RegisterRolls);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut tables = 0usize;
    let mut steps = 0usize;
    while !state.info.game_over {
        steps += 1;
        assert!(
            steps < STEP_BUDGET,
            "{dims:?} seed {seed}: no game over within {STEP_BUDGET} micro-steps"
        );
        let outcome = match state.pending_roll {
            Some(RequestedRoll::Sum2D6) if state.proc_stack_top() == Some("KickoffTable") => {
                let k = seen.tables;
                let r = if k % 2 == 1 {
                    Sum2D6::Eight
                } else {
                    TABLE[(k / 2) % TABLE.len()]
                };
                seen.tables += 1;
                tables += 1;
                seen.table[r as usize - 2] += 1;
                let kicker = state.info.kicking_this_drive;
                let before = (state.info.home_turn, state.info.away_turn, turn_of(&state, kicker));
                let out = state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Sum2D6(r)));
                if r == Sum2D6::Three {
                    // Time-out!: if the *kicking* team's turn marker is on 6, 7 or 8 both
                    // markers move one space back; otherwise both move one space forward.
                    let delta: i16 = if (6..=8).contains(&before.2) { -1 } else { 1 };
                    if delta < 0 {
                        seen.timeouts_back += 1;
                    } else {
                        seen.timeouts_forward += 1;
                    }
                    assert_eq!(
                        (state.info.home_turn as i16, state.info.away_turn as i16),
                        (before.0 as i16 + delta, before.1 as i16 + delta),
                        "{dims:?} seed {seed}: time-out with {kicker:?} kicking on turn {} (home {}, away {})",
                        before.2,
                        before.0,
                        before.1
                    );
                }
                out
            }
            Some(RequestedRoll::Sum2D6) if state.proc_stack_top() == Some("ChangingWeather") => {
                let r = TABLE[seen.weathers % TABLE.len()];
                seen.weathers += 1;
                seen.weather[r as usize - 2] += 1;
                let out = state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Sum2D6(r)));
                assert_eq!(
                    state.info.weather,
                    Weather::from(r),
                    "{dims:?} seed {seed}: weather roll {r:?}"
                );
                out
            }
            Some(requested) => {
                let result = resolve_with_rng(requested, &mut rng);
                state.step_with_roll_or_action(SomeProcInput::Roll(result))
            }
            None => {
                let actions = state.get_all_actions();
                assert!(
                    !actions.is_empty(),
                    "{dims:?} seed {seed}: no action offered at {:?}",
                    state.proc_stack_top()
                );
                let action = actions[rng.gen_range(0..actions.len())];
                state.step_with_roll_or_action(SomeProcInput::Action(action))
            }
        };
        assert!(
            matches!(
                outcome,
                MicroStepState::NeedAction | MicroStepState::NeedRoll | MicroStepState::GameOver
            ),
            "{dims:?} seed {seed}: leaked {outcome:?}"
        );
        assert!(
            state.info.home_turn <= 8 && state.info.away_turn <= 8,
            "{dims:?} seed {seed}: turn past 8"
        );
    }
    assert!(
        tables >= 2,
        "{dims:?} seed {seed}: a game rolls the table at least once a half"
    );
}

#[test]
fn every_kickoff_table_result_resolves_on_every_seven_player_board() {
    let mut seen = Seen::default();
    let mut boards = 0;
    for (w, h) in SEVEN_PLAYER_BOARDS {
        if !fits(w, h, 7) {
            eprintln!("skipped {w}x{h}/7: compiled capacity {WIDTH}x{HEIGHT}/{TEAM_SIZE}");
            continue;
        }
        boards += 1;
        let dims = BoardDims::new(w, h, 7);
        assert!(dims.kickoff_table_enabled());
        for seed in 0..6u64 {
            play(dims, 6500 + seed, &mut seen);
        }
    }
    if boards == 0 {
        return;
    }
    assert!(
        seen.table.iter().all(|&n| n > 0),
        "every table result was rolled: {:?}",
        seen.table
    );
    assert!(
        seen.weather.iter().all(|&n| n > 0),
        "every weather result was rolled: {:?}",
        seen.weather
    );
    assert!(seen.timeouts_forward > 0, "a forward time-out was checked");
}

/// The case that tells the kicking team's marker from Home's: Away received first in this half
/// and scored on its turn 6, so Away kicks on 6 while Home's marker is on 5. Back, not forward.
#[test]
fn time_out_reads_the_kicking_teams_marker_not_homes() {
    let dims = BoardDims::default();
    let mut state = GameStateBuilder::new()
        .with_board_dims(dims)
        .set_state(BuilderState::CoinToss)
        .build();
    state.set_logging_state(false);
    state.set_dice_mode(DiceMode::RegisterRolls);
    let mut rng = ChaCha8Rng::seed_from_u64(65);
    // Play to the first kickoff-table roll of the game, if this capacity has the table at all.
    let mut steps = 0;
    loop {
        steps += 1;
        assert!(steps < STEP_BUDGET);
        if state.info.game_over || (state.pending_roll.is_none() && state.proc_stack_top() == Some("Turn")) {
            assert!(
                !dims.kickoff_table_enabled(),
                "reached a turn without a kickoff-table roll"
            );
            return;
        }
        match state.pending_roll {
            Some(RequestedRoll::Sum2D6) if state.proc_stack_top() == Some("KickoffTable") => break,
            Some(requested) => {
                let r = resolve_with_rng(requested, &mut rng);
                state.step_with_roll_or_action(SomeProcInput::Roll(r));
            }
            None => {
                let actions = state.get_all_actions();
                let a = actions[rng.gen_range(0..actions.len())];
                state.step_with_roll_or_action(SomeProcInput::Action(a));
            }
        };
    }
    for (kicker, home, away, want) in [
        // Away kicks on 6, Home on 5: back.
        (TeamType::Away, 5u8, 6u8, (4u8, 5u8)),
        // Away kicks on 5, Home on 6 (a score in Home's turn): forward.
        (TeamType::Away, 6, 5, (7, 6)),
        // Home kicks on 6, Away on 5: back.
        (TeamType::Home, 6, 5, (5, 4)),
        // Home kicks on 5, Away on 6: forward.
        (TeamType::Home, 5, 6, (6, 7)),
        // The half's opening kickoff, both on 0: forward.
        (TeamType::Away, 0, 0, (1, 1)),
    ] {
        let mut s = state.clone();
        s.info.kicking_this_drive = kicker;
        s.info.home_turn = home;
        s.info.away_turn = away;
        s.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Sum2D6(Sum2D6::Three)));
        assert_eq!(
            (s.info.home_turn, s.info.away_turn),
            want,
            "time-out with {kicker:?} kicking, home on {home}, away on {away}"
        );
    }
}
