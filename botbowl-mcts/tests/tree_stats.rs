//! Plan 060: tree statistics on real searches — every descent counted once, the main line through
//! a turn end into the opponent's turn, and none at the last turn of a half.

use botbowl_data::{Phase, TreeStats};
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{PlayerStats, Position, TeamType};
use botbowl_engine::core::procedures::TURNS_PER_HALF;
use botbowl_mcts::tree_stats::opp_turn_follows;
use botbowl_mcts::{MctsBot, MctsConfig, SearchBudget};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Home at the start of its turn on an otherwise quiet pitch: two Home players, one Away player
/// far off. Small enough that a modest budget runs lines through the rest of Home's turn.
fn open_state() -> GameState {
    GameStateBuilder::new()
        .add_home_player(Position::new((5, 5)))
        .add_home_player(Position::new((3, 3)))
        .add_away_player(Position::new((12, 6)))
        .build()
}

fn bot(iters: usize, gumbel_m: u16) -> MctsBot {
    let mut cfg = MctsConfig::new();
    cfg.workers = 1;
    cfg.gumbel_m = gumbel_m;
    MctsBot::with_budget_and_config(SearchBudget::Iterations(iters), cfg)
}

fn search(state: &GameState, iters: usize, gumbel_m: u16) -> (MctsBot, TreeStats) {
    let mut b = bot(iters, gumbel_m);
    let (_, sample) = b.get_action_with_record(state);
    let tree = sample.tree.expect("a searched decision carries tree statistics");
    (b, tree)
}

/// Every descent is counted exactly once: in one end phase and one valuation.
fn assert_per_descent_totals(t: &TreeStats, budget: usize) {
    assert!(t.descents > 0 && t.descents as usize <= budget, "{t:?}");
    assert_eq!(t.ends.total(), t.descents, "one end phase per descent: {t:?}");
    let v = t.valued;
    assert_eq!(
        v.new_leaf + v.chance + v.terminal + v.solved + v.horizon,
        t.descents,
        "one valuation per descent: {t:?}"
    );
    assert!(t.plies.mean <= t.plies.p90 as f32 + 1e-3 || t.plies.p90 == t.plies.max);
    assert!(t.plies.p90 <= t.plies.max && t.own.max <= t.plies.max);
    assert!(t.reached_opp_turn >= t.ends.opp_turn + t.ends.horizon);
}

/// Home's last decisions of a turn: one slow Home player (MA 1) on the pitch, so the rest of Home's
/// turn is a handful of plies and the budget has to go into Away's turn.
fn turn_end_state() -> GameState {
    let slow = |team| {
        let mut p = PlayerStats::new_lineman(team);
        p.ma = 1;
        p
    };
    GameStateBuilder::new()
        .add_player_details(Position::new((5, 5)), TeamType::Home, slow(TeamType::Home))
        .add_player_details(Position::new((12, 6)), TeamType::Away, slow(TeamType::Away))
        .build()
}

#[test]
fn a_turn_end_search_runs_its_main_line_into_the_opponents_turn() {
    let state = turn_end_state();
    let (_, t) = search(&state, 1500, 0);
    assert_per_descent_totals(&t, 1500);
    assert!(
        matches!(t.main_line.phase, Some(Phase::OppTurn | Phase::Horizon)),
        "the main line runs through Home's turn end into Away's turn: {t:?}"
    );
    assert!(t.main_line.own >= 1 && t.main_line.plies > t.main_line.own);
    assert!(
        t.reached_opp_turn * 2 > t.descents,
        "most lines reach Away's turn: {t:?}"
    );
}

#[test]
fn a_turn_start_search_runs_lines_into_the_opponents_turn() {
    let state = open_state();
    for gumbel_m in [0, 4] {
        let (b, t) = search(&state, 1500, gumbel_m);
        assert_per_descent_totals(&t, 1500);
        assert!(t.opp_turn_follows, "mid-half: Away plays next");
        assert_eq!(t.proc.as_deref(), state.proc_stack_top());
        assert!(t.reached_opp_turn > 0, "some lines reach Away's turn: {t:?}");
        assert!(t.plies.max >= 2 && t.own.max >= 1, "{t:?}");
        assert!(t.main_line.own >= 1 && t.main_line.plies >= t.main_line.own);

        // The same numbers reach the bot's telemetry.
        let tel = &b.telemetry().tree;
        assert_eq!((tel.searches, tel.descents), (1, u64::from(t.descents)));
        assert_eq!(tel.reached_opp_turn, u64::from(t.reached_opp_turn));
        assert_eq!(tel.main_plies, u64::from(t.main_line.plies));
        assert_eq!(tel.opp_turn_follows, 1);
    }
}

/// Home kicked this half and is on its last turn: Away has played all of its turns, so no line can
/// reach an opponent's turn and every line that gets past Home's turn ends the half.
#[test]
fn the_last_turn_of_a_half_has_no_opponent_turn() {
    let mut state = open_state();
    assert!(
        state.set_kicking_this_half(TeamType::Home),
        "the builder state has a started half"
    );
    state.info.home_turn = TURNS_PER_HALF;
    state.info.away_turn = TURNS_PER_HALF;
    let (_, t) = search(&state, 800, 0);
    assert_per_descent_totals(&t, 800);
    assert!(!t.opp_turn_follows);
    assert_eq!(
        (t.ends.opp_turn, t.ends.horizon, t.reached_opp_turn),
        (0, 0, 0),
        "{t:?}"
    );
    assert!(t.ends.half_end > 0, "lines through EndTurn end the half: {t:?}");
}

/// Tree statistics are observational: a second identical search plays the same move with the same
/// root statistics (the tally only reads the tree).
#[test]
fn the_statistics_do_not_change_the_search() {
    let state = open_state();
    let mut a = bot(600, 0);
    let mut b = bot(600, 0);
    let (move_a, sa) = a.get_action_with_record(&state);
    let (move_b, sb) = b.get_action_with_record(&state);
    assert_eq!(move_a, move_b);
    assert_eq!(sa.root_visits, sb.root_visits);
    assert_eq!(sa.tree.as_ref().unwrap().descents, sb.tree.as_ref().unwrap().descents);
}

/// `opp_turn_follows` against the engine itself: random games, and at every decision whether the
/// opponent's turn counter moves before the decider's own (or the half ends). Kickoff timeouts move
/// both counters at once and are skipped.
#[test]
fn opp_turn_follows_matches_the_engine_turn_order() {
    let mut checked = 0;
    let mut follows = [0u32; 2];
    for seed in 0..6u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut state = GameStateBuilder::new_start_of_game();
        state.set_dice_mode(DiceMode::RollDice);
        state.set_seed(seed);
        state.set_logging_state(false);
        // (decider, prediction, half, home_turn, away_turn) at every decision still unresolved.
        let mut open: Vec<(TeamType, bool, u8, u8, u8)> = Vec::new();
        let mut steps = 0;
        while !state.info.game_over && steps < 20_000 {
            let (h, a, half) = (state.info.home_turn, state.info.away_turn, state.info.half);
            open.retain(|&(team, predicted, half0, h0, a0)| {
                let (own_moved, opp_moved) = match team {
                    TeamType::Home => (h != h0, a != a0),
                    TeamType::Away => (a != a0, h != h0),
                };
                let truth = if half != half0 {
                    Some(false)
                } else if own_moved && opp_moved {
                    return false; // a kickoff timeout: skip
                } else if opp_moved {
                    Some(true)
                } else if own_moved {
                    Some(false)
                } else {
                    None
                };
                match truth {
                    Some(t) => {
                        assert_eq!(predicted, t, "seed {seed}: {team:?} at half {half0} turns {h0}/{a0}");
                        checked += 1;
                        follows[usize::from(t)] += 1;
                        false
                    }
                    None => true,
                }
            });
            if let Some(team) = state.available_actions.team.filter(|_| half > 0) {
                // One open prediction per (team, turn) is enough.
                if !open.iter().any(|o| o.0 == team && (o.2, o.3, o.4) == (half, h, a)) {
                    open.push((team, opp_turn_follows(&state, team, 1), half, h, a));
                }
            }
            let actions = state.get_all_actions();
            let action = actions[rng.gen_range(0..actions.len())];
            state.step(action).unwrap();
            steps += 1;
        }
    }
    assert!(checked > 50, "enough decisions resolved: {checked}");
    assert!(follows[0] > 0 && follows[1] > 0, "both answers occur: {follows:?}");
}
