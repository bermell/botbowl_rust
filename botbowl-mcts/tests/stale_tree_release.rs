//! `MctsBot::release_stale_tree`: a two-bot game loop frees the idle side's tree once its turn is
//! over, instead of carrying it through the opponent's whole turn to be discarded at the next
//! search. It must be invisible to the search: same moves, same root statistics.

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{BoardDims, Position, TeamType, TEAM_SIZE};
use botbowl_mcts::{MctsBot, SearchBudget, TieBreak};

const W: i8 = 16;
const H: i8 = 9;
const PLAYERS: usize = 3;

fn open_state() -> Option<GameState> {
    if (botbowl_engine::core::model::WIDTH as i8) < W
        || (botbowl_engine::core::model::HEIGHT as i8) < H
        || TEAM_SIZE < PLAYERS
    {
        return None;
    }
    let carrier = Position::new((5, 4));
    let mut state = GameStateBuilder::new()
        .with_board_dims(BoardDims::new(W, H, PLAYERS))
        .add_home_player(carrier)
        .add_home_player(Position::new((4, 2)))
        .add_home_player(Position::new((4, 6)))
        .add_away_player(Position::new((11, 3)))
        .add_away_player(Position::new((11, 5)))
        .add_away_player(Position::new((13, 4)))
        .add_ball_pos(carrier)
        .build();
    state.set_seed(11);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    Some(state)
}

fn bot() -> MctsBot {
    MctsBot::new(SearchBudget::Iterations(150))
        .with_workers(1)
        .with_tie_break(TieBreak::Mover)
}

/// Play up to `steps` decisions; one line per decision, how often the idle side's cache was freed,
/// and both bots' pooled reuse telemetry.
fn play(release: bool, steps: usize) -> (Vec<String>, usize, String) {
    let Some(mut state) = open_state() else {
        return (Vec::new(), 0, String::new());
    };
    let (mut home, mut away) = (bot(), bot());
    let mut lines = Vec::new();
    let mut idle_freed = 0;
    for _ in 0..steps {
        if state.info.game_over {
            break;
        }
        let team = match state.available_actions.team {
            Some(t) => t,
            None => break,
        };
        let b = if team == TeamType::Home { &mut home } else { &mut away };
        let (action, sample) = b.get_action_with_record(&state);
        lines.push(format!(
            "{team:?} {action:?} visits={} value={:?} children={:?}",
            sample.root_visits,
            sample.root_value,
            sample
                .children
                .iter()
                .map(|c| (c.action, c.visits, c.q))
                .collect::<Vec<_>>()
        ));
        state.step(action).unwrap();
        if release {
            home.release_stale_tree(&state);
            away.release_stale_tree(&state);
            let idle = match state.available_actions.team {
                Some(TeamType::Home) => &away,
                Some(TeamType::Away) => &home,
                None => continue,
            };
            idle_freed += usize::from(!idle.has_cached_tree());
        }
    }
    let mut t = home.take_telemetry();
    t.merge(away.telemetry());
    (lines, idle_freed, format!("{:?}", t.reuse))
}

#[test]
fn releasing_stale_trees_does_not_change_the_search() {
    let (kept, _, kept_reuse) = play(false, 60);
    if kept.is_empty() {
        return; // board too small for this build
    }
    let (released, idle_freed, released_reuse) = play(true, 60);
    assert_eq!(kept.len(), released.len());
    for (i, (a, b)) in kept.iter().zip(&released).enumerate() {
        assert_eq!(a, b, "decision {i} moved");
    }
    assert!(
        idle_freed > 0,
        "the idle side's tree must actually be freed at some point"
    );
    // A freed tree is the anchor miss the next search would have had, and is reported as one.
    assert_eq!(kept_reuse, released_reuse);
}
