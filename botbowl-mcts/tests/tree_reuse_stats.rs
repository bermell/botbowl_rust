//! Plan 043: the tree-reuse outcome the bot now records for every decision.
//!
//! `run_search` has always re-rooted its cached tree when it could, but the answer was thrown away
//! — the only trace was a `was_reused` flag in a panic message. It matters because a reused tree
//! arrives with a plan already in it: the search continues from what it worked out last time
//! instead of starting over.
//!
//! What these tests pin is the *classification*, not the rate. The rate depends on the position
//! and the budget and will drift with the search; the mapping from cause to
//! [`ReuseOutcome`] must not.

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameStateBuilder};
use botbowl_engine::core::model::{BoardDims, Position, TEAM_SIZE};
use botbowl_mcts::{MctsBot, ReuseOutcome, SearchBudget};

const W: i8 = 16;
const H: i8 = 9;
const PLAYERS: usize = 3;

/// The same exhaustible one-carrier position `solved_early_stop` uses: a home carrier two squares
/// from the endzone on an otherwise empty pitch, so a modest budget settles the turn.
fn carrier_state() -> Option<botbowl_engine::core::gamestate::GameState> {
    if (botbowl_engine::core::model::WIDTH as i8) < W
        || (botbowl_engine::core::model::HEIGHT as i8) < H
        || TEAM_SIZE < PLAYERS
    {
        return None;
    }
    let carrier_pos = Position::new((3, 4));
    let mut state = GameStateBuilder::new()
        .with_board_dims(BoardDims::new(W, H, PLAYERS))
        .add_home_player(carrier_pos)
        .add_ball_pos(carrier_pos)
        .build();
    state.set_seed(0);
    state.set_dice_mode(DiceMode::RollDice);
    Some(state)
}

/// Drive a few decisions and collect the outcome of each.
fn drive(bot: &mut MctsBot, moves: usize) -> Vec<ReuseOutcome> {
    let Some(mut state) = carrier_state() else {
        return Vec::new();
    };
    let mut outcomes = Vec::new();
    for _ in 0..moves {
        if state.info.game_over {
            break;
        }
        let action = bot.get_action(&state);
        outcomes.push(
            bot.last_search()
                .expect("a completed search always leaves a summary")
                .reuse
                .outcome,
        );
        state.step(action).unwrap();
    }
    outcomes
}

/// The first decision of a game has nothing to reuse; later ones inside the same turn do.
///
/// Only the first outcome is asserted exactly. The rest are asserted to be *legal* — which
/// outcome a mid-turn decision gets depends on whether the line the game actually took was
/// materialised, and that is a property of the search, not of this accounting.
#[test]
fn the_first_decision_has_no_cache_and_later_ones_can_reuse() {
    let mut bot = MctsBot::new(SearchBudget::Iterations(400)).with_workers(1);
    let outcomes = drive(&mut bot, 4);
    if outcomes.is_empty() {
        return; // board too small for this build
    }

    assert_eq!(
        outcomes[0],
        ReuseOutcome::NoCache,
        "nothing has been searched yet, so there is no tree to re-root"
    );
    for (i, o) in outcomes.iter().enumerate().skip(1) {
        assert_ne!(*o, ReuseOutcome::NoCache, "decision {i}: a tree was cached by then");
        assert_ne!(*o, ReuseOutcome::Disabled, "decision {i}: reuse is on for this bot");
        assert_ne!(
            *o,
            ReuseOutcome::MarkerMiss,
            "decision {i}: the memory mode never changes mid-game"
        );
    }

    let t = bot.telemetry();
    assert_eq!(t.searches, outcomes.len() as u64, "one record per decision");
    assert_eq!(
        t.reuse.total.attempts(),
        outcomes.len() as u64,
        "every decision lands in exactly one outcome bucket"
    );
    assert!(
        !t.reuse.by_proc.is_empty(),
        "each decision is attributed to the procedure on top of the stack"
    );
    assert_eq!(
        t.fan.total(),
        outcomes.len() as u64,
        "the action fan is recorded once per decision"
    );
    assert!(
        t.fan.percentile(0.5).is_some_and(|p| p > 0),
        "a decision state offers at least one legal action"
    );
}

/// With reuse off, every decision reports `Disabled` — never a miss, which would wrongly suggest
/// the bot tried and failed.
#[test]
fn reuse_off_reports_disabled_rather_than_a_miss() {
    let mut bot = MctsBot::new(SearchBudget::Iterations(400))
        .with_workers(1)
        .with_tree_reuse(false);
    let outcomes = drive(&mut bot, 3);
    if outcomes.is_empty() {
        return;
    }

    assert!(
        outcomes.iter().all(|o| *o == ReuseOutcome::Disabled),
        "expected every decision to report Disabled, got {outcomes:?}"
    );
    assert_eq!(bot.telemetry().reuse.total.reused, 0);
    assert_eq!(
        bot.telemetry().reuse.total.rate(),
        Some(0.0),
        "the bot never started from a plan"
    );
}

/// The search touches the transposition table, and the per-decision delta is accounted like the
/// reuse outcome is — one contribution per decision, summing to the bot's total.
///
/// Run long enough to cross a turn boundary and to hit a rebuild. That matters: the baseline for
/// a decision's delta has to come from the tree the search actually runs on, and a decision that
/// *tries* to reuse and fails searches a different tree from the one it probed. Differencing the
/// wrong pair saturates to zero, which shows up here as `probes != hits + misses`.
#[test]
fn recombination_counters_accumulate_per_decision() {
    let mut bot = MctsBot::new(SearchBudget::Iterations(400)).with_workers(1);
    let Some(mut state) = carrier_state() else {
        return;
    };

    let mut summed = botbowl_mcts::RecombinationCounts::default();
    let mut outcomes = Vec::new();
    for _ in 0..30 {
        if state.info.game_over {
            break;
        }
        let action = bot.get_action(&state);
        let s = bot.last_search().expect("summary");
        summed.merge(&s.recombination);
        outcomes.push(s.reuse.outcome);
        state.step(action).unwrap();
    }
    assert!(
        outcomes.iter().any(|o| !o.reused()),
        "the run must include at least one rebuild, or it does not exercise the baseline; got {outcomes:?}"
    );

    let total = &bot.telemetry().recombination;
    assert_eq!(
        summed, *total,
        "the bot's totals are exactly the sum of the per-decision deltas"
    );
    assert!(total.probes > 0, "a search must probe the registry");
    assert_eq!(
        total.probes,
        total.hits + total.misses,
        "every probe is either a hit or a miss"
    );
    assert!(
        total.eq_checks >= total.hits,
        "each hit was confirmed by at least one state comparison"
    );
}
