//! Plan 053: Gumbel root search (sequential halving at the root).
//!
//! What has to hold for the A/B to mean anything: every descent's root move comes from the
//! considered set, every considered move gets its first-phase share, the budget is spent exactly,
//! the move played is a survivor, and the search is reproducible.

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{BoardDims, Position, TEAM_SIZE};
use botbowl_mcts::{ExploreStep, MctsBot, MctsConfig, RootNoiseSpec, SampleSpec, SearchBudget};

const W: i8 = 16;
const H: i8 = 9;
const PLAYERS: usize = 3;
const N: usize = 240;
const M: u16 = 8;

/// Three a side and a carrier in midfield: a wide first decision (start any of three players)
/// that a few hundred descents never solve.
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
    state.set_seed(7);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    Some(state)
}

fn bot(m: u16, scale: f32) -> MctsBot {
    let mut cfg = MctsConfig::new();
    cfg.workers = 1;
    cfg.trace_root_descents = true;
    cfg.gumbel_m = m;
    cfg.gumbel_scale = scale;
    MctsBot::with_budget_and_config(SearchBudget::Iterations(N), cfg)
}

/// Walk into the turn until the root offers more than `M` moves (a player's move fan).
fn wide_state() -> Option<GameState> {
    let mut state = open_state()?;
    let mut puct = bot(0, 0.0);
    for _ in 0..6 {
        if state.get_all_actions().len() > M as usize * 2 {
            return Some(state);
        }
        let a = puct.get_action(&state);
        state.step(a);
    }
    (state.get_all_actions().len() > M as usize * 2).then_some(state)
}

#[test]
fn halving_spends_the_budget_on_the_considered_moves_only() {
    let Some(state) = wide_state() else { return };
    let mut b = bot(M, 0.0);
    let played = b.get_action(&state);
    let s = b.last_search().unwrap();
    let descents = s.root_descents.clone().unwrap();
    let total: u32 = descents.iter().map(|(_, n)| n).sum();
    // The fresh root's first descent expands it and selects nothing; every other descent was
    // named, unless the named move was solved in the meantime (then it ran as plain PUCT, which
    // can reach a move outside the considered set). The search may stop early once every
    // survivor is solved.
    let steps = b.telemetry().iterations as usize;
    assert!(steps <= N && steps >= N / 3, "{steps} descents");
    assert_eq!(total as usize, steps - 1, "{descents:?}");

    // Without noise the considered set is the top M by prior.
    let mut by_prior: Vec<_> = s
        .children
        .iter()
        .map(|e| (e.action.clone(), e.action.prior_f32().unwrap()))
        .collect();
    by_prior.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let considered: Vec<_> = by_prior.iter().take(M as usize).map(|(a, _)| a.clone()).collect();
    let is_considered = |a: &botbowl_engine::core::model::Action| {
        considered
            .iter()
            .any(|c| matches!(c, botbowl_mcts::BbAction::Player { action, .. } if action == a))
    };
    let inside: u32 = descents.iter().filter(|(a, _)| is_considered(a)).map(|(_, n)| n).sum();
    let solved_early = s.children.iter().filter(|e| e.stats.solved).count();
    // Outside the considered set only the fallbacks of moves that got solved mid-phase, one each.
    assert!(total - inside <= (solved_early as u32).max(1) * 2, "{descents:?}");
    // Every considered move that was still open got its phase-one share, N / (3 phases · 8) = 10.
    for e in s.children.iter().filter(|e| !e.stats.solved) {
        if let botbowl_mcts::BbAction::Player { action, .. } = &e.action {
            if is_considered(action) {
                let n = descents.iter().find(|(a, _)| a == action).map_or(0, |(_, n)| *n);
                assert!(n >= 10, "{action:?} got only {n} descents");
            }
        }
    }
    assert!(
        descents.iter().any(|(a, _)| *a == played),
        "the move played was never searched"
    );
}

#[test]
fn a_gumbel_search_is_reproducible_and_noise_moves_it() {
    let Some(state) = wide_state() else { return };
    let run = |scale: f32| {
        let mut b = bot(M, scale);
        let a = b.get_action(&state);
        let mut d = b.last_search().unwrap().root_descents.clone().unwrap();
        d.sort_by_key(|(a, _)| format!("{a:?}"));
        (a, d)
    };
    assert_eq!(run(0.0), run(0.0));
    assert_eq!(run(1.0), run(1.0));
    // Noise draws a different considered set from the deterministic top M (seeded by the state).
    let quiet: Vec<_> = run(0.0).1.into_iter().map(|(a, _)| a).collect();
    let noisy: Vec<_> = run(1.0).1.into_iter().map(|(a, _)| a).collect();
    assert_ne!(quiet, noisy);
}

#[test]
fn gumbel_off_is_the_shipped_search() {
    let Some(state) = wide_state() else { return };
    let mut a = bot(0, 0.0);
    let mut b = MctsBot::with_budget_and_config(SearchBudget::Iterations(N), {
        let mut c = MctsConfig::new();
        c.workers = 1;
        c.trace_root_descents = true;
        c
    });
    assert_eq!(a.get_action(&state), b.get_action(&state));
    let da = a.last_search().unwrap().root_descents.clone();
    let db = b.last_search().unwrap().root_descents.clone();
    assert_eq!(da, db);
}

#[test]
fn plan_048_exploration_is_ignored_under_gumbel() {
    let Some(state) = wide_state() else { return };
    let mut plain = bot(M, 1.0);
    let expected = plain.get_action(&state);
    let mut explored = bot(M, 1.0);
    let step = ExploreStep {
        noise: Some(RootNoiseSpec {
            epsilon: 0.5,
            alpha: 1.0,
            seed: 3,
        }),
        sample: Some(SampleSpec {
            temperature: 1.0,
            u: 0.999,
        }),
    };
    let (action, sample, outcome) = explored.get_action_explore(&state, step);
    assert_eq!(
        action, expected,
        "the Gumbel pick is the move, whatever the plan-048 knobs say"
    );
    assert_eq!(sample.chosen_action, expected);
    assert!(!outcome.noised && !outcome.sampled && !outcome.deviated, "{outcome:?}");
}

/// A reused tree keeps the dynamics it was built with, so the root forcing must reach it too: the
/// review of 2026-10-03 found the slot rebuilt per search, which left every reused search (about
/// 60% of decisions) running plain PUCT descents under a Gumbel pick.
#[test]
fn halving_also_drives_a_reused_tree() {
    let Some(mut state) = wide_state() else { return };
    let mut b = bot(M, 0.0);
    for _ in 0..16 {
        let a = b.get_action(&state);
        // A heuristic bot answers a kickoff setup from a formation, without a search (plan 047).
        let searched = b.last_search().cloned();
        if let Some(s) =
            searched.filter(|s| s.reuse.outcome == botbowl_mcts::ReuseOutcome::Reused && s.children.len() > M as usize)
        {
            let descents = s.root_descents.clone().unwrap();
            // Every descent of a reused search is named (no expansion step), apart from fallbacks
            // of moves solved mid-phase; plain PUCT spreads over far more than M moves.
            let solved = s.children.iter().filter(|e| e.stats.solved).count();
            assert!(
                descents.len() <= M as usize + solved,
                "{} root moves descended on a reused tree: {descents:?}",
                descents.len()
            );
            return;
        }
        state.step(a);
        if state.info.game_over || state.available_actions.team.is_none() {
            break;
        }
        b.release_stale_tree(&state);
    }
    panic!("no reused root wider than {M} moves found to test");
}
