//! Mirror-invariance property tests (plan 023).
//!
//! Blood Bowl is symmetric under "reflect the board about its vertical
//! midline and swap the two teams" (`GameState::mirrored`). Home attacks
//! `x = 1` and Away attacks `x = width - 2`, so any deterministic choice
//! made in *absolute* board coordinates is a side bias waiting to happen.
//! Plan 023 has already caught two of those by measurement after asserting
//! by inspection that the code was symmetric — `ScriptedBot`'s touchback
//! tie-break (a 0.113 Home share over 344 games) and the search's
//! throw-in / bounce representatives.
//!
//! These tests replace inspection with assertions, over a few hundred
//! random mid-drive states:
//!
//! * `leaf_score` is exactly antisymmetric,
//! * `prior_for` and `should_prune` are exactly mover-relative,
//! * the legal-action set mirrors onto itself.

use botbowl_curriculum::random_start::{generate_random_start, RandomStartConfig};
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action as EngineAction, BoardDims, Position};
use botbowl_mcts::pruning::should_prune;
use botbowl_mcts::score::leaf_score;
use botbowl_mcts::{priors, BbAction};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// The 14x7 curriculum tier (engine 16x9), which every plan-023 mirror
/// measurement was taken on. Pinned here rather than read from the
/// environment so the property holds on any build.
fn tier() -> BoardDims {
    BoardDims::new(16, 9, 4)
}

fn states(n: u32, seed: u64) -> Vec<GameState> {
    let cfg = RandomStartConfig {
        board_dims: Some(tier()),
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| generate_random_start(&cfg, &mut rng)).collect()
}

fn flip(dims: BoardDims, p: Position) -> Position {
    Position::new((dims.width - 1 - p.x, p.y))
}

fn mirror_action(dims: BoardDims, a: EngineAction) -> EngineAction {
    match a {
        EngineAction::Positional(at, pos) => EngineAction::Positional(at, flip(dims, pos)),
        simple => simple,
    }
}

/// `mirrored` must be its own inverse on everything it claims to cover;
/// otherwise the tests below could pass against a map that quietly loses
/// information.
#[test]
fn mirroring_twice_is_the_identity() {
    for (i, s) in states(60, 23_003).into_iter().enumerate() {
        let back = s.mirrored().mirrored();
        assert_eq!(
            leaf_score(&back),
            leaf_score(&s),
            "state {i}: double mirror changed the leaf score"
        );
        let mut a: Vec<_> = s.get_all_actions();
        let mut b: Vec<_> = back.get_all_actions();
        a.sort();
        b.sort();
        assert_eq!(a, b, "state {i}: double mirror changed the action set");
    }
}

/// The claim plan 023 made by inspection: every term of `leaf_score`
/// flips exactly under reflection + team swap.
#[test]
fn leaf_score_is_antisymmetric_under_mirroring() {
    for (i, s) in states(300, 23_000).into_iter().enumerate() {
        assert_eq!(
            leaf_score(&s.mirrored()),
            -leaf_score(&s),
            "state {i}: leaf_score is not antisymmetric"
        );
    }
}

/// Priors and pruning must be functions of the *mover's* geometry, never
/// of absolute x. A violation here is exactly the shape of the
/// `ScriptedBot` bug: the same rule helping one side and hurting the other.
#[test]
fn priors_and_pruning_are_mover_relative() {
    let dims = tier();
    let mut checked = 0usize;
    for (i, s) in states(300, 23_001).into_iter().enumerate() {
        let m = s.mirrored();
        for a in s.get_all_actions() {
            let ma = mirror_action(dims, a);
            assert_eq!(
                priors::prior_for(&m, &BbAction::player(ma, 1.0)),
                priors::prior_for(&s, &BbAction::player(a, 1.0)),
                "state {i}: prior for {a:?} changed under mirroring"
            );
            assert_eq!(
                should_prune(&m, &ma),
                should_prune(&s, &a),
                "state {i}: pruning of {a:?} changed under mirroring"
            );
            checked += 1;
        }
    }
    assert!(checked > 1000, "expected a few thousand actions, saw {checked}");
}

/// The legal-action set itself has to be symmetric — if one side is
/// offered options the other is not, nothing downstream can fix it.
#[test]
fn legal_action_set_mirrors_onto_itself() {
    let dims = tier();
    for (i, s) in states(150, 23_002).into_iter().enumerate() {
        let mut expected: Vec<EngineAction> = s.get_all_actions().into_iter().map(|a| mirror_action(dims, a)).collect();
        let mut actual = s.mirrored().get_all_actions();
        expected.sort();
        actual.sort();
        assert_eq!(expected, actual, "state {i}: legal actions do not mirror");
    }
}

// ---------------------------------------------------------------------
// The search itself (plan 023). The properties above cover the pure
// functions MCTS is built from; this covers the search on top of them.
//
// `GameState::mirrored` deliberately does not mirror the procedure stack,
// so its output cannot be *stepped* — and MCTS steps. So the mirrored
// state here is rebuilt through `GameStateBuilder` instead: same players
// reflected and swapped, same ball, same turn/half/score context with the
// two teams exchanged. That state is a real, playable one.
//
// A symmetric search must then satisfy `root_value(mirror(s)) ==
// -root_value(s)` (`root_value` is Home-centric), so the *sum* of the two
// is pure side bias. Averaged over many states it is a direct measurement
// of "does the search over-value Home", with none of a full game's
// variance — and it can be read at two budgets, which is the handle plan
// 023 has on the phenomenon.
//
// Ignored: it runs thousands of searches. Run it explicitly:
//   cargo test --release -p botbowl-mcts --test mirror_symmetry \
//       -- --ignored --nocapture

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameStateBuilder};
use botbowl_engine::core::model::{other_team, BallState, TeamType};
use botbowl_engine::core::table::SimpleAT;
use botbowl_mcts::{MctsBot, SearchBudget};

/// Rebuild `s` reflected and team-swapped, through the builder, so the
/// result has a valid procedure stack and can be played.
fn mirror_playable(s: &GameState, dims: BoardDims) -> GameState {
    let ball_pos = match s.ball {
        BallState::Carried(id) => s.get_player(id).expect("carrier exists").position,
        BallState::OnGround(p) | BallState::InAir(p) => p,
        BallState::OffPitch => panic!("random starts always have the ball on the pitch"),
    };
    let mut builder = GameStateBuilder::new();
    builder
        .with_board_dims(dims)
        .set_state(BuilderState::Turn { turn: 1 })
        .add_ball_pos(flip(dims, ball_pos));
    for p in s.get_players_on_pitch() {
        let pos = flip(dims, p.position);
        match p.stats.team {
            TeamType::Home => builder.add_away_player(pos),
            TeamType::Away => builder.add_home_player(pos),
        };
    }
    let mut m = builder.build();
    m.set_logging_state(false);
    // The builder hands turn 1 to Home; the mirror's mover is the swap of
    // `s`'s, so pass the turn on when `s` had Home to move.
    if s.available_actions.team == Some(TeamType::Home) {
        m.step_simple(SimpleAT::EndTurn);
    }
    m.info.home_turn = s.info.away_turn;
    m.info.away_turn = s.info.home_turn;
    if s.info.half == 2 {
        m.set_half(2);
    }
    m.home.score = s.away.score;
    m.away.score = s.home.score;
    // Random starts never run a coin toss, so both of these sit at their
    // constructed default for *both* states — which would leave the pair
    // agreeing on who kicked instead of mirroring it. The engine reads
    // them at the half boundary, which a turn-8 search reaches.
    m.info.kicking_first_half = other_team(s.info.kicking_first_half);
    m.info.kicking_this_drive = other_team(s.info.kicking_this_drive);
    // The one that actually matters, and the one that is easiest to miss:
    // `Half::step` decides who takes the next team turn from *its own*
    // copy of the kicking team, not from `GameInfo`. Leave it alone and the
    // mirrored state gives its mover two consecutive turns — worth roughly
    // -57 leaf-score points of spurious "side bias" when measured.
    let kicking = s.kicking_this_half().expect("a half is in progress");
    assert!(m.set_kicking_this_half(other_team(kicking)), "mirror has no started half");
    m.set_dice_mode(DiceMode::RollDice);
    m
}

fn side_bias_at_budget(iters: usize, n: u32, seed: u64) {
    let dims = tier();
    let mut sums: Vec<f64> = Vec::new();
    let mut mirrored_pick = 0usize;
    let mut pairs = 0usize;
    // Stratified by how close the state is to the end of the half. The
    // search's horizon is "until the mover's next turn", so a turn-8 state
    // runs past the half boundary and into the kickoff machinery, while an
    // early-turn state never leaves the drive. If the asymmetry lives in
    // one stratum, that localises it.
    let mut by_turn: std::collections::BTreeMap<u8, Vec<f64>> = Default::default();
    for (i, mut s) in states(n, seed).into_iter().enumerate() {
        let mut m = mirror_playable(&s, dims);
        // The rebuild must reproduce the situation exactly, or the two
        // searches are not looking at mirrored problems.
        assert_eq!(leaf_score(&m), -leaf_score(&s), "state {i}: rebuilt mirror is not the mirror");
        // The rebuild is the instrument; if it is not an exact mirror the
        // measurement below is meaningless. Compare the two states' legal
        // sets, which is the strongest cheap check available.
        let mut expect: Vec<_> = s.get_all_actions().into_iter().map(|a| mirror_action(dims, a)).collect();
        let mut got = m.get_all_actions();
        expect.sort();
        got.sort();
        assert_eq!(expect, got, "state {i}: rebuilt mirror offers different actions");
        // Turn order has to mirror too, not just the board.
        assert_eq!(
            m.kicking_this_half().map(other_team),
            s.kicking_this_half(),
            "state {i}: rebuilt mirror does not mirror the turn order"
        );
        s.set_seed(1000 + i as u64);
        m.set_seed(1000 + i as u64);

        let mut bot_s = MctsBot::new(SearchBudget::Iterations(iters)).with_workers(1);
        let mut bot_m = MctsBot::new(SearchBudget::Iterations(iters)).with_workers(1);
        bot_s.set_seed(ChaCha8Rng::seed_from_u64(7000 + i as u64));
        bot_m.set_seed(ChaCha8Rng::seed_from_u64(7000 + i as u64));
        let (a_s, rec_s) = bot_s.get_action_with_record(&s);
        let (a_m, rec_m) = bot_m.get_action_with_record(&m);
        if let (Some(vs), Some(vm)) = (rec_s.root_value, rec_m.root_value) {
            sums.push((vs + vm) as f64);
            let turn = s.info.home_turn.max(s.info.away_turn);
            by_turn.entry(turn).or_default().push((vs + vm) as f64);
            pairs += 1;
        }
        if mirror_action(dims, a_s) == a_m {
            mirrored_pick += 1;
        }
    }
    let n = sums.len() as f64;
    let mean = sums.iter().sum::<f64>() / n;
    let var = sums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    println!(
        "budget {iters:5}  n={pairs:4}  mean(root_home(s) + root_home(mirror s)) = {mean:+8.2} \
         ± {se:.2} (se)  t = {:+.2}   mirrored root pick {mirrored_pick}/{}",
        mean / se,
        sums.len()
    );
    for (turn, v) in &by_turn {
        if v.len() < 8 {
            continue;
        }
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        println!("    turn {turn}: n={:3} mean {m:+8.2} ± {se:.2}  t = {:+.2}", v.len(), m / se);
    }
}

/// Diagnostic, not an assertion: prints the search's side bias at two
/// budgets. Plan 023's mirror-match effect is budget-dependent (0.53 at
/// 200 iterations, 0.67 at 1000), so the two rows are the interesting
/// comparison.
#[test]
#[ignore]
fn search_side_bias_by_budget() {
    // `BB_MIRROR_N` / `BB_MIRROR_BUDGETS` shrink the run for a smoke test
    // on a busy machine.
    let n: u32 = std::env::var("BB_MIRROR_N").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let budgets: Vec<usize> = match std::env::var("BB_MIRROR_BUDGETS") {
        Ok(v) => v.split(',').filter_map(|b| b.trim().parse().ok()).collect(),
        Err(_) => vec![200, 1000],
    };
    for iters in budgets {
        side_bias_at_budget(iters, n, 23_010);
    }
}
