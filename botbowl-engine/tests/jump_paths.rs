//! S3 Jumping over Players and Leap, end to end: whole random games with jumps in the paths and
//! a coach that often takes the hidden `PosAT::Jump` action, and (`#[ignore]`d) the pathfinder's
//! cost with the jump toggle off and on.
//!
//! Benchmark, release only:
//!   cargo test --release -p botbowl-engine --test jump_paths -- --ignored --nocapture

use std::time::{Duration, Instant};

use botbowl_engine::core::dices::resolve_with_rng;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{Action, BallState, MicroStepState, PlayerID, PlayerStatus, SomeProcInput};
use botbowl_engine::core::pathing::{CustomIntoIter, PathFinder, PathingEvent, PositionOrEvent};
use botbowl_engine::core::table::{PosAT, Skill};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const STEP_BUDGET: usize = 400_000;

#[derive(Default, Debug)]
struct Counts {
    /// Jump rolls resolved (`JumpProc` asked for its die), from paths and the hidden action.
    jump_rolls: usize,
    /// Hidden `PosAT::Jump` actions taken.
    hidden_jumps: usize,
}

/// One random game through `DiceMode::RegisterRolls`. Every third player has Leap. At a
/// decision with a hidden jump on offer the coach takes one half the time; otherwise a uniformly
/// random listed action. `visit` sees every `Turn` decision (a coach about to pick a player).
fn play(seed: u64, counts: &mut Counts, mut visit: impl FnMut(&GameState)) {
    let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
    state.set_logging_state(false);
    state.set_dice_mode(DiceMode::RegisterRolls);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut steps = 0;
    while !state.info.game_over {
        steps += 1;
        assert!(steps < STEP_BUDGET, "seed {seed}: no game over within {STEP_BUDGET} steps");
        let outcome = match state.pending_roll {
            Some(requested) => {
                if state.proc_stack_top() == Some("JumpProc") {
                    counts.jump_rolls += 1;
                }
                let result = resolve_with_rng(requested, &mut rng);
                state.step_with_roll_or_action(SomeProcInput::Roll(result))
            }
            None => {
                let leapless: Vec<PlayerID> = state
                    .get_players_on_pitch()
                    .filter(|p| p.id % 3 == 0 && !p.has_skill(Skill::Leap))
                    .map(|p| p.id)
                    .collect();
                for id in leapless {
                    state.get_mut_player_unsafe(id).stats.give_skill(Skill::Leap);
                }
                if state.proc_stack_top() == Some("Turn") {
                    visit(&state);
                }
                let mut hidden = Vec::new();
                state.get_available_actions().collect_hidden_actions(&mut hidden);
                hidden.retain(|a| matches!(a, Action::Positional(PosAT::Jump, _)));
                let action = if !hidden.is_empty() && rng.gen_bool(0.5) {
                    counts.hidden_jumps += 1;
                    hidden[rng.gen_range(0..hidden.len())]
                } else {
                    let actions = state.get_all_actions();
                    assert!(!actions.is_empty(), "seed {seed}: no action at {:?}", state.proc_stack_top());
                    assert!(
                        actions.iter().all(|a| !matches!(a, Action::Positional(PosAT::Jump, _))),
                        "a hidden jump was listed"
                    );
                    actions[rng.gen_range(0..actions.len())]
                };
                assert!(state.is_legal_action(&action));
                state.step_with_roll_or_action(SomeProcInput::Action(action))
            }
        };
        assert!(
            matches!(
                outcome,
                MicroStepState::NeedAction | MicroStepState::NeedRoll | MicroStepState::GameOver
            ),
            "seed {seed}: leaked {outcome:?}"
        );
    }
}

/// Random games with Leap players and jumps taken through paths and the hidden action finish
/// without tripping an engine assertion, and actually jump.
#[test]
fn random_games_with_jumps_play_through() {
    let mut counts = Counts::default();
    for seed in 0..4 {
        play(7100 + seed, &mut counts, |_| ());
    }
    assert!(counts.hidden_jumps > 0, "{counts:?}");
    assert!(counts.jump_rolls > counts.hidden_jumps, "jumps from paths too: {counts:?}");
}

/// `PathFinder::can_reach` (Secure the Ball's offer) answers exactly what the pathfinder does,
/// for every player of the acting team on the loose ball, with jumps in the paths and without.
#[test]
fn can_reach_agrees_with_the_pathfinder() {
    let mut compared = [0usize; 2];
    let mut counts = Counts::default();
    for seed in 0..4 {
        play(7200 + seed, &mut counts, |state| {
            let BallState::OnGround(ball) = state.ball else { return };
            let team = state.available_actions.team.unwrap();
            for jumps in [true, false] {
                let mut state = state.clone();
                state.set_jump_paths(jumps);
                let ids: Vec<PlayerID> = state.get_players_on_pitch_in_team(team).filter(|p| !p.used).map(|p| p.id).collect();
                for id in ids {
                    let paths = PathFinder::player_paths(&state, id).unwrap();
                    assert_eq!(PathFinder::can_reach(&state, id, ball), paths[ball].is_some(), "seed {seed} id {id}");
                    compared[usize::from(paths[ball].is_some())] += 1;
                }
            }
        });
    }
    assert!(compared[0] > 50 && compared[1] > 50, "{compared:?}");
}

/// A pathfinding sample: a state at a `Turn` decision and one of the acting team's players.
struct Sample {
    state: GameState,
    id: PlayerID,
}

fn time_paths(samples: &mut [Sample], on: bool) -> (Duration, usize) {
    let mut with_jumps = 0;
    let start = Instant::now();
    for s in samples.iter_mut() {
        s.state.set_jump_paths(on);
        let paths = PathFinder::player_paths(&s.state, s.id).unwrap();
        with_jumps += paths
            .iter()
            .flatten()
            .filter(|n| {
                n.iter()
                    .any(|e| matches!(e, PositionOrEvent::Event(PathingEvent::Jump { .. })))
            })
            .count();
    }
    (start.elapsed(), with_jumps)
}

/// The pathfinder with jumps off vs on over positions from random games (the default board):
/// as played with nobody down, with four players knocked down, and with the mover given Leap.
#[test]
#[ignore]
fn bench_jump_paths() {
    if cfg!(debug_assertions) {
        eprintln!("SKIP bench_jump_paths: release only");
        return;
    }
    let mut states = Vec::new();
    let mut counts = Counts::default();
    for seed in 0..12 {
        play(9100 + seed, &mut counts, |s| {
            if states.len() < 4000 {
                states.push(s.clone())
            }
        });
    }
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let movers = |s: &GameState| -> Vec<PlayerID> {
        let team = s.get_active_teamtype().unwrap();
        s.get_players_on_pitch()
            .filter(|p| p.stats.team == team && !p.used && p.status == PlayerStatus::Up)
            .map(|p| p.id)
            .collect()
    };
    let strip_leap = |s: &mut GameState| {
        let ids: Vec<PlayerID> = s.get_players_on_pitch().map(|p| p.id).collect();
        for id in ids {
            s.get_mut_player_unsafe(id).stats.skills.remove(&Skill::Leap);
        }
    };
    let mut nobody_down = Vec::new();
    let mut four_down = Vec::new();
    let mut leap = Vec::new();
    for s in &states {
        let mut s = s.clone();
        strip_leap(&mut s);
        // Nobody down: stand everyone up.
        let mut up = s.clone();
        let ids: Vec<PlayerID> = up.get_players_on_pitch().map(|p| p.id).collect();
        for &id in &ids {
            up.get_mut_player_unsafe(id).status = PlayerStatus::Up;
        }
        for id in movers(&up) {
            nobody_down.push(Sample { state: up.clone(), id });
        }
        // Four down: knock down four random players, then path every Standing mover left.
        let mut down = up.clone();
        for _ in 0..4 {
            let id = ids[rng.gen_range(0..ids.len())];
            down.get_mut_player_unsafe(id).status = PlayerStatus::Down;
        }
        for id in movers(&down) {
            four_down.push(Sample { state: down.clone(), id });
        }
        // Leap: the mover has Leap, nobody down.
        for id in movers(&up) {
            let mut l = up.clone();
            l.get_mut_player_unsafe(id).stats.give_skill(Skill::Leap);
            leap.push(Sample { state: l, id });
        }
    }
    println!(
        "{} states, samples: nobody down {}, four down {}, Leap {}",
        states.len(),
        nobody_down.len(),
        four_down.len(),
        leap.len()
    );
    println!("| scenario | samples | off µs/call | on µs/call | on/off | paths using a jump |");
    println!("|---|---|---|---|---|---|");
    for (name, samples) in [
        ("nobody down", &mut nobody_down),
        ("four down", &mut four_down),
        ("mover has Leap", &mut leap),
    ] {
        let mut best_off = Duration::MAX;
        let mut best_on = Duration::MAX;
        let mut jumps = 0;
        for _ in 0..7 {
            best_off = best_off.min(time_paths(samples, false).0);
            let (t, j) = time_paths(samples, true);
            best_on = best_on.min(t);
            jumps = j;
        }
        let per = |d: Duration| d.as_secs_f64() * 1e6 / samples.len() as f64;
        println!(
            "| {name} | {} | {:.2} | {:.2} | {:.3} | {} |",
            samples.len(),
            per(best_off),
            per(best_on),
            best_on.as_secs_f64() / best_off.as_secs_f64(),
            jumps
        );
    }
}
