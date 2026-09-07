//! Scratch probe (plan 032 #11): random-vs-random drives, split by kicking team.
//! Run: BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4 CARGO_TARGET_DIR=target/14x7 \
//!      cargo test --release -p botbowl-engine --test seat_probe -- --ignored --nocapture
use std::collections::BTreeMap;

use botbowl_engine::bots::{Bot, RandomBot};
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameStateBuilder};
use botbowl_engine::core::model::{Action, BallState, TeamType};
use botbowl_engine::core::table::SimpleAT;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Default, Debug)]
struct Drive {
    n: u64,
    touchback: u64,
    caught: u64,
    ground: u64,
    td_kicker: u64,
    td_receiver: u64,
    land_x: BTreeMap<i8, u64>,
    land_y: BTreeMap<i8, u64>,
    /// (kicker roles, receiver roles) fielded at this kickoff
    lineups: BTreeMap<(String, String), u64>,
}

fn roles(state: &botbowl_engine::core::gamestate::GameState, team: TeamType) -> String {
    let mut v: Vec<String> = state
        .get_players_on_pitch_in_team(team)
        .map(|p| format!("{:?}", p.stats.role).chars().next().unwrap().to_string())
        .collect();
    v.sort();
    v.concat()
}

#[test]
#[ignore]
fn seat_probe() {
    let games: u64 = std::env::var("PROBE_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    // key: (half, kicking team)
    let mut drives: BTreeMap<(u8, bool), Drive> = BTreeMap::new();
    let mut unfinished = 0u64;
    for g in 0..games {
        let seed = 33200000 + g;
        let mut state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
        state.set_seed(seed);
        state.set_dice_mode(DiceMode::RollDice);
        state.set_logging_state(false);
        let mut home = RandomBot::new();
        let mut away = RandomBot::new();
        home.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xA));
        away.set_seed(ChaCha8Rng::seed_from_u64(seed ^ 0xB));
        let mut steps = 0u32;
        let mut cur: Option<(u8, bool)> = None;
        let (mut hs, mut as_) = (0u8, 0u8);
        while !state.info.game_over && steps < 20000 {
            let team = match state.available_actions.team {
                Some(t) => t,
                None => break,
            };
            let action = if team == TeamType::Home {
                home.get_action(&state)
            } else {
                away.get_action(&state)
            };
            let kicked = matches!(action, Action::Simple(SimpleAT::KickoffAimMiddle));
            state.step(action).expect("step");
            steps += 1;
            if kicked {
                let key = (state.info.half, state.info.kicking_this_drive == TeamType::Home);
                cur = Some(key);
                let d = drives.entry(key).or_default();
                d.n += 1;
                let k = state.info.kicking_this_drive;
                *d.lineups
                    .entry((
                        roles(&state, k),
                        roles(&state, botbowl_engine::core::model::other_team(k)),
                    ))
                    .or_default() += 1;
                let top = state.proc_stack_top().unwrap_or("");
                if top == "Touchback" {
                    d.touchback += 1;
                }
                match state.ball {
                    BallState::Carried(_) => d.caught += 1,
                    BallState::OnGround(_) => d.ground += 1,
                    _ => {}
                }
                if let Some(p) = state.get_ball_position() {
                    *d.land_x.entry(p.x).or_default() += 1;
                    *d.land_y.entry(p.y).or_default() += 1;
                }
            }
            if state.home.score != hs || state.away.score != as_ {
                if let Some(key) = cur {
                    let d = drives.entry(key).or_default();
                    let scorer = if state.home.score != hs {
                        TeamType::Home
                    } else {
                        TeamType::Away
                    };
                    if (scorer == TeamType::Home) == key.1 {
                        d.td_kicker += 1
                    } else {
                        d.td_receiver += 1
                    }
                }
                hs = state.home.score;
                as_ = state.away.score;
            }
        }
        if !state.info.game_over {
            unfinished += 1;
        }
    }
    println!("games={games} unfinished={unfinished}");
    for (k, d) in &drives {
        println!(
            "half {} kicker_home {}: drives {} touchback {:.3} caught {:.3} ground {:.3} TD kicker {} receiver {}",
            k.0,
            k.1,
            d.n,
            d.touchback as f64 / d.n as f64,
            d.caught as f64 / d.n as f64,
            d.ground as f64 / d.n as f64,
            d.td_kicker,
            d.td_receiver
        );
        println!("   land_x {:?}", d.land_x);
        println!("   land_y {:?}", d.land_y);
        println!("   lineups (kicker, receiver) {:?}", d.lineups);
    }
}
