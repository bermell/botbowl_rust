//! Why a block-die decision can never reuse the search tree.
//!
//! Measured over 16 ladder games (plan 043 telemetry): `FollowUp`, `DodgeProc` and `GfiProc` reuse
//! the tree 100% of the time, `Push` 78%, `MoveAction` 65% — and `Block` **0.8%**, 1 decision in
//! 128, every failure a `LookupMiss`. That is not bad luck; it is structural, and this file pins
//! the mechanism so nobody has to rediscover it.
//!
//! **Two mechanisms, and between them they cover every miss** (26/26 measured, 14 + 12):
//!
//! 1. **The search walks past the choice.** `apply_action`'s quiescent loop calls
//!    `scripted::scripted_player_pick`, whose first arm is `block_dice::scripted_pick` — so as soon
//!    as the dice land the loop picks a die and advances. No post-roll `Block` node is created at
//!    all, and the only `Block` states in the registry are pre-roll chance nodes
//!    (`state: Init, roll: [None, None, None]`).
//! 2. **Representative dice.** When the roll *is* left to the search — the attacker-Block-only
//!    `[Pow, BothDown]` case that `scripted_pick` declines — a post-roll node does exist, but
//!    `roll_outcomes::block_outcomes` groups the 6^n face combinations by *effect* and emits one
//!    child carrying a **canonical** dice array. The game rolls actual faces, reaches the same
//!    effect, and stores those. `Block` keeps `roll` in its procedure state and `GameState`
//!    equality compares the procedure stack, so the states are unequal.
//!
//! The *real game* walks past nothing: the engine stops and asks the bot which die to use. So
//! `get_action` is called on a state the previous search could not have built — a guaranteed
//! registry miss and a guaranteed full tree rebuild. `available_actions` says as much in its own
//! comment ("the *root* state passed to `MctsBot::get_action` is itself mid-block-die").
//!
//! **The rebuild is not waste, and that is the uncomfortable part.** At such a root the bot
//! searches the die choice properly (mean fan 2.2) and picks a *different* die from
//! `scripted_pick` **51% of the time**. So inside the tree the bot models its own future block-die
//! choices as the scripted one, while in play it overrules that script in half of them — the
//! search is valuing block outcomes under a policy it does not follow.
//! `the_bot_searches_a_choice_it_models_as_scripted` measures it.
//!
//! ```sh
//! cargo test --release -p botbowl-mcts --test block_reuse -- --ignored --nocapture
//! ```

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::{BoardDims, Position, TEAM_SIZE};
use botbowl_mcts::{block_dice, MctsBot, ReuseOutcome, SearchBudget};

const W: i8 = 16;
const H: i8 = 9;
const PLAYERS: usize = 3;

/// Two attackers adjacent to a ball-carrying defender, so blocks come up early and often.
fn blocking_state(seed: u64) -> Option<GameState> {
    if (botbowl_engine::core::model::WIDTH as i8) < W
        || (botbowl_engine::core::model::HEIGHT as i8) < H
        || TEAM_SIZE < PLAYERS
    {
        return None;
    }
    let carrier = Position::new((8, 4));
    let mut state = GameStateBuilder::new()
        .with_board_dims(BoardDims::new(W, H, PLAYERS))
        .add_home_player(Position::new((7, 4)))
        .add_home_player(Position::new((7, 5)))
        .add_away_player(carrier)
        .add_away_player(Position::new((9, 5)))
        .add_ball_pos(carrier)
        .build();
    state.set_seed(seed);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    Some(state)
}

fn is_post_roll_block(state: &GameState) -> bool {
    state
        .proc_stack_peek()
        .is_some_and(|p| format!("{p:?}").starts_with("Block(") && !format!("{p:?}").contains("state: Init"))
}

/// Why the state the game reaches is never the state the search built.
///
/// Two mechanisms, and each miss is one or the other:
///
/// * **walked past** — `scripted_pick` resolved the die inside the quiescent loop, so no post-roll
///   `Block` node exists at all;
/// * **representative dice** — the roll *was* left to the search (the attacker-Block-only
///   `[Pow, BothDown]` case), so a post-roll node exists, but it carries the canonical dice array
///   `block_outcomes` emitted rather than the faces the game actually rolled, and `Block` keeps
///   `roll` in its procedure state, which `GameState`'s equality compares.
#[test]
#[ignore = "diagnostic: why Block never reuses the tree, ~1 min"]
fn a_block_miss_is_either_walked_past_or_a_representative_roll() {
    let mut misses = 0usize;
    let mut walked_past = 0usize;
    let mut representative = 0usize;
    let mut unexplained = 0usize;
    let mut shown = 0usize;

    for seed in 0..40u64 {
        let Some(mut state) = blocking_state(seed) else {
            eprintln!("board too small for this build");
            return;
        };
        let mut bot = MctsBot::new(SearchBudget::Iterations(800)).with_workers(1);
        let mut prev_dag: Vec<GameState> = Vec::new();

        for _ in 0..20 {
            if state.info.game_over || state.available_actions.team.is_none() {
                break;
            }
            let action = bot.get_action(&state);
            let s = bot.last_search().expect("summary");

            if s.reuse.proc.as_deref() == Some("Block") && s.reuse.outcome == ReuseOutcome::LookupMiss {
                misses += 1;
                let actual = format!("{:?}", state.proc_stack_peek().unwrap());
                let post_roll: Vec<String> = prev_dag
                    .iter()
                    .filter(|d| is_post_roll_block(d))
                    .map(|d| format!("{:?}", d.proc_stack_peek().unwrap()))
                    .collect();

                // The registry said miss; confirm the state really is absent.
                assert!(!prev_dag.iter().any(|d| d == &state), "seed {seed}: miss but present");

                if post_roll.is_empty() {
                    walked_past += 1;
                } else if post_roll.iter().all(|p| *p != actual) {
                    representative += 1;
                    if shown < 2 {
                        shown += 1;
                        eprintln!("\nseed {seed}: the roll was left to the search, but");
                        eprintln!("  game rolled  {actual}");
                        for p in post_roll.iter().take(2) {
                            eprintln!("  search built {p}");
                        }
                    }
                } else {
                    unexplained += 1;
                }
            }

            prev_dag = bot.dag_states().unwrap_or_default();
            if state.step(action).is_err() {
                break;
            }
        }
        if misses >= 25 {
            break;
        }
    }

    eprintln!("\nBlock lookup misses                       : {misses}");
    eprintln!("  walked past by scripted_pick (no node)  : {walked_past}");
    eprintln!("  node exists, representative dice differ : {representative}");
    eprintln!("  unexplained                             : {unexplained}");

    assert!(misses > 0, "the scenario never blocked");
    assert_eq!(
        unexplained, 0,
        "every Block miss should be one of the two known mechanisms; {unexplained} were neither, \
         so there is a third cause worth finding"
    );
}

/// Does the rebuild buy anything? The bot searches the die choice at the root while modelling its
/// future self as taking `scripted_pick`'s die. If the two always agree, the rebuild is pure cost.
#[test]
#[ignore = "diagnostic: does searching a block die ever beat the scripted pick? ~1 min"]
fn the_bot_searches_a_choice_it_models_as_scripted() {
    let mut decisions = 0usize;
    let mut agreed = 0usize;
    let mut fan_total = 0usize;
    let mut fan_one = 0usize;

    for seed in 0..40u64 {
        let Some(mut state) = blocking_state(seed) else {
            return;
        };
        let mut bot = MctsBot::new(SearchBudget::Iterations(800)).with_workers(1);

        for _ in 0..20 {
            if state.info.game_over || state.available_actions.team.is_none() {
                break;
            }
            // Ask the script *before* the bot, on the same state.
            let scripted = block_dice::scripted_pick(&state);
            let action = bot.get_action(&state);
            let s = bot.last_search().expect("summary");

            if let Some(scripted) = scripted {
                decisions += 1;
                fan_total += s.children.len();
                if s.children.len() <= 1 {
                    fan_one += 1;
                }
                if scripted == action {
                    agreed += 1;
                }
            }

            if state.step(action).is_err() {
                break;
            }
        }
        if decisions >= 40 {
            break;
        }
    }

    eprintln!("\nblock-die decisions the real game asked about : {decisions}");
    if decisions > 0 {
        eprintln!(
            "  search agreed with scripted_pick            : {agreed} ({:.0}%)",
            100.0 * agreed as f64 / decisions as f64
        );
        eprintln!(
            "  mean root fan                               : {:.1}",
            fan_total as f64 / decisions as f64
        );
        eprintln!("  roots with a single candidate               : {fan_one}");
    }
    assert!(decisions > 0, "the scenario never reached a block-die choice");
}
