//! Plan 070: the same search on the same position must give the same result.
//!
//! Two copies of one bot play a drive in lockstep (one bot per side and copy, so each keeps its own
//! reused tree, as in a drive game); at every decision the two searches' root summaries (every root
//! child's visits and Q, and the move) must be identical, and the DAG each search leaves behind
//! must pass [`MctsBot::audit_dag`] (every edge deeper and listed in the child's parent set).
//!
//! `DET_SET=<positions.json>` runs the investigation on a real drive set instead of the built-in
//! positions (`DET_COUNT`, `DET_ITERS`, `DET_STEPS`, `DET_PRESET` relative to the repo root).

use std::path::PathBuf;

use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{DiceMode, GameState};
use botbowl_engine::core::model::BoardDims;
use botbowl_mcts::{DagAudit, MctsBot, MctsConfig, SearchBudget};
use botbowl_play::bots::load_mcts_config;
use botbowl_play::drives::{position_state, PositionSet};
use botbowl_play::generate::RandomStartBias;
use rand::SeedableRng;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn env_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn bot(iters: usize, config: MctsConfig) -> MctsBot {
    let mut b = MctsBot::with_budget_and_config(SearchBudget::Iterations(iters), config).with_workers(1);
    b.set_seed(rand_chacha::ChaCha8Rng::seed_from_u64(7));
    b
}

type Fingerprint = Option<(String, Vec<(String, u32, Option<i64>)>)>;

/// The last search's move and every root child's statistics; `None` before the bot's first search
/// (a forced move does not search).
fn fingerprint(b: &MctsBot) -> Fingerprint {
    let s = b.last_search()?;
    let mut c: Vec<_> = s
        .children
        .iter()
        .map(|e| (format!("{:?}", e.action), e.stats.visits, e.stats.q_home))
        .collect();
    c.sort();
    Some((format!("{:?}", s.chosen), c))
}

#[derive(Debug, Default)]
struct Outcome {
    decisions: usize,
    /// Step of the first decision whose two searches differ in any root statistic.
    first_diff: Option<usize>,
    /// ... and of the first whose moves differ.
    first_move_diff: Option<usize>,
    audit: DagAudit,
}

fn add(a: &mut DagAudit, b: DagAudit) {
    a.nodes += b.nodes;
    a.edges += b.edges;
    a.depth_violations += b.depth_violations;
    a.unlisted_parent += b.unlisted_parent;
    a.checked += b.checked;
    a.stale_value += b.stale_value;
    a.stale_visits_only += b.stale_visits_only;
}

/// Play one drive in lockstep with two copies of the bot; stop at the first differing move.
fn lockstep(mut state: GameState, dice: u64, iters: usize, config: MctsConfig, steps: usize) -> Outcome {
    state.set_seed(dice);
    state.set_dice_mode(DiceMode::RollDice);
    state.set_logging_state(false);
    let home = state.available_actions.team;
    let mut bots = [
        bot(iters, config),
        bot(iters, config),
        bot(iters, config),
        bot(iters, config),
    ];
    let mut out = Outcome::default();
    for step in 0..steps {
        if state.info.game_over {
            break;
        }
        let Some(team) = state.available_actions.team else {
            break;
        };
        let side = usize::from(Some(team) != home);
        let (a, rest) = bots.split_at_mut(2);
        let (x, y) = (&mut a[side], &mut rest[side]);
        let ma = x.get_action(&state);
        let mb = y.get_action(&state);
        out.decisions += 1;
        for b in [&*x, &*y] {
            if let Some(au) = b.audit_dag() {
                if std::env::var_os("DET_VERBOSE").is_some() {
                    eprintln!(
                        "  step {step} reuse {:?} {au:?}",
                        b.last_search().map(|s| &s.reuse.outcome)
                    );
                }
                add(&mut out.audit, au);
            }
        }
        if out.first_diff.is_none() && fingerprint(x) != fingerprint(y) {
            out.first_diff = Some(step);
        }
        if ma != mb {
            out.first_move_diff = Some(step);
            break;
        }
        state.step(ma).expect("engine step");
        for b in bots.iter_mut() {
            b.release_stale_tree(&state);
        }
    }
    out
}

fn positions(builtin: u64) -> Vec<(GameState, u64)> {
    if let Ok(path) = std::env::var("DET_SET") {
        let set = PositionSet::load(&path).expect("position set");
        let dims = set.board_dims().expect("board");
        let count = env_or("DET_COUNT", 8usize);
        return set
            .positions()
            .into_iter()
            .take(count)
            .enumerate()
            .map(|(i, seed)| (position_state(&set.bias, dims, seed), i as u64))
            .collect();
    }
    let Ok(dims) = BoardDims::try_new(14, 7, 4) else {
        return Vec::new();
    };
    (0..builtin)
        .map(|seed| (position_state(&RandomStartBias::default(), dims, seed), seed))
        .collect()
}

#[derive(Debug, Default)]
struct Totals {
    decisions: usize,
    /// Positions with a decision whose two searches differ in any root statistic.
    diffs: usize,
    /// ... whose moves differ.
    move_diffs: usize,
    audit: DagAudit,
}

fn run(preset: &str, iters: usize, steps: usize, builtin: u64) -> Totals {
    let mut config = load_mcts_config(&repo().join(preset)).expect("preset").config;
    if std::env::var("DET_NO_REUSE").is_ok_and(|v| !v.is_empty()) {
        config.tree_reuse = false;
    }
    let mut t = Totals::default();
    for (i, (state, dice)) in positions(builtin).into_iter().enumerate() {
        let o = lockstep(state, dice, iters, config, steps);
        eprintln!(
            "position {i}: decisions {} first_diff {:?} first_move_diff {:?} audit {:?}",
            o.decisions, o.first_diff, o.first_move_diff, o.audit
        );
        t.decisions += o.decisions;
        t.diffs += usize::from(o.first_diff.is_some());
        t.move_diffs += usize::from(o.first_move_diff.is_some());
        add(&mut t.audit, o.audit);
    }
    eprintln!("TOTAL {t:?}");
    t
}

/// The gate: under the shipped preset (`tie_break = "hash"`, seeded by each node's state) and
/// under `mover`, two copies of the bot agree on every root statistic of every decision, and the
/// DAGs they leave keep every edge strictly deeper and mirrored in the child's parent set. (Stale
/// backups are counted, not asserted: a descent whose backup is withheld below a child leaves that
/// child's visit bump unweighted above it, the documented visit lag.)
#[test]
fn two_copies_of_a_bot_agree_on_every_root_statistic() {
    for preset in ["cfgs/chance_all_partial.toml", "cfgs/chance_all_partial_det.toml"] {
        let t = run(preset, 150, 16, 2);
        assert_eq!(
            (t.audit.depth_violations, t.audit.unlisted_parent),
            (0, 0),
            "{preset}: {t:?}"
        );
        assert_eq!((t.diffs, t.move_diffs), (0, 0), "{preset}: {t:?}");
    }
}

/// The investigation harness behind exp084: `DET_SET=<positions.json> DET_COUNT DET_ITERS
/// DET_STEPS DET_PRESET` (relative to the repo root), `DET_VERBOSE=1` for the audit of every search.
#[test]
#[ignore = "investigation harness (plan 070); DET_SET=… for a real drive set"]
fn the_same_search_twice_gives_the_same_result() {
    let preset = std::env::var("DET_PRESET").unwrap_or_else(|_| "cfgs/chance_all_partial.toml".into());
    let t = run(&preset, env_or("DET_ITERS", 300usize), env_or("DET_STEPS", 40usize), 4);
    assert_eq!((t.audit.depth_violations, t.audit.unlisted_parent), (0, 0), "{t:?}");
    assert_eq!((t.diffs, t.move_diffs), (0, 0), "{t:?}");
}
