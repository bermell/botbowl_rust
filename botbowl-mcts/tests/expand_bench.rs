//! Microbench for MCTS leaf-expansion cost (plan 011 step 1).
//!
//! Two `#[ignore]`d tests:
//!   * `expand_bench_main` — wall-clock per-op timings:
//!       (a) `MctsBot::get_action` (full search).
//!       (b) `GameState::clone` raw cost.
//!       (c) `BloodBowlDynamics::apply_action(StartMove)` (clone + micro_step
//!           + pathing).
//!   * `expand_bench_call_counts` — algorithmic shape: wraps
//!     `BloodBowlDynamics` in a `CountingDynamics<_>` that increments
//!     atomic counters in each `GameDynamics` method, then drives a
//!     `Tree` directly for N iterations. Reports per `tree.step()` the
//!     average count of `apply_action`, `available_actions`,
//!     `select_node`, `score_leaf`, `backprop_scores` calls.
//!
//! Both seeded with `0xCAFE_1234` (same seed as `score_td_easy.rs` /
//! `parallel_bench.rs`).
//!
//! Run:
//!   cargo test --release -p botbowl-mcts --test expand_bench \
//!       -- --ignored --nocapture

use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::Lecture;
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_engine::core::table::PosAT;
use botbowl_mcts::dynamics::BbScore;
use botbowl_mcts::{BbAction, BbPlayer, BloodBowlDynamics, MctsBot};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use recon_mcts::{GameDynamics, HashOnly, SearchTree, SelectNodeState, Tree};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SEED: u64 = 0xCAFE_1234;

fn build_score_td_easy() -> GameState {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    ScoreTdEasy::new().setup(&mut rng)
}

fn build_full_teams_turn_start() -> GameState {
    // 11 home + 11 away in a plausible post-kickoff formation, ball with
    // a home carrier at (13, 8). `GameStateBuilder::build` runs the
    // kickoff sequence then drops these players onto the pitch and
    // hands turn 1 to Home. Goal is the 30+-legal-actions fan-out plan
    // 011 targets — every fielded home player is a StartMove candidate,
    // plus StartBlitz / StartHandoff / StartPass / StartFoul / EndTurn.
    let home_players = [
        (13, 8),
        (13, 7),
        (13, 9),
        (13, 5),
        (13, 11),
        (13, 3),
        (13, 13),
        (10, 8),
        (10, 6),
        (10, 10),
        (7, 8),
    ];
    let away_players = [
        (14, 8),
        (14, 7),
        (14, 9),
        (14, 5),
        (14, 11),
        (14, 3),
        (14, 13),
        (17, 8),
        (17, 6),
        (17, 10),
        (20, 8),
    ];
    let mut state = GameStateBuilder::new()
        .set_state(BuilderState::Turn { turn: 1 })
        .add_home_players(&home_players)
        .add_away_players(&away_players)
        .add_ball((13, 8))
        .build();
    state.set_seed(SEED);
    state.set_dice_mode(DiceMode::RollDice);
    state
}

fn prep_for_search(state: &mut GameState) {
    // Mirror what `MctsBot::get_action` does up front (dynamics.rs:499)
    // so we don't over-attribute to log-Vec cloning during the bench.
    state.set_logging_state(false);
    state.clear_log();
}

fn first_start_move(state: &GameState) -> Option<EngineAction> {
    state
        .available_actions
        .get_all()
        .into_iter()
        .find(|a| matches!(a, EngineAction::Positional(PosAT::StartMove, _)))
}

fn bench_get_action(label: &str, state: &GameState, trials: usize, iters: usize) {
    let mut total = Duration::ZERO;
    let mut sink: u64 = 0;
    for _ in 0..trials {
        let mut bot = MctsBot::new(iters);
        let t0 = Instant::now();
        let action = bot.get_action(state);
        total += t0.elapsed();
        sink = sink.wrapping_add(format!("{action:?}").len() as u64);
    }
    let per_trial = total.as_secs_f64() / trials as f64;
    let per_iter_us = per_trial * 1e6 / iters as f64;
    eprintln!(
        "EXPAND_BENCH {label}/get_action_ms={:.2} trials={trials} iters={iters} \
         per_iter_us={:.2} sink={sink}",
        per_trial * 1e3,
        per_iter_us,
    );
}

fn bench_clone(label: &str, state: &GameState, iters: usize) {
    let t0 = Instant::now();
    let mut sink: u64 = 0;
    for _ in 0..iters {
        let c = state.clone();
        sink = sink.wrapping_add(c.info.home_turn as u64);
        std::hint::black_box(c);
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "EXPAND_BENCH {label}/clone_ns={:.0} iters={iters} sink={sink}",
        elapsed.as_nanos() as f64 / iters as f64,
    );
}

fn bench_apply_start_move(label: &str, state: &GameState, action: EngineAction, iters: usize) {
    // Match `MctsBot::get_action`'s dice mode so we exercise the same
    // code path the tree actually traverses.
    let mut base = state.clone();
    base.set_dice_mode(DiceMode::RegisterRolls);

    let dynamics = BloodBowlDynamics::default();
    let bb_action = BbAction::Player(action);
    let t0 = Instant::now();
    let mut applied: u64 = 0;
    for _ in 0..iters {
        let s = base.clone();
        let out = dynamics.apply_action(s, &bb_action);
        if out.is_some() {
            applied += 1;
        }
        std::hint::black_box(out);
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "EXPAND_BENCH {label}/apply_start_move_ns={:.0} iters={iters} applied={applied}",
        elapsed.as_nanos() as f64 / iters as f64,
    );
}

fn run_scenario(label: &str, mut state: GameState) {
    prep_for_search(&mut state);
    let legal = state.available_actions.get_all().len();
    let team = state.available_actions.team;
    eprintln!("EXPAND_BENCH {label}/legal_actions={legal} team={:?}", team);

    bench_get_action(label, &state, 20, 1000);
    bench_clone(label, &state, 100_000);
    match first_start_move(&state) {
        Some(a) => bench_apply_start_move(label, &state, a, 10_000),
        None => eprintln!("EXPAND_BENCH {label}/apply_start_move=SKIPPED (no StartMove legal)"),
    }
}

#[test]
#[ignore = "manual wall-clock bench — run with --ignored"]
fn expand_bench_main() {
    eprintln!("EXPAND_BENCH seed={SEED:#x}");
    run_scenario("score_td_easy", build_score_td_easy());
    run_scenario("full_teams", build_full_teams_turn_start());
}

// ─── Call-counting wrapper ────────────────────────────────────────────
//
// Wraps `BloodBowlDynamics` and atomically bumps a counter on each
// `GameDynamics` method. Drives `Tree` directly (skipping `MctsBot`) so
// we can observe per-`tree.step()` averages for:
//   * `apply_action`     — total times the engine moved a state
//   * `available_actions`— times the engine enumerated legal actions
//   * `select_node`      — times PUCT was invoked (≈ descent depth)
//   * `score_leaf`       — times a leaf got scored (≈ N expansions × N children)
//   * `backprop_scores`  — times scores were propagated
//
// This is the algorithmic-shape data plan 011 needs to decide L1 vs L3
// vs cheap-wins — CPU-% from samply tells you where time goes, this
// tells you how often each callback fires.

#[derive(Debug, Default)]
struct Counters {
    apply_action: AtomicU64,
    available_actions: AtomicU64,
    select_node: AtomicU64,
    score_leaf: AtomicU64,
    backprop_scores: AtomicU64,
}

#[derive(Debug)]
struct CountingDynamics {
    inner: BloodBowlDynamics,
    counters: Arc<Counters>,
}

impl GameDynamics for CountingDynamics {
    type Player = BbPlayer;
    type State = GameState;
    type Action = BbAction;
    type Score = BbScore;
    type ActionIter = Vec<(BbPlayer, BbAction)>;

    fn available_actions(&self, player: &Self::Player, state: &Self::State) -> Option<Self::ActionIter> {
        self.counters.available_actions.fetch_add(1, Ordering::Relaxed);
        self.inner.available_actions(player, state)
    }

    fn apply_action(&self, state: Self::State, action: &Self::Action) -> Option<Self::State> {
        self.counters.apply_action.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_action(state, action)
    }

    fn select_node<II, Q, A>(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        parent_node_state: &Self::State,
        purpose: SelectNodeState,
        scores_and_actions: II,
    ) -> Self::Action
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        Q: Deref<Target = Option<Self::Score>>,
        A: Deref<Target = Self::Action>,
    {
        self.counters.select_node.fetch_add(1, Ordering::Relaxed);
        self.inner.select_node(
            parent_score,
            parent_player,
            parent_node_state,
            purpose,
            scores_and_actions,
        )
    }

    fn backprop_scores<II, Q, A>(
        &self,
        player: &Self::Player,
        score_current: Option<&Self::Score>,
        child_scores_and_actions: II,
    ) -> Option<Self::Score>
    where
        Self: Sized,
        II: Clone + IntoIterator<Item = (Q, A)>,
        A: Deref<Target = Self::Action>,
        Q: Deref<Target = Self::Score>,
    {
        self.counters.backprop_scores.fetch_add(1, Ordering::Relaxed);
        self.inner
            .backprop_scores(player, score_current, child_scores_and_actions)
    }

    fn score_leaf(
        &self,
        parent_score: Option<&Self::Score>,
        parent_player: &Self::Player,
        state: &Self::State,
    ) -> Option<Self::Score> {
        self.counters.score_leaf.fetch_add(1, Ordering::Relaxed);
        self.inner.score_leaf(parent_score, parent_player, state)
    }
}

fn player_for_root(state: &GameState) -> BbPlayer {
    use botbowl_engine::core::model::TeamType;
    if state.pending_roll.is_some() {
        return BbPlayer::Chance;
    }
    match state.available_actions.team {
        Some(TeamType::Home) => BbPlayer::Home,
        Some(TeamType::Away) => BbPlayer::Away,
        None => BbPlayer::Chance,
    }
}

fn run_call_counts(label: &str, mut state: GameState, iters: usize) {
    // Match `MctsBot::get_action`'s setup so the tree behaves identically.
    state.set_dice_mode(DiceMode::RegisterRolls);
    state.set_logging_state(false);
    state.clear_log();

    let counters = Arc::new(Counters::default());
    let dynamics = CountingDynamics {
        inner: BloodBowlDynamics::default(),
        counters: Arc::clone(&counters),
    };
    let root_player = player_for_root(&state);
    let tree = Tree::new(dynamics, HashOnly, root_player, state);

    let t0 = Instant::now();
    for _ in 0..iters {
        tree.step();
    }
    let elapsed = t0.elapsed();

    let aa = counters.apply_action.load(Ordering::Relaxed);
    let av = counters.available_actions.load(Ordering::Relaxed);
    let sn = counters.select_node.load(Ordering::Relaxed);
    let sl = counters.score_leaf.load(Ordering::Relaxed);
    let bp = counters.backprop_scores.load(Ordering::Relaxed);
    let per_step_us = elapsed.as_nanos() as f64 / iters as f64 / 1e3;
    let f = |n: u64| n as f64 / iters as f64;
    eprintln!(
        "EXPAND_COUNTS {label}/iters={iters} per_step_us={per_step_us:.2} \
         apply_action/step={:.2} avail_actions/step={:.2} \
         select_node/step={:.2} score_leaf/step={:.2} backprop/step={:.2}",
        f(aa),
        f(av),
        f(sn),
        f(sl),
        f(bp),
    );
    eprintln!(
        "EXPAND_COUNTS {label}/totals apply_action={aa} avail_actions={av} \
         select_node={sn} score_leaf={sl} backprop_scores={bp}"
    );
}

#[test]
#[ignore = "manual call-counting bench — run with --ignored"]
fn expand_bench_call_counts() {
    eprintln!("EXPAND_COUNTS seed={SEED:#x}");
    run_call_counts("score_td_easy", build_score_td_easy(), 1000);
    run_call_counts("full_teams", build_full_teams_turn_start(), 1000);
}

#[test]
#[ignore = "manual samply target — single-threaded, long-running for profiler"]
fn expand_bench_for_samply() {
    // Same workload as `expand_bench_call_counts` but with enough iters
    // for samply (1 kHz default) to capture meaningful samples. ~1-2 s
    // total wall-clock; runs single-threaded so all samples attribute
    // cleanly to one thread.
    //
    // Recipe:
    //   RUSTFLAGS="-C debuginfo=line-tables-only" cargo test --release \
    //       -p botbowl-mcts --test expand_bench --no-run
    //   samply record --save-only -o prof.json --rate 4000 -- \
    //       target/release/deps/expand_bench-* expand_bench_for_samply \
    //       --ignored --nocapture
    eprintln!("EXPAND_COUNTS_SAMPLY seed={SEED:#x}");
    run_call_counts("score_td_easy", build_score_td_easy(), 200_000);
    run_call_counts("full_teams", build_full_teams_turn_start(), 200_000);
}
