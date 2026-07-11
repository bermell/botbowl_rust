//! Microbench for MCTS leaf-expansion cost (plan 011 step 1).
//!
//! Two `#[ignore]`d tests:
//!   * `expand_bench_main` — wall-clock per-op timings:
//!     (a) `MctsBot::get_action` (full search).
//!     (b) `GameState::clone` raw cost.
//!     (c) `BloodBowlDynamics::apply_action(StartMove)` (clone + micro_step
//!      + pathing).
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
//! These are minutes-long, many-thread, high-memory profiling runs — NOT
//! part of the routine `cargo test --ignored` bot benchmark suite. They are
//! gated behind the `expand_bench` cargo feature (see this crate's
//! `Cargo.toml`), so a plain `cargo test --release -- --ignored` compiles
//! this file to zero tests and never runs them. Run explicitly with:
//!
//!   cargo test --release -p botbowl-mcts --features expand_bench \
//!       --test expand_bench -- --ignored --nocapture
#![cfg(feature = "expand_bench")]

use botbowl_curriculum::lectures::score_td::ScoreTdEasy;
use botbowl_curriculum::Lecture;
use botbowl_engine::bots::Bot;
use botbowl_engine::core::gamestate::{BuilderState, DiceMode, GameState, GameStateBuilder};
use botbowl_engine::core::model::Action as EngineAction;
use botbowl_engine::core::table::PosAT;
use botbowl_mcts::dynamics::{BbScore, HorizonAnchor};
use botbowl_mcts::{BbAction, BbPlayer, BloodBowlDynamics, MctsBot, SearchBudget};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use recon_mcts::{GameDynamics, SearchTree, SelectNodeState, StoreState, Tree};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SEED: u64 = 0xCAFE_1234;

/// These are `#[ignore]`d manual benchmarks meant for `--release` (see the
/// module doc — every recipe passes `--release`). In a **debug** build
/// `StoreState` keeps a full `GameState` per DAG node, so they are slow and
/// memory-hungry regardless of iteration count (`full_teams@10k` was
/// OOM-killed) — and a debug build is never the real measurement run anyway.
/// So skip them entirely in debug: `cargo test --ignored` stays instant and
/// the real numbers come from `cargo test --release ... --ignored`.
///
/// Call at the top of each bench: `if skip_unless_release("name") { return; }`.
fn skip_unless_release(name: &str) -> bool {
    if cfg!(debug_assertions) {
        eprintln!("SKIP {name}: expand_bench is a --release-only benchmark (debug build)");
        true
    } else {
        false
    }
}

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
        .get_all_actions()
        .into_iter()
        .find(|a| matches!(a, EngineAction::Positional(PosAT::StartMove, _)))
}

fn bench_get_action(label: &str, state: &GameState, trials: usize, iters: usize) {
    let mut total = Duration::ZERO;
    let mut sink: u64 = 0;
    for _ in 0..trials {
        let mut bot = MctsBot::new(SearchBudget::Iterations(iters));
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
    let bb_action = BbAction::player(action, 1.0);
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

/// Run `f` on a thread with an explicit 16 MB stack, mirroring the worker
/// threads `MctsBot::get_action` spawns (dynamics.rs `WORKER_STACK_SIZE`).
///
/// `recon_mcts` tears the search DAG down *recursively*: `Node::on_drop`
/// (recon_mcts/src/tree.rs) drains a node's children, and dropping each
/// child re-enters `on_drop`, so the teardown recurses to the DAG's depth.
/// The cargo-test worker thread's default ~2 MB stack overflows once a deep
/// chain forms — which these benches do whenever a tree is dropped on the
/// test thread (a `Tree` driven directly, or an `MctsBot`'s reused/cached
/// tree). Production never trips this because it only drops trees on/after
/// its own big-stack workers. 16 MB dwarfs any depth these benches reach.
fn with_big_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn_scoped(s, f)
            .expect("failed to spawn big-stack bench thread")
            .join()
            .expect("bench thread panicked")
    })
}

fn run_scenario(label: &str, mut state: GameState) {
    prep_for_search(&mut state);
    let legal = state.get_all_actions().len();
    let team = state.available_actions.team;
    eprintln!("EXPAND_BENCH {label}/legal_actions={legal} team={:?}", team);

    with_big_stack(|| {
        bench_get_action(label, &state, 20, 1000);
        bench_clone(label, &state, 100_000);
        match first_start_move(&state) {
            Some(a) => bench_apply_start_move(label, &state, a, 10_000),
            None => eprintln!("EXPAND_BENCH {label}/apply_start_move=SKIPPED (no StartMove legal)"),
        }
    });
}

#[test]
#[ignore = "manual wall-clock bench — run with --ignored"]
fn expand_bench_main() {
    if skip_unless_release("expand_bench_main") {
        return;
    }
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
    run_call_counts_inner(label, &mut state, iters, false);
}

fn run_call_counts_horizon(label: &str, mut state: GameState, iters: usize) {
    run_call_counts_inner(label, &mut state, iters, true);
}

fn run_call_counts_inner(label: &str, state: &mut GameState, iters: usize, with_horizon: bool) {
    // Match `MctsBot::get_action`'s setup so the tree behaves identically.
    state.set_dice_mode(DiceMode::RegisterRolls);
    state.set_logging_state(false);
    state.clear_log();

    let counters = Arc::new(Counters::default());
    // Mirror MctsBot::get_action's horizon capture (plan 014). Anchor
    // is held constant for the whole search so recombination stays a
    // pure function of (state, anchor).
    let horizon = if with_horizon {
        let agent_team = state.available_actions.team.unwrap_or(state.info.team_turn);
        Some(HorizonAnchor::capture(state, agent_team))
    } else {
        None
    };
    let inner = BloodBowlDynamics {
        horizon,
        ..BloodBowlDynamics::default()
    };
    let dynamics = CountingDynamics {
        inner,
        counters: Arc::clone(&counters),
    };
    let root_player = player_for_root(state);
    let root_state = state.clone();

    // Drive the tree (and, critically, drop it) on a big stack — without a
    // horizon (`with_horizon == false`) the score_td drive builds deep DAG
    // chains whose recursive teardown overflows the test thread's default
    // ~2 MB stack. See `with_big_stack`.
    let elapsed = with_big_stack(move || {
        // StoreState matches production MctsBot::get_action (dynamics.rs:749).
        // HashOnly corrupts the DAG (collisions merge distinct states), which
        // surfaced both mid-search (illegal-action assert in micro_step) and on
        // drop (DAG re-derivation) — see the old std::mem::forget hack below.
        let tree = Tree::new(dynamics, StoreState, root_player, root_state);

        let t0 = Instant::now();
        for _ in 0..iters {
            tree.step();
        }
        let elapsed = t0.elapsed();

        // StoreState stores full state per node, so dropping the tree no longer
        // re-derives the DAG — the old `std::mem::forget(tree)` HashOnly hack is
        // gone and the tree drops normally here (on this big-stack thread).
        drop(tree);
        elapsed
    });

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
    if skip_unless_release("expand_bench_call_counts") {
        return;
    }
    eprintln!("EXPAND_COUNTS seed={SEED:#x}");
    run_call_counts("score_td_easy", build_score_td_easy(), 1000);
    run_call_counts("full_teams", build_full_teams_turn_start(), 1000);
}

// Horizon-bounded variants — one per scenario so a drop-time panic in
// one doesn't poison the next. Mirrors the real MctsBot::get_action
// path (Step F horizon active).

#[test]
#[ignore = "manual call-counting bench (horizon, plan 014) — run with --ignored"]
fn expand_counts_horizon_score_td_1k() {
    if skip_unless_release("expand_counts_horizon_score_td_1k") {
        return;
    }
    eprintln!("EXPAND_COUNTS_HORIZON seed={SEED:#x}");
    run_call_counts_horizon("score_td_easy@1k", build_score_td_easy(), 1000);
}

#[test]
#[ignore = "manual call-counting bench (horizon, plan 014) — run with --ignored"]
fn expand_counts_horizon_score_td_10k() {
    if skip_unless_release("expand_counts_horizon_score_td_10k") {
        return;
    }
    eprintln!("EXPAND_COUNTS_HORIZON seed={SEED:#x}");
    run_call_counts_horizon("score_td_easy@10k", build_score_td_easy(), 10_000);
}

#[test]
#[ignore = "manual call-counting bench (horizon, plan 014) — run with --ignored"]
fn expand_counts_horizon_full_teams_1k() {
    if skip_unless_release("expand_counts_horizon_full_teams_1k") {
        return;
    }
    eprintln!("EXPAND_COUNTS_HORIZON seed={SEED:#x}");
    run_call_counts_horizon("full_teams@1k", build_full_teams_turn_start(), 1000);
}

#[test]
#[ignore = "manual call-counting bench (horizon, plan 014) — run with --ignored"]
fn expand_counts_horizon_full_teams_10k() {
    if skip_unless_release("expand_counts_horizon_full_teams_10k") {
        return;
    }
    eprintln!("EXPAND_COUNTS_HORIZON seed={SEED:#x}");
    run_call_counts_horizon("full_teams@10k", build_full_teams_turn_start(), 10_000);
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
    if skip_unless_release("expand_bench_for_samply") {
        return;
    }
    eprintln!("EXPAND_COUNTS_SAMPLY seed={SEED:#x}");
    // 200k iters give samply enough samples in the release build this test
    // targets (debug is skipped above — never the real profiling run).
    run_call_counts("score_td_easy", build_score_td_easy(), 200_000);
    run_call_counts("full_teams", build_full_teams_turn_start(), 200_000);
}
