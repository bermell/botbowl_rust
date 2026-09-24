//! Memory-aware admission control for a worker's concurrent games.
//!
//! `spawn_game_threads` runs a fixed pool of `parallel_games` threads, but a
//! *fixed* pool size can't track a variable-size search tree: plan 042's
//! board-size curriculum now draws games anywhere from 12x6 up to 18x11
//! within one generation, and a tree's memory footprint scales with board
//! area. `scripts/train_loop.sh`'s `GEN_PARALLEL_GAMES` was measured once
//! against 14x7 (98 cells, ~400-500 MB/tree at 1000 iters) and never
//! revisited as the curriculum grew the boards — which is how a 16-GB box
//! ended up at 54% sustained memory pressure and had its whole session
//! killed by `systemd-oomd` mid-generation (2026-09-24).
//!
//! Instead of retuning that constant for a size that keeps moving, each
//! game thread checks system memory headroom against a running per-cell
//! cost estimate immediately before it starts its next game, and blocks
//! (with backoff) rather than launch one likely to tip the box into swap.
//! The estimate is seeded from the measurement above and refined at
//! runtime from observed headroom, so it tracks whatever this box and this
//! net's tree shape actually cost rather than a number frozen at one board
//! size.
//!
//! All OS memory reads live in `lib.rs` (`available_memory_mb`); this
//! module is pure arithmetic so the calibration logic is unit-testable
//! without mocking the OS.

use std::sync::atomic::{AtomicU64, Ordering};

/// Seed estimate: `scripts/train_loop.sh`'s `GEN_PARALLEL_GAMES` comment
/// measured ~400-500 MB per tree at 1000 iters on a 98-cell (14x7) board,
/// i.e. ~4.5 MB/cell. Refined at runtime by [`MemGovernor::observe`].
const DEFAULT_KB_PER_CELL: u64 = 4_500;

/// EWMA smoothing for `kb_per_cell`, in permille (0..1000). Low, since a
/// single observation mixes in every game currently in flight and can be
/// noisy — this is a slow-moving calibration, not a per-game measurement.
const EWMA_ALPHA_PERMILLE: u64 = 200;

/// Tracks a running per-cell memory cost estimate and the cells currently
/// in flight, and decides whether starting one more game of a given board
/// area is safe right now.
pub struct MemGovernor {
    kb_per_cell: AtomicU64,
    active_area: AtomicU64,
    /// System memory available (kB) with nothing in flight on this worker,
    /// sampled once at startup — the headroom everything else (the nn_server
    /// sidecar, the hub, a desktop session sharing the box) leaves us. `0`
    /// means unknown, in which case `observe` and `admits` both no-op /
    /// always-admit: we'd rather run unthrottled than block forever on a
    /// baseline we never got.
    baseline_available_kb: AtomicU64,
    floor_kb: u64,
}

impl MemGovernor {
    pub fn new(floor_mb: u32, baseline_available_mb: Option<u32>) -> Self {
        MemGovernor {
            kb_per_cell: AtomicU64::new(DEFAULT_KB_PER_CELL),
            active_area: AtomicU64::new(0),
            baseline_available_kb: AtomicU64::new(baseline_available_mb.unwrap_or(0) as u64 * 1024),
            floor_kb: floor_mb as u64 * 1024,
        }
    }

    fn predicted_cost_kb(&self, area: u32) -> u64 {
        self.kb_per_cell.load(Ordering::Relaxed) * area as u64
    }

    /// Whether a game of `area` playable cells looks safe to start given
    /// `available_kb` of current headroom. Always admits when the baseline
    /// is unknown (`0`) — never block on a signal we don't trust.
    pub fn admits(&self, area: u32, available_kb: u64) -> bool {
        if self.baseline_available_kb.load(Ordering::Relaxed) == 0 {
            return true;
        }
        available_kb >= self.floor_kb + self.predicted_cost_kb(area)
    }

    pub fn active_area(&self) -> u64 {
        self.active_area.load(Ordering::Relaxed)
    }

    pub fn account_start(&self, area: u32) {
        self.active_area.fetch_add(area as u64, Ordering::Relaxed);
    }

    pub fn account_end(&self, area: u32) {
        self.active_area.fetch_sub(area as u64, Ordering::Relaxed);
    }

    /// Refine `kb_per_cell` from a fresh headroom reading, given
    /// `active_area` playable cells were in flight when it was taken.
    /// No-op with nothing in flight (nothing to attribute the headroom
    /// drop to) or before a baseline is known.
    pub fn observe(&self, available_kb: u64, active_area: u64) {
        let baseline = self.baseline_available_kb.load(Ordering::Relaxed);
        if baseline == 0 || active_area == 0 {
            return;
        }
        let used_kb = baseline.saturating_sub(available_kb);
        let implied = used_kb / active_area;
        if implied == 0 {
            return;
        }
        let prev = self.kb_per_cell.load(Ordering::Relaxed);
        let next = (prev * (1000 - EWMA_ALPHA_PERMILLE) + implied * EWMA_ALPHA_PERMILLE) / 1000;
        self.kb_per_cell.store(next.max(1), Ordering::Relaxed);
    }
}

/// RAII admission for one game: holds `area` cells accounted against the
/// governor until dropped, so a panic mid-game (`spawn_game_threads`
/// `catch_unwind`s around the whole task, not per game) still releases it
/// during unwind instead of leaking phantom "in flight" area forever.
pub struct GameSlot<'a> {
    governor: &'a MemGovernor,
    area: u32,
}

impl<'a> GameSlot<'a> {
    pub fn new(governor: &'a MemGovernor, area: u32) -> Self {
        governor.account_start(area);
        GameSlot { governor, area }
    }
}

impl Drop for GameSlot<'_> {
    fn drop(&mut self) {
        self.governor.account_end(self.area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_baseline_never_blocks() {
        let g = MemGovernor::new(1024, None);
        assert!(g.admits(20_000, 0));
        g.observe(0, 5_000);
        assert_eq!(g.predicted_cost_kb(1), DEFAULT_KB_PER_CELL);
    }

    #[test]
    fn admits_under_floor_plus_prediction() {
        // 8 GB baseline, 1 GB floor, default ~4.5 MB/cell -> a 98-cell (14x7)
        // game predicts ~441 MB, well inside 8 GB - 1 GB of headroom.
        let g = MemGovernor::new(1024, Some(8 * 1024));
        assert!(g.admits(98, 8 * 1024 * 1024));
    }

    #[test]
    fn refuses_when_headroom_is_tight() {
        let g = MemGovernor::new(1024, Some(8 * 1024));
        // Only 1200 MB available: floor alone (1024) plus any real
        // prediction for a 98-cell board (~441 MB) blows through it.
        assert!(!g.admits(98, 1200 * 1024));
    }

    #[test]
    fn larger_board_costs_more() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        assert!(g.predicted_cost_kb(400) > g.predicted_cost_kb(98));
    }

    #[test]
    fn observe_pulls_estimate_toward_implied_cost() {
        let g = MemGovernor::new(0, Some(8 * 1024 * 1024)); // 8 TB-scale kB baseline for round numbers
        let before = g.predicted_cost_kb(1);
        // Way more than the seed estimate was used per cell.
        g.observe(8 * 1024 * 1024 - 1_000_000, 100);
        let after = g.predicted_cost_kb(1);
        assert!(after > before, "estimate should move up toward the observed cost");
    }

    #[test]
    fn observe_is_noop_with_nothing_in_flight() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        let before = g.predicted_cost_kb(1);
        g.observe(1024, 0);
        assert_eq!(g.predicted_cost_kb(1), before);
    }

    #[test]
    fn account_start_and_end_round_trip() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        assert_eq!(g.active_area(), 0);
        {
            let _slot = GameSlot::new(&g, 150);
            assert_eq!(g.active_area(), 150);
        }
        assert_eq!(g.active_area(), 0);
    }
}
