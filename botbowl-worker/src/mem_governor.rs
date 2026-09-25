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
//! **Reservation, not check-then-act.** [`MemGovernor::try_admit`] adds
//! `area` to the in-flight total *before* checking it against headroom, and
//! rolls the reservation back on failure. A plain "read headroom, check if
//! this one game fits, then start it" gate is a TOCTOU race across threads:
//! at phase start, `parallel_games` threads all wake and each reads
//! essentially the same headroom snapshot before any of them has actually
//! grown a tree, so each individually looks affordable even though their
//! *combined* cost isn't. That race is exactly what happened live on
//! 2026-09-24's relaunch — 16 threads each checked a ~7.5 GB reading
//! against their own ~650 MB prediction and all admitted, for a combined
//! ~10.4 GB commitment the same 7.5 GB baseline could never have covered,
//! and the governor never got a chance to refuse any of them. Folding the
//! reservation into the same atomic step as the check closes that gap: a
//! burst of N simultaneous callers is serialized against one shared
//! counter, so only as many as truly fit are admitted, no matter how many
//! ask at once.
//!
//! All OS memory reads live in `lib.rs` (`available_memory_mb`); this
//! module is pure arithmetic so the calibration and admission logic is
//! unit-testable without mocking the OS.

use std::sync::atomic::{AtomicU64, Ordering};

/// Convert a game into the governor's cost unit: playable cells scaled by how much search runs on
/// them.
///
/// A DAG's footprint grows with the number of iterations that expand it, not only with the board
/// it sits on, and the calibration behind [`DEFAULT_KB_PER_CELL`] was taken at
/// [`REFERENCE_ITERS`]. Left unscaled, a 4000-iteration job (plan 045's arms, and any experiment
/// that moves the budget) starts out predicting a quarter of its true cost and only catches up
/// through the EWMA — which is several games of over-admission on exactly the runs where one tree
/// is multiple GB. Scaling here keeps the stored constant meaning one thing, so a fleet mixing
/// budgets still shares one calibration.
///
/// `iters` is `None` for a time-budgeted search, where iteration count is not known up front;
/// that falls back to the reference, i.e. the old behaviour.
pub fn cost_units(area: u32, iters: Option<u64>) -> u32 {
    let iters = iters.unwrap_or(REFERENCE_ITERS).max(1);
    let scaled = (area as u64 * iters).div_ceil(REFERENCE_ITERS);
    scaled.min(u32::MAX as u64) as u32
}

/// Seed estimate: `scripts/train_loop.sh`'s `GEN_PARALLEL_GAMES` comment
/// measured ~400-500 MB per tree at 1000 iters on a 98-cell (14x7) board,
/// i.e. ~4.5 MB/cell. Refined at runtime by [`MemGovernor::observe`].
///
/// The unit is a **cost unit**, not a raw cell: one cell searched for
/// [`REFERENCE_ITERS`] iterations. See [`cost_units`].
const DEFAULT_KB_PER_CELL: u64 = 4_500;

/// The iteration count [`DEFAULT_KB_PER_CELL`] was measured at.
pub const REFERENCE_ITERS: u64 = 1_000;

/// EWMA smoothing for `kb_per_cell`, in permille (0..1000). Low, since a
/// single observation mixes in every game currently in flight and can be
/// noisy — this is a slow-moving calibration, not a per-game measurement.
const EWMA_ALPHA_PERMILLE: u64 = 200;

/// Defense-in-depth ceiling on `kb_per_cell`, as a multiple of
/// [`DEFAULT_KB_PER_CELL`]. The baseline refresh in [`MemGovernor::observe`]
/// is the real fix for calibration drift; this just bounds how bad a wrong
/// estimate can get in between refreshes, so noisy input can never leave
/// admission permanently stuck no matter what caused it.
const MAX_KB_PER_CELL_MULTIPLE: u64 = 6;

/// Tracks a running per-cell memory cost estimate and the cells currently
/// reserved (admitted, whether or not their memory use has manifested
/// yet), and decides whether reserving one more game's worth of cells is
/// safe right now.
pub struct MemGovernor {
    kb_per_cell: AtomicU64,
    active_area: AtomicU64,
    /// System memory available (kB) with nothing in flight on this worker,
    /// sampled once at startup — the headroom everything else (the nn_server
    /// sidecar, the hub, a desktop session sharing the box) leaves us. `0`
    /// means unknown, in which case `observe` no-ops and `try_admit`/
    /// `admit_unconditionally` always admit: we'd rather run unthrottled
    /// than block forever on a baseline we never got.
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

    pub fn predicted_cost_kb(&self, area: u64) -> u64 {
        self.kb_per_cell.load(Ordering::Relaxed) * area
    }

    pub fn active_area(&self) -> u64 {
        self.active_area.load(Ordering::Relaxed)
    }

    /// Reserves `area` cells against `available_kb` of headroom read just
    /// before the call, admitting only if the *total* now-reserved area
    /// (this game plus every other already-reserved one, not just this
    /// one) still fits under `available_kb - floor`. Rolls the reservation
    /// back and returns `None` on refusal, so the caller can back off and
    /// retry with a fresh reading.
    ///
    /// Always admits once the baseline is known to be `0` (unread at
    /// startup) — an unreadable headroom signal should never blockade a
    /// game indefinitely.
    pub fn try_admit(&self, area: u32, available_kb: u64) -> Option<GameSlot<'_>> {
        let reserved = self.active_area.fetch_add(area as u64, Ordering::SeqCst) + area as u64;
        if self.baseline_available_kb.load(Ordering::Relaxed) != 0 {
            let committed_kb = self.predicted_cost_kb(reserved);
            if available_kb < self.floor_kb + committed_kb {
                self.active_area.fetch_sub(area as u64, Ordering::SeqCst);
                return None;
            }
        }
        Some(GameSlot { governor: self, area })
    }

    /// Reserves `area` cells with no headroom check — used only when
    /// headroom couldn't be read at all (see `try_admit`'s doc).
    pub fn admit_unconditionally(&self, area: u32) -> GameSlot<'_> {
        self.active_area.fetch_add(area as u64, Ordering::SeqCst);
        GameSlot { governor: self, area }
    }

    /// Refine `kb_per_cell` from a fresh headroom reading, given
    /// `active_area` cost units were reserved when it was taken.
    ///
    /// With nothing reserved there is nothing to attribute a headroom
    /// change to — but that moment is also the only trustworthy "zero
    /// load" reading this worker will see after startup, so it refreshes
    /// the baseline instead of discarding the sample. Without this,
    /// `baseline_available_kb` stays pinned to whatever ambient memory
    /// looked like at connection time forever, and anything that changes
    /// later for reasons unrelated to games (a desktop session, page-cache
    /// churn) gets misattributed entirely to the next game and inflates
    /// the estimate. That inflation used to be one-directional too — a low
    /// `implied` reading was discarded rather than blended in — so once
    /// it drifted there was no way back: on 2026-09-25 `kb_per_cell` crept
    /// to ~7x its seed over a multi-hour eval and refused a game with 8 GB
    /// genuinely free. `implied` is now blended in both directions and
    /// clamped as a backstop against however bad a single noisy reading is.
    pub fn observe(&self, available_kb: u64, active_area: u64) {
        if active_area == 0 {
            if available_kb > 0 {
                self.baseline_available_kb.store(available_kb, Ordering::Relaxed);
            }
            return;
        }
        let baseline = self.baseline_available_kb.load(Ordering::Relaxed);
        if baseline == 0 {
            return;
        }
        let used_kb = baseline.saturating_sub(available_kb);
        let implied = used_kb / active_area;
        let prev = self.kb_per_cell.load(Ordering::Relaxed);
        let next = (prev * (1000 - EWMA_ALPHA_PERMILLE) + implied * EWMA_ALPHA_PERMILLE) / 1000;
        let clamped = next.clamp(1, DEFAULT_KB_PER_CELL * MAX_KB_PER_CELL_MULTIPLE);
        self.kb_per_cell.store(clamped, Ordering::Relaxed);
    }

    /// The current "zero load" reference `observe` measures against.
    /// Diagnostic / test use.
    pub fn baseline_available_kb(&self) -> u64 {
        self.baseline_available_kb.load(Ordering::Relaxed)
    }
}

/// RAII admission for one game: holds `area` cells reserved against the
/// governor until dropped, so a panic mid-game (`spawn_game_threads`
/// `catch_unwind`s around the whole task, not per game) still releases it
/// during unwind instead of leaking phantom "in flight" area forever. Only
/// ever constructed by `MemGovernor::try_admit` / `admit_unconditionally`,
/// which is what pairs its `area` with a matching reservation.
pub struct GameSlot<'a> {
    governor: &'a MemGovernor,
    area: u32,
}

impl Drop for GameSlot<'_> {
    fn drop(&mut self) {
        self.governor.active_area.fetch_sub(self.area as u64, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_baseline_never_blocks() {
        let g = MemGovernor::new(1024, None);
        assert!(g.try_admit(20_000, 0).is_some());
        g.observe(0, 5_000);
        assert_eq!(g.predicted_cost_kb(1), DEFAULT_KB_PER_CELL);
    }

    #[test]
    fn admits_under_floor_plus_prediction() {
        // 8 GB baseline, 1 GB floor, default ~4.5 MB/cell -> a 98-cell (14x7)
        // game predicts ~441 MB, well inside 8 GB - 1 GB of headroom.
        let g = MemGovernor::new(1024, Some(8 * 1024));
        assert!(g.try_admit(98, 8 * 1024 * 1024).is_some());
    }

    #[test]
    fn refuses_when_headroom_is_tight() {
        let g = MemGovernor::new(1024, Some(8 * 1024));
        // Only 1200 MB available: floor alone (1024) plus any real
        // prediction for a 98-cell board (~441 MB) blows through it.
        assert!(g.try_admit(98, 1200 * 1024).is_none());
        // A refusal must roll its reservation back, not leak it.
        assert_eq!(g.active_area(), 0);
    }

    #[test]
    fn larger_board_costs_more() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        assert!(g.predicted_cost_kb(400) > g.predicted_cost_kb(98));
    }

    #[test]
    fn cost_scales_with_the_search_budget() {
        // The whole point: a 4000-iteration game on 14x7 must not be predicted as cheaply as a
        // 1000-iteration one, or plan-045-style arms over-admit until the EWMA catches up.
        assert_eq!(cost_units(98, Some(1_000)), 98);
        assert_eq!(cost_units(98, Some(4_000)), 98 * 4);
        assert_eq!(cost_units(98, Some(500)), 49);
        // A time budget is unknowable up front: fall back to the calibration point.
        assert_eq!(cost_units(98, None), 98);
        // A search-free bot still rounds up to something non-zero rather than wrapping.
        assert_eq!(cost_units(98, Some(0)), 1);
        // And a 16x9 board at 4000 iters outprices a 14x7 board at 1000, which is the ordering
        // the admission decision actually rests on.
        let g = MemGovernor::new(0, Some(8 * 1024));
        assert!(
            g.predicted_cost_kb(cost_units(144, Some(4_000)) as u64)
                > g.predicted_cost_kb(cost_units(98, Some(1_000)) as u64)
        );
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
    fn observe_with_nothing_in_flight_does_not_touch_the_estimate() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        let before = g.predicted_cost_kb(1);
        g.observe(1024, 0);
        assert_eq!(g.predicted_cost_kb(1), before);
    }

    #[test]
    fn idle_observations_refresh_the_baseline() {
        let g = MemGovernor::new(0, Some(8 * 1024)); // 8 GB
        assert_eq!(g.baseline_available_kb(), 8 * 1024 * 1024);
        // Ambient conditions improved while nothing was in flight (or the
        // stale connection-time reading was simply wrong) — the next idle
        // observation must adopt it as the new reference, not keep the old
        // one forever.
        g.observe(12 * 1024 * 1024, 0);
        assert_eq!(g.baseline_available_kb(), 12 * 1024 * 1024);
        // A genuinely unreadable `0` must never zero the baseline out — that
        // sentinel means "unknown, never block" elsewhere and would silently
        // disable admission control rather than making it more cautious.
        g.observe(0, 0);
        assert_eq!(g.baseline_available_kb(), 12 * 1024 * 1024);
    }

    #[test]
    fn estimate_can_fall_as_well_as_rise() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        // Push it up first, as `observe_pulls_estimate_toward_implied_cost` does.
        g.observe(8 * 1024 * 1024 - 1_000_000, 100);
        let up = g.predicted_cost_kb(1);
        assert!(up > DEFAULT_KB_PER_CELL, "sanity: estimate should have risen first");
        // Repeated near-zero-implied observations (headroom back near
        // baseline) must be able to pull it back down. The old code
        // discarded every `implied == 0` sample outright, so a single noisy
        // spike could never be corrected for the rest of a run — exactly
        // what stalled a real eval on 2026-09-25.
        for _ in 0..20 {
            g.observe(8 * 1024 * 1024 - 1, 100);
        }
        let down = g.predicted_cost_kb(1);
        assert!(down < up, "estimate should be able to fall back down, {down} vs {up}");
    }

    #[test]
    fn calibration_drift_is_clamped() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        // Feed it a wildly high implied cost repeatedly (used_kb ~= the
        // whole baseline attributed to a single cost unit) — the shape of
        // noise that drifted a real run's estimate to ~7x its seed.
        for _ in 0..50 {
            g.observe(1, 1);
        }
        assert!(
            g.predicted_cost_kb(1) <= DEFAULT_KB_PER_CELL * MAX_KB_PER_CELL_MULTIPLE,
            "estimate must not drift past its clamp no matter how bad the input, got {}",
            g.predicted_cost_kb(1)
        );
    }

    #[test]
    fn reservation_round_trips_on_success() {
        let g = MemGovernor::new(0, Some(8 * 1024));
        assert_eq!(g.active_area(), 0);
        {
            let slot = g.try_admit(150, 8 * 1024 * 1024);
            assert!(slot.is_some());
            assert_eq!(g.active_area(), 150);
        }
        assert_eq!(g.active_area(), 0);
    }

    /// The bug this module exists to fix: a burst of callers who all read
    /// the *same* stale headroom snapshot (as `parallel_games` threads do
    /// at phase start, before any of them has actually grown a tree) must
    /// not all be admitted just because each one individually looked
    /// affordable against that shared reading.
    #[test]
    fn a_burst_against_one_stale_reading_does_not_all_admit() {
        // 2 GB headroom, no floor, ~4.5 MB/cell default -> ~444 cells fit.
        // 10 callers each reserving a 98-cell board (980 cells total) must
        // not all get in against the same 2 GB snapshot.
        let g = MemGovernor::new(0, Some(2 * 1024));
        let available_kb = 2 * 1024 * 1024;
        // Held alive for the whole burst, exactly like concurrent game
        // threads keep their `GameSlot` for a game's whole duration — a
        // slot dropped right after the check (as a bare `.is_some()` would)
        // releases its reservation before the next caller even asks.
        let slots: Vec<_> = (0..10).filter_map(|_| g.try_admit(98, available_kb)).collect();
        let admitted = slots.len();
        assert!(
            admitted < 10,
            "a stale shared reading admitted every caller in the burst ({admitted}/10)"
        );
        // What actually got in still fits the budget it was checked against.
        assert!(g.predicted_cost_kb(g.active_area()) <= available_kb);
    }
}
