// This module is used to emulate the behavior of `std::cell::Ref::map` for smart pointer types
// that do not provide a similar construct (e.g. `Mutex` and `RwLock`); it is slightly less
// efficient because the map is recalculated on each `Deref::deref` whereas `Ref::map` is able to
// store the results of the map using `Ref`'s internals; for the reason `map` is not implemented
// for `Mutex` and `RwLock` see: https://stackoverflow.com/q/40095383/#comment90293227_40095383

use std::ops::Deref;

// Plan 015 Step 4 — debug-only guard against the lockref-chaining pattern
// that produced the original parallel deadlock (see plan 013). A `Ref`
// wraps a live `RwLockReadGuard`; holding two simultaneously on one
// thread is the precondition for the wait-graph cycle that froze workers
// under contention. The fix in `BloodBowlDynamics::select_node` collapses
// each `(Q, A)` to a scalar inside `.map(...)` so only one `Ref` is alive
// at a time. This assertion makes regressions die loudly in debug builds
// instead of intermittently hanging in release.
//
// In release builds the counter, the cfg-gated `Drop`, and the assertion
// all disappear; `Ref` behaves identically to before.
#[cfg(all(debug_assertions, feature = "lockref-guard"))]
thread_local! {
    static LIVE_REFS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub struct Ref<In, F> {
    r: In,
    f: F,
}

impl<In, F, Out> Ref<In, F>
where
    F: Fn(&In) -> &Out,
    Out: ?Sized,
{
    pub fn new(r: In, f: F) -> Self {
        #[cfg(all(debug_assertions, feature = "lockref-guard"))]
        LIVE_REFS.with(|c| {
            let prev = c.get();
            assert!(
                prev == 0,
                "nested lockref::Ref ({}+1) — would deadlock under contention. \
                 GD::select_node / backprop_scores must collapse each (Q, A) to a scalar \
                 before pulling the next; see plan 013.",
                prev
            );
            c.set(prev + 1);
        });
        Ref { r, f }
    }
}

#[cfg(all(debug_assertions, feature = "lockref-guard"))]
impl<In, F> Drop for Ref<In, F> {
    fn drop(&mut self) {
        LIVE_REFS.with(|c| c.set(c.get() - 1));
    }
}

impl<In, F, Out> Deref for Ref<In, F>
where
    F: Fn(&In) -> &Out,
    Out: ?Sized,
{
    type Target = Out;
    fn deref(&self) -> &Self::Target {
        (self.f)(&self.r)
    }
}
