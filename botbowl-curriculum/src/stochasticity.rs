// Stochasticity-policy placeholder.
//
// The grand plan calls for target-aware dice policies (e.g. "3+ dodges
// succeed but 4+ dodges fail"). Implementing that cleanly requires an
// engine hook in `GameState::get_d6_roll` / `get_roll_result` that
// consults a policy before falling back to the FIFO queue and RNG.
//
// The first lecture (Score TD - Easy) needs no rolls, so we ship this
// crate with the policy stubbed out and revisit when the second
// lecture motivates the engine change.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DicePolicy {
    /// Use whatever was queued via `state.fixes`, then fall back to RNG.
    Default,
}

impl Default for DicePolicy {
    fn default() -> Self {
        DicePolicy::Default
    }
}
