//! Search health: how often the bot kept its plan, and what recombination cost.
//!
//! Plan 043. Two things the bot does on every decision were invisible until now.
//!
//! **Tree reuse.** Between consecutive `get_action` calls `MctsBot` tries to re-root the tree it
//! already has into the node matching the new state (plan 015 Step 1). When that works the search
//! starts from a DAG that already contains a plan; when it fails the plan is thrown away and the
//! tree is rebuilt from nothing. Five distinct things can make it fail, and they call for entirely
//! different responses — a turn boundary is expected and unavoidable, a lookup miss is not — so
//! [`ReuseOutcome`] keeps them apart, and [`TreeReuseStats`] breaks them down by the procedure on
//! top of the stack, which is what names the *kind* of decision being made.
//!
//! **Recombination.** `recon_mcts` counts registry hits and misses, and now also the state
//! comparisons behind them. Under `StoreState` a comparison clones and compares two whole
//! `GameState`s, so a rejected one is pure waste; if the rejections dominate the confirmed hits,
//! recombination is not paying for itself and the feature can go.
//!
//! Every field here is a **commutative counter**, so telemetry from any number of games, threads
//! or worker machines folds together with [`SearchTelemetry::merge`] in any order — the same rule
//! `botbowl_play::eval::LadderRow::record` encodes for eval rows.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// What happened when the bot tried to re-root its cached tree for this decision.
///
/// Ordered roughly by how early the attempt gave up, and deliberately flat (no payload) so it
/// serialises as a plain string in a trace row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReuseOutcome {
    /// Re-rooted into an existing node: the search inherited a tree that already held a plan.
    Reused,
    /// `tree_reuse` is off, so no attempt was made.
    Disabled,
    /// No cached tree — the first decision of a game, or the one after a rebuild.
    NoCache,
    /// The [`crate::HorizonAnchor`] moved: a turn boundary or a score change invalidated every Q in
    /// the cached tree. Expected and unavoidable; it is the baseline the other misses stand out
    /// against.
    AnchorMiss,
    /// The cached tree was built under a different `MemoryMode`. Only reachable if the mode is
    /// changed under a live bot.
    MarkerMiss,
    /// The registry has no node for the new root state — the search never materialised the line
    /// the game actually took.
    LookupMiss,
    /// The state is in the registry but is not a descendant of the cached tree's root, so there is
    /// no path to re-root along.
    NoPath,
}

impl ReuseOutcome {
    /// Did the search inherit a tree?
    pub fn reused(self) -> bool {
        matches!(self, ReuseOutcome::Reused)
    }

    /// Stable snake_case name, matching the serialised form.
    pub fn label(self) -> &'static str {
        match self {
            ReuseOutcome::Reused => "reused",
            ReuseOutcome::Disabled => "disabled",
            ReuseOutcome::NoCache => "no_cache",
            ReuseOutcome::AnchorMiss => "anchor_miss",
            ReuseOutcome::MarkerMiss => "marker_miss",
            ReuseOutcome::LookupMiss => "lookup_miss",
            ReuseOutcome::NoPath => "no_path",
        }
    }
}

/// One decision's reuse attempt, with the context needed to interpret it.
///
/// Carried on [`crate::SearchSummary`] so a UI can show it, and folded into [`TreeReuseStats`] for
/// the run totals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReuseDecision {
    pub outcome: ReuseOutcome,
    /// `proc_stack_top()` of the root state — what kind of decision this was. `None` only if the
    /// stack was empty, which a decision state never has.
    pub proc: Option<String>,
    /// Legal actions at the root *after* pruning: the fan the search had to cover.
    pub n_actions: usize,
    /// Edges walked to re-root. `0` when the new root was already the cached root, and always `0`
    /// when `outcome` is not [`ReuseOutcome::Reused`].
    pub path_len: usize,
}

impl ReuseDecision {
    /// The proc name for grouping, with a stable stand-in when the stack was empty.
    pub fn proc_key(&self) -> &str {
        self.proc.as_deref().unwrap_or("<none>")
    }
}

/// Per-outcome tally. One of these per procedure, plus one total.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReuseCounts {
    pub reused: u64,
    pub disabled: u64,
    pub no_cache: u64,
    pub anchor_miss: u64,
    pub marker_miss: u64,
    pub lookup_miss: u64,
    pub no_path: u64,
}

impl ReuseCounts {
    pub fn record(&mut self, outcome: ReuseOutcome) {
        match outcome {
            ReuseOutcome::Reused => self.reused += 1,
            ReuseOutcome::Disabled => self.disabled += 1,
            ReuseOutcome::NoCache => self.no_cache += 1,
            ReuseOutcome::AnchorMiss => self.anchor_miss += 1,
            ReuseOutcome::MarkerMiss => self.marker_miss += 1,
            ReuseOutcome::LookupMiss => self.lookup_miss += 1,
            ReuseOutcome::NoPath => self.no_path += 1,
        }
    }

    pub fn merge(&mut self, rhs: &ReuseCounts) {
        self.reused += rhs.reused;
        self.disabled += rhs.disabled;
        self.no_cache += rhs.no_cache;
        self.anchor_miss += rhs.anchor_miss;
        self.marker_miss += rhs.marker_miss;
        self.lookup_miss += rhs.lookup_miss;
        self.no_path += rhs.no_path;
    }

    /// Decisions counted here.
    pub fn attempts(&self) -> u64 {
        self.reused
            + self.disabled
            + self.no_cache
            + self.anchor_miss
            + self.marker_miss
            + self.lookup_miss
            + self.no_path
    }

    /// Share of decisions that inherited a tree. `None` before the first decision.
    ///
    /// Counts `disabled` and `no_cache` in the denominator deliberately: the question this answers
    /// is "how often did the bot start from a plan", not "how often did an attempt succeed".
    pub fn rate(&self) -> Option<f64> {
        let n = self.attempts();
        (n > 0).then(|| self.reused as f64 / n as f64)
    }
}

/// Reuse outcomes for a run, in total and split by the procedure that was on top of the stack.
///
/// The split is the useful part: a uniform miss rate means the horizon anchor is doing its job,
/// while one procedure missing far more than the rest points at a specific transition the search
/// is failing to materialise.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeReuseStats {
    pub total: ReuseCounts,
    /// Keyed by `proc_stack_top()`. `String` rather than `&'static str` so the whole report
    /// round-trips through JSON; the allocation happens once per distinct procedure, not per
    /// decision.
    pub by_proc: BTreeMap<String, ReuseCounts>,
}

impl TreeReuseStats {
    pub fn record(&mut self, decision: &ReuseDecision) {
        self.total.record(decision.outcome);
        let key = decision.proc_key();
        if let Some(counts) = self.by_proc.get_mut(key) {
            counts.record(decision.outcome);
        } else {
            let mut counts = ReuseCounts::default();
            counts.record(decision.outcome);
            self.by_proc.insert(key.to_string(), counts);
        }
    }

    pub fn merge(&mut self, rhs: &TreeReuseStats) {
        self.total.merge(&rhs.total);
        for (proc, counts) in &rhs.by_proc {
            self.by_proc.entry(proc.clone()).or_default().merge(counts);
        }
    }
}

/// Distribution of the root action fan, kept exactly rather than bucketed.
///
/// Production fan is bimodal — mean 20, median 6, p90 73 (plan 032 #4) — so a mean alone is
/// misleading and power-of-two buckets would blur exactly the range that matters. A sparse map is
/// exact, folds commutatively, and stays small because the fan takes few distinct values.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionFanHistogram {
    /// `n_actions` → how many decisions had that many.
    pub counts: BTreeMap<u32, u64>,
}

impl ActionFanHistogram {
    pub fn record(&mut self, n_actions: usize) {
        *self.counts.entry(n_actions as u32).or_insert(0) += 1;
    }

    pub fn merge(&mut self, rhs: &ActionFanHistogram) {
        for (n, c) in &rhs.counts {
            *self.counts.entry(*n).or_insert(0) += c;
        }
    }

    /// Decisions recorded.
    pub fn total(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Nearest-rank percentile, `q` in `[0, 1]`. `None` when nothing has been recorded.
    pub fn percentile(&self, q: f64) -> Option<u32> {
        let total = self.total();
        if total == 0 {
            return None;
        }
        // Nearest-rank: the smallest value whose cumulative count reaches ceil(q * total).
        let rank = (q.clamp(0.0, 1.0) * total as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (n, c) in &self.counts {
            seen += c;
            if seen >= rank {
                return Some(*n);
            }
        }
        self.counts.keys().next_back().copied()
    }

    /// Mean fan. `None` when nothing has been recorded.
    pub fn mean(&self) -> Option<f64> {
        let total = self.total();
        (total > 0).then(|| {
            let sum: u64 = self.counts.iter().map(|(n, c)| u64::from(*n) * c).sum();
            sum as f64 / total as f64
        })
    }
}

/// Serialisable mirror of `recon_mcts::RecombinationStats`.
///
/// `recon_mcts` is deliberately dependency-free (std only, by design), so it cannot derive serde.
/// This is the same numbers in a shape that can reach `report.json` and the browser.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecombinationCounts {
    /// Expansion probes that found the state already in the DAG — real recombination.
    pub hits: u64,
    /// Expansion probes that inserted a new node.
    pub misses: u64,
    /// `hits + misses`.
    pub probes: u64,
    /// State comparisons made inside a probe. Under `StoreState` each one clones and compares two
    /// whole `GameState`s.
    pub eq_checks: u64,
    /// Comparisons whose 64-bit hashes also agreed — as opposed to a 7-bit table-tag brush.
    pub eq_hash_equal: u64,
    /// Comparisons that returned `false`: the expensive ones that bought nothing.
    pub eq_rejects: u64,
    /// Tree-reuse lookups (`Tree::lookup_state`), counted apart from expansion.
    pub lookup_probes: u64,
    /// Tree-reuse lookups that found a node.
    pub lookup_hits: u64,
}

impl From<recon_mcts::RecombinationStats> for RecombinationCounts {
    fn from(s: recon_mcts::RecombinationStats) -> Self {
        // `len` is a level, not a counter, so it is deliberately dropped here — summing tree sizes
        // across searches would read as a total that means nothing.
        RecombinationCounts {
            hits: s.hits,
            misses: s.misses,
            probes: s.probes,
            eq_checks: s.eq_checks,
            eq_hash_equal: s.eq_hash_equal,
            eq_rejects: s.eq_rejects,
            lookup_probes: s.lookup_probes,
            lookup_hits: s.lookup_hits,
        }
    }
}

impl RecombinationCounts {
    pub fn merge(&mut self, rhs: &RecombinationCounts) {
        self.hits += rhs.hits;
        self.misses += rhs.misses;
        self.probes += rhs.probes;
        self.eq_checks += rhs.eq_checks;
        self.eq_hash_equal += rhs.eq_hash_equal;
        self.eq_rejects += rhs.eq_rejects;
        self.lookup_probes += rhs.lookup_probes;
        self.lookup_hits += rhs.lookup_hits;
    }

    /// Share of probes that recombined. `None` before the first probe.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.probes > 0).then(|| self.hits as f64 / self.probes as f64)
    }

    /// Share of comparisons that bought nothing. `None` before the first comparison.
    pub fn reject_rate(&self) -> Option<f64> {
        (self.eq_checks > 0).then(|| self.eq_rejects as f64 / self.eq_checks as f64)
    }
}

/// Everything one bot accumulated over its lifetime.
///
/// Lives on [`crate::MctsBot`] as plain `u64`s rather than atomics: `get_action` takes `&mut self`,
/// and the reuse decision is made on the owning thread before any search worker spawns. A process
/// running several differently-tuned bots at once (the web server, plan 034) therefore gets one
/// tally per bot, which a global static could not provide.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchTelemetry {
    /// Decisions made.
    pub searches: u64,
    pub reuse: TreeReuseStats,
    pub recombination: RecombinationCounts,
    pub fan: ActionFanHistogram,
}

impl SearchTelemetry {
    /// Fold one decision in. Called once per `get_action`.
    pub fn record(&mut self, decision: &ReuseDecision, recombination: RecombinationCounts) {
        self.searches += 1;
        self.reuse.record(decision);
        self.fan.record(decision.n_actions);
        self.recombination.merge(&recombination);
    }

    /// Fold another bot's (or game's, or worker's) totals in. Commutative.
    pub fn merge(&mut self, rhs: &SearchTelemetry) {
        self.searches += rhs.searches;
        self.reuse.merge(&rhs.reuse);
        self.recombination.merge(&rhs.recombination);
        self.fan.merge(&rhs.fan);
    }

    /// One grep-able line, as printed under `BLOOD_MCTS_STATS=1`.
    pub fn summary(&self) -> String {
        let pct = |v: Option<f64>| v.map_or_else(|| "n/a".to_string(), |r| format!("{:.4}", r));
        let r = &self.reuse.total;
        format!(
            "searches={} reuse={} reuse_rate={} anchor_miss={} lookup_miss={} no_path={} \
             fan_p50={} fan_p90={} recomb_hits={} recomb_misses={} recomb_hit_rate={} \
             eq_checks={} eq_rejects={} eq_reject_rate={}",
            self.searches,
            r.reused,
            pct(r.rate()),
            r.anchor_miss,
            r.lookup_miss,
            r.no_path,
            self.fan.percentile(0.5).map_or(-1i64, i64::from),
            self.fan.percentile(0.9).map_or(-1i64, i64::from),
            self.recombination.hits,
            self.recombination.misses,
            pct(self.recombination.hit_rate()),
            self.recombination.eq_checks,
            self.recombination.eq_rejects,
            pct(self.recombination.reject_rate()),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn decision(outcome: ReuseOutcome, proc: &str, n_actions: usize) -> ReuseDecision {
        ReuseDecision {
            outcome,
            proc: Some(proc.to_string()),
            n_actions,
            path_len: 0,
        }
    }

    /// The whole point of these being counters: two halves of a run, folded in either order, give
    /// the same totals as the run itself. This is what lets workers report independently.
    #[test]
    fn merging_is_commutative() {
        let mut a = SearchTelemetry::default();
        a.record(
            &decision(ReuseOutcome::Reused, "MoveAction", 6),
            RecombinationCounts::default(),
        );
        a.record(
            &decision(ReuseOutcome::AnchorMiss, "Block", 12),
            RecombinationCounts::default(),
        );

        let mut b = SearchTelemetry::default();
        b.record(
            &decision(ReuseOutcome::Reused, "Block", 4),
            RecombinationCounts::default(),
        );

        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);

        assert_eq!(ab, ba);
        assert_eq!(ab.searches, 3);
        assert_eq!(ab.reuse.total.reused, 2);
        assert_eq!(ab.reuse.by_proc["Block"].reused, 1);
        assert_eq!(ab.reuse.by_proc["Block"].anchor_miss, 1);
    }

    /// Nearest-rank, so a percentile is always a fan size that actually occurred.
    #[test]
    fn percentiles_come_from_observed_fans() {
        let mut h = ActionFanHistogram::default();
        for n in [2, 4, 4, 6, 6, 6, 8, 40, 73, 98] {
            h.record(n);
        }
        assert_eq!(h.total(), 10);
        assert_eq!(h.percentile(0.5), Some(6));
        assert_eq!(h.percentile(0.9), Some(73));
        assert_eq!(h.percentile(1.0), Some(98));
        assert_eq!(h.percentile(0.0), Some(2), "q=0 still names the smallest observed fan");
        assert_eq!(ActionFanHistogram::default().percentile(0.5), None);
    }

    /// `disabled` and `no_cache` stay in the denominator: the question is how often the bot had a
    /// plan to start from, not how often an attempt that was made succeeded.
    #[test]
    fn reuse_rate_counts_every_decision() {
        let mut c = ReuseCounts::default();
        c.record(ReuseOutcome::NoCache);
        c.record(ReuseOutcome::Reused);
        c.record(ReuseOutcome::Reused);
        c.record(ReuseOutcome::AnchorMiss);
        assert_eq!(c.attempts(), 4);
        assert_eq!(c.rate(), Some(0.5));
    }
}
