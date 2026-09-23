//! Per-decision tree-reuse trace (plan 043), for when the aggregate looks wrong.
//!
//! `report.json` and a trajectory's provenance always carry the reuse rates broken down by
//! procedure. That answers "how often" and "in what kind of position", which is usually enough.
//! What it cannot answer is "which actions, exactly" — and when one procedure shows an
//! unexpected miss rate, that is the next question.
//!
//! So this is opt-in and off by default: enabling it costs a formatted action list per decision,
//! which is real work next to nothing, and would bloat every corpus for a question nobody is
//! asking most of the time.
//!
//! One JSON object per line:
//!
//! ```json
//! {"game":7,"decision":41,"proc":"Block","n_actions":12,"outcome":"lookup_miss",
//!  "path_len":0,"actions":["Block(3,4)","Block(3,5)"]}
//! ```

use std::io::{self, Write};
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use botbowl_engine::core::gamestate::GameState;
use botbowl_mcts::{ReuseDecision, ReuseOutcome};

/// The maximum number of actions written per row.
///
/// A mid-turn move fan reaches ~100 (plan 032 #4 puts p90 at 73), and the point of the list is to
/// recognise the *kind* of decision, not to reconstruct it exactly. `n_actions` always carries
/// the true count, so a truncated row is still honest about how wide the fan was.
pub const MAX_TRACED_ACTIONS: usize = 24;

/// One decision's reuse outcome with its full context.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ReuseTraceRow {
    /// The game (eval) or trajectory seed index (generation) this decision belongs to.
    pub game: u32,
    /// Which decision of that game, from 0.
    pub decision: u32,
    /// `proc_stack_top()` — what kind of decision this was.
    pub proc: Option<String>,
    /// Legal actions after pruning: the fan the search had to cover.
    pub n_actions: usize,
    pub outcome: ReuseOutcome,
    /// Edges walked to re-root; `0` unless the outcome is `reused`.
    pub path_len: usize,
    /// The actions themselves, truncated at [`MAX_TRACED_ACTIONS`].
    pub actions: Vec<String>,
}

impl ReuseTraceRow {
    /// Build a row from a finished decision and the state it was made in.
    ///
    /// `state` must be the state the bot was asked about, so the action list matches the fan the
    /// search saw.
    pub fn new(game: u32, decision: u32, d: &ReuseDecision, state: &GameState) -> Self {
        let actions = state
            .get_all_actions()
            .into_iter()
            .take(MAX_TRACED_ACTIONS)
            .map(|a| format!("{a:?}"))
            .collect();
        ReuseTraceRow {
            game,
            decision,
            proc: d.proc.clone(),
            n_actions: d.n_actions,
            outcome: d.outcome,
            path_len: d.path_len,
            actions,
        }
    }
}

/// A JSONL sink shared by the workers of a run.
///
/// Appends, like `--per-game-out`, so a resumed run adds to the file rather than truncating it. A
/// whole line must be atomic against its peers, which is what the mutex is for — the same reason
/// the eval per-game writer has one.
#[derive(Debug)]
pub struct ReuseTraceWriter {
    out: Mutex<io::BufWriter<std::fs::File>>,
}

impl ReuseTraceWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::options().create(true).append(true).open(path)?;
        Ok(ReuseTraceWriter {
            out: Mutex::new(io::BufWriter::new(file)),
        })
    }

    /// Write one row. Errors are reported once and then swallowed: a diagnostic trace must never
    /// take down a multi-hour run.
    pub fn write(&self, row: &ReuseTraceRow) {
        let mut guard = self.out.lock().expect("reuse trace writer");
        if let Ok(line) = serde_json::to_string(row) {
            let _ = writeln!(guard, "{line}");
        }
    }

    pub fn flush(&self) {
        if let Ok(mut guard) = self.out.lock() {
            let _ = guard.flush();
        }
    }
}

impl Drop for ReuseTraceWriter {
    fn drop(&mut self) {
        self.flush();
    }
}
