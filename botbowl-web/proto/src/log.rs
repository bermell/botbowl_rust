//! The game log: one stream of everything that happened, dice included.
//!
//! The dice ticker and the text log used to be two panels fed two ways — die
//! rolls as their own message, text as a tail re-sent inside every view. One
//! stream of [`LogEntry`] replaces both: a roll is a line that happens to
//! carry die faces and the procedure that asked for it, and every line knows
//! the micro-step it was written at, so clicking it can rewind the game to
//! the position it describes ([`crate::msg::ClientMsg::RewindTo`]).

use serde::{Deserialize, Serialize};

use crate::action::TeamType;
use crate::dice::DiceEvent;

/// What kind of line this is, for styling and filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogKind {
    /// Session bookkeeping: new game, step mode, undo.
    Note,
    /// A die the engine asked for.
    Roll,
    /// A decision, by a human or a bot. `decision` names it in the decision log.
    Action,
    /// A touchdown, a drive ending, the final whistle.
    Score,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    /// 0-based position in this game's log. A rewind or an undo truncates the
    /// log ([`crate::msg::ServerMsg::LogTruncated`]), so indices are reused.
    pub index: u64,
    /// The micro-step this line was written at: the position *before* the
    /// roll or action it describes, which is what a rewind to it restores.
    pub step: usize,
    pub kind: LogKind,
    /// Whose line, when it belongs to a side.
    pub team: Option<TeamType>,
    pub text: String,
    /// The die behind a `Roll` line: faces to draw, what was asked, what came up.
    pub roll: Option<DiceEvent>,
    /// For an `Action`: the index of its [`crate::decision::DecisionRecord`].
    pub decision: Option<u64>,
}
