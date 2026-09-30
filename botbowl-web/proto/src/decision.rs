//! The decision log: one record per action anybody took, human or bot.
//!
//! A bot record carries the whole root of the search behind it
//! ([`SearchReport`]); every record — the human's too — carries the net's own
//! read of the position it was taken in ([`NetReadout`]), so "what did the
//! policy head want here" has an answer whoever was choosing. Both are
//! snapshots taken when the decision was made: they stay readable after the
//! bot's tree has been re-rooted, which is what makes the log browsable
//! backwards. Only the *latest* search of each bot can still be walked below
//! the root (see [`crate::search::SearchReport::search_id`]).

use serde::{Deserialize, Serialize};

use crate::action::{Action, TeamType};
use crate::search::SearchReport;

/// Who took a decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Decider {
    Human,
    /// `BotSpec::label()` of the seat's bot.
    Bot {
        label: String,
    },
}

/// The net's probability for one legal action.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActionPrior {
    pub action: Action,
    /// Softmax over **every** legal action, summing to 1 — not the search's
    /// PUCT prior, which is renormalised over the pruned set and rescaled to
    /// mean 1.
    pub prob: f32,
}

/// One forward pass of the net over one position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetReadout {
    /// Basename of the net that produced this.
    pub model: String,
    /// Home-centric in `[-1, 1]`.
    pub value_home: f32,
    /// Whose policy this is — the side to act.
    pub mover: TeamType,
    /// Sorted by `prob`, descending. Empty when nobody is being asked for an
    /// action (a pending roll, game over).
    pub priors: Vec<ActionPrior>,
}

impl NetReadout {
    /// The net's probability for `action`, and its rank (1 = the net's
    /// favourite) among the legal actions.
    pub fn rank_of(&self, action: Action) -> Option<(usize, f32)> {
        self.priors
            .iter()
            .position(|p| p.action == action)
            .map(|i| (i + 1, self.priors[i].prob))
    }

    pub fn prob_of(&self, action: Action) -> Option<f32> {
        self.priors.iter().find(|p| p.action == action).map(|p| p.prob)
    }
}

/// One decision, as the log shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// 0-based position in this game's decision log. An undo truncates the
    /// log ([`crate::msg::ServerMsg::DecisionsTruncated`]), so indices are
    /// reused after one.
    pub index: u64,
    pub team: TeamType,
    pub by: Decider,
    pub action: Action,
    /// `proc_stack_top()` — what kind of decision it was.
    pub proc: String,
    pub half: u8,
    /// The deciding team's turn counter.
    pub turn: u8,
    /// Legal actions offered, before any search-side pruning.
    pub n_legal: usize,
    /// The net's read of the position the decision was taken in. For an MCTS
    /// bot this is the bot's own net; for a human or the random bot it is the
    /// first net seated in the game. `None` when no seat has one.
    pub net: Option<NetReadout>,
    /// The search behind an MCTS decision.
    pub search: Option<Box<SearchReport>>,
}
