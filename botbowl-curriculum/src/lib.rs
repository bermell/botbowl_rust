pub mod lecture;
pub mod lectures;
pub mod runner;
pub mod stochasticity;

pub use lecture::{Difficulty, Lecture, LectureContext, LectureStatus};
pub use runner::{run_trials, TrialStats};
