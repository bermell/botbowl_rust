pub mod lecture;
pub mod lectures;
pub mod runner;
pub mod stochasticity;

pub use lecture::{Difficulty, Lecture, LectureContext, LectureStatus};
pub use runner::{run_trials, LectureSession, TrialStats};

use lectures::get_the_ball::{GetTheBallEasy, GetTheBallHard, GetTheBallMedium};
use lectures::score_td::{ScoreTdEasy, ScoreTdMedium};

/// Distinct lecture names (the `Lecture::name()` of every shipping lecture).
pub fn lecture_names() -> &'static [&'static str] {
    &["Score TD", "Get the ball"]
}

/// All `(name, difficulty)` pairs that resolve to a real lecture.
pub fn available_lectures() -> &'static [(&'static str, Difficulty)] {
    &[
        ("Score TD", Difficulty::Easy),
        ("Score TD", Difficulty::Medium),
        ("Get the ball", Difficulty::Easy),
        ("Get the ball", Difficulty::Medium),
        ("Get the ball", Difficulty::Hard),
    ]
}

/// Resolve a `(name, difficulty)` pair to a boxed lecture instance. `name` is
/// matched case-insensitively and trimmed.
pub fn make_lecture(name: &str, difficulty: Difficulty) -> Option<Box<dyn Lecture>> {
    let key = name.trim().to_ascii_lowercase();
    match (key.as_str(), difficulty) {
        ("score td", Difficulty::Easy) => Some(Box::new(ScoreTdEasy::new())),
        ("score td", Difficulty::Medium) => Some(Box::new(ScoreTdMedium::new())),
        ("get the ball", Difficulty::Easy) => Some(Box::new(GetTheBallEasy::new())),
        ("get the ball", Difficulty::Medium) => Some(Box::new(GetTheBallMedium::new())),
        ("get the ball", Difficulty::Hard) => Some(Box::new(GetTheBallHard::new())),
        _ => None,
    }
}
