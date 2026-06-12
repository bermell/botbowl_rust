# CLAUDE.md — botbowl-curriculum

Training-scenario harness. Depends only on `botbowl-engine`.

- `Lecture` trait describes a scenario: `setup(rng) -> GameState`, `evaluate(state, ctx) -> {Success, Failure, InProgress}`, plus metadata like `agent_team()`.
- `LectureSession` drives one trial one `micro_step` at a time (useful for live ratatui rendering); `run_trials` is the headless batch driver.
- Lectures are registered in `lib.rs::make_lecture` and surfaced through the UI's `curriculum` subcommand:
  `cargo run -p botbowl-ui -- curriculum "Score TD" --difficulty easy --bot mcts`
- Lectures typically use `DiceMode::DicePolicy` for difficulty control (e.g. `SucceedAtOrEasier`).
