# Placement Finder Plan (manual setup)

## Goal
Provide a reusable, deterministic helper that returns legal coordinates for player placement during setup. Manual setup will poll this list between placements so bots only pick valid squares and cannot dead-end themselves.

## Constraints to Enforce (current rules as encoded in the engine)
- Team half only:
  - Home: `pos.x >= get_line_of_scrimage_x(Home)`
  - Away: `pos.x <= get_line_of_scrimage_x(Away)`
- Line of scrimmage (LOS) minimum: at least `min(3, min(11, available_players))` on the LOS line in `LINE_OF_SCRIMMAGE_Y_RANGE`.
- Wide zones (wings) maximum: at most 2 in each wing (`NORTH_WING_Y_RANGE`, `SOUTH_WING_Y_RANGE`).
- Max players on pitch: 11.
- Cannot place on occupied or out-of-bounds squares.

These align with `GameState::is_setup_legal` today.

## API Proposal
Add a setup placement helper exposed from `GameState` (or a new `src/core/placement_finder.rs` module), e.g.:

```rust
pub fn get_legal_setup_positions(&self, team: TeamType) -> Vec<Position>
```

Optional supporting helpers (kept private if preferred):

```rust
struct SetupCounts { on_pitch: usize, los: usize, north: usize, south: usize }
struct SetupLimits { min_on_pitch: usize, min_los: usize }
```

## Algorithm (feasibility-aware)
Goal: each returned square must allow completing setup without violating constraints.

1. Gather counts for `team`:
   - `on_pitch` and zone counts (`los`, `north`, `south`) from current players on pitch.
   - `on_bench` from dugout reserves.
   - `min_on_pitch = min(11, on_pitch + on_bench)`.
   - `min_los = min(3, min_on_pitch)`.
   - `remaining_to_place = min_on_pitch.saturating_sub(on_pitch)`.
2. If `remaining_to_place == 0`, return empty vector (manual setup should only allow `EndSetup`).
3. Enumerate empty squares in the legal half:
   - Use `WIDTH_`, `HEIGHT_`, `get_line_of_scrimage_x(team)`, and `Position { x, y }`.
   - Skip occupied squares and `pos.is_out()`.
4. For each candidate `pos`, compute counts after placement:
   - `new_los = los + is_los(pos)`
   - `new_north = north + is_north_wing(pos)`
   - `new_south = south + is_south_wing(pos)`
5. Reject candidate if:
   - `new_north > 2` or `new_south > 2`.
6. Feasibility check (prevents dead-ends):
   - `remaining_players_after = remaining_to_place - 1`.
   - `remaining_los_needed = max(0, min_los - new_los)`.
   - `remaining_los_squares = empty_los_squares - is_los(pos)`.
   - Require both:
     - `remaining_players_after >= remaining_los_needed` and
     - `remaining_los_squares >= remaining_los_needed`.
7. Return all candidates that pass. If `remaining_players_after == remaining_los_needed`, only LOS squares should survive by design.

Notes:
- `empty_los_squares` counts empty squares on the LOS line within `LINE_OF_SCRIMMAGE_Y_RANGE` on the team’s half.
- This is O(pitch_size) per call, which is fine for setup.

## Integration Points
- Manual setup procedure (future):
  - When it is the team’s turn to place a player, call `get_legal_setup_positions(team)`.
  - Use `AvailableActions::insert_positional(PosAT::SelectPosition, positions)` (or a new setup-specific `PosAT` if preferred) to surface legal squares to bots and UIs.
  - After each placement, re-call to update availability.
- Keep `GameState::is_setup_legal` as final validation when `EndSetup` is chosen.

## Tests (TDD)
Add unit tests near `src/core/gamestate.rs` or new `src/core/placement_finder.rs`:
1. **Half constraint**: all returned positions are on the team’s half.
2. **Wing cap**: after placing two in `NORTH_WING_Y_RANGE`, no returned positions are in north wing.
3. **LOS feasibility**:
   - With `los = 1`, `remaining_to_place = 2`, `min_los = 3`, only LOS squares are returned.
   - With `los >= min_los`, returned positions may include non-LOS squares.
4. **Occupied squares**: positions already occupied are never returned.
5. **Team direction**: verify both Home and Away halves behave correctly.

## Future Extensions
- Include player-specific constraints if any skills (or kickoff events) alter setup legality.
- Add a “reason” or “zone type” in a richer API for UI hints.
- Consider caching results in `AvailableActions` if manual setup polls heavily.
