//! `GameState → tensor` encoder — the single source of feature layout,
//! shared verbatim by the offline prepare step and the live evaluator.
//!
//! Two views of one encoding. [`encode_raw`] is the source of truth and
//! yields the spatial planes as **raw `u8` counts** — every spatial channel
//! is an integer quantity (a flag, a characteristic, a tackle-zone count), so
//! `u8` is exact, not a quantisation. [`encode`] is that divided by
//! [`spatial_channel_scales`], which is what the live evaluator feeds the
//! network. The offline corpus stores the `u8` form (4× smaller on disk) and
//! the trainer applies the same scales, read from the manifest.
//!
//! Output ([`Encoded`]):
//! - `spatial`: `C × H × W` `f32`, flat in C-major/row-major order
//!   (`idx = c*H*W + y*W + x`), i.e. PyTorch `NCHW` per-sample. `H`/`W`
//!   are the runtime `board_dims` **including** the 2-cell OOB border, so
//!   a `Position` indexes the tensor directly; border cells are flagged
//!   by the `oob` plane and always masked out of the policy.
//! - `global`: `F` non-spatial features, mover-perspective.
//! - `h`/`w`/`mover`: the concrete board shape + whose move it is.
//!
//! Everything is **mover-centric**: `us_present` marks the team to move,
//! `them_present` the opponent, and the whole board is canonicalised (mover
//! attacks toward `x=1`) via [`crate::perspective`].
//!
//! **Ownership is stated once.** A square holds at most one player, so the
//! two `present` planes already say whose player is in a cell; every other
//! per-player plane (status flags, characteristics, the 39 skills) is
//! therefore stored **unpaired**, for whichever player is there. This is
//! lossless, and recoverable in the stem: for binary planes `a` and
//! `us_present = b`, `ReLU(a + b - 1.5)` is "us has `a`" and
//! `ReLU(a - b - 0.5)` is "them has `a`" — one filter each. For a
//! characteristic `v`, `ReLU(v + b - 1)` is the mover's `v` **provided
//! `v <= 1`**, which is why the normalisers below are the engine's
//! `PlayerStats::MAX_*` caps rather than round numbers.
//!
//! The tackle-zone planes stay paired: they are neighbour counts, not
//! attached to the player standing in the cell, and both teams can cover one
//! square at once.

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{other_team, BallState, PlayerStats, Position, TeamType};
use botbowl_engine::core::pathing::PathFinder;
use botbowl_engine::core::table::Skill;

use crate::perspective::{canonical_pos, mover_for};

/// The two ownership planes: `us_present`, `them_present`. The only place a
/// side is encoded, and the key every other per-player plane is read against.
const C_US_PRESENT: usize = 0;
const C_THEM_PRESENT: usize = 1;

/// First channel of the shared (unpaired) per-player block.
const PLAYER_BASE: usize = 2;
/// Offsets within the shared per-player block, relative to [`PLAYER_BASE`].
/// The channel order, the name list and the scale vector all index through
/// these, so the three cannot drift apart.
const P_STANDING: usize = 0;
const P_STUNNED: usize = 1;
const P_USED: usize = 2;
const P_MOVEMENT: usize = 3;
const P_ST: usize = 4;
const P_MA: usize = 5;
const P_AG: usize = 6;
const P_AV: usize = 7;
/// Non-skill per-player planes: the eight above.
const PLAYER_SCALARS: usize = 8;

/// First of the per-skill planes — one per [`Skill`] variant, at
/// `SKILL_BASE + skill.index()`.
const SKILL_BASE: usize = PLAYER_BASE + PLAYER_SCALARS;

/// Skill planes, in channel order: **every** [`Skill`] variant, ordered by
/// [`Skill::index`]. Derived from the engine's enum rather than a hand-picked
/// subset, so a net can reason about any skill the engine can field.
const SKILL_PLANES: [Skill; Skill::COUNT] = Skill::ALL;

/// First of the planes that are not per-player.
const SHARED_BASE: usize = SKILL_BASE + Skill::COUNT;

/// Spatial channel count `C`.
pub const SPATIAL_CHANNELS: usize = SHARED_BASE + 12;
/// Non-spatial feature count `F`.
pub const GLOBAL_FEATURES: usize = 18;

// Planes that belong to no single player.
const C_BALL_GROUND: usize = SHARED_BASE;
const C_BALL_AIR: usize = SHARED_BASE + 1;
const C_BALL_CARRIER: usize = SHARED_BASE + 2;
const C_ACTIVE: usize = SHARED_BASE + 3;
const C_US_TZ: usize = SHARED_BASE + 4;
const C_THEM_TZ: usize = SHARED_BASE + 5;
const C_OOB: usize = SHARED_BASE + 6;
/// The endzone columns. A conv tower is translation-equivariant and `oob` is
/// symmetric in `x` (`is_out` is `x <= 0 || x >= width - 1`), so **nothing
/// else in the tensor says which end the mover attacks** — canonicalisation
/// fixes the convention but cannot communicate it. Without these the policy
/// head cannot prefer a move toward the scoring end over the identical move
/// away from it.
const C_US_TD_ZONE: usize = SHARED_BASE + 7;
const C_THEM_TD_ZONE: usize = SHARED_BASE + 8;
/// Probability that the **active player** reaches this square, as the
/// pathfinder computes it — the move's cumulative success chance across every
/// dodge, GFI and pickup on the way. `0` means "no path here"; a real path is
/// floored at `1/255` so the two can never be confused.
///
/// This is the one lossy plane: the pathfinder's `f32` is quantised to 1/255.
/// [`encode_raw`] stores the quantised value and [`encode`] divides by
/// [`PATH_PROB_NORM`], so the raw/normalised identity still holds exactly —
/// the loss is at the input, not between the two views.
const C_PATH_PROB: usize = SHARED_BASE + 9;
/// Plan 042 (schema v7): board geometry the net can otherwise only infer
/// from its distance to the zero-padded tensor edge — which is exactly the
/// shortcut mixed-size training exists to take away. Both are in the
/// **canonical** frame (mover attacks toward `x = 1`), zero on OOB cells.
///
/// `dist_to_us_endzone` is `x - 1`: "three squares from scoring" is a direct
/// input instead of something to read off the border. Its opposite-end
/// twin is `playable_w - 1 - this`, recoverable once the width is a global.
/// Absolute coordinate planes were rejected: they are the memorisable
/// feature, this is the relative one that transfers.
const C_DIST_US_ENDZONE: usize = SHARED_BASE + 10;
/// `min(y - 1, height - 2 - y)`: how close to either sideline.
const C_DIST_SIDELINE: usize = SHARED_BASE + 11;

// Per-player normalisers. These are the engine's characteristic caps, not
// arbitrary round numbers: every per-player plane must land in `[0, 1]` or the
// side-factored encoding above stops being exactly recoverable. Raising a cap
// in `PlayerStats` widens the divisor here automatically.
const MOVE_NORM: f32 = PlayerStats::MAX_MOVEMENT as f32;
const ST_NORM: f32 = PlayerStats::MAX_ST as f32;
const MA_NORM: f32 = PlayerStats::MAX_MA as f32;
const AG_NORM: f32 = PlayerStats::MAX_AG as f32;
const AV_NORM: f32 = PlayerStats::MAX_AV as f32;
/// Full `u8` range: path probabilities are a fraction, not a characteristic.
const PATH_PROB_NORM: f32 = 255.0;
// Geometry normalisers (plan 042): the full 26x15/11 pitch reads as 1.0, so
// every smaller board sits inside the unit interval and the same numbers
// serve the spatial distance planes and the global size features.
const PITCH_W_NORM: f32 = 26.0;
const PITCH_H_NORM: f32 = 15.0;
const TEAM_NORM: f32 = 11.0;
/// Largest sideline distance on the full pitch: `(15 - 1) / 2`.
const SIDELINE_DIST_NORM: f32 = 7.0;

// Global-feature divisors — arbitrary but fixed, so train and inference agree
// (both go through this file). BN in the tower absorbs the rest. The `[0, 1]`
// constraint does not apply: globals are not per-player.
const TURN_NORM: f32 = 8.0;
const RR_NORM: f32 = 3.0;
const HALF_NORM: f32 = 2.0;
const SCORE_NORM: f32 = 3.0;

/// Per-channel divisor taking [`EncodedRaw::spatial`] to [`Encoded::spatial`].
///
/// Written into the prepared corpus's manifest so the trainer normalises with
/// exactly these numbers — the `u8` corpus is meaningless without them, and a
/// second hand-maintained copy in Python is precisely the train/inference skew
/// this crate exists to make impossible. Length `SPATIAL_CHANNELS`.
pub fn spatial_channel_scales() -> Vec<f32> {
    let mut scales = vec![1.0; SPATIAL_CHANNELS];
    // Only the five characteristic planes are scaled; everything else is a
    // flag or a raw count.
    scales[PLAYER_BASE + P_MOVEMENT] = MOVE_NORM;
    scales[PLAYER_BASE + P_ST] = ST_NORM;
    scales[PLAYER_BASE + P_MA] = MA_NORM;
    scales[PLAYER_BASE + P_AG] = AG_NORM;
    scales[PLAYER_BASE + P_AV] = AV_NORM;
    scales[C_PATH_PROB] = PATH_PROB_NORM;
    scales[C_DIST_US_ENDZONE] = PITCH_W_NORM;
    scales[C_DIST_SIDELINE] = SIDELINE_DIST_NORM;
    scales
}

/// The raw, pre-normalisation encoding: identical layout to [`Encoded`], but
/// the spatial planes are the underlying integers.
#[derive(Debug, Clone)]
pub struct EncodedRaw {
    /// `C × H × W`, flat C-major/row-major (`idx = c*H*W + y*W + x`).
    pub spatial: Vec<u8>,
    /// `F` non-spatial features, mover-perspective. Not quantised: it is 15
    /// values per sample, and `score_diff` is signed.
    pub global: Vec<f32>,
    /// Board height (rows, tensor H) incl. OOB border.
    pub h: usize,
    /// Board width (cols, tensor W) incl. OOB border.
    pub w: usize,
    /// The team to move (perspective anchor).
    pub mover: TeamType,
}

/// A fully encoded decision node, ready to tensorise.
#[derive(Debug, Clone)]
pub struct Encoded {
    /// `C × H × W`, flat C-major/row-major (`idx = c*H*W + y*W + x`).
    pub spatial: Vec<f32>,
    /// `F` non-spatial features, mover-perspective.
    pub global: Vec<f32>,
    /// Board height (rows, tensor H) incl. OOB border.
    pub h: usize,
    /// Board width (cols, tensor W) incl. OOB border.
    pub w: usize,
    /// The team to move (perspective anchor).
    pub mover: TeamType,
}

/// Human-readable spatial channel names, in channel order — for the
/// manifest so a dataset self-describes its layout. Length `SPATIAL_CHANNELS`.
pub fn spatial_channel_names() -> Vec<String> {
    let mut names = vec!["us_present".to_string(), "them_present".to_string()];
    // Unpaired from here: these describe whichever player occupies the cell,
    // and the two planes above say whose it is.
    names.extend(
        ["standing", "stunned", "used", "movement_left", "st", "ma", "ag", "av"]
            .into_iter()
            .map(String::from),
    );
    for sk in SKILL_PLANES {
        names.push(format!("skill_{sk:?}").to_lowercase());
    }
    names.extend(
        [
            "ball_on_ground",
            "ball_in_air",
            "ball_carrier",
            "active_player",
            "us_tackle_zones",
            "them_tackle_zones",
            "oob",
            "us_td_zone",
            "them_td_zone",
            "path_prob",
            "dist_to_us_endzone",
            "dist_to_sideline",
        ]
        .into_iter()
        .map(String::from),
    );
    debug_assert_eq!(names.len(), SPATIAL_CHANNELS);
    names
}

/// Human-readable global feature names, in order. Length `GLOBAL_FEATURES`.
pub fn global_feature_names() -> Vec<String> {
    [
        "half",
        "us_turn",
        "them_turn",
        "us_score",
        "them_score",
        "score_diff",
        "us_rerolls",
        "them_rerolls",
        "us_reroll_usable",
        "them_reroll_usable",
        "blitz_available",
        "pass_available",
        "handoff_available",
        "foul_available",
        "turnover",
        // Plan 042 (schema v7): the board itself. A pooled, fully
        // convolutional value head has no other way to know how long the
        // pitch is, and the chance of a TD in the turns left depends on it.
        "playable_w",
        "playable_h",
        "team_size",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Encode a decision (or scoreable) state into mover-centric tensors,
/// normalised for the network. Exactly [`encode_raw`] divided by
/// [`spatial_channel_scales`].
pub fn encode(state: &GameState) -> Encoded {
    let raw = encode_raw(state);
    let scales = spatial_channel_scales();
    let plane = raw.h * raw.w;
    // One divisor per plane, so the inner loop is a plain division and not an
    // integer `i / plane` per element (that alone was most of `encode`'s
    // no-path cost). Division, not a reciprocal multiply: the raw/normalised
    // identity is pinned bit-for-bit and Python divides too.
    let mut spatial = Vec::with_capacity(raw.spatial.len());
    for (chunk, &scale) in raw.spatial.chunks_exact(plane).zip(scales.iter()) {
        spatial.extend(chunk.iter().map(|&v| v as f32 / scale));
    }
    Encoded {
        spatial,
        global: raw.global,
        h: raw.h,
        w: raw.w,
        mover: raw.mover,
    }
}

/// Encode a decision (or scoreable) state into mover-centric tensors, with
/// the spatial planes left as raw integer counts. See the module docs.
pub fn encode_raw(state: &GameState) -> EncodedRaw {
    let mover = mover_for(state);
    let dims = state.board_dims;
    let h = dims.height as usize;
    let w = dims.width as usize;
    let plane = h * w;
    let mut spatial = vec![0u8; SPATIAL_CHANNELS * plane];

    // Flat index for (channel, canonical position).
    let idx = |c: usize, pos: Position| -> usize { c * plane + (pos.y as usize) * w + (pos.x as usize) };
    let cpos = |pos: Position| canonical_pos(pos, dims, mover);

    // --- Per-player planes ---
    for p in state.get_players_on_pitch() {
        let pos = cpos(p.position);
        let ours = p.stats.team == mover;
        // The one ownership bit. Everything below is unpaired and read
        // against it — see the module docs.
        spatial[idx(if ours { C_US_PRESENT } else { C_THEM_PRESENT }, pos)] = 1;

        let mut set = |off: usize, v: u8| {
            spatial[idx(PLAYER_BASE + off, pos)] = v;
        };
        use botbowl_engine::core::model::PlayerStatus;
        set(P_STANDING, matches!(p.status, PlayerStatus::Up) as u8);
        set(P_STUNNED, matches!(p.status, PlayerStatus::Stunned) as u8);
        set(P_USED, p.used as u8);
        set(P_MOVEMENT, p.total_movement_left());
        set(P_ST, p.stats.str_);
        set(P_MA, p.stats.ma);
        set(P_AG, p.stats.ag);
        set(P_AV, p.stats.av);
        for sk in SKILL_PLANES {
            spatial[idx(SKILL_BASE + sk.index(), pos)] = p.has_skill(sk) as u8;
        }

        // Tackle zones this player exerts onto its (canonical) neighbours.
        if p.has_tackle_zone() {
            let tz_c = if ours { C_US_TZ } else { C_THEM_TZ };
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let nx = pos.x + dx;
                let ny = pos.y + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let np = Position::new((nx, ny));
                    spatial[idx(tz_c, np)] += 1;
                }
            }
        }
    }

    // --- Ball planes ---
    match state.ball {
        BallState::OnGround(pos) => spatial[idx(C_BALL_GROUND, cpos(pos))] = 1,
        BallState::InAir(pos) => spatial[idx(C_BALL_AIR, cpos(pos))] = 1,
        BallState::Carried(id) => {
            if let Ok(carrier) = state.get_player(id) {
                spatial[idx(C_BALL_CARRIER, cpos(carrier.position))] = 1;
            }
        }
        BallState::OffPitch => {}
    }

    // --- Active player ---
    if let Some(id) = state.info.active_player {
        if let Ok(p) = state.get_player(id) {
            spatial[idx(C_ACTIVE, cpos(p.position))] = 1;
        }
    }

    // --- Out-of-bounds mask (logical border) ---
    for y in 0..h {
        for x in 0..w {
            let pos = Position::new((x as i8, y as i8));
            if dims.is_out(pos) {
                spatial[idx(C_OOB, pos)] = 1;
            }
        }
    }

    // --- Endzones ---
    // Read from the engine rather than assuming the canonical `x = 1`, so the
    // planes cannot drift from the rules; the canonicalisation then puts the
    // mover's own endzone at `x = 1` every time (asserted in the tests).
    for (team, channel) in [(mover, C_US_TD_ZONE), (other_team(mover), C_THEM_TD_ZONE)] {
        let x = dims.endzone_x(team);
        for y in 0..h {
            let pos = Position::new((x, y as i8));
            if !dims.is_out(pos) {
                spatial[idx(channel, cpos(pos))] = 1;
            }
        }
    }

    // --- Board geometry (plan 042) ---
    // Defined directly in the canonical frame, so no mirror is applied: the
    // mover's endzone is `x = 1` whoever moves, and the sideline distance is
    // symmetric in `y`. Tests pin both under either mover.
    for y in 1..h - 1 {
        let sideline = (y - 1).min((h - 2) - y);
        for x in 1..w - 1 {
            let pos = Position::new((x as i8, y as i8));
            spatial[idx(C_DIST_US_ENDZONE, pos)] = (x - 1) as u8;
            spatial[idx(C_DIST_SIDELINE, pos)] = sideline as u8;
        }
    }

    // --- Path probabilities for the active player ---
    //
    // **Recomputed, never read from `state.path_buffer`.** That field is
    // `#[serde(skip)]` and `#[derivative(PartialEq = "ignore")]`, so a state
    // that has been through the corpus (`prepare` deserialises JSONL) comes
    // back with `has_paths == true` and an empty buffer — the plane would be
    // all zeros in training and populated at inference, silent skew of exactly
    // the kind this crate is built to prevent. Ignoring it in `PartialEq` is
    // the second problem: two states that recombine to one DAG node may differ
    // in the buffer, which would make the prior impure.
    //
    // `PathFinder::player_paths` is a pure function of `(state, id)` and
    // reproduces the buffer's contents exactly
    // (`recomputed_paths_match_the_engines_own_buffer`). The `has_paths` gate
    // *is* serialised and *is* in `PartialEq`, so gating on it stays pure.
    if state.available_actions.has_paths() {
        if let Some(id) = state.info.active_player {
            if let Ok(paths) = PathFinder::player_paths(state, id) {
                for (pos, node) in paths.iter_position() {
                    let Some(node) = node else { continue };
                    // A reachable square always has p > 0, so floor the
                    // quantisation at 1: a long, unlikely path must not read
                    // as "unreachable".
                    let q = (node.prob * PATH_PROB_NORM).round().clamp(1.0, PATH_PROB_NORM) as u8;
                    spatial[idx(C_PATH_PROB, cpos(pos))] = q;
                }
            }
        }
    }

    // --- Global (non-spatial) features, mover-perspective ---
    let (us, them) = match mover {
        TeamType::Home => (&state.home, &state.away),
        TeamType::Away => (&state.away, &state.home),
    };
    let (us_turn, them_turn) = match mover {
        TeamType::Home => (state.info.home_turn, state.info.away_turn),
        TeamType::Away => (state.info.away_turn, state.info.home_turn),
    };
    let global = vec![
        state.info.half as f32 / HALF_NORM,
        us_turn as f32 / TURN_NORM,
        them_turn as f32 / TURN_NORM,
        us.score as f32 / SCORE_NORM,
        them.score as f32 / SCORE_NORM,
        (us.score as f32 - them.score as f32) / SCORE_NORM,
        us.rerolls as f32 / RR_NORM,
        them.rerolls as f32 / RR_NORM,
        us.can_use_reroll() as u8 as f32,
        them.can_use_reroll() as u8 as f32,
        state.info.blitz_available as u8 as f32,
        state.info.pass_available as u8 as f32,
        state.info.handoff_available as u8 as f32,
        state.info.foul_available as u8 as f32,
        state.info.turnover as u8 as f32,
        (dims.width - 2) as f32 / PITCH_W_NORM,
        (dims.height - 2) as f32 / PITCH_H_NORM,
        dims.team_size as f32 / TEAM_NORM,
    ];
    debug_assert_eq!(global.len(), GLOBAL_FEATURES);

    EncodedRaw {
        spatial,
        global,
        h,
        w,
        mover,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::{BoardDims, Position};

    #[test]
    fn name_lengths_match_channel_counts() {
        assert_eq!(spatial_channel_names().len(), SPATIAL_CHANNELS);
        assert_eq!(global_feature_names().len(), GLOBAL_FEATURES);
        // 2 present + 8 shared per-player + 39 skill + 12 shared. Pinned so a
        // change to the engine's `Skill` enum shows up here as a failing
        // test, next to the `NN_SCHEMA_VERSION` bump it requires.
        assert_eq!(SPATIAL_CHANNELS, 2 + PLAYER_SCALARS + Skill::COUNT + 12);
        assert_eq!(SPATIAL_CHANNELS, 61);
        assert_eq!(spatial_channel_scales().len(), SPATIAL_CHANNELS);
        assert_eq!(GLOBAL_FEATURES, 18);
    }

    /// Plan 042: the geometry planes are canonical-frame distances, the same
    /// under either mover, zero on the border, and the size globals read the
    /// runtime board. This is what makes a 12x5 and a 16x9 sample
    /// distinguishable to a translation-equivariant tower by something other
    /// than its distance to the zero padding.
    #[test]
    fn geometry_planes_and_size_globals_follow_the_runtime_board() {
        use botbowl_engine::core::gamestate::DiceMode;
        use botbowl_engine::core::table::SimpleAT;
        botbowl_engine::skip_if_board_smaller_than!(16, 9);
        let dims = BoardDims::new(16, 9, 4);
        let mut home = GameStateBuilder::new().with_board_dims(dims).build();
        home.set_logging_state(false);
        let mut away = GameStateBuilder::new().with_board_dims(dims).build();
        away.set_logging_state(false);
        away.set_dice_mode(DiceMode::RollDice);
        away.set_seed(0);
        away.step_simple(SimpleAT::EndTurn);
        assert_eq!(mover_for(&away), TeamType::Away, "expected the turn to pass");

        for state in [&home, &away] {
            let raw = encode_raw(state);
            let (h, w) = (raw.h, raw.w);
            let plane = h * w;
            let at = |c: usize, x: usize, y: usize| raw.spatial[c * plane + y * w + x];
            for y in 0..h {
                for x in 0..w {
                    let oob = x == 0 || x == w - 1 || y == 0 || y == h - 1;
                    if oob {
                        assert_eq!(at(C_DIST_US_ENDZONE, x, y), 0);
                        assert_eq!(at(C_DIST_SIDELINE, x, y), 0);
                    } else {
                        assert_eq!(at(C_DIST_US_ENDZONE, x, y) as usize, x - 1, "endzone dist at ({x},{y})");
                        assert_eq!(
                            at(C_DIST_SIDELINE, x, y) as usize,
                            (y - 1).min(h - 2 - y),
                            "sideline dist at ({x},{y})"
                        );
                    }
                }
            }
            // The mover's endzone column is distance 0 — the two planes agree.
            for y in 1..h - 1 {
                assert_eq!(at(C_US_TD_ZONE, 1, y), 1);
                assert_eq!(at(C_DIST_US_ENDZONE, 1, y), 0);
                assert_eq!(at(C_DIST_US_ENDZONE, w - 2, y) as usize, w - 3);
            }
            let g = &raw.global;
            assert_eq!(g[GLOBAL_FEATURES - 3], 14.0 / PITCH_W_NORM);
            assert_eq!(g[GLOBAL_FEATURES - 2], 7.0 / PITCH_H_NORM);
            assert_eq!(g[GLOBAL_FEATURES - 1], 4.0 / TEAM_NORM);
        }
        // The normalised view stays inside the unit interval for any board
        // up to the full pitch, and is exactly raw / scale.
        let enc = encode(&home);
        let plane = enc.h * enc.w;
        for c in [C_DIST_US_ENDZONE, C_DIST_SIDELINE] {
            let max = enc.spatial[c * plane..(c + 1) * plane].iter().copied().fold(0.0f32, f32::max);
            assert!(max <= 1.0, "channel {c} reached {max}");
        }
    }

    #[test]
    fn raw_u8_planes_reproduce_the_normalised_ones_exactly() {
        // The whole basis for storing the corpus as u8: every spatial channel
        // is an integer over a fixed divisor, so the round-trip is exact and
        // not a quantisation. Bit equality, not a tolerance.
        let scales = spatial_channel_scales();
        assert_eq!(scales.len(), SPATIAL_CHANNELS);
        let state = GameStateBuilder::new_start_of_game();
        let raw = encode_raw(&state);
        let enc = encode(&state);
        let plane = enc.h * enc.w;
        assert_eq!(raw.spatial.len(), enc.spatial.len());
        for (i, (&r, &f)) in raw.spatial.iter().zip(&enc.spatial).enumerate() {
            assert_eq!(
                r as f32 / scales[i / plane],
                f,
                "channel {} cell {}",
                i / plane,
                i % plane
            );
        }
        assert_eq!(raw.global, enc.global);
        assert_eq!((raw.h, raw.w, raw.mover), (enc.h, enc.w, enc.mover));
    }

    /// A u8 plane cannot hold a value that overflows it. ST/MA/AG/AV and
    /// movement are small by construction; a tackle-zone cell tops out at the
    /// 8 neighbours a square has.
    #[test]
    fn raw_planes_fit_in_u8_on_a_crowded_board() {
        let state = GameStateBuilder::new_at_kickoff();
        let raw = encode_raw(&state);
        let plane = raw.h * raw.w;
        let max_tz = raw.spatial[C_US_TZ * plane..(C_US_TZ + 1) * plane]
            .iter()
            .chain(&raw.spatial[C_THEM_TZ * plane..(C_THEM_TZ + 1) * plane])
            .copied()
            .max()
            .unwrap_or(0);
        assert!(max_tz <= 8, "a tackle-zone cell reached {max_tz}");
    }

    #[test]
    fn every_skill_has_its_own_plane_and_the_names_agree() {
        let names = spatial_channel_names();
        for sk in Skill::ALL {
            assert_eq!(names[SKILL_BASE + sk.index()], format!("skill_{sk:?}").to_lowercase());
        }
        // The skill planes are unpaired, so no name may carry a side.
        for name in &names[PLAYER_BASE..SHARED_BASE] {
            assert!(
                !name.starts_with("us_") && !name.starts_with("them_"),
                "{name} is side-tagged but lives in the shared per-player block"
            );
        }
    }

    #[test]
    fn a_skill_outside_the_old_six_reaches_its_plane() {
        // The regression the all-skills change was about: `Guard` used to
        // have no plane at all, so a guard and a plain lineman encoded
        // identically.
        use botbowl_engine::core::model::{PlayerStats, TeamType};
        let mut stats = PlayerStats::new_lineman(TeamType::Home);
        stats.give_skill(Skill::Guard);
        let pos = Position::new((5, 5));
        let mut builder = GameStateBuilder::new();
        builder.add_player_details(pos, TeamType::Home, stats);
        let state = builder.build();
        assert_eq!(mover_for(&state), TeamType::Home, "test assumes no x-mirror");

        let enc = encode(&state);
        let plane = enc.h * enc.w;
        let at = |c: usize| enc.spatial[c * plane + (pos.y as usize) * enc.w + (pos.x as usize)];
        assert_eq!(at(C_US_PRESENT), 1.0, "player is present, on our side");
        assert_eq!(at(C_THEM_PRESENT), 0.0);
        assert_eq!(at(SKILL_BASE + Skill::Guard.index()), 1.0, "guard plane is set");
        assert_eq!(at(SKILL_BASE + Skill::Block.index()), 0.0, "an unheld skill stays 0");
    }

    /// A state with the active player mid-move, where the engine has filled
    /// its own `path_buffer` — the fixture the path-plane tests need.
    fn state_with_paths() -> GameState {
        use botbowl_engine::core::model::Action;
        use botbowl_engine::core::table::PosAT;
        let carrier = Position::new((5, 5));
        let mut builder = GameStateBuilder::new();
        builder
            .add_home_player(carrier)
            .add_away_player(Position::new((8, 5)))
            .add_ball_pos(carrier);
        let mut state = builder.build();
        state.set_logging_state(false);
        state
            .step(Action::Positional(PosAT::StartMove, carrier))
            .expect("activate the carrier");
        assert!(state.available_actions.has_paths(), "fixture has no path offerings");
        state
    }

    /// The encoder recomputes the paths instead of reading `path_buffer`.
    /// That is only sound if the recomputation agrees with the engine's own
    /// buffer, square for square — this is what pins it.
    #[test]
    fn recomputed_paths_match_the_engines_own_buffer() {
        let state = state_with_paths();
        let id = state.info.active_player.expect("an active player");
        let cached = state.get_paths().expect("engine filled the buffer");
        let recomputed = PathFinder::player_paths(&state, id).expect("recompute");

        let mut compared = 0;
        for (pos, node) in cached.iter_position() {
            let mine = &recomputed[pos];
            match (node, mine) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.prob, b.prob, "probability differs at {pos:?}");
                    compared += 1;
                }
                _ => panic!("reachability differs at {pos:?}"),
            }
        }
        assert!(compared > 20, "fixture only compared {compared} squares");
    }

    /// Plan 038 open item B, settled: the engine's buffer is **not** a
    /// substitute for the recompute even when it is present. At a `BlockAction`
    /// decision the buffer holds one direct-block offering per adjacent victim
    /// (`new_direct_block_node`) while `has_paths` is just as true as it is
    /// mid-move; the plane is defined as the mover's *movement* probabilities
    /// and the recompute gives exactly that. An encoder that read the buffer
    /// "when present" would encode the two differently, silently, at every
    /// block decision. Keep this test if anyone revisits the fast path: it is
    /// the counter-example.
    #[test]
    fn the_engines_buffer_is_not_the_path_plane_at_a_block_decision() {
        use botbowl_engine::core::model::Action;
        use botbowl_engine::core::table::PosAT;
        let attacker = Position::new((5, 5));
        let mut builder = GameStateBuilder::new();
        builder
            .add_home_player(attacker)
            .add_away_player(Position::new((6, 5)))
            .add_away_player(Position::new((6, 6)));
        let mut state = builder.build();
        state.set_logging_state(false);
        state
            .step(Action::Positional(PosAT::StartBlock, attacker))
            .expect("start the block");
        assert!(
            state.available_actions.has_paths(),
            "block offerings live in the path buffer"
        );
        let id = state.info.active_player.expect("the attacker is active");

        let cached = state.get_paths().expect("engine filled the buffer");
        let recomputed = PathFinder::player_paths(&state, id).expect("recompute");
        let cached_n = cached.iter().filter(|n| n.is_some()).count();
        let recomputed_n = recomputed.iter().filter(|n| n.is_some()).count();
        assert_eq!(cached_n, 2, "one block offering per adjacent standing opponent");
        assert!(
            recomputed_n > cached_n,
            "the buffer ({cached_n} squares) and the recompute ({recomputed_n}) must differ here, \
             or item B's premise has changed and this test — not the encoder — needs revisiting"
        );
        assert!(
            cached
                .iter_position()
                .all(|(_, n)| n.as_ref().is_none_or(|n| n.get_action_type() == PosAT::Block)),
            "block offerings are Block actions, not moves"
        );
    }

    /// The reason for recomputing, stated as a test: a state that has been
    /// through the corpus keeps `has_paths == true` but loses the buffer
    /// (`#[serde(skip)]`), so an encoder that read `path_buffer` would emit an
    /// all-zero plane in training and a populated one at inference. Both
    /// encodings must be identical.
    #[test]
    fn the_path_plane_survives_the_corpus_round_trip() {
        let state = state_with_paths();
        let json = serde_json::to_string(&state).expect("serialise");
        let restored: GameState = serde_json::from_str(&json).expect("deserialise");

        assert!(restored.available_actions.has_paths(), "the gate is serialised");
        assert!(
            restored.get_paths().is_none(),
            "path_buffer is #[serde(skip)] — if this ever starts round-tripping, \
             the comment in encode_raw needs revisiting, not this assert relaxing"
        );

        let live = encode(&state);
        let back = encode(&restored);
        let plane = live.h * live.w;
        let range = C_PATH_PROB * plane..(C_PATH_PROB + 1) * plane;
        assert_eq!(
            live.spatial[range.clone()],
            back.spatial[range],
            "the path plane differs across the corpus round-trip"
        );
        assert_eq!(live.spatial, back.spatial, "some other plane differs too");
    }

    /// `0` must mean "unreachable" and nothing else: a reachable square always
    /// carries `p > 0`, so its quantised value is floored at 1.
    #[test]
    fn zero_in_the_path_plane_means_unreachable() {
        let state = state_with_paths();
        let id = state.info.active_player.unwrap();
        let paths = PathFinder::player_paths(&state, id).unwrap();
        let raw = encode_raw(&state);
        let dims = state.board_dims;
        let mover = mover_for(&state);
        let plane = raw.h * raw.w;

        let mut reachable = 0;
        for (pos, node) in paths.iter_position() {
            let c = canonical_pos(pos, dims, mover);
            let v = raw.spatial[C_PATH_PROB * plane + (c.y as usize) * raw.w + (c.x as usize)];
            match node {
                Some(n) => {
                    assert!(n.prob > 0.0, "the pathfinder produced a zero-probability path");
                    assert!(v >= 1, "reachable square {pos:?} (p={}) quantised to 0", n.prob);
                    reachable += 1;
                }
                None => assert_eq!(v, 0, "unreachable square {pos:?} has a probability"),
            }
        }
        assert!(reachable > 20, "fixture only had {reachable} reachable squares");
    }

    /// With no active player the plane is simply empty — and identically so in
    /// training and inference, which is all that is required.
    #[test]
    fn the_path_plane_is_empty_when_no_paths_are_offered() {
        let state = GameStateBuilder::new_start_of_game();
        assert!(!state.available_actions.has_paths());
        let raw = encode_raw(&state);
        let plane = raw.h * raw.w;
        assert!(
            raw.spatial[C_PATH_PROB * plane..(C_PATH_PROB + 1) * plane]
                .iter()
                .all(|&v| v == 0),
            "path plane is populated with no offerings"
        );
    }

    /// The mover's endzone must always land on the canonical `x = 1` column
    /// and the opponent's on `x = w - 2`, for **both** movers — that is the
    /// whole point of the canonicalisation, and the only thing in the tensor
    /// that says which way the mover attacks.
    #[test]
    fn td_zones_are_canonical_for_either_mover() {
        use botbowl_engine::core::gamestate::DiceMode;
        use botbowl_engine::core::table::SimpleAT;

        let mut home_to_move = GameStateBuilder::new().build();
        home_to_move.set_logging_state(false);
        assert_eq!(mover_for(&home_to_move), TeamType::Home);

        // Hand the turn over so the Away branch (x-mirrored) is covered too.
        let mut away_to_move = GameStateBuilder::new().build();
        away_to_move.set_logging_state(false);
        away_to_move.set_dice_mode(DiceMode::RollDice);
        away_to_move.set_seed(0);
        away_to_move.step_simple(SimpleAT::EndTurn);
        assert_eq!(mover_for(&away_to_move), TeamType::Away, "expected the turn to pass");

        for state in [&home_to_move, &away_to_move] {
            let enc = encode(state);
            let plane = enc.h * enc.w;
            let at = |c: usize, x: usize, y: usize| enc.spatial[c * plane + y * enc.w + x];
            let mut us_cells = 0;
            let mut them_cells = 0;
            for y in 0..enc.h {
                for x in 0..enc.w {
                    let (us, them) = (at(C_US_TD_ZONE, x, y), at(C_THEM_TD_ZONE, x, y));
                    assert!(us == 0.0 || them == 0.0, "a cell is both endzones");
                    if us == 1.0 {
                        assert_eq!(x, 1, "mover's endzone must be the canonical x = 1");
                        us_cells += 1;
                    }
                    if them == 1.0 {
                        assert_eq!(x, enc.w - 2, "opponent's endzone must be x = w - 2");
                        them_cells += 1;
                    }
                }
            }
            // One cell per in-bounds row.
            assert_eq!(us_cells, enc.h - 2, "mover endzone is not a full column");
            assert_eq!(them_cells, enc.h - 2, "opponent endzone is not a full column");
        }
    }

    /// The claim the side-factored layout rests on: a square holds at most one
    /// player, so `(us_present, them_present)` is never `(1, 1)` and the two
    /// planes alone recover the owner of every per-player feature.
    #[test]
    fn the_two_present_planes_partition_the_occupied_squares() {
        let state = GameStateBuilder::new_at_kickoff();
        let enc = encode(&state);
        let plane = enc.h * enc.w;
        let mut occupied = 0usize;
        for cell in 0..plane {
            let us = enc.spatial[C_US_PRESENT * plane + cell];
            let them = enc.spatial[C_THEM_PRESENT * plane + cell];
            assert!(us == 0.0 || them == 0.0, "cell {cell} claims a player on both sides");
            if us + them > 0.0 {
                occupied += 1;
            } else {
                // No player ⇒ every unpaired per-player plane must be clear,
                // or a feature would float free of any owner.
                for c in PLAYER_BASE..SHARED_BASE {
                    assert_eq!(enc.spatial[c * plane + cell], 0.0, "channel {c} set on an empty cell");
                }
            }
        }
        assert_eq!(
            occupied,
            state.get_players_on_pitch().count(),
            "present planes lost a player"
        );
    }

    /// Every per-player plane must land in `[0, 1]`: the stem recovers the
    /// mover's copy of a characteristic `v` as `ReLU(v + us_present - 1)`,
    /// which is only exact while `v <= 1`. `movement_left` is the one that
    /// used to break it (MA 9 + 2 GFI over a divisor of 10).
    #[test]
    fn per_player_planes_are_normalised_into_the_unit_interval() {
        use botbowl_engine::core::model::{PlayerStats, TeamType};
        let mut stats = PlayerStats::new_lineman(TeamType::Home);
        stats.ma = PlayerStats::MAX_MA;
        stats.str_ = PlayerStats::MAX_ST;
        stats.ag = PlayerStats::MAX_AG;
        stats.av = PlayerStats::MAX_AV;
        let pos = Position::new((5, 5));
        let mut builder = GameStateBuilder::new();
        builder.add_player_details(pos, TeamType::Home, stats);
        let enc = encode(&builder.build());
        let plane = enc.h * enc.w;
        for c in 0..SHARED_BASE {
            let max = enc.spatial[c * plane..(c + 1) * plane]
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            assert!(max <= 1.0, "channel {c} reached {max}, above the unit interval");
        }
    }

    #[test]
    fn shape_matches_board_dims() {
        let state = GameStateBuilder::new_start_of_game();
        let enc = encode(&state);
        let dims = state.board_dims;
        assert_eq!(enc.h, dims.height as usize);
        assert_eq!(enc.w, dims.width as usize);
        assert_eq!(enc.spatial.len(), SPATIAL_CHANNELS * enc.h * enc.w);
        assert_eq!(enc.global.len(), GLOBAL_FEATURES);
    }

    #[test]
    fn variable_dims_16x9_matches_the_channel_count() {
        // 14x7 playable tier → engine 16x9. Requires a build whose capacity
        // is at least 16x9 (the default 28x17 is).
        botbowl_engine::skip_if_board_smaller_than!(16, 9);
        let dims = BoardDims::new(16, 9, 4);
        let state = GameStateBuilder::new().with_board_dims(dims).build();
        let enc = encode(&state);
        assert_eq!(enc.h, 9);
        assert_eq!(enc.w, 16);
        assert_eq!(enc.spatial.len(), SPATIAL_CHANNELS * 9 * 16);
    }

    #[test]
    fn golden_present_plane_marks_a_placed_player() {
        // Default builder → Home receives → Home to move (no mirror). A
        // Home lineman at (5,5) must light up the us-present plane exactly
        // there.
        let mut builder = GameStateBuilder::new();
        builder.add_home_player(Position::new((5, 5)));
        let state = builder.build();
        let enc = encode(&state);
        assert_eq!(enc.mover, TeamType::Home, "default builder should leave Home to move");
        let w = enc.w;
        let flat = 5 * w + 5; // channel 0 (us_present) at (x=5,y=5)
        assert_eq!(enc.spatial[flat], 1.0, "expected present plane hit at (5,5)");
    }

    #[test]
    fn mirror_consistency_us_present_hits_canonical_squares() {
        // Whoever moves, the us-present plane must have a hit at the
        // canonical square of every fielded player on the mover's team —
        // the mirror is applied consistently (plan 017 mirror invariant).
        // Squares chosen relative to the compiled board so the test also
        // holds on the small tiers (a literal (10, 8) is off a 14x7 pitch).
        let d = BoardDims::default();
        let mut b = GameStateBuilder::new();
        b.add_home_player(Position::new((d.width / 2 - 2, d.height / 2)));
        b.add_away_player(Position::new((d.width / 2 + 2, d.height - 2)));
        let state = b.build();
        let enc = encode(&state);
        let dims = state.board_dims;
        let w = enc.w;
        let mover = enc.mover;
        for p in state.get_players_on_pitch().filter(|p| p.stats.team == mover) {
            let pos = canonical_pos(p.position, dims, mover);
            let flat = pos.y as usize * w + pos.x as usize; // channel 0 (us_present)
            assert_eq!(enc.spatial[flat], 1.0, "us-present missing at canonical {pos:?}");
        }
    }
}
