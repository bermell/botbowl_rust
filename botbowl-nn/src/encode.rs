//! `GameState → tensor` encoder — the single source of feature layout,
//! shared verbatim by the offline prepare step and the live evaluator.
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
//! Everything is **mover-centric**: side-0 planes are the team to move
//! ("us"), side-1 the opponent ("them"), and the whole board is
//! canonicalised (mover attacks toward `x=1`) via [`crate::perspective`].

use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{BallState, Position, TeamType};
use botbowl_engine::core::table::Skill;

use crate::perspective::{canonical_pos, mover_for};

/// Per-side plane count: present, standing, stunned, used, movement_left,
/// ST, MA, AG, AV, then the 6 skill planes.
pub const PER_SIDE: usize = 15;
/// Spatial channel count `C`.
pub const SPATIAL_CHANNELS: usize = 2 * PER_SIDE + 7; // = 37
/// Non-spatial feature count `F`.
pub const GLOBAL_FEATURES: usize = 15;

/// Skill planes, in channel order. Must match [`skill_plane_names`].
const SKILL_PLANES: [Skill; 6] = [
    Skill::Dodge,
    Skill::Throw,
    Skill::Block,
    Skill::Catch,
    Skill::SureHands,
    Skill::SureFeet,
];

// Global-spatial plane offsets (after the two per-side blocks).
const C_BALL_GROUND: usize = 2 * PER_SIDE; // 30
const C_BALL_AIR: usize = 2 * PER_SIDE + 1; // 31
const C_BALL_CARRIER: usize = 2 * PER_SIDE + 2; // 32
const C_ACTIVE: usize = 2 * PER_SIDE + 3; // 33
const C_US_TZ: usize = 2 * PER_SIDE + 4; // 34
const C_THEM_TZ: usize = 2 * PER_SIDE + 5; // 35
const C_OOB: usize = 2 * PER_SIDE + 6; // 36

// Light normalisation divisors — arbitrary but fixed, so train and
// inference agree (both go through this file). BN in the tower absorbs
// the rest.
const MA_NORM: f32 = 10.0;
const ST_NORM: f32 = 8.0;
const AG_NORM: f32 = 6.0;
const AV_NORM: f32 = 12.0;
const MOVE_NORM: f32 = 10.0;
const TURN_NORM: f32 = 8.0;
const RR_NORM: f32 = 3.0;
const HALF_NORM: f32 = 2.0;
const SCORE_NORM: f32 = 3.0;

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
    let mut names = Vec::with_capacity(SPATIAL_CHANNELS);
    for side in ["us", "them"] {
        for base in [
            "present",
            "standing",
            "stunned",
            "used",
            "movement_left",
            "st",
            "ma",
            "ag",
            "av",
        ] {
            names.push(format!("{side}_{base}"));
        }
        for sk in SKILL_PLANES {
            names.push(format!("{side}_skill_{sk:?}").to_lowercase());
        }
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
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Encode a decision (or scoreable) state into mover-centric tensors.
pub fn encode(state: &GameState) -> Encoded {
    let mover = mover_for(state);
    let dims = state.board_dims;
    let h = dims.height as usize;
    let w = dims.width as usize;
    let plane = h * w;
    let mut spatial = vec![0.0f32; SPATIAL_CHANNELS * plane];

    // Flat index for (channel, canonical position).
    let idx = |c: usize, pos: Position| -> usize { c * plane + (pos.y as usize) * w + (pos.x as usize) };
    let cpos = |pos: Position| canonical_pos(pos, dims, mover);

    // --- Per-player planes ---
    for p in state.get_players_on_pitch() {
        let pos = cpos(p.position);
        let side_base = if p.stats.team == mover { 0 } else { PER_SIDE };
        let mut set = |off: usize, v: f32| {
            spatial[idx(side_base + off, pos)] = v;
        };
        set(0, 1.0); // present
        use botbowl_engine::core::model::PlayerStatus;
        set(1, matches!(p.status, PlayerStatus::Up) as u8 as f32); // standing
        set(2, matches!(p.status, PlayerStatus::Stunned) as u8 as f32); // stunned
        set(3, p.used as u8 as f32); // used
        set(4, p.total_movement_left() as f32 / MOVE_NORM); // movement_left
        set(5, p.stats.str_ as f32 / ST_NORM); // ST
        set(6, p.stats.ma as f32 / MA_NORM); // MA
        set(7, p.stats.ag as f32 / AG_NORM); // AG
        set(8, p.stats.av as f32 / AV_NORM); // AV
        for (i, sk) in SKILL_PLANES.into_iter().enumerate() {
            set(9 + i, p.has_skill(sk) as u8 as f32);
        }

        // Tackle zones this player exerts onto its (canonical) neighbours.
        if p.has_tackle_zone() {
            let tz_c = if p.stats.team == mover { C_US_TZ } else { C_THEM_TZ };
            for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let nx = pos.x + dx;
                let ny = pos.y + dy;
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let np = Position::new((nx, ny));
                    spatial[idx(tz_c, np)] += 1.0;
                }
            }
        }
    }

    // --- Ball planes ---
    match state.ball {
        BallState::OnGround(pos) => spatial[idx(C_BALL_GROUND, cpos(pos))] = 1.0,
        BallState::InAir(pos) => spatial[idx(C_BALL_AIR, cpos(pos))] = 1.0,
        BallState::Carried(id) => {
            if let Ok(carrier) = state.get_player(id) {
                spatial[idx(C_BALL_CARRIER, cpos(carrier.position))] = 1.0;
            }
        }
        BallState::OffPitch => {}
    }

    // --- Active player ---
    if let Some(id) = state.info.active_player {
        if let Ok(p) = state.get_player(id) {
            spatial[idx(C_ACTIVE, cpos(p.position))] = 1.0;
        }
    }

    // --- Out-of-bounds mask (logical border) ---
    for y in 0..h {
        for x in 0..w {
            let pos = Position::new((x as i8, y as i8));
            if dims.is_out(pos) {
                spatial[idx(C_OOB, pos)] = 1.0;
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
    ];
    debug_assert_eq!(global.len(), GLOBAL_FEATURES);

    Encoded {
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
        assert_eq!(SPATIAL_CHANNELS, 37);
        assert_eq!(GLOBAL_FEATURES, 15);
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
    fn variable_dims_16x9_gives_37_9_16() {
        // 14x7 playable tier → engine 16x9. Requires a build whose capacity
        // is at least 16x9 (the default 28x17 is).
        botbowl_engine::skip_if_board_smaller_than!(16, 9);
        let dims = BoardDims::new(16, 9, 4);
        let state = GameStateBuilder::new().with_board_dims(dims).build();
        let enc = encode(&state);
        assert_eq!(enc.h, 9);
        assert_eq!(enc.w, 16);
        assert_eq!(enc.spatial.len(), 37 * 9 * 16);
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
        let mut b = GameStateBuilder::new();
        b.add_home_player(Position::new((6, 4)));
        b.add_away_player(Position::new((10, 8)));
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
