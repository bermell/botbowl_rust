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
pub const SPATIAL_CHANNELS: usize = SHARED_BASE + 9;
/// Non-spatial feature count `F`.
pub const GLOBAL_FEATURES: usize = 15;

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

// Per-player normalisers. These are the engine's characteristic caps, not
// arbitrary round numbers: every per-player plane must land in `[0, 1]` or the
// side-factored encoding above stops being exactly recoverable. Raising a cap
// in `PlayerStats` widens the divisor here automatically.
const MOVE_NORM: f32 = PlayerStats::MAX_MOVEMENT as f32;
const ST_NORM: f32 = PlayerStats::MAX_ST as f32;
const MA_NORM: f32 = PlayerStats::MAX_MA as f32;
const AG_NORM: f32 = PlayerStats::MAX_AG as f32;
const AV_NORM: f32 = PlayerStats::MAX_AV as f32;

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

/// Encode a decision (or scoreable) state into mover-centric tensors,
/// normalised for the network. Exactly [`encode_raw`] divided by
/// [`spatial_channel_scales`].
pub fn encode(state: &GameState) -> Encoded {
    let raw = encode_raw(state);
    let scales = spatial_channel_scales();
    let plane = raw.h * raw.w;
    let spatial = raw
        .spatial
        .iter()
        .enumerate()
        .map(|(i, &v)| v as f32 / scales[i / plane])
        .collect();
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
        // 2 present + 8 shared per-player + 39 skill + 7 shared. Pinned so a
        // change to the engine's `Skill` enum shows up here as a failing
        // test, next to the `NN_SCHEMA_VERSION` bump it requires.
        assert_eq!(SPATIAL_CHANNELS, 2 + PLAYER_SCALARS + Skill::COUNT + 9);
        assert_eq!(SPATIAL_CHANNELS, 58);
        assert_eq!(spatial_channel_scales().len(), SPATIAL_CHANNELS);
        assert_eq!(GLOBAL_FEATURES, 15);
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
