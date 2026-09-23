//! Board-size distributions for mixed-size generation and eval (plan 042).
//!
//! A [`SizeDist`] is a finite weighted set of playable boards. Two ways to
//! build one:
//!
//! - an explicit list, `12x5,14x7/4:3,16x9` — playable `WxH`, optional
//!   `/T` team size (else derived from the area), optional `:weight`;
//! - a **centred** distribution ([`CentredSpec`]): every legal board inside
//!   the compiled capacity and an aspect band, weighted by a log-normal in
//!   *area* around a centre, smeared by a temperature, then mixed with a
//!   uniform floor so no size ever drops out of the corpus. Temperature to
//!   infinity is plan 039's uniform sampling; temperature to zero is plan
//!   017's single tier. The centre is the one knob the loop schedules.
//!
//! Team size follows the area (`round(area / cells_per_player)`, clamped to
//! `[2, capacity]`) so the game keeps the same density at every size: plan
//! 017's tiers all sit at 25-35 cells per player, and sampling `(w, h)` with a
//! fixed roster would train on a different game at each end.
//!
//! [`SizeDist::sample`] is a **pure function of `(dist, seed)`**: the board
//! for game `g` is decided by the game's own seed, not by any RNG stream a
//! worker happens to be on, so a hub-shipped config draws the same board for
//! the same seed on every machine and a corpus stays re-derivable.

use botbowl_engine::core::model::{BoardDims, Coord, HEIGHT, TEAM_SIZE, WIDTH};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Plan 039's density: `14x7/4` is 24.5, `20x11/7` 31, `26x15/11` 35.
pub const DEFAULT_CELLS_PER_PLAYER: f64 = 26.0;
/// Aspect band (playable width / playable height) the centred grid keeps.
/// Plan 017's tiers span 1.73 (26x15) to 2.67 (8x3); 12x9 (1.33) and 10x7
/// (1.43) fall outside and are plan 039's deliberate off-distribution shapes.
pub const DEFAULT_ASPECT: (f64, f64) = (1.5, 2.8);

/// Keeps the size draw off the seed streams the game and the bots use.
const SIZE_SEED_MIX: u64 = 0x5127_ED15_7B0A_4D5E;

/// One board and its (unnormalised) sampling weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SizeEntry {
    pub dims: BoardDims,
    pub weight: f64,
}

/// A finite weighted set of boards. Weights are normalised on construction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SizeDist {
    pub entries: Vec<SizeEntry>,
    /// Self-describing provenance, stamped into the corpus and the status log.
    pub label: String,
}

/// The centred distribution's knobs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CentredSpec {
    /// Playable area (`w * h`) the distribution is centred on.
    pub centre_area: f64,
    /// Standard deviation of `ln(area / centre)`. `0` puts all the weight on
    /// the nearest legal area; large values flatten toward uniform.
    pub temperature: f64,
    /// Share of the mass spread uniformly over every legal board, whatever
    /// the centre says. Keeps every size in every generation.
    pub floor: f64,
    pub aspect_min: f64,
    pub aspect_max: f64,
    pub cells_per_player: f64,
    /// Smallest playable area to enumerate; `None` = no lower bound.
    ///
    /// Plan 042 E0: at the default density every board of area <= 60 is a
    /// 2v2, where the champion scored *less* than a scripted mirror (attack
    /// 0.95x, against 1.6-2.0x on every other board). Skill does not express
    /// on those boards, so a run that wants only boards a bot can actually
    /// play sets this to 70.
    pub min_area: Option<f64>,
    /// Largest playable area to enumerate; `None` = the compiled capacity.
    pub max_area: Option<f64>,
}

impl Default for CentredSpec {
    fn default() -> Self {
        CentredSpec {
            centre_area: 98.0,
            temperature: 0.3,
            floor: 0.2,
            aspect_min: DEFAULT_ASPECT.0,
            aspect_max: DEFAULT_ASPECT.1,
            cells_per_player: DEFAULT_CELLS_PER_PLAYER,
            min_area: None,
            max_area: None,
        }
    }
}

/// Playable label, `14x7/4`, the form every flag and status line uses.
pub fn board_label(d: BoardDims) -> String {
    format!("{}x{}/{}", d.width - 2, d.height - 2, d.team_size)
}

/// Roster size that keeps the density: `round(area / cells_per_player)`,
/// at least 2 and never above the compiled capacity.
pub fn team_size_for(playable_area: f64, cells_per_player: f64) -> usize {
    let n = (playable_area / cells_per_player).round() as i64;
    (n.max(2) as usize).min(TEAM_SIZE)
}

/// Parse one playable board: `WxH` (team size from the area) or `WxH/T`.
pub fn parse_board(s: &str, cells_per_player: f64) -> Result<BoardDims, String> {
    let s = s.trim();
    let (wh, team) = match s.split_once('/') {
        Some((wh, t)) => (
            wh,
            Some(
                t.trim()
                    .parse::<usize>()
                    .map_err(|e| format!("board {s:?}: bad team size {t:?}: {e}"))?,
            ),
        ),
        None => (s, None),
    };
    let (w, h) = wh
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("board {s:?}: expected WxH or WxH/T"))?;
    let pw: i64 = w.trim().parse().map_err(|e| format!("board {s:?}: bad width: {e}"))?;
    let ph: i64 = h.trim().parse().map_err(|e| format!("board {s:?}: bad height: {e}"))?;
    if !(1..=125).contains(&pw) || !(1..=125).contains(&ph) {
        return Err(format!("board {s:?}: dimensions out of range"));
    }
    let team = team.unwrap_or_else(|| team_size_for((pw * ph) as f64, cells_per_player));
    BoardDims::try_new((pw + 2) as Coord, (ph + 2) as Coord, team).map_err(|e| format!("board {s:?}: {e}"))
}

/// Every board the compiled binary can play whose aspect lies in the band
/// and whose playable area is within `min_area..=max_area`, with the density
/// rule's team size. Sorted by area, then width.
pub fn legal_grid(
    aspect: (f64, f64),
    cells_per_player: f64,
    min_area: Option<f64>,
    max_area: Option<f64>,
) -> Vec<BoardDims> {
    let cap_w = (WIDTH as i64) - 2;
    let cap_h = (HEIGHT as i64) - 2;
    let mut out = Vec::new();
    let mut pw = 8;
    while pw <= cap_w {
        for ph in 3..=cap_h {
            let area = (pw * ph) as f64;
            let ratio = pw as f64 / ph as f64;
            if ratio < aspect.0 || ratio > aspect.1 {
                continue;
            }
            if max_area.is_some_and(|m| area > m) || min_area.is_some_and(|m| area < m) {
                continue;
            }
            let team = team_size_for(area, cells_per_player);
            if let Ok(d) = BoardDims::try_new((pw + 2) as Coord, (ph + 2) as Coord, team) {
                out.push(d);
            }
        }
        pw += 2;
    }
    out.sort_by_key(|d| ((d.width - 2) as i64 * (d.height - 2) as i64, d.width));
    out
}

fn playable_area(d: BoardDims) -> f64 {
    ((d.width - 2) as f64) * ((d.height - 2) as f64)
}

impl SizeDist {
    /// One board, always. What an unset `--board-sizes` means once a caller
    /// has resolved the env board.
    pub fn single(dims: BoardDims) -> Self {
        SizeDist {
            entries: vec![SizeEntry { dims, weight: 1.0 }],
            label: board_label(dims),
        }
    }

    /// `12x5,14x7/4:3,16x9` — comma-separated boards with optional `:weight`.
    pub fn parse_list(spec: &str, cells_per_player: f64) -> Result<Self, String> {
        let mut entries = Vec::new();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (board, weight) = match item.rsplit_once(':') {
                Some((b, w)) => (
                    b,
                    w.trim()
                        .parse::<f64>()
                        .map_err(|e| format!("size {item:?}: bad weight {w:?}: {e}"))?,
                ),
                None => (item, 1.0),
            };
            if !(weight > 0.0) || !weight.is_finite() {
                return Err(format!("size {item:?}: weight must be positive"));
            }
            let dims = parse_board(board, cells_per_player)?;
            if entries.iter().any(|e: &SizeEntry| e.dims == dims) {
                return Err(format!("size {item:?}: listed twice"));
            }
            entries.push(SizeEntry { dims, weight });
        }
        if entries.is_empty() {
            return Err("--board-sizes is empty".into());
        }
        let label = format!(
            "list[{}]",
            entries
                .iter()
                .map(|e| if e.weight == 1.0 {
                    board_label(e.dims)
                } else {
                    format!("{}:{}", board_label(e.dims), e.weight)
                })
                .collect::<Vec<_>>()
                .join(",")
        );
        Ok(Self::normalised(entries, label))
    }

    /// Log-normal in area around `centre_area`, mixed with a uniform floor,
    /// over the legal grid. See the module docs.
    pub fn centred(spec: &CentredSpec) -> Result<Self, String> {
        if !(spec.centre_area > 0.0) {
            return Err(format!("size centre must be positive, got {}", spec.centre_area));
        }
        if !(0.0..=1.0).contains(&spec.floor) {
            return Err(format!("size floor must be in [0, 1], got {}", spec.floor));
        }
        if spec.temperature < 0.0 {
            return Err(format!("size temperature must be >= 0, got {}", spec.temperature));
        }
        if !(spec.aspect_min > 0.0 && spec.aspect_max >= spec.aspect_min) {
            return Err(format!(
                "size aspect band must satisfy 0 < min <= max, got {}-{}",
                spec.aspect_min, spec.aspect_max
            ));
        }
        let grid = legal_grid(
            (spec.aspect_min, spec.aspect_max),
            spec.cells_per_player,
            spec.min_area,
            spec.max_area,
        );
        if grid.is_empty() {
            return Err(format!(
                "no legal board fits aspect {}-{} and area {:?}..={:?} within capacity {}x{}/{}",
                spec.aspect_min,
                spec.aspect_max,
                spec.min_area,
                spec.max_area,
                WIDTH - 2,
                HEIGHT - 2,
                TEAM_SIZE
            ));
        }
        let n = grid.len() as f64;
        let ln_dist: Vec<f64> = grid
            .iter()
            .map(|d| (playable_area(*d) / spec.centre_area).ln().abs())
            .collect();
        let core: Vec<f64> = if spec.temperature == 0.0 {
            // All of the centred mass on the nearest area (ties share it).
            let best = ln_dist.iter().cloned().fold(f64::INFINITY, f64::min);
            ln_dist.iter().map(|&d| if d == best { 1.0 } else { 0.0 }).collect()
        } else {
            ln_dist
                .iter()
                .map(|&d| (-(d * d) / (2.0 * spec.temperature * spec.temperature)).exp())
                .collect()
        };
        let core_sum: f64 = core.iter().sum();
        let entries = grid
            .iter()
            .zip(&core)
            .map(|(d, c)| SizeEntry {
                dims: *d,
                weight: (1.0 - spec.floor) * c / core_sum + spec.floor / n,
            })
            .collect();
        let label = format!(
            "centred(area={:.0},T={},floor={},aspect={}-{},cpp={}{})",
            spec.centre_area,
            spec.temperature,
            spec.floor,
            spec.aspect_min,
            spec.aspect_max,
            spec.cells_per_player,
            format!(
                "{}{}",
                spec.min_area.map(|m| format!(",min={m:.0}")).unwrap_or_default(),
                spec.max_area.map(|m| format!(",max={m:.0}")).unwrap_or_default()
            )
        );
        Ok(Self::normalised(entries, label))
    }

    fn normalised(mut entries: Vec<SizeEntry>, label: String) -> Self {
        let total: f64 = entries.iter().map(|e| e.weight).sum();
        for e in &mut entries {
            e.weight /= total;
        }
        SizeDist { entries, label }
    }

    pub fn is_single(&self) -> bool {
        self.entries.len() == 1
    }

    pub fn boards(&self) -> impl Iterator<Item = BoardDims> + '_ {
        self.entries.iter().map(|e| e.dims)
    }

    /// The board for the game with this seed. Pure in `(self, seed)`.
    pub fn sample(&self, seed: u64) -> BoardDims {
        if self.entries.len() == 1 {
            return self.entries[0].dims;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ SIZE_SEED_MIX);
        let u: f64 = rng.gen();
        let mut acc = 0.0;
        for e in &self.entries {
            acc += e.weight;
            if u < acc {
                return e.dims;
            }
        }
        // Rounding at the top of the cumulative sum.
        self.entries.last().expect("non-empty").dims
    }

    /// `(board label, probability)` rows, heaviest first — for status lines.
    pub fn table(&self) -> Vec<(String, f64)> {
        let mut rows: Vec<(String, f64)> = self.entries.iter().map(|e| (board_label(e.dims), e.weight)).collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap_fits(pw: i64, ph: i64, t: usize) -> bool {
        pw + 2 <= WIDTH as i64 && ph + 2 <= HEIGHT as i64 && t <= TEAM_SIZE
    }

    #[test]
    fn team_size_follows_the_area_and_stays_in_range() {
        assert_eq!(team_size_for(98.0, 26.0), 4.min(TEAM_SIZE)); // 14x7
        assert_eq!(team_size_for(50.0, 26.0), 2); // 10x5: 1.9 rounds to 2
        assert_eq!(team_size_for(24.0, 26.0), 2); // never below 2
        assert_eq!(team_size_for(1e6, 26.0), TEAM_SIZE); // never above capacity
    }

    #[test]
    fn parse_board_accepts_both_forms_and_rejects_the_engines_rules() {
        if !cap_fits(14, 7, 4) {
            return;
        }
        let d = parse_board("14x7", 26.0).unwrap();
        assert_eq!((d.width, d.height, d.team_size), (16, 9, 4));
        let d = parse_board("14x7/3", 26.0).unwrap();
        assert_eq!(d.team_size, 3);
        assert_eq!(board_label(d), "14x7/3");
        assert!(parse_board("13x7", 26.0).unwrap_err().contains("even"));
        assert!(parse_board("14", 26.0).unwrap_err().contains("WxH"));
        assert!(parse_board("14x7/x", 26.0).unwrap_err().contains("team size"));
    }

    #[test]
    fn list_parses_weights_and_normalises() {
        if !cap_fits(14, 7, 4) {
            return;
        }
        let d = SizeDist::parse_list("12x5, 14x7:3 ,10x5/2", 26.0).unwrap();
        assert_eq!(d.entries.len(), 3);
        let total: f64 = d.entries.iter().map(|e| e.weight).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((d.entries[1].weight - 0.6).abs() < 1e-12);
        assert_eq!(d.label, "list[12x5/2,14x7/4:3,10x5/2]");
        assert!(SizeDist::parse_list("14x7,14x7", 26.0).unwrap_err().contains("twice"));
        assert!(SizeDist::parse_list("", 26.0).is_err());
        assert!(SizeDist::parse_list("14x7:0", 26.0).unwrap_err().contains("positive"));
    }

    #[test]
    fn legal_grid_respects_capacity_aspect_and_the_engine() {
        let grid = legal_grid(DEFAULT_ASPECT, 26.0, None, None);
        assert!(!grid.is_empty());
        for d in &grid {
            let (pw, ph) = ((d.width - 2) as f64, (d.height - 2) as f64);
            assert!(pw >= 8.0 && (pw as i64) % 2 == 0);
            assert!(ph >= 3.0);
            let r = pw / ph;
            assert!(r >= DEFAULT_ASPECT.0 && r <= DEFAULT_ASPECT.1, "{}", board_label(*d));
            assert!(d.width as usize <= WIDTH && d.height as usize <= HEIGHT);
            assert_eq!(d.team_size, team_size_for(pw * ph, 26.0));
        }
        // Sorted by area.
        for w in grid.windows(2) {
            assert!(playable_area(w[0]) <= playable_area(w[1]));
        }
        let small = legal_grid(DEFAULT_ASPECT, 26.0, None, Some(60.0));
        assert!(small.iter().all(|d| playable_area(*d) <= 60.0));
        assert!(small.len() < grid.len());
    }

    /// Plan 042 E0: at the density rule's 26 cells/player every board of area
    /// <= 60 is a 2v2, and the ladder showed the champion scores *less* there
    /// than a scripted mirror does (attack 0.95x, against 1.6-2.0x everywhere
    /// else) — the board is a scoring free-for-all in which skill does not
    /// express, so those games are not worth generating. `min_area` is how a
    /// run excludes them.
    #[test]
    fn min_area_excludes_the_degenerate_small_boards() {
        let grid = legal_grid(DEFAULT_ASPECT, 26.0, None, None);
        let big = legal_grid(DEFAULT_ASPECT, 26.0, Some(70.0), None);
        assert!(big.iter().all(|d| playable_area(*d) >= 70.0));
        assert!(big.len() < grid.len());
        // 70 is exactly the bound that clears every team-size-2 board.
        assert!(big.iter().all(|d| d.team_size >= 3), "{:?}", big.iter().map(|d| board_label(*d)).collect::<Vec<_>>());
        // Both bounds compose.
        let band = legal_grid(DEFAULT_ASPECT, 26.0, Some(70.0), Some(112.0));
        assert!(band
            .iter()
            .all(|d| (70.0..=112.0).contains(&playable_area(*d))));
    }

    #[test]
    fn centred_respects_min_area() {
        if !cap_fits(16, 9, 6) {
            return;
        }
        let spec = CentredSpec {
            min_area: Some(70.0),
            max_area: Some(144.0),
            ..CentredSpec::default()
        };
        let d = SizeDist::centred(&spec).unwrap();
        assert!(d.entries.iter().all(|e| e.dims.team_size >= 3));
        assert!(d.label.contains("min=70"), "{}", d.label);
        // An empty band is an error, not a silent empty distribution.
        assert!(SizeDist::centred(&CentredSpec {
            min_area: Some(1000.0),
            ..CentredSpec::default()
        })
        .is_err());
    }

    #[test]
    fn centred_puts_most_mass_near_the_centre_and_keeps_the_floor() {
        if !cap_fits(16, 9, 6) {
            return;
        }
        let spec = CentredSpec {
            centre_area: 98.0,
            temperature: 0.2,
            floor: 0.2,
            max_area: Some(144.0),
            ..CentredSpec::default()
        };
        let d = SizeDist::centred(&spec).unwrap();
        let n = d.entries.len() as f64;
        let total: f64 = d.entries.iter().map(|e| e.weight).sum();
        assert!((total - 1.0).abs() < 1e-12);
        // Every board keeps at least its share of the floor.
        for e in &d.entries {
            assert!(e.weight >= 0.2 / n - 1e-12, "{} starved", board_label(e.dims));
        }
        // The heaviest board is the one at the centre.
        let top = &d.table()[0];
        assert_eq!(top.0, "14x7/4", "{:?}", d.table());
        // Zero temperature: everything but the floor on the nearest area.
        let sharp = SizeDist::centred(&CentredSpec {
            temperature: 0.0,
            ..spec
        })
        .unwrap();
        let top = &sharp.table()[0];
        assert!((top.1 - (0.8 + 0.2 / n)).abs() < 1e-9, "{:?}", sharp.table());
        // Bad knobs are refused up front.
        assert!(SizeDist::centred(&CentredSpec { floor: 1.5, ..spec }).is_err());
        assert!(SizeDist::centred(&CentredSpec {
            aspect_min: 3.0,
            aspect_max: 2.0,
            ..spec
        })
        .is_err());
    }

    #[test]
    fn sampling_is_pure_in_the_seed_and_matches_the_weights() {
        if !cap_fits(16, 9, 6) {
            return;
        }
        let d = SizeDist::parse_list("12x5:1,14x7:3", 26.0).unwrap();
        for seed in 0..50u64 {
            assert_eq!(d.sample(seed), d.sample(seed));
        }
        let n = 20_000;
        let big = d.entries[1].dims;
        let hits = (0..n).filter(|&s| d.sample(s) == big).count() as f64 / n as f64;
        assert!((hits - 0.75).abs() < 0.02, "{hits}");
        // A single-entry distribution never touches the RNG.
        let one = SizeDist::single(big);
        assert!(one.is_single());
        assert_eq!(one.sample(123), big);
    }
}
