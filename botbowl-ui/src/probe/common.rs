//! What every probe suite shares: the net (tract, CPU, raw values), base states (replayed corpus
//! decisions and regenerated drive positions), order-preserving parallel map, distribution
//! summaries, the worst-case keeper and its dumps (JSON with ASCII boards, plus a two-state
//! `Recording` that `botbowl-ui replay` / `snapshot --replay F --step 1` show).

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use botbowl_data::{BoardCapacity, TrajectoryMeta};
use botbowl_engine::core::game_runner::Recording;
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{Action, BallState, BoardDims, PlayerStatus, Position, TeamType};
use botbowl_engine::core::procedures::AnyProc;
use botbowl_nn::encode::encode;
use botbowl_nn::eval::NnEvaluator;
use botbowl_nn::perspective::mover_for;
use botbowl_play::board_sizes::board_label;
use botbowl_play::drives::{position_state, PositionSet};
use botbowl_play::GAME_STACK_SIZE;

use crate::override_audit::{load_trajectory, replay_to};

// ---------------------------------------------------------------------------------------------
// The net
// ---------------------------------------------------------------------------------------------

/// A frozen net on the **tract** backend only. The probes never take a sidecar socket (and so
/// never touch a GPU), and they read the raw value: unclamped, unquantised, mover-centric.
pub struct Net {
    pub nn: Arc<NnEvaluator>,
}

impl Net {
    pub fn load(path: &str) -> io::Result<Net> {
        let nn = NnEvaluator::from_path(path).map_err(|e| io::Error::other(format!("{path}: {e}")))?;
        Ok(Net { nn: Arc::new(nn) })
    }

    /// The raw value head on `state`, in the mover's frame (`+1` = the team to move scores).
    pub fn value(&self, state: &GameState) -> f32 {
        let e = encode(state);
        self.nn.value_only_raw(&e.spatial, &e.global, e.h, e.w)
    }

    /// The raw value in `team`'s frame.
    pub fn value_for(&self, state: &GameState, team: TeamType) -> f32 {
        let v = self.value(state);
        if mover_for(state) == team {
            v
        } else {
            0.0 - v
        }
    }

    /// The policy over `legal` as probabilities (the search's priors are these × `len`).
    pub fn probs(&self, state: &GameState, legal: &[Action]) -> Vec<f32> {
        let n = legal.len().max(1) as f32;
        self.nn.priors(state, legal).into_iter().map(|p| p / n).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Base states
// ---------------------------------------------------------------------------------------------

/// Where a probe decision sits in the turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A `Turn` decision before any of the mover's players has acted.
    TurnStart,
    /// A `Turn` decision later in the turn.
    TurnMid,
    /// A player is activated (choosing a square, a block target, ...).
    Activation,
    /// Anything else the mover is asked (a block die, a push square, a skill, ...).
    Other,
}

pub fn kind_of(s: &GameState) -> Kind {
    if matches!(s.proc_stack_peek(), Some(AnyProc::Turn(_))) {
        let mover = mover_for(s);
        if s.get_players_on_pitch_in_team(mover).any(|p| p.used) {
            Kind::TurnMid
        } else {
            Kind::TurnStart
        }
    } else if s.info.active_player.is_some() && s.available_actions.has_paths() {
        Kind::Activation
    } else {
        Kind::Other
    }
}

/// A player decision outside the kickoff setup, with nothing pending.
pub fn is_probe_decision(s: &GameState) -> bool {
    s.available_actions.team.is_some() && s.pending_roll.is_none() && !s.info.game_over && s.setup_team().is_none()
}

pub fn at_turn_decision(s: &GameState) -> bool {
    is_probe_decision(s) && matches!(s.proc_stack_peek(), Some(AnyProc::Turn(_)))
}

pub struct Base {
    /// `file:line:sample` for a corpus decision, `set:seed` for a position.
    pub id: String,
    pub state: GameState,
    pub kind: Kind,
}

fn board_matches(filters: &[String], dims: BoardDims) -> bool {
    if filters.is_empty() {
        return true;
    }
    let label = board_label(dims);
    let wh = label.split('/').next().unwrap_or("");
    filters.iter().any(|f| f.trim() == label || f.trim() == wh)
}

#[derive(Deserialize)]
struct LiteTraj {
    meta: TrajectoryMeta,
    samples: Vec<serde::de::IgnoredAny>,
}

/// Up to `n` corpus decisions that `accept`s, sampled uniformly (seeded) from the random-start
/// trajectories on the wanted boards, each rebuilt by replay ([`replay_to`]).
pub fn corpus_bases(
    paths: &[String],
    boards: &[String],
    n: usize,
    seed: u64,
    accept: &dyn Fn(&GameState) -> bool,
) -> io::Result<Vec<Base>> {
    let capacity = BoardCapacity::current();
    let mut cands: Vec<(usize, u64, usize, usize)> = Vec::new();
    for (fi, path) in paths.iter().enumerate() {
        let mut reader =
            BufReader::new(File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))?);
        let (mut offset, mut li, mut line) = (0u64, 0usize, String::new());
        loop {
            line.clear();
            let read = reader.read_line(&mut line)?;
            if read == 0 {
                break;
            }
            let here = offset;
            offset += read as u64;
            li += 1;
            if line.trim().is_empty() {
                continue;
            }
            let Ok(t) = serde_json::from_str::<LiteTraj>(&line) else {
                eprintln!("probe: {path}:{li} does not parse, skipped (a shard still being written?)");
                continue;
            };
            // Plan 065: any capacity the board fits replays it (`BoardCapacity::holds`).
            if !capacity.holds(t.meta.board_dims) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{path}: board {:?} (corpus capacity {:?}) does not fit this binary's capacity {capacity:?}; \
                         rebuild with a bigger BOARD_SIZE_W/H/PLAYERS (scripts/probe_build.sh)",
                        t.meta.board_dims, t.meta.board_capacity
                    ),
                ));
            }
            let first_drive =
                !t.meta.extra.contains_key("drive") || t.meta.extra.get("drive").map(String::as_str) == Some("1");
            if t.meta.extra.get("mode").map(String::as_str) != Some("random-start")
                || !first_drive
                || !board_matches(boards, t.meta.board_dims)
            {
                continue;
            }
            for si in 0..t.samples.len() {
                cands.push((fi, here, li, si));
            }
        }
    }
    cands.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let mut out = Vec::new();
    let mut cache: HashMap<(usize, u64), botbowl_data::Trajectory> = HashMap::new();
    for (fi, off, li, si) in cands {
        if out.len() >= n {
            break;
        }
        if cache.len() > 64 {
            cache.clear();
        }
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry((fi, off)) {
            e.insert(load_trajectory(&paths[fi], off)?);
        }
        let traj = &cache[&(fi, off)];
        let Ok(mut state) = replay_to(traj, si) else { continue };
        if !is_probe_decision(&state) || !accept(&state) {
            continue;
        }
        state.set_logging_state(false);
        out.push(Base {
            id: format!("{}:{li}:{si}", paths[fi]),
            kind: kind_of(&state),
            state,
        });
    }
    Ok(out)
}

/// Up to `n` drive positions (fresh random-start `Turn` decisions) from frozen position sets: the
/// screened (contested) positions first, then the set's other candidates, in a seeded order.
pub fn position_bases(
    paths: &[String],
    boards: &[String],
    n: usize,
    seed: u64,
    accept: &dyn Fn(&GameState) -> bool,
) -> io::Result<Vec<Base>> {
    let mut cands: Vec<(usize, u64, bool)> = Vec::new();
    let mut sets = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        let set = PositionSet::load(p).map_err(io::Error::other)?;
        let dims = set.board_dims().map_err(io::Error::other)?;
        if !board_matches(boards, dims) {
            continue;
        }
        let kept = set.positions();
        for &s in &kept {
            cands.push((i, s, true));
        }
        for &s in &set.seeds {
            if !kept.contains(&s) {
                cands.push((i, s, false));
            }
        }
        sets.push((i, set, dims));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (mut contested, mut rest): (Vec<_>, Vec<_>) = cands.into_iter().partition(|c| c.2);
    contested.shuffle(&mut rng);
    rest.shuffle(&mut rng);
    let mut out = Vec::new();
    for (i, s, is_contested) in contested.into_iter().chain(rest) {
        if out.len() >= n {
            break;
        }
        let (_, set, dims) = sets.iter().find(|(j, _, _)| *j == i).expect("indexed set");
        let state = position_state(&set.bias, *dims, s);
        if !at_turn_decision(&state) || !accept(&state) {
            continue;
        }
        out.push(Base {
            id: format!("{}:{s}{}", set.name, if is_contested { "" } else { ":unscreened" }),
            kind: kind_of(&state),
            state,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Parallel map, statistics, worst cases
// ---------------------------------------------------------------------------------------------

/// `f(i)` for `i in 0..n` on `threads` threads, results in index order (so outputs do not depend
/// on scheduling). Each thread has the deep stack a search needs.
pub fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads.clamp(1, n.max(1)))
            .map(|t| {
                let (next, out, f) = (&next, &out, &f);
                std::thread::Builder::new()
                    .name(format!("probe-{t}"))
                    .stack_size(GAME_STACK_SIZE)
                    .spawn_scoped(scope, move || loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return;
                        }
                        let r = f(i);
                        out.lock().expect("par_map mutex")[i] = Some(r);
                    })
                    .expect("spawn probe thread")
            })
            .collect();
        for h in handles {
            h.join().expect("probe thread panicked");
        }
    });
    out.into_inner()
        .expect("par_map mutex")
        .into_iter()
        .map(|r| r.expect("every index ran"))
        .collect()
}

/// A distribution, summarised.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Dist {
    pub n: usize,
    pub mean: f32,
    pub se: f32,
    pub min: f32,
    pub p01: f32,
    pub p10: f32,
    pub p50: f32,
    pub p90: f32,
    pub p99: f32,
    pub max: f32,
}

pub fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let i = ((sorted.len() - 1) as f32 * q).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

pub fn dist(xs: &[f32]) -> Dist {
    let n = xs.len();
    if n == 0 {
        return Dist::default();
    }
    let mut s: Vec<f32> = xs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let mean = s.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var = if n > 1 {
        s.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    Dist {
        n,
        mean: mean as f32,
        se: (var / n as f64).sqrt() as f32,
        min: s[0],
        p01: quantile(&s, 0.01),
        p10: quantile(&s, 0.10),
        p50: quantile(&s, 0.50),
        p90: quantile(&s, 0.90),
        p99: quantile(&s, 0.99),
        max: s[n - 1],
    }
}

pub fn frac(xs: impl IntoIterator<Item = bool>) -> f32 {
    let (mut k, mut n) = (0usize, 0usize);
    for x in xs {
        k += x as usize;
        n += 1;
    }
    if n == 0 {
        f32::NAN
    } else {
        k as f32 / n as f32
    }
}

/// One case worth looking at: its row, and the state before and after the edit.
pub struct Case {
    pub score: f32,
    pub label: String,
    pub row: serde_json::Value,
    pub before: GameState,
    pub after: GameState,
    pub marks: Vec<(Position, char)>,
}

/// The `k` highest-scoring cases.
pub struct TopK {
    k: usize,
    items: Vec<Case>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK { k, items: Vec::new() }
    }

    pub fn push(&mut self, c: Case) {
        if self.items.len() >= self.k && self.items.last().is_some_and(|l| l.score >= c.score) {
            return;
        }
        let at = self.items.partition_point(|x| x.score >= c.score);
        self.items.insert(at, c);
        self.items.truncate(self.k);
    }

    pub fn extend(&mut self, cs: impl IntoIterator<Item = Case>) {
        for c in cs {
            self.push(c);
        }
    }

    pub fn into_items(self) -> Vec<Case> {
        self.items
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// `{out}.worst.json` (rows with ASCII boards) and one `Recording` per case under
    /// `{dir}/worst/`, viewable with `botbowl-ui replay` or `snapshot --replay F --step 1`.
    pub fn write(&self, dir: &str, stem: &str) -> io::Result<String> {
        let wdir = format!("{dir}/worst");
        std::fs::create_dir_all(&wdir)?;
        let mut out = Vec::new();
        for (k, c) in self.items.iter().enumerate() {
            let rec = format!("{wdir}/{stem}.{k:02}.rec.json");
            Recording::new(vec![c.before.clone(), c.after.clone()]).to_file(&rec);
            out.push(serde_json::json!({
                "rank": k,
                "score": c.score,
                "label": c.label,
                "row": c.row,
                "recording": rec,
                "before": ascii(&c.before, &c.marks),
                "after": ascii(&c.after, &c.marks),
            }));
        }
        let path = format!("{dir}/{stem}.worst.json");
        std::fs::write(&path, serde_json::to_string_pretty(&out)?)?;
        Ok(path)
    }
}

/// The board as text, one row per line, two characters a square. `H`/`A` a standing Home/Away
/// player, `h`/`a` down or stunned, `*` after a player is the ball carrier, `*` alone a loose
/// ball, `+` the active player, `#` out of bounds; a mark's character replaces the second one.
pub fn ascii(state: &GameState, marks: &[(Position, char)]) -> Vec<String> {
    let d = state.board_dims;
    let carrier = match state.ball {
        BallState::Carried(id) => state.get_player(id).ok().map(|p| p.position),
        _ => None,
    };
    let loose = match state.ball {
        BallState::OnGround(p) | BallState::InAir(p) => Some(p),
        _ => None,
    };
    let active = state
        .info
        .active_player
        .and_then(|id| state.get_player(id).ok())
        .map(|p| p.position);
    (0..d.height)
        .map(|y| {
            let mut row = String::new();
            for x in 0..d.width {
                let p = Position::new((x, y));
                let mark = marks.iter().find(|(q, _)| *q == p).map(|(_, c)| *c);
                let (a, mut b) = if d.is_out(p) {
                    ('#', '#')
                } else if let Some(pl) = state.get_player_at(p) {
                    let up = pl.status == PlayerStatus::Up;
                    let c = match (pl.stats.team, up) {
                        (TeamType::Home, true) => 'H',
                        (TeamType::Home, false) => 'h',
                        (TeamType::Away, true) => 'A',
                        (TeamType::Away, false) => 'a',
                    };
                    let tag = if carrier == Some(p) {
                        '*'
                    } else if active == Some(p) {
                        '+'
                    } else {
                        ' '
                    };
                    (c, tag)
                } else if loose == Some(p) {
                    ('*', ' ')
                } else {
                    ('.', ' ')
                };
                if let Some(m) = mark {
                    b = m;
                }
                row.push(a);
                row.push(b);
            }
            row
        })
        .collect()
}

/// Skills no engine rule reads (plan 064's `Inert` class), used as a placebo: their true effect on
/// the value is zero, so the net's `|ΔV|` for them is the noise floor of a one-cell edit.
pub const PLACEBO_SKILLS: [botbowl_engine::core::table::Skill; 3] = [
    botbowl_engine::core::table::Skill::Leap,
    // Plane 24, the placebo `Pass` held before it took the old `Throw` plane and a rule.
    botbowl_engine::core::table::Skill::SafePass,
    botbowl_engine::core::table::Skill::Kick,
];

/// `|ΔV|` for every placebo skill on every player of `s` (whose value is `v`).
pub fn placebo_deltas(net: &Net, s: &GameState, v: f32) -> Vec<f32> {
    let mut out = Vec::new();
    for p in s.get_players_on_pitch() {
        for k in PLACEBO_SKILLS {
            if let Ok(e) = super::edits::add_skill(s, p.id, k) {
                out.push((net.value(&e) - v).abs());
            }
        }
    }
    out
}

/// Chebyshev distance.
pub fn cheb(a: Position, b: Position) -> i8 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}

/// The direction (`-1` / `+1` in x) in which `team` scores.
pub fn attack_dx(dims: BoardDims, team: TeamType) -> i8 {
    if dims.endzone_x(team) < dims.width / 2 {
        -1
    } else {
        1
    }
}
