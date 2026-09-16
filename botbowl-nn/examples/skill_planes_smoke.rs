//! End-to-end smoke check for the all-skills encoder: generate curriculum
//! random starts, print the stat/skill variety they contain, and confirm the
//! encoded tensor lights up the matching skill plane for every player —
//! together with the `present` plane that says whose player it is, which is
//! the only thing carrying ownership since the planes were unpaired.
//!
//! `cargo run -p botbowl-nn --example skill_planes_smoke`

use std::collections::BTreeMap;

use botbowl_curriculum::random_start::{generate_random_start, RandomStartConfig};
use botbowl_engine::core::model::BoardDims;
use botbowl_engine::core::table::Skill;
use botbowl_nn::encode::{encode, spatial_channel_names, SPATIAL_CHANNELS};
use botbowl_nn::perspective::{canonical_pos, mover_for};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let cfg = RandomStartConfig {
        board_dims: Some(BoardDims::new(16, 9, 4)),
        ..Default::default()
    };
    let names = spatial_channel_names();
    // The skill planes start right after the 2 present + 8 shared per-player
    // planes; `SKILL_BASE` itself is private to the encoder.
    const SKILL_BASE: usize = 10;
    assert_eq!(names[SKILL_BASE], format!("skill_{:?}", Skill::ALL[0]).to_lowercase());
    println!("C = {SPATIAL_CHANNELS}, skill planes = {} (unpaired)", Skill::COUNT);

    let mut skill_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut stat_lines = 0usize;
    let mut checked = 0usize;

    for seed in 0..50u64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let state = generate_random_start(&cfg, &mut rng);
        let mover = mover_for(&state);
        let dims = state.board_dims;
        let enc = encode(&state);
        let plane = enc.h * enc.w;

        for p in state.get_players_on_pitch() {
            let pos = canonical_pos(p.position, dims, mover);
            let cell = |c: usize| enc.spatial[c * plane + (pos.y as usize) * enc.w + (pos.x as usize)];
            if p.stats != botbowl_engine::core::model::PlayerStats::new_lineman(p.stats.team) {
                stat_lines += 1;
            }
            // Ownership: exactly one of the two present planes, never both.
            let (us, them) = (cell(0), cell(1));
            let ours = (p.stats.team == mover) as u8 as f32;
            assert_eq!((us, them), (ours, 1.0 - ours), "present planes disagree on the owner");

            for sk in Skill::ALL {
                let want = p.has_skill(sk) as u8 as f32;
                let c = SKILL_BASE + sk.index();
                assert_eq!(cell(c), want, "plane {} wrong for {sk:?}", names[c]);
                if p.has_skill(sk) {
                    *skill_counts.entry(format!("{sk:?}")).or_default() += 1;
                }
            }
            checked += 1;
        }
    }

    println!("{checked} players encoded, {stat_lines} off the lineman baseline");
    println!("skills seen on the pitch (and verified in their planes):");
    for (sk, n) in &skill_counts {
        println!("  {sk:<12} {n}");
    }
    assert!(!skill_counts.is_empty(), "no skills were generated at all");
    println!("OK — every player's skills matched their planes");
}
