//! A fast, fixed hasher for the tree's own containers (botbowl plan 070).
//!
//! The children maps, `parents` sets, the registry and the per-walk visited sets used std's
//! `RandomState` (SipHash-1-3 with per-process random keys): about 9.5% of botbowl's search
//! instructions (exp080), and an iteration order that differs per process and per container. This
//! is the multiply-rotate hash of `rustc-hash` 2 (Firefox's FxHash family): a few instructions per
//! word, no keys, the same order in every process on every platform for the same insertions.
//!
//! The keys hashed here are an action, a node's precomputed state hash (`Node::hash`, which stays
//! SipHash: the registry's equality check is what its quality protects) or a pointer, so neither
//! hash flooding nor weak bits are a concern. `finish` rotates the high-entropy bits of the product
//! down into the bits `hashbrown` indexes buckets with.

use std::hash::{BuildHasherDefault, Hasher};

/// `rustc-hash` 2's multiplier.
const K: u64 = 0xf135_7aea_2e62_a9c5;

/// See the module docs.
#[derive(Clone, Copy, Debug, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            // The length keeps "ab" + "" and "a" + "b" apart.
            self.add(u64::from_le_bytes(word) ^ ((rest.len() as u64) << 59));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
}

/// The `BuildHasher` for every hashed container of the tree.
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;

#[cfg(test)]
mod test {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn h<T: Hash>(t: &T) -> u64 {
        FxBuildHasher::default().hash_one(t)
    }

    #[test]
    fn fixed_across_instances() {
        assert_eq!(h(&(3u32, 7u64)), h(&(3u32, 7u64)));
        assert_ne!(h(&(3u32, 7u64)), h(&(7u32, 3u64)));
        assert_ne!(h(&"ab"), h(&"a"));
    }

    #[test]
    fn spreads_aligned_pointers_over_the_low_bits() {
        // Pointer keys are multiples of the alignment; the bucket index is taken from the low bits.
        let buckets: std::collections::HashSet<u64> = (0..256u64).map(|i| h(&(i * 64)) & 255).collect();
        assert!(buckets.len() > 140, "{} distinct low bytes of 256", buckets.len());
    }
}
