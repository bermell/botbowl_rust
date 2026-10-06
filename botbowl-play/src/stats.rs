//! Paired-game statistics for the ladder (plan 051 step 1).
//!
//! `ladder_assignment` gives games `2k` and `2k+1` one seed with the sides swapped. Those two
//! games are not independent: a one-sided seed yields a win and a loss that cancel and say
//! nothing about the bots. Scoring the **pair** as one sample, with five outcomes (0, ½, 1, 1½, 2
//! points), estimates the variance correctly. Fishtest moved to the same scheme for the same
//! reason.
//!
//! [`Sprt`] is a sequential test on those pair samples. It stops a head-to-head once the
//! evidence for `s1` over `s0` (or the reverse) crosses a bound, so more games can only sharpen
//! it. A fixed-N threshold test on a point estimate can get worse with more games (plan 030).

use serde::{Deserialize, Serialize};

/// Pair results: `counts[i]` is the number of mirrored pairs where the candidate took `i` half
/// points over its two games, so index 0 is two losses and index 4 is two wins.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pentanomial {
    pub counts: [u32; 5],
}

impl Pentanomial {
    /// Fold one finished pair, given the candidate's half points in each game (0, 1 or 2).
    pub fn record(&mut self, a: u8, b: u8) {
        debug_assert!(a <= 2 && b <= 2, "a game is worth at most two half points");
        self.counts[(a + b) as usize] += 1;
    }

    pub fn pairs(&self) -> u32 {
        self.counts.iter().sum()
    }

    /// Per-game score of pair outcome `i`, in [0, 1].
    fn score(i: usize) -> f64 {
        i as f64 / 4.0
    }

    /// Mean per-game score over the pairs, in [0, 1]. 0.5 with no pairs.
    pub fn mean(&self) -> f64 {
        let n = self.pairs();
        if n == 0 {
            return 0.5;
        }
        self.counts
            .iter()
            .enumerate()
            .map(|(i, &c)| Self::score(i) * c as f64)
            .sum::<f64>()
            / n as f64
    }

    /// Variance of one pair's per-game score (population form, as fishtest uses).
    pub fn var(&self) -> f64 {
        let n = self.pairs();
        if n == 0 {
            return 0.0;
        }
        let m = self.mean();
        self.counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (Self::score(i) - m).powi(2) * c as f64)
            .sum::<f64>()
            / n as f64
    }

    /// Standard error of [`Pentanomial::mean`]. 0 with fewer than two pairs.
    pub fn se(&self) -> f64 {
        let n = self.pairs();
        if n < 2 {
            return 0.0;
        }
        (self.var() / n as f64).sqrt()
    }
}

/// A sequential probability ratio test of "the candidate scores `s1`" against "it scores `s0`",
/// with error rates `alpha` (accepting H1 when H0 holds) and `beta` (accepting H0 when H1 holds).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Sprt {
    pub s0: f64,
    pub s1: f64,
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    H0,
    H1,
    Undecided,
}

/// The test's state on one rung, as `report.json` carries it.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct SprtStatus {
    #[serde(flatten)]
    pub rule: Sprt,
    pub llr: f64,
    pub lower: f64,
    pub upper: f64,
    pub pairs: u32,
    pub verdict: Verdict,
}

/// No verdict below this many pairs: the sample variance is too unstable to trust.
pub const SPRT_MIN_PAIRS: u32 = 8;

impl Sprt {
    /// `S0:S1` or `S0:S1:ALPHA:BETA`; alpha and beta default to 0.05.
    pub fn parse(s: &str) -> Result<Self, String> {
        let parts: Vec<f64> = s
            .split(':')
            .map(|p| p.trim().parse::<f64>().map_err(|e| format!("`{p}`: {e}")))
            .collect::<Result<_, _>>()?;
        let (s0, s1, alpha, beta) = match parts[..] {
            [s0, s1] => (s0, s1, 0.05, 0.05),
            [s0, s1, a, b] => (s0, s1, a, b),
            _ => return Err(format!("expected S0:S1 or S0:S1:ALPHA:BETA, got `{s}`")),
        };
        let rule = Sprt { s0, s1, alpha, beta };
        rule.validate()?;
        Ok(rule)
    }

    fn validate(&self) -> Result<(), String> {
        let unit = |x: f64| (0.0..=1.0).contains(&x);
        if !(unit(self.s0) && unit(self.s1)) || self.s0 == self.s1 {
            return Err(format!(
                "s0 and s1 must be distinct scores in [0, 1], got {} and {}",
                self.s0, self.s1
            ));
        }
        if !(self.alpha > 0.0 && self.alpha < 0.5 && self.beta > 0.0 && self.beta < 0.5) {
            return Err(format!(
                "alpha and beta must be in (0, 0.5), got {} and {}",
                self.alpha, self.beta
            ));
        }
        Ok(())
    }

    /// `(lower, upper)`: accept H0 at or below `lower`, H1 at or above `upper`. ±2.944 at 5%/5%.
    pub fn bounds(&self) -> (f64, f64) {
        (
            (self.beta / (1.0 - self.alpha)).ln(),
            ((1.0 - self.beta) / self.alpha).ln(),
        )
    }

    /// The normal-approximation log-likelihood ratio on the pair mean (fishtest's
    /// `LLR_normalized`): `N (s1 − s0) (2ȳ − s0 − s1) / (2σ²)`. A zero sample variance (every pair
    /// alike) falls back to `s(1 − s)` at the midpoint of `s0` and `s1`, which keeps the ratio
    /// finite and errs towards caution.
    pub fn llr(&self, p: &Pentanomial) -> f64 {
        let n = p.pairs() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let mut var = p.var();
        if var <= 0.0 {
            let s = (self.s0 + self.s1) / 2.0;
            var = s * (1.0 - s);
        }
        n * (self.s1 - self.s0) * (2.0 * p.mean() - self.s0 - self.s1) / (2.0 * var)
    }

    pub fn verdict(&self, p: &Pentanomial) -> Verdict {
        if p.pairs() < SPRT_MIN_PAIRS {
            return Verdict::Undecided;
        }
        let (lower, upper) = self.bounds();
        let llr = self.llr(p);
        if llr >= upper {
            Verdict::H1
        } else if llr <= lower {
            Verdict::H0
        } else {
            Verdict::Undecided
        }
    }

    pub fn status(&self, p: &Pentanomial) -> SprtStatus {
        let (lower, upper) = self.bounds();
        SprtStatus {
            rule: *self,
            llr: self.llr(p),
            lower,
            upper,
            pairs: p.pairs(),
            verdict: self.verdict(p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn penta(counts: [u32; 5]) -> Pentanomial {
        Pentanomial { counts }
    }

    #[test]
    fn mean_and_variance_of_a_known_pentanomial() {
        // Scores 0.25, 0.5 x3, 0.75 x4, 1 x2 over ten pairs.
        let p = penta([0, 1, 3, 4, 2]);
        assert_eq!(p.pairs(), 10);
        assert!((p.mean() - 0.675).abs() < 1e-12);
        assert!((p.var() - 0.050625).abs() < 1e-12);
        assert!((p.se() - (0.050625f64 / 10.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn llr_matches_the_hand_computed_value() {
        // 10 · 0.05 · (1.35 − 1.05) / (2 · 0.050625) = 40/27.
        let rule = Sprt::parse("0.5:0.55").unwrap();
        assert!((rule.llr(&penta([0, 1, 3, 4, 2])) - 40.0 / 27.0).abs() < 1e-12);
    }

    #[test]
    fn bounds_at_five_percent() {
        let (lower, upper) = Sprt::parse("0.5:0.55").unwrap().bounds();
        assert!((lower + 2.944_438_979).abs() < 1e-6, "{lower}");
        assert!((upper - 2.944_438_979).abs() < 1e-6, "{upper}");
    }

    /// Swapping candidate and opponent mirrors the scores (x → 1 − x) and the hypotheses
    /// (s0, s1 → 1 − s1, 1 − s0). The LLR must flip sign and the verdict must swap.
    #[test]
    fn verdicts_swap_when_the_sides_swap() {
        let rule = Sprt::parse("0.5:0.55").unwrap();
        let mirrored = Sprt {
            s0: 1.0 - rule.s1,
            s1: 1.0 - rule.s0,
            ..rule
        };
        for counts in [
            [2, 10, 30, 40, 18],
            [18, 40, 30, 10, 2],
            [5, 20, 50, 20, 5],
            [0, 1, 3, 4, 2],
        ] {
            let p = penta(counts);
            let mut rev = counts;
            rev.reverse();
            let q = penta(rev);
            assert!((rule.llr(&p) + mirrored.llr(&q)).abs() < 1e-9, "{counts:?}");
            let swapped = match rule.verdict(&p) {
                Verdict::H0 => Verdict::H1,
                Verdict::H1 => Verdict::H0,
                Verdict::Undecided => Verdict::Undecided,
            };
            assert_eq!(mirrored.verdict(&q), swapped, "{counts:?}");
        }
    }

    #[test]
    fn a_clear_result_decides_and_a_coin_flip_does_not() {
        let rule = Sprt::parse("0.5:0.55").unwrap();
        assert_eq!(rule.verdict(&penta([2, 10, 30, 40, 18])), Verdict::H1);
        assert_eq!(rule.verdict(&penta([18, 40, 30, 10, 2])), Verdict::H0);
        assert_eq!(rule.verdict(&penta([1, 2, 4, 2, 1])), Verdict::Undecided);
    }

    #[test]
    fn no_verdict_below_the_minimum_pairs_or_with_zero_variance_blowing_up() {
        let rule = Sprt::parse("0.5:0.55").unwrap();
        // Seven straight double wins: overwhelming, but below the guard.
        assert_eq!(rule.verdict(&penta([0, 0, 0, 0, 7])), Verdict::Undecided);
        // Every pair alike: the variance falls back instead of dividing by zero.
        let all_draws = penta([0, 0, 40, 0, 0]);
        assert!(rule.llr(&all_draws).is_finite());
        assert_eq!(rule.llr(&Pentanomial::default()), 0.0);
    }

    #[test]
    fn parse_accepts_two_or_four_fields_and_rejects_nonsense() {
        assert_eq!(
            Sprt::parse("0.5:0.55:0.1:0.2").unwrap(),
            Sprt {
                s0: 0.5,
                s1: 0.55,
                alpha: 0.1,
                beta: 0.2
            }
        );
        for bad in ["0.5", "0.5:0.5", "0.5:1.5", "0.5:0.55:0.6:0.05", "a:b", "0.5:0.55:0.05"] {
            assert!(Sprt::parse(bad).is_err(), "{bad}");
        }
    }
}
