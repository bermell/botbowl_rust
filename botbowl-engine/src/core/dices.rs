use std::{
    cmp::{max, min},
    collections::VecDeque,
    ops::Add,
};

use rand::{distributions::Standard, prelude::Distribution, Rng};
use rand_chacha::ChaCha8Rng;

use super::{
    model::{Coord, Direction, InjuryOutcome, Weather},
    table::{NumBlockDices, SimpleAT},
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub trait RollTarget<T: Serialize + DeserializeOwned> {
    fn is_success(&self, roll: T) -> bool;
    fn add_modifer(&mut self, modifer: i8) -> &mut Self;
    fn success_prob(&self) -> f32;
}

// Shamelessly copied from https://github.com/vadorovsky/enum-try-from
macro_rules! impl_enum_try_from {
    ($(#[$meta:meta])* $vis:vis enum $name:ident {
        $($(#[$vmeta:meta])* $vname:ident $(= $val:expr)?,)*
    }, $type:ty, $err_ty:ty, $err:expr $(,)?) => {
        $(#[$meta])*
        $vis enum $name {
            $($(#[$vmeta])* $vname $(= $val)?,)*
        }

        impl TryFrom<$type> for $name {
            type Error = $err_ty;

            fn try_from(v: $type) -> Result<Self, Self::Error> {
                match v {
                    $(x if x == $name::$vname as $type => Ok($name::$vname),)*
                    _ => Err($err),
                }
            }
        }
    }
}

fn truncate_to<T: Ord>(lower_limit: T, upper_limit: T, value: T) -> T {
    max(lower_limit, min(upper_limit, value))
}

#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Deserialize, Serialize)]
pub enum Coin {
    Heads,
    Tails,
}

impl From<Coin> for SimpleAT {
    fn from(coin: Coin) -> Self {
        match coin {
            Coin::Heads => SimpleAT::Heads,
            Coin::Tails => SimpleAT::Tails,
        }
    }
}

impl Distribution<Coin> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Coin {
        match rng.gen_range(1..=2) {
            1 => Coin::Heads,
            2 => Coin::Tails,
            _ => unreachable!(),
        }
    }
}

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Deserialize, Serialize)]
    pub enum D8 {
        One = 1,
        Two,
        Three,
        Four,
        Five,
        Six,
        Seven,
        Eight,
    },
    u8,
    (),
    ()
}

impl Distribution<D8> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> D8 {
        D8::try_from(rng.gen_range(1..=8)).unwrap()
    }
}

impl From<D8> for Direction {
    fn from(roll: D8) -> Self {
        Direction::all_directions_as_array()[roll as usize - 1]
    }
}

impl From<Direction> for D8 {
    fn from(direction: Direction) -> Self {
        Direction::all_directions_iter()
            .enumerate()
            .find(|(_, &dir)| dir == direction)
            .map(|(index, _)| D8::try_from((1 + index) as u8).unwrap())
            .unwrap()
    }
}

impl From<(Coord, Coord)> for D8 {
    fn from(dxdy: (Coord, Coord)) -> Self {
        let dir: Direction = Direction::from(dxdy);
        D8::from(dir)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum BlockDice {
    Skull,
    BothDown,
    Push,
    PowPush,
    Pow,
}

impl From<BlockDice> for SimpleAT {
    fn from(roll: BlockDice) -> Self {
        match roll {
            BlockDice::Skull => SimpleAT::SelectSkull,
            BlockDice::BothDown => SimpleAT::SelectBothDown,
            BlockDice::Push => SimpleAT::SelectPush,
            BlockDice::PowPush => SimpleAT::SelectPowPush,
            BlockDice::Pow => SimpleAT::SelectPow,
        }
    }
}

impl Distribution<BlockDice> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> BlockDice {
        match rng.gen_range(1..=6) {
            1 => BlockDice::Skull,
            2 => BlockDice::BothDown,
            3 | 4 => BlockDice::Push,
            5 => BlockDice::PowPush,
            6 => BlockDice::Pow,
            _ => panic!("very wrong!"),
        }
    }
}

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Deserialize, Serialize)]
    pub enum D6 {
        One = 1,
        Two,
        Three,
        Four,
        Five,
        Six,
    },
    u8,
    (),
    ()
}
impl Add<i8> for D6 {
    type Output = D6;

    fn add(self, rhs: i8) -> Self::Output {
        D6::try_from(truncate_to(1, 6, self as i8 + rhs) as u8).unwrap()
    }
}

impl Distribution<D6> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> D6 {
        D6::try_from(rng.gen_range(1..=6)).unwrap()
    }
}

impl Add<D6> for D6 {
    type Output = Sum2D6;

    fn add(self, rhs: D6) -> Self::Output {
        Sum2D6::try_from(self as u8 + rhs as u8).unwrap()
    }
}

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Deserialize, Serialize)]
    pub enum D3 {
        One = 1,
        Two,
        Three,
    },
    u8,
    (),
    ()
}

impl Distribution<D3> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> D3 {
        D3::try_from(rng.gen_range(1..=3)).unwrap()
    }
}

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]
    pub enum D6Target {
        TwoPlus = 2,
        ThreePlus,
        FourPlus,
        FivePlus,
        SixPlus,
    },
    u8,
    (),
    ()
}

impl RollTarget<D6> for D6Target {
    fn is_success(&self, roll: D6) -> bool {
        (*self as u8) <= (roll as u8)
    }

    fn add_modifer(&mut self, modifer: i8) -> &mut D6Target {
        *self = D6Target::try_from(truncate_to(2, 6, *self as i8 - modifer) as u8).unwrap();
        self
    }

    fn success_prob(&self) -> f32 {
        const PROBS: [f32; 7] = [
            f32::NAN,
            f32::NAN,
            5.0 / 6.0,
            4.0 / 6.0,
            3.0 / 6.0,
            2.0 / 6.0,
            1.0 / 6.0,
        ];
        PROBS[*self as usize]
    }
}

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]
    pub enum Sum2D6 {
        Two = 2,
        Three,
        Four,
        Five,
        Six,
        Seven,
        Eight,
        Nine,
        Ten,
        Eleven,
        Twelve,
    },
    u8,
    (),
    ()
}

// The Weather table
impl From<Sum2D6> for Weather {
    fn from(value: Sum2D6) -> Self {
        match value {
            Sum2D6::Two => Weather::Sweltering,
            Sum2D6::Three => Weather::Sunny,
            Sum2D6::Eleven => Weather::Rain,
            Sum2D6::Twelve => Weather::Blizzard,
            _ => Weather::Nice,
        }
    }
}

impl Distribution<Sum2D6> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Sum2D6 {
        Sum2D6::try_from(rng.gen_range(1..=6) + rng.gen_range(1..=6)).unwrap()
    }
}

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]
    pub enum Sum2D6Target {
        TwoPlus = 2,
        ThreePlus,
        FourPlus,
        FivePlus,
        SixPlus,
        SevenPlus,
        EightPlus,
        NinePlus,
        TenPlus,
        ElevenPlus,
        TwelvePlus,
    },
    u8,
    (),
    ()
}

impl RollTarget<Sum2D6> for Sum2D6Target {
    fn is_success(&self, roll: Sum2D6) -> bool {
        (*self as u8) <= (roll as u8)
    }

    fn add_modifer(&mut self, modifer: i8) -> &mut Sum2D6Target {
        *self = Sum2D6Target::try_from(truncate_to(2, 12, *self as i8 - modifer) as u8).unwrap();
        self
    }

    fn success_prob(&self) -> f32 {
        const PROBS: [f32; 13] = [
            f32::NAN,
            f32::NAN,
            1.0,
            35.0 / 36.0,
            33.0 / 36.0,
            30.0 / 36.0,
            26.0 / 36.0,
            21.0 / 36.0,
            15.0 / 36.0,
            10.0 / 36.0,
            6.0 / 36.0,
            3.0 / 36.0,
            1.0 / 36.0,
        ];
        PROBS[*self as usize]
    }
}

/// Block-dice override for `DicePolicy`. The grand plan's curriculum
/// recipe is "2 dice blocks are knockdowns, 1 die is a push or even a
/// skull" — i.e., attacker-favored blocks succeed, anything else fails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockDicePolicy {
    /// No override on block dice; fall through to FIFO/RNG.
    #[default]
    Default,
    /// Attacker-favored counts (`Two`, `Three`) yield all `Pow` so the
    /// attacker selects a knockdown. One-die and uphill rolls fall
    /// through. This is the minimum needed to make a 2-dice block
    /// deterministic for "Get the ball — Hard".
    KnockdownAtAdvantage,
}

impl BlockDicePolicy {
    fn resolve(&self, num_dices: NumBlockDices) -> Option<RollResult> {
        match self {
            BlockDicePolicy::Default => None,
            BlockDicePolicy::KnockdownAtAdvantage => match num_dices {
                NumBlockDices::Two | NumBlockDices::Three => {
                    let n = u8::from(num_dices) as usize;
                    let mut dices: [Option<BlockDice>; 3] = [None, None, None];
                    for slot in dices.iter_mut().take(n) {
                        *slot = Some(BlockDice::Pow);
                    }
                    Some(RollResult::BlockDice(dices))
                }
                _ => None,
            },
        }
    }
}

/// Resolve any `RequestedRoll` by sampling fresh values from `rng`.
///
/// Used by `DiceMode::RollDice` and by `DicePolicy::resolve` for roll
/// types the policy doesn't override. Kept here rather than on
/// `GameState` so the same routine drives both modes without
/// duplicating the per-variant unpacking.
pub fn resolve_with_rng(request: RequestedRoll, rng: &mut ChaCha8Rng) -> RollResult {
    match request {
        RequestedRoll::D6 => RollResult::D6(rng.gen()),
        RequestedRoll::D6PassFail(target) => {
            if target.is_success(rng.gen()) {
                RollResult::Pass
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::D6ThreeOutcomes(low_target, high_target) => {
            let roll: D6 = rng.gen();
            if high_target.is_success(roll) {
                RollResult::Pass
            } else if low_target.is_success(roll) {
                RollResult::MiddleOutcome
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::Sum2D6 => RollResult::Sum2D6(rng.gen::<D6>() + rng.gen::<D6>()),
        RequestedRoll::Sum2D6PassFail(target) => {
            if target.is_success(rng.gen::<D6>() + rng.gen::<D6>()) {
                RollResult::Pass
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::Sum2D6ThreeOutcomes(low_target, high_target) => {
            let roll: Sum2D6 = rng.gen::<D6>() + rng.gen::<D6>();
            if high_target.is_success(roll) {
                RollResult::Pass
            } else if low_target.is_success(roll) {
                RollResult::MiddleOutcome
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::D8 => RollResult::D8(rng.gen()),
        RequestedRoll::Coin => RollResult::Coin(rng.gen()),
        RequestedRoll::Deviate => RollResult::Deviate(rng.gen(), rng.gen()),
        RequestedRoll::FoulArmor(target) => {
            let roll1: D6 = rng.gen();
            let roll2: D6 = rng.gen();
            RollResult::FoulArmor {
                broken: target.is_success(roll1 + roll2),
                ejected: roll1 == roll2,
            }
        }
        RequestedRoll::FoulInjury(ko_target, cas_target) => {
            let roll1: D6 = rng.gen();
            let roll2: D6 = rng.gen();
            let outcome = if cas_target.is_success(roll1 + roll2) {
                InjuryOutcome::Casualty
            } else if ko_target.is_success(roll1 + roll2) {
                InjuryOutcome::KO
            } else {
                InjuryOutcome::Stunned
            };
            RollResult::FoulInjury {
                outcome,
                ejected: roll1 == roll2,
            }
        }
        RequestedRoll::ThrowIn => RollResult::ThrowIn {
            direction: rng.gen(),
            distance: rng.gen::<D6>() + rng.gen::<D6>(),
        },
        RequestedRoll::BlockDice(num_dices) => {
            let mut dices: [Option<BlockDice>; 3] = [None, None, None];
            for slot in dices.iter_mut().take(u8::from(num_dices) as usize) {
                *slot = Some(rng.gen());
            }
            RollResult::BlockDice(dices)
        }
        RequestedRoll::Scatter => RollResult::Scatter(rng.gen(), rng.gen(), rng.gen()),
    }
}

/// FIFO queue of pre-pinned dice values, used by `DiceMode::FixedDice`.
///
/// Tests and builders push concrete dice values; when the engine asks
/// for a roll the corresponding queue is popped. The queue is strict:
/// `resolve_from_fixes` panics with a diagnostic if the engine requests
/// a roll the queue can't satisfy.
#[derive(Clone, Default, Serialize, Deserialize, PartialEq, Eq, Debug)]
pub struct FixedDice {
    d3_fixes: VecDeque<D3>,
    d6_fixes: VecDeque<D6>,
    blockdice_fixes: VecDeque<BlockDice>,
    d8_fixes: VecDeque<D8>,
    coin_fixes: VecDeque<Coin>,
}

impl FixedDice {
    pub fn fix_coin(&mut self, value: Coin) {
        self.coin_fixes.push_back(value);
    }
    pub fn fix_d3(&mut self, value: u8) {
        self.d3_fixes.push_back(D3::try_from(value).unwrap());
    }
    pub fn fix_d6(&mut self, value: u8) {
        self.d6_fixes.push_back(D6::try_from(value).unwrap());
    }
    pub fn fix_d8(&mut self, value: u8) {
        self.d8_fixes.push_back(D8::try_from(value).unwrap());
    }
    pub fn fix_d8_direction(&mut self, direction: Direction) {
        self.d8_fixes.push_back(D8::from(direction));
    }
    pub fn fix_blockdice(&mut self, value: BlockDice) {
        self.blockdice_fixes.push_back(value);
    }
    pub fn is_empty(&self) -> bool {
        self.d3_fixes.is_empty()
            && self.d6_fixes.is_empty()
            && self.d8_fixes.is_empty()
            && self.blockdice_fixes.is_empty()
            && self.coin_fixes.is_empty()
    }
    pub fn blockdice_fixes_len(&self) -> usize {
        self.blockdice_fixes.len()
    }
    pub fn assert_is_empty(&self) {
        assert!(
            self.is_empty(),
            "fixed dices are not empty: d3:{:?}, d6:{:?}, d8:{:?}, blockdice:{:?}, coin:{:?}",
            self.d3_fixes,
            self.d6_fixes,
            self.d8_fixes,
            self.blockdice_fixes,
            self.coin_fixes,
        );
    }
    fn pop_d3(&mut self, request: &RequestedRoll) -> D3 {
        self.d3_fixes
            .pop_front()
            .unwrap_or_else(|| panic!("FixedDice queue empty for D3 (request: {:?})", request))
    }
    fn pop_d6(&mut self, request: &RequestedRoll) -> D6 {
        self.d6_fixes
            .pop_front()
            .unwrap_or_else(|| panic!("FixedDice queue empty for D6 (request: {:?})", request))
    }
    fn pop_d8(&mut self, request: &RequestedRoll) -> D8 {
        self.d8_fixes
            .pop_front()
            .unwrap_or_else(|| panic!("FixedDice queue empty for D8 (request: {:?})", request))
    }
    fn pop_coin(&mut self, request: &RequestedRoll) -> Coin {
        self.coin_fixes
            .pop_front()
            .unwrap_or_else(|| panic!("FixedDice queue empty for Coin (request: {:?})", request))
    }
    fn pop_blockdice(&mut self, request: &RequestedRoll) -> BlockDice {
        self.blockdice_fixes
            .pop_front()
            .unwrap_or_else(|| panic!("FixedDice queue empty for BlockDice (request: {:?})", request))
    }
}

/// Resolve any `RequestedRoll` by popping pinned values from `fixes`.
///
/// Used by `DiceMode::FixedDice`. Panics with a diagnostic naming the
/// requested roll type if the relevant queue is empty.
pub fn resolve_from_fixes(request: RequestedRoll, fixes: &mut FixedDice) -> RollResult {
    match request {
        RequestedRoll::D6 => RollResult::D6(fixes.pop_d6(&request)),
        RequestedRoll::D6PassFail(target) => {
            if target.is_success(fixes.pop_d6(&request)) {
                RollResult::Pass
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::D6ThreeOutcomes(low_target, high_target) => {
            let roll = fixes.pop_d6(&request);
            if high_target.is_success(roll) {
                RollResult::Pass
            } else if low_target.is_success(roll) {
                RollResult::MiddleOutcome
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::Sum2D6 => RollResult::Sum2D6(fixes.pop_d6(&request) + fixes.pop_d6(&request)),
        RequestedRoll::Sum2D6PassFail(target) => {
            if target.is_success(fixes.pop_d6(&request) + fixes.pop_d6(&request)) {
                RollResult::Pass
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::Sum2D6ThreeOutcomes(low_target, high_target) => {
            let roll = fixes.pop_d6(&request) + fixes.pop_d6(&request);
            if high_target.is_success(roll) {
                RollResult::Pass
            } else if low_target.is_success(roll) {
                RollResult::MiddleOutcome
            } else {
                RollResult::Fail
            }
        }
        RequestedRoll::D8 => RollResult::D8(fixes.pop_d8(&request)),
        RequestedRoll::Coin => RollResult::Coin(fixes.pop_coin(&request)),
        RequestedRoll::Deviate => RollResult::Deviate(fixes.pop_d6(&request), fixes.pop_d8(&request)),
        RequestedRoll::FoulArmor(target) => {
            let roll1 = fixes.pop_d6(&request);
            let roll2 = fixes.pop_d6(&request);
            RollResult::FoulArmor {
                broken: target.is_success(roll1 + roll2),
                ejected: roll1 == roll2,
            }
        }
        RequestedRoll::FoulInjury(ko_target, cas_target) => {
            let roll1 = fixes.pop_d6(&request);
            let roll2 = fixes.pop_d6(&request);
            let outcome = if cas_target.is_success(roll1 + roll2) {
                InjuryOutcome::Casualty
            } else if ko_target.is_success(roll1 + roll2) {
                InjuryOutcome::KO
            } else {
                InjuryOutcome::Stunned
            };
            RollResult::FoulInjury {
                outcome,
                ejected: roll1 == roll2,
            }
        }
        RequestedRoll::ThrowIn => RollResult::ThrowIn {
            direction: fixes.pop_d3(&request),
            distance: fixes.pop_d6(&request) + fixes.pop_d6(&request),
        },
        RequestedRoll::BlockDice(num_dices) => {
            let mut dices: [Option<BlockDice>; 3] = [None, None, None];
            for slot in dices.iter_mut().take(u8::from(num_dices) as usize) {
                *slot = Some(fixes.pop_blockdice(&request));
            }
            RollResult::BlockDice(dices)
        }
        RequestedRoll::Scatter => {
            RollResult::Scatter(fixes.pop_d8(&request), fixes.pop_d8(&request), fixes.pop_d8(&request))
        }
    }
}

/// Target-aware override of dice resolution.
///
/// Curriculum lectures and other consumers can install a policy to pin
/// pass/fail roll outcomes by *target*, independent of the FIFO fixes queue
/// — implementing the grand plan's "3+ succeeds, 4+ fails" semantics. The
/// default variant is a no-op and preserves the existing queue/RNG path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DicePolicy {
    /// No override; fall through to the FIFO fixes queue and RNG.
    #[default]
    Default,
    /// Pass/fail rolls succeed iff the request's target is no stricter
    /// than the configured threshold. A request with a D6 target ≤ `d6`
    /// returns `Pass`; otherwise `Fail`. 2D6 pass/fail rolls use `sum2d6`
    /// analogously. Block-dice rolls consult `block_dice`. Non-pass/fail,
    /// non-block-dice rolls (raw D6/D8/scatter/coin) always fall through.
    SucceedAtOrEasier {
        d6: D6Target,
        sum2d6: Sum2D6Target,
        block_dice: BlockDicePolicy,
    },
}

impl DicePolicy {
    /// Total resolution: the policy is required to return a `RollResult`
    /// for every `RequestedRoll`. Variants that don't have a target-aware
    /// override for a given roll type delegate to `resolve_with_rng`
    /// — that way lectures can pin pickups/dodges to a fixed outcome while
    /// scatter/bounce/coin remain stochastic.
    pub fn resolve(&mut self, request: RequestedRoll, rng: &mut ChaCha8Rng) -> RollResult {
        match *self {
            DicePolicy::Default => resolve_with_rng(request, rng),
            DicePolicy::SucceedAtOrEasier { d6, sum2d6, block_dice } => match request {
                RequestedRoll::D6PassFail(target) => {
                    if (target as u8) <= (d6 as u8) {
                        RollResult::Pass
                    } else {
                        RollResult::Fail
                    }
                }
                RequestedRoll::Sum2D6PassFail(target) => {
                    if (target as u8) <= (sum2d6 as u8) {
                        RollResult::Pass
                    } else {
                        RollResult::Fail
                    }
                }
                RequestedRoll::BlockDice(num_dices) => block_dice
                    .resolve(num_dices)
                    .unwrap_or_else(|| resolve_with_rng(request, rng)),
                other => resolve_with_rng(other, rng),
            },
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum RequestedRoll {
    BlockDice(NumBlockDices),
    Coin,
    D6,
    D6PassFail(D6Target),
    D6ThreeOutcomes(D6Target, D6Target),
    D8,
    FoulArmor(Sum2D6Target),
    FoulInjury(Sum2D6Target, Sum2D6Target),
    Deviate, // TODO: this should be called deviate
    Scatter,
    Sum2D6,
    Sum2D6PassFail(Sum2D6Target),
    Sum2D6ThreeOutcomes(Sum2D6Target, Sum2D6Target),
    ThrowIn,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum RollResult {
    BlockDice([Option<BlockDice>; 3]),
    Coin(Coin),
    Pass,
    Fail,
    FoulArmor { broken: bool, ejected: bool },
    FoulInjury { outcome: InjuryOutcome, ejected: bool },
    MiddleOutcome,
    D6(D6),
    D8(D8),
    Deviate(D6, D8),
    Scatter(D8, D8, D8),
    Sum2D6(Sum2D6),
    ThrowIn { direction: D3, distance: Sum2D6 },
}

impl RequestedRoll {
    pub fn is_compatible(&self, result: RollResult) -> bool {
        match (self, result) {
            (RequestedRoll::BlockDice(_), RollResult::BlockDice(_)) => true,
            (RequestedRoll::Coin, RollResult::Coin(_)) => true,
            (RequestedRoll::D6, RollResult::D6(_)) => true,
            (RequestedRoll::D6PassFail(_), RollResult::Pass | RollResult::Fail) => true,
            (RequestedRoll::D6ThreeOutcomes(_, _), RollResult::Pass | RollResult::MiddleOutcome | RollResult::Fail) => {
                true
            }
            (RequestedRoll::D8, RollResult::D8(_)) => true,
            (RequestedRoll::FoulArmor(_), RollResult::FoulArmor { .. }) => true,
            (RequestedRoll::FoulInjury(_, _), RollResult::FoulInjury { .. }) => true,
            (RequestedRoll::Deviate, RollResult::Deviate(_, _)) => true,
            (RequestedRoll::Scatter, RollResult::Scatter(_, _, _)) => true,
            (RequestedRoll::Sum2D6, RollResult::Sum2D6(_)) => true,
            (RequestedRoll::Sum2D6PassFail(_), RollResult::Pass | RollResult::Fail) => true,
            (
                RequestedRoll::Sum2D6ThreeOutcomes(_, _),
                RollResult::Pass | RollResult::MiddleOutcome | RollResult::Fail,
            ) => true,
            (RequestedRoll::ThrowIn, RollResult::ThrowIn { .. }) => true,
            _ => false,
        }
    }
}
