use std::{
    cmp::{max, min},
    ops::Add,
};

use rand::{distributions::Standard, prelude::Distribution};

use super::{
    model::{Coord, Direction, InjuryOutcome, Weather},
    table::{NumBlockDices, SimpleAT},
};

pub trait RollTarget<T> {
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
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
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
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
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
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
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
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
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
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RequestedRoll {
    BlockDice(NumBlockDices),
    Coin,
    D6,
    D6PassFail(D6Target),
    D6ThreeOutcomes(D6Target, D6Target),
    D8,
    FoulArmor(Sum2D6Target),
    FoulInjury(Sum2D6Target, Sum2D6Target),
    Kick,
    Sum2D6,
    Sum2D6PassFail(Sum2D6Target),
    Sum2D6ThreeOutcomes(Sum2D6Target, Sum2D6Target),
    ThrowIn,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RollResult {
    BlockDice([Option<BlockDice>; 3]),
    Coin(Coin),
    Pass,
    Fail,
    FoulArmor {
        broken: bool,
        ejected: bool,
    },
    FoulInjury {
        outcome: InjuryOutcome,
        ejected: bool,
    },
    MiddleOutcome,
    D6(D6),
    D8(D8),
    Kick(D6, D8),
    Sum2D6(Sum2D6),
    ThrowIn {
        direction: D3,
        distance: Sum2D6,
    },
}
