use std::{
    cmp::{max, min},
    ops::Add,
};

use rand::{distributions::Standard, prelude::Distribution};

use super::{gamestate::DIRECTIONS, model::Position};

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
}
impl Distribution<D8> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> D8 {
        D8::try_from(rng.gen_range(1..=6)).unwrap()
    }
}

impl TryFrom<u8> for D8 {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(D8::One),
            2 => Ok(D8::Two),
            3 => Ok(D8::Three),
            4 => Ok(D8::Four),
            5 => Ok(D8::Five),
            6 => Ok(D8::Six),
            7 => Ok(D8::Seven),
            8 => Ok(D8::Eight),
            _ => Err(()),
        }
    }
}
impl From<D8> for Position {
    fn from(roll: D8) -> Self {
        Position::new(DIRECTIONS[roll as usize - 1])
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

impl Add<i8> for D6 {
    type Output = D6;

    fn add(self, rhs: i8) -> Self::Output {
        let result: u8 = truncate_to(1, 6, self as i8 + rhs) as u8;
        D6::try_from(result).unwrap()
    }
}
impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
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

trait RollTarget<T> {
    fn is_success(&self, roll: T) -> bool;
}

impl RollTarget<D6> for D6Target {
    fn is_success(&self, roll: D6) -> bool {
        (*self as u8) <= (roll as u8)
    }
}

impl Add<i8> for D6Target {
    type Output = D6Target;

    fn add(self, rhs: i8) -> Self::Output {
        let result: u8 = max(1, min(6, self as i8 + rhs)) as u8;
        D6Target::try_from(result).unwrap()
    }
}

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
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

impl_enum_try_from! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
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
