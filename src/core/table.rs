#[derive(Debug, Eq, Hash, PartialEq, Clone, Copy)]
pub enum PosAT {
    StartMove,
    Push,
    FollowUp,
    StartHandoff,
    Handoff,
    Move,
    StartBlock,
    Block,
}

#[derive(Debug, Eq, Hash, PartialEq, Clone, Copy)]
pub enum SimpleAT {
    StartGame,
    SelectBothDown,
    SelectPow,
    SelectPush,
    SelectPowPush,
    SelectSkull,
    UseReroll,
    DontUseReroll,
    EndPlayerTurn,
    EndTurn,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum AnyAT {
    Simple(SimpleAT),
    Postional(PosAT),
}
impl From<SimpleAT> for AnyAT {
    fn from(at: SimpleAT) -> Self {
        AnyAT::Simple(at)
    }
}
impl From<PosAT> for AnyAT {
    fn from(at: PosAT) -> Self {
        AnyAT::Postional(at)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Skill {
    Dodge,
    Block,
    Catch,
    SureHands,
    SureFeet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NumBlockDices {
    Three,
    Two,
    One,
    TwoUphill,
    ThreeUphill,
}

impl From<NumBlockDices> for u8 {
    fn from(value: NumBlockDices) -> Self {
        match value {
            NumBlockDices::Three => 3,
            NumBlockDices::Two => 2,
            NumBlockDices::One => 1,
            NumBlockDices::TwoUphill => 2,
            NumBlockDices::ThreeUphill => 3,
        }
    }
}
