use rand::{distributions::{Standard}, prelude::Distribution};

#[derive(Debug, Eq, Hash, PartialEq, Clone, Copy)]
pub enum PosAT{
    StartMove, 
    Move, 
    StartBlock, 
    Block,
}


#[derive(Debug, Eq, Hash, PartialEq, Clone, Copy)]
pub enum SimpleAT{
    StartGame, 
    SelectBothDown, 
    UseReroll, 
    DontUseReroll, 
    EndPlayerTurn,
    EndTurn,  
}

#[derive(Eq, Hash, PartialEq,  Debug, Clone, Copy)]
pub enum AnyAT{
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
pub enum Skill{
    Dodge, 
    Block, 
    Catch, 
    SureHands, 
    SureFeet, 
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum D6 {
    One=1,
    Two, 
    Three, 
    Four,
    Five, 
    Six, 
}

impl Distribution<D6> for Standard{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> D6 {
        D6::try_from(rng.gen_range(1..=6)).unwrap() 
    }
}

impl TryFrom<u8> for D6 {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value { 
            1 => Ok(D6::One), 
            2 => Ok(D6::Two), 
            3 => Ok(D6::Three), 
            4 => Ok(D6::Four), 
            5 => Ok(D6::Five), 
            6 => Ok(D6::Six), 
            _ => Err(()),
        }
    }
}