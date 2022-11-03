#[derive(Eq, Hash, PartialEq)]
pub enum PosAT{
    StartMove, 
    Move, 
    StartBlock, 
    Block,
}


#[derive(Eq, Hash, PartialEq)]
pub enum SimpleAT{
    StartGame, 
    SelectBothDown, 
    UseReroll, 
    DontUseReroll, 
    EndPlayerTurn, 
}

#[derive(Eq, Hash, PartialEq)]
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


pub enum Skill{
    Dodge, 
    Block, 
    Catch, 
    SureHands, 
}