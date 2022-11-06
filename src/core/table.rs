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


pub enum Skill{
    Dodge, 
    Block, 
    Catch, 
    SureHands, 
}