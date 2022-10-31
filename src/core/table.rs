pub enum PosAT{
    StartMove, 
    Move, 
    StartBlock, 
    Block,
}

pub enum SimpleAT{
    StartGame, 
    SelectBothDown, 
    UseReroll, 
    DontUseReroll, 
}

pub enum AnyAT{
    Simple(SimpleAT),
    Postional(PosAT), 
}


pub enum Skill{
    Dodge, 
    Block, 
    Catch, 
    SureHands, 
}