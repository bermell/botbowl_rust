//use super::model::{PlayerID, TeamType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]
pub enum PosAT {
    StartMove,
    StartBlitz,
    StartPass,
    StartFoul,
    SelectPosition,
    Push,
    FollowUp,
    StartHandoff,
    Handoff,
    Pass,
    Move,
    Foul,
    StartBlock,
    Block,
}

#[derive(Debug, Eq, Hash, PartialEq, PartialOrd, Ord, Clone, Copy, Serialize, Deserialize)]
pub enum SimpleAT {
    SelectBothDown,
    SelectPow,
    SelectPush,
    SelectPowPush,
    SelectSkull,
    UseReroll,
    DontUseReroll,
    EndPlayerTurn,
    EndTurn,
    Heads,
    Tails,
    Kick,
    Receive,
    SetupLine,
    EndSetup,
    KickoffAimMiddle,
    // Setup formations beyond the default line — see `Formation` in
    // `procedures/kickoff_procs.rs`. Appended at the end so the existing
    // action indices (and any model trained against them) keep their meaning.
    SetupSpread,
    SetupWedge,
    SetupZone,
}

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Skill {
    // Agility Skills
    Catch,
    Dodge,
    JumpUp,
    Leap,
    SideStep,
    Sprint,
    SureFeet,
    // Devious Skills
    DirtyPlayer,
    PileDriver,
    QuickFoul,
    Shadowing,
    SneakyGit,
    // General Skills
    Block,
    Dauntless,
    Fend,
    Frenzy,
    Kick,
    SureHands,
    Tackle,
    Wrestle,
    // Mutation Skills
    Claws,
    // Passing Skills
    Accurate,
    NervesOfSteel,
    OnTheBall,
    Pass,
    // Strength Skills
    ArmBar,
    Brawler,
    BreakTackle,
    Grab,
    Guard,
    Juggernaut,
    MightyBlow,
    StandFirm,
    // Traits
    BoneHead,
    Loner,
    Stunty,
    Throw,
    WildAnimal,
    KickOffReturn,
}
impl Skill {
    pub fn all_skills() -> Vec<Skill> {
        vec![
            Skill::Dodge,
            Skill::Throw,
            Skill::Block,
            Skill::Catch,
            Skill::SureHands,
            Skill::SureFeet,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NumBlockDices {
    ThreeUphill,
    TwoUphill,
    One,
    Two,
    Three,
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

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, Deserialize)]
// enum PrayerToNuffleEffect {
//     TreachearousTrapdoor,
//     FriendsWithTheRef(TeamType),
//     Stiletto(PlayerID),
//     IronMan(PlayerID),
//     KnuckleDusters(PlayerID),
//     BadHabit(PlayerID),
//     GreasyCleats(PlayerID),
//     BlessedStatueOfNuffle(PlayerID),
//     MalesUnderThePitch,
//     PerfectPassing(TeamType),
//     FanInteraction(TeamType),
//     NecessaryViolence(TeamType),
//     FoulingFrenzy(TeamType),
//     ThrowARock(TeamType),
//     UnderScrutiny(TeamType),
//     IntensiveTraining(PlayerID, Skill),
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerRole {
    Lineman,
    Blitzer,
    Thrower,
    Catcher,
}
