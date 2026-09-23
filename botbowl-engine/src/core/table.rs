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
    /// Every skill variant, in declaration order. This is the canonical
    /// ordering: the NN's per-skill feature planes are indexed by
    /// [`Skill::index`], so the order is part of the tensor schema — adding
    /// a variant anywhere but the end shifts every later plane and requires
    /// an `NN_SCHEMA_VERSION` bump.
    pub const ALL: [Skill; Skill::COUNT] = [
        // Agility Skills
        Skill::Catch,
        Skill::Dodge,
        Skill::JumpUp,
        Skill::Leap,
        Skill::SideStep,
        Skill::Sprint,
        Skill::SureFeet,
        // Devious Skills
        Skill::DirtyPlayer,
        Skill::PileDriver,
        Skill::QuickFoul,
        Skill::Shadowing,
        Skill::SneakyGit,
        // General Skills
        Skill::Block,
        Skill::Dauntless,
        Skill::Fend,
        Skill::Frenzy,
        Skill::Kick,
        Skill::SureHands,
        Skill::Tackle,
        Skill::Wrestle,
        // Mutation Skills
        Skill::Claws,
        // Passing Skills
        Skill::Accurate,
        Skill::NervesOfSteel,
        Skill::OnTheBall,
        Skill::Pass,
        // Strength Skills
        Skill::ArmBar,
        Skill::Brawler,
        Skill::BreakTackle,
        Skill::Grab,
        Skill::Guard,
        Skill::Juggernaut,
        Skill::MightyBlow,
        Skill::StandFirm,
        // Traits
        Skill::BoneHead,
        Skill::Loner,
        Skill::Stunty,
        Skill::Throw,
        Skill::WildAnimal,
        Skill::KickOffReturn,
    ];

    /// Number of skill variants.
    pub const COUNT: usize = 39;

    /// Dense index of a skill within [`Skill::ALL`].
    ///
    /// The match is exhaustive on purpose: adding a `Skill` variant is a
    /// **compile error here**, forcing whoever adds it to also place it in
    /// `ALL` and to consider the NN schema bump (cf. `botbowl-nn/actions.rs`).
    pub const fn index(self) -> usize {
        match self {
            Skill::Catch => 0,
            Skill::Dodge => 1,
            Skill::JumpUp => 2,
            Skill::Leap => 3,
            Skill::SideStep => 4,
            Skill::Sprint => 5,
            Skill::SureFeet => 6,
            Skill::DirtyPlayer => 7,
            Skill::PileDriver => 8,
            Skill::QuickFoul => 9,
            Skill::Shadowing => 10,
            Skill::SneakyGit => 11,
            Skill::Block => 12,
            Skill::Dauntless => 13,
            Skill::Fend => 14,
            Skill::Frenzy => 15,
            Skill::Kick => 16,
            Skill::SureHands => 17,
            Skill::Tackle => 18,
            Skill::Wrestle => 19,
            Skill::Claws => 20,
            Skill::Accurate => 21,
            Skill::NervesOfSteel => 22,
            Skill::OnTheBall => 23,
            Skill::Pass => 24,
            Skill::ArmBar => 25,
            Skill::Brawler => 26,
            Skill::BreakTackle => 27,
            Skill::Grab => 28,
            Skill::Guard => 29,
            Skill::Juggernaut => 30,
            Skill::MightyBlow => 31,
            Skill::StandFirm => 32,
            Skill::BoneHead => 33,
            Skill::Loner => 34,
            Skill::Stunty => 35,
            Skill::Throw => 36,
            Skill::WildAnimal => 37,
            Skill::KickOffReturn => 38,
        }
    }

    pub fn good_skills() -> Vec<Skill> {
        vec![
            Skill::Dodge,
            Skill::Block,
            Skill::Catch,
            Skill::JumpUp,
            Skill::SideStep,
            Skill::Guard,
            Skill::MightyBlow,
            Skill::Frenzy,
            Skill::Tackle,
            Skill::Wrestle,
            Skill::SureHands,
            Skill::StandFirm,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
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

#[cfg(test)]
mod skill_tests {
    use super::Skill;
    use std::collections::HashSet;

    /// `ALL` and `index` must agree: every variant appears exactly once, at
    /// the position its `index()` reports. The exhaustive match in `index`
    /// catches a *new* variant; this catches one that was added to the match
    /// but forgotten in `ALL` (or listed twice).
    #[test]
    fn all_skills_is_complete_and_index_is_its_position() {
        assert_eq!(Skill::ALL.len(), Skill::COUNT);
        let unique: HashSet<Skill> = Skill::ALL.into_iter().collect();
        assert_eq!(unique.len(), Skill::COUNT, "ALL contains a duplicate");
        for (i, skill) in Skill::ALL.into_iter().enumerate() {
            assert_eq!(
                skill.index(),
                i,
                "{skill:?} is at ALL[{i}] but indexes to {}",
                skill.index()
            );
        }
    }

    /// The sampling pool the curriculum draws from must be real skills, no
    /// duplicates.
    #[test]
    fn good_skills_is_a_subset_of_all_skills_without_duplicates() {
        let good = Skill::good_skills();
        let unique: HashSet<Skill> = good.iter().copied().collect();
        assert_eq!(unique.len(), good.len(), "good_skills contains a duplicate");
        for skill in good {
            assert!(skill.index() < Skill::COUNT);
        }
    }
}
