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
    /// Setup: put the player being placed (`info.active_player`) on this
    /// square of our own half. See `Setup` in `procedures/kickoff_procs.rs`.
    PlacePlayer,
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
    KickoffAimMiddle,
    /// Setup: send the player being placed to the reserves instead of
    /// fielding it. Only offered while the team can still field its minimum
    /// without that player.
    BenchPlayer,
    /// Use / don't use an optional skill (Wrestle, Stand Firm, ...). Which skill is asked about
    /// is the procedure on top of the stack.
    UseSkill,
    DontUseSkill,
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
    StripBall,
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
        // Appended, not grouped with General, so every earlier index (and NN skill plane) keeps
        // its place.
        Skill::StripBall,
    ];

    /// Number of skill variants.
    pub const COUNT: usize = 40;

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
            Skill::StripBall => 39,
        }
    }

    pub fn good_skills() -> Vec<Skill> {
        vec![
            Skill::Dodge, //implemented
            Skill::Block, //implemented
            Skill::Catch, //implemented
            Skill::JumpUp,
            Skill::SideStep,
            Skill::Guard,
            Skill::MightyBlow,
            Skill::Frenzy,
            Skill::Tackle,
            Skill::Wrestle,
            Skill::SureHands, //implemented
            Skill::StandFirm,
        ]
    }
}

/// A set of [`Skill`]s as a bitmask over [`Skill::index`].
///
/// Replaces a `HashSet<Skill>` per player (two per fielded player, counting `used_skills`): the
/// sets were cloned with every `GameState` clone and compared element-wise by hashing, which the
/// search does for every node it creates or recombines. Serializes as a sequence, exactly like
/// the `HashSet` it replaces, so stored corpora and positions read back unchanged.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct SkillSet(u64);

const _: () = assert!(Skill::COUNT <= 64, "SkillSet holds one bit per skill");

impl SkillSet {
    pub const fn new() -> SkillSet {
        SkillSet(0)
    }

    const fn bit(skill: Skill) -> u64 {
        1 << skill.index()
    }

    /// Adds `skill`; returns whether it was absent (as `HashSet::insert`).
    pub fn insert(&mut self, skill: Skill) -> bool {
        let absent = !self.contains(&skill);
        self.0 |= Self::bit(skill);
        absent
    }

    /// Removes `skill`; returns whether it was present (as `HashSet::remove`).
    pub fn remove(&mut self, skill: &Skill) -> bool {
        let present = self.contains(skill);
        self.0 &= !Self::bit(*skill);
        present
    }

    pub fn contains(&self, skill: &Skill) -> bool {
        self.0 & Self::bit(*skill) != 0
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }

    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// The skills in [`Skill::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = Skill> + '_ {
        Skill::ALL.into_iter().filter(|s| self.contains(s))
    }

    /// The order-independent set hash `GameState` has always used for skill sets: the length,
    /// then the wrapping sum of each skill's own `DefaultHasher` hash. Kept value-for-value so a
    /// state hashes exactly as it did when these were `HashSet`s; the per-skill hashes are fixed
    /// (`DefaultHasher::new` is unkeyed), so they are computed once.
    pub fn hash_unordered<H: std::hash::Hasher>(&self, h: &mut H) {
        use std::hash::Hash as _;
        static SKILL_HASH: std::sync::LazyLock<[u64; Skill::COUNT]> = std::sync::LazyLock::new(|| {
            Skill::ALL.map(|s| {
                let mut item_hasher = std::collections::hash_map::DefaultHasher::new();
                s.hash(&mut item_hasher);
                std::hash::Hasher::finish(&item_hasher)
            })
        });
        let mut acc: u64 = 0;
        let mut bits = self.0;
        while bits != 0 {
            acc = acc.wrapping_add(SKILL_HASH[bits.trailing_zeros() as usize]);
            bits &= bits - 1;
        }
        self.len().hash(h);
        acc.hash(h);
    }
}

impl std::fmt::Debug for SkillSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl FromIterator<Skill> for SkillSet {
    fn from_iter<I: IntoIterator<Item = Skill>>(iter: I) -> Self {
        let mut set = SkillSet::new();
        set.extend(iter);
        set
    }
}

impl Extend<Skill> for SkillSet {
    fn extend<I: IntoIterator<Item = Skill>>(&mut self, iter: I) {
        for s in iter {
            self.insert(s);
        }
    }
}

impl<const N: usize> From<[Skill; N]> for SkillSet {
    fn from(skills: [Skill; N]) -> Self {
        skills.into_iter().collect()
    }
}

impl Serialize for SkillSet {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de> Deserialize<'de> for SkillSet {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Vec::<Skill>::deserialize(deserializer)?.into_iter().collect())
    }
}

impl PosAT {
    pub const COUNT: usize = 15;
    /// Every variant, in `index` order. `index` is an exhaustive match, so a new variant fails to
    /// compile until it is numbered, and `pos_at_all_matches_index` checks it was added here too.
    pub const ALL: [PosAT; PosAT::COUNT] = [
        PosAT::StartMove,
        PosAT::StartBlitz,
        PosAT::StartPass,
        PosAT::StartFoul,
        PosAT::SelectPosition,
        PosAT::Push,
        PosAT::FollowUp,
        PosAT::StartHandoff,
        PosAT::Handoff,
        PosAT::Pass,
        PosAT::Move,
        PosAT::Foul,
        PosAT::StartBlock,
        PosAT::Block,
        PosAT::PlacePlayer,
    ];
    pub const fn index(self) -> usize {
        match self {
            PosAT::StartMove => 0,
            PosAT::StartBlitz => 1,
            PosAT::StartPass => 2,
            PosAT::StartFoul => 3,
            PosAT::SelectPosition => 4,
            PosAT::Push => 5,
            PosAT::FollowUp => 6,
            PosAT::StartHandoff => 7,
            PosAT::Handoff => 8,
            PosAT::Pass => 9,
            PosAT::Move => 10,
            PosAT::Foul => 11,
            PosAT::StartBlock => 12,
            PosAT::Block => 13,
            PosAT::PlacePlayer => 14,
        }
    }
}

impl SimpleAT {
    pub const COUNT: usize = 17;
    /// Every variant, in `index` order; see `PosAT::ALL`.
    pub const ALL: [SimpleAT; SimpleAT::COUNT] = [
        SimpleAT::SelectBothDown,
        SimpleAT::SelectPow,
        SimpleAT::SelectPush,
        SimpleAT::SelectPowPush,
        SimpleAT::SelectSkull,
        SimpleAT::UseReroll,
        SimpleAT::DontUseReroll,
        SimpleAT::EndPlayerTurn,
        SimpleAT::EndTurn,
        SimpleAT::Heads,
        SimpleAT::Tails,
        SimpleAT::Kick,
        SimpleAT::Receive,
        SimpleAT::KickoffAimMiddle,
        SimpleAT::BenchPlayer,
        SimpleAT::UseSkill,
        SimpleAT::DontUseSkill,
    ];
    pub const fn index(self) -> usize {
        match self {
            SimpleAT::SelectBothDown => 0,
            SimpleAT::SelectPow => 1,
            SimpleAT::SelectPush => 2,
            SimpleAT::SelectPowPush => 3,
            SimpleAT::SelectSkull => 4,
            SimpleAT::UseReroll => 5,
            SimpleAT::DontUseReroll => 6,
            SimpleAT::EndPlayerTurn => 7,
            SimpleAT::EndTurn => 8,
            SimpleAT::Heads => 9,
            SimpleAT::Tails => 10,
            SimpleAT::Kick => 11,
            SimpleAT::Receive => 12,
            SimpleAT::KickoffAimMiddle => 13,
            SimpleAT::BenchPlayer => 14,
            SimpleAT::UseSkill => 15,
            SimpleAT::DontUseSkill => 16,
        }
    }
}

/// A set of a small fieldless enum as a bitmask over its `index`, in the mould of [`SkillSet`]:
/// `Copy`, compared and hashed as one integer, iterated in `ALL` order, and serialised as the
/// sequence of variants so stored corpora read back unchanged.
macro_rules! enum_bitset {
    ($(#[$attr:meta])* $set:ident, $enum:ident, $bits:ty) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $set($bits);

        const _: () = assert!(
            $enum::COUNT <= <$bits>::BITS as usize,
            concat!(stringify!($set), " holds one bit per variant")
        );

        impl $set {
            pub const fn new() -> Self {
                Self(0)
            }

            const fn bit(v: $enum) -> $bits {
                1 << v.index()
            }

            /// Adds `v`; returns whether it was absent (as `HashSet::insert`).
            pub fn insert(&mut self, v: $enum) -> bool {
                let absent = !self.contains(&v);
                self.0 |= Self::bit(v);
                absent
            }

            /// Removes `v`; returns whether it was present (as `HashSet::remove`).
            pub fn remove(&mut self, v: &$enum) -> bool {
                let present = self.contains(v);
                self.0 &= !Self::bit(*v);
                present
            }

            pub fn contains(&self, v: &$enum) -> bool {
                self.0 & Self::bit(*v) != 0
            }

            pub fn clear(&mut self) {
                self.0 = 0;
            }

            pub fn len(&self) -> usize {
                self.0.count_ones() as usize
            }

            pub fn is_empty(&self) -> bool {
                self.0 == 0
            }

            /// The members in `ALL` order.
            pub fn iter(&self) -> impl Iterator<Item = $enum> + '_ {
                $enum::ALL.into_iter().filter(|v| self.contains(v))
            }
        }

        impl std::fmt::Debug for $set {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_set().entries(self.iter()).finish()
            }
        }

        impl FromIterator<$enum> for $set {
            fn from_iter<I: IntoIterator<Item = $enum>>(iter: I) -> Self {
                let mut set = Self::new();
                set.extend(iter);
                set
            }
        }

        impl Extend<$enum> for $set {
            fn extend<I: IntoIterator<Item = $enum>>(&mut self, iter: I) {
                for v in iter {
                    self.insert(v);
                }
            }
        }

        impl<const N: usize> From<[$enum; N]> for $set {
            fn from(items: [$enum; N]) -> Self {
                items.into_iter().collect()
            }
        }

        impl Serialize for $set {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.collect_seq(self.iter())
            }
        }

        impl<'de> Deserialize<'de> for $set {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                Ok(Vec::<$enum>::deserialize(deserializer)?.into_iter().collect())
            }
        }
    };
}

enum_bitset!(
    /// The positional action types offered on one square: the per-square entry of
    /// `AvailableActions::positional`. Two bytes in place of a 24-byte `SmallVec<[PosAT; 4]>`
    /// (which also spilled to the heap past four), so a whole pitch of offerings is one small
    /// `Copy` array and an `AvailableActions` clone is a memcpy.
    PosATSet,
    PosAT,
    u16
);

enum_bitset!(
    /// The simple (position-free) actions offered at a decision: `AvailableActions::simple`.
    /// Replaces a `HashSet<SimpleAT>` that was allocated and rehashed with every state clone.
    SimpleATSet,
    SimpleAT,
    u32
);

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
mod skill_set_tests {
    use super::{Skill, SkillSet};
    use std::collections::HashSet;
    use std::hash::Hasher;

    /// The pre-`SkillSet` hash of a `HashSet<Skill>` (`model::hash_set_unordered`), verbatim.
    fn legacy_hash(set: &HashSet<Skill>) -> u64 {
        use std::hash::Hash as _;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let mut acc: u64 = 0;
        for item in set {
            let mut item_hasher = std::collections::hash_map::DefaultHasher::new();
            item.hash(&mut item_hasher);
            acc = acc.wrapping_add(item_hasher.finish());
        }
        set.len().hash(&mut h);
        acc.hash(&mut h);
        h.finish()
    }

    fn sets() -> Vec<Vec<Skill>> {
        vec![
            vec![],
            vec![Skill::Block],
            vec![Skill::Catch, Skill::StripBall],
            vec![Skill::Dodge, Skill::Block, Skill::Guard, Skill::MightyBlow],
            Skill::ALL.to_vec(),
        ]
    }

    /// A state must hash exactly as it did with `HashSet`s, or the search would take different
    /// registry paths (and the corpus-identity check of a pure refactor would fail).
    #[test]
    fn hash_matches_the_hash_set_it_replaced() {
        for skills in sets() {
            let set: SkillSet = skills.iter().copied().collect();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            set.hash_unordered(&mut h);
            assert_eq!(h.finish(), legacy_hash(&skills.iter().copied().collect()), "{skills:?}");
        }
    }

    /// Serializes as a sequence, so JSON written from a `HashSet<Skill>` reads back.
    #[test]
    fn serde_is_compatible_with_hash_set() {
        for skills in sets() {
            let hs: HashSet<Skill> = skills.iter().copied().collect();
            let from_hs: SkillSet = serde_json::from_str(&serde_json::to_string(&hs).unwrap()).unwrap();
            let set: SkillSet = skills.iter().copied().collect();
            assert_eq!(from_hs, set);
            let back: HashSet<Skill> = serde_json::from_str(&serde_json::to_string(&set).unwrap()).unwrap();
            assert_eq!(back, hs);
        }
    }

    #[test]
    fn set_operations_behave_like_hash_set() {
        let mut set = SkillSet::new();
        assert!(set.is_empty());
        assert!(set.insert(Skill::Dodge));
        assert!(!set.insert(Skill::Dodge));
        assert!(set.insert(Skill::StripBall));
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Skill::Dodge) && !set.contains(&Skill::Block));
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![Skill::Dodge, Skill::StripBall]);
        assert!(set.remove(&Skill::Dodge));
        assert!(!set.remove(&Skill::Dodge));
        set.clear();
        assert!(set.is_empty());
    }
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
