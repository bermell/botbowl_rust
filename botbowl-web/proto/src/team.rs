//! Teams: a named roster of positions, each with stats, skills and a picture.
//!
//! The engine has no team concept — every game is two copies of one human-like roster, told
//! apart by colour. A [`TeamDef`] is the web app's: the server builds a game's two rosters from
//! the chosen defs (`server/src/teams.rs`), and the view draws each player with its position's
//! picture. Custom teams are saved as JSON under `~/.config/botbowl/teams/`.
//!
//! Stats are in the **engine's** terms, which are LRB6 values: `ag` is the old agility value
//! (higher is better; the roll target is `7 - ag`), `pa` is a pass *target* (`2..=6`, i.e.
//! `2+..6+`). Skills are carried by their display label ([`SkillInfo::label`]), so a saved team
//! is readable by hand; the server rejects a label it does not know.

use serde::{Deserialize, Serialize};

use crate::action::TeamType;
use crate::view::PlayerRole;

/// The engine's stat ceilings (`PlayerStats::MAX_*`), mirrored so the editor can clamp.
pub const MAX_MA: u8 = 10;
pub const MAX_ST: u8 = 8;
pub const MAX_AG: u8 = 6;
pub const MAX_AV: u8 = 12;

/// One position on a roster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionDef {
    /// "Blitzer", "Gutter Runner", ...
    pub name: String,
    /// The engine role. It is not cosmetic: the scripted kickoff setup ranks players by role.
    pub role: PlayerRole,
    pub ma: u8,
    pub st: u8,
    pub ag: u8,
    /// Pass target, `2..=6`.
    pub pa: u8,
    pub av: u8,
    /// Display labels, e.g. `"Sure Hands"`.
    pub skills: Vec<String>,
    /// A sprite stem under `img/iconssmall/` (`"oblitzer1"`, drawn in the team's colourway and
    /// greyed once it has acted), or `custom/<file>` — an uploaded picture, drawn as is.
    pub picture: String,
    /// The most of this position a roster may field. The **first** position is the filler and
    /// takes whatever slots the others leave, whatever its count says.
    pub max: u8,
}

/// A roster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeamDef {
    /// Unique among the teams a server offers; also the saved file's stem (slugged).
    pub name: String,
    pub positions: Vec<PositionDef>,
    /// Shipped with the server rather than saved by a user. Not stored in the JSON.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub builtin: bool,
}

/// One skill the editor can offer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillInfo {
    pub label: String,
    /// The engine implements its rules. The others exist in the enum (and the net sees them)
    /// but do nothing in a game.
    pub implemented: bool,
}

impl TeamDef {
    /// Slot `i` of a roster of `n`, as indices into `positions`: every non-filler position
    /// gets `min(max, (n / 6).max(1))` — the engine's own positional split — in order until
    /// the roster is full, and the first position takes the rest.
    pub fn roster(&self, n: usize) -> Vec<usize> {
        if self.positions.is_empty() {
            return Vec::new();
        }
        let per = (n / 6).max(1);
        let mut out = Vec::with_capacity(n);
        for (i, p) in self.positions.iter().enumerate().skip(1) {
            let take = (p.max as usize).min(per).min(n.saturating_sub(out.len() + 1));
            out.extend(std::iter::repeat_n(i, take));
        }
        let filler = n - out.len();
        // Filler first, so a small board's lone player is the roster's lineman.
        let mut roster = vec![0; filler];
        roster.extend(out);
        roster
    }

    /// Clamp everything into what the engine accepts. The server applies it before a game and
    /// before a save, so a hand-edited JSON cannot panic a session.
    pub fn sanitised(mut self) -> Self {
        self.name = self.name.trim().chars().take(40).collect();
        self.positions.truncate(16);
        for p in &mut self.positions {
            p.name = p.name.trim().chars().take(40).collect();
            p.ma = p.ma.clamp(1, MAX_MA);
            p.st = p.st.clamp(1, MAX_ST);
            p.ag = p.ag.clamp(1, MAX_AG);
            p.pa = p.pa.clamp(2, 6);
            p.av = p.av.clamp(3, MAX_AV);
            p.skills.sort();
            p.skills.dedup();
        }
        self
    }

    /// Which position a player with these numbers is, for drawing it. `None` when no position
    /// matches — a random-start drive's players are drawn from the lineman template.
    pub fn position_of(
        &self,
        role: PlayerRole,
        ma: u8,
        st: u8,
        ag: u8,
        av: u8,
        skills: &[String],
    ) -> Option<&PositionDef> {
        self.positions.iter().find(|p| {
            p.role == role && p.ma == ma && p.st == st && p.ag == ag && p.av == av && {
                let mut a = p.skills.clone();
                a.sort();
                let mut b = skills.to_vec();
                b.sort();
                a == b
            }
        })
    }

    /// The position a player of `role` is drawn as when nothing matches it exactly: the first of
    /// that role, else the filler.
    pub fn look_for_role(&self, role: PlayerRole) -> Option<&PositionDef> {
        self.positions
            .iter()
            .find(|p| p.role == role)
            .or(self.positions.first())
    }
}

/// The sprite path under `img/` for a picture.
pub fn picture_sprite(picture: &str, team: TeamType, used: bool) -> String {
    if picture.starts_with("custom/") {
        return picture.to_string();
    }
    let home = if team == TeamType::Home { "b" } else { "" };
    let not_acted = if used { "" } else { "an" };
    format!("iconssmall/{picture}{home}{not_acted}.gif")
}

/// A file-name-safe stem for a team name.
pub fn slug(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    s.trim_matches('-').to_string()
}

#[allow(clippy::too_many_arguments)]
fn pos(
    name: &str,
    role: PlayerRole,
    ma: u8,
    st: u8,
    ag: u8,
    pa: u8,
    av: u8,
    skills: &[&str],
    picture: &str,
    max: u8,
) -> PositionDef {
    PositionDef {
        name: name.into(),
        role,
        ma,
        st,
        ag,
        pa,
        av,
        skills: {
            let mut v: Vec<String> = skills.iter().map(|s| s.to_string()).collect();
            v.sort();
            v
        },
        picture: picture.into(),
        max,
    }
}

fn team(name: &str, positions: Vec<PositionDef>) -> TeamDef {
    TeamDef {
        name: name.into(),
        positions,
        builtin: true,
    }
}

/// The name of the team that reproduces the engine's own roster exactly — the one every net
/// was trained on, and the default for both sides.
pub const DEFAULT_TEAM: &str = "Human";

/// The shipped rosters. LRB6 stats, cut down to the skills the engine has: a skill it does not
/// know (Thick Skull, Horns, Regeneration, ...) is left off rather than faked.
pub fn builtin_teams() -> Vec<TeamDef> {
    use PlayerRole::*;
    vec![
        // Exactly `PlayerStats::new_*`, in the engine's roster order.
        team(
            DEFAULT_TEAM,
            vec![
                pos("Lineman", Lineman, 6, 3, 3, 4, 8, &[], "hlineman1", 16),
                pos("Blitzer", Blitzer, 7, 3, 3, 4, 9, &["Block"], "hblitzer1", 4),
                pos("Catcher", Catcher, 8, 2, 3, 5, 8, &["Catch", "Dodge"], "hcatcher1", 4),
                pos(
                    "Thrower",
                    Thrower,
                    6,
                    3,
                    3,
                    2,
                    8,
                    &["Sure Hands", "Throw"],
                    "hthrower1",
                    2,
                ),
            ],
        ),
        team(
            "Orc",
            vec![
                pos("Lineman", Lineman, 5, 3, 3, 4, 9, &[], "olineman1", 16),
                pos("Blitzer", Blitzer, 6, 3, 3, 4, 9, &["Block"], "oblitzer1", 4),
                pos("Goblin", Catcher, 6, 2, 3, 4, 7, &["Dodge", "Stunty"], "goblin1", 4),
                pos(
                    "Thrower",
                    Thrower,
                    5,
                    3,
                    3,
                    3,
                    8,
                    &["Sure Hands", "Pass"],
                    "othrower1",
                    2,
                ),
                pos("Black Orc", Lineman, 4, 4, 2, 6, 9, &[], "oblackorc1", 4),
            ],
        ),
        team(
            "Dwarf",
            vec![
                pos(
                    "Blocker",
                    Lineman,
                    4,
                    3,
                    2,
                    5,
                    9,
                    &["Block", "Tackle"],
                    "dlongbeard1",
                    16,
                ),
                pos("Blitzer", Blitzer, 5, 3, 3, 4, 9, &["Block"], "dblitzer1", 2),
                pos("Runner", Thrower, 6, 3, 3, 3, 8, &["Sure Hands"], "drunner1", 2),
                pos(
                    "Troll Slayer",
                    Blitzer,
                    5,
                    3,
                    2,
                    6,
                    8,
                    &["Block", "Dauntless", "Frenzy"],
                    "dslayer1",
                    2,
                ),
            ],
        ),
        team(
            "Skaven",
            vec![
                pos("Lineman", Lineman, 7, 3, 3, 4, 7, &[], "sklineman1", 16),
                pos("Stormvermin", Blitzer, 7, 3, 3, 4, 8, &["Block"], "skstorm1", 2),
                pos("Gutter Runner", Catcher, 9, 2, 4, 4, 7, &["Dodge"], "skrunner1", 4),
                pos(
                    "Thrower",
                    Thrower,
                    7,
                    3,
                    3,
                    2,
                    7,
                    &["Sure Hands", "Pass"],
                    "skthrower1",
                    2,
                ),
            ],
        ),
        team(
            "Wood Elf",
            vec![
                pos("Lineman", Lineman, 7, 3, 4, 3, 7, &[], "welineman1", 16),
                pos(
                    "Wardancer",
                    Blitzer,
                    8,
                    3,
                    4,
                    3,
                    7,
                    &["Block", "Dodge", "Leap"],
                    "weblitzer1",
                    2,
                ),
                pos(
                    "Catcher",
                    Catcher,
                    8,
                    2,
                    4,
                    3,
                    7,
                    &["Catch", "Dodge", "Sprint"],
                    "wecatcher1",
                    4,
                ),
                pos("Thrower", Thrower, 7, 3, 4, 2, 7, &["Pass"], "wethrower1", 2),
            ],
        ),
        team(
            "High Elf",
            vec![
                pos("Lineman", Lineman, 6, 3, 4, 3, 8, &[], "helineman1", 16),
                pos("Blitzer", Blitzer, 7, 3, 4, 3, 8, &["Block"], "heblitzer1", 2),
                pos("Catcher", Catcher, 8, 3, 4, 3, 7, &["Catch"], "hecatcher1", 4),
                pos(
                    "Thrower",
                    Thrower,
                    6,
                    3,
                    4,
                    2,
                    8,
                    &["Pass", "Sure Hands"],
                    "hethrower1",
                    2,
                ),
            ],
        ),
        team(
            "Dark Elf",
            vec![
                pos("Lineman", Lineman, 6, 3, 4, 4, 8, &[], "delineman1", 16),
                pos("Blitzer", Blitzer, 7, 3, 4, 4, 8, &["Block"], "deblitzer1", 4),
                pos(
                    "Witch Elf",
                    Catcher,
                    7,
                    3,
                    4,
                    5,
                    7,
                    &["Dodge", "Frenzy", "Jump Up"],
                    "dewitchelf1",
                    2,
                ),
                pos("Runner", Thrower, 7, 3, 4, 3, 7, &["Pass"], "dethrower1", 2),
            ],
        ),
        team(
            "Amazon",
            vec![
                pos("Linewoman", Lineman, 6, 3, 3, 4, 7, &["Dodge"], "amlineman1", 16),
                pos("Blitzer", Blitzer, 6, 3, 3, 4, 7, &["Block", "Dodge"], "amblitzer1", 4),
                pos("Catcher", Catcher, 6, 3, 3, 4, 7, &["Catch", "Dodge"], "amcatcher1", 2),
                pos("Thrower", Thrower, 6, 3, 3, 3, 7, &["Dodge", "Pass"], "amthrower1", 2),
            ],
        ),
        team(
            "Norse",
            vec![
                pos("Lineman", Lineman, 6, 3, 3, 4, 7, &["Block"], "nlineman1", 16),
                pos(
                    "Berserker",
                    Blitzer,
                    6,
                    3,
                    3,
                    4,
                    7,
                    &["Block", "Frenzy", "Jump Up"],
                    "nblitzer1",
                    2,
                ),
                pos(
                    "Runner",
                    Catcher,
                    7,
                    3,
                    3,
                    4,
                    7,
                    &["Block", "Dauntless"],
                    "ncatcher1",
                    2,
                ),
                pos("Thrower", Thrower, 6, 3, 3, 3, 7, &["Block", "Pass"], "nthrower1", 2),
            ],
        ),
        team(
            "Lizardmen",
            vec![
                pos("Skink", Catcher, 8, 2, 3, 5, 7, &["Dodge", "Stunty"], "lmskink1", 16),
                pos("Saurus", Blitzer, 6, 4, 1, 6, 9, &[], "lmsaurus1", 6),
            ],
        ),
        team(
            "Undead",
            vec![
                pos("Skeleton", Lineman, 5, 3, 2, 6, 7, &[], "uskeleton1", 16),
                pos("Wight", Blitzer, 6, 3, 3, 5, 8, &["Block"], "uwight1", 2),
                pos("Ghoul", Catcher, 7, 3, 3, 4, 7, &["Dodge"], "ughoul1", 4),
                pos("Mummy", Lineman, 3, 5, 1, 6, 9, &["Mighty Blow"], "umummy1", 2),
                pos("Zombie", Lineman, 4, 3, 2, 6, 8, &[], "uzombie1", 4),
            ],
        ),
        team(
            "Chaos",
            vec![
                pos("Beastman", Lineman, 6, 3, 3, 4, 8, &[], "cbeastman1", 16),
                pos("Chaos Warrior", Blitzer, 5, 4, 3, 5, 9, &[], "cwarrior1", 4),
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_roster_fills_every_slot_with_the_engine_split() {
        let human = builtin_teams().into_iter().find(|t| t.name == DEFAULT_TEAM).unwrap();
        // 12: the engine's 6 linemen + 2 each of blitzer/catcher/thrower.
        let r = human.roster(12);
        assert_eq!(r.len(), 12);
        assert_eq!(r.iter().filter(|&&i| i == 0).count(), 6);
        for i in 1..4 {
            assert_eq!(r.iter().filter(|&&j| j == i).count(), 2);
        }
        for n in 1..=16 {
            assert_eq!(human.roster(n).len(), n, "roster of {n}");
            assert_eq!(human.roster(n)[0], 0, "the filler comes first");
        }
    }

    #[test]
    fn caps_hold() {
        let lizards = builtin_teams().into_iter().find(|t| t.name == "Lizardmen").unwrap();
        // 18 slots at 3 per positional — the saurus's max (6) is not reached, the per cap is.
        assert_eq!(lizards.roster(18).iter().filter(|&&i| i == 1).count(), 3);
    }

    #[test]
    fn builtin_names_are_unique_and_slug() {
        let teams = builtin_teams();
        for t in &teams {
            assert_eq!(teams.iter().filter(|u| slug(&u.name) == slug(&t.name)).count(), 1);
            assert_eq!(t.clone().sanitised(), *t, "{} is already within the caps", t.name);
        }
        assert_eq!(slug("Wood Elf  #2"), "wood-elf-2");
    }

    #[test]
    fn custom_pictures_are_drawn_as_is() {
        assert_eq!(picture_sprite("custom/ab.png", TeamType::Home, true), "custom/ab.png");
        assert_eq!(
            picture_sprite("oblitzer1", TeamType::Home, false),
            "iconssmall/oblitzer1ban.gif"
        );
        assert_eq!(
            picture_sprite("oblitzer1", TeamType::Away, true),
            "iconssmall/oblitzer1.gif"
        );
    }
}
