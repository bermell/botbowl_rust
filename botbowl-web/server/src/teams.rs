//! Teams on the server: the built-ins plus the saved ones, applying a roster to a fresh game,
//! and which picture each player is drawn with.
//!
//! Saved teams are one JSON file per team, `<dir>/<slug>.json`, where `<dir>` is
//! `~/.config/botbowl/teams` by default ([`default_dir`]); uploaded pictures go to `<dir>/img/`
//! and are served at `img/custom/`. The files are re-read on every listing, so a hand-edited or
//! copied-in team shows up on the next page load.
//!
//! A player's picture cannot ride on the engine state — `PlayerStats` has no room for one, and
//! player ids are reassigned every time a player moves between dugout and pitch — so [`Looks`]
//! finds it from what the engine *does* keep: the player's team, role, stats and skills, which
//! identify its position on any roster where two positions differ in at least one of them.

use std::path::{Path, PathBuf};

use botbowl_engine::core::dices::D6Target;
use botbowl_engine::core::gamestate::GameState;
use botbowl_engine::core::model::{PlayerStats, TeamType};
use botbowl_engine::core::table::Skill;
use botbowl_web_proto::team::{self, PositionDef, SkillInfo, TeamDef};

use crate::mirror;

/// `$XDG_CONFIG_HOME/botbowl/teams`, else `~/.config/botbowl/teams` — next to the hub token.
pub fn default_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("botbowl").join("teams"))
}

/// Every skill, by its display label.
pub fn skills() -> Vec<SkillInfo> {
    let good = Skill::good_skills();
    Skill::ALL
        .into_iter()
        .map(|s| SkillInfo {
            label: mirror::skill_label(s).to_string(),
            implemented: good.contains(&s),
        })
        .collect()
}

pub fn skill_from_label(label: &str) -> Option<Skill> {
    Skill::ALL.into_iter().find(|s| mirror::skill_label(*s) == label)
}

/// The engine stats for a position. Errors on a skill label the engine does not know.
pub fn stats_of(p: &PositionDef, team: TeamType) -> Result<PlayerStats, String> {
    let mut stats = PlayerStats::new_lineman(team);
    stats.role = mirror::role_from_proto(p.role);
    stats.ma = p.ma;
    stats.str_ = p.st;
    stats.ag = p.ag;
    stats.av = p.av;
    stats.pass = D6Target::try_from(p.pa.clamp(2, 6)).expect("2..=6 is a D6 target");
    stats.skills.clear();
    for label in &p.skills {
        let skill = skill_from_label(label).ok_or_else(|| format!("{}: unknown skill {label:?}", p.name))?;
        stats.give_skill(skill);
    }
    Ok(stats)
}

/// Replace `team`'s whole dugout with `def`'s roster, slot for slot. Only valid before kickoff,
/// when every player is still in reserves.
pub fn apply(state: &mut GameState, team: TeamType, def: &TeamDef) -> Result<(), String> {
    let n = state.get_dugout().filter(|p| p.stats.team == team).count();
    let roster = def.roster(n);
    if roster.len() != n {
        return Err(format!("{} has no positions", def.name));
    }
    let stats: Vec<PlayerStats> = roster
        .iter()
        .map(|&i| stats_of(&def.positions[i], team))
        .collect::<Result<_, _>>()?;
    for (player, stats) in state.get_dugout_mut().filter(|p| p.stats.team == team).zip(stats) {
        player.stats = stats;
    }
    Ok(())
}

/// Lower every player's MA to at most `cap`, benched and fielded alike — the lobby's "no natural
/// one-turn" ([`botbowl_web_proto::msg::GameSpec::no_natural_one_turn`]).
pub fn cap_ma(state: &mut GameState, cap: u8) {
    for p in state.get_dugout_mut() {
        p.stats.ma = p.stats.ma.min(cap);
    }
    let fielded: Vec<_> = state.get_players_on_pitch().map(|p| p.id).collect();
    for id in fielded {
        if let Ok(p) = state.get_mut_player(id) {
            p.stats.ma = p.stats.ma.min(cap);
        }
    }
}

/// Which picture each player is drawn with: the two sides' teams, resolved per player by
/// [`TeamDef::position_of`], falling back to the first position of the player's role.
#[derive(Debug, Clone, Default)]
pub struct Looks {
    /// `[home, away]`.
    pub teams: [Option<TeamDef>; 2],
}

impl Looks {
    pub fn sprite(&self, stats: &PlayerStats, used: bool) -> String {
        let team = mirror::team_to_proto(stats.team);
        let role = mirror::role_to_proto(stats.role);
        let side = match stats.team {
            TeamType::Home => 0,
            TeamType::Away => 1,
        };
        let Some(def) = &self.teams[side] else {
            return role.sprite(team, used);
        };
        let skills: Vec<String> = Skill::ALL
            .into_iter()
            .filter(|s| stats.skills.contains(s))
            .map(|s| mirror::skill_label(s).to_string())
            .collect();
        def.position_of(role, stats.ma, stats.str_, stats.ag, stats.av, &skills)
            // MA may have been capped ("no natural one-turn"): the same position with more.
            .or_else(|| {
                def.positions.iter().find(|p| {
                    p.role == role
                        && p.ma >= stats.ma
                        && (p.st, p.ag, p.av) == (stats.str_, stats.ag, stats.av)
                        && p.skills.iter().all(|s| skills.contains(s))
                        && p.skills.len() == skills.len()
                })
            })
            .or_else(|| def.look_for_role(role))
            .map(|p| team::picture_sprite(&p.picture, team, used))
            .unwrap_or_else(|| role.sprite(team, used))
    }
}

/// The built-ins plus whatever is saved under `dir`.
#[derive(Debug, Clone)]
pub struct TeamStore {
    pub dir: Option<PathBuf>,
}

impl TeamStore {
    pub fn img_dir(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join("img"))
    }

    /// Built-ins first, then saved teams by name. A saved file whose name collides with a
    /// built-in, or that does not parse, is skipped with a line on stderr.
    pub fn list(&self) -> Vec<TeamDef> {
        let mut teams = team::builtin_teams();
        let Some(dir) = &self.dir else { return teams };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return teams;
        };
        let mut saved: Vec<TeamDef> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .filter_map(|p| {
                let text = std::fs::read_to_string(&p).ok()?;
                match serde_json::from_str::<TeamDef>(&text) {
                    Ok(t) => Some(TeamDef { builtin: false, ..t }.sanitised()),
                    Err(e) => {
                        eprintln!("[teams] skipping {}: {e}", p.display());
                        None
                    }
                }
            })
            .filter(|t| !teams.iter().any(|b| team::slug(&b.name) == team::slug(&t.name)))
            .collect();
        saved.sort_by(|a, b| a.name.cmp(&b.name));
        teams.extend(saved);
        teams
    }

    pub fn find(&self, name: &str) -> Option<TeamDef> {
        self.list().into_iter().find(|t| t.name == name)
    }

    fn path_of(&self, name: &str) -> Result<PathBuf, String> {
        let dir = self.dir.as_ref().ok_or("this server has no team directory")?;
        let slug = team::slug(name);
        if slug.is_empty() {
            return Err("a team needs a name".into());
        }
        Ok(dir.join(format!("{slug}.json")))
    }

    /// Check a team can be played: a name, at least one position, every skill known.
    pub fn validate(def: &TeamDef) -> Result<(), String> {
        if def.name.trim().is_empty() {
            return Err("a team needs a name".into());
        }
        if def.positions.is_empty() {
            return Err("a team needs at least one position".into());
        }
        for p in &def.positions {
            stats_of(p, TeamType::Home)?;
            if !p.picture.starts_with("custom/") && !p.picture.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Err(format!("{}: bad picture {:?}", p.name, p.picture));
            }
            if let Some(file) = p.picture.strip_prefix("custom/") {
                if !safe_file_name(file) {
                    return Err(format!("{}: bad picture {:?}", p.name, p.picture));
                }
            }
        }
        Ok(())
    }

    pub fn save(&self, def: TeamDef) -> Result<(), String> {
        let def = TeamDef { builtin: false, ..def }.sanitised();
        Self::validate(&def)?;
        if team::builtin_teams()
            .iter()
            .any(|b| team::slug(&b.name) == team::slug(&def.name))
        {
            return Err(format!(
                "{:?} is a built-in team — save it under another name",
                def.name
            ));
        }
        let path = self.path_of(&def.name)?;
        std::fs::create_dir_all(path.parent().expect("a file in a dir")).map_err(|e| e.to_string())?;
        let json = serde_json::to_string_pretty(&def).map_err(|e| e.to_string())?;
        std::fs::write(&path, json).map_err(|e| format!("could not write {}: {e}", path.display()))
    }

    pub fn delete(&self, name: &str) -> Result<(), String> {
        if team::builtin_teams().iter().any(|b| b.name == name) {
            return Err("built-in teams cannot be deleted".into());
        }
        let path = self.path_of(name)?;
        std::fs::remove_file(&path).map_err(|e| format!("could not delete {}: {e}", path.display()))
    }

    /// Store an uploaded `data:image/<png|gif|jpeg|webp>;base64,...` under `img/`, named by its
    /// content hash so the same picture uploaded twice is one file. Returns `custom/<file>`.
    pub fn save_picture(&self, data_url: &str) -> Result<String, String> {
        const MAX_BYTES: usize = 512 * 1024;
        let dir = self.img_dir().ok_or("this server has no team directory")?;
        let rest = data_url.strip_prefix("data:image/").ok_or("not an image data URL")?;
        let (kind, data) = rest.split_once(";base64,").ok_or("not a base64 data URL")?;
        let ext = match kind {
            "png" => "png",
            "gif" => "gif",
            "jpeg" | "jpg" => "jpg",
            "webp" => "webp",
            other => return Err(format!("unsupported image type {other:?} (png, gif, jpeg, webp)")),
        };
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data.trim())
            .map_err(|e| format!("bad base64: {e}"))?;
        if bytes.len() > MAX_BYTES {
            return Err(format!(
                "picture is {} KB; the limit is {} KB",
                bytes.len() / 1024,
                MAX_BYTES / 1024
            ));
        }
        let name = format!("{:016x}.{ext}", fnv1a(&bytes));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(&name), &bytes).map_err(|e| e.to_string())?;
        Ok(format!("custom/{name}"))
    }

    /// Sprite stems under `<assets>/iconssmall/` (one per player look, without the colourway and
    /// acted suffixes), then the uploaded pictures.
    pub fn pictures(&self, assets: Option<&Path>) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(Ok(entries)) = assets.map(|a| std::fs::read_dir(a.join("iconssmall"))) {
            let mut stems: Vec<String> = entries
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter_map(|n| n.strip_suffix(".gif").map(str::to_string))
                // Only the base file (away colourway, acted); the variants are derived.
                .filter(|n| !n.ends_with("an") && !n.ends_with('b'))
                .collect();
            stems.sort();
            stems.dedup();
            out.extend(stems);
        }
        if let Some(Ok(entries)) = self.img_dir().map(std::fs::read_dir) {
            let mut custom: Vec<String> = entries
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| safe_file_name(n))
                .map(|n| format!("custom/{n}"))
                .collect();
            custom.sort();
            out.extend(custom);
        }
        out
    }
}

fn safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
}

/// A stable content hash for picture file names (not a security property: the name only has to
/// dedupe identical uploads).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use botbowl_engine::core::gamestate::GameStateBuilder;
    use botbowl_engine::core::model::BoardDims;

    fn dims() -> BoardDims {
        BoardDims::try_new(16, 9, 6).unwrap_or_else(|_| BoardDims::from_env())
    }

    /// "No natural one-turn": the cap is one short of the engine's own LOS-to-end-zone
    /// distance on every board, and a capped player keeps their position's picture.
    #[test]
    fn the_one_turn_cap_falls_one_short_of_the_end_zone_and_keeps_the_pictures() {
        use botbowl_web_proto::msg::BoardSpec;
        for (w, h) in [(8, 3), (12, 5), (14, 7), (16, 9), (20, 9), (26, 15)] {
            let spec = BoardSpec::new(w, h, 3);
            let (ew, eh, n) = spec.engine_dims();
            let Ok(dims) = BoardDims::try_new(ew, eh, n) else { continue };
            assert_eq!(
                spec.no_one_turn_ma() as i16,
                dims.los_to_endzone_distance() as i16 - 1,
                "{w}x{h}"
            );
        }

        let mut state = GameStateBuilder::new_start_of_game_with(dims());
        let human = botbowl_web_proto::team::builtin_teams().remove(0);
        apply(&mut state, TeamType::Home, &human).unwrap();
        let looks = Looks { teams: [Some(human), None] };
        let before: Vec<String> = state.get_dugout().map(|p| looks.sprite(&p.stats, false)).collect();
        cap_ma(&mut state, 5);
        assert!(state.get_dugout().all(|p| p.stats.ma <= 5));
        let after: Vec<String> = state.get_dugout().map(|p| looks.sprite(&p.stats, false)).collect();
        assert_eq!(before, after, "a capped Catcher is still drawn as a Catcher");
    }

    #[test]
    fn the_default_team_is_the_engine_roster_exactly() {
        let fresh = GameStateBuilder::new_start_of_game_with(dims());
        let mut applied = fresh.clone();
        let human = team::builtin_teams()
            .into_iter()
            .find(|t| t.name == team::DEFAULT_TEAM)
            .unwrap();
        apply(&mut applied, TeamType::Home, &human).unwrap();
        apply(&mut applied, TeamType::Away, &human).unwrap();
        assert!(
            applied == fresh,
            "the Human team must reproduce the roster the nets were trained on"
        );
    }

    #[test]
    fn every_builtin_team_applies_and_draws_its_own_pictures() {
        for def in team::builtin_teams() {
            let mut state = GameStateBuilder::new_start_of_game_with(dims());
            apply(&mut state, TeamType::Away, &def).unwrap_or_else(|e| panic!("{}: {e}", def.name));
            let looks = Looks {
                teams: [None, Some(def.clone())],
            };
            for p in state.get_dugout().filter(|p| p.stats.team == TeamType::Away) {
                let sprite = looks.sprite(&p.stats, false);
                assert!(
                    def.positions.iter().any(|d| sprite.contains(&d.picture)),
                    "{}: {sprite}",
                    def.name
                );
            }
        }
    }

    #[test]
    fn unknown_skills_are_refused() {
        let mut def = team::builtin_teams().remove(0);
        def.positions[0].skills.push("Regeneration".into());
        assert!(TeamStore::validate(&def).unwrap_err().contains("Regeneration"));
    }

    #[test]
    fn save_list_delete_round_trip() {
        let dir = std::env::temp_dir().join(format!("botbowl-teams-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = TeamStore { dir: Some(dir.clone()) };
        let mut def = team::builtin_teams().remove(1);
        assert!(store.save(def.clone()).is_err(), "a built-in name is refused");
        def.name = "My Orcs".into();
        def.positions[1].skills.push("Tackle".into());
        store.save(def.clone()).unwrap();
        assert!(dir.join("my-orcs.json").is_file());
        let back = store.find("My Orcs").unwrap();
        assert!(!back.builtin);
        assert!(back.positions[1].skills.contains(&"Tackle".to_string()));
        // 1x1 transparent png.
        let png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        let pic = store.save_picture(png).unwrap();
        assert!(pic.starts_with("custom/") && pic.ends_with(".png"));
        assert!(store.pictures(None).contains(&pic));
        store.delete("My Orcs").unwrap();
        assert!(store.find("My Orcs").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
