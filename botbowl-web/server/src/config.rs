//! `~/.config/botbowl/web.toml`: where the web play app finds its sprites, nets and teams, so a
//! box does not need them copied into the checkout or passed on every command line.
//!
//! Read by `botbowl-web-server` and by `botbowl-hub serve` for its `/play/` mount. Precedence is
//! command-line flag, then this file, then the built-in default. A missing file is created with
//! every key documented and whatever this machine has filled in (the sibling sprite checkout,
//! the repo's `models/`), so it is there to edit. A file that does not parse is an error, not a
//! silent fallback: a typo'd key would otherwise look exactly like the setting being ignored.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// The parsed file. Every key is optional.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebConfig {
    /// The sprite directory of a `botbowl` checkout (`.../botbowl/web/static/img`).
    pub assets_dir: Option<PathBuf>,
    /// Directories searched for `.onnx` nets, up to three levels deep, in this order.
    #[serde(default)]
    pub models_dirs: Vec<PathBuf>,
    /// A worker's model cache, offered by the names the hub sent. Default
    /// `~/.cache/botbowl/models`; `""` turns it off.
    pub worker_cache: Option<PathBuf>,
    /// Saved teams and uploaded pictures. Default `~/.config/botbowl/teams`.
    pub teams_dir: Option<PathBuf>,
    /// The `trunk build` output of `botbowl-web/client`.
    pub dist_dir: Option<PathBuf>,
}

/// `$XDG_CONFIG_HOME/botbowl/web.toml`, else `~/.config/botbowl/web.toml` — next to `hub.token`.
pub fn default_path() -> PathBuf {
    botbowl_hub_proto::default_token_path().with_file_name("web.toml")
}

/// `~/x` → `$HOME/x`; anything else as written.
fn expand(p: &Path) -> PathBuf {
    match (p.strip_prefix("~"), std::env::var_os("HOME")) {
        (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => p.to_path_buf(),
    }
}

impl WebConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut c: WebConfig = toml::from_str(text).map_err(|e| e.to_string())?;
        c.assets_dir = c.assets_dir.as_deref().map(expand);
        c.models_dirs = c.models_dirs.iter().map(|p| expand(p)).collect();
        c.worker_cache = c.worker_cache.as_deref().map(expand);
        c.teams_dir = c.teams_dir.as_deref().map(expand);
        c.dist_dir = c.dist_dir.as_deref().map(expand);
        Ok(c)
    }

    /// Read `path`, or create it from [`template`] when it does not exist.
    pub fn load_or_create(path: &Path, assets_hint: Option<&Path>, models_hint: Option<&Path>) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let text = template(assets_hint, models_hint);
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                match std::fs::write(path, &text) {
                    Ok(()) => eprintln!("wrote {} — edit it to point at your sprites and nets", path.display()),
                    Err(e) => eprintln!("could not write {}: {e}", path.display()),
                }
                Self::parse(&text).map_err(|e| format!("the built-in template does not parse: {e}"))
            }
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// The worker cache to scan: the configured one, `None` when set to `""`, else the default.
    pub fn worker_cache(&self) -> Option<PathBuf> {
        match &self.worker_cache {
            Some(p) if p.as_os_str().is_empty() => None,
            Some(p) => Some(p.clone()),
            None => Some(botbowl_hub_proto::default_model_cache_dir()),
        }
    }
}

/// Paths given on the command line; each beats the file.
#[derive(Debug, Clone, Default)]
pub struct Flags {
    pub dist_dir: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    pub assets_dir: Option<PathBuf>,
    pub teams_dir: Option<PathBuf>,
}

/// Everything the play app needs to find, resolved.
#[derive(Debug, Clone)]
pub struct Paths {
    pub dist_dir: PathBuf,
    /// The first models directory; `extra_model_dirs` follow it.
    pub models_dir: PathBuf,
    pub extra_model_dirs: Vec<PathBuf>,
    pub assets_dir: Option<PathBuf>,
    pub teams_dir: Option<PathBuf>,
    pub worker_cache: Option<PathBuf>,
    /// The config file that was read (or created).
    pub config: PathBuf,
}

/// The repo root, from this crate's source path — the fallback for the client build and nets.
fn repo_path(relative: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(relative);
    std::fs::canonicalize(&path).unwrap_or(path)
}

/// Flag, then `web.toml` (created on first use), then the built-in default.
pub fn resolve(flags: Flags) -> Result<Paths, String> {
    let sibling_assets = Some(repo_path("../botbowl/botbowl/web/static/img")).filter(|p| p.is_dir());
    let repo_models = Some(repo_path("models")).filter(|p| p.is_dir());
    let config = default_path();
    let file = WebConfig::load_or_create(&config, sibling_assets.as_deref(), repo_models.as_deref())?;
    let mut dirs: Vec<PathBuf> = flags
        .models_dir
        .into_iter()
        .chain(file.models_dirs.iter().cloned())
        .collect();
    if dirs.is_empty() {
        dirs.push(repo_path("models"));
    }
    let models_dir = dirs.remove(0);
    Ok(Paths {
        dist_dir: flags
            .dist_dir
            .or(file.dist_dir.clone())
            .unwrap_or_else(|| repo_path("botbowl-web/client/dist")),
        extra_model_dirs: dirs.into_iter().filter(|d| *d != models_dir).collect(),
        models_dir,
        assets_dir: flags.assets_dir.or(file.assets_dir.clone()).or(sibling_assets),
        teams_dir: flags
            .teams_dir
            .or(file.teams_dir.clone())
            .or_else(crate::teams::default_dir),
        worker_cache: file.worker_cache().filter(|p| p.is_dir()),
        config,
    })
}

fn quoted(p: &Path) -> String {
    format!("{:?}", p.to_string_lossy())
}

/// The file written on first run: every key documented, the ones this machine can fill in set.
pub fn template(assets_hint: Option<&Path>, models_hint: Option<&Path>) -> String {
    let assets = match assets_hint {
        Some(p) => format!("assets_dir = {}", quoted(p)),
        None => "# assets_dir = \"~/repos/botbowl/botbowl/web/static/img\"".into(),
    };
    let models = match models_hint {
        Some(p) => format!("models_dirs = [{}]", quoted(p)),
        None => "# models_dirs = [\"~/repos/botbowl_rust/models\"]".into(),
    };
    format!(
        "# Where the web play app (botbowl-web-server, and botbowl-hub's /play/) finds things.\n\
         # Command-line flags override these; unset keys use the built-in defaults.\n\
         \n\
         # The sprite directory of a `botbowl` checkout. Player icons are not under the botbowl\n\
         # licence, so they are never copied into this repo.\n\
         {assets}\n\
         \n\
         # Directories searched for .onnx nets (three levels deep), in order.\n\
         {models}\n\
         \n\
         # A remote worker's model cache: the nets the hub shipped to this box, offered under the\n\
         # names the hub sent with them. Default ~/.cache/botbowl/models; \"\" turns it off.\n\
         # worker_cache = \"~/.cache/botbowl/models\"\n\
         \n\
         # Saved teams (<name>.json) and uploaded pictures (img/).\n\
         # teams_dir = \"~/.config/botbowl/teams\"\n\
         \n\
         # The client build (cd botbowl-web/client && trunk build --release).\n\
         # dist_dir = \"~/repos/botbowl_rust/botbowl-web/client/dist\"\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_template_parses_with_and_without_hints() {
        let c = WebConfig::parse(&template(None, None)).unwrap();
        assert!(c.assets_dir.is_none() && c.models_dirs.is_empty());
        let c = WebConfig::parse(&template(Some(Path::new("/a/img")), Some(Path::new("/r/models")))).unwrap();
        assert_eq!(c.assets_dir.as_deref(), Some(Path::new("/a/img")));
        assert_eq!(c.models_dirs, vec![PathBuf::from("/r/models")]);
    }

    #[test]
    fn a_typo_is_an_error_and_tilde_expands() {
        assert!(WebConfig::parse("asset_dir = \"/x\"").is_err());
        if let Some(home) = std::env::var_os("HOME") {
            let c = WebConfig::parse("teams_dir = \"~/t\"").unwrap();
            assert_eq!(c.teams_dir, Some(PathBuf::from(home).join("t")));
        }
        assert_eq!(WebConfig::parse("worker_cache = \"\"").unwrap().worker_cache(), None);
        assert!(WebConfig::parse("").unwrap().worker_cache().is_some());
    }

    #[test]
    fn a_missing_file_is_created_and_then_read_back() {
        let dir = std::env::temp_dir().join(format!("botbowl-web-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("web.toml");
        let c = WebConfig::load_or_create(&path, Some(Path::new("/s/img")), None).unwrap();
        assert_eq!(c.assets_dir.as_deref(), Some(Path::new("/s/img")));
        assert!(path.is_file());
        std::fs::write(&path, "assets_dir = \"/other\"\n").unwrap();
        let c = WebConfig::load_or_create(&path, Some(Path::new("/s/img")), None).unwrap();
        assert_eq!(
            c.assets_dir.as_deref(),
            Some(Path::new("/other")),
            "an existing file wins over hints"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
