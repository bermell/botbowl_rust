//! Plan 043: bot presets.
//!
//! A preset exists so two arms of an A/B can be named, committed and diffed. That only works if
//! the name pins the behaviour, which means two things have to hold: a preset must ignore the
//! environment, and a knob a preset does not mention must land on the shipped default rather than
//! whatever the machine happened to be configured with.

use std::io::Write;

use botbowl_mcts::{BackupMode, MctsConfig, PuctMode, TieBreak};
use botbowl_play::bots::{load_mcts_config, SearchConfig};

fn write_preset(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create preset");
    f.write_all(body.as_bytes()).expect("write preset");
    path
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("botbowl-bot-config-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// The control arm: a preset that sets nothing is exactly the shipped configuration.
///
/// This is what makes `cfgs/baseline.toml` an honest control — if this drifts, the "baseline" arm
/// of every A/B silently stops being the baseline.
#[test]
fn an_empty_preset_is_the_shipped_configuration() {
    let dir = tmpdir("empty");
    let path = write_preset(&dir, "baseline.toml", "# nothing overridden\n");

    let loaded = load_mcts_config(&path).expect("load");
    assert_eq!(loaded.name, "baseline", "the file stem names the configuration");
    assert_eq!(
        loaded.config,
        MctsConfig::new(),
        "an empty preset must equal the compiled defaults"
    );
}

/// The repository's own control file, checked rather than assumed.
#[test]
fn the_committed_baseline_preset_is_the_shipped_configuration() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("cfgs/baseline.toml");
    let loaded = load_mcts_config(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()));
    assert_eq!(loaded.name, "baseline");
    assert_eq!(loaded.config, MctsConfig::new());
}

/// A preset overrides what it names and nothing else, in the same vocabulary the CLI flags use.
#[test]
fn a_preset_sets_only_what_it_names() {
    let dir = tmpdir("partial");
    let path = write_preset(
        &dir,
        "aggressive.toml",
        r#"
backup = "mean"
fpu_reduction = 0.25
horizon_turns = 2
tie_break = "asc"

[puct.normalised_q]
c = 1.4
range_floor = 0.1
"#,
    );

    let loaded = load_mcts_config(&path).expect("load");
    assert_eq!(loaded.name, "aggressive");
    let c = loaded.config;
    assert_eq!(c.backup, BackupMode::Mean);
    assert_eq!(c.fpu_reduction, 0.25);
    assert_eq!(c.horizon_turns, 2);
    assert_eq!(c.tie_break, TieBreak::Asc);
    assert!(matches!(
        c.puct,
        PuctMode::NormalisedQ { c, range_floor } if c == 1.4 && range_floor == 0.1
    ));

    // Untouched knobs are the shipped defaults, not something else.
    let d = MctsConfig::new();
    assert_eq!(c.tree_reuse, d.tree_reuse);
    assert_eq!(c.virtual_loss, d.virtual_loss);
    assert_eq!(c.memory_mode, d.memory_mode);
    assert_eq!(c.horizon, d.horizon);
}

/// The reproducibility guarantee. `MctsConfig`'s `Default` is `from_env()`, so serde pointed at
/// `Default` would have quietly absorbed the environment into every unnamed field — which would
/// make a named configuration mean different things on different machines.
///
/// Single-threaded and restoring the variable, because the process environment is shared.
#[test]
fn a_preset_ignores_the_environment() {
    let dir = tmpdir("env");
    let path = write_preset(&dir, "quiet.toml", "fpu_reduction = 0.5\n");

    let key = "BLOOD_MCTS_BACKUP";
    let restore = std::env::var(key).ok();
    std::env::set_var(key, "mean");
    let loaded = load_mcts_config(&path).expect("load");
    match restore {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }

    assert_eq!(
        loaded.config.backup,
        BackupMode::Minimax,
        "a hostile BLOOD_MCTS_BACKUP must not reach a named preset"
    );
    assert_eq!(loaded.config.fpu_reduction, 0.5, "the preset's own knob still applies");
}

/// A typo must fail the run, not silently leave the knob at its default and produce an A/B whose
/// two arms are identical.
#[test]
fn an_unknown_knob_is_an_error() {
    let dir = tmpdir("typo");
    let path = write_preset(&dir, "typo.toml", "fpu_reducton = 0.25\n");

    let err = load_mcts_config(&path).expect_err("a misspelled knob must not load");
    let msg = err.to_string();
    assert!(
        msg.contains("fpu_reducton"),
        "the error should name the offending key, got: {msg}"
    );
}

/// `SearchConfig` must stay `Copy`: it is re-exported verbatim by `botbowl-hub-proto` and crosses
/// the worker protocol inside `Task`, so a non-`Copy` field would ripple through the hub.
#[test]
fn search_config_carrying_a_preset_is_still_copy() {
    fn assert_copy<T: Copy>(_: &T) {}

    let mut sc = SearchConfig::iterations(100);
    assert!(sc.config.is_none(), "the default is still `leave the bot's default`");
    sc.config = Some(MctsConfig::new());
    assert_copy(&sc);

    // And it round-trips through postcard, which is what the hub actually speaks.
    let bytes = postcard::to_allocvec(&sc).expect("encode");
    let back: SearchConfig = postcard::from_bytes(&bytes).expect("decode");
    assert_eq!(back.config, sc.config);
}
