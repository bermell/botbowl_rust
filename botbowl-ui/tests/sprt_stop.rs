//! `eval --sprt` stops a rung once its sequential test decides (plan 051 step 1), and without
//! the flag a rung plays every game it was given. Scripted vs random, like `parallel_rungs.rs`:
//! deterministic and quick, and scripted wins every game, so the verdict is known in advance.

use std::process::Command;

const CAP: u32 = 400;

fn small_tier_or_skip(test: &str) -> bool {
    let w = botbowl_engine::core::model::WIDTH;
    if w > 20 {
        eprintln!("skipped {test}: board is {w} wide; run at the 14x7 tier");
        return false;
    }
    true
}

fn run(sprt: Option<&str>, tag: &str, dir: &std::path::Path) -> serde_json::Value {
    let report = dir.join(format!("{tag}.json"));
    let mut args = vec![
        "eval".to_string(),
        "--candidate-bot".into(),
        "scripted".into(),
        "--rungs".into(),
        "random".into(),
        "--skip-lectures".into(),
        "--games".into(),
        CAP.to_string(),
        "--seed".into(),
        "77".into(),
        "--parallel-games".into(),
        "4".into(),
        "--out".into(),
        report.to_str().unwrap().into(),
    ];
    if let Some(s) = sprt {
        args.extend(["--sprt".into(), s.into()]);
    }
    let out = Command::new(env!("CARGO_BIN_EXE_botbowl-ui"))
        .args(&args)
        .output()
        .expect("run eval");
    assert!(
        out.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&report).expect("report")).expect("report json");
    report["ladder"][0].clone()
}

#[test]
fn sprt_stops_a_decided_rung_early() {
    if !small_tier_or_skip("sprt_stops_a_decided_rung_early") {
        return;
    }
    let dir = std::env::temp_dir().join(format!("botbowl-sprt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let row = run(Some("0.5:0.55"), "sprt", &dir);
    let sprt = &row["sprt"];
    assert_eq!(sprt["verdict"], "H1", "scripted beats random: {row}");
    assert!(sprt["llr"].as_f64().unwrap() >= sprt["upper"].as_f64().unwrap());
    let games = row["games"].as_u64().unwrap();
    // The verdict lands at ~31 pairs; in-flight games (at most one per worker) may overshoot.
    assert!(
        games < 100,
        "stopped at {games} games, expected well under the {CAP} cap"
    );
    assert!(row["pairs"]["counts"][4].as_u64().unwrap() >= 30, "{row}");

    let fixed = run(None, "fixed", &dir);
    assert_eq!(fixed["games"].as_u64().unwrap(), CAP as u64, "no --sprt: every game");
    assert!(fixed.get("sprt").is_none(), "no --sprt: no sprt block");
    assert_eq!(fixed["points"].as_f64().unwrap(), 1.0);

    let _ = std::fs::remove_dir_all(&dir);
}
