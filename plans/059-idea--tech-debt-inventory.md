# 059 — Technical-debt inventory

**Status:** inventory, 2026-10-07. Nothing here is started.

Six parallel review passes (engine; mcts + recon_mcts; ui/play/hub/worker/nn/data;
scripts + train; web + docs/plans; cross-cutting build/test/env) over the whole repo,
asked one question: what makes this repo annoying to work with day to day, for humans
and for agents? This file is the merged, deduplicated answer, ranked.

Caveat: the reviews started on a detached HEAD at `ae721b7` and the checkout was moved
to `master` (`5076090`, 21 commits later) mid-review. Those commits are plan 058 docs,
the `SkillSet` bitset, two `recon_mcts` descent optimisations, the forced-move skip,
web UI polish and four new scripts. None of them addresses a structural item below,
but line numbers cited in `dynamics.rs`, `tree.rs`, `session.rs`, `pitch.rs` and
`state.rs` are approximate.

Effort: S = under an hour, M = half a day to a day, L = multi-day. Payoff is for the
daily loop, not for bot strength.

---

## Part 1 — The top 20, by payoff over effort

Start at the top. Items 1–8 are each under a day and remove something that bites every
session.

| # | Item | Effort | Payoff |
|---|------|--------|--------|
| 5 | **`scripts/` is 108 flat files, 50 of them `exp0NN_*.sh` that cannot run on master** (none uses a v9 net; two use the removed `BLOOD_MCTS_BACKUP` / `--vs-backup`). Move to `scripts/archive/` with a `# Status: archived <date>, last runnable at <commit>` header; add `scripts/README.md` listing the ~15 standing tools. | S | high |
| 6 | **Four ways to configure a search, and their defaults drift:** `BLOOD_MCTS_*` env vars (25 of them), per-knob CLI flags, `cfgs/*.toml`, and the web `MctsSpec`. `eval` passes `puct: Some(raw)` so env PUCT is silently ignored there but honoured by `dataset`; the web cannot express gumbel/budget/chance/setup knobs so it cannot play the loop's bot; `impl Default for MctsConfig` reads the environment; ~20 mcts test files build bots via `from_env` so an exported `BLOOD_MCTS_BUDGET` changes test results. Target: presets are the one source of truth, env/CLI become `--set key=value` overrides on a preset, web loads a preset by name, `Default` = `new()`. | M | high |
| 7 | **No cargo profiles.** Debug MCTS tests run unoptimised (`hash_quality` 17 s, ui unittests 21 s, mcts suite ~190 s). Add `[profile.dev.package.{botbowl-engine,botbowl-mcts,recon_mcts}] opt-level = 2`, a `[profile.profiling]` that inherits release with `debug = 2` (so profiling stops invalidating the release cache via `RUSTFLAGS`), and `[workspace.dependencies]` / `[workspace.package]` to stop version drift. | S | high |
| 8 | **ui `dataset`/`eval` and hub `job generate`/`job eval` are copy-pasted flag structs (23+10+8 vs ~55 fields; 34 vs 36) that have already drifted into a bug:** the hub fixed the one-sided-preset `vs:` label panic (`botbowl-hub/src/main.rs:806`), the ui copy at `botbowl-ui/src/eval.rs:499` still panics. `CliEvaluator`/`CliCandidateBot`/`CliDifficulty`/`DatasetMode` are re-declared; `size_dist_of` is a line copy of `SizeArgs::to_dist`; hub parses and ignores `--parallel-games`, `--nn-server`, `--skip-lectures`, `--trials`. Move the arg structs into `botbowl-play` (or a `botbowl-cli-args` crate) and `#[command(flatten)]` them from both binaries. | M | high |
| 9 | **The "ignored = benchmark suite, ~2 min" claim is false.** 64 `#[ignore]`s, none with a reason string; ~50 in botbowl-mcts are manual probes (`tree_shape` ×9, `mirror_search_exact` ×9, `expand_bench` ×9, `root_visit_anomaly` ×6), two need gitignored models, one needs a 14x7 build. Add `#[ignore = "..."]` reasons, move probes/benches to `examples/` or `benches/`, and fix the root doc. | M | high |
| 10 | **Tests that silently pass by skipping.** `skip_if_board_smaller_than!` (10 uses), `open_state() -> None` returning empty Vecs, and `parallel_rungs`, `sprt_stop`, `eval_job` ×2, `capacity_parity`, `remote.rs` ×4 all print "skipped" and pass under the default 26x15 build. Green means less than it looks. Either build the small board at runtime via `with_board_dims` (engine supports it) or make them `#[ignore = "needs 16x9 build"]`. | S–M | high |
| 11 | **`BOARD_SIZE_*` does two jobs:** build-time capacity (`build.rs`) and runtime active board (`BoardDims::from_env`, called from every `GameStateBuilder::build()`). Changing the env rebuilds the world and both variants overwrite each other in the same `target/`; scripts work around it with `CARGO_TARGET_DIR=target/${W}x${H}` by convention, and root `CLAUDE.md`'s hub/worker commands don't mention it. 11 of 230 engine tests fail at 14x7/5 (engine doc says 1). Decouple the runtime board from the process env (explicit parameter, default = capacity); encode the board into the target dir automatically (xtask/wrapper). | M | high |
| 12 | **`train_loop.sh` is 1310 lines of bash with 88 env knobs, defaults from a run that no longer exists** (14x7 anchor, full games, no mc-label), only `set -u`, a "DO NOT EDIT WHILE RUNNING" hazard (bash reads by byte offset), JSON parsed with grep/sed, and the gen-0 bootstrap duplicating the main phases. Short term: promote plan-058 values to defaults, wrap in `main "$@"; exit`, generate a knob table. Long term: a Python driver or `botbowl-hub loop` with typed config. | S now / L later | high |
| 13 | **`launch_plan0NN.sh` files are 90% identical** (056 vs 058 differ in a header and 8 values). Replace with a run-config file (`runs.d/plan058.env` or TOML) plus one generic launcher. Also fix: `exec systemd-inhibit` is Linux-only and double-inhibits; the launcher depends on `runs/loopmix16x9g/positions` from a previous run (freeze into `cfgs/positions/`); hub port is 7777 in docs, 13337 in every launcher. | M | high |
| 14 | **`recon_mcts` tests are invisible from the root** (`cargo test --workspace` never runs them), its fmt rule lives only in `.cursor/rules`, it is edition 2018 vs 2021, rand 0.9 vs 0.8, with a second lockfile and a second 721 MB `target/`. Add `scripts/check_all.sh` (or an xtask) that runs fmt/clippy/test for both workspaces; retire the `.cursor` rule in favour of `cargo fmt --all` from root (which already covers path deps). | S | high |
| 15 | **No CI, no hooks, 76 clippy warning lines, no `clippy.toml`/`deny.toml`/`.cargo/config.toml`, no `rust-toolchain.toml`.** Add a `.claude/settings.json` (shared, not `.local`) with a post-edit `cargo fmt` hook, and a `[workspace.lints]` block so clippy is enforced. Fix the existing warnings with `cargo clippy --fix` (mostly `useless_conversion`, `large_enum_variant`, `ptr_arg`). | S | med–high |
| 16 | **Provenance is stringly typed.** `meta.extra` is `BTreeMap<String,String>` with ~40 literal keys written in one crate and parsed back by literal in ui, worker, tests and 8 Python scripts; `RandomStartBias` goes out key by key and comes back key by key (with a `temperature2 = temperature` bug in `override_audit.rs:312`); rung labels like `vs:mcts(nn:/abs/path.onnx) [puct=raw(c=10)]@14x7/4` are built by `format!` in two crates and parsed in `page.rs:458`, `eval_summary.py` and `anchor_curve.py`; `td_rate.py` regexes raw JSON bytes. Give `extra` a typed struct (serde, `#[serde(flatten)]` for the long tail) and a structured rung identity alongside the label. | M–L | high |
| 17 | **God files are mostly inline tests.** `block_procs.rs` 2757 lines (1730 tests), `gamestate.rs` 2423 (670), `kickoff_procs.rs` 1437 (716, ~130 commented out), `dynamics.rs` 4609 (~1030), `roll_outcomes.rs` 1495 (890), `tree.rs` 3022. Move tests to `<mod>/tests.rs` first (mechanical, zero risk), then split by the seams listed in Part 2. | M per file | high |
| 18 | **`GameState` keeps three field lists in sync by hand:** `derive(PartialEq)` via the unmaintained `derivative` crate (RUSTSEC-2024-0388), a hand-written `Clone`, a hand-written `Hash`. Only `Clone` fails the build when a field is added; `Hash`/`PartialEq` drift is exactly the invariant root `CLAUDE.md` calls a correctness bug. Derive both by destructuring (`let Self { a, b, .. } = self;` so a new field is a compile error) and drop `derivative`. | S–M | high |
| 19 | **The hub/worker `PROTOCOL_VERSION` bumps for edits in other crates** (16 bumps in 3 weeks; hub-proto re-exports `MctsConfig`, `SearchTelemetry`, `EvalGameLine` verbatim and postcard encodes positionally, so any mcts/play struct edit silently changes the wire). The handshake already requires the same commit, so the number adds bookkeeping without safety. Either drop the version in favour of the commit hash, or give the wire its own DTOs. Also: `botbowl-hub-proto` depends on `botbowl-play` and so pulls 175 crates including tract; `botbowl-hub` depends on `botbowl-web-server` so hub tests compile the web server (feature-gate the `/play/` mount). | M | med–high |
| 20 | **CLAUDE.md files total ~160 KB** (mcts 29.7 K, web 24.8 K, nn 22.6 K, hub 20 K, root 13.5 K) and are heavy on history ("used to…", "before 2026-09-29…"); several lines exceed 1000 chars. Every mcts edit pays ~43 KB of context. Keep invariants, knobs, file map and test matrix; move history to the plans. Add to root: per-board target dirs, a per-crate "what to run after touching X" table with timings. | M | med–high |

---

## Part 2 — Full list by area

### A. Build, workspace, dev loop

- A2 No `[profile.*]` in root `Cargo.toml` → top-20 #7. S / high
- A3 No `[workspace.dependencies]`/`[workspace.package]`; 23 crates at multiple versions: itertools ×4, syn ×3, hashbrown ×3, getrandom ×3, rand 0.8/0.9, **tokio-tungstenite 0.24 (worker) + 0.29 (axum)**, toml 0.8/1.1, thiserror 1/2, mio 0.8 via stale crossterm 0.27/ratatui 0.25. Bumping the worker's tungstenite drops a whole websocket stack. S–M / med
- A4 `botbowl-web/client` (leptos, wasm) is a workspace member with no `default-members`, so every host `cargo check/test --workspace` compiles leptos. Add `default-members` or cfg-gate. S / med
- A5 Nothing checks `--target wasm32-unknown-unknown`; `trunk build` is never automated; the hub serves whatever stale `dist/` is on disk; no client↔server proto version/hash in the first `ServerMsg`, so a stale build fails silently on serde mismatch. S / med–high
- A6 `botbowl-mcts` enables `recon_mcts/deterministic_hash` for dev-deps only, so `cargo test` and `cargo build` produce two artifact sets and rebuild on every switch; tests never exercise the production `RandomState` nondeterminism (memory: "seeds don't pin games"). M / med
- A7 `botbowl-nn` lists `clap` as a normal dep for the `prepare` bin only; every lib consumer builds clap derive. Also the bin is just called `prepare`. S / low
- A8 `botbowl-engine` has unused deps `ansi_term`, `arraystring`, `json-patch`; `color-backtrace` should be a dev-dep (it's installed as a global panic hook inside tests at `bots.rs:116`, `game_procs.rs:1079`, which is the source of the noisy "application panicked" output in `should_panic` tests). S / low
- A9 No `rust-toolchain.toml`; recon_mcts is edition 2018. S / low
- A10 Dead recon_mcts features: `stable` (no-op), `nightly` (gates GATs stable since 1.65), `two_player` (gates 123 lines + 150 commented-out lines). S / low
- A11 Disk: `target/` 44 G (debug 37 G + per-board dirs), `data/` 24 G of pre-v9 corpora the code rejects, `models/` holds v6/v7 nets at top level, git has 6039 loose objects (182 MiB) vs 2.4 MiB packed → `git gc`. Add `.ignore`/`.rgignore` for `data/ runs/ models/ target/` so agents' searches skip them. S / med
- A12 Crate layout is inconsistent: `botbowl-hub-proto/` flat vs `botbowl-web/proto`; hub/worker flat, web nested. M / low
- A13 No justfile/Makefile/xtask; 73 shell scripts are the only launcher layer. Covered by #5, #13, #14.

### B. Configuration surface

- B1 Env vars read by Rust: 34 production + 9 test-only, three prefixes (`BLOOD_`, `BOTBOWL_`, `BB_`), three golden-bless names (`BLESS`, `BLOOD_WRITE_GOLDEN`, `INSTA_UPDATE`). Undocumented anywhere: `BLOOD_MCTS_PUCT_C`, `_PUCT_RANGE_FLOOR`, `_GUMBEL_SCALE`, `_GUMBEL_Q_FLOOR`, `_TRACE_ROOT`, `BOTBOWL_HUB_TOKEN`, `BOTBOWL_BENCH_TRIALS`, `BLOOD_WRITE_GOLDEN`, `BB_MIRROR_*`, `BB_WALK_N`, `PARALLEL_*`, `PROBE_GAMES`. Dead: `BLOOD_MCTS_BACKUP`. No clap `#[arg(env)]` anywhere, so CLI and env are unlinked. Generate one table; warn at startup on unknown `BLOOD_*` (the worker already does, `warn_about_stale_env`). M / high
- B2 Shell scripts: 167 distinct `${VAR:-}` knobs across 73 files; `train_loop.sh` 88; `DRIVE_EVAL_EVERY` (the current driver) is in no doc; `TIER`/`RUN_DIR` are silently ignored in mixed mode in favour of `TIER_OVERRIDE`/`RUN_DIR_OVERRIDE`; `NN_SERVER_RESTARTS` is never read; `HEUR_SHARDS` is a dead path threaded through 6 sites. S / med
- B3 `MctsConfig` plumbing (`dynamics.rs`): `BloodBowlDynamics` re-declares 6 config fields with its own `Default` (a new knob touches ~5 places: struct, `new`, `from_env`, builder, gd copy); each enum has a hand-written `from_env` string match duplicating serde names and silently defaulting on unknown values (while `_MEMORY` panics and TOML uses `deny_unknown_fields`); boolean parsing accepts different spellings per knob; defaults restated as literals in `from_env`; `horizon_turns` clamped in env and web but not TOML; `parse_puct` in play duplicates the env parser. Hold a `MctsConfig` (it's `Copy`), parse enums via serde, add `MctsConfig::validate()`. M / high
- B4 `cfgs/`: README covers 7 of 19 presets (the live `gumbel16_f*` family undocumented), has no key table; `gumbel16_f1000_mean`/`f4000_mean` are byte-identical to their siblings; `diag_*` ×3 are orphaned; `validation_pairs.toml` isn't a preset but sits with them and fails `--bot-config` with a parse error. Add a test that diffs `MctsConfig` field names against the README. S / med
- B5 Config files live in 7 places with no single doc: `~/.config/botbowl/{web.toml,teams/,hub.token}`, `~/.cache/botbowl/models`, `<repo>/hub-allowed-commits.toml`, `data/web-games`, `models/`. Cache dir uses `$HOME/.cache` while token honours `XDG_CONFIG_HOME`. S / med
- B6 **Stray `hub.token` at repo root**, mode 644, ignored only via `.git/info/exclude` (not shared with clones/worktrees), differs from `~/.config/botbowl/hub.token` (also 644), and the running worker uses a third copy at `~/hub.token`. Delete, add to `.gitignore`, `chmod 600`. S / med
- B7 Value scale "1 TD = 1000" is a magic number in 7 places (`score.rs:21`, `dynamics.rs` ×4, `report.rs:35`, `botbowl-nn/src/eval.rs:431`); PUCT_C, virtual loss and `gumbel_q_floor` are all calibrated against it. One shared const. S / med
- B8 `LEAF_STATS` and `ENGINE_APPLY_ACTIONS` are process-global statics (`dynamics.rs:965, 988`), so the web server's several bots mix tallies; `LEAF_STATS` is re-exported publicly. M / low

### C. Engine

- C1 `GameState` Hash/Eq/Clone by hand + `derivative` → top-20 #18.
- C2 `GameStateBuilder::build()` plays a coin toss, two auto-setups, a seeded kickoff and two EndTurns for every test (`gamestate.rs:238-330`), and turns logging on; any setup/kickoff change breaks hundreds of unrelated tests. Construct turn state directly. M / high
- C3 Builder can't add a player with skills, so tests call `give_skill` after `build()` when `available_actions` is already computed (stale-actions hazard, e.g. `block_procs.rs:~1876`); `build()` swallows fielding errors with `_ =`. Add `add_home_player_with(pos, &[Skill])`. S / med
- C4 Splits: `gamestate.rs` → `builder.rs`, `dice_mode.rs`, `passing.rs` (its own TODO at :1699 says so), `mirror.rs`; `model.rs` (1330 lines, geometry + players + procedure protocol + actions) → `geometry.rs`, `player.rs`, `procedure.rs`, `actions.rs`; `kickoff_procs.rs` `Formation` planner (lines 189-535) → `formation.rs`; `pathing.rs` has its own `struct GameInfo<'a>` (:421) shadowing `gamestate::GameInfo`, plus generic `FixedQueue`/`CustomIntoIter` that belong elsewhere. M / med–high
- C5 Accessor conventions: `get_player` (Result) / `get_player_at` (Option) / `get_player_unsafe` (panics, 283 call sites, nothing to do with `unsafe`); `get_mut_team` vs `get_dugout_player_mut`; `step()` returns `Result` but always `Ok` and panics via `assert!` on illegal actions in release (`gamestate.rs:1260`); 108 `pub fn` on `GameState`, 324 `pub fn` crate-wide and 8 `pub(crate)` so dead-code lint sees nothing. Public typos: `get_line_of_scrimage_x`, `AnyAT::Postional`, `vicitm_id`. M / med
- C6 Procedure boilerplate: 44 `pub fn new() -> AnyProc` force a crate-wide `allow(new_ret_no_self)` and `new_pure` twins; `ProcState` has 9 variants for Done/NotDone × New/NewProcs; UseSkill/DontUseSkill prompt built 4×, UseReroll 3×, three separate `fn ask`; 15 copies of `_ => panic!("Unexpected input")`. Have `any_proc!` emit `From<T>`, collapse `ProcState` to `{keep, push}` + wait variants, add `AvailableActions::yes_no`. M / med
- C7 Callers match procedures by name string (`proc_stack_top() == Some("Block")`) in ~23 files across crates; a rename compiles and silently breaks them. `ProcKind` enum. M / med
- C8 Adding a skill means listing it 3× in `table.rs` (enum, `Skill::ALL`, `index()`); declaration order already disagrees with `ALL` (`StripBall`). One macro list. S / med
- C9 `AvailableActions.simple` is a `HashSet<SimpleAT>` (17 variants): nondeterministic iteration forces a sort, slow to clone/hash. Bitset like `SkillSet`. S / med
- C10 Three `is_out` with different meanings (`Position::is_out` is capacity and is used in gameplay at `gamestate.rs:1682`). Rename capacity versions. S / med
- C11 Errors are a `Box<dyn Error>` alias over 5 ad-hoc structs; 35 `unwrap`s in non-test `block_procs.rs`, 17 in `gamestate.rs`. One `enum EngineError`. M / med
- C12 `KickoffTable` silently no-ops 8 of 11 events (only empty comment bodies say so); `todo!()` on a live path at `ball_procs.rs:431`; GameOver offers a fake `DontUseReroll` sentinel (`game_procs.rs:288`) that leaks into encoders/bots. S / med
- C13 Dead/stale: ~130 lines of commented-out tests at `kickoff_procs.rs:1309-1437`; unused `ActionChoice`, `MissingActionError`, `EmptyProcStackError`; `SkillSet::hash_unordered` / `hash_set_unordered` leftovers from the HashSet era; `standard_state()` in `lib.rs:10` with its own "shouldn't be here" TODO; 13 TODOs; stale comments at `build.rs:3`, `model.rs:110`, `gamestate.rs:449, 585, 1218-1233`, `table.rs:228`; `dices.rs` (740 lines) has no test module; `ball_carrier_casualty` has no assertion. S / low–med
- C14 `botbowl-engine/CLAUDE.md` drift: "2-cell OOB border" (it's 1 per side), "H must be odd" (build.rs says no parity rule), "14x7 fails one test" (11 fail), `ProcState` list omits 4 variants. S / med
- C15 Every test module repeats a 6-10 line import block; ~300 `use botbowl_engine::…` downstream. A `prelude`. S / med
- C16 Engine rule changes ripple: since 2026-09-01, 70 engine commits, 20 re-blessed `lazy_mover_goldens.txt`, 22 touched `rules/README.md`. Consider whether the golden belongs in the engine crate so one commit carries both. S / med

### D. MCTS and recon_mcts

- D1 `dynamics.rs` split (4609 lines): `config.rs` (enums, `MctsConfig`, `from_env`), `gd.rs` (`GameDynamics` impl; `select_node` ~250 lines, `backprop_scores` ~180), `puct.rs`, `leaf_stats.rs`, `bot.rs` (`MctsBot`, builders, `run_search`), `inspect.rs`, `run_gumbel` → existing `gumbel.rs`, tests → `dynamics/tests.rs`. L / high
- D2 `run_search` (~440 lines) interleaves reuse lookup, telemetry baselines, DAG walk, depth histograms, `LEAF_STATS` dumps and worker spawning inside a local `macro_rules! run_with_marker`, which defeats navigation and step-debugging. Extract `fn search_on<M: StateMemory>()` + a `dump_diagnostics()` observer. M / high
- D3 `Bot::get_action` re-implements `get_action_explore`'s pick-best instead of delegating. S / med
- D4 `canonical_chance_key` allocates a `String` per child on every chance-node backprop (hot path, 6^n block outcomes); an `Ord` tuple would do. Relevant to the throughput focus. S / med
- D5 `roll_outcomes.rs`: code is lines 1-607, tests 608-1495; split `block_outcomes`/`BlockContext` into `roll_outcomes/block.rs`. M / med
- D6 `recon_mcts/src/tree.rs` (3022 lines): `state_memory`, `branch_wip`, `ArcWrap`/`WeakWrap`, stats types, `Node`, `Tree` are the seams. Dead: `Children::BranchWip` is never constructed yet matched in 6 places and the whole `branch_wip` module (520-650); `apply_atomic`, `find_parents_sorted` unused; commented-out code at :997, :1078. M / med
- D7 `recon_mcts` still publicly exports `HashOnly` (`tree.rs:462`, prelude); the "never reintroduce" rule is convention only. `#[deprecated]` or feature-gate. S / med
- D8 Per-search state on `BloodBowlDynamics` (`root_noise`, `forced_root`, `root_trace`, `horizon`, `opponent_setup`) is argued pure only in comments; a 2026-10-03 bug of exactly this kind is noted at `dynamics.rs:2222`. Add a "dynamics constant across reuse" test. M / med
- D9 Recombination stats triplicated: `RegistryInfo` atomics, `recon_mcts::RecombinationStats`, `telemetry::RecombinationCounts`. S / low
- D10 Test hygiene: `open_state()` byte-identical in 3 files, `bot()` in 5, tiny-ONNX path built 15× across 7 files, `tests/common/mod.rs` used by 12 of 30 with `#![allow(dead_code)]`; `TEST_HOME_FRAME`/`TEST_AWAY_FRAME` defined twice and unused; `assert!(true)` at `dynamics.rs:4168`; unused `Result` in `tests/gumbel_root.rs`. S / med
- D11 Stale TODO at `dynamics.rs:1030` lists pruning, cached priors, scripted block dice and scripted chance outcomes as "left to do"; all exist. Misleads agents. S / med
- D12 Docstrings enumerate accepted values and are stale: `TieBreak::from_env` omits `mover`, `ChanceModel::from_env` omits the diagnostic variants. S / low
- D13 `recon_mcts` test layout contradicts its own CLAUDE.md ("don't move tests back into root crate"): six root `tests/*.rs` exist; `tests/nim/main.rs` is auto-discovered as a 0-test integration binary; "nim" holds 2048. S / low
- D14 `#[allow]`: file-wide `type_complexity` on `tree.rs`, file-wide `missing_docs` on `game_dynamics.rs` (defeats the lib-level warn), 4 stacked in `tests/nim/lib.rs`. S / low
- D15 Unmeasured consts: `NORM_VL_REFERENCE = 300` ("no measurement behind it"), `C_VISIT`/`C_SCALE` in `gumbel.rs` not preset-reachable while `gumbel_*` are. S / low

### E. Play, UI, hub, worker, nn, data

- E1 Flag-struct duplication ui↔hub → top-20 #8.
- E2 `botbowl-play` violates its "process-agnostic" contract: reads files (`bots.rs:119`, `drives.rs:70`), writes a file (`trace.rs:84`), reads env (`MctsConfig::from_env` at `bots.rs:73`, `BudgetMode::from_env` at `:254` and `generate.rs:284`). Hidden env reads are exactly what made workers diverge; `SearchConfig::pinned_to_env` exists only to paper over it. M / med
- E3 `generate.rs:448-458` duplicates `drives::position_state` including the odd-seed `temperature2` rule; `random_start_trajectory` should call it (corpus/benchmark equivalence). S / med
- E4 `EvalGameLine::serialize` is hand-written because one impl serves pinned JSON and positional postcard; its doc says "update the field count by hand". Separate wire DTO. M / med
- E5 NN schema version defined 3× with nothing tying them: `botbowl-nn/src/bin/prepare.rs:85` (in the bin, not the lib), `model.py:23`, shape consts `model.py:15-17` (`SPATIAL_CHANNELS=61` etc. vs Rust consts). A v9 bump touched ~8-10 files. The ONNX carries no schema version and `NnEvaluator::from_path` checks nothing (v8→v9 policy-width change can load silently). `botbowl-data::FORMAT_VERSION = 1` has never moved while `meta.extra` absorbs every change. Export the Rust consts to a generated Python/JSON fixture and assert in pytest; embed schema in ONNX metadata and check on load. M / med–high
- E6 `nn_server.py` wire constants (`MAGIC`, `PROTOCOL_VERSION`, status codes, struct formats) duplicated against `botbowl-nn/src/remote.rs`; caught only by the launch handshake. Same fix as E5. S / med
- E7 `botbowl-ui` has 12 subcommands mixing TUI, pipeline stages (`mc-label` is a loop stage) and research probes, all in binary-private modules so hub/worker can't reuse them and integration tests can't reach them. Move pipeline stages into `botbowl-play`. M / med
- E8 Zero tests on CLI→request glue (`botbowl-hub/src/main.rs` 1144 lines, no `#[cfg(test)]`; `eval.rs`, `cli.rs`, `dataset.rs`). The #8 panic lives exactly there. S–M / med
- E9 Errors are `Result<_, String>` in 26 signatures across ui/play/hub, no anyhow/thiserror; `process::exit` ×11 in hub main. Logging is ad-hoc `eprintln!("[hub] …")` (hub 39, ui 31, worker 12) and scripts grep exact stdout markers (`NN_PROFILE`, `NN_SERVER_FALLBACK`), so log text is an API. M / med
- E10 Duplicates: progress line in `dataset.rs:132` and `worker/src/lib.rs:500`; free-RAM probing in `page.rs:120` (Linux-only) and `worker/src/lib.rs:810`; `RandomStartBias` defaults written out in ui `default_value_t`, play `Default`, and hub `Option` (three conventions); hub `PathBuf` absolutised vs ui `Option<String>` relative, so the same run gets a different rung label depending on which binary played it. S / med
- E11 Hub status page parses bash-written `status.md` prose and trainer progress files (`page.rs:82, 304`): a cross-language contract with no schema. M / low–med
- E13 `#[allow(clippy::too_many_arguments)]` ×6 in ui/play signal missing config structs. S / low

### F. Scripts and Python

- F1 `exp*.sh` archive + `scripts/README.md` → top-20 #5. F2 `train_loop.sh` → #12. F3 launchers → #13.
- F4 Boilerplate copy-pasted across `exp0[4-6]x_*.sh`: `BOARD_SIZE`/`CARGO_TARGET_DIR` exports (51 files), `train/.venv/bin/python` (49, instead of `uv run`), sidecar launch + socket wait loop (38), `cargo build --release` (25), `job eval --wait` (27), `status/die/down` traps (39), the inline Python heredoc summarising `report.json["ladder"]` with mean ± SE (12 files, 6 byte-identical). `exp032_lib.sh` is the only shared lib and is frozen at 14x7/cuda/`runs/loop14x7`. Extend to `scripts/lib/hub.sh` (`start_hub`, `start_sidecar`, `start_worker`, `match`) + `summary.py ladder`. M / high
- F5 `/tmp/bbnn-*.sock` hardcoded in 48 scripts; `--device cuda` in 43; `nn_server.py:1350` accepts only `cpu|cuda` (no `auto`/`mps`, unlike `bbnn.train`), so nothing runs on the macOS box. S / low
- F6 `nn_server.py` (1476 lines) is production infra living in `scripts/`, imported via `sys.path.insert` (`:79`, and `train/tests/test_nn_server.py:25`). Move to `bbnn/server/{protocol,runners,server}.py` (clean seams at `Registry`, `GraphRunner`/`EagerRunner`/`RunnerPool`, `Connection`/`Server`; bench/loadgen tools at `:1188-1336`) with a `[project.scripts]` entry. M / med
- F7 Six scripts use `sys.path.insert` hacks; `audit_value_head_bias.py:42` uses cwd-relative `"train/src"` and breaks outside repo root. S / med
- F8 31 `scripts/*.py`, none tested, including ones the loop's status depends on: `td_rate.py`, `size_curriculum.py` (moves the curriculum centre), `eval_summary.py`, `drive_curve.py`, `anchor_curve.py`, `absorb_probe.py`. M / med
- F9 Duplicated analysis: pair/pentanomial scoring in `paired_summary.py`, `pair_correlation.py`, `eval_summary.py`, `anchor_curve.py`, `side_bias_pooled.py` while SPRT lives in Rust (`validate_proxy.py:52` re-implements the LLR from `stats.rs:155`); ~20 scripts have their own `json.loads` loop, 8 their own `load*`, 10 hand-roll `sys.argv`; defaults point at long-gone `runs/loop14x7`. A `bbtools` module with `read_jsonl`, `load_report`, `pair_score`, `se`. M / med
- F10 Python tooling: no `.python-version` (requires ≥3.10, venv is 3.12), no ruff config; `ruff --isolated` finds 10 hits incl. a probably-real B023 closure-capture at `audit_corpus_stats.py:398`; 13 scripts have a shebang without the exec bit. S / low
- F11 Shellcheck (not installed; ran via `uvx shellcheck-py`): 51 warnings + 62 notes across `scripts/*.sh`, 77 `# shellcheck disable` lines; SC2054 (comma in array) at `exp056_tau_budget_drives.sh:81`, `exp063_plan054.sh:109` is probably a real bug; 12 scripts set no `set -euo` at all, `train_loop.sh` only `-u`. Add shellcheck to `check_all.sh`. S / med
- F12 Small bugs: `train_loop.sh:1029` prints `parent` when `DRIVE_REF=parent` (should use `drive_ref_of`); `[ -f "$ANCHOR" ] || die` runs even in drives-only mode so launchers must ship an anchor they never play, and fake `.mirror.done` because there's no `MIRROR_GAMES=0` path; `launch_plan058.sh:33` exports `BOARD_SIZE_*`/`CARGO_TARGET_DIR` that `train_loop.sh` immediately overwrites. S / low–med
- F13 `PROFILING.md` doesn't mention `perf_search_bench.sh`, `perf_gen_bench.sh`, `nn_throughput_probe.sh`, `BLOOD_NN_PROFILE`; `tools/` holds one file (`samply_flatten.py`). Fold into `scripts/` or move perf scripts to `tools/`. S / med

### G. Web

- G1 Web `MctsSpec` is a third hand-copied knob subset; cannot play the loop's `gumbel16_f1000` bot; `config_label` omits gumbel/setup/backup so two inspector bots can't be told apart; `MctsSpec` defaults are hand-copied from `MctsConfig::new()` with no test pinning them. Load `cfgs/*.toml` by name. M / high
- G2 No client↔server version check; `trunk build` never automated; wasm target never checked → A5.
- G3 Client has 0 tests (`state.rs` 352 lines, `inspector.rs` merge/`favours` logic are host-testable); the headless-Chrome CDP driver exists only as prose in a memory file (which also says "no browser tool", no longer true). Check in `tools/web_drive.mjs`. M / med
- G4 Large components: `pitch.rs` 1030 lines (`square()` ~118, `Debug` 99), `session.rs` 1138 (`run` ~125), `SearchDetail` 158, `Explorer` 128, `DecisionLog` 123. M / med
- G6 Rosters are stringly typed (`team.rs:166` takes `&[&str]` skills; comment at `:209` kept in sync by hand). S / low

### H. Docs, plans, agent context

- H2 Root `CLAUDE.md` Plans section (`:20-34`) is a changelog; per-plan numbers and schema details belong in the plans. Keep: current focus, live run, standing rules, index pointer. Add: `plans/STATUS.md` or `plans/README.md` with numbering/Status/idea-vs-plan conventions. M / high
- H3 Finished plans still in `plans/`: 017, 031 (probably), 035, 038, 039, 040, 043, 044, 045, 047-wide-window, 048, 050, 055. Number collisions: three 036s, two 047s, two 010s and two 019s in `completed/`, 017/017b; `exp0NN` script numbers are a separate series overlapping plan numbers (`exp056` ≠ plan 056). Status lines stale (`047-per-player-setup` says "built, not yet run"; `042` says "not started"; 001, 002, 036-e4 have none). `completed/` suffix used inconsistently. M / high
- H4 Dead references in plans: `041:262` → `scripts/dist_build.sh` (missing), `052:79` → `scripts/exp_contest_prior.py` (missing), completed 006/008/009/010 → `plans/005-…` (moved), 011/012 → `plans/011-baseline-results.md` (missing). S / low
- H5 `032-plan--ranked-experiment-queue.md` is 1363 lines, `031` is 883: append-only logs too big to read for one task. Split into open-queue + archive. M / med
- H6 Missing context agents need every session (live run, `runs/` location, worktree/target rule and file-lock note are now in root `CLAUDE.md`): per-board target dirs; a per-crate test matrix ("touched engine → `cargo test -p botbowl-engine` 230 tests 1.7 s + `-p botbowl-mcts --test hash_quality` 17 s + re-bless lazy_mover; touched nn → `uv run pytest` in `train/` + parity fixtures; touched web → wasm check + trunk"); timings (no-op check 2 s, engine-touch check 24 s, full test ~2 min of test time plus compile, clippy 63 s). S / high
- H7 Agent config is local-only: only `.claude/settings.local.json` (allows `git checkout *` broadly, which likely explains the mid-review branch switch; `skillOverrides: commit off`); no shared `.claude/settings.json`, no hooks; `recon_mcts/.cursor/rules` is Cursor-only. S / med
- H8 Memory notes stale: three "unmerged branch" entries for merged work; `tests-under-14x7-env` says 3 failures vs engine doc's 1 vs measured 11. S / med

---

## Part 3 — Suggested sequencing

1. **One afternoon of S items that remove daily friction:** #5 (archive exp scripts), #7 (profiles + workspace deps), B6 (token), A11 (`git gc`, `.ignore`), #14 (`check_all.sh`), #15 (fmt hook, clippy fix). None touches game logic; do it between generations.
2. **Config consolidation (#6, B1, B3, B4, G1):** presets as the one source of truth, env/CLI as overrides, generated knob table, web loads presets. This is the item most likely to prevent the next "why do these two runs differ" hunt.
3. **Test honesty (#9, #10, D10, C2, C3):** ignore reasons, no silent skips, cheaper builder, shared test helpers. Makes green mean green and the suite faster.
4. **Mechanical splits (#17, D1, D2, D5, D6, C4):** tests out of god files first, then module seams. Zero behaviour change, big win for agents' context budgets.
5. **Typed boundaries (#8, #16, #18, #19, E4, E5):** shared CLI arg structs, typed `meta.extra`, derived `GameState` Hash/Eq, wire DTOs, schema consts exported to Python.
6. **Long tail:** `train_loop.sh` rewrite (#12 L), launcher config files (#13), engine API cleanups (C5–C11), Python `bbtools`/`bbnn.server` (F6, F9).
