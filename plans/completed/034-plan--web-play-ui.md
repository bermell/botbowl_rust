# Web play UI: human vs bot in the browser, with search-tree overlays

**Status:** **Phases 0-3 shipped 2026-09-11.** Playable end to end: a full game against any bot
this workspace can build, on any board the binary's capacity allows, with dice, undo, and an
inspector over the MCTS bot's search. Phase 4 (engine and search on wasm32) is untouched and stays
a future idea — the design below is the record; the implementation log at the bottom is what
actually happened, including three divergences from it. POC scope only: local machine, one player,
never exposed to the internet. Supersedes the "easy for a human to play against an agent" line of
`001-grand-plan.md` §1 (the terminal UI never grew an input path and won't).

```sh
cd botbowl-web/client && trunk build --release
cargo run --release -p botbowl-web-server -- \
    --assets-dir /Users/mattias/repos/blood/botbowl/botbowl/web/static/img
```

## Goal

Play a full game against any bot the workspace can build (`Random`, `Scripted`, `MctsBot` with
heuristic or NN evaluator, any knob combination), see dice as they happen, and after each bot
decision inspect *why*: root action distribution as a heatmap on the pitch, the ranked candidate
list, the principal variation, and a click-to-expand view into the search DAG.

Not goals: auth, multi-user, persistence beyond the existing `Recording` file, pretty lobby,
mobile, production hardening.

## Decisions (settled 2026-09-10)

1. **New workspace member `botbowl-web/` holding three crates** (`proto/`, `server/`, `client/`).
   `proto` = serde message types, no engine dependency, compiles on host and wasm32.
   `server` = axum + tokio + tungstenite websocket, owns the `GameState` and the bots.
   `client` = Rust → wasm via **Leptos (CSR mode) + trunk**. Leptos over Yew/Dioxus for the
   fine-grained signals (per-square reactivity on a 26x15 grid with heat values is exactly the
   case where VDOM diffing hurts) and for being the most actively maintained. Plain
   `wasm-bindgen + web-sys` is the fallback if the framework fights us; the proto boundary means
   the choice is swappable.
2. **Bot logic server-side for the POC.** `MctsBot::get_action` runs under `std::thread::scope`
   and `recon_mcts` is inherently multi-threaded (`RwLock`/`Condvar` work stealing, cycle guard
   writes a temp file). None of that runs on wasm32. Client-side search is phase 4, speculative.
3. **Client does not depend on the engine in phases 1–3.** The server ships a fully derived
   `ViewState` (every square already annotated with legal actions, move success probability,
   block dice). This keeps the wasm build trivial and puts all game logic in one testable
   pure function `GameState -> ViewState`. Phase 4 revisits this.
4. **Server drives the engine in `DiceMode::RegisterRolls`**, rolls with its own `ChaCha8Rng` via
   `dices::resolve_with_rng`, and feeds results back through `step_with_roll_or_action`. That
   way the server *sees* every `RequestedRoll`/`RollResult` pair and streams them to the client
   as dice events (block dice faces, D6 dodge/pickup, KO/injury, kickoff). `RollDice` mode hides
   all of this inside the engine. Bonus: a "fix next roll" debug control falls out for free.
   Bots still get the normal `GameState` (they set their own dice mode internally).
5. **Pitch drawn in CSS/DOM, not the JPG backgrounds.** Backgrounds only exist for 4x3, 12x5,
   16x9, 20x9, 26x15, and our trained models are 14x7 and 8x3. A CSS grid with endzone/LOS/
   wide-zone colouring works for any `BoardDims`. Sprites and icons come from the old repo.
6. **Assets are served from the sibling checkout, not copied into this repo.** The player icons
   are explicitly not under the botbowl license (`botbowl/README.md:61`). The server takes
   `--assets-dir /Users/mattias/repos/blood/botbowl/botbowl/web/static/img` and mounts it at
   `/img/`. Nothing binary enters git.
7. **Transport: one websocket per game, JSON messages.** Full `ViewState` on every change (a few
   KB; no deltas for a POC). REST only for `/new-game` and asset/static serving.
8. **One server binary, board size chosen per game in the lobby.** Build-time env only sets
   *capacity*; `GameStateBuilder::with_board_dims` picks the runtime `BoardDims` (must be ≤
   capacity) and the NN encoder already uses runtime dims (`botbowl-nn/src/encode.rs:144`). So
   build the server at the default 26x15 capacity and let `GameSpec` carry `dims`. 14x7 is the
   first target (trained models), but the client is size-agnostic from day one and the lobby
   lists models whose `_WxH_` filename tag matches the chosen dims. Verify in phase 0 that a
   14x7 model gives identical outputs under a 26x15-capacity build vs a 14x7 build (the
   `botbowl-nn/tests/parity.rs` harness is the template).
9. **`RegisterRolls` is the contract.** If a full game through `step_with_roll_or_action` with
   external rolling hits engine asserts, that is an engine bug to fix (TDD: failing test first),
   not something to route around with log parsing.
10. **Undo is server-side only:** a stack of `GameState` clones plus the server rng, pushed at
    each human decision point. No engine changes. Unlimited.
11. **No hint mode.** The inspector explains the bot's moves; it does not advise the human.

## Architecture

```
browser (wasm, Leptos)  <--ws JSON-->  botbowl-web-server (axum)  --in-proc-->  engine + bots
        |                                     |
   /  /img/*  (static, from --assets-dir)     |-- GameSession { GameState, rng, human: TeamType,
   /  /       (trunk dist/)                   |                 bot: Box<dyn Bot>, last_search: Option<SearchReport>,
                                              |                 history: Vec<GameState> (undo), recording }
```

### proto (shared types)

- `ClientMsg`: `NewGame(GameSpec)`, `Act(Action)`, `Undo`, `ExpandNode{ path: Vec<Action> }`,
  `FixNextRoll(RollResult)`, `SaveRecording`.
  `GameSpec { dims: {w,h,team_size}, human: TeamType, bot: BotSpec, seed: Option<u64> }`.
  `BotSpec { kind: Random|Scripted|Mcts{ budget: Iters(n)|Millis(n), evaluator: Heuristic|Nn{model_path}, workers, backup, puct, fpu, horizon_turns, tree_reuse } }`.
- `ServerMsg`: `View(ViewState)`, `Dice(DiceEvent)`, `BotThinking`, `BotMoved{ action, report: SearchReport }`,
  `Node(NodeExpansion)`, `Error(String)`, `GameOver{...}`.
- `ViewState`: `dims`, `half/turn/score/rerolls/weather` from `GameInfo`/`TeamState`, `squares:
  Vec<SquareView>` (player sprite key + status flags + ball + tackle-zone count + `actions:
  Vec<PosAT>` + `move_prob: Option<f32>` + `block_dice: Option<i8>`), `dugouts` (reserves/KO/
  cas per team, from `get_dugout()`), `simple_actions: Vec<SimpleAT>`, `to_act: Option<TeamType>`,
  `proc: &str` (`proc_stack_top`), `active_player`, `log_tail: Vec<String>`.
- `SearchReport`: `root_visits`, `root_q`, `children: Vec<{ action, visits, q, prior, solved }>`,
  `pv: Vec<Action>`, `elapsed_ms`, `iterations`, `evaluator_value: Option<f32>` (NN v at root).
- `NodeExpansion { path, info: NodeSummary, children: Vec<...> }` for the tree explorer.

Engine `Action`, `PosAT`, `SimpleAT`, `Position`, `RequestedRoll`, `RollResult` already derive serde;
`proto` re-declares mirror types rather than depending on the engine so the client stays
engine-free. A round-trip test in `server` pins the mirrors to the real enums.

### server

- `GameSession` state machine per websocket: after every input, loop `step_with_roll_or_action`
  until `NeedAction` (emit `View`), rolling on `NeedRoll` (emit `Dice`) and stopping on `GameOver`.
- When `to_act == bot team`: `tokio::task::spawn_blocking` around `bot.get_action(&state)` (it
  uses its own scoped threads; fine on a blocking worker). Send `BotThinking` first so the UI can
  show a spinner and the bot's clock.
- `view::derive(&GameState) -> ViewState` in its own module, pure, unit-tested against
  `GameStateBuilder` positions. Reuses `get_all_actions()`, `get_paths()` (per-square success
  probability, same green-with-roll overlay the old JS did), `get_blockdices_from`, `get_tz_on`.
- Undo: push a `GameState` clone (plus the server rng) before each human action; `Undo` pops to
  the previous human decision point, including back across the bot's intervening turn. Unlimited.
  The opponent `MctsBot`'s `cached_tree` anchor will no longer match after an undo; it already
  discards on anchor mismatch, so no bot change is needed, just a wasted reuse.
- Recording: reuse `game_runner::Recording` so `botbowl-ui replay` can open web games.
- `--assets-dir`, `--dist-dir`, `--port` (bind `127.0.0.1` only).

### MctsBot changes (botbowl-mcts)

- `MctsBot::last_search_report() -> Option<SearchReport>`: built in `get_action` from
  `tree.get_next_move_info()` (visits/Q per root child), `prior_for`/`nn.priors` for priors,
  and a greedy-by-visits descent (or `find_path_to`) for the PV. `cached_tree` already keeps
  the tree alive between moves, so `ExpandNode{path}` can be answered later by walking children
  via `get_node_info()` without re-searching. Serialize a bounded subtree (depth ≤ 3, ≤ 12
  children per node) per request, never the whole DAG.
- `BotSpec -> MctsBot` construction without env vars. Today every knob is read from
  `BLOOD_MCTS_*` at `::new`/`get_action`; the builder methods (`with_workers`, `with_evaluator`,
  `with_backup`, `with_fpu_reduction`) cover only some. Add a `MctsConfig` struct with
  `from_env()` as the default so the CLI paths keep working and the web server can build several
  different bots in one process. Moderate refactor; do it in phase 2 rather than blocking phase 1.

### client (Leptos, wasm32)

Screens: **Lobby** (board dims, side, bot spec, seed, model file from `/models` listing filtered
to matching dims) → **Game**.

Game screen layout: pitch grid centre; dugouts left/right (`dugout-left.jpg` backgrounds are
optional); top bar half/turn/score/rerolls/proc name; right panel = action buttons (simple
actions incl. `EndTurn`, `EndSetup`, `SetupLine`, reroll yes/no, block-die pick), dice ticker,
log tail; bottom drawer = **bot inspector** (ranked candidates, PV as a clickable list, tree
explorer). Overlay toggles: legal moves, move probabilities, tackle zones, bot visit heat, bot
prior heat, NN value. Overlays are per-square data, so they are size-agnostic.

Input model, following the engine's position-based actions:
- Click a square with exactly one legal `PosAT` → send it. Several (e.g. `StartMove` vs
  `StartBlitz` on own player) → small radial/popup menu, using `icons/actions/*.gif`.
- Simple actions are buttons; block-dice selection shows `dice/*.png` faces.
- Everything the engine asks is exposed: coin toss (`Heads`/`Tails`), `Kick`/`Receive`, setup
  (`SetupLine` shortcut plus per-square `SelectPosition` for manual placement, `EndSetup`),
  kickoff aim, reroll prompts, block-die choice, push/follow-up squares. Nothing auto-resolved.
- Keyboard: `Enter`/`E` end turn, `Z` undo, `1-5` overlay toggles.

Sprites: engine has one race with four roles; map `Lineman/Blitzer/Thrower/Catcher` →
`iconssmall/h{lineman,blitzer,thrower,catcher}1{,b}{,an}.gif` (verified to exist: `b` = home
colour, `an` = has not acted yet). Status overlays from `player_status/{prone,stunned}.png`,
ball `ball/sball.gif`, carrier `decorations/holdball.png`, block dice previews
`decorations/block{1,2,3}d.gif` / `block2dagainst.gif`, injuries `bloodspots/*`. Squares are
30px to match the 28px sprites.

## Phases

**Phase 0 — spike (≤ half day).** `cargo install trunk`, `rustup target add wasm32-unknown-unknown`,
hello-world Leptos crate inside the workspace, confirm `cargo test --workspace` still passes with
the client crate as a member (host build of a CSR Leptos crate is fine; if it isn't, exclude the
client like `recon_mcts` and build it from its own directory). Confirm axum serves `trunk dist/`
plus `--assets-dir`. Also in phase 0: the headless full-game `RegisterRolls` test (decision 9) and the capacity-vs-
dims NN parity check (decision 8). Both are engine/nn tests, independent of the web crates, and
can start immediately.

**Phase 1 — playable.** proto + `view::derive` + session loop + Random/Scripted opponents +
dice events + coin/kick/setup/kickoff flow + undo. Exit criterion: play a full game vs
`ScriptedBot` on 14x7 and on 26x15 from one server binary without touching the terminal. Tests: `derive` unit tests; a server
integration test that opens a websocket, plays random legal actions to game over, and asserts
every `View` had ≥1 legal action while `to_act == human`.

**Phase 2 — MCTS opponent + inspector.** `MctsConfig`, `last_search_report`, `ExpandNode`,
heatmap/prior/PV overlays, lobby bot spec incl. NN model path. Exit criterion: play
vs `bbnet_14x7_gen0c` and see, for each bot move, the top-N candidates on the pitch and a PV I
can step through.

**Phase 3 — debugging conveniences.** `FixNextRoll`, load a `Recording`/seed/snapshot as the
start state, save recording, NN value readout per state, "step into PV": temporarily render a
hypothetical node's `GameState` (`NodeInfo.state` is `Some` under `StoreState`) on the pitch.

**Phase 4 — client-side engine and search (speculative, not POC).**
Engine on wasm32 needs: `getrandom` with `js` feature (four `from_entropy()` call sites),
`cfg`-gate `Recording::to_file/from_file`, drop `color-backtrace` and the `Instant` benchmark
helper behind a feature. That alone buys local action validation and instant legal-move
overlays. Search on wasm32 additionally needs a single-threaded `recon_mcts` step loop (no
`thread::scope`; a config flag isn't enough), `Instant` → `web-time`, the cycle-guard temp-file
path removed, and NN inference via `tract` compiled to wasm (tract claims wasm32 support;
unverified) or a Web Worker pool with `SharedArrayBuffer` for parallelism. Rough cost: a week,
mostly in `recon_mcts`. Worth it only once the server-side loop is boring.

## Risks / things to verify early

- **`RegisterRolls` for a whole game.** Only exercised over MCTS horizons so far. The phase 0
  headless test (random actions + external rolls to game over, both board sizes) will surface any
  asserts; those are engine bugs and get fixed in the engine with a failing test first.
- **Bot `get_action` on a `RegisterRolls` state.** The bots clone the state and set their own mode;
  confirm `ScriptedBot` and `RandomBot` don't assume `RollDice` on the passed reference.
- **Long searches block the session.** `spawn_blocking` keeps the websocket alive but the human
  can't undo mid-think; acceptable for a POC. Add a cancel later if it bites.
- **Env-var knobs are process-global.** Until `MctsConfig` lands, all bots in the server share
  the `BLOOD_MCTS_*` settings; fine for phase 1 (one bot per process).
- **Board-size mismatch** between chosen dims and model panics inside `NnEvaluator`; the lobby
  lists only models whose `_WxH_` filename tag matches the selected dims.
- **Capacity padding.** Everything runtime-dims-aware has been tested via the `with_board_dims`
  tests in `gamestate.rs`, but the web server is the first *long-running* consumer of a
  non-capacity board. Any `WIDTH`/`HEIGHT` constant leaking into gameplay code shows up as a
  wrong out-of-bounds edge on 14x7; the phase 0 full-game test on 14x7 under 26x15 capacity is
  the guard.
- **Workspace hygiene.** The client crate pulls `leptos`/`wasm-bindgen`/`web-sys` into the shared
  `Cargo.lock`; harmless, but `cargo test --workspace` compile time grows. Exclude if annoying.

## Resolved questions (2026-09-10)

- Framework: Leptos CSR.
- Assets: served from the sibling `botbowl` checkout path, nothing copied into git.
- Human controls every decision the engine surfaces, including coin toss, kick/receive, setup.
- Undo: unlimited, server-side clone stack, no engine involvement.
- 14x7 first; many board sizes eventually, hence one binary + runtime dims (decision 8).
- No hint mode.
- `RegisterRolls` is the contract; engine gets fixed if it breaks.

## Still open

- `MctsConfig` refactor scope: minimal (struct + `from_env` default, builder untouched) vs also
  removing the per-`get_action` env reads. Decide when phase 2 starts.
- Whether `SearchReport` should also carry per-child NN value estimates (needs a forward per
  child; cheap on 14x7, maybe not on 26x15). Defer until the overlay exists.

---

## Implementation log (2026-09-11)

### What shipped, against the phases as written

**Phase 0.** `trunk 0.21` + `wasm32-unknown-unknown` + Leptos 0.8.20 CSR; the client compiles on
*both* targets, so it stays a workspace member and `cargo test --workspace` is unaffected beyond
compile time. Decision 9 verified by `botbowl-engine/tests/register_rolls_full_game.rs` — 40 whole
games on 14x7 and 15 at the compiled capacity, every die supplied by the caller: **no engine
assert fired**, so nothing had to be fixed. Decision 8 verified by
`botbowl-nn/tests/capacity_parity.rs`: golden encoder bytes and action↔cell maps captured under a
`BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4` build are byte-identical under the default
26x15 build, and the `#[ignore]`d end-to-end check against `bbnet_14x7_gen0c` gives an identical
value and all 47 identical priors under both capacities.

**Phase 1.** `botbowl-web/{proto,server}` as designed. Exit criterion met and pinned:
`tests/play_over_websocket.rs` plays whole games over the real socket on 14x7 *and* 26x15 from one
binary, choosing only from what the `ViewState` describes, asserting every human prompt offered at
least one action.

**Phase 2.** `MctsConfig` landed in the fuller of the two forms left open (see below), plus
`MctsBot::{last_search, explore, principal_variation}` and three read-only navigation methods added
to `recon_mcts`. Exit criterion met: candidates on the pitch as a heatmap, a ranked list, and a
clickable PV.

**Phase 3.** `FixNextRoll` (pin ahead of the roll — the session never pauses on one), recording
save, resume-from-recording in the lobby, NN value on the inspector's summary line, and
"step into the PV" rendering a node's stored `GameState` on the pitch.

**Phase 4.** Not started, as planned.

### Divergences from the design

1. **Manual setup is not possible, so the client does not offer it.** §client says "per-square
   `SelectPosition` for manual placement". The engine's `Setup` procedure
   (`kickoff_procs.rs`) offers **only** `SimpleAT::SetupLine` and then `EndSetup` — there is no
   per-square placement action to expose. "Human controls every decision the engine surfaces"
   still holds; manual setup is simply not one of them. Adding it is an engine feature, not a UI
   one.
2. **`MctsConfig` took the larger of the two options in "Still open".** Minimal (struct +
   `from_env` default, builders untouched) would not have been enough: `BLOOD_MCTS_HORIZON`,
   `_WORKERS` and `_MEMORY` were re-read from the environment inside *every* `get_action`, so they
   **overrode** an explicit builder call. A lobby that says "4 workers" has to mean it. So
   `run_search` now reads only `self.config`; `MctsConfig::from_env()` is still the default, so
   every CLI path behaves exactly as before, and env vars are resolved once at construction
   instead of per call.
3. **`SearchReport` gained `search_id`, and Q is reported in one frame.** Neither was in the
   design. The bot keeps exactly one tree, so an inspector opened on an earlier move cannot be
   walked — without an id the server would happily answer about the wrong position. And signing
   each node's Q by its *own* player (the obvious reading of "mover-centric") makes the inspector
   flip sign at every ply: a root at `+0.27` whose best child reads `-0.27` looks like the bot
   picked the worst move. Everything is in the searching agent's frame now.

The "Still open" question about per-child NN value estimates is still open and still deferred: the
overlay exists, nobody has wanted the number yet.

### Bugs found and fixed on the way

- **`view::derive` panicked mid-kickoff.** `Kickoff` sets
  `BallState::InAir(aim + direction * len)` with `len` capped only at `max_scatter()`, which on a
  narrow board puts the ball at a *negative* coordinate for a step or two. `Position` is `i8`, so
  the view's unchecked `pos.y as usize` wrapped to ~2^64 and the multiply overflowed. The
  websocket test never saw it (it only observes states where the *human* is asked to move), so the
  guard is now `tests/derive_every_state.rs`, which derives a view at **every** engine pause of
  whole random games and asserts a kick actually leaves the grid in the process.
- **A panicking session was invisible.** The blocking thread died, the socket stayed open, and the
  browser clicked into nothing. `ws.rs` now selects on the session handle and reports the crash.
- **The principal variation opened on unvisited children.** Folding "never scored" in as
  `q_home.unwrap_or(i64::MIN)` and then multiplying by the mover sign turned it into `+9.2e18` for
  an `Away` node, so the PV walked straight into a `0v` child and stopped. Unscored children now
  rank last explicitly.
- **Stale wasm after a rebuild.** The dev server now sends `cache-control: no-cache`; without it
  the browser kept serving the previous `index.html` and the page ran old code that looked exactly
  like a logic bug.

### A finding for the experiment queue, deliberately not fixed here

On **every** board below the compiled default, the engine's own `SetupLine` formation fails the
engine's own `is_setup_legal`: the clamped offsets reach the line-of-scrimmage *column* but at `y`
values outside `los_y_range`, so the scrimmage count is **0** against a required 3. Measured at
16x9/4, 18x11/6 and 22x11/8; legal at 28x17/11. Nothing in the engine enforces
`is_setup_legal`, so it has never affected play — but it means a 14x7 drive opens with nobody on
the line, and it is the formation every 14x7 model was trained against. Changing it invalidates
the trained nets and every measured result in plan 032, so it belongs there (added to "Still
open"), not in a UI commit. Pinned as a test:
`botbowl-web/server/src/view.rs::the_auto_setup_formation_is_illegal_on_clamped_boards`.

### Where the code lives

`botbowl-web/CLAUDE.md` carries the architecture and the gotchas; it loads automatically when
working in that tree.
