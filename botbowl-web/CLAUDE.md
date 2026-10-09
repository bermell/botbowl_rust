# CLAUDE.md — botbowl-web

Blood Bowl in a browser — you against a bot, or two bots against each other — with every decision
logged and the net-guided search behind each bot move opened up next to the board
(`plans/034-plan--web-play-ui.md`). Three crates:

| crate | target | depends on |
|---|---|---|
| `proto/` | host **and** wasm32 | `serde` only |
| `server/` | host | engine + mcts + nn + axum/tokio |
| `client/` | wasm32 (built with `trunk`) | `proto` + leptos |

```sh
cd botbowl-web/client && trunk build --release      # once, and after any client change
cargo run --release -p botbowl-web-server -- \
    --assets-dir /path/to/botbowl/botbowl/web/static/img
# → http://127.0.0.1:8080
```

**Build the server `--release` for real play.** A debug MCTS search is one to two orders of
magnitude slower, which reads as a hung UI rather than a slow one.

`--dist-dir`, `--models-dir` and `--recordings-dir` default to paths resolved from the server
crate's own `CARGO_MANIFEST_DIR`, **not** the working directory, so the two commands above work
from anywhere — including from inside `botbowl-web/client`, which is where you just ran `trunk`.
They used to be cwd-relative, and starting the server from the client directory then served a bare
404 with `0 model(s)` and no hint why. A missing client build is now a hard error naming the path
and the command to fix it, not a warning.

## The one rule: the client is a renderer

`proto` has **no engine dependency** and the client has no game logic — no rules, no geometry, no
legality. The server ships a fully derived `ViewState` where every square already carries its
kind, sprite, legal actions, path success probability, route, block dice and tackle-zone counts,
and the client draws it. Two things fall out of that:

- the wasm build stays trivial (the engine does not compile to wasm32 today — `getrandom`,
  `Recording::to_file`, `color-backtrace`, `Instant`; see plan 034 phase 4);
- **all** the game logic sits in one pure function, `server/src/view.rs::derive`, which is
  unit-tested against `GameStateBuilder` positions and fuzzed over whole random games
  (`tests/derive_every_state.rs`).

Anything the client draws from the view but does not *fetch* is a method on `ViewState`, not a
helper in the client — `mover()`, `moves_offered()`, `threat_team()` decide the tackle-zone layer
and are unit-tested in `server/src/view.rs` against real positions. The layer paints the zones of
the mover's **opponent**, so it flips sides on its own when the bot is the one choosing; `mover()`
reads `to_act` before `team_turn` because the two disagree (an uphill block's dice are picked by
the defender). `threat_team` is `None` outside a move phase, so a kickoff or a dice prompt is not
covered in colour. Resolve it **once per board**, not once per square: it scans every square. The
layer is its own three-way toggle in the client (`TzMode`: auto / always / off, key `T`), separate
from the exclusive tint overlays, because it has to coexist with any of them and because with two
bots playing it flips sides every turn — some people want it off.

Two things the view says that a `GameState` never does: `PlayerView.used` is the engine's flag,
but the *active* player is drawn in full colour and with the "not acted" sprite however it is set
(the engine marks a player used the moment they activate, and greying them mid-move reads as
"done"); and `to_act` lights up that side's dugout and scoreboard entry.

## Click-to-act: select, then target (the mouse stays on the board)

A click on a player who may be declared does not ask "Start move / Start blitz / ...". It sends
`ClientMsg::Select { pos }`; the server answers `ServerMsg::Selection(SelectionView)` from
`view::selection`, which steps **every** legal declaration on a clone (`RegisterRolls`, so a
declaration that rolls at once — Jump Up — just offers nothing) and resolves each square to one
`IntentView { start, action }`: Move onto an empty square or the loose ball, Block on an adjacent
standing opponent and Blitz on one further away, Foul on a prone one, and for the ball carrier the
likelier of Handoff and Pass to a standing team-mate. The rule set lives server-side so the client
stays a renderer. The next click on a target sends `ClientMsg::ActChain([start, action])` — one
undo point, dice in between rolled, the chain stops quietly at the first action that is no longer
legal. A selection carries the view's `seq` and is dropped on every new board.
**Selecting a team-mate ends the active player.** Mid-activation the engine offers no
declarations (only `Turn` does), so `ViewState::reselect` lists whoever could be declared once
`EndPlayerTurn` is played (stepped on a clone, `view::after_end_player_turn`); `selection` falls back
to previewing that position and sets `SelectionView::end_first`, and `SelectionView::chain` prepends
`EndPlayerTurn` — three actions, one undo point. A prone player's own square is a Move target
too — the engine's stand-up-in-place path — so clicking a selected prone player again stands them
up where they lie. A player picture that fails to load falls back to the role's stock sprite
(`fall_back` in `pitch.rs`, with a console warning naming the missing file), and the server logs
every 404 under `img/` to stderr and `~/.cache/botbowl/missing-assets.log` (`log_missing_sprite`,
with the asset directory it looked in) — read that file first when someone reports a broken image. The declarations
sit as buttons in the action bar above the board (with End turn and every other simple action);
block dice and the odds show on the **hovered** target only. `ViewState::prompt` (`view::prompt`)
anchors a reroll / optional-skill / block-dice question on the square it concerns — the rolling
player (`SimpleProcContainer::id`), the pushed one (`Push::on`), the defender for block dice, or
whichever of attacker/defender is being asked — and the client asks it there. Pinned by the
`selection`/`prompt` tests in `view.rs`, the fuzz in `derive_every_state.rs` and
`a_selected_player_declares_and_moves_in_one_undoable_decision`.

Layout: the dugouts sit **below** the board, each on the side its team sets up on (Home's half is
the high-x one, so Away left, Home right), carrying score, turn, rerolls and the bot's search
spinner. Each end zone is painted in the colour of the team that defends it, so Home (blue)
scores in the red one — `SquareKind::EndzoneHome` is the zone Home *attacks*, coloured away.

The price is that `proto` hand-mirrors the engine's action and dice enums. `server/src/mirror.rs`
pays it: every conversion is an **exhaustive match with no wildcard arm**, so adding an engine
variant is a compile error there, and every variant round-trips in its tests.

## The session owns the game; the socket only carries JSON

One websocket per game. `ws.rs` does nothing but translate frames; `session.rs` runs on a
`spawn_blocking` thread that owns the `GameState` outright. That is not a style choice — the
engine is synchronous and `MctsBot::get_action` spawns its own `std::thread::scope` workers
inside the call, so it must not run on the async runtime. Channels in, channels out, no locks.

- **`DiceMode::RegisterRolls` is the contract** (plan 034 decision 4). The session rolls every die
  itself with its own `ChaCha8Rng` and resumes through `step_with_roll_or_action`, which is the
  only reason the UI can show dice at all — `RollDice` resolves them inside the engine where
  nothing can see them. "Pin the next roll" falls out for free. Pinned to a full game on two board
  sizes by `botbowl-engine/tests/register_rolls_full_game.rs`.
- **Either seat is a human or a bot** (`GameSpec { home: Seat, away: Seat }`); the session holds
  `bots: [Option<SessionBot>; 2]` and "whose decision" is `bots[actor].is_none()`. Bot-vs-bot has no
  human decision point, so it has no undo, and hot-seat (two humans) falls out for free.
- **Undo is a stack of `GameState` clones plus the session RNG**, pushed at each *human* decision
  point, so one undo rewinds across the bot's whole reply. No engine involvement. `MctsBot`'s
  cached tree stops matching its anchor after an undo and discards itself, which costs a wasted
  reuse and nothing else. The snapshot also carries the decision-log length, and an undo sends
  `DecisionsTruncated { keep }` so the client's log rewinds with the board.
- **The hold sits between a bot's search and its move** (`StepMode`, `GameSession::pending`).
  `advance` rolls every die straight through to the next decision; when that decision is a bot's,
  it searches, logs and reports the decision (`ServerMsg::Decision`) and *then* asks `hold()`.
  Under `Manual`/`Auto` it returns to the run loop with the move unplayed: the board on screen is
  the position the search was about, the report beside it is that search, and
  `ViewState::pending_action` names what the next `StepOnce` plays (the Step button reads
  "Play ▶ Move (12,3)"). One click = play the held move, roll through, search the next decision,
  hold again. Dice are never held — they are the engine's work, not something anyone can inspect.
  (It used to hold *before* every step, dice included, which made a bot's turn forty clicks and
  showed the search report only after its move had already landed.) Releasing is `StepOnce`, the
  `Auto` deadline, or switching to `Run`; changing speed re-arms the hold without playing the move.
  Because the hold returns to the run loop rather than sleeping inside `advance`, undo, rewind,
  roll pinning and `ExpandNode` all keep working while it holds — which is the whole point, since
  inspecting the tree mid-turn is why the pacing exists. Pinned by
  `manual_pacing_holds_on_every_bot_decision`.
- **Rewind is a real rewind, from the log** (`ClientMsg::RewindTo { step }`). The session keeps
  every micro-step (`steps`) *and* the dice RNG at each (`rngs`), so clicking a line of the game
  log or "rewind to here" on a decision puts that position back as the live game: later steps,
  log lines and decisions are dropped (`LogTruncated`, `DecisionsTruncated`, the human undo stack
  pruned), the dice stream resumes where it was, and `advance` runs — under `Manual` the bot
  searches that position again and holds, so a decision that went by can be re-read. Under `Run`
  the session switches itself to `Manual` first, or the rewind would be gone before anyone saw it.
  Bot-vs-bot has no human undo point, so this is its only way back. Pinned by
  `a_rewind_restores_an_earlier_step_and_holds_there`.
- **One game log, streamed** (`ServerMsg::Log(LogEntry)`). Dice and text used to be two panels fed
  two ways (a `Dice` message and a `log_tail` re-sent inside every view). Every line now carries
  its kind, side, micro-step, the die faces for a roll and the decision index for an action, and a
  roll line is prefixed with what it was *for* — `dice::purpose(proc_stack_top())` at the moment
  the engine paused on the request ("Dodge · D6 3+ — success", "Armour · 2D6 9+ — failure").
- **The trail and the block arrow are derived, never stored.** `ViewState::trail` is the active
  player's squares this activation, read by `view::trail` off the step snapshots — with the
  roll-free squares `MoveAction::continue_along_path` walks in one micro-step filled in from the
  pathfinder route in the last snapshot that had a path buffer. **No history goes into
  `GameState`**: a state that remembered its past would stop recombining with one that reached the
  same position another way. `ViewState::block` is the `Block` procedure found anywhere on the
  stack (`GameState::proc_stack_iter`), attacker = active player, drawn as an SVG over the grid.
  - The pacing lives on the **connection**, not the session: it is set from the game screen, must
    survive "New game", and a `SetStepMode` can arrive before the first `NewGame`.
  - `Auto` polls with `try_recv` on a 5 ms tick rather than `tokio::time::timeout`, because this
    thread is a `spawn_blocking` worker that must not touch the async runtime.
  - **`Run` yields between bot moves** (`yielded`, `wake_at() == now`). Without it a bot-vs-bot game
    played to the end inside one `advance` call, deaf to the socket — no pause, no new game. The
    yield sits in front of the *next* bot move, never behind the last one, so it can never land on
    a human's decision point (a view sent there raced the human's click against the resume). A
    yield re-sends the board at most every 100 ms: two random bots make thousands of moves a second.
- **A panicking session is reported, not silent.** The engine and the bots are full of `assert!`s;
  before `ws.rs` learned to select on the session handle, a panic left the socket open and the
  browser clicking into a dead thread.
- The session keeps its **own** event log. Engine logging both pushes to a Vec and `println!`s a
  rules trace on every micro-step — that would flood the server's stdout with something no player
  wants to read.

## One binary, any board (plan 034 decision 8)

The build-time env vars set only **capacity**; `GameSpec.board` picks the runtime `BoardDims`, and
the lobby offers the presets that fit. Build at the default 26x15/11 and one server plays 8x3,
14x7 and 26x15. This is safe because everything downstream is runtime-dims-aware, which
`botbowl-nn/tests/capacity_parity.rs` pins with golden bytes captured under a 14x7-capacity build
(encoder + action↔cell map, which together imply forward parity); confirmed end to end against
`bbnet_14x7_gen0c` — identical value and all 47 priors under both capacities.

**A model trained on another board size panics inside `NnEvaluator` rather than erroring**, so the
`_WxH_` filename tag is checked twice: the lobby only offers matching models, and the server
re-checks before loading. That convention is load-bearing here for the first time — nothing in
Rust parsed those filenames before.

## Bots are a closed enum, always on a net, and take no env vars

`bots::SessionBot` is `Random | Mcts`, not `Box<dyn Bot>`, because the inspector needs
`last_search()` / `explore()` off the MCTS bot and downcasting a trait object to get them is
worse than two variants.

**The MCTS bot always searches with the net's value *and* priors** (`Evaluator::Nn`); `MctsSpec`
carries a `model`, not an evaluator choice, and an empty one is an error, not a fallback. The
heuristic, pure-TD and value-only evaluators and the scripted bot were removed from the web app —
they remain CLI diagnostics. `SessionBot::Mcts` keeps its own `Net` handle beside the bot so the
session can read the net out on positions the search never scored (a human's decision; a root
child before the search moved its value). Tests run on the committed
`botbowl-nn/tests/fixtures/tiny.onnx` (untagged, so it fits every board).

`mcts_config` starts from `MctsConfig::new()` — **not** `from_env()`. Before plan 034, half the
knobs were resolved from `BLOOD_MCTS_*` at `::new` and the other half re-read from the environment
on *every* `get_action`, so an explicit builder call could be silently overridden and a process
could not run two differently-tuned bots at all. `MctsConfig` (in `botbowl-mcts`) fixed that;
`from_env()` remains the default for every CLI path, so nothing else changed behaviour.

## Every decision is logged

`ServerMsg::Decision(DecisionRecord)` goes out for **every** action either side takes, human or bot:
team, decider, action, proc, half/turn, legal-action count, the net's `NetReadout` of the position
(value + softmax over *all* legal actions — not the search's PUCT prior, which is over the pruned
set and rescaled to mean 1), and for an MCTS move the whole `SearchReport`. A human's decision is
read out by the first net seated in the game. Records are snapshots, so any past decision's root
stays readable; `ShowDecision { index }` re-renders the board it was taken on (the session keeps
`decision_steps`, an index into `steps`), so a logged search's heatmap can be read over the
position it was about.

Root children carry `prior_share` (prior over the searched siblings, sums to 1), `visit_prob`
(visits over sibling visits — a visit-count policy target), and `net_value` (the value head on the
child position, in the report's agent frame — `Q − net V` is what the search changed its mind
about; `None` for chance and never-visited children). The client merges the root children with the
readout's priors, so actions pruned before search show up as rows with a net probability and no
visits.

## The inspector reads one search, in one frame

- **Q is reported in the searching agent's frame everywhere** — root, candidates, PV, explorer.
  `recon_mcts` stores Home-centric Q; signing each node by its *own* player instead reads as a
  sign flip at every ply, and a root at `+0.27` whose best child says `-0.27` looks like the bot
  picked the worst move. Pinned by the frame assertions in `tests/mcts_inspector.rs`.
- **Each bot keeps exactly one tree** — its most recent search's. Every search re-roots or rebuilds
  it, so a report from an earlier move can be read but not walked. `search_id` is one counter across
  both bots; the session remembers each seat's latest (`latest_search`), `ExpandNode` routes to the
  bot whose latest it is, and refuses any other id rather than answering about the wrong position.
  With two MCTS bots, both latest searches are walkable.
- Walking uses `recon_mcts`'s `Node::get_children_info` / `Node::get_child` / `Tree::get_root_node`
  (added by this plan; `get_next_move_info` only ever reached the root's children). Inspection is
  inert — no descent, no visit bump — pinned by `recon_mcts/tests/navigate.rs`.
- Visit counts are "descents through this node", cumulative across a reused tree and frozen once a
  subtree is solved. They measure search effort, not move quality; `MctsBot` picks by aggregated Q
  for exactly that reason, and the heatmap normalises against the busiest sibling rather than the
  root's own counter.

## Two read-outs, and they answer different questions (plan 043)

- **`SearchReport.health`** is the search's own vitals, next to what it concluded: this decision's
  tree-reuse outcome plus the rate so far, and recombination's hit rate against its wasted-compare
  rate. Rendered as the inspector's "Search health" block. `anchor_miss` at a turn boundary is
  expected; the same outcome mid-turn, or a `lookup_miss`, is not.
- **`ServerMsg::Net(NetReadout)`** is the net's read of the **current** position — value
  (Home-centric in `[-1, 1]`) and policy over the side-to-act's legal actions — emitted from
  `GameSession::view` on *every* board change. The "Net priors" overlay paints it while the log is
  following the game. It is a message of its own,
  not a `ViewState` field, for two reasons: the view is re-sent in full on every step and should
  not carry a value most sessions do not have, and the point is that it updates during the
  **human's** turn — `SearchReport.evaluator_value` only ever appears after a bot move. One forward
  pass per ply is nothing next to a search, and it cannot perturb the game because a frozen net is
  a pure function of the state.
- **Both name a team rather than printing a bare signed number.** `inspector::favours` turns a
  Home-centric value into `Home 0.42`, because the question people ask of a value head is "who does
  it think scores next", and a bare `+0.42` makes the reader remember whose frame it is in.
  `evaluator_value` arrives in the *searching agent's* frame and is flipped back to Home's first.

## Sprites come from the sibling checkout, never from git

`--assets-dir /path/to/botbowl/botbowl/web/static/img` is mounted at `/img/`. The player icons are
explicitly **not** under the botbowl licence (`botbowl/README.md:61`), so nothing binary enters
this repo. Every sprite path on the wire is relative to that mount.

The pitch is drawn as a **CSS grid over the engine board** (playable squares plus the two-cell
out-of-bounds border, so a `Position` indexes the grid directly), not as one of the old repo's JPG
backgrounds — those exist for six fixed sizes, none of which are the tiers we train on.

## `~/.config/botbowl/web.toml`, and the worker's cached nets

Paths resolve flag → `~/.config/botbowl/web.toml` → built-in default (`config::resolve`, shared by
`botbowl-web-server` and the hub's `/play/`). Keys: `assets_dir`, `models_dirs` (a list),
`worker_cache`, `teams_dir`, `dist_dir`; `~` expands. A missing file is **created** on first start
with every key documented and the ones this machine can detect filled in; a file that does not
parse (`deny_unknown_fields`) stops the server — a typo'd key must not look like an ignored one.

The lobby offers the configured directories' nets, then a remote worker's model cache
(`~/.cache/botbowl/models` by default, `worker_cache = ""` turns it off) — so a box that has been
a hub worker can play the nets the hub shipped it without copying anything. Cache entries are
named by the hub's `<hex>.json` sidecar (`botbowl_hub_proto::ModelMeta`, protocol v14 —
`botbowl-hub/CLAUDE.md`) as `cache: <run>/models/<file>.onnx`, with the board tag from that file
name; an entry not yet named is listed by its hash, untagged, after the named ones (the probe still
refuses a net that does not fit the board). A cached net whose **bytes** match a local file is
listed once, as the local file — compared by size, then BLAKE3, never by name, since every run has
a `gen0`.

## Served standalone or under the hub's `/play/`

The client is built with relative URLs (`Trunk.toml` `public_url = "./"`), and the socket
(`ws.rs::endpoint`) and every sprite (`img/...`) are page-relative, so one `dist/` serves at `/`
from `botbowl-web-server` and at `/play/` from `botbowl-hub serve` (which nests this crate's
`router` whole — see `botbowl-hub/CLAUDE.md`). **Never write an absolute `/img` or `/ws` in the
client**; it would break the hub mount only.

`PlayOptions` on `AppState` holds what differs between the two: the initial pacing, a cap on
search threads (`--max-workers` / the hub's `--play-max-workers`), the team directory, and whether
`StartFrom::Recording` (the browser naming a server path) is allowed — off on the hub.
`PlayOptions::default()` is the safe one; tests opt into `StepMode::Run` explicitly.

**Defaults are cheap on purpose (the user, 2026-10-05):** a new connection starts in
`Auto { ms: 600 }`, the lobby proposes bot-vs-bot (`GameSpec::default_for`), and `MctsSpec`
defaults to `workers: Some(1)`. Opening the page shows a slow game on one thread per bot, not
a machine-wide search. Keep it that way.

## Random-start drives

`StartFrom::RandomDrive { seed }` starts from `botbowl_play::drives::position_state` with the
default `RandomStartBias` — the exact position the corpus draws for that seed — and the session
stops on `DriveStart::over` (a score, the half changing, or game over) with
`ServerMsg::DriveOver`, the same place a training trajectory stops. The drive check runs *before*
the game-over check, because a drive that runs out the second half is a drive that ended.
`seed: None` draws a fresh position each time (the "Next drive" button resends the spec).
Generated players keep their generated stats; the seats' teams only lend their pictures.

## Teams

The engine has no team concept: every game is the same four-role roster. `proto/src/team.rs`
defines `TeamDef` (positions with stats, skills, engine role, picture, max count) and twelve
built-in LRB6 rosters, cut to the skills the engine has; `server/src/teams.rs` applies a roster
to a fresh game's dugout before the coin toss (`teams::apply`) and lists, saves and deletes the
saved ones as `~/.config/botbowl/teams/<slug>.json` (uploaded pictures in `teams/img/`, served
at `img/custom/`).

- **The built-in `Human` team is the engine roster exactly**, slot for slot — pinned by
  `the_default_team_is_the_engine_roster_exactly`. It is the default for both seats, so a game
  nobody customised is the game the nets were trained on.
- **A player's picture is looked up from its stats**, not stored: `PlayerStats` has no room for it
  and player ids are reassigned whenever a player moves between dugout and pitch. `Looks` matches
  (team, role, stats, skills) against the team's positions and falls back to the first position
  of the role. Two positions identical in all of those share a picture.
- `ag` is the engine's LRB6 value (roll `7 - ag`), `pa` a pass target. The role is not cosmetic:
  the scripted kickoff setup ranks players by role.
- Skills are carried by label; a label the engine does not know is refused. The editor marks
  skills outside `Skill::good_skills()` as "no rules yet" — they exist in the enum (and the net's
  input) but do nothing in a game.

## Gotchas

- **The dev server sends `cache-control: no-cache` on everything.** Without it a browser keeps
  serving the previous `index.html` after a `trunk build` and the page silently runs stale wasm —
  which looks exactly like a code bug and costs an hour the first time.
- **A ball in flight can be off the grid.** `Kickoff` sets `BallState::InAir(aim + direction * len)`,
  `len` capped at `max_scatter()` and (on a narrow board) scaled down by `scatter_divisor()` — still
  possible to land past the border ring into negative coordinates, just rarer than before that
  scaling existed. `Position` is `i8`, so an unchecked `pos.y as usize` wraps to ~2^64; `view::index_of`
  is fallible for this reason and off-grid balls are simply not drawn. Pinned by
  `a_kickoff_deviate_off_the_grid_does_not_panic_the_view` (`derive_every_state.rs`), which forces
  the roll rather than waiting for random play to find it, since `scatter_divisor` made that unreliable.
- **A player pushed into the crowd is briefly at an out-of-bounds position while still "on
  pitch."** `Push::do_moves` moves them to the (out-of-bounds) crowd square; `Injury::new_crowd`,
  queued right after, is what actually unfields them — so `view::derive` can be called in between
  (a `check()` mid-game, or a real session's dice-pending hold) with a fielded player whose position
  fails `BoardDims::is_out`. `tackle_zones` must skip such a player rather than call
  `get_adj_positions` on them, which asserts in-bounds. Pinned by
  `a_player_mid_crowd_push_does_not_panic_the_view` (`view.rs`).
- **Setup is one decision per player, and the formations are a session shortcut.** The engine's
  `Setup` procedure asks about `info.active_player` and offers `PosAT::PlacePlayer` squares on the
  own half plus `SimpleAT::BenchPlayer` while the roster has a spare; it closes the setup itself
  once `team_size` are placed. Everyone available for the drive is *staged* on the pitch in the
  own end-zone column with `used == true` until placed, so mid-setup there are more players on
  the pitch than `team_size` — `ViewState::setup` (`SetupView`) carries the placed/waiting counts
  and the names of the `Formation`s that fit the board. `ClientMsg::AutoSetup(name)` plays the
  rest of the human's setup out with that formation, through `self.step` so the recording keeps
  every placement, as **one** undo point. `is_setup_legal` still isn't enforced by the engine,
  but every offered formation satisfies it on every board (pinned in `view.rs`).
- **A long search blocks its own session.** `spawn_blocking` keeps the socket alive, but there is
  no cancel, so the human cannot undo mid-think. Accepted for a POC. (Between searches the session
  does listen — see the `Run` yield.)

```sh
cargo test -p botbowl-web-server        # mirrors, view derivation, whole games over the socket
cargo check -p botbowl-web-client --target wasm32-unknown-unknown
```
