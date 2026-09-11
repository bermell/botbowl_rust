# CLAUDE.md — botbowl-web

Human-vs-bot Blood Bowl in a browser, with the bot's search opened up next to the board
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
- **Undo is a stack of `GameState` clones plus the session RNG**, pushed at each *human* decision
  point, so one undo rewinds across the bot's whole reply. No engine involvement. `MctsBot`'s
  cached tree stops matching its anchor after an undo and discards itself, which costs a wasted
  reuse and nothing else.
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

## Bots are a closed enum, and take no env vars

`bots::SessionBot` is `Random | Scripted | Mcts`, not `Box<dyn Bot>`, because the inspector needs
`last_search()` / `explore()` off the MCTS bot and downcasting a trait object to get them is
worse than three variants.

`mcts_config` starts from `MctsConfig::new()` — **not** `from_env()`. Before plan 034, half the
knobs were resolved from `BLOOD_MCTS_*` at `::new` and the other half re-read from the environment
on *every* `get_action`, so an explicit builder call could be silently overridden and a process
could not run two differently-tuned bots at all. `MctsConfig` (in `botbowl-mcts`) fixed that;
`from_env()` remains the default for every CLI path, so nothing else changed behaviour.

## The inspector reads one search, in one frame

- **Q is reported in the searching agent's frame everywhere** — root, candidates, PV, explorer.
  `recon_mcts` stores Home-centric Q; signing each node by its *own* player instead reads as a
  sign flip at every ply, and a root at `+0.27` whose best child says `-0.27` looks like the bot
  picked the worst move. Pinned by the frame assertions in `tests/mcts_inspector.rs`.
- **The bot keeps exactly one tree** — the most recent search's. Every search re-roots or rebuilds
  it, so a report from an earlier move can be read but not walked. `search_id` says which, and the
  server refuses a stale one rather than answering about the wrong position.
- Walking uses `recon_mcts`'s `Node::get_children_info` / `Node::get_child` / `Tree::get_root_node`
  (added by this plan; `get_next_move_info` only ever reached the root's children). Inspection is
  inert — no descent, no visit bump — pinned by `recon_mcts/tests/navigate.rs`.
- Visit counts are "descents through this node", cumulative across a reused tree and frozen once a
  subtree is solved. They measure search effort, not move quality; `MctsBot` picks by aggregated Q
  for exactly that reason, and the heatmap normalises against the busiest sibling rather than the
  root's own counter.

## Sprites come from the sibling checkout, never from git

`--assets-dir /path/to/botbowl/botbowl/web/static/img` is mounted at `/img/`. The player icons are
explicitly **not** under the botbowl licence (`botbowl/README.md:61`), so nothing binary enters
this repo. Every sprite path on the wire is relative to that mount.

The pitch is drawn as a **CSS grid over the engine board** (playable squares plus the two-cell
out-of-bounds border, so a `Position` indexes the grid directly), not as one of the old repo's JPG
backgrounds — those exist for six fixed sizes, none of which are the tiers we train on.

## Gotchas

- **The dev server sends `cache-control: no-cache` on everything.** Without it a browser keeps
  serving the previous `index.html` after a `trunk build` and the page silently runs stale wasm —
  which looks exactly like a code bug and costs an hour the first time.
- **A ball in flight can be off the grid.** `Kickoff` sets `BallState::InAir(aim + direction * len)`
  with `len` capped only at `max_scatter()`, which on a narrow board reaches past the border ring
  into negative coordinates. `Position` is `i8`, so an unchecked `pos.y as usize` wraps to ~2^64;
  `view::index_of` is fallible for this reason and off-grid balls are simply not drawn.
- **Manual setup is not possible.** The engine's `Setup` procedure offers only `SetupLine` and then
  `EndSetup` (`kickoff_procs.rs`) — there is no per-square placement action, so the UI cannot
  expose one. `is_setup_legal` exists but nothing in the engine enforces it, and below the default
  board the engine's own formation fails it (recorded in `plans/032`, "Still open").
- **A long search blocks its own session.** `spawn_blocking` keeps the socket alive, but there is
  no cancel, so the human cannot undo mid-think. Accepted for a POC.

```sh
cargo test -p botbowl-web-server        # mirrors, view derivation, whole games over the socket
cargo check -p botbowl-web-client --target wasm32-unknown-unknown
```
