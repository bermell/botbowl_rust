//! The game screen: scoreboard, dugouts, the pitch, and the action panel.
//!
//! The pitch is a CSS grid over the **engine** board — playable squares plus
//! the two-cell out-of-bounds border — because a `Position` then indexes the
//! grid directly and any board size works without an artwork asset (decision
//! 5). The JPG pitch backgrounds in the old repo only exist for six fixed
//! sizes, none of which are the ones we train on.

use std::collections::HashMap;

use botbowl_web_proto::dice::{BlockDice, RollResult};
use botbowl_web_proto::log::{LogEntry, LogKind};
use botbowl_web_proto::msg::{ClientMsg, StartFrom, StepMode};
use botbowl_web_proto::search::SearchEdge;
use botbowl_web_proto::view::{BallView, BlockView, SquareKind, SquareView, ViewState};
use botbowl_web_proto::{Action, Position, TeamType};
use leptos::prelude::*;

use crate::state::{click_target, App, Click, Menu, Overlay, TzMode};
use crate::ws;

#[component]
pub fn Game() -> impl IntoView {
    let app = expect_context::<App>();
    keyboard_shortcuts(app);

    view! {
        <div class="game">
            <Scoreboard />
            <div class="table">
                <Dugout side=0 />
                <div class="pitch-column">
                    <Board />
                    <StepControls />
                    <OverlayPicker />
                </div>
                <Dugout side=1 />
                <Panel />
            </div>
        </div>
    }
}

#[component]
fn Scoreboard() -> impl IntoView {
    let app = expect_context::<App>();
    move || {
        app.view.get().map(|v| {
            let s = v.scoreboard;
            let turn = |t: TeamType| if t == TeamType::Home { s.home_turn } else { s.away_turn };
            let seat = |t: TeamType| app.spec.get().map(|g| g.seat(t).label()).unwrap_or_default();
            let team = |t: TeamType| {
                app.spec
                    .get()
                    .map(|g| if t == TeamType::Home { g.home_team } else { g.away_team })
                    .unwrap_or_default()
            };
            let drive = app.spec.get().is_some_and(|g| g.start.is_drive());
            let (home_human, away_human) = (v.is_human(TeamType::Home), v.is_human(TeamType::Away));
            let acting = v.to_act;
            view! {
                <div class="scoreboard">
                    <div class="team home" class:you=home_human class:acting=acting == Some(TeamType::Home)>
                        <span class="name">{format!("Home · {}", team(TeamType::Home))}</span>
                        <span class="seat">{seat(TeamType::Home)}</span>
                        <span class="score">{s.home_score}</span>
                        <span class="meta">
                            {format!("turn {} · {} reroll(s){}", s.home_turn, s.home_rerolls,
                                if s.home_can_reroll { "" } else { " · used" })}
                        </span>
                    </div>
                    <div class="middle">
                        <span class="half">{format!("half {}", s.half.max(1))}</span>
                        <span class="weather">{s.weather.label()}</span>
                        <span class="proc">{v.proc.clone()}</span>
                        <span class="whose">
                            {match (s.game_over, v.to_act) {
                                (true, _) => "game over".to_string(),
                                (_, Some(t)) if v.is_human(t) && v.humans.len() == 1 => "your move".to_string(),
                                (_, Some(t)) if v.is_human(t) => format!("{t:?} (you) to act"),
                                (_, Some(t)) => format!("{t:?} to act"),
                                _ => format!("{:?}'s turn", s.team_turn),
                            }}
                        </span>
                        <span class="turnmark">
                            {format!("turn {} of the half{}", turn(s.team_turn), if drive { " · random drive" } else { "" })}
                        </span>
                    </div>
                    <div class="team away" class:you=away_human class:acting=acting == Some(TeamType::Away)>
                        <span class="name">{format!("Away · {}", team(TeamType::Away))}</span>
                        <span class="seat">{seat(TeamType::Away)}</span>
                        <span class="score">{s.away_score}</span>
                        <span class="meta">
                            {format!("turn {} · {} reroll(s){}", s.away_turn, s.away_rerolls,
                                if s.away_can_reroll { "" } else { " · used" })}
                        </span>
                    </div>
                </div>
            }
        })
    }
}

/// Squares a hovered move would pass through, for the route preview.
fn hovered_route(app: &App) -> Vec<Position> {
    let (Some(view), Some(pos)) = (app.view.get(), app.hover.get()) else {
        return Vec::new();
    };
    view.square(pos)
        .and_then(|s| s.route.as_ref())
        .map(|r| r.steps.clone())
        .unwrap_or_default()
}

/// Per-square heat for the search and net overlays, normalised against the
/// strongest square so it is always fully lit.
fn bot_heat(app: &App, overlay: Overlay) -> HashMap<Position, f32> {
    let mut heat = HashMap::new();
    let raw: Vec<(Position, f32)> = match overlay {
        Overlay::BotVisits => app
            .report()
            .map(|r| {
                r.children
                    .iter()
                    .filter_map(|c| match c.edge {
                        SearchEdge::Player(Action::Positional(_, pos)) => Some((pos, c.stats.visits as f32)),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        Overlay::NetPriors => app
            .priors_shown()
            .map(|n| {
                n.priors
                    .iter()
                    .filter_map(|p| p.action.position().map(|pos| (pos, p.prob)))
                    .collect()
            })
            .unwrap_or_default(),
        _ => return heat,
    };
    let peak = raw.iter().map(|(_, v)| *v).fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return heat;
    }
    for (pos, value) in raw {
        // Several actions can target one square (StartMove vs StartBlitz);
        // the square shows the best of them.
        let entry = heat.entry(pos).or_insert(0.0);
        *entry = entry.max(value / peak);
    }
    heat
}

#[component]
fn Board() -> impl IntoView {
    let app = expect_context::<App>();
    let route = Memo::new(move |_| hovered_route(&app));
    let heat = Memo::new(move |_| bot_heat(&app, app.overlay.get()));

    view! {
        <div class="board-wrap">
            {move || {
                app.hypothetical.get().map(|_| {
                    let what = match app.board_of.get() {
                        Some(i) => format!("board at decision #{i}"),
                        None => "board at a search-tree node".to_string(),
                    };
                    view! {
                        <div class="hypothetical-banner">
                            <span>{format!("showing the {what} — not the live game")}</span>
                            <button on:click=move |_| app.back_to_live()>"back to live"</button>
                        </div>
                    }
                })
            }}
            {move || {
                let Some(view) = app.hypothetical.get().or_else(|| app.view.get()) else {
                    return None;
                };
                let cols = view.dims.width as usize;
                // Scale the squares so a small board is not a postage stamp
                // and a full pitch still fits a laptop. Sprites are 28px, so
                // this up- or down-samples them; `image-rendering: pixelated`
                // keeps that honest rather than blurry.
                let sq = (980 / cols.max(1)).clamp(28, 46);
                let route = route.get();
                let heat = heat.get();
                let overlay = app.overlay.get();
                // A board that is not the live one takes no clicks.
                let my_turn = app.my_turn() && app.hypothetical.get().is_none();
                // Whose tackle zones to paint, resolved once for the whole
                // board: `threat_team` scans every square, so asking it per
                // square would make drawing quadratic.
                let threat = match app.tz.get() {
                    TzMode::Auto => view.threat_team(),
                    TzMode::Always => Some(view.mover().other()),
                    TzMode::Off => None,
                };
                let marks = Marks {
                    route: &route,
                    trail: &view.trail,
                    pending: view.pending_action.and_then(|a| a.position()),
                };
                let rows = view.dims.height as usize;
                Some(
                    view! {
                        <div class="pitch" style=format!("--cols: {cols}; --sq: {sq}px")>
                            {view
                                .squares
                                .iter()
                                .map(|square_view| {
                                    square(&view, square_view, &marks, &heat, overlay, my_turn, threat)
                                })
                                .collect_view()}
                            {view.block.map(|b| block_arrow(b, sq, cols, rows))}
                        </div>
                    },
                )
            }}
            <ActionMenu />
            <Thinking />
        </div>
    }
}

/// Per-square marks that come from outside the square itself.
struct Marks<'a> {
    /// The hovered move's route.
    route: &'a [Position],
    /// Where the active player has been this activation.
    trail: &'a [Position],
    /// The square a held bot move targets.
    pending: Option<Position>,
}

/// The attacker → defender arrow of a block in progress, as an SVG laid over
/// the grid. Drawn from square centre to square centre and stopped short of
/// the defender's centre so the head sits on the edge of their square rather
/// than on their face.
fn block_arrow(block: BlockView, sq: usize, cols: usize, rows: usize) -> AnyView {
    let centre = |p: Position| ((p.x as f32 + 0.5) * sq as f32, (p.y as f32 + 0.5) * sq as f32);
    let (x1, y1) = centre(block.attacker);
    let (x2, y2) = centre(block.defender);
    let (dx, dy) = (x2 - x1, y2 - y1);
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    let pull = sq as f32 * 0.42;
    let (ex, ey) = (x2 - dx / len * pull, y2 - dy / len * pull);
    let (sx, sy) = (x1 + dx / len * pull * 0.6, y1 + dy / len * pull * 0.6);
    let title = format!("block, {} dice", block.dice.signed());
    view! {
        <svg
            class="arrows"
            width=(cols * sq).to_string()
            height=(rows * sq).to_string()
            viewBox=format!("0 0 {} {}", cols * sq, rows * sq)
        >
            <title>{title}</title>
            <defs>
                <marker id="blockhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" class="head"></path>
                </marker>
            </defs>
            <line class="shadow" x1=sx y1=sy x2=ex y2=ey></line>
            <line class="shaft" x1=sx y1=sy x2=ex y2=ey marker-end="url(#blockhead)"></line>
        </svg>
    }
    .into_any()
}

fn square(
    view: &ViewState,
    sq: &SquareView,
    marks: &Marks<'_>,
    heat: &HashMap<Position, f32>,
    overlay: Overlay,
    my_turn: bool,
    threat: Option<TeamType>,
) -> AnyView {
    let app = expect_context::<App>();
    let pos = sq.pos;
    let actionable = my_turn && !sq.actions.is_empty();
    let on_route = marks.route.contains(&pos);
    let on_trail = marks.trail.contains(&pos);
    let pending_target = marks.pending == Some(pos);
    // Tackle zones of whoever is *not* moving, banded 1/2/3+ rather than
    // shaded continuously: the number changes the dodge target, so reading it
    // off at a glance matters more than a smooth gradient.
    let threat_tz = threat.map(|t| sq.tz(t)).filter(|&n| n > 0);

    let kind_class = match sq.kind {
        SquareKind::OutOfBounds => "oob",
        SquareKind::EndzoneHome => "endzone-home",
        SquareKind::EndzoneAway => "endzone-away",
        SquareKind::Scrimmage => "scrimmage",
        SquareKind::WingNorth => "wing-north",
        SquareKind::WingSouth => "wing-south",
        SquareKind::Normal => "normal",
    };

    // One overlay at a time: they all tint the same square.
    let (tint, tint_alpha) = match overlay {
        Overlay::Moves => ("move", if actionable { 0.55 } else { 0.0 }),
        Overlay::Risk => (
            "risk",
            sq.move_prob.map(|p| 1.0 - p).filter(|_| actionable).unwrap_or(0.0),
        ),
        Overlay::BotVisits | Overlay::NetPriors => ("bot", heat.get(&pos).copied().unwrap_or(0.0)),
        Overlay::None => ("none", 0.0),
    };

    let title = tooltip(sq, threat.unwrap_or_else(|| view.mover().other()));

    view! {
        <div
            class=format!("sq {kind_class} tint-{tint}")
            class:actionable=actionable
            class:on-route=on_route
            class:pending-target=pending_target
            class:has-ball=sq.ball.is_some()
            style=format!("--tint: {tint_alpha:.3}")
            title=title
            on:click=move |_| {
                match click_target(&app, pos) {
                    Click::Nothing => app.menu.set(None),
                    Click::Send(action) => {
                        app.menu.set(None);
                        ws::send(&ClientMsg::Act(action));
                    }
                    Click::OpenMenu(menu) => app.menu.set(Some(menu)),
                }
            }
            on:mouseenter=move |_| app.hover.set(Some(pos))
            on:mouseleave=move |_| app.hover.update(|h| { if *h == Some(pos) { *h = None } })
        >
            {threat_tz
                .map(|n| view! { <span class=format!("tz tz-{}", n.min(3))></span> })}
            {on_trail.then(|| view! { <span class="trail-dot"></span> })}
            {sq
                .player
                .as_ref()
                .map(|p| {
                    view! {
                        <img
                            class="player"
                            class:active=p.active
                            class:down=p.status != botbowl_web_proto::view::PlayerStatus::Up
                            // The engine marks a player `used` as soon as they are
                            // activated; grey them out only once they have *finished*.
                            class:used=p.used && !p.active
                            src=format!("img/{}", p.sprite)
                            alt=p.role.label()
                        />
                    }
                })}
            {sq
                .player
                .as_ref()
                .and_then(|p| p.status.overlay())
                .map(|o| view! { <img class="status" src=format!("img/{o}") alt="" /> })}
            {sq
                .ball
                .map(|b| {
                    let src = match b {
                        BallView::Carried => "icons/decorations/holdball.png",
                        BallView::InAir => "ball/tball.gif",
                        BallView::OnGround => "ball/sball.gif",
                    };
                    view! { <img class="ball" src=format!("img/{src}") alt="ball" /> }
                })}
            {sq
                .block_dice
                .filter(|_| actionable)
                .map(|d| view! { <img class="blockdice" src=format!("img/{}", d.badge()) alt="" /> })}
            {(overlay == Overlay::Risk && actionable && sq.move_prob.is_some_and(|p| p < 1.0))
                .then(|| {
                    view! {
                        <span class="prob">{format!("{:.0}", sq.move_prob.unwrap_or(0.0) * 100.0)}</span>
                    }
                })}
        </div>
    }
    .into_any()
}

/// Everything the square knows, as a hover tooltip — the overlays can only
/// show one dimension at a time. `threat` is the team whose tackle zones the
/// board is painting, which is the mover's opponent, not always the human's.
fn tooltip(sq: &SquareView, threat: TeamType) -> String {
    let mut parts = vec![format!("({}, {})", sq.pos.x, sq.pos.y)];
    if let Some(p) = &sq.player {
        parts.push(format!(
            "{:?} {} — ST{} MA{} AG{} AV{}{}{}",
            p.team,
            p.role.label(),
            p.st,
            p.ma,
            p.ag,
            p.av,
            if p.skills.is_empty() {
                String::new()
            } else {
                format!(" · {}", p.skills.join(", "))
            },
            if p.used { " · has acted" } else { "" },
        ));
        if p.status != botbowl_web_proto::view::PlayerStatus::Up {
            parts.push(format!("{:?}", p.status));
        }
    }
    let tz = sq.tz(threat);
    if tz > 0 {
        parts.push(format!("{tz} {threat:?} tackle zone(s)"));
    }
    if let Some(p) = sq.move_prob {
        parts.push(format!("route succeeds {:.0}%", p * 100.0));
    }
    if let Some(route) = &sq.route {
        if !route.rolls.is_empty() {
            parts.push(route.rolls.join(" · "));
        }
    }
    if let Some(d) = sq.block_dice {
        parts.push(format!("{} block dice", d.signed()));
    }
    parts.join("\n")
}

/// When a square offers several actions (`StartMove` vs `StartBlitz` on your
/// own player), ask rather than guess.
#[component]
fn ActionMenu() -> impl IntoView {
    let app = expect_context::<App>();
    move || {
        app.menu.get().map(|Menu { pos, actions }| {
            view! {
                <div class="action-menu">
                    <div class="menu-title">{format!("({}, {})", pos.x, pos.y)}</div>
                    {actions
                        .into_iter()
                        .map(|at| {
                            view! {
                                <button on:click=move |_| {
                                    app.menu.set(None);
                                    ws::send(&ClientMsg::Act(Action::Positional(at, pos)));
                                }>
                                    {at.icon()
                                        .map(|icon| view! { <img src=format!("img/{icon}") alt="" /> })}
                                    {at.label()}
                                </button>
                            }
                        })
                        .collect_view()}
                    <button class="cancel" on:click=move |_| app.menu.set(None)>
                        "Cancel"
                    </button>
                </div>
            }
        })
    }
}

#[component]
fn Thinking() -> impl IntoView {
    let app = expect_context::<App>();
    move || {
        app.thinking
            .get()
            .map(|label| view! { <div class="thinking"><span class="spinner"></span>{label}</div> })
    }
}

/// How fast the bot is allowed to play. The point of holding is not the
/// animation — it is that a held board is a board you can still inspect: the
/// search report next to it is the one that produced the move about to be
/// taken, and undo, the tree explorer and roll pinning all keep working while
/// the session holds.
#[component]
fn StepControls() -> impl IntoView {
    let app = expect_context::<App>();
    let set = move |mode: StepMode| {
        app.step_mode.set(mode);
        ws::send(&ClientMsg::SetStepMode(mode));
    };
    let is = move |mode: StepMode| app.step_mode.get() == mode;

    view! {
        <div class="steps">
            <span class="label">"Bot pace"</span>
            <button class="pace" class:on=move || is(StepMode::Run) on:click=move |_| set(StepMode::Run)>
                "Run"
            </button>
            <button
                class="pace"
                class:on=move || is(StepMode::Manual)
                on:click=move |_| set(StepMode::Manual)
            >
                "Step"
            </button>
            <button
                class="pace"
                class:on=move || app.step_mode.get().millis().is_some()
                on:click=move |_| set(StepMode::Auto { ms: app.step_ms.get() })
            >
                "Auto"
            </button>
            // `on:input` only moves the label; the mode change goes out on
            // `on:change`, so dragging the slider does not spray the socket.
            <input
                type="range"
                min="50"
                max="3000"
                step="50"
                prop:value=move || app.step_ms.get().to_string()
                on:input=move |ev| {
                    if let Ok(ms) = event_target_value(&ev).parse::<u64>() {
                        app.step_ms.set(ms);
                    }
                }
                on:change=move |_| {
                    if app.step_mode.get().millis().is_some() {
                        set(StepMode::Auto { ms: app.step_ms.get() });
                    }
                }
            />
            <span class="ms">{move || format!("{} ms", app.step_ms.get())}</span>
            <button
                class="stepone"
                disabled=move || !app.paused()
                on:click=move |_| ws::send(&ClientMsg::StepOnce)
            >
                {move || {
                    // While held, the button says what it will play: the move
                    // the search beside the board just chose.
                    match app.view.get().and_then(|v| v.pending_action) {
                        Some(a) => format!("Play ▶ {}", a.describe()),
                        None => "Step ▶".to_string(),
                    }
                }}
                <span class="key">"→"</span>
            </button>
            {move || {
                app.paused().then(|| view! { <span class="held">"held after the search"</span> })
            }}
        </div>
    }
}

#[component]
fn OverlayPicker() -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="overlays">
            {Overlay::ALL
                .into_iter()
                .map(|o| {
                    view! {
                        <button
                            class="overlay"
                            class:on=move || app.overlay.get() == o
                            on:click=move |_| app.overlay.set(o)
                        >
                            <span class="key">{o.key().to_string()}</span>
                            {o.label()}
                        </button>
                    }
                })
                .collect_view()}
            <button
                class="overlay tzmode"
                class:on=move || app.tz.get() != TzMode::Off
                title="tackle zones of the side not moving: auto = while a move is being chosen, always, or off"
                on:click=move |_| app.tz.update(|t| *t = t.next())
            >
                <span class="key">"T"</span>
                {move || format!("Tackle zones: {}", app.tz.get().label())}
            </button>
        </div>
    }
}

#[component]
fn Dugout(side: usize) -> impl IntoView {
    let app = expect_context::<App>();
    move || {
        app.view.get().and_then(|v| {
            let dugout = v.dugouts.get(side)?.clone();
            // The side whose decision is next: the dugout lights up in its
            // colour so a bot-vs-bot game reads at a glance.
            let acting = v.to_act == Some(dugout.team);
            let boxes = [
                botbowl_web_proto::view::DugoutPlace::Reserves,
                botbowl_web_proto::view::DugoutPlace::KnockOut,
                botbowl_web_proto::view::DugoutPlace::Injured,
                botbowl_web_proto::view::DugoutPlace::Ejected,
            ];
            Some(view! {
                <div class=format!("dugout team-{:?}", dugout.team) class:acting=acting>
                    <h3>
                        {format!("{:?}", dugout.team)}
                        {acting.then(|| view! { <span class="to-act">"to act"</span> })}
                    </h3>
                    {boxes
                        .into_iter()
                        .map(|place| {
                            let players: Vec<_> = dugout
                                .players
                                .iter()
                                .filter(|p| p.place == place)
                                .cloned()
                                .collect();
                            let (count, empty) = (players.len(), players.is_empty());
                            view! {
                                <div class="box" class:empty=empty>
                                    <span class="box-label">
                                        {format!("{} ({})", place.label(), count)}
                                    </span>
                                    <div class="bench">
                                        {players
                                            .into_iter()
                                            .map(|p| {
                                                view! {
                                                    <img
                                                        src=format!("img/{}", p.sprite)
                                                        title=p.role.label()
                                                        alt=p.role.label()
                                                    />
                                                }
                                            })
                                            .collect_view()}
                                    </div>
                                </div>
                            }
                        })
                        .collect_view()}
                </div>
            })
        })
    }
}

#[component]
fn Panel() -> impl IntoView {
    view! {
        <div class="panel">
            <SimpleActions />
            <GameOver />
            <GameLog />
            <Debug />
        </div>
    }
}

/// The game log: dice, decisions and notes in one stream, newest first. Every
/// line is a point in the game — click it and the server rewinds there
/// (`RewindTo`), so a position that has gone by can be put back, searched
/// again and inspected. The decision log below the board does the same from
/// the search's side.
#[component]
fn GameLog() -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="log">
            <h3>"Log" <span class="hint">"click a line to rewind to it"</span></h3>
            <div class="log-lines">
                <For
                    each=move || app.log.get().into_iter().rev()
                    key=|e| (e.index, e.step)
                    children=move |e: LogEntry| {
                        let step = e.step;
                        let decision = e.decision;
                        let kind = match e.kind {
                            LogKind::Note => "note",
                            LogKind::Roll => "roll",
                            LogKind::Action => "action",
                            LogKind::Score => "score",
                        };
                        let team = match e.team {
                            Some(TeamType::Home) => "team-H",
                            Some(TeamType::Away) => "team-A",
                            None => "",
                        };
                        let fixed = e.roll.as_ref().is_some_and(|r| r.fixed);
                        let faces = e.roll.as_ref().map(|r| r.faces.clone()).unwrap_or_default();
                        view! {
                            <div
                                class=format!("line {kind} {team}")
                                class:fixed=fixed
                                title=format!("rewind to step {step}")
                                on:click=move |_| {
                                    if let Some(index) = decision {
                                        app.selected.set(Some(index));
                                    }
                                    ws::send(&ClientMsg::RewindTo { step });
                                }
                            >
                                <span class="step">{step}</span>
                                <span class="faces">
                                    {faces
                                        .into_iter()
                                        .map(|f| {
                                            view! {
                                                <img src=format!("img/{}", f.img) title=f.label.clone() alt=f.label />
                                            }
                                        })
                                        .collect_view()}
                                </span>
                                <span class="text">{e.text}</span>
                            </div>
                        }
                    }
                />
            </div>
        </div>
    }
}

#[component]
fn SimpleActions() -> impl IntoView {
    let app = expect_context::<App>();
    view! {
        <div class="actions">
            {move || {
                let Some(view) = app.view.get() else { return None };
                let enabled = app.my_turn();
                Some(
                    view! {
                        <>
                            <div class="buttons">
                                {view
                                    .simple_actions
                                    .iter()
                                    .cloned()
                                    .map(|a| {
                                        let at = a.at;
                                        view! {
                                            <button
                                                class="act"
                                                disabled=!enabled
                                                on:click=move |_| ws::send(&ClientMsg::Act(Action::Simple(at)))
                                            >
                                                {a
                                                    .img
                                                    .clone()
                                                    .map(|img| {
                                                        view! { <img src=format!("img/{img}") alt="" /> }
                                                    })}
                                                {a.label.clone()}
                                            </button>
                                        }
                                    })
                                    .collect_view()}
                            </div>
                            {view
                                .setup
                                .clone()
                                .map(|setup| {
                                    let who = if view.is_human(setup.team) && view.humans.len() == 1 {
                                        "Setup".to_string()
                                    } else {
                                        format!("{:?} setup", setup.team)
                                    };
                                    view! {
                                        <div class="setup-panel">
                                            <div class="setup-note">
                                                {format!(
                                                    "{who}: placing player {} of {} ({} waiting)",
                                                    setup.placed + 1,
                                                    setup.team_size,
                                                    setup.waiting,
                                                )}
                                            </div>
                                            <div class="buttons">
                                                {setup
                                                    .formations
                                                    .iter()
                                                    .cloned()
                                                    .map(|name| {
                                                        let label = format!("{name} setup");
                                                        view! {
                                                            <button
                                                                class="act"
                                                                disabled=!enabled
                                                                on:click=move |_| ws::send(&ClientMsg::AutoSetup(name.clone()))
                                                            >
                                                                {label}
                                                            </button>
                                                        }
                                                    })
                                                    .collect_view()}
                                            </div>
                                        </div>
                                    }
                                })}
                            <div class="tools">
                                <button
                                    disabled=!view.can_undo
                                    on:click=move |_| ws::send(&ClientMsg::Undo)
                                >
                                    "Undo (Z)"
                                </button>
                                <button on:click=move |_| {
                                    app.reset_game();
                                    app.view.set(None);
                                }>"New game"</button>
                            </div>
                        </>
                    },
                )
            }}
        </div>
    }
}

#[component]
fn GameOver() -> impl IntoView {
    let app = expect_context::<App>();
    // A drive that ended: who scored, and a button for the next one — the same spec, so a
    // drive with no pinned position seed draws a fresh position, and a pinned one replays it.
    let drive = move || {
        app.drive_over.get().map(|(attacker, scored, home, away)| {
            let next = move |_| {
                let Some(spec) = app.spec.get_untracked() else { return };
                app.reset_game();
                app.spec.set(Some(spec.clone()));
                ws::send(&ClientMsg::NewGame(spec));
            };
            let replay = matches!(
                app.spec.get_untracked().map(|s| s.start),
                Some(StartFrom::RandomDrive { seed: Some(_) })
            );
            view! {
                <div class="gameover">
                    <span class="result">
                        {match scored {
                            Some(t) if t == attacker => format!("{t:?} scored"),
                            Some(t) => format!("{t:?} scored against the drive"),
                            None => format!("{attacker:?}'s drive ended without a score"),
                        }}
                    </span>
                    <span class="score">{format!("{home} — {away}")}</span>
                    <button class="next-drive" on:click=next>
                        {if replay { "Replay this drive" } else { "Next drive" }}
                    </button>
                </div>
            }
        })
    };
    let game = move || {
        app.game_over.get().map(|(winner, home, away)| {
            view! {
                <div class="gameover">
                    <span class="result">
                        {match winner {
                            Some(t) => format!("{t:?} wins"),
                            None => "Draw".to_string(),
                        }}
                    </span>
                    <span class="score">{format!("{home} — {away}")}</span>
                </div>
            }
        })
    };
    view! {
        {drive}
        {game}
    }
}

/// `1`-`5` pick an overlay, `T` cycles the tackle-zone layer, `Z` undoes,
/// `E`/`Enter` ends the turn, `Esc` closes a menu, `→` steps the bot.
fn keyboard_shortcuts(app: App) {
    let handle = window_event_listener(leptos::ev::keydown, move |ev| {
        // Never steal a key from a text field in the lobby.
        let key = ev.key();
        match key.as_str() {
            "Escape" => app.menu.set(None),
            // One key for both halves of the gesture: it takes the held step,
            // and on a session that is running free it puts the brakes on
            // first — otherwise "press → to watch the bot" needs you to have
            // reached for the mode buttons beforehand, which is exactly the
            // moment you have already missed.
            "ArrowRight" => {
                ev.prevent_default();
                if app.paused() {
                    ws::send(&ClientMsg::StepOnce);
                } else {
                    app.step_mode.set(StepMode::Manual);
                    ws::send(&ClientMsg::SetStepMode(StepMode::Manual));
                }
            }
            "z" | "Z" => {
                if app.view.get().is_some_and(|v| v.can_undo) {
                    ws::send(&ClientMsg::Undo);
                }
            }
            "t" | "T" => app.tz.update(|t| *t = t.next()),
            "e" | "E" | "Enter" => {
                if app.my_turn() {
                    let has_end_turn = app.view.get().is_some_and(|v| {
                        v.simple_actions
                            .iter()
                            .any(|a| a.at == botbowl_web_proto::SimpleAT::EndTurn)
                    });
                    if has_end_turn {
                        ws::send(&ClientMsg::Act(Action::Simple(botbowl_web_proto::SimpleAT::EndTurn)));
                    }
                }
            }
            other => {
                if let Some(overlay) = Overlay::ALL.into_iter().find(|o| o.key().to_string() == other) {
                    app.overlay.set(overlay);
                }
            }
        }
    });
    // The listener lives as long as the app does.
    on_cleanup(move || handle.remove());
}

/// Debug controls (phase 3 of plan 034). These exist because the server rolls
/// the dice itself — in `DiceMode::RollDice` the engine resolves rolls
/// internally and nothing outside could pin one.
///
/// The pin is set *ahead* of the roll, because the session never pauses on a
/// roll: it resolves and streams it in the same step. If the pinned value does
/// not fit the roll the engine actually asks for, the server says so and rolls
/// normally rather than forcing an incompatible result into the engine.
#[component]
fn Debug() -> impl IntoView {
    let app = expect_context::<App>();
    let open = RwSignal::new(false);
    let filename = RwSignal::new(String::from("web-game.json"));

    let pin = move |roll: Option<RollResult>| ws::send(&ClientMsg::FixNextRoll(roll));

    view! {
        <div class="debug">
            <button class="drawer-handle" on:click=move |_| open.update(|o| *o = !*o)>
                {move || if open.get() { "▾ debug" } else { "▸ debug" }}
            </button>
            <Show when=move || open.get()>
                <div class="debug-body">
                    // Plan 043: the net's read of the position *right now*. Unlike the
                    // inspector's `net favours …`, which is a by-product of the bot's last
                    // search, this updates on every board change — so it answers "who does the
                    // network think scores next" during your own turn.
                    <div class="valuation">
                        <span class="label">"Net valuation"</span>
                        <span class="value">
                            {move || match app.net_now.get() {
                                Some(n) => format!("favours {} ({})", crate::inspector::favours(n.value_home), n.model),
                                None => "— (no network in play)".to_string(),
                            }}
                        </span>
                    </div>
                    <div class="pin">
                        <span class="label">"Pin next roll"</span>
                        <div class="pin-buttons">
                            <button on:click=move |_| pin(Some(RollResult::Pass))>"Pass"</button>
                            <button on:click=move |_| pin(Some(RollResult::Fail))>"Fail"</button>
                            {(1u8..=6)
                                .map(|v| {
                                    view! {
                                        <button on:click=move |_| pin(Some(RollResult::D6 { value: v }))>
                                            {v.to_string()}
                                        </button>
                                    }
                                })
                                .collect_view()}
                            {[
                                BlockDice::Skull,
                                BlockDice::BothDown,
                                BlockDice::Push,
                                BlockDice::PowPush,
                                BlockDice::Pow,
                            ]
                                .into_iter()
                                .map(|face| {
                                    view! {
                                        <button
                                            title=face.label()
                                            on:click=move |_| {
                                                pin(Some(RollResult::BlockDice { faces: vec![face] }))
                                            }
                                        >
                                            <img src=format!("img/{}", face.img()) alt=face.label() />
                                        </button>
                                    }
                                })
                                .collect_view()}
                        </div>
                        {move || {
                            app.pinned
                                .get()
                                .map(|roll| {
                                    view! {
                                        <div class="pinned">
                                            {format!("pinned: {roll:?}")}
                                            <button on:click=move |_| pin(None)>"clear"</button>
                                        </div>
                                    }
                                })
                        }}
                    </div>
                    <div class="save">
                        <span class="label">"Save recording"</span>
                        <input
                            type="text"
                            prop:value=move || filename.get()
                            on:input=move |ev| filename.set(event_target_value(&ev))
                        />
                        <button on:click=move |_| {
                            ws::send(&ClientMsg::SaveRecording { path: filename.get() })
                        }>"Save"</button>
                        {move || {
                            app.saved
                                .get()
                                .map(|path| view! { <div class="hint">{format!("wrote {path}")}</div> })
                        }}
                        <p class="hint">
                            "opens in `cargo run -p botbowl-ui -- replay <file>`"
                        </p>
                    </div>
                </div>
            </Show>
        </div>
    }
}
