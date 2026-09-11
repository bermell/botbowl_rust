//! The lobby: pick a board, a side, and an opponent.
//!
//! Everything offered here comes from the server's `LobbyInfo` — the board
//! presets it can actually run at its compiled capacity, and the models it
//! found on disk. Model choice is filtered by the `_WxH_` filename tag,
//! because a net trained on another board size panics inside the evaluator
//! rather than erroring.

use botbowl_web_proto::msg::{BotSpec, Budget, ClientMsg, EvaluatorSpec, GameSpec, MctsSpec, ModelInfo, StartFrom};
use botbowl_web_proto::TeamType;
use leptos::prelude::*;

use crate::state::App;
use crate::ws;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Random,
    Scripted,
    Mcts,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Eval {
    Heuristic,
    PureTd,
    Nn,
    NnValue,
}

#[component]
pub fn Lobby() -> impl IntoView {
    let app = expect_context::<App>();

    let board_index = RwSignal::new(0usize);
    let play_home = RwSignal::new(true);
    let kind = RwSignal::new(Kind::Scripted);
    let eval = RwSignal::new(Eval::Heuristic);
    let model = RwSignal::new(String::new());
    let use_millis = RwSignal::new(false);
    let iterations = RwSignal::new(2000usize);
    let millis = RwSignal::new(1000u64);
    let workers = RwSignal::new(0usize);
    let seed = RwSignal::new(String::new());
    let recording = RwSignal::new(String::new());
    let step = RwSignal::new(0usize);

    // The board the form is currently on, defaulting to the server's own
    // suggestion (14x7 where it fits).
    let boards = move || app.lobby.get().map(|l| l.boards.clone()).unwrap_or_default();
    let board = move || boards().get(board_index.get()).copied();

    // Only models whose filename tag matches; an untagged model is offered
    // everywhere because we cannot know what it was trained on.
    let models = move || -> Vec<ModelInfo> {
        let Some(board) = board() else { return Vec::new() };
        app.lobby
            .get()
            .map(|l| {
                l.models
                    .into_iter()
                    .filter(|m| m.board_tag.as_deref().is_none_or(|t| t == board.tag()))
                    .collect()
            })
            .unwrap_or_default()
    };

    // Default the board selection to whatever the server proposed.
    Effect::new(move |_| {
        if let Some(lobby) = app.lobby.get() {
            if let Some(i) = lobby.boards.iter().position(|b| *b == lobby.defaults.board) {
                board_index.set(i);
            }
        }
    });

    let start = move |_| {
        let Some(board) = board() else { return };
        let bot = match kind.get() {
            Kind::Random => BotSpec::Random,
            Kind::Scripted => BotSpec::Scripted,
            Kind::Mcts => {
                let chosen = model.get();
                let chosen = if chosen.is_empty() {
                    models().first().map(|m| m.path.clone()).unwrap_or_default()
                } else {
                    chosen
                };
                let evaluator = match eval.get() {
                    Eval::Heuristic => EvaluatorSpec::Heuristic,
                    Eval::PureTd => EvaluatorSpec::PureTd,
                    Eval::Nn => EvaluatorSpec::Nn { model: chosen },
                    Eval::NnValue => EvaluatorSpec::NnValue { model: chosen },
                };
                BotSpec::Mcts(MctsSpec {
                    budget: if use_millis.get() {
                        Budget::Millis(millis.get())
                    } else {
                        Budget::Iterations(iterations.get())
                    },
                    evaluator,
                    workers: (workers.get() > 0).then(|| workers.get()),
                    ..Default::default()
                })
            }
        };
        let spec = GameSpec {
            board,
            human: if play_home.get() {
                TeamType::Home
            } else {
                TeamType::Away
            },
            bot,
            seed: seed.get().trim().parse::<u64>().ok(),
            start: match recording.get().trim() {
                "" => StartFrom::CoinToss,
                path => StartFrom::Recording {
                    path: path.to_string(),
                    step: step.get(),
                },
            },
        };
        app.reset_game();
        app.spec.set(Some(spec.clone()));
        ws::send(&ClientMsg::NewGame(spec));
    };

    view! {
        <div class="lobby">
            <h1>"Blood Bowl"</h1>
            <p class="subtitle">
                "Play a full game against any bot this build can make, and see why it moved."
            </p>

            <section>
                <h2>"Board"</h2>
                <div class="choices">
                    {move || {
                        boards()
                            .into_iter()
                            .enumerate()
                            .map(|(i, b)| {
                                view! {
                                    <button
                                        class="choice"
                                        class:on=move || board_index.get() == i
                                        on:click=move |_| board_index.set(i)
                                    >
                                        <span class="big">{format!("{}x{}", b.width, b.height)}</span>
                                        <span class="small">{format!("{} a side", b.team_size)}</span>
                                    </button>
                                }
                            })
                            .collect_view()
                    }}
                </div>
            </section>

            <section>
                <h2>"You play"</h2>
                <div class="choices">
                    <button class="choice" class:on=move || play_home.get() on:click=move |_| play_home.set(true)>
                        <span class="big">"Home"</span>
                        <span class="small">"attacks left, sets up second"</span>
                    </button>
                    <button class="choice" class:on=move || !play_home.get() on:click=move |_| play_home.set(false)>
                        <span class="big">"Away"</span>
                        <span class="small">"attacks right, kicks first"</span>
                    </button>
                </div>
            </section>

            <section>
                <h2>"Opponent"</h2>
                <div class="choices">
                    <button class="choice" class:on=move || kind.get() == Kind::Random on:click=move |_| kind.set(Kind::Random)>
                        <span class="big">"Random"</span>
                        <span class="small">"uniform over legal actions"</span>
                    </button>
                    <button class="choice" class:on=move || kind.get() == Kind::Scripted on:click=move |_| kind.set(Kind::Scripted)>
                        <span class="big">"Scripted"</span>
                        <span class="small">"the heuristic baseline"</span>
                    </button>
                    <button class="choice" class:on=move || kind.get() == Kind::Mcts on:click=move |_| kind.set(Kind::Mcts)>
                        <span class="big">"MCTS"</span>
                        <span class="small">"search, with an inspector"</span>
                    </button>
                </div>
            </section>

            <Show when=move || kind.get() == Kind::Mcts>
                <section class="knobs">
                    <h2>"Search"</h2>
                    <label>
                        "Budget"
                        <span class="row">
                            <select on:change=move |ev| use_millis.set(event_target_value(&ev) == "ms")>
                                <option value="iters">"iterations"</option>
                                <option value="ms">"milliseconds"</option>
                            </select>
                            {move || {
                                if use_millis.get() {
                                    view! {
                                        <input
                                            type="number"
                                            min="1"
                                            prop:value=move || millis.get().to_string()
                                            on:input=move |ev| {
                                                if let Ok(v) = event_target_value(&ev).parse() {
                                                    millis.set(v)
                                                }
                                            }
                                        />
                                    }
                                        .into_any()
                                } else {
                                    view! {
                                        <input
                                            type="number"
                                            min="1"
                                            prop:value=move || iterations.get().to_string()
                                            on:input=move |ev| {
                                                if let Ok(v) = event_target_value(&ev).parse() {
                                                    iterations.set(v)
                                                }
                                            }
                                        />
                                    }
                                        .into_any()
                                }
                            }}
                        </span>
                    </label>

                    <label>
                        "Evaluator"
                        <select on:change=move |ev| {
                            eval.set(match event_target_value(&ev).as_str() {
                                "pure-td" => Eval::PureTd,
                                "nn" => Eval::Nn,
                                "nn-value" => Eval::NnValue,
                                _ => Eval::Heuristic,
                            })
                        }>
                            <option value="heuristic">"heuristic (scripted value + priors)"</option>
                            <option value="pure-td">"pure TD (unshaped value)"</option>
                            <option value="nn">"NN (value + priors)"</option>
                            <option value="nn-value">"NN value, scripted priors"</option>
                        </select>
                    </label>

                    <Show when=move || matches!(eval.get(), Eval::Nn | Eval::NnValue)>
                        <label>
                            "Model"
                            {move || {
                                let available = models();
                                if available.is_empty() {
                                    view! {
                                        <span class="warn">
                                            "no model on this server matches "
                                            {board().map(|b| b.tag()).unwrap_or_default()}
                                        </span>
                                    }
                                        .into_any()
                                } else {
                                    view! {
                                        <select on:change=move |ev| model.set(event_target_value(&ev))>
                                            {available
                                                .into_iter()
                                                .map(|m| {
                                                    view! { <option value=m.path.clone()>{m.name}</option> }
                                                })
                                                .collect_view()}
                                        </select>
                                    }
                                        .into_any()
                                }
                            }}
                        </label>
                    </Show>

                    <label>
                        "Workers"
                        <input
                            type="number"
                            min="0"
                            prop:value=move || workers.get().to_string()
                            on:input=move |ev| { workers.set(event_target_value(&ev).parse().unwrap_or(0)) }
                        />
                        <span class="hint">"0 = all cores"</span>
                    </label>
                </section>
            </Show>

            <section class="knobs">
                <label>
                    "Resume"
                    <input
                        type="text"
                        placeholder="path to a recording (optional)"
                        prop:value=move || recording.get()
                        on:input=move |ev| recording.set(event_target_value(&ev))
                    />
                    <input
                        type="number"
                        min="0"
                        style="width: 90px"
                        prop:value=move || step.get().to_string()
                        on:input=move |ev| step.set(event_target_value(&ev).parse().unwrap_or(0))
                    />
                    <span class="hint">"a botbowl-ui replay file, and the micro-step to resume at"</span>
                </label>
                <label>
                    "Seed"
                    <input
                        type="text"
                        placeholder="random"
                        prop:value=move || seed.get()
                        on:input=move |ev| seed.set(event_target_value(&ev))
                    />
                    <span class="hint">"seeds the dice, not the search"</span>
                </label>
            </section>

            <button class="start" on:click=start disabled=move || board().is_none()>
                "Kick off"
            </button>

            <footer>{move || app.lobby.get().map(|l| l.server).unwrap_or_default()}</footer>
        </div>
    }
}
