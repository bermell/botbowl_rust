//! The lobby: pick a board, then who plays each side.
//!
//! Either side can be you, the random bot, or MCTS on a net (value and priors
//! both from the net — the only search the web app offers). Two bots against
//! each other is how you watch two nets, or two budgets, play.
//!
//! Everything offered here comes from the server's `LobbyInfo` — the board
//! presets it can actually run at its compiled capacity, and the models it
//! found on disk. Model choice is filtered by the `_WxH_` filename tag,
//! because a net trained on another board size panics inside the evaluator
//! rather than erroring.

use botbowl_web_proto::msg::{BotSpec, Budget, ClientMsg, GameSpec, MctsSpec, ModelInfo, Seat, StartFrom};
use leptos::prelude::*;

use crate::state::App;
use crate::ws;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Human,
    Random,
    Mcts,
}

/// One side's choices, kept as signals so the form survives re-renders.
#[derive(Clone, Copy)]
struct SeatForm {
    kind: RwSignal<Kind>,
    /// The model path picked in the dropdown; empty (or no longer on offer
    /// after a board change) means "the first one that fits".
    model: RwSignal<String>,
    use_millis: RwSignal<bool>,
    iterations: RwSignal<usize>,
    millis: RwSignal<u64>,
    workers: RwSignal<usize>,
}

impl SeatForm {
    fn new(kind: Kind) -> Self {
        SeatForm {
            kind: RwSignal::new(kind),
            model: RwSignal::new(String::new()),
            use_millis: RwSignal::new(false),
            iterations: RwSignal::new(2000),
            millis: RwSignal::new(1000),
            workers: RwSignal::new(0),
        }
    }

    /// Take a seat the server proposed.
    fn load(&self, seat: &Seat) {
        match seat {
            Seat::Human => self.kind.set(Kind::Human),
            Seat::Bot(BotSpec::Random) => self.kind.set(Kind::Random),
            Seat::Bot(BotSpec::Mcts(m)) => {
                self.kind.set(Kind::Mcts);
                self.model.set(m.model.clone());
            }
        }
    }

    /// The model this seat would load: the one picked, if it still fits the
    /// board, else the newest that does.
    fn effective_model(&self, available: &[ModelInfo]) -> Option<String> {
        let picked = self.model.get();
        available
            .iter()
            .find(|m| m.path == picked)
            .or_else(|| available.first())
            .map(|m| m.path.clone())
    }

    fn seat(&self, available: &[ModelInfo]) -> Seat {
        match self.kind.get() {
            Kind::Human => Seat::Human,
            Kind::Random => Seat::Bot(BotSpec::Random),
            Kind::Mcts => Seat::Bot(BotSpec::Mcts(MctsSpec {
                budget: if self.use_millis.get() {
                    Budget::Millis(self.millis.get())
                } else {
                    Budget::Iterations(self.iterations.get())
                },
                model: self.effective_model(available).unwrap_or_default(),
                workers: (self.workers.get() > 0).then(|| self.workers.get()),
                ..Default::default()
            })),
        }
    }
}

#[component]
pub fn Lobby() -> impl IntoView {
    let app = expect_context::<App>();

    let board_index = RwSignal::new(0usize);
    let home = SeatForm::new(Kind::Human);
    let away = SeatForm::new(Kind::Mcts);
    let seed = RwSignal::new(String::new());
    let recording = RwSignal::new(String::new());
    let step = RwSignal::new(0usize);

    // The board the form is currently on, defaulting to the server's own
    // suggestion (14x7 where it fits).
    let boards = move || app.lobby.get().map(|l| l.boards.clone()).unwrap_or_default();
    let board = move || boards().get(board_index.get()).copied();

    // Only models whose filename tag matches; an untagged model is offered
    // everywhere because we cannot know what it was trained on.
    let models = Signal::derive(move || -> Vec<ModelInfo> {
        let Some(board) = board() else { return Vec::new() };
        app.lobby
            .get()
            .map(|l| l.models.into_iter().filter(|m| m.fits(board)).collect())
            .unwrap_or_default()
    });

    // Default the form to whatever the server proposed.
    Effect::new(move |_| {
        if let Some(lobby) = app.lobby.get() {
            if let Some(i) = lobby.boards.iter().position(|b| *b == lobby.defaults.board) {
                board_index.set(i);
            }
            home.load(&lobby.defaults.home);
            away.load(&lobby.defaults.away);
        }
    });

    // An MCTS seat with no net to load cannot start.
    let blocked = move || {
        let none = models.get().is_empty();
        board().is_none() || (none && (home.kind.get() == Kind::Mcts || away.kind.get() == Kind::Mcts))
    };

    let start = move |_| {
        let Some(board) = board() else { return };
        let available = models.get();
        let spec = GameSpec {
            board,
            home: home.seat(&available),
            away: away.seat(&available),
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
                "Play the net-guided search, or watch two bots play each other, and see why every move was made."
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

            <div class="seats">
                <SeatPicker title="Home" note="attacks left, sets up second" form=home models=models />
                <SeatPicker title="Away" note="attacks right, kicks first" form=away models=models />
            </div>

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
                    <span class="hint">"seeds the dice and the random bot, not the search"</span>
                </label>
            </section>

            <button class="start" on:click=start disabled=blocked>
                "Kick off"
            </button>

            <footer>{move || app.lobby.get().map(|l| l.server).unwrap_or_default()}</footer>
        </div>
    }
}

/// Who plays one side, and — for MCTS — on which net and at what budget.
#[component]
fn SeatPicker(
    title: &'static str,
    note: &'static str,
    form: SeatForm,
    models: Signal<Vec<ModelInfo>>,
) -> impl IntoView {
    let kind = form.kind;
    let choice = move |k: Kind, big: &'static str, small: &'static str| {
        view! {
            <button class="choice" class:on=move || kind.get() == k on:click=move |_| kind.set(k)>
                <span class="big">{big}</span>
                <span class="small">{small}</span>
            </button>
        }
    };

    view! {
        <section class="seat-picker">
            <h2>{title} <span class="hint">{note}</span></h2>
            <div class="choices">
                {choice(Kind::Human, "You", "play it from this browser")}
                {choice(Kind::Random, "Random", "uniform over legal actions")}
                {choice(Kind::Mcts, "MCTS", "net value + priors, with an inspector")}
            </div>
            <Show when=move || kind.get() == Kind::Mcts>
                <div class="knobs">
                    <label>
                        "Model"
                        {move || {
                            let available = models.get();
                            if available.is_empty() {
                                view! { <span class="warn">"no model on this server fits this board"</span> }
                                    .into_any()
                            } else {
                                let current = form.effective_model(&available).unwrap_or_default();
                                view! {
                                    <select on:change=move |ev| form.model.set(event_target_value(&ev))>
                                        {available
                                            .into_iter()
                                            .map(|m| {
                                                let selected = m.path == current;
                                                view! {
                                                    <option value=m.path.clone() selected=selected>
                                                        {m.name}
                                                    </option>
                                                }
                                            })
                                            .collect_view()}
                                    </select>
                                }
                                    .into_any()
                            }
                        }}
                    </label>
                    <label>
                        "Budget"
                        <span class="row">
                            <select on:change=move |ev| form.use_millis.set(event_target_value(&ev) == "ms")>
                                <option value="iters" selected=move || !form.use_millis.get()>
                                    "iterations"
                                </option>
                                <option value="ms" selected=move || form.use_millis.get()>
                                    "milliseconds"
                                </option>
                            </select>
                            {move || {
                                if form.use_millis.get() {
                                    view! {
                                        <input
                                            type="number"
                                            min="1"
                                            prop:value=move || form.millis.get().to_string()
                                            on:input=move |ev| {
                                                if let Ok(v) = event_target_value(&ev).parse() {
                                                    form.millis.set(v)
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
                                            prop:value=move || form.iterations.get().to_string()
                                            on:input=move |ev| {
                                                if let Ok(v) = event_target_value(&ev).parse() {
                                                    form.iterations.set(v)
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
                        "Workers"
                        <input
                            type="number"
                            min="0"
                            prop:value=move || form.workers.get().to_string()
                            on:input=move |ev| form.workers.set(event_target_value(&ev).parse().unwrap_or(0))
                        />
                        <span class="hint">"0 = all cores"</span>
                    </label>
                </div>
            </Show>
        </section>
    }
}
