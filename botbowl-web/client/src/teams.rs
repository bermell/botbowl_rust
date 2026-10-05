//! The team editor: pick a team, change its positions' stats, skills and pictures, and save it
//! on the server (`~/.config/botbowl/teams/<name>.json`).
//!
//! Built-in teams are read-only; editing one and saving under a new name is how a custom team
//! starts. The draft is one signal holding a whole `TeamDef`. Inputs commit on `change` rather
//! than `input`, because every commit re-renders the position rows and an `input` handler
//! would take the focus out of the field on each keystroke.

use botbowl_web_proto::msg::ClientMsg;
use botbowl_web_proto::team::{self, PositionDef, TeamDef, DEFAULT_TEAM};
use botbowl_web_proto::view::PlayerRole;
use botbowl_web_proto::TeamType;
use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::state::{App, Screen};
use crate::ws;

const ROLES: [PlayerRole; 4] = [
    PlayerRole::Lineman,
    PlayerRole::Blitzer,
    PlayerRole::Thrower,
    PlayerRole::Catcher,
];

/// The `<img src>` for a picture, drawn in the away colourway, not yet acted.
fn picture_src(picture: &str) -> String {
    format!("img/{}", team::picture_sprite(picture, TeamType::Away, false))
}

fn blank_position() -> PositionDef {
    PositionDef {
        name: "Lineman".into(),
        role: PlayerRole::Lineman,
        ma: 6,
        st: 3,
        ag: 3,
        pa: 4,
        av: 8,
        skills: Vec::new(),
        picture: "hlineman1".into(),
        max: 16,
    }
}

#[component]
pub fn TeamEditor() -> impl IntoView {
    let app = expect_context::<App>();
    let first = app
        .teams
        .get_untracked()
        .into_iter()
        .find(|t| t.name == DEFAULT_TEAM)
        .or_else(|| app.teams.get_untracked().into_iter().next());
    let draft = RwSignal::new(first.clone().unwrap_or_else(|| TeamDef {
        name: "New team".into(),
        positions: vec![blank_position()],
        builtin: false,
    }));
    // The saved team the draft was loaded from, if any.
    let source = RwSignal::new(first.map(|t| t.name));
    // The position whose picture is being chosen.
    let picking = RwSignal::new(None::<usize>);
    let can_save = move || app.lobby.get().is_some_and(|l| l.can_save_teams);

    // An upload finished: put it on the position it was for.
    Effect::new(move |_| {
        if let Some(path) = app.uploaded.get() {
            if let Some(i) = picking.get_untracked() {
                draft.update(|d| {
                    if let Some(p) = d.positions.get_mut(i) {
                        p.picture = path.clone();
                    }
                });
            }
            app.uploaded.set(None);
            picking.set(None);
        }
    });

    let load = move |t: TeamDef| {
        source.set(Some(t.name.clone()));
        draft.set(t);
        app.notice.set(None);
    };

    let clashes_with_builtin = move || {
        let name = draft.with(|d| team::slug(&d.name));
        team::builtin_teams().iter().any(|b| team::slug(&b.name) == name)
    };

    let save = move |_| {
        let mut def = draft.get();
        def.builtin = false;
        source.set(Some(def.name.clone()));
        ws::send(&ClientMsg::SaveTeam(def));
    };

    let delete = move |_| {
        let name = draft.with(|d| d.name.clone());
        ws::send(&ClientMsg::DeleteTeam { name });
        if let Some(t) = app.teams.get_untracked().into_iter().next() {
            load(t);
        }
    };

    view! {
        <div class="lobby teams">
            <h1>"Teams"</h1>
            <p class="subtitle">
                "Built-in rosters are read-only — change one and save it under a new name. Saved teams live on the server in "
                <code>"~/.config/botbowl/teams/"</code> "."
            </p>

            <section>
                <h2>"Team"</h2>
                <div class="choices">
                    {move || {
                        app.teams
                            .get()
                            .into_iter()
                            .map(|t| {
                                let name = t.name.clone();
                                let builtin = t.builtin;
                                let looks = t.positions.first().map(|p| p.picture.clone()).unwrap_or_default();
                                view! {
                                    <button
                                        class="choice team-choice"
                                        class:on=move || source.get().as_deref() == Some(name.as_str())
                                        on:click=move |_| load(t.clone())
                                    >
                                        <img src=picture_src(&looks) alt="" />
                                        <span class="big">{t.name.clone()}</span>
                                        <span class="small">{if builtin { "built-in" } else { "saved" }}</span>
                                    </button>
                                }
                            })
                            .collect_view()
                    }}
                    <button
                        class="choice"
                        on:click=move |_| {
                            source.set(None);
                            draft.set(TeamDef {
                                name: "New team".into(),
                                positions: vec![blank_position()],
                                builtin: false,
                            });
                        }
                    >
                        <span class="big">"+ New"</span>
                        <span class="small">"from scratch"</span>
                    </button>
                </div>
            </section>

            <section class="knobs">
                <label>
                    "Name"
                    <input
                        type="text"
                        prop:value=move || draft.with(|d| d.name.clone())
                        on:change=move |ev| draft.update(|d| d.name = event_target_value(&ev))
                    />
                    <Show when=clashes_with_builtin>
                        <span class="warn">"a built-in's name — rename to save"</span>
                    </Show>
                </label>
            </section>

            <section>
                <h2>
                    "Positions"
                    <span class="hint">
                        " the first is the filler: it takes every slot the others leave. AG is the engine's (roll 7−AG); PA is the pass target."
                    </span>
                </h2>
                <table class="positions">
                    <thead>
                        <tr>
                            <th>"Picture"</th>
                            <th>"Position"</th>
                            <th>"Role"</th>
                            <th>"MA"</th>
                            <th>"ST"</th>
                            <th>"AG"</th>
                            <th>"PA"</th>
                            <th>"AV"</th>
                            <th>"Max"</th>
                            <th>"Skills"</th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody>
                        {move || {
                            draft
                                .get()
                                .positions
                                .into_iter()
                                .enumerate()
                                .map(|(i, p)| view! { <PositionRow i=i p=p draft=draft picking=picking /> })
                                .collect_view()
                        }}
                    </tbody>
                </table>
                <button
                    class="link"
                    on:click=move |_| draft.update(|d| d.positions.push(blank_position()))
                >
                    "+ Add position"
                </button>
            </section>

            <div class="team-actions">
                <button
                    class="start"
                    on:click=save
                    disabled=move || !can_save() || clashes_with_builtin() || draft.with(|d| d.positions.is_empty())
                >
                    "Save team"
                </button>
                <Show when=move || {
                    let name = draft.with(|d| d.name.clone());
                    app.teams.get().iter().any(|t| t.name == name && !t.builtin)
                }>
                    <button class="danger" on:click=delete>
                        "Delete"
                    </button>
                </Show>
                <button on:click=move |_| app.screen.set(Screen::Lobby)>"Back to the lobby"</button>
                <span class="hint">
                    {move || {
                        if !can_save() {
                            "this server has no team directory, so nothing can be saved".to_string()
                        } else {
                            app.notice.get().unwrap_or_default()
                        }
                    }}
                </span>
            </div>

            <PicturePicker draft=draft picking=picking />
        </div>
    }
}

/// One stat cell: a number input committing on change, clamped to `lo..=hi`.
fn stat_cell(
    draft: RwSignal<TeamDef>,
    i: usize,
    value: u8,
    lo: u8,
    hi: u8,
    set: fn(&mut PositionDef, u8),
) -> impl IntoView {
    view! {
        <td>
            <input
                type="number"
                class="stat"
                min=lo.to_string()
                max=hi.to_string()
                prop:value=value.to_string()
                on:change=move |ev| {
                    if let Ok(v) = event_target_value(&ev).trim().parse::<u8>() {
                        draft.update(|d| {
                            if let Some(p) = d.positions.get_mut(i) {
                                set(p, v.clamp(lo, hi));
                            }
                        });
                    }
                }
            />
        </td>
    }
}

#[component]
fn PositionRow(i: usize, p: PositionDef, draft: RwSignal<TeamDef>, picking: RwSignal<Option<usize>>) -> impl IntoView {
    let app = expect_context::<App>();
    let edit = move |f: Box<dyn FnOnce(&mut PositionDef)>| {
        draft.update(|d| {
            if let Some(p) = d.positions.get_mut(i) {
                f(p)
            }
        })
    };
    let role = p.role;
    let skills = p.skills.clone();
    view! {
        <tr>
            <td>
                <button class="picture" title="choose a picture" on:click=move |_| picking.set(Some(i))>
                    <img src=picture_src(&p.picture) alt="" />
                </button>
            </td>
            <td>
                <input
                    type="text"
                    class="pos-name"
                    prop:value=p.name.clone()
                    on:change=move |ev| {
                        let v = event_target_value(&ev);
                        edit(Box::new(move |p| p.name = v))
                    }
                />
            </td>
            <td>
                <select on:change=move |ev| {
                    let v = event_target_value(&ev);
                    if let Some(r) = ROLES.into_iter().find(|r| r.label() == v) {
                        edit(Box::new(move |p| p.role = r))
                    }
                }>
                    {ROLES
                        .into_iter()
                        .map(|r| view! { <option value=r.label() selected=r == role>{r.label()}</option> })
                        .collect_view()}
                </select>
            </td>
            {stat_cell(draft, i, p.ma, 1, team::MAX_MA, |p, v| p.ma = v)}
            {stat_cell(draft, i, p.st, 1, team::MAX_ST, |p, v| p.st = v)}
            {stat_cell(draft, i, p.ag, 1, team::MAX_AG, |p, v| p.ag = v)}
            {stat_cell(draft, i, p.pa, 2, 6, |p, v| p.pa = v)}
            {stat_cell(draft, i, p.av, 3, team::MAX_AV, |p, v| p.av = v)}
            {stat_cell(draft, i, p.max, 0, 16, |p, v| p.max = v)}
            <td class="skills">
                {skills
                    .iter()
                    .cloned()
                    .map(|s| {
                        let label = s.clone();
                        view! {
                            <span class="skill-chip">
                                {s.clone()}
                                <button
                                    title="remove"
                                    on:click=move |_| {
                                        let label = label.clone();
                                        edit(Box::new(move |p| p.skills.retain(|x| *x != label)))
                                    }
                                >
                                    "×"
                                </button>
                            </span>
                        }
                    })
                    .collect_view()}
                <select
                    class="add-skill"
                    on:change=move |ev| {
                        let v = event_target_value(&ev);
                        if !v.is_empty() {
                            edit(Box::new(move |p| {
                                if !p.skills.contains(&v) {
                                    p.skills.push(v);
                                    p.skills.sort();
                                }
                            }))
                        }
                    }
                >
                    <option value="" selected=true>
                        "+ skill"
                    </option>
                    {move || {
                        app.lobby
                            .get()
                            .map(|l| l.skills)
                            .unwrap_or_default()
                            .into_iter()
                            .filter(|s| !skills.contains(&s.label))
                            .map(|s| {
                                let text = if s.implemented {
                                    s.label.clone()
                                } else {
                                    format!("{} (no rules yet)", s.label)
                                };
                                view! { <option value=s.label.clone()>{text}</option> }
                            })
                            .collect_view()
                    }}
                </select>
            </td>
            <td>
                <button
                    class="link danger"
                    title="remove this position"
                    on:click=move |_| draft.update(|d| {
                        if d.positions.len() > 1 {
                            d.positions.remove(i);
                        }
                    })
                >
                    "remove"
                </button>
            </td>
        </tr>
    }
}

/// A modal grid of every picture the server offers, plus an upload.
#[component]
fn PicturePicker(draft: RwSignal<TeamDef>, picking: RwSignal<Option<usize>>) -> impl IntoView {
    let app = expect_context::<App>();
    let filter = RwSignal::new(String::new());
    let choose = move |picture: String| {
        if let Some(i) = picking.get_untracked() {
            draft.update(|d| {
                if let Some(p) = d.positions.get_mut(i) {
                    p.picture = picture;
                }
            });
        }
        picking.set(None);
    };

    let upload = move |ev: leptos::ev::Event| {
        let Some(input) = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()) else {
            return;
        };
        let Some(file) = input.files().and_then(|f| f.get(0)) else {
            return;
        };
        let Ok(reader) = web_sys::FileReader::new() else { return };
        let reader_in = reader.clone();
        let on_load = Closure::<dyn FnMut()>::new(move || {
            if let Some(data_url) = reader_in.result().ok().and_then(|r| r.as_string()) {
                ws::send(&ClientMsg::UploadPicture { data_url });
            }
        });
        reader.set_onload(Some(on_load.as_ref().unchecked_ref()));
        on_load.forget();
        let _ = reader.read_as_data_url(&file);
    };

    move || {
        picking.get().map(|_| {
            view! {
                <div class="modal-backdrop" on:click=move |_| picking.set(None)>
                    <div class="modal" on:click=|ev| ev.stop_propagation()>
                        <div class="modal-head">
                            <input
                                type="text"
                                placeholder="filter (e.g. orc, elf, skaven: sk)"
                                prop:value=move || filter.get()
                                on:input=move |ev| filter.set(event_target_value(&ev))
                            />
                            <label class="upload">
                                "Upload…"
                                <input type="file" accept="image/png,image/gif,image/jpeg,image/webp" on:change=upload />
                            </label>
                            <button on:click=move |_| picking.set(None)>"Close"</button>
                        </div>
                        <div class="picture-grid">
                            {move || {
                                let f = filter.get().to_lowercase();
                                app.pictures
                                    .get()
                                    .into_iter()
                                    .filter(|p| f.is_empty() || p.to_lowercase().contains(&f))
                                    .map(|p| {
                                        let title = p.clone();
                                        let chosen = p.clone();
                                        view! {
                                            <button class="picture" title=title on:click=move |_| choose(chosen.clone())>
                                                <img src=picture_src(&p) alt="" loading="lazy" />
                                            </button>
                                        }
                                    })
                                    .collect_view()
                            }}
                        </div>
                    </div>
                </div>
            }
        })
    }
}
