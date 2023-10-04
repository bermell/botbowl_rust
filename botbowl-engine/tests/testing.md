
extern crate botbowl_engine;
use ansi_term::Colour::Red;
use itertools::Either;
use botbowl_engine::core::dices::BlockDice;
use botbowl_engine::core::dices::Coin;
use botbowl_engine::core::dices::D6Target;
use botbowl_engine::core::dices::D6;
use botbowl_engine::core::dices::D8;
use botbowl_engine::core::gamestate;
use botbowl_engine::core::model::*;
use botbowl_engine::core::pathing::CustomIntoIter;
use botbowl_engine::core::pathing::NodeIteratorItem;
use botbowl_engine::core::table::*;
use botbowl_engine::core::{
    gamestate::{GameState, GameStateBuilder},
    model::{Action, DugoutPlace, PlayerStats, Position, TeamType, HEIGHT_, WIDTH_},
    pathing::{PathFinder, PathingEvent},
    table::PosAT,
};
use botbowl_engine::standard_state;
use std::{
    collections::{HashMap, HashSet},
    iter::{repeat_with, zip},
};
