
extern crate rust_bb;
use ansi_term::Colour::Red;
use itertools::Either;
use rust_bb::core::dices::BlockDice;
use rust_bb::core::dices::Coin;
use rust_bb::core::dices::D6Target;
use rust_bb::core::dices::D6;
use rust_bb::core::dices::D8;
use rust_bb::core::gamestate;
use rust_bb::core::model::*;
use rust_bb::core::pathing::CustomIntoIter;
use rust_bb::core::pathing::NodeIteratorItem;
use rust_bb::core::table::*;
use rust_bb::core::{
    gamestate::{GameState, GameStateBuilder},
    model::{Action, DugoutPlace, PlayerStats, Position, TeamType, HEIGHT_, WIDTH_},
    pathing::{PathFinder, PathingEvent},
    table::PosAT,
};
use rust_bb::standard_state;
use std::{
    collections::{HashMap, HashSet},
    iter::{repeat_with, zip},
};
