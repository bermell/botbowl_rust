use std::{error, fmt};

use crate::core::model;
use model::*;

#[derive(Debug, Clone, Copy)]
pub struct InvalidPlayerId {
    pub id: PlayerID,
}
impl error::Error for InvalidPlayerId {}
impl fmt::Display for InvalidPlayerId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Not valid PlayerId: {}", self.id)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IllegalMovePosition {
    pub position: Position,
}

impl error::Error for IllegalMovePosition {}
impl fmt::Display for IllegalMovePosition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Can't move peice to already occupied position: {:?}",
            self.position
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EmptyProcStackError;
impl error::Error for EmptyProcStackError {}
impl fmt::Display for EmptyProcStackError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GameState procstack unexpectidly empty")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IllegalActionError {
    pub action: Action,
}

impl error::Error for IllegalActionError {}
impl fmt::Display for IllegalActionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Action is not allowed: {:?}", self.action)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MissingActionError {}

impl error::Error for MissingActionError {}
impl fmt::Display for MissingActionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Action is missing")
    }
}
