use std::{
    collections::HashSet,
    ops::{Add, Index, IndexMut, Neg},
    usize,
};

use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;

const SIZE: usize = 4;

pub type SqVal = u32;
pub type Grid = [[SqVal; SIZE]; SIZE];

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct Game2048 {
    pub board: Grid,
    pub score: u32,
    pub state: GameState,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub enum GameState {
    Done,
    WaitingForAction,
    WaitingForRandom,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct Coord {
    pub row: usize,
    pub col: usize,
}
impl Coord {
    pub fn all_coords() -> impl Iterator<Item = Coord> {
        (0..SIZE).flat_map(move |row| (0..SIZE).map(move |col| Coord { row, col }))
    }
}

impl Add<Direction> for Coord {
    type Output = Coord;

    fn add(self, other: Direction) -> Coord {
        let Coord { mut row, mut col } = self;
        match other {
            Direction::Up if row == 0 => row = usize::MAX,
            Direction::Up => row -= 1,
            Direction::Down => row += 1,
            Direction::Left if col == 0 => col = usize::MAX,
            Direction::Left => col -= 1,
            Direction::Right => col += 1,
        };
        Coord { row, col }
    }
}
impl Neg for Direction {
    type Output = Direction;

    fn neg(self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

impl Index<Coord> for Game2048 {
    type Output = u32;

    fn index(&self, index: Coord) -> &Self::Output {
        &self.board[index.row][index.col]
    }
}

impl IndexMut<Coord> for Game2048 {
    fn index_mut(&mut self, index: Coord) -> &mut Self::Output {
        &mut self.board[index.row][index.col]
    }
}

#[allow(dead_code)]
impl Game2048 {
    pub fn new_game(coord: Coord, value: u32) -> Game2048 {
        let mut board = [[0; SIZE]; SIZE];
        board[coord.row][coord.col] = value;

        Game2048 {
            board,
            score: 0,
            state: GameState::WaitingForAction,
        }
    }

    fn new_custom(board: Grid) -> Game2048 {
        assert!(board.len() == SIZE);

        Game2048 {
            board,
            score: 0,
            state: GameState::WaitingForAction,
        }
    }

    pub fn available_chance(&self) -> Vec<(Coord, u32, f32)> {
        debug_assert!(self.state == GameState::WaitingForRandom);
        let mut chance_actions = Vec::new();
        for coord in self.empty_squares() {
            for value in 1..=2 {
                chance_actions.push((coord, value));
            }
        }
        if chance_actions.len() == 0 {
            return vec![];
        }
        let len_chance = chance_actions.len();
        // prob is one over the number of empty square
        let prob: f32 = 1.0 / len_chance as f32;

        chance_actions
            .iter()
            .map(|(coord, value)| (*coord, *value, prob))
            .collect()
    }

    pub fn available_action(&self) -> HashSet<Direction> {
        debug_assert!(self.state == GameState::WaitingForAction);
        let mut aa: HashSet<Direction> = HashSet::new();
        let all_dirs = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ]
        .iter()
        .cloned()
        .collect::<HashSet<_>>();

        for coord in self.filled_squares() {
            if aa == all_dirs {
                break;
            }

            for &dir in (&all_dirs - &aa).iter() {
                let next = coord + dir;
                if !self.in_bounds(next) {
                    continue;
                }
                if self[coord] == self[next] || self[next] == 0 {
                    aa.insert(dir);
                }
            }
        }

        aa
    }
    pub fn filled_squares(&self) -> Vec<Coord> {
        Coord::all_coords()
            .filter(|&coord| self[coord] != 0)
            .collect()
    }

    pub fn empty_squares(&self) -> HashSet<Coord> {
        Coord::all_coords()
            .filter(|&coord| self[coord] == 0)
            .collect()
    }
    pub fn in_bounds(&self, coord: Coord) -> bool {
        coord.row < SIZE && coord.col < SIZE
    }
    pub fn step_random(&mut self, coord: Coord, value: u32) {
        self[coord] = value;
        if self.empty_squares().is_empty() {
            self.state = GameState::Done;
        } else {
            self.state = GameState::WaitingForAction;
        }
    }
    pub fn step_action(&mut self, direction: Direction) {
        //validate input
        if self.state != GameState::WaitingForAction {
            panic!("Invalid state in step_action {:?}", self.state);
        }
        let actions = self.available_action();
        if !actions.contains(&direction) {
            panic!("Invalid action");
        }
        self.step_generic(direction);
        self.state = GameState::WaitingForRandom;
    }

    /// Only moves the tiles so they are "packed" together in the direction of the move
    /// does not merge tiles
    fn shift(&mut self, start: Coord, dir: Direction) {
        loop {
            let mut moved = false;
            let mut current = start;
            let mut next = current + dir;
            while self.in_bounds(current) && self.in_bounds(next) {
                if self[current] == 0 && self[next] != 0 {
                    self[current] = self[next];
                    self[next] = 0;
                    moved = true;
                }
                current = next;
                next = next + dir;
            }
            if !moved {
                break;
            }
        }
    }

    fn step_generic(&mut self, direction: Direction) {
        let range_ = 0..SIZE;
        let m = SIZE - 1;
        let start_squares: Vec<Coord> = match direction {
            Direction::Up => range_.map(|i| Coord { row: 0, col: i }).collect(),
            Direction::Down => range_.map(|i| Coord { row: m, col: i }).collect(),
            Direction::Left => range_.map(|i| Coord { row: i, col: 0 }).collect(),
            Direction::Right => range_.map(|i| Coord { row: i, col: m }).collect(),
        };
        let dir = -direction;

        for start in start_squares {
            self.shift(start, dir);
            let mut current = start;
            while self.in_bounds(current) {
                if self[current] == 0 {
                    current = current + dir;
                    continue;
                }

                let next = current + dir;
                if !self.in_bounds(next) {
                    break;
                }
                if self[next] == 0 {
                    current = next;
                    continue;
                }
                if self[current] == self[next] {
                    // Merge the tiles!
                    // increment score with 2 to the power for the value
                    self.score += 2u32.pow(self[current]);
                    self[current] += 1;
                    self[next] = 0;
                    // move other values
                    self.shift(current, dir);
                    // reset current to start
                    current = start;
                } else {
                    current = next;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use rand::Rng;

    use crate::game_2048::Game2048;

    use super::{Coord, Direction, GameState};

    #[test]
    fn available_actions() {
        let game = Game2048::new_custom([
            //comment to force format
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 1],
        ]);
        let aa = game.available_action();
        assert!(aa.contains(&Direction::Up));
        assert!(aa.contains(&Direction::Left));
    }

    #[test]
    fn available_actions_more() {
        let game = Game2048::new_custom([
            //comment to force format
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 1, 1],
        ]);
        let aa = game.available_action();
        assert!(aa.contains(&Direction::Up));
        assert!(aa.contains(&Direction::Left));
        assert!(aa.contains(&Direction::Right));
    }

    #[test]
    fn random_input() {
        let mut game = Game2048::new_game(Coord { row: 2, col: 3 }, 2);
        game.step_action(Direction::Down);
        if game.state != GameState::WaitingForRandom {
            panic!("Invalid state");
        }
        let coords = game.empty_squares();
        assert_eq!(coords.len(), 15);
        // check that all are in bounds
        for coord in coords {
            assert!(game.in_bounds(coord));
        }

        game.step_random(Coord { col: 0, row: 0 }, 2);
        let expected = [
            //commnt
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
        ];
        assert_eq!(game.board, expected);
    }

    #[test]
    fn step_right() {
        let mut game = Game2048::new_custom([
            //comment
            [1, 3, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 2, 3, 2],
        ]);
        game.step_action(Direction::Right);

        let expected = [
            //comment
            [0, 0, 1, 3],
            [0, 0, 1, 2],
            [0, 0, 0, 2],
            [0, 2, 3, 2],
        ];
        assert_eq!(game.board, expected);
    }
    #[test]
    fn step_down() {
        let mut game = Game2048::new_custom([
            //comment
            [0, 0, 1, 3],
            [0, 0, 1, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]);
        game.step_action(Direction::Down);
        let expected = [
            //comment
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 3],
            [0, 0, 2, 2],
        ];
        assert_eq!(game.board, expected);
    }

    #[test]
    fn step_left() {
        let mut game = Game2048::new_custom([
            //comment
            [1, 1, 1, 1],
            [0, 2, 2, 3],
            [0, 5, 0, 5],
            [0, 0, 0, 0],
        ]);
        game.step_action(Direction::Left);
        let expected = [
            //comment
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [6, 0, 0, 0],
            [0, 0, 0, 0],
        ];
        assert_eq!(game.board, expected);
    }
    #[test]
    fn play_to_finish() {
        let mut game = Game2048::new_game(Coord { row: 0, col: 0 }, 2);

        let mut rng = rand::rng();
        while game.state != GameState::Done {
            let aa = game.available_action();
            assert!(!aa.is_empty());

            let aa_vec = aa.iter().cloned().collect::<Vec<_>>();
            let random_index = rng.random_range(0..aa_vec.len());
            let dir = aa_vec[random_index];
            game.step_action(dir);

            let chances = game.available_chance();
            assert!(!chances.is_empty());
            let random_index = rng.random_range(0..chances.len());
            let cc = chances[random_index];
            game.step_random(cc.0, cc.1);
        }
        assert_eq!(game.state, GameState::Done);
    }
}
