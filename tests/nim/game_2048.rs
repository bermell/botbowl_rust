use std::{
    collections::HashSet,
    ops::{Add, Index, IndexMut, Neg},
};

use rand::Rng;

const SIZE: usize = 4;

/// Serpentine preference: large tiles should sit toward bottom-left (highest weight at (3,0)).
const SNAKE_WEIGHT: [[i64; SIZE]; SIZE] = [
    [1, 2, 3, 4],
    [8, 7, 6, 5],
    [9, 10, 11, 12],
    [16, 15, 14, 13],
];

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
        if chance_actions.is_empty() {
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
        // TODO: cache this?
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
        debug_assert!(self.state == GameState::WaitingForRandom);
        debug_assert!(self[coord] == 0);
        self[coord] = value;

        self.state = GameState::WaitingForAction;

        if self.available_action().is_empty() {
            self.state = GameState::Done;
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

    /// One pass over adjacent equal tiles: each tile merges at most once (standard 2048).
    fn merge_line(values: Vec<SqVal>) -> (Vec<SqVal>, u32) {
        let mut score_add = 0u32;
        let mut i = 0usize;
        let mut out = Vec::new();
        while i < values.len() {
            if i + 1 < values.len() && values[i] == values[i + 1] {
                let e = values[i];
                out.push(e + 1);
                score_add += 2u32.pow(e + 1);
                i += 2;
            } else {
                out.push(values[i]);
                i += 1;
            }
        }
        (out, score_add)
    }

    fn step_generic(&mut self, direction: Direction) {
        match direction {
            Direction::Left => {
                for row in 0..SIZE {
                    let values: Vec<SqVal> = (0..SIZE)
                        .map(|col| self.board[row][col])
                        .filter(|&v| v != 0)
                        .collect();
                    let (merged, add) = Self::merge_line(values);
                    self.score += add;
                    for c in 0..SIZE {
                        self.board[row][c] = 0;
                    }
                    for (i, &v) in merged.iter().enumerate() {
                        self.board[row][i] = v;
                    }
                }
            }
            Direction::Right => {
                for row in 0..SIZE {
                    let values: Vec<SqVal> = (0..SIZE)
                        .rev()
                        .map(|col| self.board[row][col])
                        .filter(|&v| v != 0)
                        .collect();
                    let (merged, add) = Self::merge_line(values);
                    self.score += add;
                    for c in 0..SIZE {
                        self.board[row][c] = 0;
                    }
                    for (i, &v) in merged.iter().enumerate() {
                        self.board[row][SIZE - 1 - i] = v;
                    }
                }
            }
            Direction::Up => {
                for col in 0..SIZE {
                    let values: Vec<SqVal> = (0..SIZE)
                        .map(|row| self.board[row][col])
                        .filter(|&v| v != 0)
                        .collect();
                    let (merged, add) = Self::merge_line(values);
                    self.score += add;
                    for r in 0..SIZE {
                        self.board[r][col] = 0;
                    }
                    for (i, &v) in merged.iter().enumerate() {
                        self.board[i][col] = v;
                    }
                }
            }
            Direction::Down => {
                for col in 0..SIZE {
                    let values: Vec<SqVal> = (0..SIZE)
                        .rev()
                        .map(|row| self.board[row][col])
                        .filter(|&v| v != 0)
                        .collect();
                    let (merged, add) = Self::merge_line(values);
                    self.score += add;
                    for r in 0..SIZE {
                        self.board[r][col] = 0;
                    }
                    for (i, &v) in merged.iter().enumerate() {
                        self.board[SIZE - 1 - i][col] = v;
                    }
                }
            }
        }
    }

    /// Static evaluation after a slide, before the random tile (`WaitingForRandom`).
    /// Combines cumulative score, empty cells, and snake-shaped monotonicity (tile exponents).
    pub fn static_eval_after_action(g: &Game2048) -> i64 {
        debug_assert_eq!(g.state, GameState::WaitingForRandom);
        const EMPTY_WEIGHT: i64 = 256;
        let empty = g.empty_squares().len() as i64;
        let mut snake = 0i64;
        for row in 0..SIZE {
            for col in 0..SIZE {
                let v = g.board[row][col] as i64;
                snake += v * SNAKE_WEIGHT[row][col];
            }
        }
        g.score as i64 + empty * EMPTY_WEIGHT + snake
    }

    /// One-step greedy: pick the legal slide with highest [`static_eval_after_action`].
    /// Directions are sorted so ties are resolved deterministically.
    pub fn best_direction_heuristic(&self) -> Direction {
        debug_assert_eq!(self.state, GameState::WaitingForAction);
        let mut dirs: Vec<Direction> = self.available_action().into_iter().collect();
        dirs.sort_unstable();
        dirs.into_iter()
            .max_by_key(|&d| {
                let mut g = self.clone();
                g.step_action(d);
                Self::static_eval_after_action(&g)
            })
            .expect("WaitingForAction implies at least one legal move")
    }

    /// Uniform random legal moves until the game ends (same policy as the random-play baseline).
    pub fn random_rollout_to_end<R: Rng>(&mut self, rng: &mut R) {
        while self.state != GameState::Done {
            match self.state {
                GameState::WaitingForAction => {
                    let mut dirs: Vec<Direction> = self.available_action().into_iter().collect();
                    dirs.sort_unstable();
                    let idx = rng.random_range(0..dirs.len());
                    self.step_action(dirs[idx]);
                }
                GameState::WaitingForRandom => {
                    let mut chances = self.available_chance();
                    chances.sort_by(|a, b| {
                        a.0.row
                            .cmp(&b.0.row)
                            .then_with(|| a.0.col.cmp(&b.0.col))
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    let idx = rng.random_range(0..chances.len());
                    let (c, v, _) = chances[idx];
                    self.step_random(c, v);
                }
                GameState::Done => break,
            }
        }
    }
}

#[cfg(test)]
mod test {

    use rand::Rng;
    use rand::SeedableRng;

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
            [2, 2, 0, 0],
            [3, 3, 0, 0],
            [6, 0, 0, 0],
            [0, 0, 0, 0],
        ];
        assert_eq!(game.board, expected);
    }

    #[test]
    fn game_done_when_no_available_actions() {
        let mut game = Game2048::new_custom([
            //comment
            [1, 1, 3, 0],
            [4, 5, 6, 7],
            [1, 2, 3, 4],
            [4, 5, 6, 7],
        ]);
        game.step_action(Direction::Up);
        assert_eq!(game.state, GameState::WaitingForRandom);
        assert!(!game.available_chance().is_empty());

        game.step_random(Coord { row: 3, col: 3 }, 1);
        assert_eq!(game.state, GameState::WaitingForAction);

        let aa = game.available_action();
        assert_eq!(aa.len(), 2);
        assert!(aa.contains(&Direction::Left));
        assert!(aa.contains(&Direction::Right));

        game.step_action(Direction::Left);
        assert_eq!(game.state, GameState::WaitingForRandom);

        game.step_random(Coord { row: 0, col: 3 }, 1);
        assert_eq!(game.state, GameState::Done);
    }
    /// In standard 2048, each tile merges at most once per move. Four equal tiles
    /// become two merged tiles (e.g. four 2s → two 4s), not one doubled merge.
    #[test]
    fn four_equal_tiles_do_not_chain_merge_in_one_move() {
        let mut game =
            Game2048::new_custom([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
        game.step_action(Direction::Left);
        let expected = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        assert_eq!(game.board, expected);
    }

    /// Score for a merge should equal the face value of the resulting tile (standard 2048).
    /// Tile exponents are log2: two 2s (exp 1) merge to 4 (exp 2) → +4 points.
    #[test]
    fn merge_score_adds_value_of_resulting_tile() {
        let mut game =
            Game2048::new_custom([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]);
        game.step_action(Direction::Left);
        assert_eq!(game.board[0][0], 2);
        assert_eq!(game.score, 4);
    }

    #[test]
    fn best_direction_heuristic_is_among_available() {
        let game = Game2048::new_custom([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 1]]);
        let aa = game.available_action();
        let pick = game.best_direction_heuristic();
        assert!(aa.contains(&pick));
    }

    #[test]
    fn static_eval_prefers_slide_that_opens_more_space() {
        // Two legal moves; one leaves more empty cells after the slide.
        let base = Game2048::new_custom([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 1]]);
        let mut left = base;
        left.step_action(Direction::Left);
        let mut up = base;
        up.step_action(Direction::Up);
        let score_left = Game2048::static_eval_after_action(&left);
        let score_up = Game2048::static_eval_after_action(&up);
        assert!(
            score_left > score_up,
            "left should score higher (more empty cells after merge)"
        );
    }

    #[test]
    fn heuristic_play_to_finish_seeded() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut game = Game2048::new_game(
            Coord {
                row: rng.random_range(0usize..4),
                col: rng.random_range(0usize..4),
            },
            2,
        );
        while game.state != GameState::Done {
            let dir = game.best_direction_heuristic();
            game.step_action(dir);
            let chances = game.available_chance();
            assert!(!chances.is_empty());
            let idx = rng.random_range(0..chances.len());
            let (c, v, _) = chances[idx];
            game.step_random(c, v);
        }
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
