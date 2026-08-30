use core::panic;
use derivative::Derivative;
use itertools::Itertools;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{
    cmp::{max, min},
    collections::HashSet,
};

use crate::core::{model, procedures::CoinToss};

use model::*;

use super::{
    bb_errors::{IllegalMovePosition, InvalidPlayerId},
    dices::{
        resolve_from_fixes, resolve_with_rng, BlockDice, Coin, D6Target, DicePolicy, FixedDice, RequestedRoll,
        RollResult, RollTarget,
    },
    procedures::{AnyProc, GameOver, Half},
    table::{NumBlockDices, PosAT, SimpleAT},
};

pub enum BuilderState {
    Turn { turn: u8 },
    Setup { turn: u8 },
    Kickoff { turn: u8 },
    CoinToss,
}

pub struct GameStateBuilder {
    home_players: Vec<Position>,
    away_players: Vec<Position>,
    ball_pos: Option<Position>,
    state: BuilderState,
    /// Runtime board override for `build()`. `None` → read `BoardDims::from_env()`.
    board_dims: Option<BoardDims>,
}

impl GameStateBuilder {
    /// creates a gamestate where away won the coin toss and choose to kick
    /// next is setup for away as defence and home as offence
    pub fn new_at_setup() -> GameState {
        let mut state: GameState = GameStateBuilder::new_start_of_game();

        state.fix_coin(Coin::Heads);
        state.step_simple(SimpleAT::Heads); //Away

        state.step_simple(SimpleAT::Kick); //Away
        state
    }

    /// creates a gamestate where away won the coin toss and choose to kick, and
    /// both teams setup their line of scrimmage
    pub fn new_at_kickoff() -> GameState {
        let mut state: GameState = GameStateBuilder::new_start_of_game();

        state.fix_coin(Coin::Heads);
        state.step_simple(SimpleAT::Heads); //Away

        state.step_simple(SimpleAT::Kick); //Away

        state.step_simple(SimpleAT::SetupLine); //Away
        state.step_simple(SimpleAT::EndSetup); //Away

        state.step_simple(SimpleAT::SetupLine); //Home
        state.step_simple(SimpleAT::EndSetup); //Home
        state
    }
    ///creates a gamestate with two human teams at very beginning of a gamestate
    ///which right now is the coin toss. (but later should be pregame which does weather roll abd
    ///such)
    pub fn new_start_of_game() -> GameState {
        GameStateBuilder::new_start_of_game_with(BoardDims::from_env())
    }
    pub fn new_start_of_game_with(board_dims: BoardDims) -> GameState {
        let mut state = GameStateBuilder::empty_state_with(board_dims);

        // Dugout
        let place = DugoutPlace::Reserves;
        // Role split scales with roster size; linemen take the remainder.
        // At roster_per_team()=12 this is the historical 6 linemen + 2 each of
        // blitzer/catcher/thrower. The per-tier distribution on smaller boards
        // is a deferred curriculum decision — this is just a sane default.
        let roster_per_team = state.board_dims.roster_per_team();
        let positionals = (roster_per_team / 6).max(1);
        let linemen = roster_per_team - 3 * positionals;
        for team in [TeamType::Home, TeamType::Away] {
            for _ in 0..linemen {
                state.dugout_add_new_player(PlayerStats::new_lineman(team), place);
            }
            for _ in 0..positionals {
                state.dugout_add_new_player(PlayerStats::new_blitzer(team), place);
            }
            for _ in 0..positionals {
                state.dugout_add_new_player(PlayerStats::new_catcher(team), place);
            }
            for _ in 0..positionals {
                state.dugout_add_new_player(PlayerStats::new_thrower(team), place);
            }
        }

        state.proc_stack = vec![GameOver::new(), Half::new(2), Half::new(1), CoinToss::new()];
        state.step_simple(SimpleAT::EndTurn);
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::Heads)));
        assert!(state.is_legal_action(&Action::Simple(SimpleAT::Tails)));
        // available_actions: AvailableActions::new_empty(),
        state
    }
    pub fn new() -> GameStateBuilder {
        GameStateBuilder {
            home_players: Vec::new(),
            away_players: Vec::new(),
            ball_pos: None,
            state: BuilderState::Turn { turn: 1 },
            board_dims: None,
        }
    }
    /// Override the runtime board for `build()` — the testing hook that lets one
    /// binary exercise any board `<=` the compiled capacity without a recompile.
    pub fn with_board_dims(&mut self, board_dims: BoardDims) -> &mut GameStateBuilder {
        self.board_dims = Some(board_dims);
        self
    }
    pub fn add_str(&mut self, start_pos: Position, s: &str) -> &mut GameStateBuilder {
        let mut pos = start_pos;
        let start_x = pos.x;
        let mut newline = false;
        for c in s.chars() {
            assert!(!pos.is_out());
            match c {
                'a' => self.away_players.push(pos),
                'h' => self.home_players.push(pos),
                'H' => {
                    self.home_players.push(pos);
                    self.ball_pos = Some(pos);
                }
                'A' => {
                    self.away_players.push(pos);
                    self.ball_pos = Some(pos);
                }
                '\n' => newline = true,
                _ => (),
            }
            if newline {
                pos.y += 1;
                pos.x = start_x;
                newline = false;
            } else {
                pos.x += 1;
            }
        }
        self
    }
    pub fn add_home_player(&mut self, position: Position) -> &mut GameStateBuilder {
        self.home_players.push(position);
        self
    }
    pub fn set_state(&mut self, state: BuilderState) -> &mut GameStateBuilder {
        self.state = state;
        self
    }

    pub fn add_away_player(&mut self, position: Position) -> &mut GameStateBuilder {
        self.away_players.push(position);
        self
    }

    pub fn add_home_players(&mut self, players: &[(Coord, Coord)]) -> &mut GameStateBuilder {
        players
            .iter()
            .for_each(|(x, y)| self.home_players.push(Position::new((*x, *y))));
        self
    }

    pub fn add_away_players(&mut self, players: &[(Coord, Coord)]) -> &mut GameStateBuilder {
        players
            .iter()
            .for_each(|(x, y)| self.away_players.push(Position::new((*x, *y))));
        self
    }

    pub fn add_ball(&mut self, xy: (Coord, Coord)) -> &mut GameStateBuilder {
        self.ball_pos = Some(Position::new((xy.0, xy.1)));
        self
    }

    pub fn add_ball_pos(&mut self, position: Position) -> &mut GameStateBuilder {
        self.ball_pos = Some(position);
        self
    }

    pub fn empty_state() -> GameState {
        GameStateBuilder::empty_state_with(BoardDims::from_env())
    }
    pub fn empty_state_with(board_dims: BoardDims) -> GameState {
        GameState {
            board_dims,
            fielded_players: Default::default(),
            home: TeamState::new(),
            away: TeamState::new(),
            board: Default::default(),
            ball: BallState::OffPitch,
            bounce_squares: SmallVec::new(),
            dugout_players: Default::default(),
            proc_stack: Vec::new(),
            //new_procs: VecDeque::new(),
            available_actions: AvailableActions::new_empty(),
            path_buffer: None,
            rng: ChaCha8Rng::from_entropy(),
            info: GameInfo::new(),
            dice_mode: DiceMode::default(),
            pending_roll: None,
            registered_roll: None,
            log: Vec::new(),
            print_log: false,
            next_input: None,
        }
    }
    pub fn build(&mut self) -> GameState {
        let dims = self.board_dims.unwrap_or_else(BoardDims::from_env);
        let mut state = GameStateBuilder::new_start_of_game_with(dims);

        let user_turn = match self.state {
            BuilderState::CoinToss => return state,
            BuilderState::Kickoff { turn } => turn,
            BuilderState::Setup { turn } => turn,
            BuilderState::Turn { turn } => turn,
        };
        // TODO: we should set the dice policy to fixed here. But likely needs changes elsewhere
        assert!(user_turn > 0, "turn must be positive");
        assert_eq!(state.info.home_turn, 0);
        assert_eq!(state.info.away_turn, 0);

        state.fix_coin(Coin::Heads);
        state.step_simple(SimpleAT::Heads); //Away

        state.step_simple(SimpleAT::Kick); //Away

        //increase turn counter according to user wish
        state.info.home_turn += user_turn - 1;
        state.info.away_turn += user_turn - 1;

        state.step_simple(SimpleAT::SetupLine); //Away
        state.step_simple(SimpleAT::EndSetup); //Away

        state.step_simple(SimpleAT::SetupLine); //Home
        state.step_simple(SimpleAT::EndSetup); //Home

        if let BuilderState::Kickoff { .. } = self.state {
            return state;
        }

        // Fast-forward through the kickoff to reach the receiving team's turn.
        // Its outcome is discarded below (players and ball are reset), so resolve
        // it with the RNG under a fixed seed — board-independent, unlike the old
        // hand-pinned dice script which assumed a 28x17 pitch (a fixed "scatter
        // up 5" lands out of bounds on shorter boards).
        state.set_dice_mode(DiceMode::RollDice);
        state.set_seed(0);
        state.step_simple(SimpleAT::KickoffAimMiddle);
        // A scattered kick can land out of bounds, prompting the receiver to
        // place the touchback ball. Resolve any such pre-turn choice.
        let mut guard = 0;
        while !state.get_available_actions().get_simple().contains(&SimpleAT::EndTurn) {
            let mut actions = Vec::new();
            state.get_available_actions().collect_non_path_actions(&mut actions);
            let action = actions
                .into_iter()
                .find(|a| matches!(a, Action::Positional(..)))
                .expect("kickoff stalled with no resolvable action before the turn");
            state.step(action).unwrap();
            guard += 1;
            assert!(guard < 8, "kickoff fast-forward did not reach a turn");
        }
        // Back to the default FixedDice mode for the caller's test rolls.
        state.set_dice_mode(DiceMode::default());
        // Drop the ball before clearing players: a receiver may have caught the
        // kick, and unfield_player refuses to remove the ball carrier.
        state.set_ball(BallState::OffPitch);
        state.clear_all_players().unwrap();

        for position in self.home_players.iter() {
            let player_stats = PlayerStats::new_lineman(TeamType::Home);
            _ = state.add_new_player_to_field(player_stats, *position)
        }

        for position in self.away_players.iter() {
            let player_stats = PlayerStats::new_lineman(TeamType::Away);
            _ = state.add_new_player_to_field(player_stats, *position)
        }

        if let Some(pos) = self.ball_pos {
            state.set_ball(match state.get_player_at(pos) {
                None => BallState::OnGround(pos),
                Some(p) if p.status == PlayerStatus::Up => BallState::Carried(p.id),
                _ => panic!(),
            })
        }
        // decrease turn counter before calling endturn twice
        // (need to call end turn here to refresh available actions)
        state.step_simple(SimpleAT::EndTurn);
        state.step_simple(SimpleAT::EndTurn);
        state.info.home_turn -= 1;
        state.info.away_turn -= 1;

        state.set_logging_state(true);
        state
    }
}

impl Default for GameStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct GameInfo {
    pub half: u8,
    pub home_turn: u8,
    pub away_turn: u8,
    pub winner: Option<TeamType>,
    pub turnover: bool,
    pub active_player: Option<PlayerID>,
    pub player_action_type: Option<PosAT>,
    pub team_turn: TeamType,
    pub game_over: bool,
    pub weather: Weather,
    pub kicking_first_half: TeamType,
    pub kickoff_by_team: Option<TeamType>,
    pub kicking_this_drive: TeamType,
    pub handoff_available: bool,
    pub foul_available: bool,
    pub pass_available: bool,
    pub blitz_available: bool,
    pub handle_td_by: Option<PlayerID>,
    /// True iff the currently-active player picked up the ball during this
    /// activation and has not yet started a follow-up move action. Set in
    /// `PickupProc::apply_success`, cleared at activation start
    /// (`set_active_player`), when `Turn` clears the active player, and when
    /// the player selects their next move action in `MoveAction`.
    ///
    /// Part of the hashed/compared state on purpose: a player who *just*
    /// picked up the ball and one who was *already* carrying can otherwise
    /// reach a byte-identical `GameState`, and the MCTS bot needs to treat
    /// them differently (the former is owed one extra move action — see
    /// `botbowl-mcts` pruning rule P8). Keeping the distinction in the state
    /// keeps that pruning a pure function of `(state, action)`.
    #[serde(default)]
    pub pickup_this_activation: bool,
    pub blitz_this_activation: bool,
}
impl GameInfo {
    fn new() -> GameInfo {
        GameInfo {
            half: 0,
            active_player: None,
            team_turn: TeamType::Away,
            game_over: false,
            winner: None,
            weather: Weather::Nice,
            kicking_first_half: TeamType::Away,
            home_turn: 0,
            away_turn: 0,
            player_action_type: None,
            handoff_available: true,
            pass_available: true,
            foul_available: true,
            blitz_available: true,
            handle_td_by: None,
            kickoff_by_team: None,
            kicking_this_drive: TeamType::Away,
            turnover: false,
            pickup_this_activation: false,
            blitz_this_activation: false,
        }
    }
}
/// Dice resolution strategy for a `GameState`.
///
/// Replaces the previous four-flag setup (`fixes` / `dice_policy` /
/// `expose_rolls` / `rng_enabled`) with one explicit mode per caller
/// intent. Modes are mutually exclusive and switched via
/// `GameState::set_dice_mode`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiceMode {
    /// Production play and search rollouts. All rolls drawn from
    /// `GameState::rng`. `state.fix_*` and `state.step_with_roll`
    /// panic in this mode.
    RollDice,
    /// Tests and `GameStateBuilder` setup. FIFO queue of pinned dice
    /// values; `resolve_from_fixes` panics on empty for a requested
    /// roll.
    FixedDice(FixedDice),
    /// MCTS bot / interactive search. `step()` is forbidden; callers
    /// drive the engine via `micro_step` (which returns mid-procedure
    /// with `pending_roll` set) and `step_with_roll(result)` to resume.
    RegisterRolls,
    /// Lectures / scripted scenarios. The policy is required to
    /// resolve every requested roll — it may internally consult the
    /// engine's RNG for roll types it doesn't override.
    DicePolicy(DicePolicy),
}

impl Default for DiceMode {
    /// Defaults to `FixedDice` so tests and `GameStateBuilder` work
    /// without ceremony. Production callers (`BotGameRunnerBuilder`,
    /// MCTS, lectures) explicitly call `set_dice_mode` after construction.
    fn default() -> Self {
        DiceMode::FixedDice(FixedDice::default())
    }
}
#[derive(Derivative)]
#[derivative(PartialEq)]
#[derive(Serialize, Deserialize, Debug)]
pub struct GameState {
    pub info: GameInfo,
    pub home: TeamState,
    pub away: TeamState,

    /// The runtime-active board (logical dimensions + team size). Constant for
    /// the life of a game; sourced from env / the builder at construction. The
    /// physical arrays below stay at compile-time capacity — see `BoardDims`.
    pub board_dims: BoardDims,

    // Both arrays are sized for the whole roster: a player is either fielded or
    // in the dugout, and at most ROSTER_PER_TEAM per team can exist. (SetupLine
    // can field more than TEAM_SIZE when the roster is smaller than the
    // formation template, so fielded must hold the full roster, not 2*TEAM_SIZE.)
    fielded_players: [Option<FieldedPlayer>; 2 * ROSTER_PER_TEAM],
    dugout_players: [Option<DugoutPlayer>; 2 * ROSTER_PER_TEAM],
    board: FullPitch<Option<PlayerID>>,
    pub ball: BallState,
    /// Squares the ball has already occupied during the current in-air
    /// bounce / throw-in sequence. Maintained exclusively through
    /// `set_ball`: every `InAir(pos)` transition appends `pos`, and any
    /// transition to a settled state (`OnGround` / `Carried` / `OffPitch`)
    /// clears it. Invariant: non-empty *only* while `ball` is `InAir`.
    /// Its purpose is to let the MCTS search skip bounce directions that
    /// lead back to a square the ball has already bounced from.
    #[serde(default)]
    pub bounce_squares: SmallVec<[Position; 8]>,
    proc_stack: Vec<AnyProc>,
    pub available_actions: Box<AvailableActions>,
    /// Reusable backing storage for `MoveAction`/`BlockAction`'s path
    /// offerings. `available_actions.has_paths` is the gate — when false,
    /// this buffer's contents are stale/unreachable. Lazy-allocated on the
    /// first producer call; preserved across micro_steps so we don't pay
    /// the 4KB alloc + 476-slot drop every frame. `#[serde(skip)]` because
    /// the buffer is recoverable scratch (the producer rebuilds on the
    /// next decision). Cloned conditionally — see `Clone for GameState`
    /// below — so non-pathing states stay cheap to clone (the MCTS hot
    /// path).
    #[serde(skip, default)]
    #[derivative(PartialEq = "ignore")]
    pub(crate) path_buffer: Option<Box<FullPitch<Option<std::sync::Arc<super::pathing::Node>>>>>,
    /// Single source of truth for how the engine resolves dice. See
    /// `DiceMode` docs; switch via `set_dice_mode`.
    #[serde(default)]
    pub(crate) dice_mode: DiceMode,
    /// In `DiceMode::RegisterRolls`, the roll the engine paused on
    /// after a procedure returned `ProcState::NeedRoll`. The caller
    /// observes this, calls `step_with_roll(result)` to resume.
    /// Included in `PartialEq` because "about to resolve a 3+ dodge"
    /// is a different situation than "ready to act" — MCTS chance
    /// nodes hinge on this distinction.
    #[serde(default, skip)]
    pub pending_roll: Option<RequestedRoll>,
    /// The roll provided by the most recent `step_with_roll` call,
    /// consumed on the next `micro_step` to satisfy the `pending_roll`
    /// the engine paused on. Only used in `DiceMode::RegisterRolls`.
    #[serde(default, skip)]
    #[derivative(PartialEq = "ignore")]
    registered_roll: Option<RollResult>,

    #[serde(skip)]
    #[derivative(PartialEq = "ignore")]
    next_input: Option<ProcInput>,

    #[serde(skip, default = "ChaCha8Rng::from_entropy")]
    #[derivative(PartialEq = "ignore")]
    pub(crate) rng: ChaCha8Rng,

    // Log is a debug/audit accumulator only — two states that reached the
    // same game situation via different histories should compare equal.
    // Excluded from PartialEq so MCTS state recombination works.
    #[derivative(PartialEq = "ignore")]
    log: Vec<String>,
    #[derivative(PartialEq = "ignore")]
    print_log: bool,
}

impl Eq for GameState {}

// Hand-rolled Clone instead of derived: `path_buffer` is reusable scratch
// allocated up to 4KB on the heap. Cloning it eagerly would regress the MCTS
// hot path, where GameState clones happen for every tree expansion and the
// vast majority of cloned states aren't currently inside a `MoveAction` /
// `BlockAction` decision. Clone the buffer only when `available_actions.has_paths`
// says it holds live offerings for the cloned state.
impl Clone for GameState {
    fn clone(&self) -> Self {
        let path_buffer = if self.available_actions.has_paths {
            self.path_buffer.clone()
        } else {
            None
        };
        GameState {
            info: self.info.clone(),
            home: self.home,
            away: self.away,
            board_dims: self.board_dims,
            fielded_players: self.fielded_players.clone(),
            dugout_players: self.dugout_players.clone(),
            board: self.board,
            ball: self.ball,
            bounce_squares: self.bounce_squares.clone(),
            proc_stack: self.proc_stack.clone(),
            available_actions: self.available_actions.clone(),
            path_buffer,
            dice_mode: self.dice_mode.clone(),
            pending_roll: self.pending_roll,
            registered_roll: self.registered_roll,
            next_input: self.next_input,
            rng: self.rng.clone(),
            log: self.log.clone(),
            print_log: self.print_log,
        }
    }
}

// Hand-rolled Hash that covers the canonical "situation" fields only.
// Used by MCTS-style transposition tables. The hash is intentionally
// conservative — collisions are corrected by PartialEq, so undercounting
// fields is safe; overcounting (e.g. hashing the RNG state or the log)
// would prevent useful recombination.
impl std::hash::Hash for GameState {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        // GameInfo discriminators
        self.info.half.hash(h);
        self.info.home_turn.hash(h);
        self.info.away_turn.hash(h);
        (self.info.team_turn as u8).hash(h);
        self.info.game_over.hash(h);
        self.info.turnover.hash(h);
        self.info.active_player.hash(h);
        self.info.pickup_this_activation.hash(h);
        self.info.blitz_this_activation.hash(h);
        self.info.handle_td_by.hash(h);
        (self.info.kicking_first_half as u8).hash(h);
        (self.info.kicking_this_drive as u8).hash(h);
        self.info.kickoff_by_team.map(|t| t as u8).hash(h);

        // Scores
        self.home.score.hash(h);
        self.away.score.hash(h);

        // Fielded players slot-by-slot
        for slot in &self.fielded_players {
            match slot {
                None => 0u8.hash(h),
                Some(p) => {
                    1u8.hash(h);
                    p.id.hash(h);
                    p.position.x.hash(h);
                    p.position.y.hash(h);
                    (p.status as u8).hash(h);
                    p.used.hash(h);
                    p.moves.hash(h);
                    (p.stats.team as u8).hash(h);
                    p.stats.str_.hash(h);
                    p.stats.ma.hash(h);
                    p.stats.ag.hash(h);
                    p.stats.av.hash(h);
                }
            }
        }

        // Ball
        match &self.ball {
            BallState::OffPitch => 0u8.hash(h),
            BallState::OnGround(p) => {
                1u8.hash(h);
                p.x.hash(h);
                p.y.hash(h);
            }
            BallState::Carried(id) => {
                2u8.hash(h);
                id.hash(h);
            }
            BallState::InAir(p) => {
                3u8.hash(h);
                p.x.hash(h);
                p.y.hash(h);
            }
        }
        // Two in-air states with different already-bounced-through squares are
        // distinct search situations (they permit different next bounces).
        for p in &self.bounce_squares {
            p.x.hash(h);
            p.y.hash(h);
        }

        // Dice mode + procedure-stack-top discriminator. The full
        // proc_stack is heavy and changes shape often — using only the
        // top procedure's name catches "what the engine is about to do
        // next" without paying for a deep recursive hash. Per-variant
        // payloads (fixed-dice queue, policy parameters) are search-
        // irrelevant — collisions are corrected by PartialEq.
        std::mem::discriminant(&self.dice_mode).hash(h);
        self.proc_stack.len().hash(h);
        self.proc_stack_top().hash(h);
        // pending_roll is part of "what happens next" — two states that
        // differ only in whether they're paused on a roll are distinct.
        match &self.pending_roll {
            None => 0u8.hash(h),
            Some(r) => {
                1u8.hash(h);
                std::mem::discriminant(r).hash(h);
            }
        }
    }
}

/// Log a formatted message into the game's log buffer **only when
/// logging is on**. The `format!` is gated behind `is_logging()`, so a
/// disabled logger costs one bool load per `micro_step` instead of
/// formatting and discarding the whole `{:?}` tree of `ProcState`,
/// `AvailableActions`, etc. — that formatting was ~10% of wall-clock
/// in random-vs-random games before this macro existed.
#[macro_export]
macro_rules! game_log {
    ($state:expr, $($arg:tt)*) => {
        if $state.is_logging() {
            $state.log(format!($($arg)*));
        }
    };
}

impl GameState {
    pub fn log(&mut self, s: String) {
        if !self.print_log {
            return;
        }
        println!("{}", s);
        self.log.push(s);
    }

    #[inline]
    pub fn is_logging(&self) -> bool {
        self.print_log
    }
    pub fn get_log(&self) -> &Vec<String> {
        &self.log
    }
    pub fn set_logging_state(&mut self, state: bool) {
        self.print_log = state;
    }
    /// Drop all log entries accumulated so far. Useful before starting
    /// a workload that clones state many times (e.g. an MCTS search):
    /// otherwise every clone copies the historical log Vec.
    pub fn clear_log(&mut self) {
        self.log.clear();
    }
    pub fn get_dugout(&self) -> impl Iterator<Item = &DugoutPlayer> {
        self.dugout_players.iter().flatten()
    }
    pub fn get_dugout_mut(&mut self) -> impl Iterator<Item = &mut DugoutPlayer> {
        self.dugout_players.iter_mut().flatten()
    }
    pub fn dugout_add_new_player(&mut self, player_stats: PlayerStats, place: DugoutPlace) {
        let id = match self
            .dugout_players
            .iter()
            .enumerate()
            .find(|(_, player)| player.is_none())
        {
            Some((id, _)) => id,
            None => panic!("Not room in gamestate of another dugout player!"),
        };
        self.dugout_players[id] = Some(DugoutPlayer {
            stats: player_stats,
            place,
            id,
        })
    }
    pub fn get_dugout_player(&self, id: DugoutPlayerID) -> Option<&DugoutPlayer> {
        self.dugout_players[id].as_ref()
    }
    pub fn get_dugout_player_mut(&mut self, id: DugoutPlayerID) -> Option<&mut DugoutPlayer> {
        self.dugout_players[id].as_mut()
    }

    pub fn field_dugout_player(&mut self, dugout_id: DugoutPlayerID, position: Position) {
        let DugoutPlayer { stats, place, .. } = self.dugout_players[dugout_id].take().unwrap();
        assert_eq!(place, DugoutPlace::Reserves, "Must field from reserves_box");
        self.add_new_player_to_field(stats, position).unwrap();
    }
    pub fn get_available_actions(&self) -> &AvailableActions {
        &self.available_actions
    }
    pub fn home_to_act(&self) -> bool {
        self.get_available_actions()
            .get_team()
            .map(|team| team == TeamType::Home)
            .unwrap_or(false)
    }
    pub fn away_to_act(&self) -> bool {
        self.get_available_actions()
            .get_team()
            .map(|team| team == TeamType::Away)
            .unwrap_or(false)
    }

    pub fn set_active_player(&mut self, id: PlayerID) {
        debug_assert!(self.get_player(id).is_ok());
        self.info.active_player = Some(id);
        // Fresh activation — clear any pickup-bonus owed to a prior activation.
        self.info.pickup_this_activation = false;
        self.info.blitz_this_activation = false;
    }

    pub fn get_active_player(&self) -> Option<&FieldedPlayer> {
        self.info.active_player.and_then(|id| self.get_player(id).ok())
    }
    pub fn get_active_player_mut(&mut self) -> Option<&mut FieldedPlayer> {
        self.info.active_player.and_then(|id| self.get_mut_player(id).ok())
    }
    pub fn get_endzone_x(&self, team: TeamType) -> Coord {
        self.board_dims.endzone_x(team)
    }
    /// Out of bounds of the runtime-active board (gameplay OOB). For array-bound
    /// sanity checks use `Position::is_out` instead.
    pub fn is_out(&self, pos: Position) -> bool {
        self.board_dims.is_out(pos)
    }
    pub fn is_on_team_side(&self, pos: Position, team: TeamType) -> bool {
        self.board_dims.is_on_team_side(pos, team)
    }
    pub fn set_seed(&mut self, state: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(state);
    }

    pub fn proc_stack_top(&self) -> Option<&'static str> {
        self.proc_stack.last().map(|p| p.name())
    }

    /// Read-only peek at the procedure on top of the stack, for callers
    /// (e.g. the MCTS roll-outcome scripting) that need the procedure's
    /// data, not just the name `proc_stack_top` gives.
    pub fn proc_stack_peek(&self) -> Option<&AnyProc> {
        self.proc_stack.last()
    }

    pub fn dice_mode(&self) -> &DiceMode {
        &self.dice_mode
    }

    /// Switch the dice resolution strategy. Asserts no `pending_roll`
    /// is outstanding so a mode switch can't strand a paused
    /// `RegisterRolls` engine in a mode that doesn't know how to
    /// resume it.
    pub fn set_dice_mode(&mut self, mode: DiceMode) {
        debug_assert!(
            self.pending_roll.is_none(),
            "set_dice_mode called while pending_roll is set"
        );
        self.registered_roll = None;
        self.dice_mode = mode;
    }

    /// Relabel the in-progress half as `half` (1 or 2), dropping any pending
    /// unstarted half so the game ends when this half's turns run out. For
    /// scenario/training-state generators; the turn counters are untouched.
    pub fn set_half(&mut self, half: u8) {
        debug_assert!(half == 1 || half == 2);
        self.info.half = half;
        if half == 2 {
            self.proc_stack
                .retain(|p| !matches!(p, AnyProc::Half(h) if !h.started));
        }
    }

    fn fixes_mut(&mut self) -> &mut FixedDice {
        match &mut self.dice_mode {
            DiceMode::FixedDice(fixes) => fixes,
            other => panic!(
                "FixedDice access requires DiceMode::FixedDice, current mode is {:?}",
                std::mem::discriminant(other)
            ),
        }
    }

    /// Push a `Coin` value onto the fixed-dice queue. Panics if the
    /// current mode is not `DiceMode::FixedDice`.
    pub fn fix_coin(&mut self, value: Coin) {
        self.fixes_mut().fix_coin(value);
    }
    /// Push a `D3` value onto the fixed-dice queue. Panics if the
    /// current mode is not `DiceMode::FixedDice`.
    pub fn fix_d3(&mut self, value: u8) {
        self.fixes_mut().fix_d3(value);
    }
    /// Push a `D6` value onto the fixed-dice queue. Panics if the
    /// current mode is not `DiceMode::FixedDice`.
    pub fn fix_d6(&mut self, value: u8) {
        self.fixes_mut().fix_d6(value);
    }
    /// Push a `D8` value onto the fixed-dice queue. Panics if the
    /// current mode is not `DiceMode::FixedDice`.
    pub fn fix_d8(&mut self, value: u8) {
        self.fixes_mut().fix_d8(value);
    }
    /// Push a `D8` direction onto the fixed-dice queue. Panics if the
    /// current mode is not `DiceMode::FixedDice`.
    pub fn fix_d8_direction(&mut self, direction: Direction) {
        self.fixes_mut().fix_d8_direction(direction);
    }
    /// Push a `BlockDice` value onto the fixed-dice queue. Panics if
    /// the current mode is not `DiceMode::FixedDice`.
    pub fn fix_blockdice(&mut self, value: BlockDice) {
        self.fixes_mut().fix_blockdice(value);
    }
    /// True iff the current mode is `FixedDice` with an empty queue,
    /// or any non-FixedDice mode (which has no queue at all).
    pub fn fixes_is_empty(&self) -> bool {
        match &self.dice_mode {
            DiceMode::FixedDice(fixes) => fixes.is_empty(),
            _ => true,
        }
    }
    pub fn blockdice_fixes_len(&self) -> usize {
        match &self.dice_mode {
            DiceMode::FixedDice(fixes) => fixes.blockdice_fixes_len(),
            _ => 0,
        }
    }

    /// Where `kicking_team`'s kickoff aims: the centre of the *receiving*
    /// half. Note the argument is the team that **kicks**, not the receiver.
    ///
    /// The Away branch is written as the mirror of the Home one rather than as
    /// `3w/4`, which is one column too deep at every width divisible by 4 (28
    /// and 16 included) and gave the two sides materially different kickoffs
    /// — see plan 023 B1. `w - 1 - w/4` is mirror-symmetric by construction at
    /// every width, which is the invariant
    /// `kickoff_aim_is_centre_of_receiving_half_and_mirror_symmetric` pins.
    pub fn get_best_kickoff_aim_for(&self, kicking_team: TeamType) -> Position {
        let w = self.board_dims.width;
        let mid_y = self.board_dims.height / 2 - 1;
        let away_half_centre = w / 4;
        match kicking_team {
            TeamType::Home => Position::new((away_half_centre, mid_y)),
            TeamType::Away => Position::new((w - 1 - away_half_centre, mid_y)),
        }
    }

    pub fn get_team_from_player(&self, id: PlayerID) -> Result<&TeamState> {
        self.get_player(id)
            .map(|player| player.stats.team)
            .map(|team| self.get_team(team))
    }

    pub fn get_mut_team_from_player(&mut self, id: PlayerID) -> Result<&mut TeamState> {
        self.get_player(id)
            .map(|player| player.stats.team)
            .map(|team| self.get_mut_team(team))
    }

    pub fn get_team(&self, team: TeamType) -> &TeamState {
        match team {
            TeamType::Home => &self.home,
            TeamType::Away => &self.away,
        }
    }

    pub fn get_mut_team(&mut self, team: TeamType) -> &mut TeamState {
        match team {
            TeamType::Home => &mut self.home,
            TeamType::Away => &mut self.away,
        }
    }
    pub fn get_active_players_team(&self) -> Option<&TeamState> {
        self.info
            .active_player
            .and_then(|id| self.get_player(id).ok())
            .map(|player| self.get_team(player.stats.team))
    }

    pub fn get_active_players_team_mut(&mut self) -> Option<&mut TeamState> {
        self.get_mut_team_from_player(self.info.active_player.unwrap()).ok()
    }
    pub fn get_active_teamtype(&self) -> Option<TeamType> {
        self.available_actions.get_team()
    }

    pub fn get_player_id_at(&self, p: Position) -> Option<PlayerID> {
        self.get_player_id_at_coord(p.x, p.y)
    }
    pub fn get_player_at(&self, p: Position) -> Option<&FieldedPlayer> {
        self.get_player_at_coord(p.x, p.y)
    }

    pub fn get_player_id_at_coord(&self, x: Coord, y: Coord) -> Option<PlayerID> {
        //unwrap is OK here because if you're requesting negative indicies, you want the program to crash!
        // let xx = usize::try_from(x).unwrap();
        // let yy = usize::try_from(y).unwrap();
        // self.board[xx][yy]
        self.board[Position::new((x, y))]
    }
    pub fn get_player_at_coord(&self, x: Coord, y: Coord) -> Option<&FieldedPlayer> {
        match self.get_player_id_at_coord(x, y) {
            None => None,
            Some(id) => Some(self.get_player(id).unwrap()),
            //above unwrap is safe for bad input. If it panics it's an interal logical error!
        }
    }

    pub fn get_player_unsafe(&self, id: PlayerID) -> &FieldedPlayer {
        self.fielded_players[id].as_ref().unwrap()
    }

    pub fn get_mut_player_unsafe(&mut self, id: PlayerID) -> &mut FieldedPlayer {
        self.fielded_players[id].as_mut().unwrap()
    }

    pub fn get_player(&self, id: PlayerID) -> Result<&FieldedPlayer> {
        match &self.fielded_players[id] {
            Some(player) => Ok(player),
            None => Err(Box::new(InvalidPlayerId { id })),
        }
    }

    pub fn get_adj_positions(&self, position: Position) -> impl Iterator<Item = Position> {
        debug_assert!(!position.is_out());
        Direction::all_directions_iter().map(move |&direction| position + direction)
    }

    pub fn get_adj_players(&self, p: Position) -> impl Iterator<Item = &FieldedPlayer> + '_ {
        self.get_adj_positions(p)
            .filter_map(|adj_pos| self.get_player_at(adj_pos))
    }

    pub fn get_mut_player_at_unsafe(&mut self, p: Position) -> &mut FieldedPlayer {
        self.get_mut_player_unsafe(self.get_player_id_at(p).unwrap())
    }

    pub fn get_mut_player(&mut self, id: PlayerID) -> Result<&mut FieldedPlayer> {
        match &mut self.fielded_players[id] {
            Some(player) => Ok(player),
            None => Err(Box::new(InvalidPlayerId { id })),
        }
    }
    pub fn get_catch_target(&self, id: PlayerID) -> Result<D6Target> {
        let player = self.get_player(id)?;
        let mut target = player.ag_target();
        let team = player.stats.team;
        target.add_modifer(
            -(self
                .get_adj_players(player.position)
                .filter(|player_| player_.stats.team != team && player_.has_tackle_zone())
                .count() as i8),
        );

        if let Weather::Rain = self.info.weather {
            target.add_modifer(-1);
        }
        Ok(target)
    }

    /// Sets the ball state while maintaining `bounce_squares`. Every
    /// `InAir(pos)` transition records `pos` as a square the ball has
    /// occupied this bounce/throw-in sequence; settling the ball
    /// (`OnGround` / `Carried` / `OffPitch`) clears the record. All engine
    /// ball transitions must go through this method rather than assigning
    /// `self.ball` directly, or the invariant breaks.
    pub fn set_ball(&mut self, ball: BallState) {
        match ball {
            BallState::InAir(pos) => self.bounce_squares.push(pos),
            _ => self.bounce_squares.clear(),
        }
        self.ball = ball;
    }

    pub fn get_ball_position(&self) -> Option<Position> {
        match self.ball {
            BallState::OffPitch => None,
            BallState::OnGround(pos) => Some(pos),
            BallState::Carried(id) => Some(self.get_player(id).unwrap().position),
            BallState::InAir(pos) => Some(pos),
        }
    }

    pub fn get_tz_on_except_from_id(&self, id: PlayerID, except_from_id: PlayerID) -> u8 {
        let player = self.get_player_unsafe(id);
        let team = player.stats.team;

        self.get_adj_players(player.position)
            .filter(|adj_player| {
                adj_player.stats.team != team && adj_player.has_tackle_zone() && adj_player.id != except_from_id
            })
            .count() as u8
    }

    pub fn get_tz_on(&self, id: PlayerID) -> u8 {
        let player = self.get_player_unsafe(id);
        let team = player.stats.team;

        self.get_adj_players(player.position)
            .filter(|adj_player| adj_player.stats.team != team && adj_player.has_tackle_zone())
            .count() as u8
    }

    pub fn get_blockdices(&self, attacker: PlayerID, defender: PlayerID) -> NumBlockDices {
        let attacker_pos = self.get_player_unsafe(attacker).position;
        self.get_blockdices_from(attacker, attacker_pos, defender)
    }

    pub fn get_blockdices_from(&self, attacker: PlayerID, attacker_pos: Position, defender: PlayerID) -> NumBlockDices {
        let attr = self.get_player_unsafe(attacker);
        let defr = self.get_player_unsafe(defender);

        debug_assert_ne!(attr.stats.team, defr.stats.team);
        debug_assert_eq!(attacker_pos.distance_to(&defr.position), 1);
        // debug_assert!(attr.has_tackle_zone());
        debug_assert_eq!(defr.status, PlayerStatus::Up);

        let mut attr_str = attr.stats.str_;
        let mut defr_str = defr.stats.str_;

        attr_str += self
            .get_adj_players(defr.position)
            .filter(|attr_assister| {
                attr_assister.id != attr.id
                    && attr_assister.stats.team == attr.stats.team
                    && attr_assister.has_tackle_zone()
                    && self.get_tz_on_except_from_id(attr_assister.id, defr.id) == 0
                //what is guard anyway?
            })
            .count() as u8;

        defr_str += self
            .get_adj_players(attacker_pos)
            .filter(|defr_assister| {
                defr_assister.id != defr.id
                    && defr_assister.stats.team == defr.stats.team
                    && defr_assister.has_tackle_zone()
                    && self.get_tz_on_except_from_id(defr_assister.id, attr.id) == 0
                //what is guard anyway?
            })
            .count() as u8;

        if attr_str > 2 * defr_str {
            NumBlockDices::Three
        } else if attr_str > defr_str {
            NumBlockDices::Two
        } else if attr_str == defr_str {
            NumBlockDices::One
        } else if 2 * attr_str < defr_str {
            NumBlockDices::ThreeUphill
        } else {
            NumBlockDices::TwoUphill
        }
    }
    pub fn get_line_of_scrimage_x(&self, team: TeamType) -> Coord {
        self.board_dims.los_x(team)
    }
    pub fn move_player(&mut self, id: PlayerID, new_pos: Position) -> Result<()> {
        let old_pos = self.get_player(id)?.position;
        if let Some(occupied_id) = self.board[new_pos] {
            panic!(
                "Tried to move {}, to {:?} but it was already occupied by {}",
                id, new_pos, occupied_id
            );
            //return Err(Box::new(IllegalMovePosition{position: new_pos} ))
        }
        self.board[old_pos] = None;
        self.get_mut_player(id)?.position = new_pos;
        self.board[new_pos] = Some(id);
        Ok(())
    }
    pub fn get_players_on_pitch(&self) -> impl Iterator<Item = &FieldedPlayer> {
        self.fielded_players.iter().filter_map(|x| x.as_ref())
    }
    pub fn get_players_on_pitch_mut(&mut self) -> impl Iterator<Item = &mut FieldedPlayer> {
        self.fielded_players.iter_mut().filter_map(|x| x.as_mut())
    }
    pub fn get_players_on_pitch_in_team(&self, team: TeamType) -> impl Iterator<Item = &FieldedPlayer> {
        self.get_players_on_pitch().filter(move |p| p.stats.team == team)
    }
    pub fn add_new_player_to_field(&mut self, player_stats: PlayerStats, position: Position) -> Result<PlayerID> {
        if self.board[position].is_some() {
            return Err(Box::new(IllegalMovePosition { position }));
        }

        let id = match self
            .fielded_players
            .iter()
            .enumerate()
            .find(|(_, player)| player.is_none())
        {
            Some((id, _)) => id,
            None => panic!("Not room in gamestate of another fielded player!"),
        };

        self.board[position] = Some(id);
        self.fielded_players[id] = Some(FieldedPlayer {
            id,
            stats: player_stats,
            position,
            status: PlayerStatus::Up,
            used: false,
            moves: 0,
            used_skills: HashSet::new(),
        });
        Ok(id)
    }

    pub fn unfield_player(&mut self, id: PlayerID, place: DugoutPlace) -> Result<()> {
        if let BallState::Carried(carrier_id) = self.ball {
            assert_ne!(carrier_id, id);
        }
        if matches!(self.info.active_player, Some(active_id) if active_id == id) {
            self.info.active_player = None;
        }

        let FieldedPlayer { stats, position, .. } = self.fielded_players[id].take().unwrap();

        self.dugout_add_new_player(stats, place);

        self.board[position] = None;
        Ok(())
    }

    pub fn unfield_all_players(&mut self) -> Result<()> {
        #[allow(clippy::needless_collect)]
        let player_id_on_pitch: Vec<PlayerID> = self.get_players_on_pitch().map(|player| player.id).collect();

        player_id_on_pitch
            .into_iter()
            .for_each(|id| self.unfield_player(id, DugoutPlace::Reserves).unwrap());
        Ok(())
    }
    pub fn clear_all_players(&mut self) -> Result<()> {
        self.unfield_all_players().unwrap();
        self.dugout_players = Default::default();
        Ok(())
    }

    pub fn step_with_roll_or_action(&mut self, roll_or_action: SomeProcInput) -> MicroStepState {
        // This function should eventually be on only one called from mcts bot's apply action.
        // it either takes an action or a roll results and applies micro_step until either:
        //  - needs a roll result
        //  - needs an action
        //  - game over
        //
        //  To do this we need to change micro_step(): it should be a private function that takes a
        //  ProcInput and returns a MicroStepState (GameOver, NeedAction, NeedRoll, or RunAgain). If:
        //  - RunAgain: call micro_step again with ProcInput::Nothing
        //  - NeedAction(aa): set available_actions to aa and return
        //  - NeedRoll(requested_roll): set pending_roll to requested_roll and return
        //  - GameOver: set game_over to true and return

        // an we need to refactor step() to call this function with an action.
        // Step however is only called when the dicepolicy is not RegisterRolls, so if it gets a
        // requested roll back it will just resolve it and call this function again.

        let mut proc_input = ProcInput::from(roll_or_action);
        loop {
            let micro_step_state = self.micro_step(proc_input);
            if micro_step_state == MicroStepState::RunAgain {
                proc_input = ProcInput::Nothing;
                continue;
            } else {
                return micro_step_state;
            }
        }
    }

    fn micro_step(&mut self, proc_input: ProcInput) -> MicroStepState {
        if self.info.game_over {
            return MicroStepState::GameOver;
        }

        // check that arg is correct - can consider doing debug asserts here..
        match proc_input {
            ProcInput::Nothing => {
                assert!(self.pending_roll.is_none());
                assert!(self.available_actions.is_empty());
            }
            ProcInput::Action(action) => {
                assert!(self.pending_roll.is_none());
                assert!(self.is_legal_action(&action));
            }
            ProcInput::Roll(roll_result) => {
                assert!(self
                    .pending_roll
                    .expect("micro_step called with ProcInput::Roll but no pending_roll")
                    .is_compatible(roll_result));
                assert!(self.available_actions.is_empty());
            }
        }
        let mut top_proc = self
            .proc_stack
            .pop()
            .expect("proc_stack was empty at start of micro_step - should never happen");

        match &proc_input {
            ProcInput::Action(a) => crate::game_log!(self, "STEPPING: {:?}\n  action={:?}", top_proc, a),
            ProcInput::Roll(_) => crate::game_log!(self, "STEPPING: {:?}\n  input={:?}", top_proc, proc_input),
            ProcInput::Nothing => crate::game_log!(self, "STEPPING: {:?}", top_proc),
        }

        let proc_return = top_proc.step(self, proc_input);

        crate::game_log!(self, "  result:   {:?}", proc_return);

        // Conditional reset: NeedAction(_) overwrites available_actions
        // immediately below, NeedActionInPlace was just written by the
        // producer in `step` (and would be lost by an unconditional reset
        // here). For every other transition, clear stale AA + path buffer
        // contents. Note: clearing AA only drops the (small) AvailableActions
        // body; the `Box<AvailableActions>` allocation is reused. The path
        // buffer's `FullPitch` allocation is also reused — we just drop the
        // Arc<Node> payload (release search results) via `clear_path_buffer`.
        if !matches!(proc_return, ProcState::NeedAction(_) | ProcState::NeedActionInPlace) {
            if self.available_actions.has_paths {
                self.clear_path_buffer();
            }
            *self.available_actions = AvailableActions::default();
        }
        self.pending_roll = None;

        match proc_return {
            ProcState::NotDoneNewProcs(new_procs) => {
                self.proc_stack.push(top_proc);
                self.proc_stack.extend(new_procs);
                MicroStepState::RunAgain
            }
            ProcState::DoneNewProcs(new_procs) => {
                self.proc_stack.extend(new_procs);
                MicroStepState::RunAgain
            }
            ProcState::NotDoneNew(new_proc) => {
                self.proc_stack.push(top_proc);
                self.proc_stack.push(new_proc);
                MicroStepState::RunAgain
            }
            ProcState::DoneNew(new_proc) => {
                self.proc_stack.push(new_proc);
                MicroStepState::RunAgain
            }
            ProcState::NotDone => {
                self.proc_stack.push(top_proc);
                MicroStepState::RunAgain
            }
            ProcState::Done => MicroStepState::RunAgain,
            ProcState::NeedAction(aa) => {
                // Old-style producer returns a freshly-built Box. Drop the
                // currently-held AA (with any stale paths) and adopt the new one.
                if self.available_actions.has_paths {
                    self.clear_path_buffer();
                }
                self.available_actions = aa;
                self.proc_stack.push(top_proc);
                MicroStepState::NeedAction
            }
            ProcState::NeedActionInPlace => {
                // Producer wrote `state.available_actions` (and `path_buffer`
                // if `has_paths`) during `step`. Nothing to copy.
                self.proc_stack.push(top_proc);
                MicroStepState::NeedAction
            }
            ProcState::NeedRoll(requested_roll) => {
                self.proc_stack.push(top_proc);
                self.pending_roll = Some(requested_roll);
                MicroStepState::NeedRoll
            }
        }
    }

    pub fn step(&mut self, action: Action) -> Result<()> {
        // Match the legacy step() behavior of dropping the action when
        // nothing is asking for one — initial-state setups and a few
        // post-transition paths in GameStateBuilder pass a placeholder
        // action just to drive the engine to the next decision.
        let initial_state = if self.available_actions.is_empty() && self.pending_roll.is_none() {
            // No input expected: drive the engine forward with Nothing.
            let mut s = MicroStepState::RunAgain;
            while s == MicroStepState::RunAgain {
                s = self.micro_step(ProcInput::Nothing);
            }
            s
        } else {
            self.step_with_roll_or_action(SomeProcInput::Action(action))
        };

        let mut micro_step_state = initial_state;
        loop {
            if micro_step_state == MicroStepState::GameOver || micro_step_state == MicroStepState::NeedAction {
                break;
            } else if micro_step_state == MicroStepState::NeedRoll {
                let roll_result = self.get_roll_result();
                micro_step_state = self.step_with_roll_or_action(SomeProcInput::Roll(roll_result));
            } else {
                panic!(
                    "step_with_roll_or_action returned unexpected state: {:?}",
                    micro_step_state
                );
            }
        }
        debug_assert!(!self.available_actions.is_empty() || self.info.game_over);
        Ok(())
    }

    fn get_roll_result(&mut self) -> RollResult {
        let requested_roll = self
            .pending_roll
            .expect("get_roll_result called but no pending_roll was set");
        match &mut self.dice_mode {
            DiceMode::RollDice => resolve_with_rng(requested_roll, &mut self.rng),
            DiceMode::FixedDice(fixes) => resolve_from_fixes(requested_roll, fixes),
            DiceMode::DicePolicy(policy) => policy.resolve(requested_roll, &mut self.rng),
            DiceMode::RegisterRolls => unreachable!(
                "DiceMode::RegisterRolls: micro_step pauses on NeedRoll, so \
                 get_roll_result must be reached only via a registered roll \
                 (handled inline in micro_step) — never as a direct call"
            ),
        }
    }

    pub fn is_legal_action(&self, action: &Action) -> bool {
        if self.available_actions.is_legal_non_path_action(*action) {
            return true;
        }
        // Path-style positional actions are backed by `path_buffer`.
        if let Action::Positional(at, pos) = action {
            if let Some(node) = self.get_paths().and_then(|p| p[*pos].as_ref()) {
                return node.get_action_type() == *at;
            }
        }
        false
    }

    /// Borrow the path offerings produced by the last `MoveAction` /
    /// `BlockAction`. Returns `None` unless `available_actions.has_paths` is
    /// true — even when a buffer was allocated for an earlier decision, it's
    /// stale until a producer refills it.
    pub fn get_paths(&self) -> Option<&FullPitch<Option<std::sync::Arc<super::pathing::Node>>>> {
        if self.available_actions.has_paths {
            self.path_buffer.as_deref()
        } else {
            None
        }
    }

    /// Move the path node at `pos` out of the buffer. Used by `MoveAction`
    /// and `BlockAction` to consume the chosen offering. Returns `None` if
    /// `has_paths` is false or there's no offering at that position.
    pub fn take_path(&mut self, pos: Position) -> Option<std::sync::Arc<super::pathing::Node>> {
        if !self.available_actions.has_paths {
            return None;
        }
        self.path_buffer.as_mut().and_then(|buf| buf[pos].take())
    }

    /// Gather every legal action — simple, positional, and path-style. Sorts
    /// at the end because `AvailableActions::simple` is a `HashSet` whose
    /// iteration order is randomized per run, and `RandomBot::get_action`
    /// indexes by RNG-generated position — sort makes seeded games (and the
    /// UI snapshot tests that depend on them) deterministic.
    pub fn get_all_actions(&self) -> Vec<Action> {
        let mut out = Vec::with_capacity(32);
        self.available_actions.collect_non_path_actions(&mut out);
        if let Some(paths) = self.get_paths() {
            for (pos, node_opt) in paths.iter_position() {
                if let Some(node) = node_opt {
                    out.push(Action::Positional(node.get_action_type(), pos));
                }
            }
        }
        out.sort();
        out
    }

    /// Lazy-allocate then return a mutable handle to the path buffer. Used by
    /// `MoveAction` / `BlockAction` producers to fill in place. The flag on
    /// `available_actions` is the producer's responsibility to set.
    pub(crate) fn take_path_buffer(&mut self) -> Box<FullPitch<Option<std::sync::Arc<super::pathing::Node>>>> {
        self.path_buffer.take().unwrap_or_default()
    }

    /// Drop every Arc payload in the path buffer in place, retaining the 4KB
    /// allocation. Cheap when most slots are already `None`. Used both by
    /// producers (before refill) and by `micro_step` when a non-NeedAction
    /// transition makes the buffer's contents stale.
    pub(crate) fn clear_path_buffer(&mut self) {
        if let Some(buf) = self.path_buffer.as_mut() {
            for slot in buf.iter_mut() {
                *slot = None;
            }
        }
        self.available_actions.has_paths = false;
    }

    /// Restore a buffer that was taken via `take_path_buffer`, marking
    /// `has_paths` so consumers can read it. Cheap (two pointer writes).
    pub(crate) fn install_path_buffer(&mut self, buf: Box<FullPitch<Option<std::sync::Arc<super::pathing::Node>>>>) {
        self.path_buffer = Some(buf);
        self.available_actions.has_paths = true;
    }

    pub fn is_setup_legal(&self, team: TeamType) -> bool {
        let mut north_wing = 0;
        let mut south_wing = 0;
        let mut line_of_scrimage = 0;
        let num_players_on_pitch = self.get_players_on_pitch_in_team(team).count();
        let num_players_on_bench = self
            .get_dugout()
            .filter(|player| player.stats.team == team && player.place == DugoutPlace::Reserves)
            .count();
        let num_available_players = num_players_on_bench + num_players_on_pitch;
        let team_size = self.board_dims.team_size;
        let min_people_on_pitch = team_size.min(num_available_players);
        let min_people_on_scrimage = 3.min(num_available_players);

        if num_players_on_pitch < min_people_on_pitch || num_players_on_pitch > team_size {
            return false;
        }
        let line_of_scrimage_x = self.get_line_of_scrimage_x(team);
        let los_y_range = self.board_dims.los_y_range();
        let south_wing_y_range = self.board_dims.south_wing_y_range();
        let north_wing_y_range = self.board_dims.north_wing_y_range();

        for pos in self.get_players_on_pitch_in_team(team).map(|p| p.position) {
            if self.is_out(pos)
                || (team == TeamType::Home && pos.x < line_of_scrimage_x)
                || (team == TeamType::Away && pos.x > line_of_scrimage_x)
            {
                return false;
            }

            if pos.x == line_of_scrimage_x && los_y_range.contains(&pos.y) {
                line_of_scrimage += 1;
            } else if south_wing_y_range.contains(&pos.y) {
                south_wing += 1;
            } else if north_wing_y_range.contains(&pos.y) {
                north_wing += 1;
            }
        }
        north_wing <= 2 && south_wing <= 2 && line_of_scrimage >= min_people_on_scrimage
    }

    pub fn step_simple(&mut self, action: SimpleAT) {
        self.step(Action::Simple(action)).unwrap();
        self.assert_fixes_empty();
    }

    pub fn step_positional(&mut self, action: PosAT, position: Position) {
        self.step(Action::Positional(action, position)).unwrap();
        self.assert_fixes_empty();
    }

    /// In `FixedDice` mode, assert the queue has been fully consumed.
    /// In other modes this is a no-op — those modes have no queue.
    pub fn assert_fixes_empty(&self) {
        if let DiceMode::FixedDice(fixes) = &self.dice_mode {
            fixes.assert_is_empty();
        }
    }

    pub fn get_interception_positions(from: Position, to: Position) -> impl Iterator<Item = Position> {
        let fr_y = from.y as i16;
        let to_y = to.y as i16;
        let fr_x = from.x as i16;
        let to_x = to.x as i16;
        let max_x = max(fr_x, to_x) + 1;
        let min_x = min(fr_x, to_x) - 1;
        let max_y = max(fr_y, to_y) + 1;
        let min_y = min(fr_y, to_y) - 1;
        let dx = to_x - fr_x;
        let dy = to_y - fr_y;
        let distance_squared = dx * dx + dy * dy;
        let distance = (distance_squared as f32).sqrt();

        ((min_x)..=(max_x))
            .cartesian_product((min_y)..=(max_y))
            .filter(move |(x, y)| ((*x - fr_x).pow(2) + (*y - fr_y).pow(2)) <= distance_squared)
            .filter(move |(x, y)| ((*x - to_x).pow(2) + (*y - to_y).pow(2)) <= distance_squared)
            .filter(move |(x, y)| 1.2 >= ((to_x - x) * dy - (to_y - y) * dx).abs() as f32 / distance)
            .map(|(x, y)| Position::new((x as i8, y as i8)))
            .filter(|pos| !pos.is_out())
            .filter(move |pos| *pos != from && *pos != to)
    }

    pub fn get_intercepters(&self, team: TeamType, from: Position, to: Position) -> Vec<(Position, D6Target)> {
        // TODO: thos function needs tests!!!
        GameState::get_interception_positions(from, to)
            .filter_map(|pos| {
                self.get_player_at(pos)
                    .filter(|p| p.stats.team == team && p.can_catch())
                    .and_then(|p| self.get_catch_target(p.id).ok())
                    .map(|target| (pos, target))
            })
            .collect::<Vec<(Position, D6Target)>>()
    }

    pub fn get_pass_modifier(&self, id: usize, from: Position, to: Position) -> Option<i8> {
        // TODO: move this whole function to pathing and cache it there.
        const MATRIX: [[i8; 14]; 14] = [
            [8, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 9],
            [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 9],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 9],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 9, 9],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 9, 9],
            [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 9, 9, 9],
            [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 9, 9, 9],
            [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 9, 9, 9, 9],
            [2, 2, 2, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9],
            [3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9],
            [3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ];
        // 8 - passing to oneself, not possible
        // 9 - hail mary pass - skill not implemented
        // 0 - quick pass
        // 1 - short pass
        // 2 - long pass
        // 3 - long bomb

        let delta = to - from;
        let (dx, dy) = (delta.dx.unsigned_abs() as usize, delta.dy.unsigned_abs() as usize);
        let distance_modifier = MATRIX.get(dx)?.get(dy)?;
        if *distance_modifier == 8 {
            panic!("Passing to oneself is not possible: from {} to {}", from, to);
        } else if *distance_modifier == 9 {
            return None;
        }

        let Some(team) = self.get_player(id).ok().map(|p| p.stats.team) else {
            //return None;
            panic!("Player not found");
        };
        let tackle_zones = self
            .get_adj_players(from)
            .filter(|adj_p| adj_p.stats.team != team && adj_p.has_tackle_zone())
            .count() as i8;
        // TODO: weather effect
        let sum_modifiers = -tackle_zones - *distance_modifier;
        Some(sum_modifiers)
    }
    pub fn get_pass_target(&self, id: usize, from: Position, to: Position) -> Option<D6Target> {
        let Some(player) = self.get_player(id).ok() else {
            //return None;
            panic!("Player not found");
        };
        let mut target = player.pass_target();
        let modifiers = self.get_pass_modifier(id, from, to)?;
        target.add_modifer(modifiers);
        Some(target)
    }
}

#[cfg(test)]
mod gamestate_tests {
    use itertools::Itertools;
    use serde_json;

    use crate::{
        core::{
            dices::D6,
            gamestate::{BuilderState, GameState},
            model::{BallState, DugoutPlace, PlayerStats, Position, Result, TeamType, HEIGHT_, WIDTH, WIDTH_},
        },
        standard_state,
    };
    use std::{collections::HashSet, io::Write, iter::repeat_with};

    use super::GameStateBuilder;

    #[test]
    fn set_half_relabels_and_drops_pending_half() {
        use crate::core::procedures::AnyProc;
        use crate::core::table::SimpleAT;

        let mut state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 1 })
            .add_home_player(Position::new((5, 5)))
            .add_away_player(Position::new((10, 5)))
            .build();
        assert_eq!(state.info.half, 1);

        state.set_half(2);
        assert_eq!(state.info.half, 2);
        let half_procs = state.proc_stack.iter().filter(|p| matches!(p, AnyProc::Half(_))).count();
        assert_eq!(half_procs, 1, "the pending unstarted half should be dropped");

        // Ending every remaining turn must now end the game — no second-half
        // kickoff in between. From (home 1, away 0) that is 15 end-turns to
        // reach 8/8, after which GameOver runs.
        let mut guard = 0;
        while !state.info.game_over {
            state.step_simple(SimpleAT::EndTurn);
            guard += 1;
            assert!(guard <= 16, "game did not end after the relabeled half");
        }
        assert_eq!(state.info.half, 2);
    }

    #[test]
    fn symmetric_interception_positions() {
        for (dx, dy) in (0..14).cartesian_product(0..14) {
            let from = Position::new((3, 3));
            let to = from + (dx, dy);
            let to_from = GameState::get_interception_positions(from, to);
            let from_to = GameState::get_interception_positions(to, from);
            assert_eq!(to_from.collect::<HashSet<_>>(), from_to.collect::<HashSet<_>>());
        }
    }
    #[test]
    fn interception_positions() {
        let correct_thing = [
            [
                ".......................",
                ".......................",
                ".....oo.................",
                "....XooX...............",
                ".....oo................",
                ".......................",
                ".......................",
            ],
            [
                ".......................",
                ".......................",
                ".......................",
                "....oooX................",
                "....Xooo..............",
                ".....................",
                ".......................",
            ],
            [
                ".......................",
                ".......................",
                ".....ooX...............",
                "....oooo................",
                "....Xoo...............",
                ".....................",
                ".......................",
            ],
            [
                ".......................",
                "......oX...............",
                ".....ooo...............",
                "....ooo.................",
                "....Xo................",
                ".....................",
                ".......................",
            ],
        ];
        for s in correct_thing {
            let mut to_from = Vec::new();
            let mut correct_intercepters = HashSet::new();
            for (y, line) in s.iter().enumerate() {
                for (x, c) in line.chars().enumerate() {
                    let p = Position::new((x as i8 + 2, y as i8 + 2));
                    if c == 'X' {
                        to_from.push(p);
                    } else if c == 'o' {
                        correct_intercepters.insert(p);
                    } else {
                        assert_eq!(c, '.');
                    }
                }
            }

            assert!(to_from.len() == 2);
            let from = to_from.pop().unwrap();
            let to = to_from.pop().unwrap();

            let calc_intercepters = GameState::get_interception_positions(from, to).collect::<HashSet<Position>>();
            // assert_eq!(calc_intercepters, correct_intercepters);
            if calc_intercepters != correct_intercepters {
                // create ss, a clone of s.
                println!("from: {:?}, to: {:?}", from, to);
                println!("calc_intercepters: {:?}", calc_intercepters);
                let mut ss: Vec<Vec<char>> = (0..HEIGHT_).map(|_| vec!['.'; WIDTH]).collect();
                let correctly_addeds = correct_intercepters.intersection(&calc_intercepters);
                println!("correctly_addeds: {:?}", correctly_addeds);
                let wrongly_addeds = calc_intercepters.difference(&correct_intercepters);
                println!("wrongly_addeds: {:?}", wrongly_addeds);
                for wrongly_added in wrongly_addeds {
                    assert!(!wrongly_added.is_out());
                    let (x, y) = wrongly_added.to_usize().unwrap();
                    ss[y][x] = 'a';
                }
                let wrongly_missings = correct_intercepters.difference(&calc_intercepters);
                println!("wrongly_missings: {:?}", wrongly_missings);
                for wrongly_missing in wrongly_missings {
                    assert!(!wrongly_missing.is_out());
                    let (x, y) = wrongly_missing.to_usize().unwrap();
                    ss[y][x] = 'm';
                }
                let (x, y) = from.to_usize().unwrap();
                ss[y][x] = 'X';
                let (x, y) = to.to_usize().unwrap();
                ss[y][x] = 'X';

                let error_strs: Vec<String> = ss
                    .iter()
                    .enumerate()
                    .map(|(i, line)| format!("{}: {}", i, line.iter().collect::<String>()).to_string())
                    .filter(|s| s.contains('a') || s.contains('m') || s.contains('X'))
                    .collect();
                let error_str: String = error_strs.join("\n");
                assert_eq!(
                    calc_intercepters,
                    correct_intercepters,
                    "\n{}\n\nshould have been\n{}\n",
                    error_str,
                    s.iter().join("\n")
                );
            }
        }
    }

    /// The kickoff aim must be the centre of the *receiving* half, and the two
    /// branches must be exact mirrors of each other.
    ///
    /// This asserts the *property*, not the formula. The previous version of
    /// this test restated `WIDTH_ / 4` and `WIDTH_ * 3 / 4` verbatim, so it
    /// pinned the off-by-one it was supposed to catch (plan 023 B1): at every
    /// width divisible by 4, `3w/4` is one column deeper than the mirror of
    /// `w/4`, which handed the receiving team a systematically different
    /// kickoff depending on which side it was.
    #[test]
    fn kickoff_aim_is_centre_of_receiving_half_and_mirror_symmetric() {
        use crate::core::model::TeamType;
        let state = GameStateBuilder::new().build();
        let dims = state.board_dims;
        let home_kicks = state.get_best_kickoff_aim_for(TeamType::Home);
        let away_kicks = state.get_best_kickoff_aim_for(TeamType::Away);

        assert_eq!(home_kicks.y, away_kicks.y, "both aims sit on the same row");
        assert_eq!(
            away_kicks.x,
            dims.width - 1 - home_kicks.x,
            "aims must mirror under x -> width-1-x (got home {} / away {} on width {})",
            home_kicks.x,
            away_kicks.x,
            dims.width
        );

        // The ball is kicked *into the receiving team's half*.
        assert!(
            dims.is_on_team_side(home_kicks, TeamType::Away),
            "Home's kick must land in Away's half, got {home_kicks:?}"
        );
        assert!(
            dims.is_on_team_side(away_kicks, TeamType::Home),
            "Away's kick must land in Home's half, got {away_kicks:?}"
        );
        assert!(!dims.is_out(home_kicks) && !dims.is_out(away_kicks), "aims are in bounds");
    }

    #[test]
    fn test_unfield_all_players() {
        let mut state = GameStateBuilder::new()
            .add_home_players(&[(1, 2), (2, 2), (3, 1)])
            .add_away_players(&[(5, 2), (5, 5), (2, 3)])
            .add_ball((3, 2))
            .build();
        assert_eq!(state.get_players_on_pitch().count(), 6);
        state.unfield_all_players().unwrap();

        assert_eq!(state.get_players_on_pitch().count(), 0);
    }
    #[test]
    fn test_clear_all_players() {
        let mut state = GameStateBuilder::new()
            .add_home_players(&[(1, 2), (2, 2), (3, 1)])
            .add_away_players(&[(5, 2), (5, 5), (2, 3)])
            .add_ball((3, 2))
            .build();
        assert_eq!(state.get_dugout().count(), 0);
        assert_eq!(state.get_players_on_pitch().count(), 6);

        state.clear_all_players().unwrap();

        assert_eq!(state.get_players_on_pitch().count(), 0);
        assert_eq!(state.get_dugout().count(), 0);
    }
    #[test]
    fn test_build_game_custom_turn() {
        let state = GameStateBuilder::new()
            .set_state(BuilderState::Turn { turn: 3 })
            .build();
        assert_eq!(state.info.home_turn, 3);
        assert_eq!(state.info.away_turn, 2);
    }

    #[test]
    fn test_kickoff_game_custom_turn() {
        let state = GameStateBuilder::new()
            .set_state(BuilderState::Kickoff { turn: 7 })
            .build();
        assert_eq!(state.info.home_turn, 6);
        assert_eq!(state.info.away_turn, 6);
    }
    #[test]
    fn state_from_str() {
        let mut field = "".to_string();
        field += " aa\n";
        field += " Aa\n";
        field += "h  \n";
        let first_pos = Position::new((5, 1));
        let state = GameStateBuilder::new().add_str(first_pos, &field).build();
        assert_eq!(
            state.get_player_at(Position::new((5, 3))).unwrap().stats.team,
            TeamType::Home
        );

        assert_eq!(
            state.get_player_at(Position::new((6, 2))).unwrap().stats.team,
            TeamType::Away
        );

        let id = state.get_player_id_at_coord(6, 2).unwrap();
        assert_eq!(state.ball, BallState::Carried(id));
    }

    #[test]
    fn player_unique_id_and_correct_positions() {
        let state = standard_state();

        let mut ids = HashSet::new();
        for x in 0..WIDTH_ {
            for y in 0..HEIGHT_ {
                let pos = Position::new((x, y));
                if let Some(player) = state.get_player_at(pos) {
                    assert_eq!(player.position, pos);
                    assert!(ids.insert(player.id));
                }
            }
        }
        assert_eq!(0, ids.into_iter().filter(|id| *id >= 22).count());
    }

    #[test]
    fn adjescent() {
        let state = standard_state();
        let num_adj = state.get_adj_players(Position::new((2, 2))).count();
        assert_eq!(num_adj, 3);
    }

    #[test]
    fn mutate_player() {
        let mut state = standard_state();

        assert!(!(state.get_player(0).unwrap().used));
        state.get_mut_player(0).unwrap().used = true;
        assert!(state.get_player(0).unwrap().used);
    }

    #[test]
    fn move_player() -> Result<()> {
        let mut state = standard_state();
        let id = 1;
        let old_pos = Position::new((2, 2));
        let new_pos = Position::new((crate::core::model::WIDTH_ / 2, crate::core::model::HEIGHT_ / 2));

        assert_eq!(state.get_player_id_at(old_pos), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, old_pos);
        assert!(state.get_player_id_at(new_pos).is_none());

        state.move_player(id, new_pos)?;

        assert!(state.get_player_id_at(old_pos).is_none());
        assert_eq!(state.get_player_id_at(new_pos), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, new_pos);
        Ok(())
    }
    #[test]
    fn field_a_player() -> Result<()> {
        let mut state = standard_state();
        let player_stats = PlayerStats::new_lineman(TeamType::Home);
        let position = Position::new((crate::core::model::WIDTH_ / 2, crate::core::model::HEIGHT_ / 2));

        assert!(state.get_player_id_at(position).is_none());

        let id = state.add_new_player_to_field(player_stats, position).unwrap();

        assert_eq!(state.get_player_id_at(position), Some(id));
        assert_eq!(state.get_player(id).unwrap().position, position);

        state.unfield_player(id, DugoutPlace::Reserves)?;

        assert!(state.get_player_id_at(position).is_none());
        Ok(())
    }
    #[test]
    fn rng_seed_in_gamestate() -> Result<()> {
        use rand::Rng;
        let mut state = standard_state();
        state.set_dice_mode(crate::core::gamestate::DiceMode::RollDice);
        let seed = 5;
        state.set_seed(seed);

        fn get_random_rolls(state: &mut GameState) -> Vec<D6> {
            repeat_with(|| state.rng.gen::<D6>()).take(200).collect()
        }

        let numbers: Vec<D6> = get_random_rolls(&mut state);
        let different_numbers = get_random_rolls(&mut state);
        assert_ne!(numbers, different_numbers);

        state.set_seed(seed);
        let same_numbers = get_random_rolls(&mut state);

        assert_eq!(numbers, same_numbers);

        Ok(())
    }
    #[test]
    fn serialize_gamestate() {
        let state = standard_state();

        let serialized = serde_json::to_string(&state).unwrap();
        let mut file = std::fs::File::create("serialized_test.json").unwrap();
        file.write_all(serialized.as_bytes()).unwrap();

        let json_str = std::fs::read_to_string("serialized_test.json").unwrap();
        let deserialized: GameState = serde_json::from_str(&json_str).unwrap();
        assert_eq!(state, deserialized);
        std::fs::remove_file("serialized_test.json").unwrap();
    }

    use crate::core::dices::{RequestedRoll, RollResult};
    use crate::core::gamestate::DiceMode;
    use crate::core::model::{Action, MicroStepState, SomeProcInput};
    use crate::core::table::PosAT;

    #[test]
    #[should_panic(expected = "FixedDice access requires DiceMode::FixedDice")]
    fn fix_d6_panics_in_rolldice_mode() {
        let mut state = standard_state();
        state.set_dice_mode(DiceMode::RollDice);
        state.fix_d6(5);
    }

    #[test]
    #[should_panic(expected = "no pending_roll")]
    fn step_with_roll_or_action_panics_without_pending_roll() {
        // Feeding a Roll when the engine isn't paused on a roll request
        // is a programming error — micro_step asserts pending_roll is set.
        let mut state = standard_state();
        state.set_dice_mode(DiceMode::RegisterRolls);
        let _ = state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Pass));
    }

    #[test]
    #[should_panic(expected = "DiceMode::RegisterRolls")]
    fn step_panics_in_registerrolls_mode() {
        // step() resolves rolls inline via get_roll_result; that branch
        // is unreachable! in RegisterRolls mode — the MCTS caller is
        // expected to drive chance nodes via step_with_roll_or_action.
        let start_pos = Position::new((2, 5));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(td_pos)
            .build();
        state.set_dice_mode(DiceMode::RegisterRolls);
        state.step(Action::Positional(PosAT::StartMove, start_pos)).unwrap();
        // Move onto the ball triggers a pickup D6PassFail; step()'s
        // inline roll resolution then hits the unreachable!.
        state.step(Action::Positional(PosAT::Move, td_pos)).unwrap();
    }

    #[test]
    #[should_panic(expected = "FixedDice queue empty for D6")]
    fn fixed_dice_empty_queue_panics_with_diagnostic() {
        // Setup: state with FixedDice mode and an empty queue. Drive the
        // engine into a roll request and verify the panic message names D6.
        let start_pos = Position::new((2, 5));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(td_pos)
            .build();
        // Mode is FixedDice(empty) by default after build(). The pickup
        // request (D6PassFail) will look for a d6 fix and find none.
        state.step_positional(PosAT::StartMove, start_pos);
        state.step_positional(PosAT::Move, td_pos);
    }

    #[test]
    fn dice_policy_unmatched_rolls_use_rng() {
        // Pickup forced to succeed by policy; the scatter/bounce/etc.
        // dice that the policy doesn't override are drawn from RNG.
        // Two seeded runs with different seeds should diverge on the
        // unmatched-roll outcomes (or, at minimum, both reach Success
        // without panicking — the policy is total).
        let start_pos = Position::new((2, 5));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(td_pos)
            .build();
        state.set_dice_mode(DiceMode::DicePolicy(
            crate::core::dices::DicePolicy::SucceedAtOrEasier {
                d6: crate::core::dices::D6Target::ThreePlus,
                sum2d6: crate::core::dices::Sum2D6Target::SevenPlus,
                block_dice: crate::core::dices::BlockDicePolicy::Default,
            },
        ));
        state.set_seed(42);
        state.step_positional(PosAT::StartMove, start_pos);
        state.step_positional(PosAT::Move, td_pos);
        assert_eq!(state.home.score, 1);
    }

    #[test]
    fn register_rolls_round_trip() {
        // RegisterRolls: step_with_roll_or_action drives the engine and
        // pauses on NeedRoll so the caller can supply the chance outcome.
        let start_pos = Position::new((2, 5));
        let td_pos = Position::new((1, 5));
        let mut state = GameStateBuilder::new()
            .add_home_player(start_pos)
            .add_ball_pos(td_pos)
            .build();
        state.set_dice_mode(DiceMode::RegisterRolls);

        let after_start =
            state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::StartMove, start_pos)));
        assert_eq!(
            after_start,
            MicroStepState::NeedAction,
            "StartMove should yield the next Move-target decision"
        );

        let after_move = state.step_with_roll_or_action(SomeProcInput::Action(Action::Positional(PosAT::Move, td_pos)));
        assert_eq!(
            after_move,
            MicroStepState::NeedRoll,
            "moving onto the ball should pause for the pickup roll"
        );
        assert!(
            matches!(state.pending_roll, Some(RequestedRoll::D6PassFail(_))),
            "expected pickup PassFail pending, got {:?}",
            state.pending_roll
        );

        state.step_with_roll_or_action(SomeProcInput::Roll(RollResult::Pass));
        assert_eq!(state.home.score, 1, "successful pickup scores");
    }
}

/// Runtime board sizing (plan 017): one binary plays a board smaller than the
/// compiled capacity, chosen at runtime via `with_board_dims` — no recompile.
#[cfg(test)]
mod runtime_board_dims_tests {
    use super::GameStateBuilder;
    use crate::bots::RandomBot;
    use crate::core::game_runner::BotGameRunnerBuilder;
    use crate::core::gamestate::BuilderState;
    use crate::core::model::{BoardDims, Position, TeamType, TEAM_SIZE};

    // A 16x9 engine board (14x7 playable) with 3 players — the plan's small tier.
    // Runs whenever the compiled capacity is at least this big.
    const W: i8 = 16;
    const H: i8 = 9;
    const PLAYERS: usize = 3;

    fn fits_capacity() -> bool {
        (crate::core::model::WIDTH as i8) >= W && (crate::core::model::HEIGHT as i8) >= H && TEAM_SIZE >= PLAYERS
    }

    #[test]
    fn runtime_small_board_geometry() {
        if !fits_capacity() {
            return;
        }
        let dims = BoardDims::new(W, H, PLAYERS);
        let state = GameStateBuilder::new().with_board_dims(dims).build();
        assert_eq!(state.board_dims, dims, "runtime dims propagate to the built state");

        // Logical out-of-bounds: last playable column is width-2 = 14.
        assert!(
            !state.is_out(Position::new((14, 4))),
            "(14,4) is the last playable column"
        );
        assert!(
            state.is_out(Position::new((15, 4))),
            "column 15 is the logical OOB border"
        );
        // The physical array is larger than the logical board at default capacity,
        // yet squares beyond the logical board are still out of bounds.
        crate::skip_if_board_smaller_than!(28, 17);
        assert!(
            !Position::new((20, 4)).is_out(),
            "col 20 is inside the *physical* 28-wide array (capacity check)",
        );
        assert!(
            state.is_out(Position::new((20, 4))),
            "...but logically out on the 16-wide board"
        );
    }

    #[test]
    fn runtime_small_board_derived_geometry() {
        if !fits_capacity() {
            return;
        }
        let state = GameStateBuilder::new()
            .with_board_dims(BoardDims::new(W, H, PLAYERS))
            .build();
        // Symmetric line of scrimmage around width/2 = 8.
        assert_eq!(state.get_line_of_scrimage_x(TeamType::Home), 8);
        assert_eq!(state.get_line_of_scrimage_x(TeamType::Away), 7);
        // End zones sit on the innermost playable columns.
        assert_eq!(state.get_endzone_x(TeamType::Home), 1);
        assert_eq!(state.get_endzone_x(TeamType::Away), W - 2);
        // Fewer than 7 players → degenerate kickoff (no event table).
        assert!(!state.board_dims.kickoff_table_enabled());
    }

    #[test]
    fn runtime_small_board_plays_full_game() {
        if !fits_capacity() {
            return;
        }
        // A full game from the coin toss on the runtime-small board exercises
        // setup, kickoff, bounces and throw-ins end-to-end — all without a rebuild.
        for seed in 0..5u64 {
            let state = GameStateBuilder::new()
                .with_board_dims(BoardDims::new(W, H, PLAYERS))
                .set_state(BuilderState::CoinToss)
                .build();
            assert_eq!(state.board_dims.width, W);
            let mut bot_game = BotGameRunnerBuilder::new()
                .set_home_bot(Box::new(RandomBot::new()))
                .set_away_bot(Box::new(RandomBot::new()))
                .set_state(state)
                .set_seed(seed)
                .build();
            let result = bot_game.run();
            let _ = result; // completing without panic is the assertion
        }
    }

    #[test]
    #[should_panic(expected = "exceeds compiled capacity")]
    fn board_dims_over_capacity_panics() {
        // One column wider than the compiled physical array must be rejected —
        // the arrays can't hold it, so you'd have to recompile larger.
        let over = (crate::core::model::WIDTH as i8) + 2;
        let _ = BoardDims::new(over, H, PLAYERS);
    }
}
