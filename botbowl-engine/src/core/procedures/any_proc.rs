use serde::{Deserialize, Serialize};

use crate::core::model::Procedure;
use crate::core::procedures::ball_procs::{
    Bounce, Catch, Deflect, DeflectOrResolve, Pass, PickupProc, ThrowIn, Touchback, Touchdown,
};

use crate::core::procedures::block_procs::{Block, BlockAction, FollowUp, KnockDown, Push};
use crate::core::procedures::casualty_procs::{Armor, Ejection, Injury};
use crate::core::procedures::game_procs::{
    ChooseKickReceive, CoinToss, GameOver, Half, KOWakeUp, Turn, TurnStunned,
    TurnoverIfPossessionLost,
};
use crate::core::procedures::kickoff_procs::{
    ChangingWeather, Kickoff, KickoffTable, LandKickoff, Setup,
};
use crate::core::procedures::movement_procs::{DodgeProc, GfiProc, MoveAction, StandUp};

use crate::core::procedures::procedure_tools::SimpleProcContainer;

/// Define `AnyProc` and its `Debug`, `name`, `Procedure::step` impls from a
/// single source of truth so each new procedure only has to be added once.
macro_rules! any_proc {
    ( $( $variant:ident($ty:ty) ),* $(,)? ) => {
        #[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
        pub enum AnyProc {
            $( $variant($ty), )*
        }

        impl std::fmt::Debug for AnyProc {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $( Self::$variant(arg0) => f.debug_tuple(stringify!($variant)).field(arg0).finish(), )*
                }
            }
        }

        impl AnyProc {
            pub fn name(&self) -> &'static str {
                match self {
                    $( Self::$variant(_) => stringify!($variant), )*
                }
            }
        }

        impl Procedure for AnyProc {
            fn step(
                &mut self,
                game_state: &mut crate::core::gamestate::GameState,
                input: crate::core::model::ProcInput,
            ) -> crate::core::model::ProcState {
                match self {
                    $( Self::$variant(arg) => arg.step(game_state, input), )*
                }
            }
        }
    };
}

any_proc! {
    Armor(Armor),
    Block(Block),
    BlockAction(BlockAction),
    Bounce(Bounce),
    Catch(SimpleProcContainer<Catch>),
    ChangingWeather(ChangingWeather),
    ChooseKickReceive(ChooseKickReceive),
    CoinToss(CoinToss),
    Deflect(SimpleProcContainer<Deflect>),
    DeflectOrResolve(DeflectOrResolve),
    DodgeProc(SimpleProcContainer<DodgeProc>),
    Ejection(Ejection),
    FollowUp(FollowUp),
    GameOver(GameOver),
    GfiProc(SimpleProcContainer<GfiProc>),
    Half(Half),
    Injury(Injury),
    KOWakeUp(KOWakeUp),
    Kickoff(Kickoff),
    KickoffTable(KickoffTable),
    KnockDown(KnockDown),
    LandKickoff(LandKickoff),
    MoveAction(MoveAction),
    Pass(Pass),
    PickupProc(SimpleProcContainer<PickupProc>),
    Push(Push),
    Setup(Setup),
    StandUp(StandUp),
    ThrowIn(ThrowIn),
    Touchback(Touchback),
    Touchdown(Touchdown),
    Turn(Turn),
    TurnStunned(TurnStunned),
    TurnoverIfPossessionLost(TurnoverIfPossessionLost),
}
