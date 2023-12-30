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
#[derive(Serialize, Deserialize, Clone)]
pub enum AnyProc {
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

impl std::fmt::Debug for AnyProc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Armor(arg0) => f.debug_tuple("Armor").field(arg0).finish(),
            Self::Block(arg0) => f.debug_tuple("Block").field(arg0).finish(),
            Self::BlockAction(arg0) => f.debug_tuple("BlockAction").field(arg0).finish(),
            Self::Bounce(arg0) => f.debug_tuple("Bounce").field(arg0).finish(),
            Self::Catch(arg0) => f.debug_tuple("Catch").field(arg0).finish(),
            Self::ChangingWeather(arg0) => f.debug_tuple("ChangingWeather").field(arg0).finish(),
            Self::ChooseKickReceive(arg0) => {
                f.debug_tuple("ChooseKickReceive").field(arg0).finish()
            }
            Self::CoinToss(arg0) => f.debug_tuple("CoinToss").field(arg0).finish(),
            Self::Deflect(arg0) => f.debug_tuple("Deflect").field(arg0).finish(),
            Self::DeflectOrResolve(arg0) => f.debug_tuple("DeflectOrResolve").field(arg0).finish(),
            Self::DodgeProc(arg0) => f.debug_tuple("DodgeProc").field(arg0).finish(),
            Self::Ejection(arg0) => f.debug_tuple("Ejection").field(arg0).finish(),
            Self::FollowUp(arg0) => f.debug_tuple("FollowUp").field(arg0).finish(),
            Self::GameOver(arg0) => f.debug_tuple("GameOver").field(arg0).finish(),
            Self::GfiProc(arg0) => f.debug_tuple("GfiProc").field(arg0).finish(),
            Self::Half(arg0) => f.debug_tuple("Half").field(arg0).finish(),
            Self::Injury(arg0) => f.debug_tuple("Injury").field(arg0).finish(),
            Self::KOWakeUp(arg0) => f.debug_tuple("KOWakeUp").field(arg0).finish(),
            Self::Kickoff(arg0) => f.debug_tuple("Kickoff").field(arg0).finish(),
            Self::KickoffTable(arg0) => f.debug_tuple("KickoffTable").field(arg0).finish(),
            Self::KnockDown(arg0) => f.debug_tuple("KnockDown").field(arg0).finish(),
            Self::LandKickoff(arg0) => f.debug_tuple("LandKickoff").field(arg0).finish(),
            Self::MoveAction(arg0) => f.debug_tuple("MoveAction").field(arg0).finish(),
            Self::Pass(arg0) => f.debug_tuple("Pass").field(arg0).finish(),
            Self::PickupProc(arg0) => f.debug_tuple("PickupProc").field(arg0).finish(),
            Self::Push(arg0) => f.debug_tuple("Push").field(arg0).finish(),
            Self::Setup(arg0) => f.debug_tuple("Setup").field(arg0).finish(),
            Self::StandUp(arg0) => f.debug_tuple("StandUp").field(arg0).finish(),
            Self::ThrowIn(arg0) => f.debug_tuple("ThrowIn").field(arg0).finish(),
            Self::Touchback(arg0) => f.debug_tuple("Touchback").field(arg0).finish(),
            Self::Touchdown(arg0) => f.debug_tuple("Touchdown").field(arg0).finish(),
            Self::Turn(arg0) => f.debug_tuple("Turn").field(arg0).finish(),
            Self::TurnStunned(arg0) => f.debug_tuple("TurnStunned").field(arg0).finish(),
            Self::TurnoverIfPossessionLost(arg0) => f
                .debug_tuple("TurnoverIfPossessionLost")
                .field(arg0)
                .finish(),
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
            AnyProc::Armor(arg) => arg.step(game_state, input),
            AnyProc::Block(arg) => arg.step(game_state, input),
            AnyProc::BlockAction(arg) => arg.step(game_state, input),
            AnyProc::Bounce(arg) => arg.step(game_state, input),
            AnyProc::Catch(arg) => arg.step(game_state, input),
            AnyProc::ChangingWeather(arg) => arg.step(game_state, input),
            AnyProc::ChooseKickReceive(arg) => arg.step(game_state, input),
            AnyProc::CoinToss(arg) => arg.step(game_state, input),
            AnyProc::Deflect(arg) => arg.step(game_state, input),
            AnyProc::DeflectOrResolve(arg) => arg.step(game_state, input),
            AnyProc::DodgeProc(arg) => arg.step(game_state, input),
            AnyProc::Ejection(arg) => arg.step(game_state, input),
            AnyProc::FollowUp(arg) => arg.step(game_state, input),
            AnyProc::GameOver(arg) => arg.step(game_state, input),
            AnyProc::GfiProc(arg) => arg.step(game_state, input),
            AnyProc::Half(arg) => arg.step(game_state, input),
            AnyProc::Injury(arg) => arg.step(game_state, input),
            AnyProc::KOWakeUp(arg) => arg.step(game_state, input),
            AnyProc::Kickoff(arg) => arg.step(game_state, input),
            AnyProc::KickoffTable(arg) => arg.step(game_state, input),
            AnyProc::KnockDown(arg) => arg.step(game_state, input),
            AnyProc::LandKickoff(arg) => arg.step(game_state, input),
            AnyProc::MoveAction(arg) => arg.step(game_state, input),
            AnyProc::Pass(arg) => arg.step(game_state, input),
            AnyProc::PickupProc(arg) => arg.step(game_state, input),
            AnyProc::Push(arg) => arg.step(game_state, input),
            AnyProc::Setup(arg) => arg.step(game_state, input),
            AnyProc::StandUp(arg) => arg.step(game_state, input),
            AnyProc::ThrowIn(arg) => arg.step(game_state, input),
            AnyProc::Touchback(arg) => arg.step(game_state, input),
            AnyProc::Touchdown(arg) => arg.step(game_state, input),
            AnyProc::Turn(arg) => arg.step(game_state, input),
            AnyProc::TurnStunned(arg) => arg.step(game_state, input),
            AnyProc::TurnoverIfPossessionLost(arg) => arg.step(game_state, input),
        }
    }
}
