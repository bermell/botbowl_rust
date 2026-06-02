//! First-iteration scripted bot. Translated from the Python reference at
//! `botbowl/examples/scripted_bot_example.py` but limited to a small heuristic ladder
//! that needs no engine changes. The ladder is the natural extension point — add
//! a step before "End turn" to grow capability.

use std::collections::VecDeque;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::bots::Bot;
use crate::core::gamestate::GameState;
use crate::core::model::{
    other_team, Action, AvailableActions, BallState, FieldedPlayer, PlayerID, PlayerStatus, Position, TeamType,
};
use crate::core::pathing::{Node, PathFinder};
use crate::core::table::{NumBlockDices, PosAT, SimpleAT, Skill};

/// Minimum success probability for the bot to attempt a path that ends in a
/// touchdown. Set low enough that we still take a 2-GFI run from MA-7 away
/// when there's no safer alternative — drifting closer is worthless if the
/// carrier won't get another turn (curriculum lectures, end of half, etc.).
const TD_ATTEMPT_PROB_THRESHOLD: f32 = 0.5;

pub struct ScriptedBot {
    actions: VecDeque<Action>,
    rng: ChaCha8Rng,
}

impl ScriptedBot {
    pub fn new() -> Self {
        Self {
            actions: VecDeque::new(),
            rng: ChaCha8Rng::from_entropy(),
        }
    }
}

impl Default for ScriptedBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for ScriptedBot {
    fn get_action(&mut self, state: &GameState) -> Action {
        // Drain any queued actions that are still legal in the current state.
        while let Some(action) = self.actions.pop_front() {
            if state.is_legal_action(&action) {
                return action;
            }
        }

        let (action, queue) = decide(state);
        for follow_up in queue {
            self.actions.push_back(follow_up);
        }
        if !state.is_legal_action(&action) {
            // Heuristic produced something the engine didn't expect — fall back so the
            // game keeps moving instead of panicking.
            if let Some(fallback) = first_legal_simple_or_any(state) {
                return fallback;
            }
        }
        action
    }

    fn set_seed(&mut self, rng: ChaCha8Rng) {
        self.rng = rng;
    }
}

fn first_legal_simple_or_any(state: &GameState) -> Option<Action> {
    let all = state.get_all_actions();
    // Prefer the most "neutral" simple action.
    for preferred in [
        SimpleAT::EndTurn,
        SimpleAT::EndPlayerTurn,
        SimpleAT::EndSetup,
        SimpleAT::DontUseReroll,
    ] {
        let candidate = Action::Simple(preferred);
        if all.contains(&candidate) {
            return Some(candidate);
        }
    }
    all.into_iter().next()
}

/// Returns the first action to play now and an ordered queue of follow-up actions to enqueue.
fn decide(state: &GameState) -> (Action, Vec<Action>) {
    let aa = state.available_actions.as_ref();
    let simple = aa.get_simple();

    // Coin toss
    if simple.contains(&SimpleAT::Tails) || simple.contains(&SimpleAT::Heads) {
        return (Action::Simple(SimpleAT::Tails), vec![]);
    }

    // Kick / receive choice
    if simple.contains(&SimpleAT::Receive) || simple.contains(&SimpleAT::Kick) {
        return (Action::Simple(SimpleAT::Receive), vec![]);
    }

    // Setup — let the engine pick a default formation, then end setup on the next call.
    if simple.contains(&SimpleAT::SetupLine) {
        return (
            Action::Simple(SimpleAT::SetupLine),
            vec![Action::Simple(SimpleAT::EndSetup)],
        );
    }
    if simple.contains(&SimpleAT::EndSetup) {
        return (Action::Simple(SimpleAT::EndSetup), vec![]);
    }

    // Kickoff aim
    if simple.contains(&SimpleAT::KickoffAimMiddle) {
        return (Action::Simple(SimpleAT::KickoffAimMiddle), vec![]);
    }

    // Block die selection
    if let Some(action) = pick_block_die(state, simple) {
        return (action, vec![]);
    }

    // Reroll decisions. The procedure on top of the stack tells us context.
    if simple.contains(&SimpleAT::UseReroll) {
        return (reroll_decision(state), vec![]);
    }

    // Push / follow-up — single positional choices.
    if let Some(action) = first_positional_for(state, PosAT::Push) {
        return (action, vec![]);
    }
    if let Some(action) = first_positional_for(state, PosAT::FollowUp) {
        return (action, vec![]);
    }

    // Mid-action: an active player exists and the engine has populated paths.
    // Choose a destination from the precomputed paths if one is useful; otherwise
    // EndPlayerTurn.
    if state.info.active_player.is_some() {
        if let Some((action, follow_up)) = pick_destination(state) {
            let mut tail = vec![];
            if let Some(f) = follow_up {
                tail.push(f);
            }
            return (action, tail);
        }
        if simple.contains(&SimpleAT::EndPlayerTurn) {
            return (Action::Simple(SimpleAT::EndPlayerTurn), vec![]);
        }
    }

    // Top of turn: produce a plan.
    if let Some(planned) = make_plan(state) {
        return planned;
    }

    if simple.contains(&SimpleAT::EndTurn) {
        return (Action::Simple(SimpleAT::EndTurn), vec![]);
    }
    if simple.contains(&SimpleAT::EndPlayerTurn) {
        return (Action::Simple(SimpleAT::EndPlayerTurn), vec![]);
    }

    // Should never happen if every NeedAction branch is covered above.
    (
        first_legal_simple_or_any(state).expect("no legal action available"),
        vec![],
    )
}

/// During a MoveAction, the engine offers precomputed paths in `available_actions.paths`.
/// Choose the best destination for the active player. Returns the action to play now
/// plus an optional follow-up (e.g. on standup, a subsequent EndPlayerTurn).
fn pick_destination(state: &GameState) -> Option<(Action, Option<Action>)> {
    let active = state.get_active_player()?;
    let team = active.stats.team;
    let endzone_x = state.get_endzone_x(team);
    let is_carrier = matches!(state.ball, BallState::Carried(id) if id == active.id);

    let paths = state.get_paths()?;

    // 1) If the carrier can score from here at p >= TD_ATTEMPT_PROB_THRESHOLD, do it.
    if is_carrier {
        let mut best_td: Option<std::sync::Arc<Node>> = None;
        for (pos, node_opt) in paths.iter_position() {
            if pos.x != endzone_x {
                continue;
            }
            let Some(node) = node_opt else { continue };
            if node.prob < TD_ATTEMPT_PROB_THRESHOLD {
                continue;
            }
            match &best_td {
                Some(b) if b.prob >= node.prob => (),
                _ => best_td = Some(node.clone()),
            }
        }
        if let Some(node) = best_td {
            return Some((Action::Positional(node.get_action_type(), node.position), None));
        }
    }

    // 2) Drift towards the endzone on a guaranteed (prob == 1) path if we're carrying.
    if is_carrier {
        let current_distance = (endzone_x - active.position.x).abs();
        let mut best: Option<(i8, std::sync::Arc<Node>)> = None;
        for (pos, node_opt) in paths.iter_position() {
            let Some(node) = node_opt else { continue };
            if (node.prob - 1.0).abs() > 1e-6 {
                continue;
            }
            let dist = (endzone_x - pos.x).abs();
            if dist >= current_distance {
                continue;
            }
            match &best {
                Some((best_d, _)) if *best_d <= dist => (),
                _ => best = Some((dist, node.clone())),
            }
        }
        if let Some((_, node)) = best {
            return Some((Action::Positional(node.get_action_type(), node.position), None));
        }
    }

    // 3) Non-carrier: if we were started to do a pickup, move to the ball.
    if !is_carrier {
        if let BallState::OnGround(ball_pos) = state.ball {
            if let Some(Some(node)) =
                paths
                    .iter_position()
                    .find_map(|(pos, node)| if pos == ball_pos { Some(node.clone()) } else { None })
            {
                if node.prob >= 0.33 {
                    return Some((Action::Positional(node.get_action_type(), node.position), None));
                }
            }
        }
    }

    // 4) Block action: pick the best adjacent up opponent.
    {
        let mut best: Option<std::sync::Arc<Node>> = None;
        for (_, node_opt) in paths.iter_position() {
            let Some(node) = node_opt else { continue };
            if node.get_action_type() != PosAT::Block {
                continue;
            }
            let dice = match node.get_block_dice() {
                Some(d) => d,
                None => continue,
            };
            if !matches!(dice, NumBlockDices::Two | NumBlockDices::Three) {
                continue;
            }
            match &best {
                Some(b) => {
                    let b_dice = b.get_block_dice().unwrap();
                    if dice > b_dice {
                        best = Some(node.clone());
                    }
                }
                None => best = Some(node.clone()),
            }
        }
        if let Some(node) = best {
            return Some((Action::Positional(node.get_action_type(), node.position), None));
        }
    }

    None
}

fn first_positional_for(state: &GameState, at: PosAT) -> Option<Action> {
    for action in state.get_all_actions() {
        if let Action::Positional(action_at, _) = action {
            if action_at == at {
                return Some(action);
            }
        }
    }
    None
}

fn pick_block_die(state: &GameState, simple: &std::collections::HashSet<SimpleAT>) -> Option<Action> {
    let is_block_choice = simple.contains(&SimpleAT::SelectPow)
        || simple.contains(&SimpleAT::SelectPowPush)
        || simple.contains(&SimpleAT::SelectPush)
        || simple.contains(&SimpleAT::SelectBothDown)
        || simple.contains(&SimpleAT::SelectSkull);
    if !is_block_choice {
        return None;
    }

    // Defender down outcomes are best.
    if simple.contains(&SimpleAT::SelectPow) {
        return Some(Action::Simple(SimpleAT::SelectPow));
    }
    // Pow-push (defender stumbles / falls): take it unless defender has Dodge and we lack Tackle.
    if simple.contains(&SimpleAT::SelectPowPush) {
        let (attacker, defender) = active_block_attacker_defender(state);
        let defender_dodges = defender.map(|d| d.has_skill(Skill::Dodge)).unwrap_or(false);
        let attacker_has_tackle = attacker
            .map(|a| a.has_skill(Skill::Block /* TODO: Tackle when added */))
            .unwrap_or(false);
        let _ = attacker_has_tackle;
        // Tackle isn't in the Skill enum yet, so the choice collapses to "always take it";
        // when defender has Dodge they'd just dodge it out, but it's still better than
        // a plain push or both-down.
        if !defender_dodges {
            return Some(Action::Simple(SimpleAT::SelectPowPush));
        }
        // Defender has dodge — fall through, push is safer.
    }

    if simple.contains(&SimpleAT::SelectBothDown) {
        // Take both-down if we have Block and they don't.
        let (attacker, defender) = active_block_attacker_defender(state);
        let attacker_has_block = attacker.map(|a| a.has_skill(Skill::Block)).unwrap_or(false);
        let defender_has_block = defender.map(|d| d.has_skill(Skill::Block)).unwrap_or(false);
        if attacker_has_block && !defender_has_block {
            return Some(Action::Simple(SimpleAT::SelectBothDown));
        }
    }

    if simple.contains(&SimpleAT::SelectPush) {
        return Some(Action::Simple(SimpleAT::SelectPush));
    }
    if simple.contains(&SimpleAT::SelectPowPush) {
        return Some(Action::Simple(SimpleAT::SelectPowPush));
    }
    if simple.contains(&SimpleAT::SelectBothDown) {
        return Some(Action::Simple(SimpleAT::SelectBothDown));
    }
    if simple.contains(&SimpleAT::SelectSkull) {
        return Some(Action::Simple(SimpleAT::SelectSkull));
    }
    None
}

fn active_block_attacker_defender(state: &GameState) -> (Option<&FieldedPlayer>, Option<&FieldedPlayer>) {
    // Best-effort: attacker is the active player; defender is the adjacent up opponent.
    let attacker = state.get_active_player();
    let defender = attacker.and_then(|att| {
        state
            .get_adj_players(att.position)
            .find(|p| p.stats.team != att.stats.team && p.status == PlayerStatus::Up)
    });
    (attacker, defender)
}

fn reroll_decision(state: &GameState) -> Action {
    let proc_name = state.proc_stack_top();
    match proc_name {
        // Always reroll for these — failure is a turnover or a wasted player action.
        Some("DodgeProc") | Some("PickupProc") | Some("GfiProc") | Some("Catch") | Some("Pass") => {
            Action::Simple(SimpleAT::UseReroll)
        }
        // Block: reroll if the best available outcome is bad for us.
        Some("Block") => {
            // Conservatively skip the reroll — v1 leaves a richer policy for later.
            Action::Simple(SimpleAT::DontUseReroll)
        }
        _ => Action::Simple(SimpleAT::DontUseReroll),
    }
}

fn my_team(state: &GameState) -> Option<TeamType> {
    state.available_actions.get_team()
}

/// Top-of-turn planner. Returns (action_now, follow_up_queue). The action_now is a
/// START_xxx that activates a specific player; `pick_destination` (called on subsequent
/// get_action invocations once the engine pushes MoveAction) will choose the destination.
fn make_plan(state: &GameState) -> Option<(Action, Vec<Action>)> {
    let team = my_team(state)?;

    // Step 1: stand up downed (not stunned) marked own players.
    if let Some(player) = state
        .get_players_on_pitch_in_team(team)
        .filter(|p| !p.used)
        .filter(|p| p.status == PlayerStatus::Down)
        .find(|p| state.get_tz_on(p.id) > 0)
    {
        if state.is_legal_action(&Action::Positional(PosAT::StartMove, player.position)) {
            return Some((Action::Positional(PosAT::StartMove, player.position), vec![]));
        }
    }

    // Step 2: score a touchdown with our ball carrier, or drift toward the endzone.
    if let Some(carrier) = ball_carrier(state) {
        if carrier.stats.team == team && !carrier.used {
            let can_score = PathFinder::safest_path_to_endzone(state, carrier.id)
                .ok()
                .flatten()
                .map(|p| p.prob >= TD_ATTEMPT_PROB_THRESHOLD)
                .unwrap_or(false);
            if can_score && state.is_legal_action(&Action::Positional(PosAT::StartMove, carrier.position)) {
                return Some((Action::Positional(PosAT::StartMove, carrier.position), vec![]));
            }

            if state.get_tz_on(carrier.id) == 0 {
                if let Ok(paths) = PathFinder::player_paths(state, carrier.id) {
                    let endzone_x = state.get_endzone_x(team);
                    let current_distance = (endzone_x - carrier.position.x).abs();
                    let any_safe_progress = paths.iter_position().any(|(pos, node_opt)| {
                        node_opt.as_ref().map_or(false, |n| {
                            (n.prob - 1.0).abs() < 1e-6 && (endzone_x - pos.x).abs() < current_distance
                        })
                    });
                    if any_safe_progress
                        && state.is_legal_action(&Action::Positional(PosAT::StartMove, carrier.position))
                    {
                        return Some((Action::Positional(PosAT::StartMove, carrier.position), vec![]));
                    }
                }
            }
        }
    }

    // Step 3: safe blocks (2DB+).
    if let Some((attacker_pos, _)) = best_safe_block(state, team) {
        if state.is_legal_action(&Action::Positional(PosAT::StartBlock, attacker_pos)) {
            return Some((Action::Positional(PosAT::StartBlock, attacker_pos), vec![]));
        }
    }

    // Step 3.5: opponent holds the ball. If a 2-dice blitz onto the
    // carrier is achievable by moving a still-fresh player adjacent to
    // them (using existing adjacent friendlies as assists), start it.
    // The follow-up moves and the block itself are handled by the
    // in-blitz branch in `pick_destination`.
    if let BallState::Carried(carrier_id) = state.ball {
        if let Ok(carrier) = state.get_player(carrier_id) {
            if carrier.stats.team != team {
                if let Some(blitzer_pos) = plan_blitz_against_carrier(state, team, carrier_id) {
                    if state.is_legal_action(&Action::Positional(PosAT::StartBlitz, blitzer_pos)) {
                        return Some((Action::Positional(PosAT::StartBlitz, blitzer_pos), vec![]));
                    }
                }
            }
        }
    }

    // Step 4: pickup the ball if it's on the ground.
    if matches!(state.ball, BallState::OnGround(_)) {
        if let Some(ball_pos) = state.get_ball_position() {
            let mut best: Option<(f32, Position)> = None;
            for player in state
                .get_players_on_pitch_in_team(team)
                .filter(|p| !p.used && p.status == PlayerStatus::Up)
            {
                let distance = player.position.distance_to(&ball_pos);
                if distance as u8 > player.stats.ma + 2 {
                    continue;
                }
                if let Ok(Some(path)) = PathFinder::safest_path_to(state, player.id, ball_pos) {
                    let prob = path.prob;
                    if prob < 0.33 {
                        continue;
                    }
                    match best {
                        Some((best_p, _)) if best_p >= prob => (),
                        _ => best = Some((prob, player.position)),
                    }
                }
            }
            if let Some((_, start)) = best {
                if state.is_legal_action(&Action::Positional(PosAT::StartMove, start)) {
                    return Some((Action::Positional(PosAT::StartMove, start), vec![]));
                }
            }
        }
    }

    // Step 5: nothing useful to do — end the turn.
    if state.available_actions.get_simple().contains(&SimpleAT::EndTurn) {
        return Some((Action::Simple(SimpleAT::EndTurn), vec![]));
    }
    None
}

fn ball_carrier(state: &GameState) -> Option<&FieldedPlayer> {
    match state.ball {
        BallState::Carried(id) => state.get_player(id).ok(),
        _ => None,
    }
}

/// Find a home player whose blitz onto the enemy carrier would deliver
/// 2+ dice in our favour. Picks the player+square combination with the
/// best (dice_count, path_prob) tuple; ties broken by path probability.
///
/// Returns the player's *current* position so the caller can issue
/// `StartBlitz` on them.
fn plan_blitz_against_carrier(state: &GameState, team: TeamType, carrier_id: PlayerID) -> Option<Position> {
    let carrier_pos = state.get_player(carrier_id).ok()?.position;

    let mut best: Option<(NumBlockDices, f32, Position)> = None;

    for blitzer in state
        .get_players_on_pitch_in_team(team)
        .filter(|p| !p.used && p.status == PlayerStatus::Up)
    {
        let paths = match PathFinder::player_paths(state, blitzer.id) {
            Ok(p) => p,
            Err(_) => continue,
        };

        for (pos, node_opt) in paths.iter_position() {
            // We need a square adjacent to the carrier — that's where the
            // attacker must stand to deliver the block.
            if pos.distance_to(&carrier_pos) != 1 {
                continue;
            }
            // Don't try to "move" onto the carrier itself (occupied).
            if state.get_player_at(pos).is_some() && pos != blitzer.position {
                continue;
            }
            let Some(node) = node_opt else { continue };
            // Path probability includes any dodges/GFI needed to reach the
            // square. Keep the floor reasonably high so we don't trade a
            // safe assist for a coin-flip blitz.
            if node.prob < 0.66 {
                continue;
            }
            let dice = state.get_blockdices_from(blitzer.id, pos, carrier_id);
            if !matches!(dice, NumBlockDices::Two | NumBlockDices::Three) {
                continue;
            }
            let candidate = (dice, node.prob, blitzer.position);
            match &best {
                Some((best_dice, best_prob, _)) if (*best_dice, *best_prob) >= (candidate.0, candidate.1) => {}
                _ => best = Some(candidate),
            }
        }
    }

    best.map(|(_, _, pos)| pos)
}

fn best_safe_block(state: &GameState, team: TeamType) -> Option<(Position, Position)> {
    let opp = other_team(team);
    let mut best: Option<(NumBlockDices, Position, Position)> = None;
    for attacker in state
        .get_players_on_pitch_in_team(team)
        .filter(|p| !p.used && p.status == PlayerStatus::Up)
    {
        for defender in state
            .get_adj_players(attacker.position)
            .filter(|p| p.stats.team == opp && p.status == PlayerStatus::Up)
        {
            let dice = state.get_blockdices(attacker.id, defender.id);
            // Only take blocks with at least 2 dice in our favour.
            let safe = matches!(dice, NumBlockDices::Two | NumBlockDices::Three);
            if !safe {
                continue;
            }
            match &best {
                Some((best_dice, _, _)) if *best_dice >= dice => (),
                _ => best = Some((dice, attacker.position, defender.position)),
            }
        }
    }
    best.map(|(_, a, d)| (a, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::RandomBot;
    use crate::core::game_runner::BotGameRunnerBuilder;
    use crate::core::gamestate::{BuilderState, GameStateBuilder};

    /// Smoke test — the bot pair runs a game to completion without panicking.
    #[test]
    fn scripted_vs_random_runs_to_completion() {
        for seed in 0..3u64 {
            let mut runner = BotGameRunnerBuilder::new()
                .set_home_bot(Box::new(ScriptedBot::new()))
                .set_away_bot(Box::new(RandomBot::new()))
                .set_seed(seed)
                .build();
            runner.run();
        }
    }

    /// At a clean turn start with a 2DB matchup available, the ladder issues a
    /// START_BLOCK rather than ending the turn.
    #[test]
    fn ladder_starts_block_with_2db_advantage() {
        // Two home attackers (str 3 each) adjacent to one away defender (str 3) at (9,8).
        // The attacker at (10,8) has a teammate at (11,8) as an assist on the opposite
        // side of (9,8)? No — assist needs to be adjacent to the *defender*. Move the
        // second home player to (10,7) to be adjacent to defender (9,8) for the assist.
        let mut state = GameStateBuilder::new()
            .add_home_player(Position::new((10, 8)))
            .add_home_player(Position::new((10, 7)))
            .add_away_player(Position::new((9, 8)))
            .set_state(BuilderState::Turn { turn: 1 })
            .build();

        // Ensure it's Home's turn; if not, end Away's turn first.
        while state.available_actions.get_team() != Some(TeamType::Home) {
            state.step_simple(SimpleAT::EndTurn);
        }

        let (action, _) = decide(&state);
        assert!(
            state.is_legal_action(&action),
            "ladder produced an illegal action: {:?}",
            action
        );
        assert!(
            matches!(action, Action::Positional(PosAT::StartBlock, _)),
            "expected START_BLOCK, got {:?}",
            action
        );
    }

    /// When the only legal options are EndTurn/EndPlayerTurn, the bot ends the turn cleanly.
    #[test]
    fn coin_toss_returns_tails() {
        // CoinToss is the very first decision (Away picks heads/tails).
        let state = GameStateBuilder::new().set_state(BuilderState::CoinToss).build();
        let (action, _) = decide(&state);
        assert_eq!(action, Action::Simple(SimpleAT::Tails));
    }
}
