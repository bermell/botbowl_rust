//! One websocket per worker: handshake, then translate frames to `Inner`
//! calls. The first frame must be `Hello` within a few seconds; the reply
//! is `Welcome` or `Reject` and, on reject, the socket closes.

use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;

use botbowl_hub_proto::{decode, encode, BuildInfo, Capacity, RejectReason, ToHub, ToWorker, PROTOCOL_VERSION};

use crate::allowlist::Allowlist;
use crate::state::WorkerConn;
use crate::Hub;

/// Streams for a worker that did not say: one per core, one per GiB.
fn size_parallel(cores: u16, ram_mb: u32) -> u16 {
    let by_ram = if ram_mb == 0 {
        u16::MAX
    } else {
        (ram_mb / 1024).clamp(1, u16::MAX as u32) as u16
    };
    cores.max(1).min(by_ram)
}

/// How a worker's commit was cleared, for the log line. An admission through the allowlist is
/// an assertion by the operator that wants to be visible after the fact, not a silent success.
pub(crate) enum CommitVerdict {
    Same,
    AnyAllowed,
    Allowlisted,
}

fn check(hub: &Hub, protocol: u32, token: &str, build: &BuildInfo) -> Result<CommitVerdict, RejectReason> {
    if protocol != PROTOCOL_VERSION {
        return Err(RejectReason::Protocol { hub: PROTOCOL_VERSION });
    }
    if token != hub.cfg.token {
        return Err(RejectReason::BadToken);
    }
    let mine = BuildInfo::current();
    // Capacity first: a board the worker cannot hold is a hard fact about its binary, and no
    // allowlist entry should ever be read as excusing it.
    if build.capacity != mine.capacity {
        return Err(RejectReason::Capacity {
            hub: Capacity::compiled(),
        });
    }
    // Plan 042 left this open: `capacity` is the *compile-time* ceiling, but a task that names no
    // board plays `BoardDims::from_env()` on whichever worker gets it. Two boxes built from the
    // same commit with different `BOARD_SIZE_*` would then quietly contribute different games to
    // one corpus or one rung. The active board is part of the handshake for the same reason the
    // commit is.
    if build.env_board != mine.env_board {
        return Err(RejectReason::Board { hub: mine.env_board });
    }
    // A dirty worker is refused whatever the commit says, unless the hub is dirty too — a dirty
    // tree has no name, so neither an allowlist nor `--allow-commit-mismatch` can assert anything
    // about it. (Developing the worker itself means a dirty hub, which does relax this.)
    if build.dirty && !mine.dirty {
        return Err(RejectReason::Dirty);
    }
    if hub.cfg.allow_commit_mismatch {
        return Ok(CommitVerdict::AnyAllowed);
    }
    if build.commit == mine.commit {
        return Ok(CommitVerdict::Same);
    }
    // Re-read per handshake: the operator updates the file *after* committing, and a worker
    // retrying every 30 s then rejoins on its own (see `botbowl-worker`'s reconnect loop).
    match Allowlist::load(&hub.cfg.allowed_commits, &mine.commit) {
        Some(list) if list.admits(&build.commit) => Ok(CommitVerdict::Allowlisted),
        Some(list) => {
            eprintln!("[hub] {}", list.describe());
            Err(RejectReason::Commit { hub: mine.commit })
        }
        None => Err(RejectReason::Commit { hub: mine.commit }),
    }
}

pub async fn handle(mut socket: WebSocket, hub: Hub) {
    // -- handshake ---------------------------------------------------------
    let first = tokio::time::timeout(Duration::from_secs(10), socket.recv()).await;
    let hello = match first {
        Ok(Some(Ok(Message::Binary(b)))) => decode::<ToHub>(&b).ok(),
        _ => None,
    };
    let Some(ToHub::Hello {
        protocol,
        token,
        build,
        triple,
        name,
        cores,
        ram_mb,
        cached_models,
        parallel_games,
    }) = hello
    else {
        let _ = socket.close().await;
        return;
    };
    match check(&hub, protocol, &token, &build) {
        Ok(CommitVerdict::Same) => {}
        Ok(CommitVerdict::AnyAllowed) if build.commit != BuildInfo::current().commit => {
            eprintln!(
                "[hub] {name:?} ({triple}) is on commit {} vs the hub's {} -- admitted by --allow-commit-mismatch",
                short(&build.commit),
                short(&BuildInfo::current().commit),
            );
        }
        Ok(CommitVerdict::AnyAllowed) => {}
        Ok(CommitVerdict::Allowlisted) => {
            // Loud on purpose: these games are written into the corpus under the *hub's*
            // commit, so this line is the only record of which binary actually played them.
            eprintln!(
                "[hub] {name:?} ({triple}) is on commit {} vs the hub's {} -- admitted by {}",
                short(&build.commit),
                short(&BuildInfo::current().commit),
                hub.cfg.allowed_commits.display(),
            );
        }
        Err(reason) => {
            eprintln!("[hub] rejected {name:?} ({triple}): {reason}");
            let _ = socket
                .send(Message::Binary(encode(&ToWorker::Reject { reason }).into()))
                .await;
            let _ = socket.close().await;
            return;
        }
    }
    let parallel = parallel_games.unwrap_or_else(|| size_parallel(cores, ram_mb)).max(1);
    if socket
        .send(Message::Binary(
            encode(&ToWorker::Welcome {
                parallel_games: parallel,
            })
            .into(),
        ))
        .await
        .is_err()
    {
        return;
    }

    // -- register ----------------------------------------------------------
    let (tx, mut rx) = mpsc::unbounded_channel::<ToWorker>();
    let wid = hub.inner.lock().unwrap().add_worker(WorkerConn {
        name: name.clone(),
        build,
        triple,
        cores,
        ram_mb,
        parallel_games: parallel,
        tx,
        known_models: cached_models.into_iter().collect(),
        tasks: Default::default(),
        games_done: 0,
        last_seen: Instant::now(),
    });
    hub.changed.notify_waiters();

    let (mut sink, mut stream) = socket.split();
    let writer = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sink.send(Message::Binary(encode(&msg).into())).await.is_err() {
                break;
            }
        }
        let _ = sink.close().await;
    });

    // -- serve -------------------------------------------------------------
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(Message::Binary(b)) => {
                // Every frame is proof of life, not just the heartbeat: a
                // worker streaming results is plainly alive even if its
                // heartbeat task were wedged.
                hub.inner.lock().unwrap().seen(wid);
                match decode::<ToHub>(&b) {
                    Ok(ToHub::EvalGameDone { task, line }) => {
                        hub.inner.lock().unwrap().eval_game_done(wid, task, line);
                        hub.changed.notify_waiters();
                    }
                    Ok(ToHub::TrajectoryDone {
                        task,
                        game,
                        samples,
                        zstd_json,
                    }) => {
                        hub.inner
                            .lock()
                            .unwrap()
                            .trajectory_done(wid, task, game, samples, zstd_json);
                        hub.changed.notify_waiters();
                    }
                    Ok(ToHub::TaskFailed { task, error }) => {
                        hub.inner.lock().unwrap().task_failed(wid, task, error);
                        hub.changed.notify_waiters();
                    }
                    Ok(ToHub::Heartbeat { .. }) => {}
                    Ok(ToHub::Hello { .. }) => {}
                    Err(e) => {
                        eprintln!("[hub] worker {wid} sent an undecodable frame: {e}");
                        break;
                    }
                }
            }
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(_) => {}
        }
    }

    hub.inner.lock().unwrap().remove_worker(wid);
    hub.changed.notify_waiters();
    writer.abort();
}

fn short(commit: &str) -> String {
    commit.chars().take(12).collect()
}
