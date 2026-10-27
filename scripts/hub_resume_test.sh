#!/usr/bin/env bash
# Plan 069 item 5: a generate job survives its hub dying, end to end, with the real binaries.
#
#   scripts/hub_resume_test.sh [OUT_DIR]      # BIN=dir with botbowl-hub + botbowl-worker (release)
#
# One real hub and one real worker (heuristic search, small boards, so no GPU and no net), on
# PORT (default 13411; never the loop's or the web hub's port). First an uninterrupted reference
# run, then the same job through five interruptions:
#   kill9    the hub is killed -9 mid-phase, a torn line and a lost follow-on record are added to
#            the shards (what a kill mid-write leaves), the hub restarts; the waiting client
#            notices the new hub and resubmits with --resume on its own;
#   term     the hub gets SIGTERM mid-phase (as Slurm and train_loop.sh send it): the shards must
#            end on whole lines at once; then restart and the client resumes;
#   relaunch hub and client both killed -9 (a Slurm job cut); a fresh hub and a fresh
#            `job generate --resume` finish it (what the next job's loop does);
#   deadline the client gets --deadline: it drains the job and exits 75; resubmitted, it finishes;
#   worker   the worker keeps running across a hub restart, so it sends results for the dead
#            hub's tasks to the new one, which must drop them, never write them.
# Every run must end with each shard holding exactly its seeds, each game's records once (drive 1,
# then its drive 2 if any), no torn line — and the same records per seed as the reference
# (heuristic search on one thread is a pure function of the seed).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$REPO/target/release}"
PORT="${PORT:-13411}"
OUT="${1:-$(mktemp -d /tmp/hub_resume.XXXXXX)}"
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
HUB="$BIN/botbowl-hub"; WORKER="$BIN/botbowl-worker"
for b in "$HUB" "$WORKER"; do [ -x "$b" ] || { echo "missing $b (set BIN)"; exit 2; }; done
TOKEN="$OUT/hub.token"   # this test's own token: the machine's token file is never touched
URL="http://127.0.0.1:$PORT"
GAMES="${GAMES:-60}"     # per shard
SHARDS="0 1 2 3"
unset BLOOD_MCTS_BUDGET BLOOD_NN_SERVER

HUB_PID=""; WORKER_PID=""
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/test.log"; }
hub_up() {
    "$HUB" serve --bind "127.0.0.1:$PORT" --token-file "$TOKEN" --no-play --model-index-dir "$OUT/none" \
        >> "$OUT/hub.log" 2>&1 &
    HUB_PID=$!
    local i=0
    until "$HUB" status --hub "$URL" --token-file "$TOKEN" > /dev/null 2>&1; do
        i=$((i + 1)); [ "$i" -gt 60 ] && { log "FAIL: hub did not come up"; exit 1; }; sleep 0.5
    done
}
worker_up() {
    "$WORKER" --hub "ws://127.0.0.1:$PORT/ws" --token-file "$TOKEN" --name w --parallel-games 4 \
        --cache-dir "$OUT/cache" --reconnect-max-secs 2 >> "$OUT/worker.log" 2>&1 &
    WORKER_PID=$!
}
cleanup() { kill "$HUB_PID" "$WORKER_PID" 2> /dev/null; wait 2> /dev/null; }
trap cleanup EXIT

# The job, as train_loop.sh submits it, small: heuristic, 8 descents, one search thread.
job() {   # job DIR MODE [extra args]
    local dir="$1" mode="$2"; shift 2
    "$HUB" job generate --hub "$URL" --token-file "$TOKEN" --mode random-start --games "$GAMES" \
        --seed-base 7000000 --shard-seed-stride 100000 --mcts-iters 8 --mcts-workers 1 \
        --evaluator heuristic --board-sizes 12x5,14x7 --next-drive \
        --shards "$SHARDS" --label "resume test" "$mode" --out-dir "$dir" --wait "$@"
}
lines() { cat "$1"/shard*.jsonl 2> /dev/null | wc -l; }
wait_lines() {   # wait_lines DIR N: until N records are on disk
    local i=0
    until [ "$(lines "$1")" -ge "$2" ]; do
        i=$((i + 1)); [ "$i" -gt 1200 ] && { log "FAIL: no progress in $1"; exit 1; }; sleep 0.25
    done
}
check() {   # check DIR: python verdict against the reference
    "${PY:-python3}" - "$1" "$OUT/ref" "$GAMES" $SHARDS << 'EOF'
import json, sys, os
d, ref, games, shards = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
def read(path):
    raw = open(path, 'rb').read()
    assert raw.endswith(b'\n'), f'{path}: ends mid-line'
    out = {}
    for n, line in enumerate(raw.split(b'\n')[:-1]):
        t = json.loads(line)  # every line whole
        m = t['meta']
        out.setdefault(m['seed'], []).append((m['extra'].get('drive'), len(t['samples']), t['outcome']['home_score'], t['outcome']['away_score']))
    return out
bad, same, diff, follow, unchecked = [], 0, 0, 0, 0
for k in shards:
    got, want = read(f'{d}/shard{k}.jsonl'), read(f'{ref}/shard{k}.jsonl')
    base = 7000000 + int(k) * 100000
    if set(got) != set(range(base, base + games)):
        bad.append(f'shard{k}: seeds {sorted(set(range(base, base + games)) ^ set(got))} missing or foreign')
    for s, recs in got.items():
        drives = [r[0] for r in recs]
        if drives not in (['1'], ['1', '2']):
            bad.append(f'shard{k} seed {s}: records {drives}')
        follow += len(recs) == 2
        if s not in want: unchecked += 1
        elif recs == want[s]: same += 1
        else: diff += 1
print(f'{sum(1 for _ in shards)} shards, {same} games identical to the reference, {diff} different'
      + (f', {unchecked} beyond it' if unchecked else '') + f', {follow} with a follow-on drive')
if bad:
    print('FAIL: ' + '; '.join(bad[:10])); sys.exit(1)
if diff:
    print('FAIL: games differ from the reference'); sys.exit(1)
EOF
}
verdict() {   # verdict NAME DIR
    if out=$(check "$2"); then log "PASS $1: $out"; else log "FAIL $1: $out"; FAILED=1; fi
}
FAILED=0
: > "$OUT/test.log"
log "out $OUT, bin $BIN, port $PORT, $GAMES games x shards $SHARDS"

# -- reference ---------------------------------------------------------------
hub_up; worker_up
T0=$(date +%s)
job "$OUT/ref" --truncate > "$OUT/ref.client.log" 2>&1 || { log "FAIL: reference run"; exit 1; }
TOTAL=$(lines "$OUT/ref")
log "reference: $TOTAL records in $(( $(date +%s) - T0 )) s"
verdict reference "$OUT/ref"

# -- kill -9 mid-phase, torn tail, client resubmits ---------------------------
job "$OUT/kill9" --resume > "$OUT/kill9.client.log" 2>&1 &
CLIENT=$!
wait_lines "$OUT/kill9" $((TOTAL / 3))
kill -9 "$HUB_PID"; wait "$HUB_PID" 2> /dev/null
# What a kill mid-write leaves: half a line on one shard; on another, a game whose follow-on
# record is missing (its scoring first drive is the last line).
head -c 700 "$OUT/ref/shard1.jsonl" >> "$OUT/kill9/shard0.jsonl"
"${PY:-python3}" - "$OUT/kill9" << 'EOF' | tee -a "$OUT/test.log"
import json, sys
# On shard 1, 2 or 3: cut after the last first-drive record that is followed by its drive 2.
for k in (1, 2, 3):
    p = f'{sys.argv[1]}/shard{k}.jsonl'
    lines = open(p, 'rb').read().split(b'\n')[:-1]
    for i in range(len(lines) - 2, -1, -1):
        a, b = json.loads(lines[i]), json.loads(lines[i + 1])
        if b['meta']['extra'].get('drive') == '2' and a['meta']['seed'] == b['meta']['seed']:
            open(p, 'wb').write(b'\n'.join(lines[:i + 1]) + b'\n')
            print(f'kill9: shard{k} cut between seed {a["meta"]["seed"]} and its follow-on record')
            sys.exit(0)
print('kill9: no follow-on record on disk yet to cut')
EOF
log "kill9: hub killed at $(lines "$OUT/kill9") records; restarting"
hub_up
wait "$CLIENT"; RC=$?
grep -q "submitted again" "$OUT/kill9.client.log" || { log "FAIL kill9: the client did not resubmit"; FAILED=1; }
[ "$RC" -eq 0 ] || { log "FAIL kill9: client exit $RC"; FAILED=1; }
verdict kill9 "$OUT/kill9"

# -- SIGTERM: whole lines at once ----------------------------------------------
job "$OUT/term" --resume > "$OUT/term.client.log" 2>&1 &
CLIENT=$!
wait_lines "$OUT/term" $((TOTAL / 2))
kill -TERM "$HUB_PID"; wait "$HUB_PID" 2> /dev/null
TORN=0
for f in "$OUT"/term/shard*.jsonl; do [ -s "$f" ] && [ "$(tail -c 1 "$f" | od -An -c | tr -d ' ')" != '\n' ] && TORN=1; done
[ "$TORN" -eq 0 ] && log "term: every shard ends on a whole line after SIGTERM" || { log "FAIL term: a torn line after SIGTERM"; FAILED=1; }
grep -q "SIGTERM: exiting" "$OUT/hub.log" || { log "FAIL term: no clean-exit line in hub.log"; FAILED=1; }
hub_up
wait "$CLIENT" || { log "FAIL term: client exit $?"; FAILED=1; }
verdict term "$OUT/term"

# -- relaunch: hub and client both gone, a fresh client resumes ------------------
job "$OUT/relaunch" --resume > "$OUT/relaunch.client.log" 2>&1 &
CLIENT=$!
wait_lines "$OUT/relaunch" $((TOTAL / 2))
kill -9 "$CLIENT" "$HUB_PID"; wait "$CLIENT" "$HUB_PID" 2> /dev/null
log "relaunch: hub and client killed at $(lines "$OUT/relaunch") records"
hub_up
job "$OUT/relaunch" --resume >> "$OUT/relaunch.client.log" 2>&1 || { log "FAIL relaunch: exit $?"; FAILED=1; }
grep "resuming" "$OUT/hub.log" | tail -4 | sed 's/^/    /' | tee -a "$OUT/test.log"
verdict relaunch "$OUT/relaunch"

# -- deadline: drained, exit 75, resubmitted ----------------------------------------
# Ten times the games, so the job outlives the client's first poll (5 s in), where it is drained;
# the client exits 75 once the games then in flight are in. Seeds past the reference's are checked
# for shape only.
DL=$(( $(date +%s) + 25 ))
GAMES=$((GAMES * 10)) job "$OUT/deadline" --resume --deadline "$DL" --drain-secs 23 > "$OUT/deadline.client.log" 2>&1
RC=$?
AT=$(lines "$OUT/deadline")
[ "$RC" -eq 75 ] && log "deadline: client exit 75 at $AT records" || { log "FAIL deadline: exit $RC (want 75)"; FAILED=1; }
grep -q "draining" "$OUT/hub.log" || { log "FAIL deadline: the job was never drained"; FAILED=1; }
GAMES=$((GAMES * 10)) job "$OUT/deadline" --resume > "$OUT/deadline2.client.log" 2>&1 || { log "FAIL deadline: resubmit exit $?"; FAILED=1; }
grep -q "no longer draining" "$OUT/hub.log" && log "deadline: resubmission attached to the drained job" || { log "FAIL deadline: resubmission made a new job"; FAILED=1; }
GAMES=$((GAMES * 10)) verdict deadline "$OUT/deadline"

# -- the worker's stale results ---------------------------------------------------
STALE=$(grep -c "holds seed" "$OUT/hub.log")
JOINS=$(grep -c '"w" joined' "$OUT/hub.log")
log "worker: one process, $JOINS connections over the hub restarts; $STALE result(s) of a dead hub's task reached a live task (must be 0: task ids are random per hub)"
[ "$STALE" -eq 0 ] || FAILED=1

[ "$FAILED" -eq 0 ] && log "ALL PASS" || log "SOME FAILED (see $OUT)"
exit "$FAILED"
