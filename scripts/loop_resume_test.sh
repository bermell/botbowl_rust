#!/usr/bin/env bash
# Plan 069 item 5: train_loop.sh's generate phase spans a deadline, in a scratch run dir.
#
#   scripts/loop_resume_test.sh [OUT_DIR]     # BIN=dir with botbowl-hub + botbowl-worker (release)
#
# Launch 1 runs the loop with LOOP_DEADLINE a minute away: the generate phase starts, the job
# client drains it PHASE_DRAIN_SECS before the deadline, the loop exits 75 with the games written
# so far on disk (whole lines). Launch 2, no deadline, resumes the phase (`genNN generate resumes`,
# the hub's `resuming` lines) and finishes it; a STOP file made while it runs ends the loop at the
# next boundary, before anything that needs a GPU, a net or the trainer. Each shard must then hold
# exactly its seeds, each game's records once.
#
# Heuristic games on 12x5/14x7 boards, a dummy champion (the heuristic evaluator never opens it),
# BOTBOWL_BIN_DIR with stand-ins for botbowl-ui/prepare (the generate phase does not run them), its
# own hub port and token. Never touches runs/, models/ or the machine's hub token.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')"
BIN="${BIN:-$REPO/target/release}"
S="${1:-$(mktemp -d /tmp/loop_resume.XXXXXX)}"
mkdir -p "$S" && S="$(cd "$S" && pwd)"
mkdir -p "$S/bin" "$S/models" "$S/run"
for b in botbowl-hub botbowl-worker; do
    [ -x "$BIN/$b" ] || { echo "missing $BIN/$b (set BIN)"; exit 2; }
    ln -sf "$BIN/$b" "$S/bin/$b"
done
for b in botbowl-ui prepare; do printf '#!/bin/sh\necho "%s: not part of this test" >&2; exit 1\n' "$b" > "$S/bin/$b"; chmod +x "$S/bin/$b"; done
echo "not a net" > "$S/models/champ.onnx"; echo "not a net" > "$S/models/anchor.onnx"
echo "$S/models/champ.onnx" > "$S/run/champion.txt"
touch "$S/run/.mirror.done"

export PY="${PY:-$MAIN/train/.venv/bin/python}"
export BOTBOWL_BIN_DIR="$S/bin" RUN_DIR_OVERRIDE="$S/run" MODEL_DIR="$S/models" TIER_OVERRIDE=resumetest
export INIT_CHAMPION="$S/models/champ.onnx" ANCHOR="$S/models/anchor.onnx"
export SIZE_MODE=list SIZE_LIST="12x5,14x7" EVALUATOR=heuristic NN_SERVER=off NN_BACKEND=sidecar
export MCTS_ITERS="${MCTS_ITERS:-100}" GAMES_PER_SHARD="${GAMES_PER_SHARD:-60}" GEN_PARALLEL_GAMES=4 NEXT_DRIVE=1
export MC_LABEL_PLAYOUTS=0 NET_CHECK=off MAX_GENS=1
export HUB_PORT="${PORT:-13412}" HUB_TOKEN_FILE="$S/hub.token" HUB_ALLOW_FROM=""
export GEN_RESUME=on PHASE_DRAIN_SECS=10 GEN_MIN_SLICE_SECS=5 STOP_MARGIN_SECS=0
unset BLOOD_MCTS_BUDGET BLOOD_NN_SERVER LOOP_DEADLINE
GEN="$S/run/gen01"
FAILED=0
log() { echo "[$(date +%T)] $*" | tee -a "$S/test.log"; }
lines() { cat "$GEN"/shard*.jsonl 2> /dev/null | wc -l; }

# -- launch 1: cut by the deadline mid-generate ----------------------------------
log "launch 1: LOOP_DEADLINE in 40 s, drain 10 s before"
LOOP_DEADLINE=$(( $(date +%s) + 40 )) EXIT_ON_SIGNAL=1 "$REPO/scripts/train_loop.sh" > "$S/launch1.out" 2>&1
RC=$?
AT=$(lines)
[ "$RC" -eq 75 ] && log "launch 1: exit 75 with $AT records on disk" || { log "FAIL launch 1: exit $RC (want 75)"; FAILED=1; }
grep -q "gen01 generate stopped before the end" "$S/run/status.md" || { log "FAIL launch 1: no deadline line on status.md"; FAILED=1; }
grep -q "draining" "$S/run/hub.log" || { log "FAIL launch 1: the job was never drained"; FAILED=1; }
for f in "$GEN"/shard*.jsonl; do
    [ -s "$f" ] && [ "$(tail -c 1 "$f" | od -An -c | tr -d ' ')" != '\n' ] && { log "FAIL launch 1: $f ends mid-line"; FAILED=1; }
done
[ -e "$GEN/.generated" ] && { log "FAIL launch 1: marked generated"; FAILED=1; }
[ "$AT" -gt 0 ] || log "note: nothing was written before the deadline (slow box?)"

# -- launch 2: resumes, finishes; STOP once it is under way -------------------------
log "launch 2: no deadline; STOP after the phase resumes"
"$REPO/scripts/train_loop.sh" > "$S/launch2.out" 2>&1 &
LOOP=$!
i=0
until [ "$(grep -c 'gen01 generate: 8x' "$S/run/status.md")" -ge 2 ]; do
    i=$((i + 1)); [ "$i" -gt 600 ] && { log "FAIL launch 2: never reached generate"; kill "$LOOP"; exit 1; }; sleep 0.5
done
touch "$S/run/STOP"
wait "$LOOP"; RC=$?
[ "$RC" -eq 0 ] || { log "FAIL launch 2: exit $RC (want 0 at the STOP)"; FAILED=1; }
grep "gen01 generate" "$S/run/status.md" | sed 's/^/    /' | tee -a "$S/test.log"
grep "resuming" "$S/run/hub.log" | sed 's/^/    /' | tee -a "$S/test.log"
[ -e "$GEN/.generated" ] || { log "FAIL launch 2: gen01 not marked generated"; FAILED=1; }
"$PY" - "$GEN" "$GAMES_PER_SHARD" << 'EOF' | tee -a "$S/test.log" || FAILED=1
import json, sys
d, games = sys.argv[1], int(sys.argv[2])
bad, total, follow = [], 0, 0
for k in range(8):
    raw = open(f'{d}/shard{k}.jsonl', 'rb').read()
    if not raw.endswith(b'\n'): bad.append(f'shard{k} ends mid-line')
    seen = {}
    for line in raw.split(b'\n')[:-1]:
        m = json.loads(line)['meta']
        seen.setdefault(m['seed'], []).append(m['extra'].get('drive'))
    base = 10000000 + 1000000 + k * 100000   # SEED_BASE + G*1e6 + K*1e5, G = 1
    if set(seen) != set(range(base, base + games)): bad.append(f'shard{k}: seed set differs')
    bad += [f'shard{k} seed {s}: {v}' for s, v in seen.items() if v not in (['1'], ['1', '2'])]
    total += len(seen); follow += sum(len(v) == 2 for v in seen.values())
print(f'{total} games in 8 shards, {follow} with a follow-on drive')
if bad: print('FAIL: ' + '; '.join(bad[:10])); sys.exit(1)
EOF
[ "$FAILED" -eq 0 ] && log "ALL PASS ($S)" || log "SOME FAILED ($S)"
exit "$FAILED"
