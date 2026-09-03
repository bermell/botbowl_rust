#!/usr/bin/env bash
# Overnight search-strength matrix — plan 027.
#
# Waits for the training loop to exit (it has STOP armed and finishes after
# gen04's gate), then runs head-to-head arms that differ in exactly one search
# parameter, same weights on both sides. Arms run in priority order and each
# writes its own report, so a run cut short still yields every completed arm.
#
#   nohup scripts/exp_overnight.sh > /dev/null 2>&1 &
#   touch runs/exp-search/STOP     # exits after the current arm

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"

RUN_DIR="$REPO/runs/loop14x7"
OUT="$REPO/runs/exp-search"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/14x7}"
export CARGO_TARGET_DIR
UI="$CARGO_TARGET_DIR/release/botbowl-ui"
PY="$REPO/train/.venv/bin/python"
SUMMARY="$REPO/scripts/eval_summary.py"
SOCK=/tmp/bbnn-exp.sock
PARALLEL="${PARALLEL:-6}"
GAMES="${GAMES:-60}"

mkdir -p "$OUT"
LOG="$OUT/exp.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# ---- wait for the production loop to release the box ------------------------
while pgrep -f "bash scripts/train_loop.sh" > /dev/null 2>&1; do
    [ -e "$OUT/STOP" ] && { log "STOP before start"; exit 0; }
    sleep 60
done
log "loop has exited; box is free"

MODEL=$(cat "$RUN_DIR/champion.txt")
[ -f "$MODEL" ] || { log "FATAL: champion $MODEL missing"; exit 1; }
log "champion: $(basename "$MODEL")"

# ---- build the horizon-aware binary -----------------------------------------
# The loop's binary predates commit 4b38838, so --horizon-turns would be an
# unknown flag. Build now that nothing is running.
if ! nice -n 5 cargo build --release -p botbowl-ui >> "$LOG" 2>&1; then
    log "FATAL: cargo build failed"; exit 1
fi
"$UI" eval --help 2>&1 | grep -q -- "--horizon-turns" \
    || { log "FATAL: built binary has no --horizon-turns"; exit 1; }
log "build ok, --horizon-turns present"

# ---- sidecar ----------------------------------------------------------------
NN_PID=""
if [ -f "${MODEL%.onnx}.pt" ]; then
    rm -f "$SOCK"
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda \
        --model "$MODEL" --stats-every 600 >> "$OUT/nn_server.log" 2>&1 &
    NN_PID=$!
    i=0
    while [ ! -S "$SOCK" ]; do
        i=$((i + 1))
        if [ "$i" -gt 120 ] || ! kill -0 "$NN_PID" 2>/dev/null; then
            log "WARN: sidecar did not come up — arms will run on tract (slower)"
            NN_PID=""; break
        fi
        sleep 1
    done
    [ -n "$NN_PID" ] && log "sidecar up (pid $NN_PID)"
fi
cleanup() { [ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
NN_ARGS=""
[ -n "$NN_PID" ] && NN_ARGS="--nn-server $SOCK"

# ---- arms -------------------------------------------------------------------
# name | cand_eval | cand_iters | cand_horizon | opp_eval | opp_iters | opp_horizon | games | seed
# Priority order: the two that change production settings first, then the
# confounded one. E2 arms use nn-value on both sides = the production config.
ARMS="
e2b-2000v1000|nn-value|2000|1|nn-value|1000|1|60|81000000
e2a-1000v500|nn-value|1000|1|nn-value|500|1|60|82000000
e1-priors-1000|nn|1000|1|nn-value|1000|1|60|83000000
e3-horizon2v1|nn-value|500|2|nn-value|500|1|60|84000000
e2c-4000v2000|nn-value|4000|1|nn-value|2000|1|40|85000000
"

for ARM in $ARMS; do
    [ -e "$OUT/STOP" ] && { log "STOP file — exiting"; break; }
    NAME=$(echo "$ARM" | cut -d'|' -f1)
    CE=$(echo "$ARM"   | cut -d'|' -f2)
    CI=$(echo "$ARM"   | cut -d'|' -f3)
    CH=$(echo "$ARM"   | cut -d'|' -f4)
    OE=$(echo "$ARM"   | cut -d'|' -f5)
    OI=$(echo "$ARM"   | cut -d'|' -f6)
    OH=$(echo "$ARM"   | cut -d'|' -f7)
    NG=$(echo "$ARM"   | cut -d'|' -f8)
    SD=$(echo "$ARM"   | cut -d'|' -f9)
    REPORT="$OUT/$NAME.json"
    [ -e "$REPORT" ] && { log "$NAME already done, skipping"; continue; }

    log "$NAME: candidate $CE iters=$CI horizon=$CH  vs  opponent $OE iters=$OI horizon=$OH, $NG games x$PARALLEL"
    START=$SECONDS
    # shellcheck disable=SC2086
    if "$UI" eval \
            --evaluator "$CE" --model "$MODEL" \
            --mcts-iters "$CI" --horizon-turns "$CH" \
            --vs-evaluator "$OE" --vs-model "$MODEL" \
            --opponent-iters "$OI" --vs-horizon-turns "$OH" \
            --vs-games "$NG" --seed "$SD" \
            --skip-lectures --skip-fixed-rungs \
            --parallel-games "$PARALLEL" $NN_ARGS \
            --out "$REPORT" > "$OUT/$NAME.log" 2>&1; then
        log "$NAME done ($(((SECONDS - START) / 60)) min): $("$PY" "$SUMMARY" "$REPORT" 2>/dev/null || echo 'see json')"
    else
        log "$NAME FAILED after $(((SECONDS - START) / 60)) min — see $NAME.log"
        tail -n 3 "$OUT/$NAME.log" >> "$LOG" 2>/dev/null
    fi
done
log "matrix finished"
