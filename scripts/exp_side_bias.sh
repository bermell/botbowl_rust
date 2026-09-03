#!/usr/bin/env bash
# Stream 3 — measure the Home/Away side bias directly (plan 027).
#
# Why this exists. Every parameter arm in the matrix has come back ~0.50, but
# pooling the side splits across 562 decided games shows Away winning 55.7%
# (p~0.007), and 62% across the mirror-like arms where both sides run the same
# net (n=116). The side you play is worth more than any parameter we are tuning.
#
# It hid for a long time because a *candidate's* win rate cannot see it: seats
# alternate Home/Away, so a side advantage cancels out of the headline number.
# Plan 021's "0.40 mirror anomaly" is a different statistic — a seat effect —
# and at 19/48 decided (p~0.15) it is not significant and does not reproduce in
# this matrix's near-mirrors (0.508 / 0.517 / 0.604).
#
# So: run true mirrors, identical bots on both sides, and read the side split
# rather than the seat split. Heuristic first because it needs no NN and no GPU
# — if the bias is in the engine or the board it will show there, which also
# tells us whether it is a bot problem or a game problem.
#
# Each arm also writes --per-game-out, which logs `kicking_first_half` per
# game. That is what turns this from "measure the bias" into "decompose it":
# game_procs.rs:90 gives the *receiving* team the first turn of each round, so
# the *kicking* team takes the last turn of each half — the last word, which is
# worth a lot. That asymmetry is symmetric across Home/Away only if the coin is
# fair, and the per-game rows let us check the coin and the last-turn effect
# separately instead of inferring either.
#
# Waits for stream 2 so the box is not oversubscribed.
#
#   nohup scripts/exp_side_bias.sh > /dev/null 2>&1 &
#   touch runs/exp-search/STOP3

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4

RUN_DIR="$REPO/runs/loop14x7"
OUT="$REPO/runs/exp-search"
UI="${CARGO_TARGET_DIR:-$REPO/target/14x7}/release/botbowl-ui"
PY="$REPO/train/.venv/bin/python"
SUMMARY="$REPO/scripts/eval_summary.py"
SOCK=/tmp/bbnn-exp.sock
PARALLEL="${PARALLEL:-6}"

LOG="$OUT/exp3.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

while pgrep -f "bash scripts/exp_overnight2.sh" > /dev/null 2>&1; do
    [ -e "$OUT/STOP3" ] && { log "STOP3 before start"; exit 0; }
    sleep 60
done
log "stream 2 done; starting side-bias mirrors"

MODEL=$(cat "$RUN_DIR/champion.txt")

# name | evaluator | model? | iters | games | seed
# A true mirror: identical evaluator, identical iters, identical horizon.
# The candidate's points should sit at 0.50 by construction; the number we
# actually want is the side split in the same row.
ARMS="
mirror-heuristic|heuristic|no|1000|120|91000000
mirror-nnvalue|nn-value|yes|1000|100|92000000
mirror-heuristic-500|heuristic|no|500|120|93000000
"

for ARM in $ARMS; do
    [ -e "$OUT/STOP3" ] && { log "STOP3 — exiting"; break; }
    NAME=$(echo "$ARM" | cut -d'|' -f1)
    EV=$(echo "$ARM"   | cut -d'|' -f2)
    USEM=$(echo "$ARM" | cut -d'|' -f3)
    IT=$(echo "$ARM"   | cut -d'|' -f4)
    NG=$(echo "$ARM"   | cut -d'|' -f5)
    SD=$(echo "$ARM"   | cut -d'|' -f6)
    REPORT="$OUT/$NAME.json"
    [ -e "$REPORT" ] && { log "$NAME already done, skipping"; continue; }

    MARGS=""; NN_ARGS=""
    if [ "$USEM" = "yes" ]; then
        MARGS="--model $MODEL --vs-model $MODEL"
        [ -S "$SOCK" ] && NN_ARGS="--nn-server $SOCK"
    fi

    log "$NAME: true mirror, $EV both sides, iters=$IT, $NG games x$PARALLEL"
    START=$SECONDS
    # shellcheck disable=SC2086
    if "$UI" eval \
            --evaluator "$EV" $MARGS \
            --mcts-iters "$IT" --opponent-iters "$IT" \
            --vs-evaluator "$EV" \
            --vs-games "$NG" --seed "$SD" \
            --skip-lectures --skip-fixed-rungs \
            --parallel-games "$PARALLEL" $NN_ARGS \
            --per-game-out "$OUT/$NAME.games.jsonl" \
            --out "$REPORT" > "$OUT/$NAME.log" 2>&1; then
        log "$NAME done ($(((SECONDS - START) / 60)) min): $("$PY" "$SUMMARY" "$REPORT" 2>/dev/null || echo 'see json')"
    else
        log "$NAME FAILED after $(((SECONDS - START) / 60)) min — see $NAME.log"
        tail -n 3 "$OUT/$NAME.log" >> "$LOG" 2>/dev/null
    fi
done
log "side-bias stream finished"
