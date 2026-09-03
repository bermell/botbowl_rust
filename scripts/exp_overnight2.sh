#!/usr/bin/env bash
# Second concurrent stream for the plan-027 matrix.
#
# Stream 1 (exp_overnight.sh) runs at --parallel-games 6 but the box sits at
# load ~4/8: the games block on the sidecar rather than saturating CPU, so
# there is room for a second stream, and more concurrent requests should also
# batch better on the GPU. This one runs at 4 for a combined ~10 game threads
# on 8 cores — mild oversubscription, which costs wall time per arm but raises
# total throughput.
#
# Arms here follow from E2b (2000 v 1000 = 0.508, dead even): if 1000 is
# already past saturation the informative direction is *downward*, and each
# halving that holds is a halving of generation cost. These are all cheap
# low-iteration arms, so they finish fast and add a lot per hour.
#
#   nohup scripts/exp_overnight2.sh > /dev/null 2>&1 &
#   touch runs/exp-search/STOP2     # exits after the current arm

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4

RUN_DIR="$REPO/runs/loop14x7"
OUT="$REPO/runs/exp-search"
UI="${CARGO_TARGET_DIR:-$REPO/target/14x7}/release/botbowl-ui"
PY="$REPO/train/.venv/bin/python"
SUMMARY="$REPO/scripts/eval_summary.py"
SOCK=/tmp/bbnn-exp.sock          # shared with stream 1; the sidecar is multi-client
PARALLEL="${PARALLEL:-4}"

LOG="$OUT/exp2.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
MODEL=$(cat "$RUN_DIR/champion.txt")
[ -f "$MODEL" ] || { log "FATAL: champion missing"; exit 1; }
NN_ARGS=""
[ -S "$SOCK" ] && NN_ARGS="--nn-server $SOCK"
log "stream 2 start: $(basename "$MODEL"), parallel $PARALLEL, sidecar=${NN_ARGS:-none}"

# name | cand_eval | cand_iters | cand_horizon | opp_eval | opp_iters | opp_horizon | games | seed
# Finding the floor: keep halving until strength actually drops.
# e1b extends the 200-iter prior screen (24 games so far, 0.604) toward 100.
ARMS="
e2d-500v250|nn-value|500|1|nn-value|250|1|60|86000000
e1b-priors-200ext|nn|200|1|nn-value|200|1|76|87000000
e2e-250v125|nn-value|250|1|nn-value|125|1|60|88000000
e2f-1000v250|nn-value|1000|1|nn-value|250|1|60|89000000
"

for ARM in $ARMS; do
    [ -e "$OUT/STOP2" ] && { log "STOP2 — exiting"; break; }
    NAME=$(echo "$ARM" | cut -d'|' -f1)
    CE=$(echo "$ARM"   | cut -d'|' -f2); CI=$(echo "$ARM" | cut -d'|' -f3)
    CH=$(echo "$ARM"   | cut -d'|' -f4); OE=$(echo "$ARM" | cut -d'|' -f5)
    OI=$(echo "$ARM"   | cut -d'|' -f6); OH=$(echo "$ARM" | cut -d'|' -f7)
    NG=$(echo "$ARM"   | cut -d'|' -f8); SD=$(echo "$ARM" | cut -d'|' -f9)
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
log "stream 2 finished"
