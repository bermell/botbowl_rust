#!/usr/bin/env bash
# exp057: at what search budget does gen21's root *decision* stop changing? It complements exp056
# (strength vs budget on drives) with a search-only measurement: the same random-start roots
# re-searched at a ladder of real-descent budgets, 3 independent repeats per cell, compared against
# the largest budget with a noise floor (scripts/convergence_topk.py). That covers top-1
# agreement, the top-3 sets, regret in the reference's own Q, and the TV of the cq training
# targets. The loop's "500 visits" is about 255 descents, so 250 is in the ladder.
#
# Waits for exp056 to finish (it needs the GPU and the memory), then measures one reference tree's
# peak memory to pick the reference budget and the thread count, so it cannot OOM the box the way
# exp056's first launch did.
#
#   scripts/exp057_convergence_topk.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp057"; mkdir -p "$OUT"
MODEL="$REPO/models/az_v7/bbnet_mix16x9_gen21.onnx"
STATES="${STATES:-60}"; REPEATS="${REPEATS:-3}"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET   # `--budgets` must count real descents (BudgetMode::Iterations)
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; SOCK=/tmp/bbnn-exp057.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
NN_PID=""
cleanup() { [ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD); waiting for exp056"
until grep -qE "\] (done$|FATAL)" "$REPO/runs/exp056/status.md" 2>/dev/null; do sleep 120; done
status "exp056 finished; building"
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || die "build"
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$MODEL" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"

# Pilot: peak RSS of one 16000-descent search on the bigger board, from a mid-turn root.
/usr/bin/time -f "%M" -o "$OUT/pilot.rss" "$UI" convergence --states 2 --repeats 1 --budgets 16000 \
    --board 16x9/6 --advance 1 --evaluator nn --model "$MODEL" --nn-server "$SOCK" \
    --out "$OUT/pilot.jsonl" >> "$OUT/pilot.log" 2>&1 || die "pilot"
RSS_MB=$(( $(tail -1 "$OUT/pilot.rss") / 1024 ))
AVAIL_MB=$(free -m | awk '/^Mem:/{print $7}')
REF=16000
PARALLEL=$(( (AVAIL_MB - 4096) * 10 / (RSS_MB * 13) ))   # 30% margin, 4 GB reserve
if [ "$PARALLEL" -lt 2 ]; then
    REF=8000; PARALLEL=$(( (AVAIL_MB - 4096) * 10 / (RSS_MB * 7) ))   # an 8000 tree is about half
fi
[ "$PARALLEL" -gt 8 ] && PARALLEL=8
[ "$PARALLEL" -lt 1 ] && PARALLEL=1
BUDGETS="125,250,500,1000,2000,4000,$REF"
[ "$REF" = 8000 ] && BUDGETS="125,250,500,1000,2000,4000,8000"
status "pilot: a 16000-descent search peaks at ${RSS_MB} MB, ${AVAIL_MB} MB available -> reference $REF, $PARALLEL threads, budgets $BUDGETS"

SECONDS=0
for board in 14x7/4 16x9/6; do
    for adv in 0 1; do
        tag="${board%%/*}_a$adv"
        [ -s "$OUT/conv_$tag.jsonl" ] && grep -q "wrote" "$OUT/conv_$tag.log" 2>/dev/null && continue
        "$UI" convergence --states "$STATES" --repeats "$REPEATS" --budgets "$BUDGETS" --board "$board" \
            --advance "$adv" --evaluator nn --model "$MODEL" --nn-server "$SOCK" --parallel "$PARALLEL" \
            --seed $(( 91000000 + adv * 100000 )) --out "$OUT/conv_$tag.jsonl" > "$OUT/conv_$tag.log" 2>&1 \
            || die "convergence $tag"
        status "$tag done ($((SECONDS / 60)) min): $(tail -1 "$OUT/conv_$tag.log")"
    done
done
"$PY" "$REPO/scripts/convergence_topk.py" "$OUT"/conv_*.jsonl > "$OUT/summary.txt" 2>&1 || die "summary"
status "summary in $OUT/summary.txt"
status "done"
