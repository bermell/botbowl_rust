#!/usr/bin/env bash
# Plan 024 Stage 4: how many concurrent inference streams can we actually
# offer, and what does each way of offering them cost?
#
# The server's own --loadgen curve answers "what could the sidecar deliver
# at N streams" with the search taken out. This answers the other half:
# what a *real* generator can offer, and therefore which of the plan's
# three stream sources (more shard processes, more games per process, more
# MCTS workers per tree) is worth building.
#
# Every arm generates the same total number of games at the same iteration
# budget, so wall clock is directly comparable. `forwards/s` comes from the
# NN_PROFILE counter, which is process-global, so it is summed across
# shards.
#
#   scripts/nn_throughput_probe.sh <arm> [args...]
#
# Arms:
#   tract   S G          S shards x G games, no server (the baseline)
#   server  S G          S shards x G games, --nn-server
#   workers S G W        S shards, W MCTS workers per tree (one tree)
#   games   S G P        S shards, P parallel games per process (P trees)
set -uo pipefail
# `bc` emits '.' but printf %f parses per locale; a sv_SE box then rejects
# every number this script computes.
export LC_ALL=C

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export BOARD_SIZE_W=${BOARD_SIZE_W:-14}
export BOARD_SIZE_H=${BOARD_SIZE_H:-7}
export BOARD_PLAYERS=${BOARD_PLAYERS:-4}
export CARGO_TARGET_DIR="$REPO/target/14x7"
export BLOOD_NN_PROFILE=1

BIN="$CARGO_TARGET_DIR/release/botbowl-ui"
MODEL=${MODEL:-"$REPO/models/bbnet_14x7_bench.onnx"}
SOCK=${SOCK:-/tmp/bbnn.sock}
ITERS=${ITERS:-1000}
SEED=${SEED:-424000}
OUT=${OUT:-/tmp/nn_probe}

arm=${1:?arm}; shards=${2:?shards}; games=${3:?games}; extra=${4:-1}
mkdir -p "$OUT"
# All three, or an arm with fewer shards than its predecessor sums that
# predecessor's leftover CPU times and reports more cores than the box has.
rm -f "$OUT"/shard*.log "$OUT"/shard*.jsonl "$OUT"/shard*.time

flags=(--mode random-start --games "$games" --mcts-iters "$ITERS"
       --evaluator nn-value --model "$MODEL" --truncate)
case "$arm" in
  tract)   ;;
  server)  flags+=(--nn-server "$SOCK") ;;
  workers) flags+=(--nn-server "$SOCK" --mcts-workers "$extra") ;;
  games)   flags+=(--nn-server "$SOCK" --parallel-games "$extra") ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac

echo "== arm=$arm shards=$shards games=$games extra=$extra iters=$ITERS"
t0=$(date +%s.%N)
for k in $(seq 0 $((shards - 1))); do
  # `%U %S` per shard: CPU seconds are the metric that matters once the
  # box is saturated, because then wall clock only says how much of the
  # machine an arm managed to grab, not how efficiently it used it.
  /usr/bin/time -f '%U %S' -o "$OUT/shard$k.time" \
    "$BIN" dataset "${flags[@]}" \
      --seed $((SEED + k * 100000)) \
      --out "$OUT/shard$k.jsonl" > "$OUT/shard$k.log" 2>&1 &
done
wait
t1=$(date +%s.%N)

wall=$(echo "$t1 - $t0" | bc)
# The last NN_PROFILE line of each shard is its cumulative exit line.
fw=$(grep -h '^NN_PROFILE' "$OUT"/shard*.log | awk -F'forwards=' '{print $2}' \
     | awk '{print $1}' | paste -sd+ | bc)
fw=${fw:-0}
cpu=$(cat "$OUT"/shard*.time | awk '{s+=$1+$2} END{printf "%.1f", s}')
sys=$(cat "$OUT"/shard*.time | awk '{s+=$2} END{printf "%.1f", s}')
tot_games=$((shards * games))
printf 'arm=%-7s shards=%-2s games/shard=%-2s extra=%-2s | wall %6.1fs | cpu %7.1fs (%4.1f cores, sys %2.0f%%) | %7s fw | %5.0f fw/s | %6.2f s/game | %5.0f us cpu/fw\n' \
  "$arm" "$shards" "$games" "$extra" "$wall" "$cpu" \
  "$(echo "$cpu / $wall" | bc -l)" "$(echo "100 * $sys / $cpu" | bc -l)" "$fw" \
  "$(echo "$fw / $wall" | bc -l)" "$(echo "$wall / $tot_games" | bc -l)" \
  "$(echo "$cpu * 1000000 / $fw" | bc -l)"
grep -h 'NN_SERVER_FALLBACK' "$OUT"/shard*.log | head -2
