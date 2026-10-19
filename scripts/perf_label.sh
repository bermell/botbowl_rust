#!/usr/bin/env bash
# MC-label throughput, sidecar vs in-process (plan 066 §3.4): `botbowl-ui mc-label` over the first
# TRAJ trajectories of a corpus, one stream per trajectory, timed end to end.
#
#   scripts/perf_label.sh OUT_DIR NET.onnx CORPUS.jsonl "sidecar inproc"
#
# Prints samples/min (samples x playouts are what the GPU sees), the process's CPU and peak RSS,
# the sidecar's CPU and RSS, and the batcher's mean batch. NET.ts must sit beside NET.onnx for
# `inproc`. Env: BOARD (20x11/7), CANVAS (13x22), TRAJ (48), PLAYOUTS (8).
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; NET="${2:?net.onnx}"; CORPUS="${3:?corpus}"; ARMS="${4:-sidecar inproc}"
BOARD="${BOARD:-20x11/7}"; CANVAS="${CANVAS:-13x22}"; TRAJ="${TRAJ:-48}"; PLAYOUTS="${PLAYOUTS:-8}"
PY="${PY:-$REPO/train/.venv/bin/python}"
BW="${BOARD%%x*}"; REST="${BOARD#*x}"; BH="${REST%%/*}"; BP="${REST#*/}"
export BOARD_SIZE_W="$BW" BOARD_SIZE_H="$BH" BOARD_PLAYERS="$BP"
mkdir -p "$OUT"
head -n "$TRAJ" "$CORPUS" > "$OUT/label_in.jsonl"
SAMPLES=$(grep -o '"chosen_action"' "$OUT/label_in.jsonl" | wc -l)
HZ=$(getconf CLK_TCK)
ticks() { awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null || echo 0; }
rss() { awk '/VmHWM/ {print int($2 / 1024)}' "/proc/$1/status" 2>/dev/null || echo 0; }
for ARM in $ARMS; do
    if [ "$ARM" = inproc ]; then
        T="$REPO/target/${BW}x${BH}-torch"; F=(--features torch); NN=(--nn-backend inproc)
    else
        T="$REPO/target/${BW}x${BH}"; F=(); NN=(--nn-server "/tmp/bbnn-label-$$.sock")
    fi
    CARGO_TARGET_DIR="$T" cargo build --release -j 4 -p botbowl-ui "${F[@]}" >> "$OUT/build.log" 2>&1 || { echo build failed >&2; exit 1; }
    SPID=""
    if [ "$ARM" != inproc ]; then
        rm -f "/tmp/bbnn-label-$$.sock"
        "$PY" scripts/nn_server.py --socket "/tmp/bbnn-label-$$.sock" --device cuda --model "$NET" --canvas "$CANVAS" \
            --stats-every 30 > "$OUT/nn_server.$ARM.log" 2>&1 & SPID=$!
        for _ in $(seq 120); do [ -S "/tmp/bbnn-label-$$.sock" ] && break; sleep 1; done
    fi
    rm -rf "$OUT/out.$ARM"; mkdir -p "$OUT/out.$ARM"
    S0=$(date +%s.%N); SC0=$( [ -n "$SPID" ] && ticks "$SPID" || echo 0)
    BLOOD_NN_CANVAS="$CANVAS" BLOOD_NN_INPROC_STATS=30 "$T/release/botbowl-ui" mc-label --corpus "$OUT/label_in.jsonl" \
        --model "$NET" "${NN[@]}" --playouts "$PLAYOUTS" --parallel "$TRAJ" --seed 56066 --out-dir "$OUT/out.$ARM" \
        > "$OUT/label.$ARM.log" 2>&1 & LPID=$!
    PEAK=0; SPEAK=0
    while kill -0 "$LPID" 2>/dev/null; do
        r=$(rss "$LPID"); [ "${r:-0}" -gt "$PEAK" ] && PEAK=$r
        [ -n "$SPID" ] && { r=$(rss "$SPID"); [ "${r:-0}" -gt "$SPEAK" ] && SPEAK=$r; }
        CPU=$(ticks "$LPID"); SCPU=$( [ -n "$SPID" ] && ticks "$SPID" || echo 0)
        sleep 5
    done
    wait "$LPID"; S1=$(date +%s.%N)
    if [ -n "$SPID" ]; then MB=$(grep 'samples/s' "$OUT/nn_server.$ARM.log" | tail -1 | grep -o 'mean_batch=[0-9.]*'); kill "$SPID"; wait "$SPID" 2>/dev/null
    else MB=$(grep 'NN_INPROC_STATS' "$OUT/label.$ARM.log" | tail -1 | grep -o 'mean_batch=[0-9.]*'); fi
    awk -v arm="$ARM" -v s="$SAMPLES" -v p="$PLAYOUTS" -v t0="$S0" -v t1="$S1" -v c="$CPU" -v sc="$SCPU" -v sc0="$SC0" -v hz="$HZ" \
        -v pk="$PEAK" -v spk="$SPEAK" -v mb="$MB" 'BEGIN { d = t1 - t0
        printf "%-8s %d samples x %d playouts in %.0f s: %.0f samples/min; CPU %.0f%% + sidecar %.0f%%; peak RSS %d MB + sidecar %d MB; %s\n",
            arm, s, p, d, 60 * s / d, 100 * c / hz / d, 100 * (sc - sc0) / hz / d, pk, spk, mb }' | tee -a "$OUT/summary.txt"
done
[ "$(echo "$ARMS" | wc -w)" -ge 2 ] && "$PY" scripts/parity066.py corpus "$OUT/out.sidecar" "$OUT/out.inproc" | tee -a "$OUT/summary.txt"
