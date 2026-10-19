#!/usr/bin/env bash
# Steady-state generation throughput (plan 058 §9): for each stream count, run the loop's
# generation (`dataset --mode random-start --next-drive`, the generation preset, one GPU sidecar)
# far longer than the measurement, skip a warm-up, then count decisions written in a fixed window.
# Unlike perf_gen_bench.sh there is no tail effect (no stream ever runs out of games).
#
#   scripts/perf_gen_steady.sh OUT_DIR NET.onnx "28 48 64" [WORKERS_LIST]
#   WARM=240 WINDOW=600 scripts/perf_gen_steady.sh runs/perf/steady models/az_v7/x.onnx "48 64" "1 2"
#   SIDECARS=2 scripts/perf_gen_steady.sh ...     # the streams split over two sidecars and two processes
#   BACKEND=inproc scripts/perf_gen_steady.sh ... # plan 066: one process, the net in it (needs NET.ts)
#
# Per run it prints decisions/min, records/min, the generator's CPU (% of one core) and peak RSS,
# the sidecars' CPU and RSS (0 in-process), the box's lowest MemAvailable, mean GPU utilisation, the
# batcher's mean batch and GPU samples/s. A run whose MemAvailable falls under MEM_FLOOR_MB (default
# 1200) is stopped early and marked so.
#
# Env: BOARD (16x9/6; the build capacity), CANVAS (11x18), SIZES (the 16x9 loop's centred mix),
# CONFIG, ITERS, SEED, WARM, WINDOW, SIDECARS (1), BACKEND (sidecar | inproc). The 20x11 loop:
#   BOARD=20x11/7 CANVAS=13x22 CONFIG=cfgs/chance_all_partial_gen.toml \
#   SIZES="--size-centre 220 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8 \
#          --size-max-area 220 --size-min-area 70 --cells-per-player 26" ...
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; NET="${2:?net.onnx}"; LIST="${3:-28 48 64}"; WLIST="${4:-1}"
CONFIG="${CONFIG:-cfgs/gumbel16_f1000_gen.toml}"; ITERS="${ITERS:-1000}"; SEED="${SEED:-58100}"
WARM="${WARM:-240}"; WINDOW="${WINDOW:-600}"; MEM_FLOOR_MB="${MEM_FLOOR_MB:-1200}"
BOARD="${BOARD:-16x9/6}"; CANVAS="${CANVAS:-11x18}"; BACKEND="${BACKEND:-sidecar}"
mkdir -p "$OUT"
BW="${BOARD%%x*}"; REST="${BOARD#*x}"; BH="${REST%%/*}"; BP="${REST#*/}"
export BOARD_SIZE_W="$BW" BOARD_SIZE_H="$BH" BOARD_PLAYERS="$BP"
FEATURES=()
if [ "$BACKEND" = inproc ]; then
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/${BW}x${BH}-torch}"
    FEATURES=(--features torch)
else
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/${BW}x${BH}}"
fi
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="${PY:-$REPO/train/.venv/bin/python}"
cargo build --release -j 4 -p botbowl-ui "${FEATURES[@]}" >> "$OUT/build.log" 2>&1 || { echo "build failed" >&2; exit 1; }
DEFAULT_SIZES="--size-centre 144 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8 --size-max-area 144 --size-min-area 70 --cells-per-player 26"
SIZES="${SIZES:-$DEFAULT_SIZES}"

decisions() { cat "$1" 2>/dev/null | grep -o '"chosen_action"' | wc -l; }
records() { cat "$1" 2>/dev/null | wc -l; }
cpu_ticks() { awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null || echo 0; }
rss_mb() { awk '/VmRSS/ {print int($2 / 1024)}' "/proc/$1/status" 2>/dev/null || echo 0; }
mem_avail_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }

for W in $WLIST; do
for P in $LIST; do
    # SIDECARS=N splits the streams over N sidecars, each with its own dataset process. In-process
    # there is one process and no sidecar.
    N="${SIDECARS:-1}"; [ "$BACKEND" = inproc ] && N=1
    TAG="p${P}w${W}s${N}${BACKEND:0:1}"
    NNS=(); DSS=(); FILES=()
    if [ "$BACKEND" != inproc ]; then
        for i in $(seq "$N"); do
            SOCK="/tmp/bbnn-steady-$$-$i.sock"; rm -f "$SOCK"
            "$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$NET" --stats-every 30 \
                --canvas "$CANVAS" > "$OUT/nn_server.$TAG.$i.log" 2>&1 & NNS+=($!)
        done
    fi
    for i in $(seq "$N"); do
        NN=()
        if [ "$BACKEND" = inproc ]; then
            NN=(--nn-backend inproc)
        else
            SOCK="/tmp/bbnn-steady-$$-$i.sock"
            for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
            NN=(--nn-server "$SOCK")
        fi
        FILE="$OUT/games.$TAG.$i.jsonl"; FILES+=("$FILE")
        # shellcheck disable=SC2086
        BLOOD_NN_CANVAS="$CANVAS" BLOOD_NN_INPROC_STATS=10 \
        "$UI" dataset --mode random-start --next-drive --games $((P * 50)) --seed $((SEED + 100000 * i)) \
            --bot-config "$CONFIG" --mcts-iters "$ITERS" --mcts-workers "$W" --evaluator nn --model "$NET" \
            "${NN[@]}" --parallel-games $((P / N)) $SIZES --out "$FILE" --truncate \
            > "$OUT/dataset.$TAG.$i.log" 2>&1 & DSS+=($!)
    done
    alive() { local d; for d in "${DSS[@]}"; do kill -0 "$d" 2>/dev/null || return 1; done; }
    sum_decisions() { local f t=0; for f in "${FILES[@]}"; do t=$((t + $(decisions "$f"))); done; echo $t; }
    sum_records() { local f t=0; for f in "${FILES[@]}"; do t=$((t + $(records "$f"))); done; echo $t; }
    sum_cpu() { local d t=0; for d in "$@"; do t=$((t + $(cpu_ticks "$d"))); done; echo $t; }
    PEAK_RSS=0; PEAK_SRSS=0; MIN_AVAIL=999999; GPU_SUM=0; GPU_N=0; STOPPED=""
    sample() {   # one 10 s tick of the resource readings
        local rss=0 srss=0 d avail gpu
        for d in "${DSS[@]}"; do rss=$((rss + $(rss_mb "$d"))); done
        for d in "${NNS[@]}"; do srss=$((srss + $(rss_mb "$d"))); done
        [ "$rss" -gt "$PEAK_RSS" ] && PEAK_RSS=$rss
        [ "$srss" -gt "$PEAK_SRSS" ] && PEAK_SRSS=$srss
        avail=$(mem_avail_mb); [ "$avail" -lt "$MIN_AVAIL" ] && MIN_AVAIL=$avail
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -n "$gpu" ] && { GPU_SUM=$((GPU_SUM + gpu)); GPU_N=$((GPU_N + 1)); }
        if [ "$avail" -lt "$MEM_FLOOR_MB" ]; then STOPPED="MEM (MemAvailable ${avail} MB)"; fi
    }
    T=0
    while [ "$T" -lt "$WARM" ] && alive && [ -z "$STOPPED" ]; do sleep 10; T=$((T + 10)); sample; done
    D0=$(sum_decisions); R0=$(sum_records); C0=$(sum_cpu "${DSS[@]}"); SC0=$(sum_cpu "${NNS[@]}"); S0=$(date +%s)
    L0=$(grep -c 'NN_INPROC_STATS' "$OUT/dataset.$TAG.1.log" 2>/dev/null || echo 0)
    GPU_SUM=0; GPU_N=0
    T=0
    while [ "$T" -lt "$WINDOW" ] && alive && [ -z "$STOPPED" ]; do sleep 10; T=$((T + 10)); sample; done
    D1=$(sum_decisions); R1=$(sum_records); C1=$(sum_cpu "${DSS[@]}"); SC1=$(sum_cpu "${NNS[@]}"); S1=$(date +%s)
    for d in "${DSS[@]}"; do kill "$d" 2>/dev/null; wait "$d" 2>/dev/null; done
    MB=""; SPS=0
    if [ "$BACKEND" = inproc ]; then
        # Stats lines come every 10 s and are cumulative since load: the window's samples/s and
        # mean batch are the differences between the line before it and its last line.
        read -r SPS MB < <(grep 'NN_INPROC_STATS' "$OUT/dataset.$TAG.1.log" | awk -v l0="$L0" '
            { split($0, a, "batches="); split(a[2], b, " "); bt = b[1]
              split($0, c, "samples="); split(c[2], d, " "); sm = d[1]
              n++; if (n == l0) { b0 = bt; s0 = sm }; b1 = bt; s1 = sm }
            END { k = n - l0; if (k < 1) { print 0, "?"; exit }
                  printf "%d %.2f\n", (s1 - s0) / (10 * k), (b1 > b0) ? (s1 - s0) / (b1 - b0) : 0 }')
    else
        for i in $(seq "$N"); do
            st=$(grep 'samples/s' "$OUT/nn_server.$TAG.$i.log" | tail -1)
            MB="$MB$(echo "$st" | grep -o 'mean_batch=[0-9.]*' | cut -d= -f2) "
            SPS=$((SPS + $(echo "$st" | grep -o '[0-9]* samples/s' | cut -d' ' -f1)))
        done
    fi
    for n in "${NNS[@]}"; do kill "$n"; wait "$n" 2>/dev/null; done
    rm -f /tmp/bbnn-steady-$$-*.sock
    HZ=$(getconf CLK_TCK)
    awk -v p="$P" -v w="$W" -v n="$N" -v be="$BACKEND" -v d0="$D0" -v d1="$D1" -v r0="$R0" -v r1="$R1" -v c0="$C0" -v c1="$C1" \
        -v sc0="$SC0" -v sc1="$SC1" -v s0="$S0" -v s1="$S1" -v hz="$HZ" -v rss="$PEAK_RSS" -v srss="$PEAK_SRSS" -v avail="$MIN_AVAIL" \
        -v gs="$GPU_SUM" -v gn="$GPU_N" -v mb="$MB" -v sps="$SPS" -v stop="$STOPPED" 'BEGIN {
        m = (s1 - s0) / 60; if (m <= 0) m = 1e-9
        printf "%s streams %3d x workers %d, %d process(es): %6.1f decisions/min, %5.1f records/min; CPU %4.0f%% + sidecars %4.0f%%; peak RSS %5d MB + sidecars %5d MB, min MemAvailable %5d MB; GPU %3.0f%%; mean_batch %s; %d samples/s%s\n",
            be, p, w, n, (d1 - d0) / m, (r1 - r0) / m, 100 * (c1 - c0) / hz / (s1 - s0), 100 * (sc1 - sc0) / hz / (s1 - s0),
            rss, srss, avail, gn ? gs / gn : -1, mb, sps, stop != "" ? "  STOPPED: " stop : ""
    }' | tee -a "$OUT/summary.txt"
    rm -f "${FILES[@]}"
done
done
