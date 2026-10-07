#!/usr/bin/env bash
# Steady-state generation throughput (plan 058 §9): for each stream count, run the loop's
# generation (`dataset --mode random-start --next-drive`, the generation preset, one GPU sidecar)
# far longer than the measurement, skip a warm-up, then count decisions written in a fixed window.
# Unlike perf_gen_bench.sh there is no tail effect (no stream ever runs out of games).
#
#   scripts/perf_gen_steady.sh OUT_DIR NET.onnx "28 48 64" [WORKERS_LIST]
#   WARM=240 WINDOW=600 scripts/perf_gen_steady.sh runs/perf/steady models/az_v7/x.onnx "48 64" "1 2"
#   SIDECARS=2 scripts/perf_gen_steady.sh ...     # the streams split over two sidecars and two processes
#
# Per run it prints decisions/min, records/min, the generator's CPU (% of one core) and peak RSS,
# the box's lowest MemAvailable, mean GPU utilisation and the sidecar's mean batch. A run whose
# MemAvailable falls under MEM_FLOOR_MB (default 1200) is stopped early and marked so.
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; NET="${2:?net.onnx}"; LIST="${3:-28 48 64}"; WLIST="${4:-1}"
CONFIG="${CONFIG:-cfgs/gumbel16_f1000_gen.toml}"; ITERS="${ITERS:-1000}"; SEED="${SEED:-58100}"
WARM="${WARM:-240}"; WINDOW="${WINDOW:-600}"; MEM_FLOOR_MB="${MEM_FLOOR_MB:-1200}"
mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || { echo "build failed" >&2; exit 1; }
SIZES="--size-centre 144 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8 --size-max-area 144 --size-min-area 70 --cells-per-player 26"

decisions() { cat "$1" 2>/dev/null | grep -o '"chosen_action"' | wc -l; }
records() { cat "$1" 2>/dev/null | wc -l; }
cpu_ticks() { awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null || echo 0; }
mem_avail_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }

for W in $WLIST; do
for P in $LIST; do
    # SIDECARS=N splits the streams over N sidecars, each with its own dataset process.
    N="${SIDECARS:-1}"; TAG="p${P}w${W}s${N}"
    NNS=(); DSS=(); FILES=()
    for i in $(seq "$N"); do
        SOCK="/tmp/bbnn-steady-$$-$i.sock"; rm -f "$SOCK"
        "$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$NET" --stats-every 30 \
            --canvas 11x18 > "$OUT/nn_server.$TAG.$i.log" 2>&1 & NNS+=($!)
    done
    for i in $(seq "$N"); do
        SOCK="/tmp/bbnn-steady-$$-$i.sock"
        for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
        FILE="$OUT/games.$TAG.$i.jsonl"; FILES+=("$FILE")
        # shellcheck disable=SC2086
        "$UI" dataset --mode random-start --next-drive --games $((P * 50)) --seed $((SEED + 100000 * i)) \
            --bot-config "$CONFIG" --mcts-iters "$ITERS" --mcts-workers "$W" --evaluator nn --model "$NET" \
            --nn-server "$SOCK" --parallel-games $((P / N)) $SIZES --out "$FILE" --truncate \
            > "$OUT/dataset.$TAG.$i.log" 2>&1 & DSS+=($!)
    done
    alive() { local d; for d in "${DSS[@]}"; do kill -0 "$d" 2>/dev/null || return 1; done; }
    sum_decisions() { local f t=0; for f in "${FILES[@]}"; do t=$((t + $(decisions "$f"))); done; echo $t; }
    sum_records() { local f t=0; for f in "${FILES[@]}"; do t=$((t + $(records "$f"))); done; echo $t; }
    sum_cpu() { local d t=0; for d in "${DSS[@]}"; do t=$((t + $(cpu_ticks "$d"))); done; echo $t; }
    PEAK_RSS=0; MIN_AVAIL=999999; GPU_SUM=0; GPU_N=0; STOPPED=""
    sample() {   # one 10 s tick of the resource readings
        local rss=0 r d avail gpu
        for d in "${DSS[@]}"; do
            r=$(awk '/VmRSS/ {print int($2 / 1024)}' "/proc/$d/status" 2>/dev/null || echo 0); rss=$((rss + ${r:-0}))
        done
        [ "$rss" -gt "$PEAK_RSS" ] && PEAK_RSS=$rss
        avail=$(mem_avail_mb); [ "$avail" -lt "$MIN_AVAIL" ] && MIN_AVAIL=$avail
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        [ -n "$gpu" ] && { GPU_SUM=$((GPU_SUM + gpu)); GPU_N=$((GPU_N + 1)); }
        if [ "$avail" -lt "$MEM_FLOOR_MB" ]; then STOPPED="MEM (MemAvailable ${avail} MB)"; fi
    }
    T=0
    while [ "$T" -lt "$WARM" ] && alive && [ -z "$STOPPED" ]; do sleep 10; T=$((T + 10)); sample; done
    D0=$(sum_decisions); R0=$(sum_records); C0=$(sum_cpu); S0=$(date +%s)
    GPU_SUM=0; GPU_N=0
    T=0
    while [ "$T" -lt "$WINDOW" ] && alive && [ -z "$STOPPED" ]; do sleep 10; T=$((T + 10)); sample; done
    D1=$(sum_decisions); R1=$(sum_records); C1=$(sum_cpu); S1=$(date +%s)
    for d in "${DSS[@]}"; do kill "$d" 2>/dev/null; wait "$d" 2>/dev/null; done
    MB=""; SPS=0
    for i in $(seq "$N"); do
        st=$(grep 'samples/s' "$OUT/nn_server.$TAG.$i.log" | tail -1)
        MB="$MB$(echo "$st" | grep -o 'mean_batch=[0-9.]*' | cut -d= -f2) "
        SPS=$((SPS + $(echo "$st" | grep -o '[0-9]* samples/s' | cut -d' ' -f1)))
    done
    for n in "${NNS[@]}"; do kill "$n"; wait "$n" 2>/dev/null; done
    rm -f /tmp/bbnn-steady-$$-*.sock
    HZ=$(getconf CLK_TCK)
    awk -v p="$P" -v w="$W" -v n="$N" -v d0="$D0" -v d1="$D1" -v r0="$R0" -v r1="$R1" -v c0="$C0" -v c1="$C1" \
        -v s0="$S0" -v s1="$S1" -v hz="$HZ" -v rss="$PEAK_RSS" -v avail="$MIN_AVAIL" \
        -v gs="$GPU_SUM" -v gn="$GPU_N" -v mb="$MB" -v sps="$SPS" -v stop="$STOPPED" 'BEGIN {
        m = (s1 - s0) / 60; if (m <= 0) m = 1e-9
        printf "streams %3d x workers %d, %d sidecar(s): %6.1f decisions/min, %5.1f records/min; CPU %4.0f%%; peak RSS %5d MB, min MemAvailable %5d MB; GPU %3.0f%%; mean_batch %s; %d samples/s%s\n",
            p, w, n, (d1 - d0) / m, (r1 - r0) / m, 100 * (c1 - c0) / hz / (s1 - s0), rss, avail,
            gn ? gs / gn : -1, mb, sps, stop != "" ? "  STOPPED: " stop : ""
    }' | tee -a "$OUT/summary.txt"
    rm -f "${FILES[@]}"
done
done
