#!/usr/bin/env bash
# exp086 (plan 069 item 4b): MC labels streamed inside a generation vs the separate label phase.
#
#   scripts/exp086_genlabel.sh OUT [phases]          # phases: eq prod time (default all three)
#
#   eq    equivalence, deterministic GPU (BLOOD_NN_BUCKETS=1, cuDNN deterministic): a 2-shard
#         generation with streamed labels; `job label` and `botbowl-ui mc-label` over the same raw
#         shards; the three mc/ outputs compared byte for byte (`cmp`).
#   prod  the same at the loop's GPU settings (buckets to 64, nondeterministic batching): streamed vs
#         `job label` labels compared sample by sample (last-bit batching differences can flip a
#         near-tied argmax in a playout, as two separate label runs can).
#   time  wall time and GPU use: sequential (generate with G streams, then `job label` on the same
#         G streams) against streamed (G game + L label streams in one worker), for each L in
#         LS; every arm plays the same seeds. positions/s per kind from the queue's stats lines.
#
# Env: BIN (dir with botbowl-hub, botbowl-worker, botbowl-ui built with torch at 20x11/7; default
# target/agents/label/20x11/release), NET (default models/az_v7/bbnet_mix16x9v9_gen58.onnx, .ts
# beside it), PORT (13501), GAMES (8 per shard for eq/prod), TGAMES (24 per shard for time),
# G (24), LS ("8 24"), PLAYOUTS (8). Never touches runs/ or port 13337.
set -uo pipefail
export LC_ALL=C
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="$(git -C "$REPO" worktree list | head -1 | awk '{print $1}')"
OUT="${1:?out dir}"; shift
PHASES="${*:-eq prod time}"
BIN="${BIN:-$MAIN/target/agents/label/20x11/release}"
NET="${NET:-$MAIN/models/az_v7/bbnet_mix16x9v9_gen58.onnx}"
PORT="${PORT:-13501}"
GAMES="${GAMES:-8}"; TGAMES="${TGAMES:-24}"; G="${G:-24}"; LS="${LS:-8 24}"; PLAYOUTS="${PLAYOUTS:-8}"
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
TOKEN="$OUT/hub.token"
HUB_URL="http://127.0.0.1:$PORT"
SIZES=(--size-centre 220 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8 --size-max-area 220
       --size-min-area 70 --cells-per-player 26)
GEN=(--mode random-start --next-drive --mcts-iters 1000 --bot-config "$REPO/cfgs/chance_all_partial_gen.toml"
     --evaluator nn --model "$NET" "${SIZES[@]}" --shard-seed-stride 100000 --batch 1 --truncate)
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/exp086.log"; }

HUB_PID=""; WORKER_PID=""; SMI_PID=""
hub_start() {
    "$BIN/botbowl-hub" serve --bind "127.0.0.1:$PORT" --token-file "$TOKEN" --no-play \
        --model-index-dir "$OUT/noindex" --registry-dir "$OUT/noindex" --allowed-commits "$OUT/no-allowlist.toml" \
        >> "$OUT/hub.log" 2>&1 &
    HUB_PID=$!
    mkdir -p "$OUT/noindex"
    for _ in $(seq 60); do grep -q "listening on" "$OUT/hub.log" 2>/dev/null && return 0; sleep 1; done
    log "hub did not start"; exit 1
}
# $1 game streams, $2 label streams, $3 log; the rest is env (VAR=value ...) for the worker.
worker_start() {
    local g="$1" l="$2" wlog="$3"; shift 3
    env "$@" BLOOD_NN_INPROC_STATS=10 "$BIN/botbowl-worker" --hub "ws://127.0.0.1:$PORT/ws" --token-file "$TOKEN" \
        --name exp086 --parallel-games "$g" --label-streams "$l" --nn-backend inproc --nn-device cuda:0 \
        --cache-dir "$OUT/cache" >> "$wlog" 2>&1 &
    WORKER_PID=$!
    sleep 3
}
worker_stop() { [ -n "$WORKER_PID" ] && { kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null; }; WORKER_PID=""; }
smi_start() { nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 > "$1" 2>/dev/null & SMI_PID=$!; }
smi_stop() { [ -n "$SMI_PID" ] && kill "$SMI_PID" 2>/dev/null; SMI_PID=""; }
cleanup() { smi_stop; worker_stop; [ -n "$HUB_PID" ] && kill "$HUB_PID" 2>/dev/null; }
trap cleanup EXIT
job() { "$BIN/botbowl-hub" job "$1" --hub "$HUB_URL" --token-file "$TOKEN" "${@:2}"; }
stream_flags() {  # $1 out dir of the generation, $2 seed
    echo "--mc-label-playouts $PLAYOUTS --mc-label-seed $2 --mc-label-out-dir $1/mc --mc-label-chunk-samples 32"
}
label_job() {  # $1 generation dir, $2 out dir, $3 seed
    job label --corpus "$1/shard0.jsonl" "$1/shard1.jsonl" --model "$NET" --playouts "$PLAYOUTS" --seed "$3" \
        --chunk-samples 32 --out-dir "$2" --wait
}
compare() {  # $1 $2 dirs of mc shards: sample-level label agreement
    "$MAIN/train/.venv/bin/python" - "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
n = diff = traj = tdiff = 0
for f in sorted(a.glob("shard*.jsonl")):
    for la, lb in zip(open(f), open(b / f.name)):
        ta, tb = json.loads(la), json.loads(lb)
        traj += 1
        va = [s.get("outcome_value") for s in ta["samples"]]
        vb = [s.get("outcome_value") for s in tb["samples"]]
        d = sum(x != y for x, y in zip(va, vb))
        n += len(va); diff += d; tdiff += d > 0
print(f"{traj} trajectories, {n} samples: {diff} labels differ ({100.0 * diff / max(n, 1):.2f}%), in {tdiff} trajectories")
EOF
}

hub_start
log "commit $(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet || echo -dirty), net $NET, bins $BIN"

if [[ " $PHASES " == *" eq "* ]]; then
    D="$OUT/eq"; mkdir -p "$D"
    DET=(BLOOD_NN_BUCKETS=1 BLOOD_NN_DETERMINISTIC=1)
    worker_start 12 4 "$D/worker.log" "${DET[@]}"
    log "eq: generate 2x$GAMES games with streamed labels (deterministic GPU)"
    job generate "${GEN[@]}" --games "$GAMES" --seed-base 85000000 --shards "0 1" --out-dir "$D/gen" \
        $(stream_flags "$D/gen" 56085) --label "exp086 eq" --wait > "$D/generate.log" 2>&1 || { log "eq generate failed"; exit 1; }
    label_job "$D/gen" "$D/job_mc" 56085 > "$D/job_label.log" 2>&1 || { log "eq job label failed"; exit 1; }
    worker_stop
    env "${DET[@]}" "$BIN/botbowl-ui" mc-label --corpus "$D/gen/shard0.jsonl" "$D/gen/shard1.jsonl" --model "$NET" \
        --nn-backend inproc --nn-device cuda:0 --playouts "$PLAYOUTS" --parallel 8 --seed 56085 \
        --out-dir "$D/local_mc" 2> "$D/mc_label.log" || { log "eq mc-label failed"; exit 1; }
    for k in 0 1; do
        for o in job_mc local_mc; do
            if cmp -s "$D/gen/mc/shard$k.jsonl" "$D/$o/shard$k.jsonl"; then r=identical; else r=DIFFERENT; fi
            log "eq: streamed mc/shard$k.jsonl vs $o: $r ($(wc -l < "$D/gen/shard$k.jsonl") records)"
        done
    done
    log "eq: streamed vs job label: $(compare "$D/gen/mc" "$D/job_mc")"
    log "eq: streamed vs mc-label:  $(compare "$D/gen/mc" "$D/local_mc")"
fi

if [[ " $PHASES " == *" prod "* ]]; then
    D="$OUT/prod"; mkdir -p "$D"
    worker_start 12 4 "$D/worker.log"
    log "prod: generate 2x$GAMES games with streamed labels (the loop's GPU settings)"
    job generate "${GEN[@]}" --games "$GAMES" --seed-base 85100000 --shards "0 1" --out-dir "$D/gen" \
        $(stream_flags "$D/gen" 56086) --label "exp086 prod" --wait > "$D/generate.log" 2>&1 || { log "prod generate failed"; exit 1; }
    label_job "$D/gen" "$D/job_mc" 56086 > "$D/job_label.log" 2>&1 || { log "prod job label failed"; exit 1; }
    worker_stop
    for k in 0 1; do
        cmp -s "$D/gen/mc/shard$k.jsonl" "$D/job_mc/shard$k.jsonl" && r=identical || r=different
        log "prod: streamed mc/shard$k.jsonl vs job label: $r"
    done
    log "prod: streamed vs job label: $(compare "$D/gen/mc" "$D/job_mc")"
fi

# Positions/s per kind from a worker log's stats lines: the last line minus the first after $2 s.
rates() {
    "$MAIN/train/.venv/bin/python" - "$1" <<'EOF'
import re, sys
xs = []
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("NN_INPROC_STATS"):
        s = int(re.search(r"samples=(\d+)", line).group(1))
        m = re.search(r"label_samples=(\d+)", line)
        xs.append((s, int(m.group(1)) if m else 0))
if not xs:
    print("no stats"); sys.exit()
s, l = xs[-1]
print(f"forwards {s} (label {l}, gen {s - l})")
EOF
}
gpu_mean() { awk -F', ' '{u+=$2; n++} END {if (n) printf "%.0f%% over %d s", u/n, n}' "$1"; }

if [[ " $PHASES " == *" time "* ]]; then
    D="$OUT/time"; mkdir -p "$D"
    SEED=85200000
    # Sequential: the loop today (generate, then the label phase as a hub job on the same streams).
    A="$D/seq"; mkdir -p "$A"
    worker_start "$G" 0 "$A/worker.log"
    smi_start "$A/gpu_gen.csv"; t0=$(date +%s)
    job generate "${GEN[@]}" --games "$TGAMES" --seed-base "$SEED" --shards "0 1" --out-dir "$A/gen" \
        --label "exp086 seq generate" --wait > "$A/generate.log" 2>&1 || { log "seq generate failed"; exit 1; }
    t1=$(date +%s); smi_stop; smi_start "$A/gpu_label.csv"
    label_job "$A/gen" "$A/gen/mc" 56087 > "$A/job_label.log" 2>&1 || { log "seq label failed"; exit 1; }
    t2=$(date +%s); smi_stop; worker_stop
    log "time seq G=$G: generate $((t1 - t0)) s (GPU $(gpu_mean "$A/gpu_gen.csv")) + label $((t2 - t1)) s (GPU $(gpu_mean "$A/gpu_label.csv")) = $((t2 - t0)) s; $(cat "$A"/gen/shard*.jsonl | wc -l) records, $(grep -o 'samples x' "$A/job_label.log" | wc -l) shards labelled; $(rates "$A/worker.log")"
    for L in $LS; do
        B="$D/stream_L$L"; mkdir -p "$B"
        worker_start "$G" "$L" "$B/worker.log"
        smi_start "$B/gpu.csv"; t0=$(date +%s)
        job generate "${GEN[@]}" --games "$TGAMES" --seed-base "$SEED" --shards "0 1" --out-dir "$B/gen" \
            $(stream_flags "$B/gen" 56087) --label "exp086 stream L$L" --wait > "$B/generate.log" 2>&1 || { log "stream L$L failed"; exit 1; }
        t2=$(date +%s); smi_stop; worker_stop
        GEN_DONE=$(grep -o 'wrote [0-9]* trajectories / [0-9]* samples in [0-9]* s' "$B/generate.log" | grep -o 'in [0-9]* s')
        log "time stream G=$G L=$L: generate+labels $((t2 - t0)) s (games done $GEN_DONE; GPU $(gpu_mean "$B/gpu.csv")); $(cat "$B"/gen/shard*.jsonl | wc -l) records; $(rates "$B/worker.log")"
    done
fi
log "done"
