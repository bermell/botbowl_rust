#!/usr/bin/env bash
# exp078 (plan 067): do 16-playout MC value labels train a better net than the loop's 8?
#
# One generation step, exp069/exp073 style, at a plateau generation P of the live loop. The parent is
# the loop's gen P net (the generator and warm start of gen P+1); the window is exactly what
# train_loop.sh's prepare uses for gen P+1 (window_shards: gens P-1..P+1, train shards 0-3,5,6, val
# shards 4,7, each gen's mc/ copy). Two arms, everything identical but the train labels:
#   A  the loop's own 8-playout labels;
#   B  the same shards topped up to 16 playouts (`job label --top-up 8`, exact: plan 067), each gen
#      under the net that made its labels (read back from the labels' tag).
# Both arms: prepare --policy-target cq --tau 100 --value-blend 1.0; bbnn.train with the loop's recipe
# (warm start from gen P's .pt, lr 5e-5, 3 epochs, eval every 1000 + at 250/500, select on combined,
# value weight 0.36 per drive, --freeze-bn) and one shared --seed, so both arms see the same batches
# and flips. Selection (plan 067 §4): both arms restore on the SAME val set, the window's val shards
# with 16-playout labels (SEL=16; SEL=8 = the loop's own val labels). Then, per arm and the parent:
# absorption on gen P+1's raw held-out shards (the loop's probe) and on the window val set under 8-
# and 16-playout labels, both value benches (paired vs the parent), the net check (gen P+1's held-out
# shards), and with DRIVES=1 paired drives B vs A (C1 both sides, both drive sets, SPRT as the loop's
# benchmark). The top-up's records give the label noise directly (scripts/label_noise.py).
#
#   P=46 scripts/exp078_label_topup.sh                  # outputs: runs/exp078/ (status.md, label_noise.txt, A/, B/, ...)
#   P=46 DRIVES=1 scripts/exp078_label_topup.sh         # + drives B vs A (resumes: every finished step is skipped)
#
# When: after the loop has mc-labelled gen P+1 (gen(P+1)/.mc_labelled) and been stopped before it
# trains (touch STOP during that mc-label: it exits "before gen(P+1) prepare"). With WAIT_LOOP=1 (the
# default) the script waits for both, then has the GPU to itself: the top-up (~10-12 h, see plan 067
# §6), two trainings, probes, benches and net checks. It never touches the loop's run dir or models
# (nets without a .ts get one exported into OUT/nets) and never relaunches the loop.
#
# Env (defaults in brackets):
#   P (required)                RUN [runs/loopmix16x9v9]  M [models/az_v7]  TIER [mix16x9v9]  OUT [runs/exp078]
#   TOPUP [8] playouts added    TOPUP_VIA [hub] hub|local SEED [78] trainer seed (both arms)
#   SEL [16] the val labels both arms restore on: 16 = the topped-up ones (8 + TOPUP; "16" names them in
#            file names whatever TOPUP is), 8 = the loop's own
#   EXTERNAL_HUB  submit the top-up and drives to this running hub (same commit, protocol v21+) and start
#                 no hub or worker; else own hub on HUB_BIND:PORT [127.0.0.1:13378] + one local worker
#                 of STREAMS [48] in-process streams. PORT=13337 HUB_BIND=0.0.0.0 with the loop stopped
#                 lets the helper workers (which dial 13337) rejoin and help.
#   DRIVES [0]  DRIVE_CAP [800]  DRIVE_SPRT [0.5:0.55]  DRIVE_POSITIONS [the loop's two sets]
#   NET_CHECK [on]  NET_CHECK_LADDER ["64 250 1000 4000"]  NET_CHECK_BOARDS [16x9,20x11]  OVERRIDES (net_check.sh)
#   VALUE_BENCH / VALUE_BENCH_2 [runs/value_bench/v9_gen01_val.jsonl / v9_20x11_val.jsonl]  VB_ROWS [all]
#   EPOCHS [3]  TRAIN_DEVICE [cuda]  TRAIN_EXTRA [""] appended to bbnn.train  WAIT_LOOP [1]  TRAIN_GPU_MB [2500]
#   Smoke only: SMOKE_LINES=N copies the first N lines of every window shard (raw and mc) into OUT/smoke_run
#   and runs on that; DIRTY_OK=1 allows a dirty tree.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
P="${P:?P = the plateau generation of the loop (its net is the parent; window P-1..P+1)}"
RUN="${RUN:-$REPO/runs/loopmix16x9v9}"; M="${M:-$REPO/models/az_v7}"; TIER="${TIER:-mix16x9v9}"
OUT="${OUT:-$REPO/runs/exp078}"; case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac; mkdir -p "$OUT"
TRAIN_SHARDS="0 1 2 3 5 6"; VAL_SHARDS="4 7"; WINDOW_GENS=3
TOPUP="${TOPUP:-8}"; TOPUP_VIA="${TOPUP_VIA:-hub}"; SEL="${SEL:-16}"; SEED="${SEED:-78}"
PORT="${PORT:-13378}"; HUB_BIND="${HUB_BIND:-127.0.0.1}"; STREAMS="${STREAMS:-48}"; CHUNK="${CHUNK:-32}"
DRIVES="${DRIVES:-0}"; DRIVE_CAP="${DRIVE_CAP:-800}"; DRIVE_SPRT="${DRIVE_SPRT:-0.5:0.55}"
DRIVE_POSITIONS="${DRIVE_POSITIONS:-$REPO/runs/loopmix16x9g/positions/contested_16x9_gen04g.json,$(ls -t "$RUN"/positions/contested_20x11_*.json 2>/dev/null | head -1)}"
DRIVE_POSITIONS="${DRIVE_POSITIONS%,}"
NET_CHECK="${NET_CHECK:-on}"; NET_CHECK_LADDER="${NET_CHECK_LADDER:-64 250 1000 4000}"
export NET_CHECK_BOARDS="${NET_CHECK_BOARDS:-16x9,20x11}"
C1="$REPO/cfgs/chance_all_partial.toml"
VALUE_BENCH="${VALUE_BENCH:-$REPO/runs/value_bench/v9_gen01_val.jsonl}"
VALUE_BENCH_2="${VALUE_BENCH_2:-$REPO/runs/value_bench/v9_20x11_val.jsonl}"
VB_ROWS="${VB_ROWS:-}"
EPOCHS="${EPOCHS:-3}"; TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"; TRAIN_EXTRA="${TRAIN_EXTRA:-}"
WAIT_LOOP="${WAIT_LOOP:-1}"; TRAIN_GPU_MB="${TRAIN_GPU_MB:-2500}"
# The loop's capacity and in-process backend (plan 065/066), in a target dir of our own: building into
# target/20x11 would swap the loop's binaries under it.
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/exp078}"
export NN_BACKEND=inproc; CANVAS="13x22"; export BLOOD_NN_CANVAS="$CANVAS"
unset BLOOD_MCTS_BUDGET
BIN="$CARGO_TARGET_DIR/release"; UI="$BIN/botbowl-ui"; HUB="$BIN/botbowl-hub"; WORKER="$BIN/botbowl-worker"; PREPARE="$BIN/prepare"
PY="${PY:-$REPO/train/.venv/bin/python}"; export PYTHONPATH="$REPO/train/src${PYTHONPATH:+:$PYTHONPATH}"
TOK="$HOME/.config/botbowl/hub.token"; HUB_URL="${EXTERNAL_HUB:-http://127.0.0.1:$PORT}"
PREP_ARGS=(--policy-target cq --tau 100 --value-blend 1.0)
STATUS="$OUT/status.md"
status() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
die() { status "FATAL: $*"; exit 1; }
gg() { printf 'gen%02d' "$1"; }
HUB_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM
loop_running() { ps -eo args | grep -q "^bash scripts/train_loop.s[h]\|/scripts/train_loop.s[h]"; }
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

source "$REPO/scripts/lib/git.sh"
[ "${DIRTY_OK:-0}" = 1 ] || tree_is_clean || die "dirty tree: commit first (the top-up's provenance is this commit)"
PARENT="${PARENT:-$M/bbnet_${TIER}_$(gg "$P")}"   # without extension
NEXT="$RUN/$(gg $((P + 1)))"
status "start: commit $(git rev-parse --short HEAD), P=$P (parent $(basename "$PARENT")), window $(gg $((P - 1)))-$(gg $((P + 1))), +$TOPUP playouts via $TOPUP_VIA, select on val$SEL, seed $SEED"

# ---- smoke: a few lines of every window shard -----------------------------------------------------
if [ -n "${SMOKE_LINES:-}" ]; then
    SRC="$RUN"; RUN="$OUT/smoke_run"
    for g in $(seq $((P - WINDOW_GENS + 2)) $((P + 1))); do
        d="$RUN/$(gg "$g")"; mkdir -p "$d/mc"; touch "$d/.generated" "$d/.mc_labelled"
        for k in $TRAIN_SHARDS $VAL_SHARDS; do
            [ -s "$d/shard$k.jsonl" ] || head -n "$SMOKE_LINES" "$SRC/$(gg "$g")/shard$k.jsonl" > "$d/shard$k.jsonl"
            [ -s "$d/mc/shard$k.jsonl" ] || head -n "$SMOKE_LINES" "$SRC/$(gg "$g")/mc/shard$k.jsonl" > "$d/mc/shard$k.jsonl"
        done
    done
    NEXT="$RUN/$(gg $((P + 1)))"
    status "smoke: first $SMOKE_LINES lines of every window shard, in $RUN"
fi

# ---- build (the loop's exact commands, so net_check.sh / value_bench.sh find nothing to rebuild) ----
if [ ! -e "$OUT/.built-$(git rev-parse --short HEAD)" ]; then
    ls "$HOME"/.cache/botbowl/cuda_nvcc/*/include/crt/host_config.h > /dev/null 2>&1 || [ -n "${BOTBOWL_CUDA_INCLUDE:-}" ] \
        || PY="$PY" scripts/fetch_cuda_headers.sh >> "$OUT/build.log" 2>&1 || die "fetch_cuda_headers.sh — see build.log"
    cargo build --release -j 4 -p botbowl-hub >> "$OUT/build.log" 2>&1 || die "build (hub) — see build.log"
    cargo build --release -j 4 -p botbowl-nn -p botbowl-worker --features botbowl-worker/torch >> "$OUT/build.log" 2>&1 \
        || die "build (worker) — see build.log"
    cargo build --release -j 4 -p botbowl-ui --features botbowl-ui/torch >> "$OUT/build.log" 2>&1 || die "build (ui) — see build.log"
    touch "$OUT/.built-$(git rev-parse --short HEAD)"
fi

# ---- the window, as train_loop.sh's window_shards for gen P+1 ------------------------------------
WIN=()
for g in $(seq $((P - WINDOW_GENS + 2)) $((P + 1))); do [ -e "$RUN/$(gg "$g")/.generated" ] && WIN+=("$g"); done
[ "${#WIN[@]}" -eq "$WINDOW_GENS" ] || die "window $(gg $((P - 1)))-$(gg $((P + 1))) incomplete: ${WIN[*]}"
if [ "$WAIT_LOOP" = 1 ]; then
    # A loop stopped before it labelled gen P+1 (STOP touched during generate exits "before
    # genP+1 mc-label") will never label it: say so instead of waiting forever.
    until [ -e "$NEXT/.mc_labelled" ]; do
        loop_running || die "$(gg $((P + 1))) is not mc-labelled and the loop is not running (STOP before its mc-label?): relaunch the loop, touch STOP during that mc-label, then rerun"
        sleep 300
    done
    if loop_running; then status "waiting for the loop to stop (touch $RUN/STOP during $(gg $((P + 1))) mc-label)"; fi
    while loop_running; do sleep 120; done
fi
[ -e "$NEXT/.mc_labelled" ] || die "$(gg $((P + 1))) is not mc-labelled"
# shards ARM KIND: arm A (8) reads the loop's mc/ shards, arm B (16) our topped-up ones.
shards() {
    local arm="$1" kind="$2" g k list out=""
    [ "$kind" = train ] && list="$TRAIN_SHARDS" || list="$VAL_SHARDS"
    for g in "${WIN[@]}"; do for k in $list; do
        if [ "$arm" = 8 ]; then out="$out $RUN/$(gg "$g")/mc/shard$k.jsonl"; else out="$out $OUT/mc16/$(gg "$g")/shard$k.jsonl"; fi
    done; done
    echo "$out"
}
for f in $(shards 8 train) $(shards 8 val); do [ -s "$f" ] || die "$f missing (the window must be MC-labelled)"; done

# ---- nets: every net loaded in-process needs NET.ts beside NET.onnx --------------------------------
mkdir -p "$OUT/nets"
# with_ts NET.onnx -> prints a path to the same net with a .ts beside it (the original when it has one,
# else a copy in OUT/nets with one exported from its .pt; the file name is kept, so the stem matches).
# Runs in $(...): it reports failure by its exit code, never through `die`.
with_ts() {
    local onnx="$1" b; b="$(basename "$onnx" .onnx)"
    if [ -s "${onnx%.onnx}.ts" ]; then echo "$onnx"; return 0; fi
    if [ ! -s "$OUT/nets/$b.ts" ]; then
        [ -f "${onnx%.onnx}.pt" ] || { echo "no ${onnx%.onnx}.pt to export $b.ts from" >> "$OUT/export.log"; return 1; }
        cp "$onnx" "${onnx%.onnx}.pt" "$OUT/nets/" || return 1
        "$PY" -m bbnn.export_ts "$OUT/nets/$b.pt" --out "$OUT/nets/$b.ts" --canvas "$CANVAS" >> "$OUT/export.log" 2>&1 || return 1
    fi
    echo "$OUT/nets/$b.onnx"
}
[ -f "$PARENT.pt" ] || die "no parent $PARENT.pt"
PARENT_ONNX="$(with_ts "$PARENT.onnx")" || die "no .ts for the parent — see export.log"

# ---- hub (own, unless EXTERNAL_HUB) ---------------------------------------------------------------
hub_up() {
    [ -n "${EXTERNAL_HUB:-}" ] && { "$HUB" status --hub "$HUB_URL" --token-file "$TOK" > /dev/null 2>&1 || die "no hub at $HUB_URL"; return 0; }
    [ -n "$HUB_PID" ] && kill -0 "$HUB_PID" 2>/dev/null && return 0
    "$HUB" status --hub "$HUB_URL" --token-file "$TOK" > /dev/null 2>&1 && die "something already serves $HUB_URL"
    "$HUB" serve --bind "$HUB_BIND:$PORT" --token-file "$TOK" --run-dir "$OUT" --no-play >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
    for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" > /dev/null 2>&1 && break; sleep 1; done
    "$HUB" status --hub "$HUB_URL" --token-file "$TOK" > /dev/null 2>&1 || die "hub did not come up — see hub.log"
    "$WORKER" --hub "ws://127.0.0.1:$PORT/ws" --token-file "$TOK" --name local --parallel-games "$STREAMS" \
        --cache-dir "$OUT/worker-cache" --nn-backend inproc --mem-floor-mb 1536 >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
    status "hub up on $HUB_BIND:$PORT (pid $HUB_PID), local worker x$STREAMS in-process (pid $WORKER_PID)"
}
hub_down() {
    for p in $WORKER_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    WORKER_PID="" HUB_PID=""
}

# ---- 1. top up every window gen's train+val shards -------------------------------------------------
# The net that labelled a gen is the one its tag names (the generator of that gen); the top-up refuses
# any other (botbowl_play::mc_label::check_topup_model).
tag_model() { "$PY" -c 'import json,sys
for line in open(sys.argv[1]):
    v = json.loads(line)["meta"]["extra"].get("value_label", "")
    if v.startswith("mc_policy("):
        print(v[v.index(",model=") + 7:v.rindex(",seed=")]); break' "$1"; }
PIDS=()
for g in "${WIN[@]}"; do
    G2="$(gg "$g")"; D="$OUT/mc16/$G2"; IN=""
    for k in $TRAIN_SHARDS $VAL_SHARDS; do IN="$IN $RUN/$G2/mc/shard$k.jsonl"; done
    done_all=1; for k in $TRAIN_SHARDS $VAL_SHARDS; do [ -s "$D/shard$k.jsonl" ] || done_all=0; done
    [ "$done_all" = 1 ] && continue
    NET="$(tag_model "$RUN/$G2/mc/shard${VAL_SHARDS%% *}.jsonl")"; [ -f "$NET" ] || die "$G2: its labels name $NET, which is not there"
    NET="$(with_ts "$NET")" || die "$G2: no .ts for its labelling net — see export.log"; mkdir -p "$D"
    status "$G2 top-up: $(echo $TRAIN_SHARDS $VAL_SHARDS | wc -w) shards +$TOPUP playouts under $(basename "$NET") via $TOPUP_VIA"
    if [ "$TOPUP_VIA" = hub ]; then
        hub_up
        # shellcheck disable=SC2086
        "$HUB" job label --hub "$HUB_URL" --token-file "$TOK" --label "exp078 $G2 top-up +$TOPUP" \
            --corpus $IN --model "$NET" --top-up "$TOPUP" --chunk-samples "$CHUNK" --out-dir "$D" --wait \
            > "$D/topup.log" 2>&1 &
        PIDS+=("$!:$G2")
    else
        # shellcheck disable=SC2086
        "$UI" mc-label --corpus $IN --model "$NET" --nn-backend inproc --top-up "$TOPUP" --parallel "$STREAMS" \
            --out-dir "$D" 2> "$D/topup.log" || die "$G2 top-up — see $D/topup.log"
    fi
done
for pg in "${PIDS[@]}"; do wait "${pg%%:*}" || die "${pg#*:} top-up failed — see $OUT/mc16/${pg#*:}/topup.log"; done
for g in "${WIN[@]}"; do
    G2="$(gg "$g")"
    for k in $TRAIN_SHARDS $VAL_SHARDS; do [ -s "$OUT/mc16/$G2/shard$k.jsonl" ] || die "$G2 top-up: shard$k missing"; done
    [ -e "$OUT/mc16/$G2/.reported" ] && continue
    status "$G2 top-up done: $(grep -o '([0-9]* left unlabelled)' "$OUT/mc16/$G2/topup.log" | tr -dc '0-9\n' | awk '{s+=$1} END {print s+0}') trajectories not topped up (kept at 8)"
    touch "$OUT/mc16/$G2/.reported"
done
# The worker holds ~1 GB of GPU memory: down before training, up again for the drives.
hub_down

# ---- 2. the label noise, measured --------------------------------------------------------------------
if [ ! -s "$OUT/label_noise.txt" ]; then
    {
        for g in "${WIN[@]}"; do echo "## $(gg "$g")"; "$PY" scripts/label_noise.py "$OUT/mc16/$(gg "$g")"/topup/shard*.jsonl; done
        echo "## window"; "$PY" scripts/label_noise.py "$OUT"/mc16/gen*/topup/shard*.jsonl
    } > "$OUT/label_noise.txt" 2>&1
fi
status "label noise: $(grep '^LABEL_NOISE' "$OUT/label_noise.txt" | tail -1)"

# ---- 3. prepare + train both arms --------------------------------------------------------------------
prep() {   # prep OUTDIR files...
    local out="$1"; shift
    [ -e "$out/.done" ] && return 0
    rm -rf "$out"
    "$PREPARE" --in "$@" --out "$out" "${PREP_ARGS[@]}" >> "$OUT/prepare.log" 2>&1 || die "prepare $out — see prepare.log"
    touch "$out/.done"
}
wait_gpu() { until [ "$(gpu_free_mb)" -ge "$TRAIN_GPU_MB" ]; do sleep 60; done; }
# shellcheck disable=SC2046
prep "$OUT/prepared_val$SEL" $(shards "$SEL" val)
for arm in A B; do
    L=8; [ "$arm" = B ] && L=16; D="$OUT/$arm"; mkdir -p "$D"
    [ -s "$D/net.onnx" ] && [ -s "$D/net.ts" ] && continue
    # shellcheck disable=SC2046
    prep "$D/prepared_train" $(shards "$L" train)
    wait_gpu; SECONDS=0
    status "$arm: train on ${L}-playout labels from $(basename "$PARENT").pt (select on val$SEL), GPU free $(gpu_free_mb) MB"
    # shellcheck disable=SC2086
    "$PY" -m bbnn.train --data "$D/prepared_train" --val-data "$OUT/prepared_val$SEL" --epochs "$EPOCHS" \
        --device "$TRAIN_DEVICE" --init "$PARENT.pt" --lr 5e-5 --select-on combined --eval-every 1000 \
        --value-weight 0.36 --freeze-bn --eval-at 250,500 --per-drive-value-weight --seed "$SEED" $TRAIN_EXTRA \
        --out "$D/net.pt" --onnx "$D/net.onnx" --progress "$D/train.progress.json" > "$D/train.log" 2>&1 \
        || die "$arm training — see $D/train.log"
    "$PY" -m bbnn.export_ts "$D/net.pt" --out "$D/net.ts" --canvas "$CANVAS" >> "$OUT/export.log" 2>&1 || die "export_ts $arm"
    rm -rf "$D/prepared_train"
    status "$arm trained ($((SECONDS / 60)) min): $(grep 'restored best-val weights' "$D/train.log" | tail -1)$(grep 'warm-start baseline' "$D/train.log" | tail -1 | sed 's/.*val_value/ (baseline val_value/;s/$/)/')"
done
# The loop's own gen P+1 net, when it exists: an unseeded replicate of arm A (selected on val8).
LOOPNET="$M/bbnet_${TIER}_$(gg $((P + 1)))"
NETS=("parent=$PARENT" "A=$OUT/A/net" "B=$OUT/B/net")
[ -f "$LOOPNET.pt" ] && [ -z "${SMOKE_LINES:-}" ] && NETS+=("loop=$LOOPNET")

# ---- 4. absorption: gen P+1's raw held-out shards (the loop's probe), the window val at 8 and 16 ------
# shellcheck disable=SC2046
prep "$OUT/prepared_val$((24 - SEL))" $(shards $((24 - SEL)) val)
RAW=""; for k in $VAL_SHARDS; do RAW="$RAW $NEXT/shard$k.jsonl"; done
# shellcheck disable=SC2086
prep "$OUT/probe_raw" $RAW
PROBE=(); for n in "${NETS[@]}"; do PROBE+=("${n%%=*}=${n#*=}.pt"); done
for v in raw val8 val16; do
    dir="$OUT/prepared_$v"; [ "$v" = raw ] && dir="$OUT/probe_raw"
    [ -s "$OUT/absorb_$v.txt" ] || { wait_gpu; "$PY" scripts/absorb_probe.py --summary --device "${PROBE_DEVICE:-cuda}" --val "$dir" "${PROBE[@]}" > "$OUT/absorb_$v.txt" 2>&1; }
    status "absorption ($v labels): $(grep -E '^ABSORB' "$OUT/absorb_$v.txt" | sed 's/^ABSORB //' | paste -sd'|')"
done
rm -rf "$OUT/probe_raw"

# ---- 5. value benches, paired vs the parent ------------------------------------------------------------
# Each bench row names its corpus relative to the checkout it was frozen in: made absolute here (and cut
# to VB_ROWS rows when set), so the bench reads the same states from any checkout.
bench_copy() {
    "$PY" - "$1" "$2" "${VB_ROWS:-0}" <<'PY'
import json, os, sys
src, dst, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(src))))
with open(src) as f, open(dst, "w") as out:
    for i, line in enumerate(f):
        if rows and i >= rows:
            break
        r = json.loads(line)
        if not os.path.isabs(r["corpus"]):
            r["corpus"] = os.path.join(root, r["corpus"])
        out.write(json.dumps(r) + "\n")
PY
}
mkdir -p "$OUT/benches"
VB_NETS=()
for n in "${NETS[@]}"; do
    net="${n#*=}.onnx"; [ "${n%%=*}" = parent ] && net="$PARENT_ONNX"
    if [ "${n%%=*}" = loop ]; then net="$(with_ts "$net")" || { status "WARN: no .ts for the loop's net: left out"; continue; }; fi
    VB_NETS+=("${n%%=*}=$net")
done
for vb in "value_bench|$VALUE_BENCH" "value_bench_2|$VALUE_BENCH_2"; do
    name="${vb%%|*}"; file="${vb#*|}"; [ -f "$file" ] || continue
    [ -s "$OUT/benches/$name.jsonl" ] || bench_copy "$file" "$OUT/benches/$name.jsonl"
    [ -s "$OUT/$name/summary.txt" ] || { wait_gpu; scripts/value_bench.sh "$OUT/benches/$name.jsonl" "$OUT/$name" "${VB_NETS[@]}" > "$OUT/$name.log" 2>&1; } \
        || status "WARN: $name failed — see $OUT/$name.log"
    status "$name: $(grep -E '^VALUE_BENCH' "$OUT/$name/summary.txt" 2>/dev/null | sed 's/^VALUE_BENCH_*//' | paste -sd'|')"
done

# ---- 6. net check on gen P+1's held-out shards -------------------------------------------------------
if [ "$NET_CHECK" = on ]; then
    mkdir -p "$OUT/nc_corpus"; for k in $VAL_SHARDS; do ln -sf "$NEXT/shard$k.jsonl" "$OUT/nc_corpus/shard$k.jsonl"; done
    for arm in A B; do
        [ -s "$OUT/$arm/net_check/net_check.txt" ] || { wait_gpu; scripts/net_check.sh "$OUT/$arm/net.onnx" "$OUT/nc_corpus" \
            "$OUT/$arm/net_check" "$C1" "$NET_CHECK_LADDER" > "$OUT/$arm/net_check.log" 2>&1; } \
            || status "WARN: $arm net check failed — see $OUT/$arm/net_check.log"
        status "$arm net check: $(cat "$OUT/$arm/net_check/net_check.txt" 2>/dev/null)"
    done
fi

# ---- 7. drives B vs A (DRIVES=1) ------------------------------------------------------------------------
if [ "$DRIVES" = 1 ]; then
    dir="$OUT/drives_B_vs_A"
    if [ ! -s "$dir/report.json" ]; then
        hub_up; mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"; SECONDS=0
        "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp078 B (16-playout labels) vs A (8)" \
            --evaluator nn --model "$OUT/B/net.onnx" --vs-evaluator nn --vs-model "$OUT/A/net.onnx" \
            --mcts-iters 1000 --games 30 --bot-config "$C1" --vs-config "$C1" --seed 0 --skip-fixed-rungs \
            --positions "$DRIVE_POSITIONS" --sprt "$DRIVE_SPRT" --vs-games "$DRIVE_CAP" \
            --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
            || die "drives — see $dir/eval.log"
        status "drives done ($((SECONDS / 60)) min)"
    fi
    status "drives B vs A: $("$PY" - "$dir/report.json" <<'PY'
import json, re, sys
r = json.load(open(sys.argv[1]))
out = []
for row in r["ladder"]:
    m = re.search(r"drives\(([^)]*)\)", row["opponent"]); s = row.get("sprt") or {}
    out.append(f"{m.group(1) if m else row.get('board')}: {row['points']:.3f} ± {row.get('points_se', 0):.3f} "
               f"{s.get('verdict', '-')} after {s.get('pairs', '?')} pairs")
print(" | ".join(out))
PY
)"
    hub_down
fi
status "done"
