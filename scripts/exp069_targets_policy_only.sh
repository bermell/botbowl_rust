#!/usr/bin/env bash
# Plan 059 (exp069): policy targets re-tested under the mean backup, judged by the quick numbers only.
# One generation step, gen06 -> gen07, replayed with other targets: the same window the loop's gen07
# trained on (gen05-07, MC-labelled), the same recipe (lr 5e-5, --freeze-bn, 3 epochs, restore on
# combined val), only the policy target changes. The loop's own gen07 is the cq tau 100 arm; a
# tau 100 re-run measures the training noise of every number below.
#
# Per arm: absorption probe on gen07's held-out shards (one cq-100 prepared set for all arms, so the
# columns compare; read P(played) / log P(played) / top-1, which do not depend on the target), the
# value bench, and policy-only paired drives vs the parent gen06 on both contested position sets
# (cfgs/policy_only.toml on both sides, tract on the CPU: one forward per decision).
#
#   scripts/exp069_targets_policy_only.sh            # out: runs/exp069/, status in runs/exp069/status.md
# Env: PAIRS (policy-only drives per board = 2*PAIRS, default 600), PO_THREADS (default 3).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="$REPO/runs/exp069"; mkdir -p "$OUT"
RUN="$REPO/runs/loopmix16x9v9"; M="$REPO/models/az_v7"
PARENT="$M/bbnet_mix16x9v9_gen06"; LOOP07="$M/bbnet_mix16x9v9_gen07"
PAIRS="${PAIRS:-600}"; PO_THREADS="${PO_THREADS:-3}"
POS="$REPO/runs/loopmix16x9g/positions"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PREPARE="$CARGO_TARGET_DIR/release/prepare"
PY="$REPO/train/.venv/bin/python"
status() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/status.md"; }

# name|prepare target args
ARMS=(
    "cq100r|--policy-target cq --tau 100"
    "cq50|--policy-target cq --tau 50"
    "cq30|--policy-target cq --tau 30"
    "gumbel|--policy-target gumbel --gumbel-c-visit 50 --gumbel-c-scale 0.1 --gumbel-min-range 50"
)

# The loop's gen07 must be trained first: its window is ours, and its train step must not share the GPU.
until grep -q "gen07 train done" "$RUN/status.md"; do sleep 60; done
status "start: commit $(git rev-parse --short HEAD); parent $(basename "$PARENT"), window gen05-07 (MC labels)"
[ -x "$PREPARE" ] || cargo build --release -p botbowl-nn --bin prepare >> "$OUT/build.log" 2>&1

shards() {   # $1 = train|val: the loop's window_shards for gen07 (MC-labelled where present)
    local g k f out=""
    for g in 05 06 07; do for k in 0 1 2 3 4 5 6 7; do
        case " 4 7 " in *" $k "*) [ "$1" = val ] || continue ;; *) [ "$1" = train ] || continue ;; esac
        f="$RUN/gen$g/mc/shard$k.jsonl"; [ -s "$f" ] || f="$RUN/gen$g/shard$k.jsonl"
        out="$out $f"
    done; done
    echo "$out"
}
TRAIN_IN=$(shards train); VAL_IN=$(shards val)

gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for spec in "${ARMS[@]}"; do
    name="${spec%%|*}"; targs="${spec#*|}"; A="$OUT/$name"
    [ -f "$A/net.onnx" ] && { status "$name: trained already"; continue; }
    mkdir -p "$A"; rm -rf "$A/prepared_train" "$A/prepared_val"
    # shellcheck disable=SC2086
    "$PREPARE" --in $TRAIN_IN --out "$A/prepared_train" $targs --value-blend 1.0 > "$A/prepare.log" 2>&1 \
        && "$PREPARE" --in $VAL_IN --out "$A/prepared_val" $targs --value-blend 1.0 >> "$A/prepare.log" 2>&1 \
        || { status "$name: prepare FAILED — see $A/prepare.log"; continue; }
    while [ "$(gpu_free_mb)" -lt 2500 ]; do sleep 60; done
    status "$name: training ($targs), GPU free $(gpu_free_mb) MB"
    SECONDS=0
    # shellcheck disable=SC2086
    if "$PY" -m bbnn.train --data "$A/prepared_train" --val-data "$A/prepared_val" --epochs 3 --device cuda \
            --init "$PARENT.pt" --lr 5e-5 --select-on combined --eval-every 1000 \
            --value-weight 0.25 --per-drive-value-weight --freeze-bn --eval-at 250,500 \
            --out "$A/net.pt" --onnx "$A/net.onnx" > "$A/train.log" 2>&1; then
        status "$name: trained ($((SECONDS / 60)) min): $(grep 'restored best-val weights' "$A/train.log" | tail -1)"
    else
        status "$name: training FAILED — see $A/train.log"
    fi
    rm -rf "$A/prepared_train" "$A/prepared_val"
done

NETS=("loop07=$LOOP07")
for spec in "${ARMS[@]}"; do name="${spec%%|*}"; [ -f "$OUT/$name/net.onnx" ] && NETS+=("$name=$OUT/$name/net"); done

# Absorption probe: gen07's held-out shards, one cq-100 target for every arm.
rm -rf "$OUT/probe_val"
"$PREPARE" --in "$RUN/gen07/shard4.jsonl" "$RUN/gen07/shard7.jsonl" --out "$OUT/probe_val" \
    --policy-target cq --tau 100 --value-blend 1.0 > "$OUT/probe_prepare.log" 2>&1
PROBE_ARGS=("parent=$PARENT.pt"); for n in "${NETS[@]}"; do PROBE_ARGS+=("${n%%=*}=${n#*=}.pt"); done
"$PY" scripts/absorb_probe.py --summary --val "$OUT/probe_val" "${PROBE_ARGS[@]}" > "$OUT/absorb.txt" 2>&1
status "absorption probe (vs parent gen06, cq-100 target):"; grep -E '^ABSORB|^net|parent|loop07|cq|gumbel' "$OUT/absorb.txt" | tee -a "$OUT/status.md"
rm -rf "$OUT/probe_val"

# Value bench, paired against the parent.
VB_ARGS=("parent=$PARENT.onnx"); for n in "${NETS[@]}"; do VB_ARGS+=("${n%%=*}=${n#*=}.onnx"); done
scripts/value_bench.sh "$RUN/../value_bench/v9_gen01_val.jsonl" "$OUT/value_bench" "${VB_ARGS[@]}" > "$OUT/value_bench.log" 2>&1
status "value bench:"; grep -E '^VALUE_BENCH' "$OUT/value_bench/summary.txt" | tee -a "$OUT/status.md"

# Policy-only paired drives vs the parent, both boards (tract on the CPU).
for n in "${NETS[@]}"; do
    name="${n%%=*}"; net="${n#*=}.onnx"
    for board in 14x7 16x9; do
        D="$OUT/po/$name.$board"; mkdir -p "$D"
        [ -s "$D/report.json" ] || "$UI" eval --model "$net" --evaluator nn --bot-config cfgs/policy_only.toml \
            --vs-config cfgs/policy_only.toml --vs-model "$PARENT.onnx" --vs-evaluator nn --mcts-iters 8 --seed 0 \
            --skip-fixed-rungs --positions "$POS/contested_${board}_gen04g.json" --vs-games $((2 * PAIRS)) \
            --parallel-games "$PO_THREADS" --out "$D/report.json" > "$D/eval.log" 2>&1
        status "policy-only $name vs parent @$board: $(grep -o 'pts [0-9.]* ± [0-9.]* ([0-9]* pairs)' "$D/eval.log" | tail -1)"
    done
done
status "done"
