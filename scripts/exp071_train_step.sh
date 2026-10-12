#!/usr/bin/env bash
# exp071: the train step (plan 058 §"the fine-tune barely moves its own val loss"; registry
# "Gradient balance"). One generation step, gen26 -> gen27, replayed with a different train step: the
# window the loop's gen27 trained on (gen25-27, all C1 corpora, MC-labelled), the loop's recipe, one
# change per arm. The loop's own gen27 is the control (cq tau 100, value weight 0.25, lr 5e-5).
#
# Arms (plan in registry/EXPERIMENTS.md#exp071):
#   tau50_vw36   cq tau 50 with the value weight rebalanced to keep tau 100's policy:value trunk-gradient
#                ratio (0.25 x 0.646/0.455 = 0.36, the gradient probe of 2026-10-09)
#   tau100_vw36  the rebalance alone: is 0.25 too low now that labels are MC averages?
#   lr1e4        lr 1e-4 (2x), everything else the loop's: does the step move the net at all?
# Per arm: absorption on gen27's held-out shards (one cq-100 target for all), the value bench paired
# vs the parent, and the trainer's own val movement (first check -> restored, and the final step).
#
# Waits for the loop to exit on its STOP file (set it after gen27's training starts), runs the arms
# on the free GPU, then (RELAUNCH=1, the default) relaunches the loop.
#   scripts/exp071_train_step.sh     # out: runs/exp071/, status in runs/exp071/status.md
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="$REPO/runs/exp071"; mkdir -p "$OUT"
RUN="$REPO/runs/loopmix16x9v9"; M="$REPO/models/az_v7"
PARENT="$M/bbnet_mix16x9v9_gen26"; LOOP27="$M/bbnet_mix16x9v9_gen27"
RELAUNCH="${RELAUNCH:-1}"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
PREPARE="$CARGO_TARGET_DIR/release/prepare"; PY="$REPO/train/.venv/bin/python"
status() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/status.md"; }

# name|prepare target args|train args
ARMS=(
    "tau50_vw36|--policy-target cq --tau 50|--lr 5e-5 --value-weight 0.36"
    "tau100_vw36|--policy-target cq --tau 100|--lr 5e-5 --value-weight 0.36"
    "lr1e4|--policy-target cq --tau 100|--lr 1e-4 --value-weight 0.25"
)

# The loop's gen27 must be trained (its window is ours), and the loop must have stopped (the GPU).
until grep -q "gen27 train done" "$RUN/status.md"; do sleep 60; done
until grep -q "STOP file present" <(tail -3 "$RUN/status.md") && ! ps -eo args | grep -q "^bash scripts/train_loop.s[h]"; do sleep 60; done
status "start: commit $(git rev-parse --short HEAD); parent $(basename "$PARENT"), window gen25-27 (C1, MC labels)"

shards() {   # $1 = train|val: the loop's window for gen27 (MC-labelled where present)
    local g k f out=""
    for g in 25 26 27; do for k in 0 1 2 3 4 5 6 7; do
        case " 4 7 " in *" $k "*) [ "$1" = val ] || continue ;; *) [ "$1" = train ] || continue ;; esac
        f="$RUN/gen$g/mc/shard$k.jsonl"; [ -s "$f" ] || f="$RUN/gen$g/shard$k.jsonl"
        out="$out $f"
    done; done
    echo "$out"
}
TRAIN_IN=$(shards train); VAL_IN=$(shards val)

for spec in "${ARMS[@]}"; do
    IFS='|' read -r name targs trargs <<< "$spec"; A="$OUT/$name"
    [ -f "$A/net.onnx" ] && { status "$name: trained already"; continue; }
    mkdir -p "$A"; rm -rf "$A/prepared_train" "$A/prepared_val"
    # shellcheck disable=SC2086
    "$PREPARE" --in $TRAIN_IN --out "$A/prepared_train" $targs --value-blend 1.0 > "$A/prepare.log" 2>&1 \
        && "$PREPARE" --in $VAL_IN --out "$A/prepared_val" $targs --value-blend 1.0 >> "$A/prepare.log" 2>&1 \
        || { status "$name: prepare FAILED — see $A/prepare.log"; continue; }
    status "$name: training ($targs; $trargs)"
    SECONDS=0
    # shellcheck disable=SC2086
    if "$PY" -m bbnn.train --data "$A/prepared_train" --val-data "$A/prepared_val" --epochs 3 --device cuda \
            --init "$PARENT.pt" $trargs --select-on combined --eval-every 1000 \
            --per-drive-value-weight --freeze-bn --eval-at 250,500 \
            --out "$A/net.pt" --onnx "$A/net.onnx" > "$A/train.log" 2>&1; then
        status "$name: trained ($((SECONDS / 60)) min): $(grep 'restored best-val weights' "$A/train.log" | tail -1)"
    else
        status "$name: training FAILED — see $A/train.log"
    fi
    rm -rf "$A/prepared_train" "$A/prepared_val"
done

NETS=("loop27=$LOOP27")
for spec in "${ARMS[@]}"; do name="${spec%%|*}"; [ -f "$OUT/$name/net.onnx" ] && NETS+=("$name=$OUT/$name/net"); done

# The trainer's own val movement: first check (step 250), the restored step, the final step.
for n in "${NETS[@]}"; do
    name="${n%%=*}"; log="$OUT/$name/train.log"; [ "$name" = loop27 ] && log="$RUN/gen27/train.log"
    status "val $name: step250 [$(grep -E '^step +250 ' "$log" | grep -oE 'val_policy [0-9.]+ +val_value [0-9.]+')] final [$(grep '(final)' "$log" | grep -oE 'val_policy [0-9.]+ +val_value [0-9.]+')] | $(grep 'restored best' "$log" | cut -c1-80)"
done

# Absorption probe: gen27's held-out shards, one cq-100 target for every arm.
rm -rf "$OUT/probe_val"
"$PREPARE" --in "$RUN/gen27/mc/shard4.jsonl" "$RUN/gen27/mc/shard7.jsonl" --out "$OUT/probe_val" \
    --policy-target cq --tau 100 --value-blend 1.0 > "$OUT/probe_prepare.log" 2>&1
PROBE_ARGS=("parent=$PARENT.pt"); for n in "${NETS[@]}"; do PROBE_ARGS+=("${n%%=*}=${n#*=}.pt"); done
"$PY" scripts/absorb_probe.py --summary --val "$OUT/probe_val" "${PROBE_ARGS[@]}" > "$OUT/absorb.txt" 2>&1
status "absorption probe (vs parent gen26, cq-100 target):"; grep -E '^ABSORB|^net|parent|loop27|tau|lr1e4' "$OUT/absorb.txt" | tee -a "$OUT/status.md"
rm -rf "$OUT/probe_val"

# Value bench, paired against the parent.
VB_ARGS=("parent=$PARENT.onnx"); for n in "${NETS[@]}"; do VB_ARGS+=("${n%%=*}=${n#*=}.onnx"); done
scripts/value_bench.sh "$RUN/../value_bench/v9_gen01_val.jsonl" "$OUT/value_bench" "${VB_ARGS[@]}" > "$OUT/value_bench.log" 2>&1
status "value bench:"; grep -E '^VALUE_BENCH' "$OUT/value_bench/summary.txt" | tee -a "$OUT/status.md"

if [ "$RELAUNCH" = 1 ]; then
    rm -f "$RUN/STOP"
    EXTRA_ALLOW=ae721b7 setsid nohup scripts/launch_plan058.sh > /dev/null 2>&1 < /dev/null &
    status "loop relaunched"
fi
status "done"
