#!/usr/bin/env bash
# exp073: is a 96x8 net better *in the loop's regime*? exp072's from-scratch 96x8 nets fit gen10-31's
# average target and lost the test set to the loop's fine-tuned 64x6 gen31 (KL 0.104 vs 0.055) yet
# played it even. Here each is fine-tuned on exactly the window and recipe the loop's gen33 trained on
# (gen31-33, MC labels, cq tau 100, lr 5e-5, 3 epochs, --freeze-bn, value weight 0.36 per-drive), so the
# only difference from gen33 is the starting net (exp072's 96x8 vs the 64x6 lineage).
# Per arm: absorption + test numbers on gen33's held-out shards vs gen33, the value bench paired vs
# gen33, and drives vs gen33 under C1 (1000 descents, both sets, 300 FIXED pairs per board, no SPRT:
# exp072's SPRT stopped 14x7 after 16-38 pairs).
# Waits for the loop to be stopped (set STOP / stop_after.sh), trains on the free GPU, relaunches the
# loop (RELAUNCH=1, default), then plays the drives on its hub.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="$REPO/runs/exp073"; mkdir -p "$OUT"
RUN="$REPO/runs/loopmix16x9v9"; M="$REPO/models/az_v7"; LOOP33="$M/bbnet_mix16x9v9_gen33"
RELAUNCH="${RELAUNCH:-1}"; PAIRS="${PAIRS:-300}"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
PREPARE="$CARGO_TARGET_DIR/release/prepare"; HUB="$CARGO_TARGET_DIR/release/botbowl-hub"
PY="$REPO/train/.venv/bin/python"; TOK="$HOME/.config/botbowl/hub.token"
C1="$REPO/cfgs/chance_all_partial.toml"; POS="$REPO/runs/loopmix16x9g/positions"
status() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/status.md"; }
loop_running() { ps -eo args | grep -q "^bash scripts/train_loop.s[h]"; }
ARMS=(A96x8 B96x8h)

until grep -q "gen33 train done" "$RUN/status.md"; do sleep 60; done
status "start: commit $(git rev-parse --short HEAD); arms ${ARMS[*]} from runs/exp072, window gen31-33, vs gen33"
shards() { local g k f out=""; for g in 31 32 33; do for k in 0 1 2 3 4 5 6 7; do
    case " 4 7 " in *" $k "*) [ "$1" = val ] || continue ;; *) [ "$1" = train ] || continue ;; esac
    f="$RUN/gen$g/mc/shard$k.jsonl"; [ -s "$f" ] || f="$RUN/gen$g/shard$k.jsonl"; out="$out $f"; done; done; echo "$out"; }
if [ ! -e "$OUT/.prepared" ]; then
    rm -rf "$OUT/prepared_train" "$OUT/prepared_val"
    # shellcheck disable=SC2046
    "$PREPARE" --in $(shards train) --out "$OUT/prepared_train" --policy-target cq --tau 100 --value-blend 1.0 > "$OUT/prepare.log" 2>&1 \
      && "$PREPARE" --in $(shards val) --out "$OUT/prepared_val" --policy-target cq --tau 100 --value-blend 1.0 >> "$OUT/prepare.log" 2>&1 \
      && touch "$OUT/.prepared" || { status "prepare FAILED"; exit 1; }
fi
need_gpu=0; for a in "${ARMS[@]}"; do [ -f "$OUT/$a/net.onnx" ] || need_gpu=1; done
if [ "$need_gpu" = 1 ]; then
    until ! loop_running && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 60; done
    status "GPU free"
fi
for a in "${ARMS[@]}"; do
    A="$OUT/$a"; [ -f "$A/net.onnx" ] && continue; mkdir -p "$A"; SECONDS=0
    status "$a: fine-tune from runs/exp072/$a/net.pt (the loop's recipe)"
    "$PY" -m bbnn.train --data "$OUT/prepared_train" --val-data "$OUT/prepared_val" --epochs 3 --device cuda \
        --init "$REPO/runs/exp072/$a/net.pt" --lr 5e-5 --select-on combined --eval-every 1000 \
        --value-weight 0.36 --per-drive-value-weight --freeze-bn --eval-at 250,500 \
        --out "$A/net.pt" --onnx "$A/net.onnx" > "$A/train.log" 2>&1 \
        && status "$a: trained ($((SECONDS / 60)) min): $(grep 'restored best-val weights' "$A/train.log" | tail -1)" \
        || status "$a: training FAILED — see $A/train.log"
done
rm -rf "$OUT/prepared_train" "$OUT/prepared_val"
if [ ! -s "$OUT/absorb.txt" ]; then
    "$PREPARE" --in "$RUN/gen33/mc/shard4.jsonl" "$RUN/gen33/mc/shard7.jsonl" --out "$OUT/probe_val" \
        --policy-target cq --tau 100 --value-blend 1.0 > "$OUT/probe_prepare.log" 2>&1
    args=("gen33=$LOOP33.pt"); for a in "${ARMS[@]}"; do args+=("$a=$OUT/$a/net.pt"); done
    "$PY" scripts/absorb_probe.py --summary --val "$OUT/probe_val" "${args[@]}" > "$OUT/absorb.txt" 2>&1
    rm -rf "$OUT/probe_val"
fi
status "test set (gen33 shards 4,7) vs gen33:"; grep -E '^net|^gen33|^A96|^B96|^ABSORB' "$OUT/absorb.txt" | tee -a "$OUT/status.md"
if [ ! -s "$OUT/value_bench/summary.txt" ]; then
    args=("gen33=$LOOP33.onnx"); for a in "${ARMS[@]}"; do args+=("$a=$OUT/$a/net.onnx"); done
    scripts/value_bench.sh "$REPO/runs/value_bench/v9_gen01_val.jsonl" "$OUT/value_bench" "${args[@]}" > "$OUT/value_bench.log" 2>&1
fi
status "value bench (paired vs gen33):"; grep -E '^VALUE_BENCH' "$OUT/value_bench/summary.txt" | tee -a "$OUT/status.md"
for a in "${ARMS[@]}"; do cp "$OUT/$a/net.pt" "$M/exp073_$a.pt"; cp "$OUT/$a/net.onnx" "$M/exp073_$a.onnx"; done
if [ "$RELAUNCH" = 1 ] && ! loop_running; then
    rm -f "$RUN/STOP"; EXTRA_ALLOW=ae721b7 setsid nohup scripts/launch_plan058.sh > /dev/null 2>&1 < /dev/null &
    status "loop relaunched"; sleep 180
fi
for a in "${ARMS[@]}"; do for b in 14x7 16x9; do
    dir="$OUT/$a/drives_$b"; [ -s "$dir/report.json" ] && continue; mkdir -p "$dir"; SECONDS=0
    "$HUB" job eval --hub http://127.0.0.1:13337 --token-file "$TOK" --label "exp073 $a (fine-tuned) vs gen33 @$b, $PAIRS fixed pairs" \
        --evaluator nn --model "$M/exp073_$a.onnx" --vs-evaluator nn --vs-model "$LOOP33.onnx" \
        --mcts-iters 1000 --bot-config "$C1" --vs-config "$C1" --seed 0 --skip-fixed-rungs \
        --positions "$POS/contested_${b}_gen04g.json" --vs-games $((2 * PAIRS)) \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
        || { status "drives $a @$b FAILED — see $dir/eval.log"; continue; }
    status "drives $a vs gen33 @$b ($((SECONDS / 60)) min): $(grep -oE 'pts [0-9.]+ ± [0-9.]+ \([0-9]+ pairs\)' "$dir/eval.log" | tail -1)"
done; done
status "done"
