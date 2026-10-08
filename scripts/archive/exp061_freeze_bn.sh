#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# exp061: review finding 2 — BatchNorm normalises single-board training batches by their own
# statistics but inference uses one global running average. Does training under frozen BN (the
# warm start's running statistics, `bbnn.train --freeze-bn`) give a better net?
#
# gen04's recipe exactly (its own prepared gen02-04 window, warm from gen03, 3 epochs, lr 2e-4,
# the loop's value weighting), seed 1, two arms: unfrozen (a same-seed control) and frozen.
#   1. Train both now (GPU), compare per-board val (printed by the trainer, eval mode).
#   2. After exp060d has finished (one drive experiment at a time), drives: frozen vs control,
#      both under the search we will use (cfgs/gumbel16_f1000.toml, 1000 descents), on the
#      gen04-screened contested sets, SPRT 0.5:0.55, at most 800 per board.
#
#   scripts/exp061_freeze_bn.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp061"; mkdir -p "$OUT"
RUN="$REPO/runs/loopmix16x9d1k"; M="$REPO/models/az_v7"; GEN03="$M/bbnet_mix16x9d1k_gen03"
TRAIN_DIR="$RUN/gen04/prepared_train"; VAL_DIR="$RUN/gen04/prepared_val"
GUMBEL="$REPO/cfgs/gumbel16_f1000.toml"; POS="$REPO/runs/exp059/positions"
SPRT=0.5:0.55; CAP=800
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp061.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
ls -d "$TRAIN_DIR"/dims_* "$VAL_DIR"/dims_* > /dev/null 2>&1 || die "gen04's prepared window is gone"

# ---- 1. train ------------------------------------------------------------------------------------
SECONDS=0
for arm in control frozen; do
    out="$M/exp061_$arm"
    [ -s "$out.onnx" ] && continue
    extra=""; [ "$arm" = frozen ] && extra="--freeze-bn"
    # shellcheck disable=SC2086
    (cd "$REPO/train" && "$PY" -m bbnn.train --data "$TRAIN_DIR" --val-data "$VAL_DIR" --epochs 3 --device cuda \
        --init "$GEN03.pt" --lr 2e-4 --select-on combined --eval-every 2500 --value-weight 0.25 \
        --per-drive-value-weight --seed 1 $extra --out "$out.pt" --onnx "$out.onnx" > "$OUT/train_$arm.log" 2>&1) \
        || die "train $arm"
    status "trained $arm ($((SECONDS / 60)) min): $(grep -E '^restored' "$OUT/train_$arm.log")"
done
# Per-board val at each arm's restored checkpoint: the starred step's per-board lines.
"$PY" - "$OUT/train_control.log" "$OUT/train_frozen.log" > "$OUT/val_by_board.txt" <<'PY'
import re, sys
def best(path):
    rows, cur = {}, {}
    for line in open(path):
        m = re.match(r"\s+val@(dims_\S+): val_policy (\S+)\s+val_value (\S+)", line)
        if m:
            cur[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        elif line.startswith("step") or line.startswith("epoch"):
            if line.rstrip().endswith("*"):
                rows = dict(cur)
            cur = {}
    return rows
c, f = best(sys.argv[1]), best(sys.argv[2])
print(f"{'board':10} {'value ctrl':>10} {'value frozen':>12} {'policy ctrl':>11} {'policy frozen':>13}")
for k in sorted(c):
    print(f"{k:10} {c[k][1]:10.4f} {f.get(k, (0, 0))[1]:12.4f} {c[k][0]:11.4f} {f.get(k, (0, 0))[0]:13.4f}")
PY
status "per-board val at the restored checkpoints in $OUT/val_by_board.txt"

# ---- 2. drives, after exp060d ----------------------------------------------------------------------
until tail -1 "$REPO/runs/exp060/drives_d/status.md" 2>/dev/null | grep -qE "\] (done|FATAL)"; do sleep 120; done
for _ in $(seq 60); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 || break; sleep 10; done
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub still serves $HUB_URL"
status "exp060d finished; drives"
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$M/exp061_control.onnx" --max-models 4 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 8 --mem-floor-mb 1536 \
    --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
dir="$OUT/frozen_vs_control"
if [ ! -s "$dir/report.json" ]; then
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "BN frozen vs control (Gumbel, drives)" \
        --evaluator nn --model "$M/exp061_frozen.onnx" --bot-config "$GUMBEL" --mcts-iters 1000 \
        --vs-evaluator nn --vs-model "$M/exp061_control.onnx" --vs-config "$GUMBEL" --opponent-iters 1000 \
        --games 30 --seed 61000 --skip-fixed-rungs \
        --positions "$POS/contested_14x7_gen04.json,$POS/contested_16x9_gen04.json" --sprt "$SPRT" --vs-games "$CAP" \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 || die "drives"
fi
status "frozen vs control: $("$PY" - "$dir/report.json" <<'PY'
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
status "done"
