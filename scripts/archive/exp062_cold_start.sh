#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# exp062: the user's cold-start question for the Gumbel loop. gen01 of runs/loopmix16x9g trained
# warm (from d1k gen04) on its own 2400 Gumbel drives; this trains the same data from random
# weights, then plays it against gen04 on the loop's drive benchmark (Gumbel both sides, 1000
# descents, the gen04g contested sets, SPRT 0.5:0.55), submitted to the running loop's hub so it
# shares its workers. Compare with gen01's own result (0.490 / 0.482 vs gen04).
#
#   scripts/exp062_cold_start.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp062"; mkdir -p "$OUT"
G1="$REPO/runs/loopmix16x9g/gen01"; M="$REPO/models/az_v7"; COLD="$M/exp062_gen01data_cold"
POS="$REPO/runs/loopmix16x9g/positions"; GUMBEL="$REPO/cfgs/gumbel16_f1000.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; PREPARE="$CARGO_TARGET_DIR/release/prepare"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
if [ ! -e "$OUT/.prepared" ]; then
    T=""; V=""; for k in 0 1 2 3 5 6; do T="$T $G1/shard$k.jsonl"; done; for k in 4 7; do V="$V $G1/shard$k.jsonl"; done
    # shellcheck disable=SC2086
    "$PREPARE" --in $T --out "$OUT/prepared_train" --policy-target cq --tau 100 --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 || die "prepare train"
    # shellcheck disable=SC2086
    "$PREPARE" --in $V --out "$OUT/prepared_val" --policy-target cq --tau 100 --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 || die "prepare val"
    touch "$OUT/.prepared"
fi
if [ ! -s "$COLD.onnx" ]; then
    SECONDS=0
    # From random weights: the loop's scratch rate and more passes than a warm fine-tune's 3.
    (cd "$REPO/train" && "$PY" -m bbnn.train --data "$OUT/prepared_train" --val-data "$OUT/prepared_val" \
        --epochs 15 --device cuda --lr 1e-3 --select-on combined --eval-every 2500 --value-weight 0.25 \
        --per-drive-value-weight --seed 1 --out "$COLD.pt" --onnx "$COLD.onnx" > "$OUT/train.log" 2>&1) || die "train"
    status "trained ($((SECONDS / 60)) min): $(grep -E '^restored' "$OUT/train.log")"
fi
dir="$OUT/cold_vs_gen04"; mkdir -p "$dir"
[ -s "$dir/report.json" ] || "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp062 cold-start gen01-data vs gen04 (drives)" \
    --evaluator nn --model "$COLD.onnx" --bot-config "$GUMBEL" --mcts-iters 1000 \
    --vs-evaluator nn --vs-model "$M/bbnet_mix16x9d1k_gen04.onnx" --vs-config "$GUMBEL" --opponent-iters 1000 \
    --games 30 --seed 0 --skip-fixed-rungs \
    --positions "$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json" --sprt 0.5:0.55 --vs-games 800 \
    --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 || die "drives"
status "cold vs gen04: $("$PY" "$REPO/scripts/eval_summary.py" "$dir/report.json" | grep -o 'drives([^)]*)@[^ ]* pts [0-9.]*\|\[paired SE [^]]*\]\|\[SPRT [^]]*\]' | paste -sd' ')"
status "done"
