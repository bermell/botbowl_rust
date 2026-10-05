#!/usr/bin/env bash
# exp064b: plan 055 — mean backup + q floor 4000 together, vs policy-only on the same net, for
# g_gen05 and d1k gen04. Submitted to exp064's running hub (its phase 2 runs one 4000-descent
# match on 6 local streams, which leaves capacity). Same format as exp064: fixed 300 pairs per
# board, --seed 64000, points are the search's.
#
#   scripts/exp064b_combined.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp064"; M="$REPO/models/az_v7"; POS="$REPO/runs/loopmix16x9g/positions"
SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
CFG="$REPO/cfgs/gumbel16_f4000_mean.toml"; POLICY="$REPO/cfgs/policy_only.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
run() {
    local name="$1" net="$2" dir="$OUT/$1"
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp064b $name: gumbel16_f4000_mean@1000 vs policy-only" \
        --evaluator nn --model "$net" --bot-config "$CFG" --mcts-iters 1000 \
        --vs-evaluator nn --vs-model "$net" --vs-config "$POLICY" --opponent-iters 8 \
        --games 30 --seed 64000 --skip-fixed-rungs --positions "$SETS" --vs-games 600 \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
        || { status "WARN: $name failed — see $dir/eval.log"; return 1; }
    status "$name (gumbel16_f4000_mean@1000, $(basename "$net" .onnx)) vs policy: $("$PY" - "$dir/report.json" <<'PY'
import json, math, re, sys
rows = json.load(open(sys.argv[1]))["ladder"]
parts = [f"{re.search(r'@(\S+)$', r['opponent']).group(1)} {r['points']:.3f} ± {r['points_se']:.3f}" for r in rows]
mean = sum(r["points"] for r in rows) / len(rows)
se = math.sqrt(sum(r["points_se"] ** 2 for r in rows)) / len(rows)
print(" | ".join(parts) + f" | mean {mean:.3f} ± {se:.3f}")
PY
)"
}
run MEANF4000 "$M/bbnet_mix16x9g_gen05.onnx" & A=$!
run MEANF4000_d1k "$M/bbnet_mix16x9d1k_gen04.onnx" & B=$!
wait $A $B
