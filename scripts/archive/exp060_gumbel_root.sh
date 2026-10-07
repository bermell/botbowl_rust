#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# exp060: plan 053 step 2 — does Gumbel root search (sequential halving over the top 16 root moves,
# cfgs/gumbel16_iters.toml) buy back the 16x9 search budget? Drives only, plus a no-games probe.
#
#   A. Convergence (no games): gen21 with Gumbel at 250/500/1000/2000 descents on exp057's states
#      and seeds, scored against exp057's 16000-descent PUCT references next to exp057's own PUCT
#      rows (scripts/convergence_topk.py takes both files).
#   B1. gen04 with Gumbel at 1000 descents vs gen04 with PUCT at 1000 descents (exact_iters), on the
#       gen04-screened contested sets from exp059, SPRT 0.5:0.55, at most 800 drives per board.
#   B2. 16x9 only: gen04 with Gumbel at 1000 vs gen04 with PUCT at 4000. Level or better means the
#       16x9 budget problem is solved at a quarter of the cost. Fewer streams: 4000-descent trees.
#
# Protocol v11: a remote worker must be rebuilt at this commit (or an allowlisted one) to help.
#
#   scripts/exp060_gumbel_root.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp060"; mkdir -p "$OUT"
M="$REPO/models/az_v7"; GEN21="$M/bbnet_mix16x9_gen21.onnx"; GEN04="$M/bbnet_mix16x9d1k_gen04.onnx"
GUMBEL="$REPO/cfgs/gumbel16_iters.toml"; ITERS="$REPO/cfgs/exact_iters.toml"
POS="$REPO/runs/exp059/positions"
SETS="$POS/contested_14x7_gen04.json,$POS/contested_16x9_gen04.json"
SPRT=0.5:0.55; CAP=800
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET   # every arm names its budget mode in its preset
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp060.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub already serves $HUB_URL"
cargo build --release -p botbowl-ui -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"

nn_up() {
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$1" --max-models 4 \
        --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
    for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
}
nn_down() { kill "$NN_PID" 2>/dev/null; wait "$NN_PID" 2>/dev/null; NN_PID=""; rm -f "$SOCK"; }

# ---- A. convergence on exp057's states ---------------------------------------------------------
SECONDS=0
nn_up "$GEN21"
for board in 14x7/4 16x9/6; do
    for adv in 0 1; do
        tag="${board%%/*}_a$adv"
        [ -s "$OUT/conv_gumbel_$tag.jsonl" ] && grep -q "wrote" "$OUT/conv_gumbel_$tag.log" 2>/dev/null && continue
        "$UI" convergence --states 60 --repeats 3 --budgets 250,500,1000,2000 --board "$board" --advance "$adv" \
            --evaluator nn --model "$GEN21" --nn-server "$SOCK" --parallel 8 --bot-config "$GUMBEL" \
            --seed $(( 91000000 + adv * 100000 )) --out "$OUT/conv_gumbel_$tag.jsonl" > "$OUT/conv_gumbel_$tag.log" 2>&1 \
            || die "convergence $tag"
        status "convergence $tag done ($((SECONDS / 60)) min)"
    done
done
nn_down
for b in 14x7 16x9; do
    "$PY" "$REPO/scripts/convergence_topk.py" "$REPO"/runs/exp057/conv_${b}_a*.jsonl "$OUT"/conv_gumbel_${b}_a*.jsonl \
        > "$OUT/summary_$b.txt" 2>&1 || die "summary $b"
done
status "convergence summaries in $OUT/summary_{14x7,16x9}.txt"

# ---- B. drives -------------------------------------------------------------------------------
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
nn_up "$GEN04"
worker_up() {
    "$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games "$1" --mem-floor-mb 1536 \
        --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
}
worker_down() { kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null; WORKER_PID=""; }
# drives NAME LABEL SETS OPP_ITERS — gen04 Gumbel@1000 vs gen04 PUCT@OPP_ITERS, waited for.
drives() {
    local name="$1" label="$2" sets="$3" opp="$4" dir="$OUT/$1"
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "$label" --evaluator nn --model "$GEN04" \
        --bot-config "$GUMBEL" --mcts-iters 1000 --vs-config "$ITERS" --opponent-iters "$opp" \
        --games 30 --seed 60000 --skip-fixed-rungs --vs-evaluator nn --vs-model "$GEN04" \
        --positions "$sets" --sprt "$SPRT" --vs-games "$CAP" \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 || die "$name"
    status "$name: $("$PY" - "$dir/report.json" <<'PY'
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
}
SECONDS=0
worker_up 12
drives b1_gumbel1000_vs_puct1000 "gen04 Gumbel@1000 vs PUCT@1000 (drives)" "$SETS" 1000
worker_down
worker_up 4
drives b2_gumbel1000_vs_puct4000_16x9 "gen04 Gumbel@1000 vs PUCT@4000, 16x9 (drives)" "$POS/contested_16x9_gen04.json" 4000
status "done ($((SECONDS / 60)) min of drives)"
