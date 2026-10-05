#!/usr/bin/env bash
# exp066: plan 055 §5-6 — does more search help under the mean backup? Waits for exp065, then:
#   1. scripts/net_check.sh on g_gen05 (the cheap audit curve, ladder 64/250/1000/4000).
#   2. The ground truth: drive matches of g_gen05's Gumbel search (cfgs/gumbel16_f1000.toml, now
#      mean backup) at 250 / 1000 / 4000 descents against policy-only on the same net, paired, the
#      gen04g contested sets, fixed 300 pairs per board, --seed 66000. Points are the search's.
#      250 and 1000 run together on 12 local streams; 4000 then runs alone on 6 (tree memory).
# The two curves are compared in plan 055 §6: if they rank the budgets alike, the cheap one is the
# standing check.
#
#   scripts/exp066_budget_curve.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="$REPO/runs/exp066"; mkdir -p "$OUT"
M="$REPO/models/az_v7"; NET="$M/bbnet_mix16x9g_gen05.onnx"; POS="$REPO/runs/loopmix16x9g/positions"
SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
GUMBEL="$REPO/cfgs/gumbel16_f1000.toml"; POLICY="$REPO/cfgs/policy_only.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp066.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
down() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; WORKER_PID="" NN_PID="" HUB_PID=""; rm -f "$SOCK"; }
trap down EXIT INT TERM
git diff --quiet || die "dirty tree"
status "start: commit $(git rev-parse --short HEAD); waiting for exp065"
until grep -q '] done' "$REPO/runs/exp065/status.md" 2>/dev/null; do sleep 30; done
status "exp065 done"

# ---- 1. the cheap curve ---------------------------------------------------------------------------
if [ ! -s "$OUT/netcheck/net_check.txt" ]; then
    scripts/net_check.sh "$NET" "$REPO/runs/loopmix16x9g/gen06" "$OUT/netcheck" "$GUMBEL" "64 250 1000 4000" \
        > "$OUT/netcheck.log" 2>&1 || status "WARN: net_check failed — see $OUT/netcheck.log"
fi
status "net check: $(cat "$OUT/netcheck/net_check.txt" 2>/dev/null)"

# ---- 2. the drive curve ---------------------------------------------------------------------------
cargo build --release -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub already serves $HUB_URL"
LAST=$(git log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$(git rev-list --abbrev-commit "$LAST"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > "$REPO/hub-allowed-commits.toml"
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$NET" --max-models 2 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
worker_up() {
    "$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games "$1" --mem-floor-mb 1536 \
        --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
}
match() {
    local iters="$1" dir="$OUT/G$1"
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp066 G$iters: gumbel16_f1000 (mean)@$iters vs policy-only" \
        --evaluator nn --model "$NET" --bot-config "$GUMBEL" --mcts-iters "$iters" \
        --vs-evaluator nn --vs-model "$NET" --vs-config "$POLICY" --opponent-iters 8 \
        --games 30 --seed 66000 --skip-fixed-rungs --positions "$SETS" --vs-games 600 \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
        || { status "WARN: G$iters failed — see $dir/eval.log"; return 1; }
    status "G$iters vs policy: $("$PY" - "$dir/report.json" <<'PY'
import json, math, re, sys
rows = json.load(open(sys.argv[1]))["ladder"]
parts = [f"{re.search(r'@(\S+)$', r['opponent']).group(1)} {r['points']:.3f} ± {r['points_se']:.3f}" for r in rows]
mean = sum(r["points"] for r in rows) / len(rows)
se = math.sqrt(sum(r["points_se"] ** 2 for r in rows)) / len(rows)
print(" | ".join(parts) + f" | mean {mean:.3f} ± {se:.3f}")
PY
)"
}
SECONDS=0
worker_up 12
match 250 & A=$!
match 1000 & B=$!
wait $A $B
kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null
worker_up 6
match 4000
status "done ($((SECONDS / 60)) min of drives)"
