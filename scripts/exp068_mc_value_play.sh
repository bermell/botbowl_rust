#!/usr/bin/env bash
# exp068: plan 056 — does arm F's value head (MC-averaged labels, benchmark RMS 0.236 vs the control's
# 0.280) make the search stronger in play? The benchmark cannot say on its own: F's labels and the
# benchmark's truth are both policy-only playouts under g_gen05. Waits for exp067's net check of D,
# then, alongside each other:
#   1. scripts/net_check.sh on F (gen06, the same seed as A and D: the same decision sample).
#   2. Drives, F's search vs A's search, both cfgs/gumbel16_f1000.toml @1000 (mean backup): the two
#      nets share g_gen05's policy (the absorb probe), so this is the value head's effect on play.
#   3. Drives, F's search @1000 vs F's own policy-only (plan 056 §4's confirmation), for
#      comparison with exp066's g_gen05 at 1000 (0.542).
# Paired, the gen04g contested sets, fixed 300 pairs per board, --seed 68000; points are the first
# net's. Own hub on :13338 with its own allowlist, so exp066's hub and worker are untouched.
#
#   scripts/exp068_mc_value_play.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="$REPO/runs/exp068"; mkdir -p "$OUT"
ARMS="$REPO/runs/exp067/arms"; F="$ARMS/F.onnx"; A="$ARMS/A.onnx"
POS="$REPO/runs/loopmix16x9g/positions"
SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
GUMBEL="$REPO/cfgs/gumbel16_f1000.toml"; POLICY="$REPO/cfgs/policy_only.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
PORT=13338; HUB_URL="http://127.0.0.1:$PORT"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp068.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
down() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; WORKER_PID="" NN_PID="" HUB_PID=""; rm -f "$SOCK"; }
trap down EXIT INT TERM
git diff --quiet || die "dirty tree"
status "start: commit $(git rev-parse --short HEAD); waiting for exp067's net check of D"
until grep -q 'net check D:' "$REPO/runs/exp067/status.md" 2>/dev/null; do sleep 30; done

# ---- 1. the cheap curve, in the background --------------------------------------------------------
(
    scripts/net_check.sh "$F" "$REPO/runs/loopmix16x9g/gen06" "$OUT/netcheck_F" "$GUMBEL" "64 250 1000 4000" \
        > "$OUT/netcheck_F.log" 2>&1 || status "WARN: net_check F failed — see $OUT/netcheck_F.log"
    status "net check F: $(cat "$OUT/netcheck_F/net_check.txt" 2>/dev/null)"
) & NC=$!

# ---- 2-3. the drives ------------------------------------------------------------------------------
cargo build --release -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub already serves $HUB_URL"
LAST=$(git log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$(git rev-list --abbrev-commit "$LAST"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > "$OUT/allowed.toml"
"$HUB" serve --bind "0.0.0.0:$PORT" --token-file "$TOK" --allowed-commits "$OUT/allowed.toml" --run-dir "$OUT" \
    >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$F" --max-models 3 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub "ws://127.0.0.1:$PORT/ws" --token-file "$TOK" --name local68 --parallel-games 6 --mem-floor-mb 1536 \
    --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!

# match NAME LABEL CAND CAND_CFG OPP OPP_CFG OPP_ITERS
match() {
    local name="$1" label="$2" cand="$3" ccfg="$4" opp="$5" ocfg="$6" oiters="$7" dir="$OUT/$1"
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp068 $label" \
        --evaluator nn --model "$cand" --bot-config "$ccfg" --mcts-iters 1000 \
        --vs-evaluator nn --vs-model "$opp" --vs-config "$ocfg" --opponent-iters "$oiters" \
        --games 30 --seed 68000 --skip-fixed-rungs --positions "$SETS" --vs-games 600 \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
        || { status "WARN: $name failed — see $dir/eval.log"; return 1; }
    status "$name ($label): $("$PY" - "$dir/report.json" <<'PY'
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
match FvA "F gumbel@1000 vs A gumbel@1000" "$F" "$GUMBEL" "$A" "$GUMBEL" 1000
match FvP "F gumbel@1000 vs F policy-only" "$F" "$GUMBEL" "$F" "$POLICY" 8
wait $NC
status "done ($((SECONDS / 60)) min)"
