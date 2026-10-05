#!/usr/bin/env bash
# exp068b: exp068's F-vs-policy match, moved to a hub on :13337 so the laptop (outside the LAN, only
# 13337 forwarded) can help once exp066 releases that port. exp068's controller was stopped after
# it started FvA; FvA finishes on :13338 with the local worker, and this script collects it.
#   1. Waits for exp066 to finish (it owns :13337 until then).
#   2. Hub on :13337, local worker (6 streams), the laptop reconnects on its own. The allowlist adds
#      exp066's commits: the only game-code commit since (99ae314) is prepare's TD(lambda), which a
#      worker never runs.
#   3. FvP: F gumbel@1000 vs F policy-only, exp068's settings (seed 68000, 300 pairs per board).
#   4. When FvA's report lands: its status line, then stops exp068's orphaned :13338 hub/sidecar/worker.
#
#   scripts/exp068b_fvp_on_13337.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="$REPO/runs/exp068"
F="$REPO/runs/exp067/arms/F.onnx"
POS="$REPO/runs/loopmix16x9g/positions"
SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
GUMBEL="$REPO/cfgs/gumbel16_f1000.toml"; POLICY="$REPO/cfgs/policy_only.toml"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
PORT=13337; HUB_URL="http://127.0.0.1:$PORT"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp068b.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: exp068b: $*"; exit 1; }
summary() {
    "$PY" - "$1" <<'PY'
import json, math, re, sys
rows = json.load(open(sys.argv[1]))["ladder"]
parts = [f"{re.search(r'@(\S+)$', r['opponent']).group(1)} {r['points']:.3f} ± {r['points_se']:.3f}" for r in rows]
mean = sum(r["points"] for r in rows) / len(rows)
se = math.sqrt(sum(r["points_se"] ** 2 for r in rows)) / len(rows)
print(" | ".join(parts) + f" | mean {mean:.3f} ± {se:.3f}")
PY
}
HUB_PID="" NN_PID="" WORKER_PID=""
down() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; WORKER_PID="" NN_PID="" HUB_PID=""; rm -f "$SOCK"; }
trap down EXIT INT TERM
git diff --quiet || die "dirty tree"
status "exp068b start: commit $(git rev-parse --short HEAD); FvP waits for exp066 to free :13337"

# FvA's collector, in the background: exp068's controller is gone, so nobody else reports it.
(
    until [ -s "$OUT/FvA/report.json" ]; do sleep 30; done
    status "FvA (F gumbel@1000 vs A gumbel@1000): $(summary "$OUT/FvA/report.json")"
    sleep 10
    # shellcheck disable=SC2046
    kill $(cat "$OUT/orphans.pids") 2>/dev/null; rm -f /tmp/bbnn-exp068.sock
    status "stopped exp068's :13338 hub, sidecar and worker"
) & COLLECT=$!

until grep -q '] done' "$REPO/runs/exp066/status.md" 2>/dev/null; do sleep 30; done
for _ in $(seq 60); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 || break; sleep 5; done
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub still serves $HUB_URL"
cargo build --release -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"
LAST=$(git log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git rev-parse --short HEAD)" \
    "$( (git rev-list --abbrev-commit "$LAST"^..HEAD; echo 8cc6ce7 2a8fe4e 14d915a e7d5c1e 381c245 e8c0144 d45e391 | tr ' ' '\n') \
        | sort -u | sed 's/.*/"&"/' | paste -sd,)" > "$OUT/allowed_13337.toml"
"$HUB" serve --bind "0.0.0.0:$PORT" --token-file "$TOK" --allowed-commits "$OUT/allowed_13337.toml" --run-dir "$OUT" \
    >> "$OUT/hub_13337.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$F" --max-models 2 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server_13337.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub "ws://127.0.0.1:$PORT/ws" --token-file "$TOK" --name local68b --parallel-games 6 --mem-floor-mb 1536 \
    --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker_13337.log" 2>&1 & WORKER_PID=$!
status "exp068b: hub on :$PORT up; FvP starts"

dir="$OUT/FvP"
if [ ! -s "$dir/report.json" ]; then
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp068 F gumbel@1000 vs F policy-only" \
        --evaluator nn --model "$F" --bot-config "$GUMBEL" --mcts-iters 1000 \
        --vs-evaluator nn --vs-model "$F" --vs-config "$POLICY" --opponent-iters 8 \
        --games 30 --seed 68000 --skip-fixed-rungs --positions "$SETS" --vs-games 600 \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
        || die "FvP failed — see $dir/eval.log"
fi
status "FvP (F gumbel@1000 vs F policy-only): $(summary "$dir/report.json")"
wait $COLLECT
status "done"
