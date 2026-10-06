#!/usr/bin/env bash
# Plan 051 step 3: gold results for the frozen validation pairs (cfgs/validation_pairs.toml).
# Every pair is a full fixed-N head-to-head under one search for both seats, played with the
# current binary into runs/validation/<name>/. A pair that already has gold is skipped, never
# overwritten. A binary change that touches play needs a new output root (GOLD_ROOT), not a rerun.
#
#   scripts/plan051_gold.sh                       # all pairs, jobs share the hub round-robin
#   GOLD_ROOT=runs/validation_<commit> scripts/plan051_gold.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PAIRS="$REPO/cfgs/validation_pairs.toml"
ROOT="${GOLD_ROOT:-$REPO/runs/validation}"; mkdir -p "$ROOT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-gold.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$ROOT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD), pairs $(basename "$PAIRS")"
cargo build --release -p botbowl-hub -p botbowl-worker >> "$ROOT/build.log" 2>&1 || die "build"

# name|candidate|opponent per line, plus the shared settings.
read -r CONFIG ITERS BOARDS GAMES SEED < <("$PY" -c '
import sys, tomllib
c = tomllib.load(open(sys.argv[1], "rb"))
print(c["config"], c["mcts_iters"], c["boards"], c["games_per_board"], c["seed"])' "$PAIRS")
mapfile -t ROWS < <("$PY" -c '
import sys, tomllib
for p in tomllib.load(open(sys.argv[1], "rb"))["pair"]:
    print("|".join((p["name"], p["candidate"], p["opponent"])))' "$PAIRS")
[ "${#ROWS[@]}" -gt 0 ] || die "no pairs in $PAIRS"
CFG="$REPO/cfgs/$CONFIG.toml"; [ -f "$CFG" ] || die "no preset $CFG"
cp "$PAIRS" "$ROOT/validation_pairs.toml"

"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$ROOT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
FIRST_MODEL="$REPO/$(echo "${ROWS[0]}" | cut -d'|' -f2)"
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$FIRST_MODEL" --stats-every 300 --canvas 11x18 >> "$ROOT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 14 --cache-dir "$ROOT/worker-cache" --nn-server "$SOCK" >> "$ROOT/worker.log" 2>&1 & WORKER_PID=$!

declare -A J
SECONDS=0
for row in "${ROWS[@]}"; do
    IFS='|' read -r name cand opp <<< "$row"
    out="$ROOT/$name"
    if [ -e "$out/.done" ]; then status "$name: gold exists, skipped"; continue; fi
    [ -e "$out" ] && die "$name: $out exists without .done; move it aside, never overwrite gold"
    mkdir -p "$out"
    for m in "$cand" "$opp"; do [ -f "$REPO/$m" ] || die "$name: missing $m"; done
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$REPO/$cand" \
        --bot-config "$CFG" --mcts-iters "$ITERS" --games 30 --seed "$SEED" --skip-fixed-rungs \
        --board-sizes "$BOARDS" --cells-per-player 26 \
        --vs-games "$GAMES" --vs-evaluator nn --vs-model "$REPO/$opp" \
        --per-game-out "$out/eval.games.jsonl" --out "$out/report.json" --wait > "$out/eval.log" 2>&1 &
    J[$name]=$!
done
status "submitted ${#J[@]} pair(s): $CONFIG at $ITERS, $GAMES per board on $BOARDS, seed $SEED"
for name in "${!J[@]}"; do
    wait "${J[$name]}" || die "$name: eval failed, see $ROOT/$name/eval.log"
    touch "$ROOT/$name/.done"
    status "$name done ($((SECONDS / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$ROOT/$name/report.json" 2>&1 | tr '\n' ' ' | cut -c1-900)"
done
status "done ($((SECONDS / 60)) min)"
