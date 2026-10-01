#!/usr/bin/env bash
# Plan 051 step 5, after the gold block: build and screen the drive position sets, run P2 (SPRT
# on paired contested drives) R times on every validation pair, and score P1/P2 against gold.
# Waits for scripts/plan051_gold.sh to finish first. Resumable: finished steps leave markers.
#
#   scripts/plan051_proxies.sh
#   REPS=3 SCREEN_CANDIDATES=500 SCREEN_PLAYOUTS=4 scripts/plan051_proxies.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GOLD="$REPO/runs/validation"; OUT="$REPO/runs/plan051_proxy"; POS="$OUT/positions"; mkdir -p "$POS"
REPS="${REPS:-3}"; CANDIDATES="${SCREEN_CANDIDATES:-500}"; PLAYOUTS="${SCREEN_PLAYOUTS:-4}"
SPRT="${SPRT:-0.5:0.55}"; DRIVE_CAP="${DRIVE_CAP:-2000}"; BAND="${BAND:-0.25:0.75}"
REFERENCE="$REPO/models/az_v7/bbnet_mix16x9_gen21.onnx"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-proxy.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD); waiting for the gold block"
until grep -qE "\] (done \(|FATAL)" "$GOLD/status.md" 2>/dev/null; do sleep 60; done
grep -q "FATAL" "$GOLD/status.md" && die "gold block failed; see $GOLD/status.md"
for _ in $(seq 120); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 || break; sleep 5; done
status "gold done; building"
cargo build --release -p botbowl-ui -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"
CFG="$REPO/cfgs/$("$PY" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["config"])' "$GOLD/validation_pairs.toml").toml"
ITERS=$("$PY" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["mcts_iters"])' "$GOLD/validation_pairs.toml")
mapfile -t ROWS < <("$PY" -c '
import sys, tomllib
for p in tomllib.load(open(sys.argv[1], "rb"))["pair"]:
    print("|".join((p["name"], p["candidate"], p["opponent"])))' "$GOLD/validation_pairs.toml")

"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$REFERENCE" --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 14 --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!

# job NAME OUTDIR MODEL VS_MODEL SEED [extra args] — one drive job over both position sets.
job() {
    local name="$1" out="$2" model="$3" vs="$4" seed="$5"; shift 5
    mkdir -p "$out"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --evaluator nn --model "$model" \
        --bot-config "$CFG" --mcts-iters "$ITERS" --games 30 --seed "$seed" --skip-fixed-rungs \
        --vs-evaluator nn --vs-model "$vs" "$@" \
        --per-game-out "$out/eval.games.jsonl" --out "$out/report.json" --wait > "$out/eval.log" 2>&1
}

# ---- positions: write, screen with the reference against itself, keep the contested ----------
if [ ! -e "$POS/.screened" ]; then
    SECONDS=0
    "$UI" positions --board 14x7/4 --count "$CANDIDATES" --seed-base 70000000 --name cand_14x7 --out "$POS/cand_14x7.json" 2>> "$STATUS" || die "positions 14x7"
    "$UI" positions --board 16x9/6 --count "$CANDIDATES" --seed-base 71000000 --name cand_16x9 --out "$POS/cand_16x9.json" 2>> "$STATUS" || die "positions 16x9"
    job screen "$POS/screen" "$REFERENCE" "$REFERENCE" 52000 \
        --positions "$POS/cand_14x7.json,$POS/cand_16x9.json" --vs-games $((CANDIDATES * PLAYOUTS)) || die "screen job"
    for b in 14x7 16x9; do
        "$PY" "$REPO/scripts/positions_screen.py" --set "$POS/cand_$b.json" --games "$POS/screen/eval.games.jsonl" \
            --reference "$REFERENCE" --band "$BAND" --min-playouts "$PLAYOUTS" --name "contested_$b" \
            --out "$POS/contested_$b.json" >> "$STATUS" 2>&1 || die "screen $b"
    done
    touch "$POS/.screened"
    status "screen done ($((SECONDS / 60)) min)"
fi
SETS="$POS/contested_14x7.json,$POS/contested_16x9.json"

# ---- P2: SPRT on paired contested drives, R reps, the pairs of one rep side by side -----------
for r in $(seq 1 "$REPS"); do
    [ -e "$OUT/p2/.rep$r" ] && continue
    SECONDS=0
    declare -A J=()
    for row in "${ROWS[@]}"; do
        IFS='|' read -r name cand opp <<< "$row"
        out="$OUT/p2/$name/r$r"
        [ -e "$out/report.json" ] && continue
        job "$name" "$out" "$REPO/$cand" "$REPO/$opp" $((53000 + r * 1000)) \
            --positions "$SETS" --sprt "$SPRT" --vs-games "$DRIVE_CAP" &
        J[$name]=$!
    done
    for name in "${!J[@]}"; do wait "${J[$name]}" || die "P2 $name r$r"; done
    touch "$OUT/p2/.rep$r"
    status "P2 rep $r done ($((SECONDS / 60)) min)"
done

"$PY" "$REPO/scripts/validate_proxy.py" --gold "$GOLD" --proxy "P2=$OUT/p2" --sprt "$SPRT" > "$OUT/validation.txt" 2>&1 || die "validate"
status "validation: $(tail -3 "$OUT/validation.txt" | tr '\n' ' ')"
status "done"
