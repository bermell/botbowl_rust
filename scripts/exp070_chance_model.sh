#!/usr/bin/env bash
# exp070: plan 061's chance-model arms, drives only (plan 051 paired contested drives, SPRT,
# pentanomial pairs, at most CAP drives per board). Same net both sides, 1000 descents, every arm a
# `gumbel16_f1000` with one chance change against `cfgs/gumbel16_f1000.toml` (C1 against the B
# winner first). One job at a time, in plan 061 §7's order; a finished arm (report.json present) is
# skipped, so a relaunch resumes.
#
#   scripts/exp070_chance_model.sh                     # every arm
#   ARMS="A1 A2" scripts/exp070_chance_model.sh        # a subset
#   B_WINNER=chance_sampled ARMS=C1 scripts/exp070_chance_model.sh
#
# Starts its own hub on PORT (default 13338, so it never meets the loop's 13337), one local worker
# and one GPU sidecar. Protocol v18: a remote worker must be rebuilt at this commit to help.
# Env: NET (gen13 .onnx), OUT (runs/exp070), PORT, STREAMS (12), CAP (800).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$REPO/runs/exp070}"; case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac; mkdir -p "$OUT"
NET="${NET:-$REPO/models/az_v7/bbnet_mix16x9v9_gen13.onnx}"
POS="$REPO/runs/loopmix16x9g/positions"; SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
CONTROL="$REPO/cfgs/gumbel16_f1000.toml"; CAP="${CAP:-800}"; PORT="${PORT:-13338}"; STREAMS="${STREAMS:-12}"
# A2 (passes) is deferred by default: plan 061 §2 found ~no passes in real games or in search.
ARMS="${ARMS:-A1 A3 B1 B2 B3 B4 C1}"; B_WINNER="${B_WINNER:-chance_partial}"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:$PORT"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp070.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD), net $(basename "$NET"), arms $ARMS"
cargo build --release -p botbowl-ui -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub already serves $HUB_URL"
"$HUB" serve --bind "127.0.0.1:$PORT" --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$NET" --max-models 2 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub "ws://127.0.0.1:$PORT/ws" --token-file "$TOK" --name local --parallel-games "$STREAMS" \
    --mem-floor-mb 1536 --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!

# drives NAME CANDIDATE_PRESET OPPONENT_PRESET SPRT
drives() {
    local name="$1" cand="$2" opp="$3" sprt="$4" dir="$OUT/$1"
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" \
        --label "exp070 $name: $(basename "$cand" .toml) vs $(basename "$opp" .toml)" \
        --evaluator nn --model "$NET" --vs-evaluator nn --vs-model "$NET" --mcts-iters 1000 \
        --bot-config "$cand" --vs-config "$opp" --seed 0 --skip-fixed-rungs \
        --positions "$SETS" --sprt "$sprt" --vs-games "$CAP" \
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

C="$REPO/cfgs"
SECONDS=0
for arm in $ARMS; do
    case "$arm" in
        A1) drives A1_bounce "$C/chance_bounce.toml" "$CONTROL" 0.47:0.5 ;;
        A2) drives A2_pass "$C/chance_pass.toml" "$CONTROL" 0.47:0.5 ;;
        A3) drives A3_throw_in "$C/chance_throw_in.toml" "$CONTROL" 0.47:0.5 ;;
        B1) drives B1_partial "$C/chance_partial.toml" "$CONTROL" 0.5:0.55 ;;
        B2) drives B2_mass90 "$C/chance_mass90.toml" "$CONTROL" 0.5:0.55 ;;
        B3) drives B3_sampled "$C/chance_sampled.toml" "$CONTROL" 0.5:0.55 ;;
        B4) drives B4_widen "$C/chance_widen.toml" "$CONTROL" 0.5:0.55 ;;
        C1)
            # Every roll model under the B winner's backup: first against the winner alone, then the
            # control.
            ALL="$OUT/presets/chance_all_${B_WINNER#chance_}.toml"; mkdir -p "$OUT/presets"
            grep -v '^chance_backup\|^chance_mass\|^chance_widen' "$C/chance_all_partial.toml" > "$ALL"
            grep '^chance_backup\|^chance_mass\|^chance_widen' "$C/$B_WINNER.toml" >> "$ALL"
            drives C1_vs_winner "$ALL" "$C/$B_WINNER.toml" 0.5:0.55
            drives C1_vs_control "$ALL" "$CONTROL" 0.47:0.5
            ;;
        *) die "unknown arm $arm" ;;
    esac
done
status "done ($((SECONDS / 60)) min of drives)"
