#!/usr/bin/env bash
# Which part of the roll-model fix loses to the legacy search? exp050 (B) measured the full fix at
# 0.415 against legacy with the heuristic evaluator (200 games, 14x7). Each diagnostic variant here
# restores one piece of the legacy model and plays legacy under the same settings and seeds.
#   diag_exact_scripted_pass   exact, but passes still fumble
#   diag_exact_through_half    exact rolls, horizon runs through half time
#   diag_injury_only           only the injury roll exact
# Single process (`botbowl-ui eval`), its own target dir so running jobs' binaries stay untouched.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp051"; mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6
export CARGO_TARGET_DIR="$REPO/target/16x9diag"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }
git -C "$REPO" diff --quiet || { status "FATAL: dirty tree"; exit 1; }
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || { status "FATAL: build"; exit 1; }
UI="$CARGO_TARGET_DIR/release/botbowl-ui"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
for v in exact_scripted_pass exact_through_half injury_only; do
    [ -s "$OUT/report_$v.json" ] && continue
    SECONDS=0
    nice "$UI" eval --evaluator heuristic --bot-config "$REPO/cfgs/diag_$v.toml" \
        --vs-evaluator heuristic --vs-config "$REPO/cfgs/legacy_chance_visits.toml" \
        --mcts-iters 500 --seed 0 --skip-lectures --skip-fixed-rungs --vs-games 200 \
        --board-sizes 14x7 --cells-per-player 26 --parallel-games "${PARALLEL:-4}" \
        --per-game-out "$OUT/$v.games.jsonl" --out "$OUT/report_$v.json" > "$OUT/eval_$v.log" 2>&1 \
        || { status "FATAL: $v failed"; exit 1; }
    status "$v vs legacy done ($((SECONDS / 60)) min): $("$REPO/train/.venv/bin/python" "$REPO/scripts/eval_summary.py" "$OUT/report_$v.json" 2>&1 | tr '\n' ' ' | cut -c1-400)"
done
status "done"
