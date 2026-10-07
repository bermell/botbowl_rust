#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# exp064: plan 055 phase 1 — does the search beat the bare policy, and which search does?
# Every match is a search configuration (the candidate) against policy-only on the SAME net
# (cfgs/policy_only.toml at 8 descents: the prior's argmax), paired drives on the gen04g contested
# sets, fixed 300 pairs per board, --seed 64000. Points are the SEARCH's: above 0.5 means search
# beats policy.
#
#   0. Waits for exp063's gate line, then stops exp063, so its E2b training never starts and the
#      loop is not relaunched (the user, 2026-10-04: no loop until search beats policy).
#   1. At 1000 descents and below, all concurrent (22 streams with the laptop):
#        N0   policy-only vs policy-only                 null check, must read 0.5
#        G250, G1000                                    Gumbel f1000 budget curve
#        P1000                                          PUCT (cfgs/exact_iters.toml)
#        MEAN, F4000, F300, H2                          Gumbel ablations: mean backup, q floor 4000
#                                                       and 300, two-turn horizon
#        G1000_d1k                                      Gumbel f1000 on d1k gen04 (a second net)
#   2. G4000: Gumbel f1000 at 4000 descents, the local worker at 6 streams (tree memory).
#
#   scripts/exp064_search_vs_policy.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp064"; mkdir -p "$OUT"
M="$REPO/models/az_v7"; POS="$REPO/runs/loopmix16x9g/positions"
SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
C="$REPO/cfgs"; POLICY="$C/policy_only.toml"
NET="$M/bbnet_mix16x9g_gen05.onnx"; D1K="$M/bbnet_mix16x9d1k_gen04.onnx"
DRIVES=600   # per position set: 300 pairs
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp064.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
down() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; WORKER_PID="" NN_PID="" HUB_PID=""; rm -f "$SOCK"; }
trap down EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"

# ---- 0. take over from exp063 at its gate ---------------------------------------------------------
# Pinned by its exact command line (never pgrep -f: it matches this script's own shell).
E63=$(ps -eo pid=,args= | awk '$2 == "bash" && $3 ~ /exp063_plan054\.sh$/ {print $1}' | head -1)
if [ -n "$E63" ]; then
    status "waiting for exp063 (pid $E63) to reach its gate"
    until grep -q 'gate:' "$REPO/runs/exp063/status.md" 2>/dev/null || ! kill -0 "$E63" 2>/dev/null; do sleep 5; done
    kill "$E63" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$E63" 2>/dev/null || break; sleep 1; done
    status "exp063 stopped at its gate: $(grep 'gate:' "$REPO/runs/exp063/status.md" | tail -1 | cut -c23-)"
    echo "[$(date '+%F %T')] stopped by exp064 after the gate (plan 055: no loop or E2b until search beats policy)" >> "$REPO/runs/exp063/status.md"
    sleep 5
fi
for _ in $(seq 60); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 || break; sleep 2; done
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub still serves $HUB_URL"

cargo build --release -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build"
LAST=$(git -C "$REPO" log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git -C "$REPO" rev-parse --short HEAD)" \
    "$(git -C "$REPO" rev-list --abbrev-commit "$LAST"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > "$REPO/hub-allowed-commits.toml"
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$NET" --max-models 4 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
worker_up() {
    "$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games "$1" --mem-floor-mb 1536 \
        --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
}

# match NAME NET CONFIG ITERS: the search configuration vs policy-only on the same net.
match() {
    local name="$1" net="$2" cfg="$3" iters="$4" dir="$OUT/$1"
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp064 $name: $(basename "$cfg" .toml)@$iters vs policy-only" \
        --evaluator nn --model "$net" --bot-config "$cfg" --mcts-iters "$iters" \
        --vs-evaluator nn --vs-model "$net" --vs-config "$POLICY" --opponent-iters 8 \
        --games 30 --seed 64000 --skip-fixed-rungs --positions "$SETS" --vs-games "$DRIVES" \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
        || { status "WARN: $name failed — see $dir/eval.log"; return 1; }
    status "$name ($(basename "$cfg" .toml)@$iters, $(basename "$net" .onnx)) vs policy: $("$PY" - "$dir/report.json" <<'PY'
import json, math, re, sys
rows = json.load(open(sys.argv[1]))["ladder"]
parts = [f"{re.search(r'@(\S+)$', r['opponent']).group(1)} {r['points']:.3f} ± {r['points_se']:.3f}" for r in rows]
mean = sum(r["points"] for r in rows) / len(rows)
se = math.sqrt(sum(r["points_se"] ** 2 for r in rows)) / len(rows)
print(" | ".join(parts) + f" | mean {mean:.3f} ± {se:.3f}")
PY
)"
}

# ---- 1. at 1000 descents and below ----------------------------------------------------------------
SECONDS=0
worker_up 12
PIDS=""
match N0 "$NET" "$POLICY" 8 & PIDS="$PIDS $!"
match G250 "$NET" "$C/gumbel16_f1000.toml" 250 & PIDS="$PIDS $!"
match G1000 "$NET" "$C/gumbel16_f1000.toml" 1000 & PIDS="$PIDS $!"
match P1000 "$NET" "$C/exact_iters.toml" 1000 & PIDS="$PIDS $!"
match MEAN "$NET" "$C/gumbel16_f1000_mean.toml" 1000 & PIDS="$PIDS $!"
match F4000 "$NET" "$C/gumbel16_f4000.toml" 1000 & PIDS="$PIDS $!"
match F300 "$NET" "$C/gumbel16_f300.toml" 1000 & PIDS="$PIDS $!"
match H2 "$NET" "$C/gumbel16_f1000_h2.toml" 1000 & PIDS="$PIDS $!"
match G1000_d1k "$D1K" "$C/gumbel16_f1000.toml" 1000 & PIDS="$PIDS $!"
# shellcheck disable=SC2086
wait $PIDS
status "phase 1 done ($((SECONDS / 60)) min)"

# ---- 2. 4000 descents, fewer local streams ---------------------------------------------------------
kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null
worker_up 6
match G4000 "$NET" "$C/gumbel16_f1000.toml" 4000
status "done ($((SECONDS / 60)) min)"
