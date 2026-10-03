#!/usr/bin/env bash
# exp060c: plan 053, after exp060/exp060b. Drives only, one job at a time (exp060 ran a probe and a
# drive match together and ran the box out of memory).
#
#   1. Probe floors 2000 and 4000 on exp057's turn-start states (`--advance 0` only: an advanced
#      state cannot be matched across runs), next to exp060b's floors 50/200/1000.
#   2. Pick the floor with the lowest mean regret at 1000 descents over both boards (advance 0).
#   3. Drives with that floor: gen04 Gumbel@1000 vs gen04 PUCT@1000 on both gen04-screened sets,
#      then 16x9 only vs PUCT@4000. SPRT 0.5:0.55, at most 800 drives per board.
#
#   scripts/exp060c_gumbel_drives.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp060/drives_c"; QF="$REPO/runs/exp060/qfloor"; mkdir -p "$OUT" "$QF/presets"
M="$REPO/models/az_v7"; GEN21="$M/bbnet_mix16x9_gen21.onnx"; GEN04="$M/bbnet_mix16x9d1k_gen04.onnx"
ITERS="$REPO/cfgs/exact_iters.toml"; POS="$REPO/runs/exp059/positions"
SPRT=0.5:0.55; CAP=800
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp060c.sock
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

# ---- 1. probe the larger floors at advance 0 -----------------------------------------------------
nn_up "$GEN21"
for f in 2000 4000; do
    printf 'budget_mode = "iterations"\ngumbel_m = 16\ngumbel_q_floor = %s\n' "$f" > "$QF/presets/gumbel16_f$f.toml"
    for board in 14x7/4 16x9/6; do
        tag="f${f}_${board%%/*}_a0"
        [ -s "$QF/conv_$tag.jsonl" ] && continue
        "$UI" convergence --states 60 --repeats 3 --budgets 250,500,1000,2000 --board "$board" --advance 0 \
            --evaluator nn --model "$GEN21" --nn-server "$SOCK" --parallel 8 --bot-config "$QF/presets/gumbel16_f$f.toml" \
            --seed 91000000 --out "$QF/conv_$tag.jsonl" > "$QF/conv_$tag.log" 2>&1 || die "$tag"
    done
done
nn_down
for b in 14x7 16x9; do
    "$PY" "$REPO/scripts/convergence_topk.py" "$REPO"/runs/exp057/conv_${b}_a0.jsonl "$REPO"/runs/exp060/conv_gumbel_${b}_a0.jsonl \
        "$QF"/conv_f*_${b}_a0.jsonl > "$OUT/summary_a0_$b.txt" 2>&1 || die "summary $b"
done
# ---- 2. the floor with the lowest mean regret at 1000 over both boards ---------------------------
FLOOR=$("$PY" - "$OUT/summary_a0_14x7.txt" "$OUT/summary_a0_16x9.txt" <<'PY'
import re, sys, collections
reg = collections.defaultdict(list)
for f in sys.argv[1:]:
    for line in open(f):
        m = re.match(r"\s+gumbel16_f(\d+) 1000\s+\S+\s+\S+\s+\S+\s+(\S+)", line)
        if m:
            reg[int(m.group(1))].append(float(m.group(2)))
best = min((sum(v) / len(v), k) for k, v in reg.items() if len(v) >= 2)  # rows repeat under "advance all"
print(best[1])
PY
) || die "floor choice"
status "probe done; regret at 1000 (advance 0): $(grep -hE '^ +(1000|gumbel16_f[0-9]+ 1000) ' "$OUT"/summary_a0_*.txt | awk '{print $1$2, $(NF-5)}' | paste -sd' '); chosen floor $FLOOR"
GUMBEL="$QF/presets/gumbel16_f$FLOOR.toml"

# ---- 3. drives -----------------------------------------------------------------------------------
git -C "$REPO" diff --quiet d37a1dc HEAD -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts \
    || status "note: protocol v12 — a remote worker must be rebuilt at this commit to help"
"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
nn_up "$GEN04"
worker_up() {
    "$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games "$1" --mem-floor-mb 1536 \
        --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
}
worker_down() { kill "$WORKER_PID" 2>/dev/null; wait "$WORKER_PID" 2>/dev/null; WORKER_PID=""; }
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
drives "f${FLOOR}_vs_puct1000" "gen04 Gumbel(f$FLOOR)@1000 vs PUCT@1000 (drives)" \
    "$POS/contested_14x7_gen04.json,$POS/contested_16x9_gen04.json" 1000
worker_down
worker_up 4
drives "f${FLOOR}_vs_puct4000_16x9" "gen04 Gumbel(f$FLOOR)@1000 vs PUCT@4000, 16x9 (drives)" "$POS/contested_16x9_gen04.json" 4000
status "done ($((SECONDS / 60)) min of drives)"
