#!/usr/bin/env bash
# exp065: plan 055 phase 2 — the override audit on the merged code (mean backup hardcoded).
# Two nets x two searches, each on data its net generated and was never trained on, eval boards only
# (14x7, 16x9: no sub-70-cell non-games), 1000 overrides x 64 paired policy-only playouts.
#   g05_f1000   g_gen05    on loopmix16x9g/gen06 (all 8 shards)   cfgs/gumbel16_f1000.toml @1000
#   g05_f4000   g_gen05    on loopmix16x9g/gen06                  cfgs/gumbel16_f4000.toml @1000
#   d1k_f1000   d1k gen04  on loopmix16x9g/gen01 (all 8 shards)   cfgs/gumbel16_f1000.toml @1000
#   d1k_f4000   d1k gen04  on loopmix16x9g/gen01                  cfgs/gumbel16_f4000.toml @1000
# One GPU sidecar serves both nets; four audits at 2 threads each (8 cores).
#
#   scripts/exp065_override_audit.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="$REPO/runs/exp065"; mkdir -p "$OUT"
M="$REPO/models/az_v7"; L="$REPO/runs/loopmix16x9g"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"; SOCK=/tmp/bbnn-exp065.sock
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
NN_PID=""; trap '[ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"' EXIT INT TERM
git diff --quiet || die "dirty tree"
status "start: commit $(git rev-parse --short HEAD)"
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || die "build"
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$M/bbnet_mix16x9g_gen05.onnx" --max-models 4 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
shards() { for k in 0 1 2 3 4 5 6 7; do echo "$L/$1/shard$k.jsonl"; done; }
audit() {
    local name="$1" net="$2" gen="$3" cfg="$4"
    [ -s "$OUT/$name.done" ] && return 0
    # shellcheck disable=SC2046
    "$UI" override-audit --corpus $(shards "$gen") --model "$net" --nn-server "$SOCK" \
        --search-config "cfgs/$cfg.toml" --search-iters 1000 --decisions 1000 --playouts 64 --parallel 2 \
        --board 14x7,16x9 --seed 65000 --out "$OUT/$name.jsonl" > "$OUT/$name.log" 2>&1 \
        || { status "WARN: $name failed — see $OUT/$name.log"; return 1; }
    "$PY" scripts/override_audit_summary.py "$OUT/$name.jsonl" > "$OUT/$name.summary.txt" 2>&1
    echo done > "$OUT/$name.done"
    status "$name done: $(grep -m1 'all overrides' "$OUT/$name.summary.txt" | tr -s ' ')"
}
SECONDS=0
audit g05_f1000 "$M/bbnet_mix16x9g_gen05.onnx" gen06 gumbel16_f1000 & A=$!
audit g05_f4000 "$M/bbnet_mix16x9g_gen05.onnx" gen06 gumbel16_f4000 & B=$!
audit d1k_f1000 "$M/bbnet_mix16x9d1k_gen04.onnx" gen01 gumbel16_f1000 & C=$!
audit d1k_f4000 "$M/bbnet_mix16x9d1k_gen04.onnx" gen01 gumbel16_f4000 & D=$!
wait $A $B $C $D
"$PY" scripts/override_audit_summary.py "$OUT"/g05_*.jsonl "$OUT"/d1k_*.jsonl > "$OUT/summary_all.txt" 2>&1
status "done ($((SECONDS / 60)) min)"
