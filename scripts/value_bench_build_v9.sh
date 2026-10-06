#!/usr/bin/env bash
# Plan 058: re-freeze the value benchmark under the new rules (schema v9, 16 more skills, per-player
# setup) from the loop's gen01 held-out shards (4, 7: never trained on), with gen01's generator.
# override-audit --all keeps every sampled decision; the search is irrelevant here (only MC(s) =
# policy.mc_* and V(s) are used), so it runs at a token budget.
#   scripts/value_bench_build_v9.sh RUN_DIR NET.onnx OUT.jsonl
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
RUN="${1:?run dir}"; NET="${2:?net}"; OUT="${3:?out jsonl}"
G="$RUN/gen01"; W="$(dirname "$OUT")/v9_build"; mkdir -p "$W"
until [ -e "$G/.generated" ]; do sleep 60; done
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"; SOCK="/tmp/bbnn-vbuild-$$.sock"
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$NET" --canvas 11x18 > "$W/nn_server.log" 2>&1 & NN=$!
trap 'kill $NN 2>/dev/null; rm -f "$SOCK"' EXIT
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
"$UI" override-audit --corpus "$G/shard4.jsonl" "$G/shard7.jsonl" --model "$NET" --nn-server "$SOCK" \
    --search-config cfgs/gumbel16_f1000.toml --search-iters 16 --all --decisions 3000 --playouts 48 \
    --parallel 24 --board 14x7,16x9 --seed 58100 --out "$W/rows.jsonl" > "$W/audit.log" 2>&1 || exit 1
"$PY" scripts/value_bench_freeze.py "$W/rows.jsonl" --out "$OUT" 2>&1 | tail -1
