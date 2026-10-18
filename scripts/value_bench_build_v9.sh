#!/usr/bin/env bash
# Plan 058: re-freeze the value benchmark under the new rules (schema v9, 16 more skills, per-player
# setup) from the loop's gen01 held-out shards (4, 7: never trained on), with gen01's generator.
# override-audit --all keeps every sampled decision; the search is irrelevant here (only MC(s) =
# policy.mc_* and V(s) are used), so it runs at a token budget.
#   scripts/value_bench_build_v9.sh RUN_DIR NET.onnx OUT.jsonl
#
# Plan 065 generalises it to any generation and board set, defaults unchanged (the v9 bench):
#   GEN=gen44 BOARDS=20x11 BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 \
#       scripts/value_bench_build_v9.sh runs/loopmix16x9v9 NET.onnx runs/value_bench/v9_20x11_gen44_val.jsonl
# Env: GEN (default gen01), BOARDS (default 14x7,16x9), SHARDS (held-out shards, default "4 7"),
# DECISIONS (default 3000), SEED (default 58100), BOARD_SIZE_W/H, BOARD_PLAYERS (capacity, default 16x9/6).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
RUN="${1:?run dir}"; NET="${2:?net}"; OUT="${3:?out jsonl}"
GEN="${GEN:-gen01}"; BOARDS="${BOARDS:-14x7,16x9}"; SHARDS="${SHARDS:-4 7}"; DECISIONS="${DECISIONS:-3000}"; SEED="${SEED:-58100}"
G="$RUN/$GEN"; W="${OUT%.jsonl}_build"; [ "$GEN" = gen01 ] && [ "$BOARDS" = 14x7,16x9 ] && W="$(dirname "$OUT")/v9_build"
mkdir -p "$W"
until [ -e "$G/.generated" ]; do sleep 60; done
export BOARD_SIZE_W="${BOARD_SIZE_W:-16}" BOARD_SIZE_H="${BOARD_SIZE_H:-9}" BOARD_PLAYERS="${BOARD_PLAYERS:-6}"
export CARGO_TARGET_DIR="$REPO/target/${BOARD_SIZE_W}x${BOARD_SIZE_H}"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"; SOCK="/tmp/bbnn-vbuild-$$.sock"
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$NET" \
    --canvas "$((BOARD_SIZE_H + 2))x$((BOARD_SIZE_W + 2))" > "$W/nn_server.log" 2>&1 & NN=$!
trap 'kill $NN 2>/dev/null; rm -f "$SOCK"' EXIT
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
CORPUS=(); for k in $SHARDS; do CORPUS+=("$G/shard$k.jsonl"); done
"$UI" override-audit --corpus "${CORPUS[@]}" --model "$NET" --nn-server "$SOCK" \
    --search-config cfgs/gumbel16_f1000.toml --search-iters 16 --all --decisions "$DECISIONS" --playouts 48 \
    --parallel 24 --board "$BOARDS" --seed "$SEED" --out "$W/rows.jsonl" > "$W/audit.log" 2>&1 || exit 1
"$PY" scripts/value_bench_freeze.py "$W/rows.jsonl" --out "$OUT" 2>&1 | tail -1
