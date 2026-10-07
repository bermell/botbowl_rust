#!/usr/bin/env bash
# Search CPU benchmark that a busy box cannot distort (plan 058 §7): instructions retired
# (`perf stat -e instructions:u`) for a fixed, exactly reproducible workload, plus a hash of the
# trajectories it wrote. Run it on two commits: equal hashes mean the search is unchanged, and the
# instruction ratio is the CPU saving. Needs `kernel.perf_event_paranoid <= 2`.
#
#   scripts/perf_search_bench.sh OUT_DIR [TAG]
#
# Reproducibility: `deterministic_hash` (fixed children-map hasher) is built in, and
# `fixed_getrandom.c` is preloaded so std's RandomState keys are fixed too. The hash sorts the
# JSON of HashSet fields (`skills`, `used_skills`, `simple`), whose order is not part of the game.
# Builds into target/perf-16x9, never the loop's target/16x9.
#
# Env: GAMES (3), SEED (4242), ITERS (1000), EVALUATOR (heuristic; or nn with MODEL=…onnx, tract),
# CONFIG (cfgs/gumbel16_f1000_gen.toml), CORE (7: the core it is pinned to, niced).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"; source "$REPO/scripts/lib/git.sh"
OUT="${1:?out dir}"; TAG="${2:-$(git rev-parse --short HEAD)$(dirty_suffix)}"
GAMES="${GAMES:-3}"; SEED="${SEED:-4242}"; ITERS="${ITERS:-1000}"; EVALUATOR="${EVALUATOR:-heuristic}"
CONFIG="${CONFIG:-cfgs/gumbel16_f1000_gen.toml}"; CORE="${CORE:-7}"
mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/perf-16x9"
cargo build --release -p botbowl-ui --features deterministic_hash >> "$OUT/build.log" 2>&1 \
    || { echo "build failed, see $OUT/build.log" >&2; exit 1; }
SHIM="$CARGO_TARGET_DIR/fixed_getrandom.so"
[ "$SHIM" -nt scripts/fixed_getrandom.c ] || gcc -O2 -shared -fPIC -o "$SHIM" scripts/fixed_getrandom.c
NN_ARGS=(); [ "$EVALUATOR" = nn ] && NN_ARGS=(--model "${MODEL:?MODEL=net.onnx for EVALUATOR=nn}")
LD_PRELOAD="$SHIM" taskset -c "$CORE" nice -n 19 perf stat -x, -e instructions:u -o "$OUT/$TAG.stat" \
    "$CARGO_TARGET_DIR/release/botbowl-ui" dataset --mode random-start --board-sizes 16x9/6 \
    --games "$GAMES" --seed "$SEED" --mcts-iters "$ITERS" --bot-config "$CONFIG" \
    --evaluator "$EVALUATOR" "${NN_ARGS[@]}" --out "$OUT/$TAG.jsonl" --truncate > /dev/null 2> "$OUT/$TAG.log"
INS=$(grep instructions "$OUT/$TAG.stat" | cut -d, -f1)
HASH=$(python3 - "$OUT/$TAG.jsonl" <<'PY'
import hashlib, json, sys
SETS = ("skills", "used_skills", "simple")
def canon(x):
    if isinstance(x, dict):
        return {k: sorted(v) if k in SETS and isinstance(v, list) else canon(v) for k, v in x.items()}
    return [canon(v) for v in x] if isinstance(x, list) else x
h = hashlib.md5()
for line in open(sys.argv[1]):
    d = json.loads(line); d.pop("meta", None)
    h.update(json.dumps(canon(d), sort_keys=True).encode())
print(h.hexdigest()[:12])
PY
)
echo "$TAG instructions=$INS trajectories=$HASH" | tee -a "$OUT/results.txt"
