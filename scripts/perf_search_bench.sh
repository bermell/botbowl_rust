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
# JSON of set fields (`skills`, `used_skills`, `simple`, and the per-square lists under `data`),
# whose order is not part of the game, and leaves out each sample's `tree` block (plan 060: it
# describes the search, not the game).
# Builds into target/perf-16x9 (target/perf-WxH for another CAP), never the loop's target/16x9 or
# target/20x11.
#
# Env: GAMES (3), SEED (4242), ITERS (1000), EVALUATOR (heuristic; or nn with MODEL=…onnx, tract),
# CONFIG (cfgs/gumbel16_f1000_gen.toml), CORE (7: the core it is pinned to, niced).
# Plan 069 §1 knobs (unset = exactly the plan-058 workload above):
#   CAP=20x11/7        build capacity (BOARD_SIZE_*), into target/perf-20x11;
#   SIZES=20x11/7      the --board-sizes list (default 16x9/6), or SIZES=mix for the loop's
#                      centred mix (centre 220, T 0.3, floor 0.2, aspect 1.5-2.8, areas 70..220,
#                      26 cells a player);
#   NEXT_DRIVE=1       --next-drive, as the loop generates;
#   BACKEND=inproc     --nn-backend: inproc = libtorch on the GPU (MODEL's .ts beside it; one lane,
#                      cuDNN deterministic, so one stream is bit-reproducible); null = no net, value
#                      0 and uniform priors at once, the encoder still runs. Either builds the
#                      `torch` feature (TORCH=1 builds it for a heuristic run too, so one binary
#                      serves every arm; a torch build always restarts the fixed getrandom stream);
#   STREAMS=1          --parallel-games (above 1 only reproducible with BACKEND=null or heuristic);
#   EVENTS=instructions:u   perf stat events (e.g. instructions:u,cycles:u,task-clock for IPC);
#   BIN=path           run this prebuilt botbowl-ui instead of building (a saved copy of a commit's
#                      binary, so the tree can change while a matrix runs).
# The loop's generation workload (plan 069 §1):
#   CAP=20x11/7 SIZES=20x11/7 NEXT_DRIVE=1 CONFIG=cfgs/chance_all_partial_gen.toml EVALUATOR=nn \
#     BACKEND=inproc MODEL=models/az_v7/bbnet_mix16x9v9_gen48.onnx scripts/perf_search_bench.sh OUT
# Besides the result line, OUT/TAG.counts has decisions, searches, descents and forwards (from the
# trajectories' telemetry and the NN_PROFILE line) and instructions per descent and per forward.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"; source "$REPO/scripts/lib/git.sh"
OUT="${1:?out dir}"; TAG="${2:-$(git rev-parse --short HEAD)$(dirty_suffix)}"
GAMES="${GAMES:-3}"; SEED="${SEED:-4242}"; ITERS="${ITERS:-1000}"; EVALUATOR="${EVALUATOR:-heuristic}"
CONFIG="${CONFIG:-cfgs/gumbel16_f1000_gen.toml}"; CORE="${CORE:-7}"
CAP="${CAP:-16x9/6}"; SIZES="${SIZES:-16x9/6}"; BACKEND="${BACKEND:-}"; STREAMS="${STREAMS:-1}"
EVENTS="${EVENTS:-instructions:u}"
mkdir -p "$OUT"
CW="${CAP%%x*}"; CR="${CAP#*x}"; CH="${CR%%/*}"; CP="${CR#*/}"
TORCH="${TORCH:-${BACKEND:+1}}"
FEATURES=deterministic_hash; [ -n "$TORCH" ] && FEATURES="$FEATURES,torch"
export BOARD_SIZE_W="$CW" BOARD_SIZE_H="$CH" BOARD_PLAYERS="$CP" CARGO_TARGET_DIR="$REPO/target/perf-${CW}x${CH}"
if [ -z "${BIN:-}" ]; then
    cargo build --release -j "${BUILD_JOBS:-3}" -p botbowl-ui --features "$FEATURES" >> "$OUT/build.log" 2>&1 \
        || { echo "build failed, see $OUT/build.log" >&2; exit 1; }
    BIN="$CARGO_TARGET_DIR/release/botbowl-ui"
fi
SHIM="$CARGO_TARGET_DIR/fixed_getrandom.so"
[ "$SHIM" -nt scripts/fixed_getrandom.c ] || gcc -O2 -shared -fPIC -o "$SHIM" scripts/fixed_getrandom.c
NN_ARGS=(); [ "$EVALUATOR" = nn ] && NN_ARGS=(--model "${MODEL:?MODEL=net.onnx for EVALUATOR=nn}")
[ -n "$BACKEND" ] && NN_ARGS+=(--nn-backend "$BACKEND")
if [ "$SIZES" = mix ]; then
    SIZE_ARGS=(--size-centre 220 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8
               --size-max-area 220 --size-min-area 70 --cells-per-player 26)
else
    SIZE_ARGS=(--board-sizes "$SIZES")
fi
[ -n "${NEXT_DRIVE:-}" ] && SIZE_ARGS+=(--next-drive)
# A torch build loads libtorch, which may draw on getrandom itself: restart the fixed stream on
# every call so std gets the same RandomState keys either way (plan 066 gate C).
[ -n "$TORCH" ] && export FIXRAND_PER_CALL=1 BLOOD_NN_DETERMINISTIC=1 BLOOD_NN_LANES="${BLOOD_NN_LANES:-1}"
LD_PRELOAD="$SHIM" BLOOD_NN_PROFILE=1 taskset -c "$CORE" nice -n 19 perf stat -x, -e "$EVENTS" -o "$OUT/$TAG.stat" \
    "$BIN" dataset --mode random-start "${SIZE_ARGS[@]}" \
    --games "$GAMES" --seed "$SEED" --mcts-iters "$ITERS" --bot-config "$CONFIG" --parallel-games "$STREAMS" \
    --evaluator "$EVALUATOR" "${NN_ARGS[@]}" --out "$OUT/$TAG.jsonl" --truncate > "$OUT/$TAG.out" 2> "$OUT/$TAG.log"
INS=$(grep instructions "$OUT/$TAG.stat" | cut -d, -f1)
HASH=$(python3 - "$OUT/$TAG.jsonl" <<'PY'
import hashlib, json, sys
SETS = ("skills", "used_skills", "simple")
def leaves_sorted(x):
    if isinstance(x, list):
        return sorted(x) if all(isinstance(v, str) for v in x) else [leaves_sorted(v) for v in x]
    return x
def canon(x):
    if isinstance(x, dict):
        return {k: sorted(v) if k in SETS and isinstance(v, list) else leaves_sorted(v) if k == "data" else canon(v)
                for k, v in x.items()}
    return [canon(v) for v in x] if isinstance(x, list) else x
h = hashlib.md5()
for line in open(sys.argv[1]):
    d = json.loads(line); d.pop("meta", None)
    for s in d.get("samples", []):
        s.pop("tree", None)  # plan 060's tree statistics describe the search, they are not the game
    h.update(json.dumps(canon(d), sort_keys=True).encode())
print(h.hexdigest()[:12])
PY
)
python3 - "$OUT/$TAG.jsonl" "$OUT/$TAG.out" "$INS" > "$OUT/$TAG.counts" <<'PY'
import json, re, sys
decisions = searches = descents = 0
boards = {}
for line in open(sys.argv[1]):
    d = json.loads(line)
    decisions += len(d.get("samples", []))
    m = d.get("meta", {}); x = m.get("extra", {}) or {}
    searches += int(x.get("searches", 0)); descents += int(x.get("iterations", 0))
    b = m.get("board_dims"); b = f"{b['width'] - 2}x{b['height'] - 2}/{b.get('team_size')}" if isinstance(b, dict) and "width" in b else str(b)
    boards[b] = boards.get(b, 0) + 1
prof = {}
for line in open(sys.argv[2]):
    if line.startswith("NN_PROFILE forwards="):
        prof = dict(kv.split("=", 1) for kv in line.split()[1:])
ins = int(sys.argv[3]); fw = int(prof.get("forwards", 0))
print("boards " + " ".join(f"{k}:{v}" for k, v in sorted(boards.items())))
print(f"records={sum(boards.values())} decisions={decisions} searches={searches} descents={descents} "
      f"forwards={fw} memo_calls={prof.get('memo_calls', 0)} memo_hits={prof.get('memo_hits', 0)}")
print(f"instructions={ins} per_descent={ins / max(descents, 1):.0f} "
      f"per_forward={(ins / fw) if fw else 0:.0f} forwards_per_descent={fw / max(descents, 1):.3f} "
      f"forwards_per_search={fw / max(searches, 1):.1f}")
PY
echo "$TAG instructions=$INS trajectories=$HASH $(tail -1 "$OUT/$TAG.counts")" | tee -a "$OUT/results.txt"
