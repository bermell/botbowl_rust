#!/usr/bin/env bash
# exp080 / plan 069 §1: instructions per descent and per forward at the loop's generation settings
# (20x11/7 capacity, C1 preset, 1000 descents, --next-drive, random-start), three evaluators:
#   heur    the scripted heuristic (no net, formation setups);
#   null    --nn-backend null: the NN code path with no network (value 0, uniform priors; the
#           encoder runs) — search + encoder CPU without inference;
#   inproc  gen48 through libtorch on the GPU, one stream, one lane, cuDNN deterministic; counted
#           only after the net is loaded (AFTER_LOAD), user and kernel.
# Boards: 20x11/7, 16x9/6 and the loop's centred mix. heur/null start-up is ~1.5G instructions
# (the GAMES=0 arms), small and stable enough to subtract.
#
#   scripts/exp080_cpu_matrix.sh OUT_DIR [ARM_FILTER_REGEX]      (BIN=, MODEL=, CORE= pass through)
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; FILTER="${2:-.}"
MODEL="${MODEL:-$REPO/models/az_v7/bbnet_mix16x9v9_gen48.onnx}"
COMMON=(TORCH=1 CAP=20x11/7 NEXT_DRIVE=1 CONFIG=cfgs/chance_all_partial_gen.toml
        EVENTS=instructions:u,cycles:u,task-clock,instructions:k,cycles:k MODEL="$MODEL")
run() {  # tag sizes evaluator backend games [extra env...]
    local tag="$1" sizes="$2" ev="$3" be="$4" games="$5"; shift 5
    [[ "$tag" =~ $FILTER ]] || return 0
    env "${COMMON[@]}" SIZES="$sizes" EVALUATOR="$ev" BACKEND="$be" GAMES="$games" "$@" \
        scripts/perf_search_bench.sh "$OUT" "$tag"
}
run heur_0 20x11/7 heuristic "" 0
run null_0 20x11/7 nn null 0
for b in 20x11/7 16x9/6 mix; do
    t="${b%%/*}"
    run "heur_$t" "$b" heuristic "" 20
    run "null_$t" "$b" nn null 20
    run "inproc_$t" "$b" nn inproc 3 AFTER_LOAD=1
done
