#!/usr/bin/env bash
# exp075 (plan 064): the adversarial probe suites on one or more nets, CPU only (tract), pinned to
# two cores and niced so a live loop on the same box is not disturbed.
#
#   scripts/exp075_probes.sh OUT_DIR SUITE NET.onnx [NET.onnx ...]
#   SUITE: skills | symmetry | dominance | lone-mover | stability | hillclimb | all
#   env: REPO_DATA (where runs/ and models/ live; default this checkout, else the main one),
#        CORPUS (default gen39 val shard 4), POSITIONS (contested 16x9 set), BOARD (16x9),
#        STATES, BUDGETS, STARTS, STEPS, CPUS (taskset list, default 6,7), PAR (threads, 2), SEED
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?out dir}"; SUITE="${2:?suite}"; shift 2
[ $# -ge 1 ] || { echo "need at least one net" >&2; exit 2; }
DATA="${REPO_DATA:-$REPO}"
[ -f "$DATA/runs/loopmix16x9g/positions/contested_16x9_gen04g.json" ] \
    || DATA="$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir | xargs dirname)"
CORPUS="${CORPUS:-$DATA/runs/loopmix16x9v9/gen39/shard4.jsonl}"
POSITIONS="${POSITIONS:-$DATA/runs/loopmix16x9g/positions/contested_16x9_gen04g.json}"
BOARD="${BOARD:-16x9}"; CPUS="${CPUS:-6,7}"; PAR="${PAR:-2}"; SEED="${SEED:-64000}"
UI="$("$REPO/scripts/probe_build.sh")"
mkdir -p "$OUT"
unset BLOOD_NN_SERVER
run() { # suite, extra args...
    local suite="$1"; shift
    for net in "${NETS[@]}"; do
        echo "== $suite $(basename "$net") $(date +%T)" >&2
        taskset -c "$CPUS" nice -n 19 "$UI" probe "$suite" --model "$net" --board "$BOARD" \
            --parallel "$PAR" --seed "$SEED" --out-dir "$OUT" "$@" 2> "$OUT/${suite}.$(basename "$net" .onnx).log"
    done
}
NETS=("$@")
case_suite() {
    case "$1" in
        skills)     run skills --corpus "$CORPUS" --positions "$POSITIONS" --states "${STATES:-150}" ;;
        symmetry)   run symmetry --corpus "$CORPUS" --positions "$POSITIONS" --states "${STATES:-300}" ;;
        dominance)  run dominance --corpus "$CORPUS" --positions "$POSITIONS" --states "${STATES:-300}" ;;
        lone-mover) run lone-mover --positions "$POSITIONS" --states "${STATES:-12}" --starts "${STARTS:-12}" \
                        --budgets "${BUDGETS:-250,1000}" ;;
        stability)  run stability --corpus "$CORPUS" --positions "$POSITIONS" --states "${STATES:-20}" \
                        --budgets "${BUDGETS:-250,1000,4000}" ;;
        hillclimb)  run hillclimb --positions "$POSITIONS" --states "${STATES:-60}" --steps "${STEPS:-80}" ;;
        *) echo "unknown suite $1" >&2; exit 2 ;;
    esac
}
if [ "$SUITE" = all ]; then
    for s in symmetry skills dominance hillclimb lone-mover stability; do case_suite "$s"; done
else
    case_suite "$SUITE"
fi
