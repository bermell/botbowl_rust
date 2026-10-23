#!/usr/bin/env bash
# Plan 069 item 6: how `prepare` and the trainer's data loader scale with the corpus — a report, not
# a fix. A generation window on the cluster may be 10-100x today's; this replicates the given shards
# R times (the same files named R times on `prepare --in`, so the page cache serves the repeats:
# the numbers are CPU and memory, not file-system reads) and records, per R:
#   prepare: wall seconds, max RSS, output bytes and samples (train only: the val split is small);
#   loader:  open time and RSS of `bbnn.data.open_prepared` (what is read into memory vs mmapped),
#            and samples/s over LOADER_BATCHES shuffled batches of 256, with 0 and LOADER_WORKERS
#            worker processes (bbnn.train --loader-workers), CPU only.
#
#   scripts/prepare_scale.sh OUT_DIR "SHARD.jsonl ..." ["1 3 10"]
#
# Output is deleted after each R (KEEP=1 keeps it). Stops before an R whose estimated output (R x
# the R=1 output) would leave less than MIN_FREE_GB (70) free. Env: CARGO_TARGET_DIR (default
# target/hubscale), BUILD_JOBS (3), PY (the trainer's python; default the main checkout's venv),
# LOADER_BATCHES (300), LOADER_WORKERS (4), PREPARE_ARGS (default the loop's: --policy-target cq
# --tau 100 --value-blend 1.0).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${1:?out dir}"; SHARDS="${2:?shards}"; FACTORS="${3:-1 3 10}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
export BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/target/hubscale}"
MAIN="$(git -C "$REPO" worktree list --porcelain | awk 'NR==1{print $2}')"
PY="${PY:-$MAIN/train/.venv/bin/python}"
export PYTHONPATH="$REPO/train/src${PYTHONPATH:+:$PYTHONPATH}"
PREPARE_ARGS="${PREPARE_ARGS:---policy-target cq --tau 100 --value-blend 1.0}"
MIN_FREE_GB="${MIN_FREE_GB:-70}"
cargo build --release -j "${BUILD_JOBS:-3}" -p botbowl-nn --bin prepare
PREPARE="$CARGO_TARGET_DIR/release/prepare"
free_gb() { df -BG "$OUT" | awk 'NR==2{gsub(/G/,"",$4); print $4}'; }
REPORT="$OUT/prepare_scale.md"
{
    echo "# prepare / loader scale ($(date '+%F %T'), commit $(git rev-parse --short HEAD))"
    echo
    echo "input: $SHARDS ($(cat $SHARDS | wc -l) lines, $(du -ch $SHARDS | tail -1 | cut -f1)); prepare $PREPARE_ARGS"
    echo
    echo "| R | lines | prepare s | prepare max RSS MB | out GB | samples | open s | open RSS MB | loader samples/s (0 workers) | loader samples/s (${LOADER_WORKERS:-4} workers) |"
    echo "|---|---|---|---|---|---|---|---|---|---|"
} > "$REPORT"
ONE_GB=0
for R in $FACTORS; do
    need=$((ONE_GB * R + MIN_FREE_GB))
    if [ "$ONE_GB" -gt 0 ] && [ "$(free_gb)" -lt "$need" ]; then
        echo "stopping before R=$R: $(free_gb) GB free, ~$((ONE_GB * R)) GB needed + $MIN_FREE_GB kept" | tee -a "$REPORT"
        break
    fi
    IN=(); for _ in $(seq "$R"); do for s in $SHARDS; do IN+=("$s"); done; done
    D="$OUT/r$R"; rm -rf "$D"
    # shellcheck disable=SC2086
    /usr/bin/time -v -o "$OUT/r$R.time" "$PREPARE" --in "${IN[@]}" --out "$D" $PREPARE_ARGS > "$OUT/r$R.prepare.log" 2>&1
    WALL=$(awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$OUT/r$R.time")
    RSS=$(awk -F': ' '/Maximum resident set size/{printf "%d", $2/1024}' "$OUT/r$R.time")
    BYTES=$(du -sb "$D" | cut -f1)
    [ "$ONE_GB" -eq 0 ] && ONE_GB=$(((BYTES / R) / 1000000000 + 1))
    LOADER=$("$PY" - "$D" "${LOADER_BATCHES:-300}" "${LOADER_WORKERS:-4}" <<'PY'
import resource, sys, time
from bbnn.data import make_loader, open_prepared
d, batches, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = time.time()
ds = open_prepared(d)
open_s = time.time() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
rates = []
for w in (0, workers):
    loader = make_loader(ds, 256, shuffle=True, workers=w)
    it = iter(loader)
    next(it)  # worker start-up and the first page faults are not throughput
    t, n = time.time(), 0
    for _ in range(batches):
        try:
            b = next(it)
        except StopIteration:
            break
        n += len(b["value"]) if isinstance(b, dict) else len(b[0])
    rates.append(n / (time.time() - t))
    del it, loader
print(f"{len(ds)} {open_s:.2f} {rss:.0f} {rates[0]:.0f} {rates[1]:.0f}")
PY
)
    read -r SAMPLES OPEN_S OPEN_RSS RATE0 RATEW <<< "$LOADER"
    echo "| $R | $(cat "${IN[@]}" | wc -l) | $WALL | $RSS | $(awk -v b="$BYTES" 'BEGIN{printf "%.2f", b/1e9}') | $SAMPLES | $OPEN_S | $OPEN_RSS | $RATE0 | $RATEW |" | tee -a "$REPORT"
    [ "${KEEP:-0}" = 1 ] || rm -rf "$D"
done
echo "wrote $REPORT"
