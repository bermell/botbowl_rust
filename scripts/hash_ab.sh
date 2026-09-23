#!/usr/bin/env bash
# Production-shaped A/B for a `GameState::hash` change (plan 044).
#
# The microbenchmark in `botbowl-mcts/tests/hash_bench.rs` isolates the search; this runs whole
# ladder games through `botbowl-ui eval`, which is the shape production actually has — game
# lengths vary, the bot is rebuilt per rung, and the telemetry lands in `report.json` exactly as
# it would in a real run.
#
# Two arms alternate so that machine drift hits both equally, and each arm is repeated so the
# spread is visible rather than assumed.
#
#   scripts/hash_ab.sh <baseline-worktree> <out-dir> [rounds] [games]
#
# The baseline worktree is a checkout of the commit to compare against, built with its own
# CARGO_TARGET_DIR so the two binaries never overwrite each other.
set -euo pipefail

BASE_WT=${1:?usage: hash_ab.sh <baseline-worktree> <out-dir> [rounds] [games]}
OUT=${2:?usage: hash_ab.sh <baseline-worktree> <out-dir> [rounds] [games]}
ROUNDS=${3:-3}
GAMES=${4:-12}

HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"

echo "building both arms..."
( cd "$HERE" && cargo build --release -p botbowl-ui >/dev/null )
( cd "$BASE_WT" && CARGO_TARGET_DIR="$BASE_WT-target" cargo build --release -p botbowl-ui >/dev/null )

run_arm() {
  local arm=$1 bin=$2 round=$3
  local report="$OUT/$arm-r$round.json"
  # Heuristic evaluator on purpose: with an NN, tract forward passes dominate the wall clock and
  # would mask whatever the hash does. Fixed seed per round so both arms face the same games.
  /usr/bin/time -p "$bin" eval \
      --skip-lectures --rungs scripted \
      --games "$GAMES" --mcts-iters 600 --mcts-workers 1 \
      --seed "$((1000 + round))" \
      --out "$report" --per-game-out "$OUT/$arm-r$round.games.jsonl" \
      >/dev/null 2>"$OUT/$arm-r$round.time"
  local secs
  secs=$(awk '/^real/ {print $2}' "$OUT/$arm-r$round.time")
  python3 - "$report" "$arm" "$round" "$secs" <<'PY'
import json, sys
report, arm, rnd, secs = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
t = json.load(open(report)).get("telemetry")
if not t:
    print(f"{arm} r{rnd}: no telemetry in {report}"); raise SystemExit
r = t["recombination"]
print(
    f"EVAL_AB arm={arm} round={rnd} secs={secs:.2f} searches={t['searches']} "
    f"nodes={r['misses']} probes={r['probes']} "
    f"eq_per_probe={r['eq_checks']/max(r['probes'],1):.4f} "
    f"eq_rejects={r['eq_rejects']} "
    f"hit_rate={r['hits']/max(r['probes'],1):.4f} "
    f"nodes_per_sec={r['misses']/secs:.0f}"
)
PY
}

for r in $(seq 1 "$ROUNDS"); do
  # Alternate within each round, so a machine that gets busier over time slows both arms equally.
  run_arm old "$BASE_WT-target/release/botbowl-ui" "$r"
  run_arm new "$HERE/target/release/botbowl-ui" "$r"
done | tee "$OUT/summary.txt"

echo
echo "wrote $OUT/summary.txt"
