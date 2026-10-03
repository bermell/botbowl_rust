#!/usr/bin/env bash
# exp059: the policy-target question at the new budget (plan 049) and plan 050's WDL step 1, run
# while the d1k loop is paused after gen04. Drives only (the user's rule for this phase).
#
# Every arm repeats gen04's own recipe with one change: the gen02-04 window of runs/loopmix16x9d1k
# (all at 1000 descents), warm from gen03, lr 2e-4, 3 epochs, the loop's value weighting. So gen04
# is the control and needs no retraining.
#
#   A. targets: cq tau 50, cq tau 20, Gumbel σ(q), Gumbel σ(q) with --gumbel-min-range 50, each
#      against gen04 on paired contested drives (SPRT 0.5:0.55, at most 800 per board). The
#      positions are re-screened with gen04 first, from the same 500 candidates per board.
#   B. WDL step 1 (plan 050): arms A (tanh, blend 0.5 = gen04's recipe), B (tanh, blend 1.0) and
#      C (WDL, blend 1.0), 3 seeds each, all scored on val shards prepared at blend 1.0 (the raw
#      drive outcome) by scripts/wdl_summary.py. Trainer only.
#   C. the loop's own gen04 benchmark (drives vs gen21) into runs/loopmix16x9d1k/gen04/drives,
#      with the loop's exact arguments, so the loop picks it up when it resumes.
#
# Resumable: finished steps leave markers or reports and are skipped.
#
#   scripts/exp059_targets_and_wdl.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp059"; mkdir -p "$OUT"
RUN="$REPO/runs/loopmix16x9d1k"; M="$REPO/models/az_v7"
GEN03="$M/bbnet_mix16x9d1k_gen03"; GEN04="$M/bbnet_mix16x9d1k_gen04"; GEN21="$M/bbnet_mix16x9_gen21"
CAND="$REPO/runs/plan051_proxy/positions"; CANDIDATES=500; PLAYOUTS=4; BAND=0.25:0.75
SPRT=0.5:0.55; CAP=800
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9" BLOOD_MCTS_BUDGET=visits
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp059.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"
"$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub already serves $HUB_URL (is the loop running?)"
cargo build --release -p botbowl-hub -p botbowl-worker -p botbowl-nn >> "$OUT/build.log" 2>&1 || die "build"

# The laptop is built at d37a1dc; since then nothing a worker's games touch has changed.
git -C "$REPO" diff --quiet d37a1dc HEAD -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts \
    && printf 'hub_commit = "%s"\nallow = ["d37a1dc"]\n' "$(git -C "$REPO" rev-parse --short HEAD)" > "$REPO/hub-allowed-commits.toml"

"$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
"$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$GEN04.onnx" --max-models 8 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
"$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 12 --mem-floor-mb 1536 \
    --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!

# job NAME DIR LABEL MODEL VS_MODEL SEED [args] — one drive job in the background, pid in J[NAME].
declare -A J
job() {
    local name="$1" dir="$2" label="$3" model="$4" vs="$5" seed="$6"; shift 6
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "$label" --evaluator nn --model "$model" \
        --mcts-iters 500 --games 30 --seed "$seed" --skip-fixed-rungs --vs-evaluator nn --vs-model "$vs" "$@" \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 &
    J[$name]=$!
}
collect() {
    local name
    for name in "$@"; do
        [ -n "${J[$name]:-}" ] || continue
        wait "${J[$name]}" || die "$name failed, see its eval.log"
        unset "J[$name]"
    done
}
drive_line() { "$PY" - "$1" <<'PY'
import json, re, sys
r = json.load(open(sys.argv[1]))
out = []
for row in r["ladder"]:
    m = re.search(r"drives\(([^)]*)\)", row["opponent"]); s = row.get("sprt") or {}
    out.append(f"{m.group(1) if m else row.get('board')}: {row['points']:.3f} ± {row.get('points_se', 0):.3f} "
               f"{s.get('verdict', '-')} after {s.get('pairs', '?')} pairs")
print(" | ".join(out))
PY
}

# ---- C. the loop's gen04 benchmark, and the gen04 screen, on the hub from the start ----------
SETS21="$REPO/cfgs/positions/contested_14x7.json,$REPO/cfgs/positions/contested_16x9.json"
job gen04 "$RUN/gen04/drives" "gen04 drives vs gen21" "$GEN04.onnx" "$GEN21.onnx" 0 \
    --positions "$SETS21" --sprt "$SPRT" --vs-games "$CAP"
POS="$OUT/positions"
if [ ! -e "$POS/.screened" ]; then
    job screen "$POS/screen" "re-screen positions with gen04" "$GEN04.onnx" "$GEN04.onnx" 59000 \
        --positions "$CAND/cand_14x7.json,$CAND/cand_16x9.json" --vs-games $((CANDIDATES * PLAYOUTS))
fi
status "on the hub: gen04 drives vs gen21 (the loop's benchmark), the gen04 re-screen"

# ---- training: prepare once per target, train every arm (GPU, alongside the hub) -------------
# The loop's window and split for gen04: train shards 0-3,5,6 and val shards 4,7 of gen02-04.
TRAIN_IN=""; VAL_IN=""
for g in 02 03 04; do
    for k in 0 1 2 3 5 6; do TRAIN_IN="$TRAIN_IN $RUN/gen$g/shard$k.jsonl"; done
    for k in 4 7; do VAL_IN="$VAL_IN $RUN/gen$g/shard$k.jsonl"; done
done
# prep NAME TARGET_ARGS: prepared_train and prepared_val under $OUT/prep/NAME.
prep() {
    local name="$1" args="$2" d="$OUT/prep/$1"
    [ -e "$d/.done" ] && return 0
    rm -rf "$d"; mkdir -p "$d"
    # shellcheck disable=SC2086
    "$PREPARE" --in $TRAIN_IN --out "$d/prepared_train" $args >> "$OUT/prepare.log" 2>&1 || die "prepare $name train"
    # shellcheck disable=SC2086
    "$PREPARE" --in $VAL_IN --out "$d/prepared_val" $args >> "$OUT/prepare.log" 2>&1 || die "prepare $name val"
    touch "$d/.done"
}
# train_arm OUT_PREFIX TRAIN_DIR VAL_DIR [extra train args] — gen04's training, one change at a time.
train_arm() {
    local out="$1" tdir="$2" vdir="$3"; shift 3
    [ -s "$out.pt" ] && [ -e "$out.done" ] && return 0
    "$PY" -m bbnn.train --data "$tdir" --val-data "$vdir" --epochs 3 --device cuda --init "$GEN03.pt" --lr 2e-4 \
        --select-on combined --eval-every 2500 --value-weight 0.25 --per-drive-value-weight "$@" \
        --out "$out.pt" > "$out.train.log" 2>&1 || die "train $(basename "$out")"
    touch "$out.done"
}

SECONDS=0
cd "$REPO/train" || die "cd train"
declare -A TARGET=(
    [tau50]="--policy-target cq --tau 50"
    [tau20]="--policy-target cq --tau 20"
    [gumbel]="--policy-target gumbel"
    [gumbel_mr50]="--policy-target gumbel --gumbel-min-range 50"
)
for arm in tau50 tau20 gumbel gumbel_mr50; do
    [ -s "$M/exp059_$arm.onnx" ] && continue
    prep "$arm" "${TARGET[$arm]} --value-blend 0.5"
    mkdir -p "$OUT/arms"
    train_arm "$OUT/arms/$arm" "$OUT/prep/$arm/prepared_train" "$OUT/prep/$arm/prepared_val" --onnx "$M/exp059_$arm.onnx"
    cp "$OUT/arms/$arm.pt" "$M/exp059_$arm.pt"   # the sidecar serves the .pt next to the .onnx
    status "trained $arm ($((SECONDS / 60)) min): $(grep -E '^restored' "$OUT/arms/$arm.train.log")"
    rm -rf "$OUT/prep/$arm"
done

# WDL step 1: blend 0.5 train (arm A, gen04's own data), blend 1.0 train (B, C), blend 1.0 val (all).
prep wdl_b05 "--policy-target cq --tau 100 --value-blend 0.5"
prep wdl_b10 "--policy-target cq --tau 100 --value-blend 1.0"
mkdir -p "$OUT/wdl"
VAL10="$OUT/prep/wdl_b10/prepared_val"
for s in 1 2 3; do
    train_arm "$OUT/wdl/A_s$s" "$OUT/prep/wdl_b05/prepared_train" "$VAL10" --seed "$s"
    train_arm "$OUT/wdl/B_s$s" "$OUT/prep/wdl_b10/prepared_train" "$VAL10" --seed "$s"
    train_arm "$OUT/wdl/C_s$s" "$OUT/prep/wdl_b10/prepared_train" "$VAL10" --seed "$s" --value-head wdl
done
status "WDL arms trained ($((SECONDS / 60)) min)"
if [ ! -s "$OUT/wdl/summary.txt" ]; then
    ARGS=(); for f in "$OUT"/wdl/[ABC]_s[0-9].pt; do ARGS+=("$(basename "$f" .pt)=$f"); done
    "$PY" "$REPO/scripts/wdl_summary.py" --val "$VAL10" "${ARGS[@]}" > "$OUT/wdl/summary.txt" 2>&1 || die "wdl summary"
fi
status "WDL step 1: $(sed -n '/^per arm/,/^$/p' "$OUT/wdl/summary.txt" | tail -n +2 | tr -s ' ' | paste -sd'|')"
rm -rf "$OUT/prep"
cd "$REPO" || die "cd"

# ---- A. targets vs gen04 on positions screened with gen04 ------------------------------------
collect screen
if [ ! -e "$POS/.screened" ]; then
    for b in 14x7 16x9; do
        "$PY" "$REPO/scripts/positions_screen.py" --set "$CAND/cand_$b.json" --games "$POS/screen/eval.games.jsonl" \
            --reference "$GEN04.onnx" --band "$BAND" --min-playouts "$PLAYOUTS" --name "contested_${b}_gen04" \
            --out "$POS/contested_${b}_gen04.json" >> "$STATUS" 2>&1 || die "screen $b"
    done
    touch "$POS/.screened"
fi
SETS04="$POS/contested_14x7_gen04.json,$POS/contested_16x9_gen04.json"
for arm in tau50 tau20 gumbel gumbel_mr50; do
    job "$arm" "$OUT/drives/$arm" "$arm vs gen04 (drives)" "$M/exp059_$arm.onnx" "$GEN04.onnx" 59100 \
        --positions "$SETS04" --sprt "$SPRT" --vs-games "$CAP"
done
status "on the hub: tau50, tau20, gumbel, gumbel_mr50 vs gen04 on drives"

collect gen04
status "gen04 vs gen21 (loop benchmark): $(drive_line "$RUN/gen04/drives/report.json")"
for arm in tau50 tau20 gumbel gumbel_mr50; do
    collect "$arm"
    status "$arm vs gen04: $(drive_line "$OUT/drives/$arm/report.json")"
done
status "done ($((SECONDS / 60)) min)"
