#!/usr/bin/env bash
# Plan 029 stages 1 and 2 — does more data make better nets?
#
# Three arms differing ONLY in how many generations of corpus they train on:
#   D1  gen07              117k samples   pure `nn` regime
#   D3  gen05-gen07        351k samples   pure `nn` regime
#   D7  gen01-gen07        818k samples   4 nn-value + 3 nn
#
# Everything else is held equal, and each of those is a trap plan 029 identified:
#
#  - IDENTICAL INIT. One BBNet is materialised once (`--epochs 0`) and `--init`ed
#    into every arm at --lr 1e-3, which is bit-exact from-scratch training rather
#    than a warm start (--init loads weights only). Not warm-started from gen03:
#    the champion already encodes gen00-02, so the k=1 arm would arrive knowing
#    what it is supposed to lack, biasing hard toward "data does nothing".
#    The `epoch -1 (warm-start baseline)` line every arm prints is a free
#    assertion that the init really was shared — checked below before any games.
#
#  - FIXED STEPS, NOT EPOCHS. 110,000 steps for every arm (production's current
#    count: gen07 ran 10 epochs x 10,957 batches). At fixed *epochs* the 818k
#    pool would take 7x the gradient updates of the 117k pool, so "more data
#    wins" would be indistinguishable from "more updates wins".
#
#  - FIXED VALIDATION CADENCE. Every 2,500 steps, so all three arms get 44
#    checkpoints. Per-epoch validation would hand D1 thirty candidates and D7
#    four, putting a confound inside best-val restore itself.
#
#  - ONE COMMON HOLDOUT for all arms (gen05-07 val shards, already prepared).
#    Disjoint by construction: arms train on TRAIN_SHARDS 0,1,2,3,5,6 only, the
#    holdout is VAL_SHARDS 4,7.
#
#  - JUDGED BY GAMES. val_value has moved opposite to strength four times in this
#    project (plans 027, 028). Each arm plays D1, 120 games, paired Home/Away.
#
#   nohup scripts/exp_data_scaling.sh > /dev/null 2>&1 &
#   touch runs/exp-data/STOP     # exits at the next stage boundary

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4
export PATH="$HOME/.cargo/bin:$PATH"

RUN_DIR="$REPO/runs/loop14x7"
OUT="$REPO/runs/exp-data"
UI="$REPO/target/14x7/release/botbowl-ui"
PREPARE="$REPO/target/14x7/release/prepare"
PY="$REPO/train/.venv/bin/python"
SOCK=/tmp/bbnn-data.sock
HOLD="$RUN_DIR/gen07/prepared_val/dims_16x9"
STEPS="${STEPS:-110000}"
EVAL_EVERY="${EVAL_EVERY:-2500}"
GAMES="${GAMES:-120}"
PARALLEL="${PARALLEL:-6}"
INIT_SEED="${INIT_SEED:-20260906}"

mkdir -p "$OUT"
LOG="$OUT/exp.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
die() { log "FATAL: $*"; exit 1; }
stopped() { [ -e "$OUT/STOP" ]; }

log "=== plan 029 stages 1-2 start ==="
[ -d "$HOLD" ] || die "holdout missing: $HOLD"
log "holdout $HOLD ($(du -sh "$HOLD" | cut -f1))"

# ---- 0. reclaim disk from stale prepared dirs -------------------------------
for g in gen00 gen01; do
    if [ -d "$RUN_DIR/$g/prepared_train" ] || [ -d "$RUN_DIR/$g/prepared_val" ]; then
        rm -rf "$RUN_DIR/$g/prepared_train" "$RUN_DIR/$g/prepared_val"
        rm -f "$RUN_DIR/$g/.prepared"
        log "pruned stale prepared_* from $g (regenerable from its shards)"
    fi
done
log "disk free $(df -BG / | awk 'NR==2{print $4}')"

# ---- 1. the one shared initialisation ---------------------------------------
INIT="$OUT/arm_init.pt"
if [ ! -f "$INIT" ]; then
    "$PY" -m bbnn.train --data "$HOLD" --epochs 0 --seed "$INIT_SEED" \
        --out "$INIT" --device cpu >> "$LOG" 2>&1 || die "could not materialise the shared init"
    log "shared init written: $INIT (seed $INIT_SEED)"
fi

# ---- 2. prepare the pools ---------------------------------------------------
# name | generations
prep_for() {
    # NB: separate `local` statements. bash expands every argument to `local`
    # before running it, so `local a="$1" b="$a"` sees the OLD $a — which under
    # `set -u` is an unbound-variable error that returns non-zero from inside a
    # command substitution, where its stderr is invisible.
    local name="$1"
    local gens="$2"
    local dir="$OUT/prep_$name"
    local inputs="" g k
    if [ -d "$dir" ]; then echo "$dir"; return 0; fi
    for g in $gens; do for k in 0 1 2 3 5 6; do inputs="$inputs $RUN_DIR/$g/shard$k.jsonl"; done; done
    # shellcheck disable=SC2086
    "$PREPARE" --in $inputs --out "$dir" >> "$LOG" 2>&1 || return 1
    echo "$dir"
}

# D3 reuses the pool the loop already built for gen07 — WINDOW_GENS=3 at G=7 is
# exactly gen05+gen06+gen07, and the streaming writer is byte-identical to the
# buffering one that wrote it (verified against all seven .npy files).
D3_POOL="$RUN_DIR/gen07/prepared_train"
[ -d "$D3_POOL/dims_16x9" ] || die "expected gen07/prepared_train to exist for D3"

# ---- 3. train the arms ------------------------------------------------------
train_arm() {
    local name="$1" pool="$2"
    local pt="$OUT/$name.pt" onnx="$OUT/$name.onnx"
    [ -f "$onnx" ] && { log "$name already trained"; return 0; }
    local dims; dims=$(ls -d "$pool"/dims_* 2>/dev/null | head -1)
    [ -n "$dims" ] || { log "FATAL: no dims dir under $pool"; return 1; }
    log "$name: training on $dims for $STEPS steps, val every $EVAL_EVERY"
    local t0=$SECONDS
    if ! "$PY" -m bbnn.train --data "$dims" --val-data "$HOLD" \
            --init "$INIT" --lr 1e-3 --seed "$INIT_SEED" \
            --max-steps "$STEPS" --eval-every "$EVAL_EVERY" --select-on combined \
            --device auto --out "$pt" --onnx "$onnx" \
            > "$OUT/$name.train.log" 2>&1; then
        log "$name FAILED — see $name.train.log"; return 1
    fi
    log "$name trained ($(((SECONDS - t0) / 60)) min): $(grep 'restored best-val' "$OUT/$name.train.log" | tail -1)"
    log "$name baseline: $(grep 'warm-start baseline' "$OUT/$name.train.log" | tail -1)"
}

play() {
    local cand="$1" opp="$2" seed="$3" tag="$4"
    local rep="$OUT/$tag.json"
    [ -e "$rep" ] && { log "$tag already played"; return 0; }
    log "$tag: $cand vs $opp, $GAMES games x$PARALLEL at 1000 iters"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    if ! "$UI" eval --evaluator nn --model "$OUT/$cand.onnx" \
            --mcts-iters 1000 \
            --vs-evaluator nn --vs-model "$OUT/$opp.onnx" \
            --vs-games "$GAMES" --seed "$seed" \
            --skip-lectures --skip-fixed-rungs \
            --parallel-games "$PARALLEL" ${NN_ARGS:-} \
            --per-game-out "$OUT/$tag.games.jsonl" \
            --out "$rep" > "$OUT/$tag.log" 2>&1; then
        log "$tag FAILED — see $tag.log"; return 1
    fi
    log "$tag done ($(((SECONDS - t0) / 60)) min): $("$PY" "$REPO/scripts/eval_summary.py" "$rep" 2>/dev/null || echo see json)"
    "$PY" "$REPO/scripts/paired_summary.py" "$OUT/$tag.games.jsonl" >> "$LOG" 2>&1 || true
}

# ---- stage 1: the screen ----------------------------------------------------
stopped && { log "STOP before stage 1"; exit 0; }
D1_POOL=$(prep_for d1 "gen07" 2>>"$LOG") || die "prepare D1 failed (see log)"
log "D1 pool $D1_POOL"
train_arm d1 "$D1_POOL" || die "D1 training failed"
train_arm d3 "$D3_POOL" || die "D3 training failed"

# The init assertion: every arm must report the same pre-training val_value.
BASES=$(grep -h 'warm-start baseline' "$OUT"/d1.train.log "$OUT"/d3.train.log | grep -oE 'val_value [0-9.]+' | sort -u)
if [ "$(echo "$BASES" | wc -l)" -ne 1 ]; then
    die "arms did NOT share an initialisation — baselines differ: $(echo $BASES)"
fi
log "init assertion passed — all arms start at $BASES"

# sidecar for the matches; the registry resolves each client's own model path
NN_ARGS=""
if [ -f "$OUT/d1.pt" ]; then
    rm -f "$SOCK"
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda \
        --model "$OUT/d1.onnx" --max-models 4 --stats-every 600 >> "$OUT/nn_server.log" 2>&1 &
    NN_PID=$!
    i=0; while [ ! -S "$SOCK" ]; do
        i=$((i+1)); { [ "$i" -gt 120 ] || ! kill -0 "$NN_PID" 2>/dev/null; } && { log "WARN: sidecar down, using tract"; NN_PID=""; break; }
        sleep 1
    done
    [ -n "${NN_PID:-}" ] && { NN_ARGS="--nn-server $SOCK"; log "sidecar up (pid $NN_PID)"; }
fi
cleanup() { [ -n "${NN_PID:-}" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM

play d3 d1 96000000 "s1-d3-vs-d1" || true
log "=== stage 1 complete ==="

# ---- stage 2: the wide arm --------------------------------------------------
stopped && { log "STOP before stage 2"; exit 0; }
D7_POOL=$(prep_for d7 "gen01 gen02 gen03 gen04 gen05 gen06 gen07" 2>>"$LOG") || die "prepare D7 failed (see log)"
log "D7 pool $D7_POOL ($(du -sh "$D7_POOL" | cut -f1)), disk free $(df -BG / | awk 'NR==2{print $4}')"
train_arm d7 "$D7_POOL" || die "D7 training failed"
B7=$(grep 'warm-start baseline' "$OUT/d7.train.log" | grep -oE 'val_value [0-9.]+')
[ "$B7" = "$BASES" ] || log "WARN: D7 baseline $B7 differs from $BASES — initialisation not shared!"
play d7 d1 97000000 "s2-d7-vs-d1" || true
log "=== stage 2 complete — box left free, loop still stopped ==="
