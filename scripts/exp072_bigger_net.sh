#!/usr/bin/env bash
# exp072 (plan 063): a bigger net trained from scratch on the clean v9 corpora, against a control of
# today's shape trained the same way and against the loop's own gen31.
#
# Data: runs/loopmix16x9v9 gen10..gen31, MC-labelled shards (genG/mc/shardK.jsonl), cq tau 100 targets,
# value blend 1.0 (the MC label). Train = shards 0-3,5,6 of gen10-31; val (checkpoint selection) =
# shards 4,7 of gen10-30; test = shards 4,7 of gen31 alone (never used for selection). One prepare
# for all arms (~40 GB in $OUT/prepared_*; removed at the end unless KEEP_PREPARED=1).
#
# Arms (same data, schedule, epochs, seed):
#   C64x6   today's 64x6 from scratch — the control: "fresh training on all the data" without "bigger"
#   A96x8   96x8 trunk, today's heads
#   B96x8h  96x8 + value_conv 64, value_pool meanmax, value_hidden 128, policy_conv 96
# Recipe: batch 256, Adam, lr 1e-3 peak, 2000 warmup steps then cosine to 0, EPOCHS (10) passes,
# validation every 5000 steps, restore on val_policy + val_value, value weight 0.36, per-drive value
# weight, BatchNorm live (no --freeze-bn: that is for fine-tunes), 6 loader workers.
#
# Per arm, in phase order (each skipped when its output exists, so a relaunch resumes):
#   train    the GPU training above (an unfinished arm restarts from scratch)
#   bench    sidecar forward cost on the GPU (`nn_server.py --bench`, 11x18 canvas, batch 1..32)
#   absorb   scripts/absorb_probe.py on the gen31 test set, every arm vs the loop's gen31
#   vbench   scripts/value_bench.sh on runs/value_bench/v9_gen01_val.jsonl, paired vs gen31
#   drives   paired contested drives vs the loop's gen31, C1 search (cfgs/chance_all_partial.toml)
#            both sides at 1000 descents, both position sets, SPRT 0.5:0.55, at most 800 per board.
#            On the loop's hub when one answers on 127.0.0.1:13337, else on a hub of our own (13338)
#            with one sidecar and one local worker.
#
# The GPU belongs to the loop: nothing here touches it until the loop has exited (no
# `bash scripts/train_loop.sh` process and no CUDA process on the card). Stop it with
#   nohup scripts/stop_after.sh "gen31 train done" runs/loopmix16x9v9 > /dev/null 2>&1 &
# or STOP_LOOP=1 here, which touches its STOP file once gen31 has trained. The loop is relaunched
# only with RELAUNCH=1 (after the GPU phases, before the drives, which then share its hub).
#
#   scripts/exp072_bigger_net.sh                          # everything; status in runs/exp072/status.md
#   ARMS="A96x8" PHASES="train absorb" scripts/exp072_bigger_net.sh
# Env: OUT EPOCHS BATCH LR WARMUP EVAL_EVERY LOADER_WORKERS SEED ARMS PHASES STOP_LOOP RELAUNCH
#      KEEP_PREPARED STREAMS CAP PORT
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
OUT="${OUT:-$REPO/runs/exp072}"; case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac; mkdir -p "$OUT"
RUN="$REPO/runs/loopmix16x9v9"; M="$REPO/models/az_v7"; LOOP31="$M/bbnet_mix16x9v9_gen31"
FIRST_GEN=10; LAST_GEN=31; TRAIN_SHARDS="0 1 2 3 5 6"; VAL_SHARDS="4 7"
EPOCHS="${EPOCHS:-10}"; BATCH="${BATCH:-256}"; LR="${LR:-1e-3}"; WARMUP="${WARMUP:-2000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"; LOADER_WORKERS="${LOADER_WORKERS:-6}"; SEED="${SEED:-72}"
ARMS="${ARMS:-C64x6 A96x8 B96x8h}"; PHASES="${PHASES:-train bench absorb vbench drives}"
STOP_LOOP="${STOP_LOOP:-0}"; RELAUNCH="${RELAUNCH:-0}"; KEEP_PREPARED="${KEEP_PREPARED:-0}"
STREAMS="${STREAMS:-24}"; CAP="${CAP:-800}"; PORT="${PORT:-13338}"
POS="$REPO/runs/loopmix16x9g/positions"; SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
C1="$REPO/cfgs/chance_all_partial.toml"; VBENCH="$REPO/runs/value_bench/v9_gen01_val.jsonl"
TARGETS="--policy-target cq --tau 100 --value-blend 1.0"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
# This checkout's trainer, whatever the venv's editable install points at.
export PYTHONPATH="$REPO/train/src${PYTHONPATH:+:$PYTHONPATH}"
PREPARE="$CARGO_TARGET_DIR/release/prepare"; HUB="$CARGO_TARGET_DIR/release/botbowl-hub"
WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"; PY="$REPO/train/.venv/bin/python"
TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp072.sock
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
die() { status "FATAL: $*"; exit 1; }
has() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

arch_of() {   # the bbnn.train shape flags of arm $1
    case "$1" in
        C64x6)  echo "--width 64 --blocks 6" ;;
        A96x8)  echo "--width 96 --blocks 8" ;;
        B96x8h) echo "--width 96 --blocks 8 --value-conv 64 --value-pool meanmax --value-hidden 128 --policy-conv 96" ;;
        *) return 1 ;;
    esac
}
for a in $ARMS; do arch_of "$a" > /dev/null || die "unknown arm $a"; done

HUB_PID="" NN_PID="" WORKER_PID=""
cleanup() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; rm -f "$SOCK"; }
trap cleanup EXIT INT TERM

git diff --quiet && git diff --cached --quiet || die "dirty tree: commit first (the nets are stamped with this commit)"
status "start: commit $(git rev-parse --short HEAD), arms $ARMS, phases $PHASES; epochs $EPOCHS batch $BATCH lr $LR warmup $WARMUP eval-every $EVAL_EVERY workers $LOADER_WORKERS seed $SEED"

# ---- 0. the data must be complete: gen31 trained means gen31 is MC-labelled and its net exists --------
until [ -e "$RUN/gen31/.trained" ] && [ -f "$LOOP31.pt" ] && [ -f "$LOOP31.onnx" ]; do sleep 120; done
if [ "$STOP_LOOP" = 1 ] && [ ! -e "$OUT/.loop_stopped" ]; then
    touch "$RUN/STOP"; status "STOP placed in $RUN (STOP_LOOP=1)"
fi

shards() {   # $1 = train|val|test: MC-labelled shard files
    local g k ks gens out=""
    case "$1" in
        train) gens=$(seq "$FIRST_GEN" "$LAST_GEN"); ks="$TRAIN_SHARDS" ;;
        val)   gens=$(seq "$FIRST_GEN" $((LAST_GEN - 1))); ks="$VAL_SHARDS" ;;
        test)  gens="$LAST_GEN"; ks="$VAL_SHARDS" ;;
    esac
    for g in $gens; do for k in $ks; do
        f="$RUN/$(printf 'gen%02d' "$g")/mc/shard$k.jsonl"
        [ -s "$f" ] || { echo "missing $f" >&2; return 1; }
        out="$out $f"
    done; done
    echo "$out"
}

# ---- 1. prepare (CPU; may run while the loop still uses the GPU) --------------------------------------
NEED_PREP=0; for p in train absorb; do has "$p" && NEED_PREP=1; done
if [ "$NEED_PREP" = 1 ] && [ ! -e "$OUT/.prepared" ]; then
    cargo build --release -p botbowl-nn --bin prepare >> "$OUT/build.log" 2>&1 || die "build prepare"
    SECONDS=0
    for part in train val test; do
        IN=$(shards "$part") || die "corpus incomplete for $part"
        rm -rf "$OUT/prepared_$part"
        # shellcheck disable=SC2086
        "$PREPARE" --in $IN --out "$OUT/prepared_$part" $TARGETS >> "$OUT/prepare.log" 2>&1 || die "prepare $part — see prepare.log"
    done
    touch "$OUT/.prepared"
    status "prepared ($((SECONDS / 60)) min): train $(du -sh "$OUT/prepared_train" | cut -f1), val $(du -sh "$OUT/prepared_val" | cut -f1), test $(du -sh "$OUT/prepared_test" | cut -f1); disk free $(df -BG "$OUT" | awk 'NR==2{print $4}')"
fi

# ---- 2. wait for the GPU ------------------------------------------------------------------------------
loop_running() { ps -eo args= | grep -q '^bash scripts/train_loop\.s[h]'; }
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
trained_arms() { local a; for a in $ARMS; do [ -e "$OUT/$a/.trained" ] && echo "$a"; done; }
# has_rows FILE PREFIX NAME...: FILE has a line starting with PREFIX NAME and a space for every NAME
# (absorb_probe's table rows, value_bench's "== name" headers) — a failed or a smaller run does not.
has_rows() {
    local f="$1" pre="$2" n; shift 2
    [ -s "$f" ] || return 1
    for n in "$@"; do grep -q "^$pre$n " "$f" || return 1; done
}
absorb_done() { has_rows "$OUT/absorb.txt" "" loop31 $(trained_arms); }
vbench_done() { has_rows "$OUT/value_bench/summary.txt" "== " loop31 $(trained_arms); }
# Whether a requested GPU phase still has work. A resume whose GPU phases are all done (say, only the
# drives are left after RELAUNCH=1 brought the loop back) must not wait for the loop to exit again.
gpu_work_left() {
    local a
    if has train; then for a in $ARMS; do [ -e "$OUT/$a/.trained" ] || return 0; done; fi
    if has bench; then for a in loop31 $(trained_arms); do [ -s "$OUT/bench.$a.txt" ] || return 0; done; fi
    [ -n "$(trained_arms)" ] || return 1
    if has absorb && ! absorb_done; then return 0; fi
    if has vbench && ! vbench_done; then return 0; fi
    return 1
}
if gpu_work_left; then
    n=0
    while loop_running || gpu_busy; do
        [ $((n % 30)) -eq 0 ] && status "waiting for the loop to exit and the GPU to be free (stop it: scripts/stop_after.sh or STOP_LOOP=1)"
        n=$((n + 1)); sleep 60
    done
    touch "$OUT/.loop_stopped"
    status "GPU free"
fi

# ---- 3. train -----------------------------------------------------------------------------------------
if has train; then
    for a in $ARMS; do
        A="$OUT/$a"
        [ -e "$A/.trained" ] && { status "$a: trained already"; continue; }
        rm -rf "$A"; mkdir -p "$A"
        status "$a: training from scratch ($(arch_of "$a"))"
        SECONDS=0
        # shellcheck disable=SC2046
        if "$PY" -m bbnn.train --data "$OUT/prepared_train" --val-data "$OUT/prepared_val" --device cuda \
                $(arch_of "$a") --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
                --lr-schedule cosine --warmup-steps "$WARMUP" --eval-every "$EVAL_EVERY" --select-on combined \
                --value-weight 0.36 --per-drive-value-weight --loader-workers "$LOADER_WORKERS" --seed "$SEED" \
                --out "$A/net.pt" --onnx "$A/net.onnx" --progress "$A/train.progress.json" > "$A/train.log" 2>&1; then
            touch "$A/.trained"
            status "$a: trained ($((SECONDS / 60)) min; $(grep -m1 '^model:' "$A/train.log")): $(grep 'restored best-val weights' "$A/train.log" | tail -1)"
            status "$a: last check $(grep -E '^step .*val_policy' "$A/train.log" | tail -1 | grep -oE 'val_policy [0-9.]+ +val_value [0-9.]+ +val_top1 [0-9.]+')"
        else
            status "$a: training FAILED — see $A/train.log"
        fi
    done
fi
NETS=(); for a in $ARMS; do [ -e "$OUT/$a/.trained" ] && NETS+=("$a"); done

# ---- 4. inference cost on the GPU ---------------------------------------------------------------------
# The sidecar's own timing: eager and CUDA-graph per-batch times at batch 1..32 on the 11x18 canvas
# (the 16x9 board, the loop's largest), and the fit t(b) = F + g*b. g is what generation pays per sample.
if has bench; then
    for n in loop31 "${NETS[@]}"; do
        f="$OUT/bench.$n.txt"; [ -s "$f" ] && continue
        net="$OUT/$n/net.onnx"; [ "$n" = loop31 ] && net="$LOOP31.onnx"
        # Through a .tmp, so a failed bench is retried on a resume instead of passing for done.
        if "$PY" scripts/nn_server.py --bench --device cuda --model "$net" --warm-sizes 11x18 --max-batch 32 > "$f.tmp" 2>&1; then
            mv "$f.tmp" "$f"
        else
            status "WARN: bench $n failed — see $f.tmp"
        fi
    done
    status "inference cost (sidecar --bench, 11x18, graph path):"
    for n in loop31 "${NETS[@]}"; do
        [ -s "$OUT/bench.$n.txt" ] || continue
        status "  $n: $(grep -E 'fit 11x18 graph' "$OUT/bench.$n.txt" | sed 's/.*graph: //') | $(grep -E 'batch +(12|24):' "$OUT/bench.$n.txt" | sed -E 's/.*batch +([0-9]+):.*graph +([0-9]+) us\/batch \( *([0-9.]+) us\/sample\).*/b\1 \3 us\/sample/' | paste -sd' ')"
    done
fi

# ---- 5. absorption on the gen31 test set --------------------------------------------------------------
if has absorb && ! absorb_done && [ "${#NETS[@]}" -gt 0 ]; then
    ARGS=("loop31=$LOOP31.pt"); for n in "${NETS[@]}"; do ARGS+=("$n=$OUT/$n/net.pt"); done
    "$PY" scripts/absorb_probe.py --summary --device cuda --val "$OUT/prepared_test" "${ARGS[@]}" > "$OUT/absorb.txt" 2>&1 \
        || status "WARN: absorption probe failed — see $OUT/absorb.txt"
    status "test set (gen31 shards 4,7; cq-100 target, MC value label), every arm vs the loop's gen31:"
    grep -E '^(net|loop31|C64|A96|B96|ABSORB)' "$OUT/absorb.txt" | tee -a "$STATUS"
fi

# ---- 6. value bench -----------------------------------------------------------------------------------
if has vbench && ! vbench_done && [ "${#NETS[@]}" -gt 0 ]; then
    ARGS=("loop31=$LOOP31.onnx"); for n in "${NETS[@]}"; do ARGS+=("$n=$OUT/$n/net.onnx"); done
    scripts/value_bench.sh "$VBENCH" "$OUT/value_bench" "${ARGS[@]}" > "$OUT/value_bench.log" 2>&1 \
        || status "WARN: value bench failed — see $OUT/value_bench.log"
    status "value bench (paired vs loop31):"; grep -E '^VALUE_BENCH' "$OUT/value_bench/summary.txt" | tee -a "$STATUS"
fi

if [ "$KEEP_PREPARED" != 1 ] && [ -e "$OUT/.prepared" ] && has train && has absorb; then
    all=1; for a in $ARMS; do [ -e "$OUT/$a/.trained" ] || all=0; done
    # Only once the probe has every arm's row: a failed probe leaves its traceback in absorb.txt.
    if [ "$all" = 1 ] && absorb_done; then
        rm -rf "$OUT"/prepared_*; rm -f "$OUT/.prepared"; status "prepared dirs removed (KEEP_PREPARED=1 keeps them)"
    fi
fi

if [ "$RELAUNCH" = 1 ] && ! loop_running; then
    rm -f "$RUN/STOP"
    EXTRA_ALLOW=ae721b7 setsid nohup scripts/launch_plan058.sh > /dev/null 2>&1 < /dev/null &
    status "loop relaunched (RELAUNCH=1)"; sleep 120
fi

# ---- 7. drives vs the loop's gen31 --------------------------------------------------------------------
if has drives && [ "${#NETS[@]}" -gt 0 ]; then
    # A sidecar resolves a worker's content-addressed ONNX against the directory of its --model, and
    # every sidecar here (ours or the loop's) is started on a net in models/az_v7: the arms go there.
    for n in "${NETS[@]}"; do
        for ext in pt onnx; do
            cmp -s "$OUT/$n/net.$ext" "$M/exp072_$n.$ext" || cp "$OUT/$n/net.$ext" "$M/exp072_$n.$ext"
        done
    done
    if "$HUB" status --hub http://127.0.0.1:13337 --token-file "$TOK" > /dev/null 2>&1; then
        HUB_URL=http://127.0.0.1:13337
        status "drives on the loop's hub ($HUB_URL); its sidecars serve the arm nets (keep --max-models in mind)"
    else
        HUB_URL="http://127.0.0.1:$PORT"
        cargo build --release -p botbowl-ui -p botbowl-hub -p botbowl-worker >> "$OUT/build.log" 2>&1 || die "build hub/worker"
        "$HUB" status --hub "$HUB_URL" --token-file "$TOK" > /dev/null 2>&1 && die "a hub already serves $HUB_URL"
        "$HUB" serve --bind "127.0.0.1:$PORT" --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
        for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" > /dev/null 2>&1 && break; sleep 1; done
        "$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$LOOP31.onnx" --max-models 4 \
            --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
        for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
        "$WORKER" --hub "ws://127.0.0.1:$PORT/ws" --token-file "$TOK" --name local --parallel-games "$STREAMS" \
            --mem-floor-mb 1536 --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
        status "drives on a hub of our own ($HUB_URL, $STREAMS local streams, one sidecar)"
    fi
    for n in "${NETS[@]}"; do
        dir="$OUT/$n/drives"
        [ -s "$dir/report.json" ] && { status "drives $n: done already: $("$PY" scripts/eval_summary.py "$dir/report.json")"; continue; }
        mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
        SECONDS=0
        "$HUB" job eval --hub "$HUB_URL" --token-file "$TOK" --label "exp072 $n vs gen31 (drives, C1)" \
            --evaluator nn --model "$M/exp072_$n.onnx" --vs-evaluator nn --vs-model "$LOOP31.onnx" \
            --mcts-iters 1000 --bot-config "$C1" --vs-config "$C1" --seed 0 --skip-fixed-rungs \
            --positions "$SETS" --sprt 0.5:0.55 --vs-games "$CAP" \
            --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
            || { status "drives $n FAILED — see $dir/eval.log"; continue; }
        status "drives $n vs gen31 ($((SECONDS / 60)) min): $("$PY" scripts/eval_summary.py "$dir/report.json")"
    done
fi
status "done"
