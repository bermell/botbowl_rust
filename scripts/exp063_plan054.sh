#!/usr/bin/env bash
# exp063: plan 054 E2/E4/E5 and the gate into E8 (the relaunched loop). Resumable: a finished
# match keeps its report.json, a finished arm its .pt.
#
#   A. On one hub, concurrently (both seats cfgs/gumbel16_f1000.toml at 1000 descents unless noted,
#      the gen04g contested sets, FIXED 300 pairs per board, no SPRT, --seed 63000 everywhere):
#        D1  plan054_w1_cq50_lr5e5  vs g_gen05     the candidate recipe
#        D2  plan054_w1_cq100_lr5e5 vs g_gen05     the lr change alone
#        D4  loop g_gen06           vs g_gen05     what the old recipe did in the same step
#        E5  g_gen05 policy_only@8  vs g_gen05     the search's edge over its own policy
#      Meanwhile on the GPU, E4 probes: cq50 at lr 5e-5 and 2e-5 with the restore fix (the
#      relaunch recipe), scored with scripts/absorb_probe.py.
#   B. Gate (pre-registered in plan 054 E2): an arm qualifies when its two-board mean is at least
#      0.5 + 2 SE and neither board is below 0.5 - 2 SE. Of the qualifiers, the higher mean wins.
#   C. Winner: scripts/launch_plan054.sh <winner> <its tau> starts runs/loopmix16x9g054, and D3
#      (plan054_w1_cq30_lr5e5 vs g_gen05) is submitted to the loop's hub if D1 read above 0.5.
#      No winner: E2b, the amplified test — d1k gen04 fine-tuned on all seven Gumbel generations
#      (cq100 at 2e-4, the old recipe, vs cq50 at 5e-5 with the restore fix), each against d1k
#      gen04 on the same drives — then stop for the user.
#
#   scripts/exp063_plan054.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/runs/exp063"; mkdir -p "$OUT"
M="$REPO/models/az_v7"; L="$REPO/runs/loopmix16x9g"; POS="$L/positions"
SETS="$POS/contested_14x7_gen04g.json,$POS/contested_16x9_gen04g.json"
GUMBEL="$REPO/cfgs/gumbel16_f1000.toml"; POLICY="$REPO/cfgs/policy_only.toml"
G05="$M/bbnet_mix16x9g_gen05.onnx"; G06="$M/bbnet_mix16x9g_gen06.onnx"; D1K="$M/bbnet_mix16x9d1k_gen04.onnx"
DRIVES=600   # per position set: 300 pairs
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
unset BLOOD_MCTS_BUDGET
HUB="$CARGO_TARGET_DIR/release/botbowl-hub"; WORKER="$CARGO_TARGET_DIR/release/botbowl-worker"
PREPARE="$CARGO_TARGET_DIR/release/prepare"
HUB_URL="http://127.0.0.1:13337"; TOK="$HOME/.config/botbowl/hub.token"; SOCK=/tmp/bbnn-exp063.sock
PY="$REPO/train/.venv/bin/python"
STATUS="$OUT/status.md"; status() { echo "[$(date '+%F %T')] $*" >> "$STATUS"; }; die() { status "FATAL: $*"; exit 1; }
HUB_PID="" NN_PID="" WORKER_PID=""
down() { for p in $WORKER_PID $NN_PID $HUB_PID; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done; WORKER_PID="" NN_PID="" HUB_PID=""; rm -f "$SOCK"; }
trap down EXIT INT TERM
git -C "$REPO" diff --quiet || die "dirty tree"
status "start: commit $(git -C "$REPO" rev-parse --short HEAD)"

up() {
    "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && die "a hub already serves $HUB_URL"
    local last; last=$(git -C "$REPO" log -1 --format=%h -- botbowl-engine botbowl-mcts botbowl-nn botbowl-play botbowl-worker botbowl-hub-proto recon_mcts)
    printf 'hub_commit = "%s"\nallow = [%s]\n' "$(git -C "$REPO" rev-parse --short HEAD)" \
        "$(git -C "$REPO" rev-list --abbrev-commit "$last"^..HEAD | sed 's/.*/"&"/' | paste -sd,)" > "$REPO/hub-allowed-commits.toml"
    "$HUB" serve --bind 0.0.0.0:13337 --token-file "$TOK" --run-dir "$OUT" >> "$OUT/hub.log" 2>&1 & HUB_PID=$!
    for _ in $(seq 30); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 1; done
    "$PY" "$REPO/scripts/nn_server.py" --socket "$SOCK" --device cuda --model "$G05" --max-models 8 \
        --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
    for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done; [ -S "$SOCK" ] || die "nn_server"
    "$WORKER" --hub ws://127.0.0.1:13337/ws --token-file "$TOK" --name local --parallel-games 12 --mem-floor-mb 1536 \
        --cache-dir "$OUT/worker-cache" --nn-server "$SOCK" >> "$OUT/worker.log" 2>&1 & WORKER_PID=$!
}

# match NAME LABEL CAND CAND_CFG CAND_ITERS OPP [HUB_URL]: paired drives, fixed 300 pairs per board.
match() {
    local name="$1" label="$2" cand="$3" ccfg="$4" citers="$5" opp="$6" url="${7:-$HUB_URL}" dir="$OUT/$1"
    [ -s "$dir/report.json" ] && return 0
    mkdir -p "$dir"; rm -f "$dir/eval.games.jsonl"
    "$HUB" job eval --hub "$url" --token-file "$TOK" --label "exp063 $label" \
        --evaluator nn --model "$cand" --bot-config "$ccfg" --mcts-iters "$citers" \
        --vs-evaluator nn --vs-model "$opp" --vs-config "$GUMBEL" --opponent-iters 1000 \
        --games 30 --seed 63000 --skip-fixed-rungs --positions "$SETS" --vs-games "$DRIVES" \
        --per-game-out "$dir/eval.games.jsonl" --out "$dir/report.json" --wait > "$dir/eval.log" 2>&1 \
        || { status "WARN: $name failed — see $dir/eval.log"; return 1; }
    status "$name ($label): $(summary "$dir/report.json")"
}
summary() {
    "$PY" - "$1" <<'PY'
import json, re, sys
r = json.load(open(sys.argv[1]))
print(" | ".join(f"{re.search(r'@(\S+)$', row['opponent']).group(1)} {row['points']:.3f} ± {row['points_se']:.3f} "
                 f"({row['games']} drives)" for row in r["ladder"]))
PY
}

# probe_arm NAME DATA VAL INIT LR [EXTRA...]: one fine-tune with the relaunch recipe's flags.
probe_arm() {
    local name="$1" data="$2" val="$3" init="$4" lr="$5"; shift 5
    [ -e "$OUT/arms/$name.done" ] && return 0   # the trainer persists its best .pt while it runs
    mkdir -p "$OUT/arms"
    (cd "$REPO/train" && "$PY" -m bbnn.train --data "$data" --val-data "$val" --epochs 3 --device cuda \
        --init "$init" --lr "$lr" --eval-every 1000 --value-weight 0.25 --per-drive-value-weight --seed 1 \
        --out "$OUT/arms/$name.pt" "$@" > "$OUT/arms/$name.log" 2>&1) || { status "WARN: arm $name failed"; return 1; }
    touch "$OUT/arms/$name.done"
    status "arm $name: $(grep -E '^restored' "$OUT/arms/$name.log")"
}

cargo build --release -p botbowl-hub -p botbowl-worker -p botbowl-nn >> "$OUT/build.log" 2>&1 || die "build"

# ---- A. the matches, and the E4 probes alongside ------------------------------------------------
SECONDS=0
up
match D1 "cq50@5e-5 vs g_gen05" "$M/plan054_w1_cq50_lr5e5.onnx" "$GUMBEL" 1000 "$G05" & P1=$!
match D2 "cq100@5e-5 vs g_gen05" "$M/plan054_w1_cq100_lr5e5.onnx" "$GUMBEL" 1000 "$G05" & P2=$!
match D4 "loop gen06 vs g_gen05" "$G06" "$GUMBEL" 1000 "$G05" & P4=$!
match E5 "g_gen05 policy-only vs g_gen05 Gumbel@1000" "$G05" "$POLICY" 8 "$G05" & P5=$!

G6="$L/gen06"; T6=""; for k in 0 1 2 3 5 6; do T6="$T6 $G6/shard$k.jsonl"; done
P="$OUT/prep"; mkdir -p "$P"
if [ ! -e "$P/.g06" ]; then
    # shellcheck disable=SC2086
    "$PREPARE" --in $T6 --out "$P/g06_train_cq50" --policy-target cq --tau 50 --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 \
        && "$PREPARE" --in "$G6/shard4.jsonl" "$G6/shard7.jsonl" --out "$P/g06_val_cq50" --policy-target cq --tau 50 \
            --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 && touch "$P/.g06" || die "prepare gen06"
fi
FIX=(--freeze-bn --init-candidate --eval-at 250,500 --select-on policy)
probe_arm e4_cq50_5e5_fix "$P/g06_train_cq50" "$P/g06_val_cq50" "$M/bbnet_mix16x9g_gen05.pt" 5e-5 "${FIX[@]}"
probe_arm e4_cq50_2e5_fix "$P/g06_train_cq50" "$P/g06_val_cq50" "$M/bbnet_mix16x9g_gen05.pt" 2e-5 "${FIX[@]}"
"$PY" "$REPO/scripts/absorb_probe.py" --summary --val "$P/g06_val_cq50" generator="$M/bbnet_mix16x9g_gen05.pt" \
    w1_cq50_5e5="$M/plan054_w1_cq50_lr5e5.pt" e4_cq50_5e5_fix="$OUT/arms/e4_cq50_5e5_fix.pt" \
    e4_cq50_2e5_fix="$OUT/arms/e4_cq50_2e5_fix.pt" > "$OUT/e4_probe.txt" 2>&1
status "E4 probes (gen06 held-out, cq50 target): $(grep '^ABSORB' "$OUT/e4_probe.txt" | sed 's/^ABSORB //' | paste -sd'|')"

wait $P1 $P2 $P4 $P5
status "phase A done ($((SECONDS / 60)) min)"

# ---- B. the gate --------------------------------------------------------------------------------
VERDICT=$("$PY" - "$OUT" 2>>"$STATUS" <<'PY'
import json, math, os, sys
out = sys.argv[1]
arms = {"D1": ("plan054_w1_cq50_lr5e5", 50), "D2": ("plan054_w1_cq100_lr5e5", 100)}
best = None
for d, (net, tau) in arms.items():
    f = os.path.join(out, d, "report.json")
    if not os.path.exists(f):
        continue
    rows = json.load(open(f))["ladder"]
    if len(rows) != 2:
        continue
    mean = sum(r["points"] for r in rows) / 2
    se = math.sqrt(sum(r["points_se"] ** 2 for r in rows)) / 2
    ok = mean >= 0.5 + 2 * se and all(r["points"] >= 0.5 - 2 * r["points_se"] for r in rows)
    print(f"{d} {net}: mean {mean:.3f} ± {se:.3f} -> {'qualifies' if ok else 'no'}", file=sys.stderr)
    if ok and (best is None or mean > best[0]):
        best = (mean, net, tau)
print(f"{best[1]} {best[2]}" if best else "NONE")
PY
)
status "gate: $VERDICT"
D1_MEAN=$("$PY" -c "import json,sys; r=json.load(open(sys.argv[1]))['ladder']; print(sum(x['points'] for x in r)/len(r))" "$OUT/D1/report.json" 2>/dev/null || echo 0)
down

# ---- C. relaunch, or the amplified test ---------------------------------------------------------
if [ "$VERDICT" != NONE ]; then
    read -r NET TAU <<< "$VERDICT"
    status "E8: launching the loop from $NET at cq tau $TAU (runs/loopmix16x9g054)"
    trap - EXIT INT TERM
    nohup "$REPO/scripts/launch_plan054.sh" "$M/$NET.onnx" "$TAU" > "$OUT/launch_plan054.out" 2>&1 &
    if "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.5 else 1)" "$D1_MEAN"; then
        for _ in $(seq 180); do "$HUB" status --hub "$HUB_URL" --token-file "$TOK" >/dev/null 2>&1 && break; sleep 10; done
        match D3 "cq30@5e-5 vs g_gen05 (on the loop's hub)" "$M/plan054_w1_cq30_lr5e5.onnx" "$GUMBEL" 1000 "$G05"
    fi
    status "done"
    exit 0
fi

status "no recipe passed the gate: E2b, the amplified test on gen01-07"
TR=""; for g in 01 02 03 04 05 06; do for k in 0 1 2 3 4 5 6 7; do TR="$TR $L/gen$g/shard$k.jsonl"; done; done
for k in 0 1 2 3 5 6; do TR="$TR $L/gen07/shard$k.jsonl"; done
for tau in 100 50; do
    [ -e "$P/.all_cq$tau" ] && continue
    # shellcheck disable=SC2086
    "$PREPARE" --in $TR --out "$P/all_train_cq$tau" --policy-target cq --tau $tau --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 \
        && "$PREPARE" --in "$L/gen07/shard4.jsonl" "$L/gen07/shard7.jsonl" --out "$P/g07_val_cq$tau" --policy-target cq \
            --tau $tau --value-blend 0.5 >> "$OUT/prepare.log" 2>&1 && touch "$P/.all_cq$tau" || die "prepare all cq$tau"
done
probe_arm e2b_cq100_2e4 "$P/all_train_cq100" "$P/g07_val_cq100" "$M/bbnet_mix16x9d1k_gen04.pt" 2e-4 --freeze-bn --select-on combined
probe_arm e2b_cq50_5e5_fix "$P/all_train_cq50" "$P/g07_val_cq50" "$M/bbnet_mix16x9d1k_gen04.pt" 5e-5 "${FIX[@]}"
for a in e2b_cq100_2e4 e2b_cq50_5e5_fix; do
    [ -s "$M/plan054_$a.onnx" ] && continue
    cp "$OUT/arms/$a.pt" "$M/plan054_$a.pt"
    (cd "$REPO/train" && "$PY" -c "import sys, torch; from bbnn.model import BBNet; from bbnn.export import export_onnx
export_onnx(BBNet.from_state_dict(torch.load(sys.argv[1], map_location='cpu')).eval(), sys.argv[2])" \
        "$M/plan054_$a.pt" "$M/plan054_$a.onnx") || die "export $a"
done
up
match E2b_ctl "d1k gen04 + gen01-07, cq100@2e-4, vs d1k gen04" "$M/plan054_e2b_cq100_2e4.onnx" "$GUMBEL" 1000 "$D1K" & Q1=$!
match E2b_new "d1k gen04 + gen01-07, cq50@5e-5+fix, vs d1k gen04" "$M/plan054_e2b_cq50_5e5_fix.onnx" "$GUMBEL" 1000 "$D1K" & Q2=$!
wait $Q1 $Q2
status "E2b done; the loop is NOT relaunched — plan 054 says decide with the user"
