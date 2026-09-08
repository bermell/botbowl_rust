#!/usr/bin/env bash
# Plan 032 #3 — re-tune PUCT_C under learned priors, then FPU reduction.
#
# Screen, not verdict: four 120-game arms against the production setting
# (c=10, plain FPU), same gen03 net on both sides, shared seed base. The
# plan's power table says a +0.03 effect is not resolvable at 120 games, so
# a winner here gets a properly sized match afterwards; an arm within 1 SE of
# 0.50 is simply "no screen signal".
#
# The plan says to run the c sweep "on the winner" of #2. Stage 2's result
# picks the backup rule for every arm here: mean if it scored >= 0.55 against
# minimax, otherwise minimax. Override with BACKUP=mean|minimax.
#
#   nohup scripts/exp032_s3_puct_fpu.sh > /dev/null 2>&1 &
source "$(dirname "$0")/exp032_lib.sh"

CHAMP="$MODELS/bbnet_14x7_gen03.onnx"

pick_backup() {
    [ -n "${BACKUP:-}" ] && { echo "$BACKUP"; return; }
    local rep="$OUT/s2-mean-vs-minimax.json"
    [ -f "$rep" ] || { echo minimax; return; }
    "$PY" - "$rep" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
row = [x for x in r["ladder"] if x["opponent"].startswith("vs:")][-1]
pts = (row["wins"] + 0.5 * row["draws"]) / row["games"]
print("mean" if pts >= 0.55 else "minimax")
EOF
}

BK="$(pick_backup)"
log "=== stage 3: PUCT_C sweep + FPU reduction, backup=$BK for all arms ==="
# #1b (exp032_s1b_d8h.sh) outranks these screens: it holds this marker from
# launch until its match is done, so the two evals never share the cores.
if [ -e "$OUT/s1b.pending" ]; then
    log "stage 3: waiting for #1b (s1b.pending) before the screens"
    while [ -e "$OUT/s1b.pending" ]; do stopped && exit 0; sleep 60; done
fi
start_sidecar "$CHAMP"

# Every arm: candidate = the variant, opponent = production c=10, k=0.
common=(--backup "$BK" --vs-backup "$BK")
stopped || play "s3-c3-vs-c10"   "$CHAMP" "$CHAMP" "${common[@]}" --puct-c 3  --vs-puct-c 10
stopped || play "s3-c30-vs-c10"  "$CHAMP" "$CHAMP" "${common[@]}" --puct-c 30 --vs-puct-c 10
stopped || play "s3-fpu100-vs-k0" "$CHAMP" "$CHAMP" "${common[@]}" --fpu-reduction 100 --vs-fpu-reduction 0
stopped || play "s3-fpu300-vs-k0" "$CHAMP" "$CHAMP" "${common[@]}" --fpu-reduction 300 --vs-fpu-reduction 0
log "=== stage 3 complete ==="
