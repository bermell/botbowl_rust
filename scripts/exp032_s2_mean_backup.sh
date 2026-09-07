#!/usr/bin/env bash
# Plan 032 #2 — mean backup vs minimax backup, same gen03 net both sides.
#
# Part A (pre-committed cheap check): two small random-start corpora on the
# same seeds, one per backup rule, then `audit_value_head_bias.py` on each.
# Plan 031 D1 put the search-added optimism at +0.10 under minimax; if the
# mean backup does not pull that toward the bare leaf's +0.03 the mechanism
# story is wrong and the match below is not worth its five hours.
#
# Part B: the 120-game match, candidate = gen03 with --backup mean, opponent
# = gen03 with the default minimax. Seed base shared with stage 1.
#
#   nohup scripts/exp032_s2_mean_backup.sh > /dev/null 2>&1 &
source "$(dirname "$0")/exp032_lib.sh"

CORPUS_GAMES="${CORPUS_GAMES:-150}"
CORPUS_SEED="${CORPUS_SEED:-32100000}"
CHAMP="$MODELS/bbnet_14x7_gen03.onnx"
CHAMP_PT="$MODELS/bbnet_14x7_gen03.pt"

log "=== stage 2: mean backup (plan 032 #2) ==="
start_sidecar "$CHAMP"

# corpus TAG  (BLOOD_MCTS_BACKUP is read by MctsBot::new inside `dataset`)
corpus() {
    local tag="$1"; local out="$OUT/$tag.jsonl"
    [ -s "$out" ] && [ -e "$out.done" ] && { log "$tag corpus exists"; return 0; }
    log "$tag: $CORPUS_GAMES random-start games, seed $CORPUS_SEED, BLOOD_MCTS_BACKUP=${BLOOD_MCTS_BACKUP:-unset}"
    local t0=$SECONDS
    # shellcheck disable=SC2086
    if ! "$UI" dataset --mode random-start --games "$CORPUS_GAMES" --seed "$CORPUS_SEED" \
            --mcts-iters 1000 --evaluator nn --model "$CHAMP" $NN_ARGS \
            --parallel-games "$PARALLEL" --truncate --out "$out" > "$OUT/$tag.log" 2>&1; then
        log "$tag corpus FAILED — see $tag.log"; return 1
    fi
    touch "$out.done"
    log "$tag corpus done ($(((SECONDS - t0) / 60)) min, $(wc -l < "$out") games)"
}

audit() {
    local tag="$1"
    local prep="$OUT/prep_$tag"
    [ -d "$prep" ] || "$PREPARE" --in "$OUT/$tag.jsonl" --out "$prep" > "$OUT/prep_$tag.log" 2>&1 \
        || { log "prepare $tag FAILED"; return 1; }
    local dims; dims="$(ls -d "$prep"/dims_* | head -1)"
    log "--- audit_value_head_bias $tag ---"
    "$PY" "$REPO/scripts/audit_value_head_bias.py" "$dims" "$CHAMP_PT" "$OUT/$tag.jsonl" \
        > "$OUT/audit_$tag.txt" 2>&1 || { log "audit $tag FAILED"; return 1; }
    grep -E "bare NN leaf gap|search gap|added by search|share of" "$OUT/audit_$tag.txt" | tee -a "$LOG"
}

stopped && exit 0
BLOOD_MCTS_BACKUP=minimax corpus "s2-corpus-minimax"
stopped && exit 0
BLOOD_MCTS_BACKUP=mean corpus "s2-corpus-mean"
audit "s2-corpus-minimax"
audit "s2-corpus-mean"

stopped && exit 0
play "s2-mean-vs-minimax" "$CHAMP" "$CHAMP" --backup mean
log "=== stage 2 complete ==="
