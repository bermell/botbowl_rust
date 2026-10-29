#!/usr/bin/env bash
# Plan 071 R7 (§5.3): corpus retention for a run dir on the cluster. DRY RUN BY DEFAULT: prints what
# it would delete and compress and touches nothing; --apply acts.
#
#   hpc/arrhenius/retain_corpora.sh [--apply] [--window N] [--keep-extra K] [--protect "G G ..."]
#                                   [--prepared-root DIR] RUN_DIR
#
# The live part of a run is every generation from C - WINDOW - KEEP_EXTRA + 1 up, where C is the
# newest genNN directory (finished or in progress) and WINDOW the loop's WINDOW_GENS: the window
# the next training reads, plus KEEP_EXTRA (2) generations of margin. Nothing in it is touched, nor
# in a generation not yet marked .generated and .trained, nor in a --protect'ed one (e.g. a window
# an A/B arm is retraining on). For each older generation, per §5.3:
#
#   prepared_train/, prepared_val/ (also under --prepared-root/genNN), probe_val/,
#   shadow_probe_val/                     deleted (regenerable from the shards)
#   raw train shards (TRAIN_SHARDS)       deleted once mc/ holds them: .mc_labelled is set and
#                                         mc/shardK.jsonl[.zst] has exactly as many lines (mc-label
#                                         writes every trajectory back, labelled or not); without
#                                         an mc/ copy they are the only one, and are compressed
#   mc/ train shards                      compressed (zstd; kept for a from-scratch retrain)
#   val shards (VAL_SHARDS), raw and mc/  compressed, kept (every future bench and rescreen)
#   everything else (eval games, reports, logs, nets)   untouched
#
# Compression writes shardK.jsonl.zst.part, tests it, renames it to shardK.jsonl.zst and only then
# removes the original, so an interrupted run leaves either the original or a verified .zst (a
# leftover .part is replaced next time). Read a compressed shard with `zstd -dc` (`--long=27`).
# Idempotent: run it after every generation, or daily.
#
# Knobs (env): TRAIN_SHARDS ("0 1 2 3 5 6") and VAL_SHARDS ("4 7"), the loop's split; WINDOW_GENS
# (else the newest RUN_DIR/slurm/env/job-*.loop.env snapshot's, else 3); ZSTD (zstd), ZSTD_LEVEL (10),
# ZSTD_THREADS (0 = all cores).

set -euo pipefail

APPLY=0 WINDOW="" KEEP_EXTRA=2 PROTECT="" PREPARED_ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --window) WINDOW="$2"; shift 2 ;;
        --keep-extra) KEEP_EXTRA="$2"; shift 2 ;;
        --protect) PROTECT="$2"; shift 2 ;;
        --prepared-root) PREPARED_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "retain_corpora: unknown flag $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -eq 1 ] || { echo "usage: retain_corpora.sh [--apply] [--window N] [--keep-extra K] [--protect LIST] [--prepared-root DIR] RUN_DIR" >&2; exit 2; }
RUN_DIR="${1%/}"
[ -d "$RUN_DIR" ] || { echo "retain_corpora: no such run dir: $RUN_DIR" >&2; exit 2; }
TRAIN_SHARDS="${TRAIN_SHARDS:-0 1 2 3 5 6}"
VAL_SHARDS="${VAL_SHARDS:-4 7}"
ZSTD="${ZSTD:-zstd}"; ZSTD_LEVEL="${ZSTD_LEVEL:-10}"; ZSTD_THREADS="${ZSTD_THREADS:-0}"

WINDOW_FROM="--window"
if [ -z "$WINDOW" ] && [ -n "${WINDOW_GENS:-}" ]; then WINDOW="$WINDOW_GENS"; WINDOW_FROM="WINDOW_GENS"; fi
if [ -z "$WINDOW" ]; then
    snap="$(ls -t "$RUN_DIR"/slurm/env/job-*.loop.env 2>/dev/null | head -1 || true)"
    if [ -n "$snap" ]; then
        WINDOW="$(grep -oE '(^|[ ;])WINDOW_GENS=[0-9]+' "$snap" | tail -1 | grep -oE '[0-9]+$' || true)"
        WINDOW_FROM="$(basename "$snap")"
    fi
fi
[ -n "$WINDOW" ] || { WINDOW=3; WINDOW_FROM="the loop's default"; }
case "$WINDOW$KEEP_EXTRA" in *[!0-9]*) echo "retain_corpora: --window/--keep-extra must be integers" >&2; exit 2 ;; esac

# Generations by number, newest last.
GENS=()
for d in "$RUN_DIR"/gen[0-9]*; do
    [ -d "$d" ] || continue
    n="${d##*/gen}"; case "$n" in *[!0-9]*) continue ;; esac
    GENS+=("$((10#$n))")
done
[ "${#GENS[@]}" -gt 0 ] || { echo "retain_corpora: no genNN dirs in $RUN_DIR"; exit 0; }
mapfile -t GENS < <(printf '%s\n' "${GENS[@]}" | sort -n)
C="${GENS[-1]}"
KEEP_FROM=$((C - WINDOW - KEEP_EXTRA + 1))

FREED=0 SHRINK=0 ACTIONS=0
human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "${1}B"; }
size_of() { du -sb "$1" 2>/dev/null | cut -f1 || echo 0; }
lines_of() {   # lines of a shard, compressed or not
    case "$1" in
        *.zst) "$ZSTD" -q -dc --long=27 "$1" | wc -l ;;
        *) wc -l < "$1" ;;
    esac
}
mc_copy() {    # the mc/ copy of shard $2 in generation dir $1, if any
    if [ -s "$1/mc/shard$2.jsonl" ]; then echo "$1/mc/shard$2.jsonl"
    elif [ -s "$1/mc/shard$2.jsonl.zst" ]; then echo "$1/mc/shard$2.jsonl.zst"
    fi
}
delete() {     # delete path $1 (file or dir) because $2
    local s; s="$(size_of "$1")"
    echo "  DELETE    ${1#"$RUN_DIR"/} ($(human "$s")): $2"
    FREED=$((FREED + s)); ACTIONS=$((ACTIONS + 1))
    [ "$APPLY" = 1 ] && rm -rf -- "$1"
    return 0
}
compress() {   # zstd file $1 in place (-> $1.zst), verified before the original goes
    local f="$1" s; s="$(size_of "$f")"
    echo "  COMPRESS  ${f#"$RUN_DIR"/} ($(human "$s"))"
    ACTIONS=$((ACTIONS + 1)); SHRINK=$((SHRINK + s))
    [ "$APPLY" = 1 ] || return 0
    if [ -e "$f.zst" ]; then
        # A previous run renamed a verified .zst into place but stopped before removing the original.
        "$ZSTD" -q -t --long=27 "$f.zst" || { echo "  WARN      $f.zst fails its test: left both for a look" >&2; return 0; }
        rm -f -- "$f"; return 0
    fi
    rm -f -- "$f.zst.part"
    "$ZSTD" -q "-$ZSTD_LEVEL" "-T$ZSTD_THREADS" --long=27 "$f" -o "$f.zst.part" \
        && "$ZSTD" -q -t --long=27 "$f.zst.part" \
        && mv -- "$f.zst.part" "$f.zst" \
        && rm -f -- "$f" \
        || { echo "  WARN      compressing $f failed: original kept" >&2; rm -f -- "$f.zst.part"; }
}

echo "retain_corpora: $RUN_DIR — newest gen$(printf '%02d' "$C"), window $WINDOW ($WINDOW_FROM) + $KEEP_EXTRA kept: gens >= $(printf '%02d' $((KEEP_FROM < 0 ? 0 : KEEP_FROM))) untouched$([ -n "$PROTECT" ] && echo ", protected: $PROTECT")$([ "$APPLY" = 1 ] || echo '; DRY RUN')"
if [ "$APPLY" = 1 ] && ! command -v "$ZSTD" > /dev/null 2>&1; then
    echo "retain_corpora: $ZSTD not found (set ZSTD=...)" >&2; exit 2
fi

for g in "${GENS[@]}"; do
    [ "$g" -lt "$KEEP_FROM" ] || continue
    GG="$(printf 'gen%02d' "$g")"; D="$RUN_DIR/$GG"
    if [[ " $PROTECT " == *" $g "* || " $PROTECT " == *" $GG "* ]]; then echo "$GG: protected, skipped"; continue; fi
    if [ ! -e "$D/.generated" ] || [ ! -e "$D/.trained" ]; then echo "$GG: not marked .generated and .trained, skipped"; continue; fi
    echo "$GG:"
    for p in "$D/prepared_train" "$D/prepared_val" "$D/probe_val" "$D/shadow_probe_val" \
             ${PREPARED_ROOT:+"$PREPARED_ROOT/$GG/prepared_train" "$PREPARED_ROOT/$GG/prepared_val"}; do
        [ -e "$p" ] && delete "$p" "prepared data, regenerable from the shards"
    done
    # As the loop's prune_prepared: the marker goes with the dirs, so nothing skips a prepare it needs.
    if [ "$APPLY" = 1 ]; then rm -f -- "$D/.prepared"; fi
    for K in $TRAIN_SHARDS; do
        raw="$D/shard$K.jsonl"; mc="$(mc_copy "$D" "$K")"
        if [ -e "$raw" ]; then
            if [ -n "$mc" ] && [ -e "$D/.mc_labelled" ]; then
                nr="$(lines_of "$raw")"; nm="$(lines_of "$mc")"
                if [ "$nr" -eq "$nm" ] && [ "$nr" -gt 0 ]; then
                    delete "$raw" "raw train shard, $nr trajectories also in $(basename "$mc")"
                else
                    echo "  KEEP      ${raw#"$RUN_DIR"/}: $nr lines but ${mc#"$RUN_DIR"/} has $nm"
                fi
            else
                compress "$raw"   # no mc/ copy: the only one
            fi
        fi
        [ -f "$D/mc/shard$K.jsonl" ] && compress "$D/mc/shard$K.jsonl"
    done
    for K in $VAL_SHARDS; do
        [ -f "$D/shard$K.jsonl" ] && compress "$D/shard$K.jsonl"
        [ -f "$D/mc/shard$K.jsonl" ] && compress "$D/mc/shard$K.jsonl"
    done
done

echo "retain_corpora: $ACTIONS actions; $(human "$FREED") deleted, $(human "$SHRINK") compressed (~25x smaller)$([ "$APPLY" = 1 ] || echo '. Dry run: nothing changed; --apply to act')"
