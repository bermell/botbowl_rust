#!/usr/bin/env bash
# Plan 071 R7 (§5.2): what goes home from the cluster, in ONE rsync (one password + 2FA login).
#
#   hpc/arrhenius/pull_home.sh [--dry-run] [--print-filter] LOGIN P RUN [RUN ...]
#
#   LOGIN  the login node, as ssh knows it (user@login.hpc.arrhenius.naiss.se, or an ssh-config alias);
#          `local` reads P from this machine (tests, or a checkout that mounts project storage)
#   P      the project directory on the cluster (loop.env's $P), e.g. /nobackup/proj/disk/naiss2026-NN/botbowl
#   RUN    run names (loop.env's RUN_NAME): $P/runs/RUN and $P/models/RUN
#
# Pulls, into a mirror of $P's layout (HPC_MIRROR, default <repo>/runs/hpc/; additive, never deletes):
#   nets           models/RUN/*.{pt,onnx,ts} (every generation; .ts includes the fp16/bf16 variants)
#   run records    runs/RUN/: status.md, *.txt (champion, latest net, size centre), *.tsv (size_curve),
#                  *.json, slurm/chain.{log,state}, positions/;
#                  per generation: train.log, shadow_train.log, absorb.txt and every other *.txt, *.json
#                  (td_rate, train progress), *.tsv, verdict; in any subdirectory (drives, ladders,
#                  net_check, value_bench*): report.json, eval.games.jsonl, eval.args, *summary*.txt,
#                  net_check.txt; anything named *tree_stats* (plan 060)
#   positions      positions/ (the contested sets) and value_bench/ (the frozen benches)
#   pilot          runs/hpc_tune/**/results.{csv,md} and meta.json
# Not corpora: shard*.jsonl, mc/, prepared_*/ and per-sample bench outputs stay on the cluster
# (retention: hpc/arrhenius/retain_corpora.sh).
#
# Then the nets are hard-linked (copied across filesystems) into <repo>/models/hpc_RUN/, where the web
# app, value bench and drive scripts look for them. Nothing else outside the mirror is written. The
# registry is edited by hand from the pulled files (CLAUDE.md "Registry").
#
# Knobs: HPC_MIRROR (the mirror), HPC_MODELS (default <repo>/models; nets go to HPC_MODELS/hpc_RUN),
# RSYNC_EXTRA (more rsync flags, split on spaces, e.g. "--bwlimit=50m"; ssh options go in rsync's own
# RSYNC_RSH, e.g. RSYNC_RSH="ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h -o ControlPersist=1h"),
# PULL_EXTRA_FILTER (a file of extra rsync filter rules, read before the defaults).
# --dry-run passes -n to rsync and links nothing; --print-filter prints the filter list and exits.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRY=0 PRINT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run|-n) DRY=1; shift ;;
        --print-filter) PRINT=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --) shift; break ;;
        -*) echo "pull_home: unknown flag $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -ge 3 ] || { echo "usage: pull_home.sh [--dry-run] [--print-filter] LOGIN P RUN [RUN ...]" >&2; exit 2; }
LOGIN="$1" P="${2%/}"; shift 2
RUNS=("$@")
HPC_MIRROR="${HPC_MIRROR:-$REPO/runs/hpc}"
HPC_MODELS="${HPC_MODELS:-$REPO/models}"

# rsync filter rules: first match wins; a directory must be included for anything under it to be
# reached; `-m` then drops the directories nothing was taken from.
filter_rules() {
    local r
    if [ -n "${PULL_EXTRA_FILTER:-}" ]; then cat "$PULL_EXTRA_FILTER"; fi
    echo "# never corpora or prepared data, wherever they sit"
    echo "- shard*.jsonl"
    echo "- shard*.jsonl.zst"
    echo "- mc/"
    echo "- prepared_*/"
    echo "- probe_val/"
    echo "- shadow_probe_val/"
    echo "- worker-cache*/"
    for r in "${RUNS[@]}"; do
        echo "# nets of $r"
        echo "+ /models/"
        echo "+ /models/$r/"
        echo "+ /models/$r/*.pt"
        echo "+ /models/$r/*.onnx"
        echo "+ /models/$r/*.ts"
        echo "# run records of $r"
        echo "+ /runs/"
        echo "+ /runs/$r/"
        echo "+ /runs/$r/status.md"
        echo "+ /runs/$r/*.txt"
        echo "+ /runs/$r/*.tsv"
        echo "+ /runs/$r/*.json"
        echo "+ /runs/$r/slurm/"
        echo "+ /runs/$r/slurm/chain.log"
        echo "+ /runs/$r/slurm/chain.state"
        echo "+ /runs/$r/positions/***"
        echo "+ /runs/$r/gen[0-9]*/"
        echo "+ /runs/$r/gen[0-9]*/train.log"
        echo "+ /runs/$r/gen[0-9]*/shadow_train.log"
        echo "+ /runs/$r/gen[0-9]*/*.txt"
        echo "+ /runs/$r/gen[0-9]*/*.json"
        echo "+ /runs/$r/gen[0-9]*/*.tsv"
        echo "+ /runs/$r/gen[0-9]*/verdict"
        echo "+ /runs/$r/gen[0-9]*/eval.games.jsonl"
        echo "+ /runs/$r/gen[0-9]*/**/"
        echo "+ /runs/$r/gen[0-9]*/**/report.json"
        echo "+ /runs/$r/gen[0-9]*/**/eval.games.jsonl"
        echo "+ /runs/$r/gen[0-9]*/**/eval.args"
        echo "+ /runs/$r/gen[0-9]*/**/*summary*.txt"
        echo "+ /runs/$r/gen[0-9]*/**/net_check.txt"
        echo "+ /runs/$r/**/*tree_stats*"
    done
    echo "# positions, benches, pilot outputs (shared by every run)"
    echo "+ /positions/***"
    echo "+ /value_bench/***"
    echo "+ /runs/"
    echo "+ /runs/hpc_tune/"
    echo "+ /runs/hpc_tune/**/"
    echo "+ /runs/hpc_tune/**/results.csv"
    echo "+ /runs/hpc_tune/**/results.md"
    echo "+ /runs/hpc_tune/**/meta.json"
    echo "# everything else stays"
    echo "- *"
}

if [ "$PRINT" = 1 ]; then filter_rules; exit 0; fi

FILTER="$(mktemp)"; trap 'rm -f "$FILTER"' EXIT
filter_rules > "$FILTER"
SRC="$LOGIN:$P/"; [ "$LOGIN" = local ] && SRC="$P/"
mkdir -p "$HPC_MIRROR"
FLAGS=(-a -m --info=stats1 --filter="merge $FILTER")
[ "$DRY" = 1 ] && FLAGS+=(-n -v)
echo "pull_home: $SRC -> $HPC_MIRROR/ (runs: ${RUNS[*]})$([ "$DRY" = 1 ] && echo ', DRY RUN')"
# shellcheck disable=SC2086  # RSYNC_EXTRA is plain flags, split on purpose
rsync "${FLAGS[@]}" ${RSYNC_EXTRA:-} "$SRC" "$HPC_MIRROR/"

[ "$DRY" = 1 ] && exit 0
for r in "${RUNS[@]}"; do
    src="$HPC_MIRROR/models/$r"
    [ -d "$src" ] || { echo "pull_home: no nets for $r (yet)"; continue; }
    dst="$HPC_MODELS/hpc_$r"
    mkdir -p "$dst"
    n=0
    for f in "$src"/*.pt "$src"/*.onnx "$src"/*.ts; do
        [ -f "$f" ] || continue
        ln -f "$f" "$dst/" 2>/dev/null || cp -p "$f" "$dst/"
        n=$((n + 1))
    done
    echo "pull_home: $n net files of $r in $dst"
done
