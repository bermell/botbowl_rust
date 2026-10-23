#!/usr/bin/env bash
# Plan 068 §7 / exp079: the validation arms, in turn, on the 16x9 contested set, deterministic preset,
# fixed size (no SPRT stop; scripts/gate_replay.py replays both tests offline).
#   null      gen48 vs gen48, gated (report)
#   g45gate   gen45 vs gen48, gated (report)
#   g45plain  gen45 vs gen48, ungated (today's rung)
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
V="$REPO/scripts/exp079_validate.sh"
export SPRT= PARALLEL="${PARALLEL:-8}"
ARM=null CAND=gen48 REF=gen48 GATE=report CAP="${CAP_NULL:-120}" "$V"
ARM=g45gate CAND=gen45 REF=gen48 GATE=report CAP="${CAP_PAIR:-160}" "$V"
ARM=g45plain CAND=gen45 REF=gen48 GATE= CAP="${CAP_PAIR:-160}" "$V"
echo "validate done $(date +%T)"
