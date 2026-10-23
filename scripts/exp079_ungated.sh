#!/usr/bin/env bash
# Plan 068 / exp079: the ungated gen45-vs-gen48 arm, run only once the loop's generate phase is steady
# and the box has room: MemAvailable >= 4.5 GB and >= 2 GB free on the GPU. 8 streams, stopped after
# MINUTES or as soon as MemAvailable falls below 3 GB.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${MAIN_CHECKOUT:-/home/mattias/repos/botbowl_rust}"
STATUS="$R/runs/loopmix16x9v9/status.md"
MINUTES="${MINUTES:-75}"
until grep -q "gen50 generate:" "$STATUS"; do sleep 60; done
echo "generate started; settling $(date +%T)"
sleep 600
mem() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
echo "MemAvailable $(mem) MB, GPU free $(gpu_free) MiB $(date +%T)"
if [ "$(mem)" -lt 4500 ] || [ "$(gpu_free)" -lt 2048 ]; then echo "no room: not running the ungated arm"; exit 0; fi
ARM=g45plain CAND=gen45 REF=gen48 GATE= CAP=200 SPRT= PARALLEL=8 "$REPO/scripts/exp079_validate.sh" &
end=$(( $(date +%s) + MINUTES * 60 ))
sleep 30
while [ "$(date +%s)" -lt "$end" ]; do
  pids=$(ps -eo pid,args | awk 'index($0, "val_g45plain/") && /release\/botbowl-ui/ {print $1}')
  [ -z "$pids" ] && break
  if [ "$(mem)" -lt 3000 ]; then echo "MemAvailable $(mem) MB: stopping $(date +%T)"; break; fi
  sleep 15
done
for pid in $(ps -eo pid,args | awk 'index($0, "val_g45plain/") && /release\/botbowl-ui/ {print $1}'); do kill "$pid"; done
wait
echo "ungated done $(date +%T): $(wc -l < "$R/runs/exp079/val_g45plain/eval.games.jsonl" 2>/dev/null || echo 0) lines"
