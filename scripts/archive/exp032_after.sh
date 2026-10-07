#!/usr/bin/env bash
# Status: archived 2026-10-07 (pre-schema-v9); last runnable at ae721b7 or earlier
# Run a stage script once a given pid (an earlier stage) has exited, so
# stages queue up without contending for the 8 logical cores.
#   nohup scripts/exp032_after.sh <pid> scripts/exp032_s2_mean_backup.sh > /dev/null 2>&1 &
pid="$1"; shift
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
exec "$@"
