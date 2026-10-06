#!/usr/bin/env bash
# Plan 055 §6: the standing net check (< 1 h). Does more search help this net?
#
# For each rung of a budget ladder, `botbowl-ui override-audit` on the same seeded decision sample
# from a corpus the net generated (and was not trained on), eval boards only, then one summary line:
# the search's gain per decision over the bare policy at each budget (must be >= 0 and must not drop
# as the budget rises — the user's criterion), whether the curve is monotone, the value head's error
# against Monte Carlo (RMS after removing MC sampling noise, and bias), the realised-on-predicted
# slope, and the smallest Q-gap bucket whose overrides realise a positive gain (the override margin
# to set). All rungs run at once on one GPU sidecar.
#
#   scripts/net_check.sh NET.onnx GEN_DIR OUT_DIR [CONFIG] [LADDER]
#   scripts/net_check.sh models/az_v7/bbnet_mix16x9g_gen05.onnx runs/loopmix16x9g/gen06 runs/netcheck/g05
#
# CONFIG defaults to cfgs/gumbel16_f1000.toml; LADDER to "64 250 1000 4000". Env: OVERRIDES (per
# rung, default 250), PLAYOUTS (default 32), PARALLEL (threads per rung, default 2), BOARDS
# (default 14x7,16x9), SEED (default 55100).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
NET="${1:?net .onnx}"; GEN="${2:?generation dir with shard*.jsonl}"; OUT="${3:?out dir}"
CFG="${4:-cfgs/gumbel16_f1000.toml}"; LADDER="${5:-64 250 1000 4000}"
OVERRIDES="${OVERRIDES:-250}"; PLAYOUTS="${PLAYOUTS:-32}"; PARALLEL="${PARALLEL:-2}"
BOARDS="${BOARDS:-14x7,16x9}"; SEED="${SEED:-55100}"
mkdir -p "$OUT"
export BOARD_SIZE_W=16 BOARD_SIZE_H=9 BOARD_PLAYERS=6 CARGO_TARGET_DIR="$REPO/target/16x9"
UI="$CARGO_TARGET_DIR/release/botbowl-ui"; PY="$REPO/train/.venv/bin/python"
SOCK="/tmp/bbnn-netcheck-$$.sock"
NN_PID=""; trap '[ -n "$NN_PID" ] && kill "$NN_PID" 2>/dev/null; rm -f "$SOCK"' EXIT INT TERM
cargo build --release -p botbowl-ui >> "$OUT/build.log" 2>&1 || { echo "build failed — see $OUT/build.log" >&2; exit 1; }
"$PY" scripts/nn_server.py --socket "$SOCK" --device cuda --model "$NET" --max-models 2 \
    --stats-every 300 --canvas 11x18 >> "$OUT/nn_server.log" 2>&1 & NN_PID=$!
for _ in $(seq 120); do [ -S "$SOCK" ] && break; sleep 1; done
[ -S "$SOCK" ] || { echo "nn_server did not come up — see $OUT/nn_server.log" >&2; exit 1; }
SHARDS=$(ls "$GEN"/shard*.jsonl)
SECONDS=0
PIDS=""
for b in $LADDER; do
    [ -s "$OUT/b$b.jsonl" ] && [ -e "$OUT/b$b.done" ] && continue
    # shellcheck disable=SC2086
    ( "$UI" override-audit --corpus $SHARDS --model "$NET" --nn-server "$SOCK" --search-config "$CFG" \
        --search-iters "$b" --decisions "$OVERRIDES" --playouts "$PLAYOUTS" --parallel "$PARALLEL" \
        --board "$BOARDS" --seed "$SEED" --out "$OUT/b$b.jsonl" > "$OUT/b$b.log" 2>&1 \
      && "$PY" scripts/override_audit_summary.py "$OUT/b$b.jsonl" > "$OUT/b$b.summary.txt" 2>&1 \
      && touch "$OUT/b$b.done" ) &
    PIDS="$PIDS $!"
done
# shellcheck disable=SC2086
wait $PIDS
"$PY" - "$OUT" "$LADDER" "$NET" "$CFG" "$SECONDS" <<'PY' | tee "$OUT/net_check.txt"
import math, re, sys
out, ladder, net, cfg, secs = sys.argv[1], [int(b) for b in sys.argv[2].split()], sys.argv[3], sys.argv[4], int(sys.argv[5])
num = r"([+-]?\d+\.\d+)"
pts, extra = [], {}
for b in ladder:
    try:
        t = open(f"{out}/b{b}.summary.txt").read()
    except OSError:
        pts.append((b, None, None)); continue
    m = re.search(rf"search gain per decision {num} ± {num}", t)
    pts.append((b, float(m.group(1)), float(m.group(2))) if m else (b, None, None))
    if b == max(ladder) or b == 1000:
        g = {}
        for k, pat in (("rms", r"RMS net error ~(\d+\.\d+)"), ("bias", rf"bias mean\[V\(s\) - MC\(s\)\]\s+{num}"),
                       ("slope", rf"realised on predicted: slope {num}")):
            mm = re.search(pat, t)
            g[k] = float(mm.group(1)) if mm else math.nan
        # Smallest Q-gap bucket whose overrides realise > 0 beyond 2 SE: the margin to try (1 SE picked
        # noise at ~250 overrides per rung: exp066's <0.01 bucket at +0.022 ± 0.018).
        margin = "none"
        sec = t.split("by Q gap", 1)[1].split("\n\n", 1)[0] if "by Q gap" in t else ""
        for line in sec.splitlines()[1:]:
            mm = re.match(rf"\s+(\S+)\s+\d+\s+\S+\s+\S+\s+\S+\s+{num}(?: ± (\d+\.\d+))?", line)
            if mm and mm.group(1) != "unscored" and mm.group(3) and float(mm.group(2)) > 2 * float(mm.group(3)):
                margin = mm.group(1); break
        g["margin"] = margin
        extra[b] = g
curve = " | ".join(f"@{b} " + (f"{g:+.4f} ± {s:.4f}" if g is not None else "n/a") for b, g, s in pts)
ok = [p for p in pts if p[1] is not None]
drops = [f"{a[0]}->{c[0]}" for a, c in zip(ok, ok[1:]) if c[1] < a[1] - 2 * math.hypot(a[2], c[2])]
neg = [str(b) for b, g, s in ok if g < -2 * s]
verdict = "MONOTONE" if not drops and not neg else "BROKEN (" + ", ".join(
    ([f"drop {d}" for d in drops]) + ([f"below policy at {n}" for n in neg])) + ")"
e = extra.get(1000) or extra.get(max(ladder)) or {}
print(f"net_check {net.rsplit('/', 1)[-1]} [{cfg.rsplit('/', 1)[-1]}] gain/decision {curve} -> {verdict}; "
      f"value RMS {e.get('rms', math.nan):.3f} bias {e.get('bias', math.nan):+.3f}; "
      f"realised/predicted slope {e.get('slope', math.nan):+.2f}; overrides pay from Q gap {e.get('margin', '?')} "
      f"({secs // 60} min)")
PY
