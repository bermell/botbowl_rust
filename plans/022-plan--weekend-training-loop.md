# Automated weekend training loop (gen-1+) on 14x7

**Status:** Ready to launch (built 2026-08-12). Automates plan 021 §Next steps 3 — the gen-1 self-play loop — as an unattended multi-generation run with ssh-monitorable progress. The regime itself (drive-bounded corpus, nn-value generator, mixed heuristic hedge, best-val training, report-card eval, promotion gate) is exactly plan 021's; this plan only adds the orchestration and the missing eval instrument.

## What was built

1. **Net-vs-net ladder rung** (`botbowl-ui eval`): `--vs-evaluator <kind> --vs-model PATH` appends a rung against an arbitrary MCTS opponent (runs at `--opponent-iters`, default `--mcts-iters`). `--skip-fixed-rungs` drops random/scripted/mcts-heuristic, leaving only the vs rung — used for mirror matches and pure promotion gates. This is the instrument plan 021's promotion gate (≥55% vs previous gen) needed.
2. **Home/Away split on every ladder row** (`wins_as_home/losses_as_home/wins_as_away/losses_as_away`, in JSON and the printed table). Alternating sides cancels a side bias out of the aggregate `win_rate`, so the plan-021 mirror anomaly (open issue 5) is only diagnosable with the split.
3. **`scripts/train_loop.sh`** — the loop driver (first automation in the repo). Bash, resume-safe via `.done` markers, self-caffeinating on macOS.
4. **`scripts/eval_summary.py`** — report JSON → one status line; `--gate 0.55` exit code implements the promotion decision.

## The regime (one generation)

All at 14x7 (`BOARD_SIZE_W=14 BOARD_SIZE_H=7 BOARD_PLAYERS=4`), 1000 MCTS iters, tuned plan-021 placement biases (CLI defaults):

1. **Generate** 8 parallel shards × 600 drive-bounded random-start games (~4.8k games/gen — sized for ~4–5 generations per weekend; plan 021 showed data shape beats scale). Shards 0–4 use the **current champion** via `nn-value`; shards 5–7 are a **heuristic hedge** (37.5%, per plan 020's mixed-transition design). Seeds: `10_000_000 + gen*1_000_000 + shard*100_000` — disjoint from all prior corpora.
2. **Prepare**: train = shards 0–3,5,6; val = shards 4+7 held out whole (one nn shard + one heuristic shard, so val matches the train mix).
3. **Train** 10 epochs, best-val restore (checkpoint persisted on every improvement — a killed run keeps its best net), export `models/bbnet_14x7_genNN.{pt,onnx}`.
4. **Report card** (`--skip-lectures`, 30 paired games/rung, seed 0 = comparable to plan 021's tables): random, scripted, mcts-heuristic, **vs champion** (nn-value both sides).
5. **Promotion gate**: win_rate ≥ 0.55 on the vs-champion rung → the new net becomes the generator for the next generation. On failure the champion stays, the loop continues with fresh seeds, and the rejected net + report stay on disk for inspection.

Pre-flight (runs once): **100-game heuristic mirror match** (settles plan 021 open issue 5 via the new Home/Away split before any weekend deltas are trusted).

Initial champion: `models/bbnet_14x7_db.onnx` (the 93%-vs-teacher gen-0 net).

## Operating it

```sh
# launch (survives logout; the script re-execs itself under caffeinate -is)
nohup scripts/train_loop.sh > /dev/null 2>&1 &

# monitor over ssh
tail -f runs/loop14x7/status.md            # one line per phase — the dashboard
tail -f runs/loop14x7/loop.log             # detail + warnings
tail -f runs/loop14x7/gen01/shard0.log     # live per-game generator progress
cat  runs/loop14x7/champion.txt            # current champion
cat  runs/loop14x7/gen01/report.json       # full report card

# stop cleanly (exits at the next phase boundary), resume by relaunching —
# completed phases are skipped via .done markers
touch runs/loop14x7/STOP
```

Knobs (env vars): `MAX_GENS` (10), `GAMES_PER_SHARD` (600), `EVAL_GAMES` (30), `MIRROR_GAMES` (100), `EPOCHS` (10), `GATE` (0.55), `MCTS_ITERS` (1000), `INIT_CHAMPION`, `RUN_DIR`, `MODEL_DIR` (the last two exist so dry runs never touch `runs/loop14x7` or `models/`).

Layout per generation: `runs/loop14x7/genNN/` holds `shard{0..7}.{jsonl,log}`, `prepared_{train,val}/`, `train.log`, `eval.log`, `report.json`, `verdict` (PROMOTED/REJECTED). Models land in `models/bbnet_14x7_genNN.{pt,onnx}` — every generation kept for the strength ladder over time.

## Expected wall clock (from plan 020/021 costs)

Per generation: generation ~6–10 h (nn-value ≈ 30–70 s/game, heuristic shards finish early), prepare minutes, train well under an hour, eval ~2–4 h (120 full games, mostly at 1000-iter search both sides). ≈ 9–14 h/generation → **~4–5 generations over a weekend**. Disk: a few GB per generation (JSONL + npy); the script logs free disk at each generate phase.

## Failure behavior

- A generator shard that dies (the plan-020/021 OOM-kill pattern) is a **warning**, not a failure — its partial JSONL is used. Only an empty/missing shard aborts.
- Any prepare/train/eval failure writes a `FATAL:` line to `status.md` and exits; relaunching resumes from the failed phase.
- Dry-run validated 2026-08-12: full loop (mirror → generate → prepare → train → eval → gate) plus resume-skip, with tiny knobs in a scratch `RUN_DIR`.

## After the weekend

- The `status.md` PROMOTED/REJECTED trail + per-gen `report.json` (fixed seed 0, fixed rungs) is the strength ladder — plot win rates by generation.
- If gens promote: revisit NN priors (`--evaluator nn` vs `nn-value` card on the newest champion, plan 021 §Next steps 4).
- If gens repeatedly reject: suspects are (a) 30-game gate noise (17/30 needed), (b) self-play data collapsing in variety (check TDs/drive and scoreless % in shard logs vs the 0.79/21% gen-0 baselines), (c) champion-relative labels drifting — consider merging corpora across generations before retraining.
- Mirror-match verdict (runs/loop14x7/mirror.json, Home/Away split): if a real side bias shows, every ladder number inherits it — audit before fine-grained cross-gen comparisons.

## Cross-references

- plan 021 — the regime this automates (§Next steps 3), gate definition, gen-0 baselines.
- plan 020 — mixed-transition hedge rationale, best-val training practice, cost table.
- plan 019 — random-start generator and biases.
