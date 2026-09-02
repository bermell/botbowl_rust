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

> **Superseded 2026-09-02 — read plan 024 before using any number in this section or in the Run 1 table
> below.** Two things changed underneath them. (1) Plan 024 made NN leaf evaluation batched and GPU-resident
> and added `--parallel-games` to both `dataset` and `eval`: generation is ~3.8× faster and eval is no longer
> serial, so "generation 6–10 h, eval 2–4 h" is no longer the shape. (2) Plan 023's mover-tagging fix
> (`e107f06`) made the search correct and therefore busier — a drive now takes ~1.36× more decisions, so the
> *heuristic* baseline in the Run 1 table (gen00, 69 min) is unreachable on the current engine; the same work
> measures ~18.8 s/game per shard. The corpus health statistics are unchanged (TDs/drive 0.75 vs 0.79). Run 1's
> numbers remain valid as history, not as expectations.

Per generation: generation ~6–10 h (nn-value ≈ 30–70 s/game, heuristic shards finish early), prepare minutes, train well under an hour, eval ~2–4 h (120 full games, mostly at 1000-iter search both sides). ≈ 9–14 h/generation → **~4–5 generations over a weekend**. Disk: a few GB per generation (JSONL + npy); the script logs free disk at each generate phase.

## Failure behavior

- A generator shard that dies (the plan-020/021 OOM-kill pattern) is a **warning**, not a failure — its partial JSONL is used. Only an empty/missing shard aborts.
- Any prepare/train/eval failure writes a `FATAL:` line to `status.md` and exits; relaunching resumes from the failed phase.
- Dry-run validated 2026-08-12: full loop (mirror → generate → prepare → train → eval → gate) plus resume-skip, with tiny knobs in a scratch `RUN_DIR`.

## Run 1 (2026-08-28 → 08-30): paused after gen02 training, deliberately

First real run of the loop, on the Linux host. Bootstrapped its own gen-0 (no
`bbnet_14x7_db.onnx` on this machine), then two full generations.

| | mirror | gen00 | gen01 | gen02 |
|---|---|---|---|---|
| generate | — | 69 min (heuristic) | 863 min | 697 min |
| train (best val_value) | — | 0.4080 @ ep 4 | 0.3909 @ ep 1 | 0.3970 @ ep 2 |
| eval | 117 min | — | 690 min | **not run** |

gen01 report card (30 games/rung, seed 0 — comparable to plan 021's tables):
random 1.00, scripted 0.80, mcts-heuristic 0.90, **vs gen00 0.60 ⇒ PROMOTED**.
Against plan 021's `bbnet_14x7_db` reference (1.00 / 0.50 / 0.93) that is a clear
gain on the scripted rung and level on the teacher, with zero losses to it — so
the from-scratch bootstrap reached the reference net's class in two generations.

**Why it was paused rather than left running.** The pre-flight mirror match found
a real side bias, and the investigation (`plans/023-idea--home-away-side-bias.md`)
verified two engine bugs in the kickoff path. They do **not** touch the
drive-bounded training corpus, but they *do* affect every full game — which is
every ladder rung and every promotion gate. Continuing would have produced more
generations whose promote/reject decisions were measured under rules about to
change, at ~26 h per datapoint. Two cheaper things wanted the idle machine first:
`plans/025-plan--search-budget-convergence.md` (~40 min, and may cut `MCTS_ITERS`
2-3x for the price of changing a constant) and plan 023's deferred mirror re-runs.

**State at the pause.** Champion is `bbnet_14x7_gen01.onnx` (gen02 was trained but
never gated). All three nets are in `models/`. Markers: gen00/01/02 all have
`.generated .prepared .trained`; only gen01 has `.evaluated`. gen02's eval was
killed mid-flight and left no `report.json`, so it re-runs on resume — which is
what we want, since a post-fix gate is the trustworthy one.

**To resume:** `rm runs/loop14x7/STOP` first (a `STOP` file is in place, so a
relaunch exits immediately otherwise), then `nohup scripts/train_loop.sh &`.
Completed phases are skipped. Note that after fixing the `kicking_first_half`
train/eval mismatch (plan 023) the existing corpora become stale and the run
should start from a fresh `RUN_DIR` instead.

## After the weekend

- The `status.md` PROMOTED/REJECTED trail + per-gen `report.json` (fixed seed 0, fixed rungs) is the strength ladder — plot win rates by generation.
- If gens promote: revisit NN priors (`--evaluator nn` vs `nn-value` card on the newest champion, plan 021 §Next steps 4).
- If gens repeatedly reject: suspects are (a) 30-game gate noise (17/30 needed), (b) self-play data collapsing in variety (check TDs/drive and scoreless % in shard logs vs the 0.79/21% gen-0 baselines), (c) champion-relative labels drifting — consider merging corpora across generations before retraining.
- Mirror-match verdict (runs/loop14x7/mirror.json, Home/Away split): **a real side bias showed** — Home took 0.645 of points over 100 games (p ≈ 0.003). Every ladder number inherits it, so audit before fine-grained cross-gen comparisons. Aggregate `win_rate` still cancels it (rungs are paired Home/Away), so the promotion gate stays valid. Investigation, verified bugs and deferred tests: `plans/023-idea--home-away-side-bias.md`.
- Note when reading any report card: the aggregate `win_rate` and the `TD x:y` column are **candidate-relative and pooled across sides**, so neither can reveal a side bias — only the Home/Away split can, and the side-relative TD split is not currently recorded at all.

## Linux training host (added 2026-08-28)

The loop was authored on macOS; it now also runs on the Linux box, which is the GPU
training host (the Mac has no usable GPU). Changes to `scripts/train_loop.sh` and the
trainer, all no-ops on macOS:

- **Gen-0 bootstrap.** `models/` is gitignored, so a fresh clone has no
  `bbnet_14x7_db.onnx` and the loop died on the champion check. It now builds its own
  initial champion when none exists: heuristic-only corpus across all 8 shards, the same
  train/val shard split, best-val training, promoted with no gate (no incumbent to beat).
  Bootstrap seeds use `G=0` — disjoint from every generation. Skipped entirely when a
  champion is present, so hand-copying the original `.onnx` still works and just skips it.
  Sized by `BOOTSTRAP_GAMES_PER_SHARD` (defaults to `GAMES_PER_SHARD`).
- **Sleep inhibition**: `caffeinate` on macOS, `systemd-inhibit` on Linux.
- **`df -g` → `free_gb()`**: GNU `df` rejects `-g`, so the disk-free field in the generate
  status line was silently empty on Linux.
- **`TRAIN_DEVICE`** (default `auto`) passed to the trainer as `--device`.
- **`CARGO_TARGET_DIR` defaults to `target/14x7`**: board size is a build-time env var, so
  the 14x7 binaries and a default-board `cargo test --workspace` were evicting each other
  from a shared target dir and forcing a full rebuild on every switch.

**GPU note (the trap worth remembering).** The default PyPI `torch==2.13.0+cu130` ships
sm_75+ kernels; the host's GTX 1060 is sm_61. `torch.cuda.is_available()` returns **True**
and every kernel launch then fails with `no kernel image is available for execution on the
device`. `train/pyproject.toml` now pins torch to PyTorch's **cu126** index for Linux only
(`marker = "sys_platform == 'linux'"`), so macOS keeps the PyPI wheel; cu126 is the same
torch 2.13.0 and its sm_60 cubins are forward-compatible with Pascal 6.1. Because
`is_available()` is not a usable capability check here, `bbnn.train.resolve_device()`
probes with a real kernel launch: `auto` falls back to CPU with a printed reason, an
explicit `--device cuda` raises rather than silently training on CPU.

Validated end-to-end with tiny knobs in a scratch `RUN_DIR`/`MODEL_DIR`: bootstrap → gen01
→ eval → PROMOTED (exit 0), plus a resume run that skipped every completed phase.

## Cross-references

- plan 021 — the regime this automates (§Next steps 3), gate definition, gen-0 baselines.
- plan 020 — mixed-transition hedge rationale, best-val training practice, cost table.
- plan 019 — random-start generator and biases.
