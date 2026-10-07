# scripts/

Standing tools: the loop, its launchers, benchmarks and the analysis scripts the loop or the
plans still use. Every `.sh` documents its arguments and env knobs in its header; every `.py` in
its docstring (`--help` where it parses flags). Run them from the repo root; Python ones with
`train/.venv/bin/python` (or `uv run` from `train/`).

`archive/` holds finished one-off experiments (`exp*.sh`, their report scripts, `launch_vl0.sh`)
kept for the plans that cite them. They predate schema v9 and the current CLI: each carries a
`# Status: archived …` line naming the last commit it ran at. Don't extend them; copy what you
need into a new script.

`lib/git.sh` — `require_clean_tree`, `dirty_suffix`, `game_crates` (source it; one definition of
"dirty" and of the code a worker's games run).

## The loop

| Script | What | Example |
|---|---|---|
| `train_loop.sh` | The generation loop: generate → mc-label → prepare → train → eval, gateless. ~90 env knobs; don't edit while it runs (bash reads it by offset). | via a launcher |
| `launch_plan058.sh` | The live loop (`runs/loopmix16x9v9`): plan 058's knobs, allowlist, then `exec train_loop.sh`. | `scripts/launch_plan058.sh` |
| `launch_plan056.sh` | The previous loop (`runs/loopmix16x9g056`), kept as the template the 058 launcher diffs against. | — |
| `launch_plan054.sh`, `launch_gumbel.sh`, `launch_d1k.sh`, `launch_mix16x9.sh`, `az_from_scratch.sh` | Earlier loop launchers cited by plans 032/042/045/049/053/054; not runnable on schema v9 nets as is. | — |
| `stop_after.sh` | Place the loop's `STOP` file once a given line appears in `status.md`. | `scripts/stop_after.sh 'gen09 eval:' runs/<run>` |
| `size_curriculum.py` | Moves the board-size centre from the corpus TD rate (called by the loop, `SIZE_MODE=centred`). | — |
| `td_rate.py`, `drive_curve.py`, `anchor_curve.py`, `eval_summary.py`, `absorb_probe.py` | Per-generation status numbers the loop prints: TD rate, SPRT drive results, anchor curve, report one-liner, did training absorb the search. | `scripts/drive_curve.py runs/<run>` |
| `anchor_backfill.sh` | Score already-trained generations against the frozen anchor with the loop's settings. | — |
| `nn_server.py` | The batched GPU inference sidecar the loop and workers talk to (plan 024). | started by the loop |

## Per-net checks

| Script | What | Example |
|---|---|---|
| `net_check.sh` | Plan 055's standing check (< 1 h): does more search help this net (budget ladder via `override-audit`). | `scripts/net_check.sh NET.onnx GEN_DIR OUT_DIR` |
| `value_bench.sh` | Score value heads on the frozen MC value benchmark (minutes per net). | `scripts/value_bench.sh BENCH.jsonl OUT net=NET.onnx` |
| `value_bench_build_v9.sh`, `value_bench_freeze.py`, `value_bench_summary.py` | Re-freeze the benchmark from held-out shards; summarise `value-bench` output. | — |
| `override_audit_summary.py` | Summarise `botbowl-ui override-audit` rows. | — |

## Throughput (current focus — see `PROFILING.md`)

| Script | What | Example |
|---|---|---|
| `perf_search_bench.sh` | Search CPU as instructions retired, plus a trajectory hash (equal hash = unchanged search). Linux `perf`. | `scripts/perf_search_bench.sh runs/perf/a` |
| `perf_gen_bench.sh` | Generation games/min and sidecar stats at several `--parallel-games`. | `scripts/perf_gen_bench.sh runs/perf/base NET "12 24 36" 48` |
| `perf_two_sidecars.sh` | Is the sidecar's front end the bottleneck? Two sidecars, two generators at once. | — |
| `nn_throughput_probe.sh` | Which stream source feeds the sidecar best. | `scripts/nn_throughput_probe.sh <arm>` |
| `measure_search_speedup.sh`, `hash_ab.sh` | Production-shaped A/B for a search or `GameState::hash` change (plans 035/044). | — |
| `fixed_getrandom.c` | `LD_PRELOAD` shim pinning std's `RandomState` keys (used by `perf_search_bench.sh`). | — |

## Ranking and validation (plan 051)

| Script | What |
|---|---|
| `plan051_gold.sh`, `plan051_proxies.sh`, `validate_proxy.py` | Gold results for the validation pairs; score ranking proxies (SPRT on drives) against them. |
| `positions_screen.py` | Screen a drive-rung position set down to its contested positions. |
| `paired_summary.py`, `pair_correlation.py` | Score a rung by seed pair; how much pairing buys. |

## Analysis and audits

| Script | What |
|---|---|
| `corpus_skills_setup.py`, `show_setups.py` | How the loop's bots use skills and kickoff setups; draw chosen setups. |
| `target_stats.py`, `wdl_summary.py`, `weight_distance.py` | Policy-target statistics; value heads on held-out drives; how far training moved a net. |
| `convergence_summary.py`, `convergence_topk.py`, `puct_sweep_summary.py` | Analyse `botbowl-ui convergence` dumps. |
| `side_bias_summary.py`, `side_bias_pooled.py`, `kickoff_effect.py` | Home/Away bias across eval logs. |
| `audit_*.py` | Plan 031 corpus / encoding / prior / value-bias audits. |
| `make_random_net.py` | A randomly initialised BBNet (gen-0 seed for a from-scratch run). |
