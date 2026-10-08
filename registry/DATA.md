# Data registry — corpora and frozen benchmark sets

What was generated, by which code and net, with which settings. Newest first within a section.
Nets are in [NETS.md](NETS.md), experiments in [EXPERIMENTS.md](EXPERIMENTS.md), head-to-head
results in [MATCHES.md](MATCHES.md). **Keep this current:** add a row whenever a corpus or a
benchmark set is produced (see the root CLAUDE.md).

Conventions:
- **commit** is the one stamped into the corpus (`generate.log`'s "wrote … (commit …)", each
  trajectory's `meta`). A loop relaunched mid-generation keeps the commit of the code that played
  the games.
- **Seeds:** the loop's shard K of generation G uses `10000000 + G·1e6 + K·1e5 + game`
  (`SEED_BASE`, `--shard-seed-stride 100000`). A `--next-drive` record shares its first drive's seed.
- **Held-out:** shards 4 and 7 of every loop generation are validation only (never trained on).
- **Usable** = readable by the current code (schema v9, the rules since 2026-10-06's master merge).

## Loop `runs/loopmix16x9v9` (plan 058; launched 2026-10-06 21:37)

Common settings, all generations (`scripts/launch_plan058.sh` → `scripts/train_loop.sh`):
- `job generate --mode random-start --next-drive`, preset `cfgs/gumbel16_f1000_gen.toml` (Gumbel
  root m=16, q floor 1000, scale 1 noise), `--mcts-iters 1000`, mean backup (hardcoded), evaluator
  `nn` on the GPU sidecar (`--canvas 11x18`);
- board sizes centred: `--size-centre 144 --size-temperature 0.3 --size-floor 0.2 --size-aspect
  1.5-2.8 --size-max-area 144 --size-min-area 70 --cells-per-player 26` (build 16x9/6; 12 boards
  from 12x5 to 16x9, 16x9/6 ~15%);
- MC value labels (plan 056): `botbowl-ui mc-label --playouts 8 --seed 56000+G`, policy-only
  playouts under the generator net, every train and val shard → `genG/mc/shardK.jsonl` (the
  training input; raw shards stay alongside).

| gen | generator net | games (records) | samples | commit | streams | notes |
|---|---|---|---|---|---|---|
| 11 | v9 gen10 | 8×400 (5820) | 202,818 | f75eda5 | 48 local, 2 sidecars, laptop 10 (v17) from the start; **108 min** | laptop ~800 decisions/min at full speed, more than either local worker |
| 10 | v9 gen09 (first τ=50 net) | 8×400 (5801) | 197,976 | f75eda5 | 48 local, 2 sidecars; laptop 10 from 21:40; 235 min (shared ~3.5 h with gen09's drives) | first corpus with the 14x5 throw-in fix: 14x5 depth 9.2 plies, 48% chance (gen09: 63, 93%); `runs/plan060/gen10_tree_stats.txt` |
| 09 | v9 gen08 | 8×400 (5785) | 199,544 | 3c25109 | 48 local, 2 sidecars, no laptop (protocol v17); 180 min | first corpus with plan 060's per-sample `tree` block (`runs/plan060/gen09_tree_stats.txt`) |
| 08 | v9 gen07 | **1361 drives (2458 records), stopped at 42%** | 82,522 | 43a45c6 | 48 local, 2 sidecars | stopped for exp069's GPU (the user); the hub filled shards in order, so the records were **redistributed by seed over the 8 shards** (originals in `gen08/partial_original/`) |
| 07 | v9 gen06 | 8×400 (5810) | 204,956 | 43a45c6 | 48 local, 2 sidecars, laptop part | generate shared with gen06's drive benchmark |
| 06 | v9 gen05 | 8×400 (5797) | 201,364 | 43a45c6 | 48 local, 2 sidecars, laptop 5 | first at 400/shard and two sidecars; 114 min |
| 05 | v9 gen04 | 8×300 (4355) | 151,799 | 407b335 | 28 local, laptop part | first corpus from the bot that plays forced moves without search |
| 04 | v9 gen03 | 8×300 (4310) | 150,520 | ae721b7 | 36 local, laptop late | last corpus with searched forced moves (20% of samples have < 2 children; `prepare` drops them from gen04's training on) |
| 03 | v9 gen02 | 8×300 (4324) | 149,454 | ae721b7 | 36 local + laptop | |
| 02 | v9 gen01 | 8×300 (4349) | 152,111 | ae721b7 | 36 local + laptop | |
| 01 | init (g056 gen04 v9) | 8×300 (4344) | 154,960 | ae721b7 | 36 local + laptop 10 | |

**Known defect in gen01-09 (fixed in 13df1bb, live from the gen10 relaunch):** on 14x5/3 (~2.4%
of drives) the search's scripted throw-in moved the ball 0 squares, so about a third of 14x5
decisions spent their budget in an endless, never-valued edge-bounce chain (plan 060 §6). Their
visit targets and root values are poor; every other board is unaffected (byte-identical games).

Per-generation corpus statistics (TD rates, skill use, setups) are in plan 058 §6;
`scripts/corpus_skills_setup.py runs/loopmix16x9v9 --gens N` recomputes them.

## Frozen benchmark sets

| file | what | built by | settings |
|---|---|---|---|
| `runs/value_bench/v9_gen01_val.jsonl` | **the value benchmark under the v9 rules** (plan 056 §2 / 058 §6): 3023 states from the v9 loop's gen01 held-out shards (4, 7), random-start drive-1 states only | `scripts/value_bench_build_v9.sh runs/loopmix16x9v9 models/az_v7/bbnet_mix16x9g056_gen04_v9.onnx …` (2026-10-07, ae721b7) | `override-audit --all --decisions 3000 --playouts 48 --seed 58100 --board 14x7,16x9`, MC under the init net's policy; mean MC SE 0.071 |
| `runs/loopmix16x9g/positions/contested_{14x7,16x9}_gen04g.json` | the contested drive start positions every drive match uses (plan 051) | screened 2026-10-03 at a3c041d with the Gumbel loop's gen04 | 500 biased random seeds per board (the file stores seeds and bias, so the engine rebuilds the positions under the current rules) |
| `runs/value_bench/{g05_gen06,d1k_gen01}.jsonl` | pre-v9 value benchmarks (plan 056) | 2026-10-05 | not readable by v9 code |

## Pre-v9 corpora (not usable)

Every corpus before 2026-10-06 is unreadable by the current code: schema v9's action layout
(per-player setup, `UseSkill`), 16 new skills, `StripBall`. Runs: `runs/loopmix16x9g056` (plan 056
§7, MC labels), `runs/loopmix16x9g` (Gumbel, plan 053), `runs/loopmix16x9d1k`, `runs/loopmix16x9vl0`,
`runs/loopmix16x9`, `runs/az14x7v6`; their settings are in the plans and memory notes.
