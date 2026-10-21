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
| 47 | v9 gen46 (96x8 + heads) | 8×150 (2123) | 127,446 | 0f27dd1 | 48 local in-process; 302 min | 20x11/7, centre 220; TD rate 0.749 (large boards 0.713); C1; MC labels 298 min in-process (net check overlapping) |
| 46 | v9 gen45 (96x8 + heads) | 8×150 (2112) | 127,586 | 0f27dd1 | 48 local in-process; 353 min (with gen45 drives overlapped) | 20x11/7, centre 220; TD rate 0.744 (large boards 0.703); C1; MC labels 306 min in-process (shared the GPU with the 196-min net check) |
| 45 | v9 gen44 (96x8 + heads) | 8×150 (2114) | 124,861 | 33d66aa (labels 0f27dd1) | 48 local, 2 sidecars; 333 min | 20x11/7 capacity, centre 220; TD rate 0.738 (20x11/20x10/18x11 0.695); C1; the loop paused at STOP before mc-label (2026-10-19 14:48-20:57) for plan 066; **MC labels 216 min in-process (plan 066, first in-process generation step), 48 label streams, ~580 samples/min** |
| 44 | v9 gen43 (96x8 + heads) | 8×150 (2098) | 127,617 | cf83459 | 48 local, 2 sidecars; 506 min | **first 20x11/7-capacity corpus** (plan 065, exp076): `--size-centre 220 --size-max-area 220`, 22 boards from 12x6/3 to 20x11/7 (20x11/7 255 records, 20x10/7 241, 18x11/7 220, 16x9/6 79); kickoff table on; TD rate 0.75 (20x11 0.68); C1; MC labels 319 min at 96 label streams (swapped the box; 48 from gen45). The first 16x9 attempt at gen44 died at 18:11 when the disk filled. |
| 43 | v9 gen42 (96x8 + heads) | 8×400 (5872) | 200,979 | 61cf72b | 48 local, 2 sidecars; 620 min | C1; MC labels 317 min; the last 16x9-capacity corpus |
| 42 | v9 gen41 (96x8 + heads) | 8×400 (5816) | 200,051 | dde8a82 | 48 local, 2 sidecars; 392 min | C1; MC labels 238 min |
| 41 | v9 gen40 (96x8 + heads) | 8×400 (5808) | 198,819 | 36b8597 | 48 local, 2 sidecars; 230 min | C1; MC labels 129 min |
| 40 | v9 gen39 (96x8 + heads) | 8×400 (5820) | 199,021 | 36b8597 | 48 local, 2 sidecars; 347 min, shared with gen39's drives | C1; MC labels 126 min |
| 39 | **v9 gen38 (96x8 + heads)** | 8×400 (5796) | 199,981 | bff9be0 | 48 local, 2 sidecars; **381 min** | **the first corpus the 96x8 generates** (exp074); C1; MC labels 238 min |
| 38 | v9 gen37 (64x6, the last) | 8×400 (5801) | 200,702 | eadbe8a | 48 local, 2 sidecars; 163 min | C1; **labelled by the 96x8** shadow96h_gen37 (exp074 swap), 208 min |
| 37 | v9 gen36 (64x6) | 8×400 (5804) | 204,997 | 02fe686 | 48 local, 2 sidecars; 209 min, shared with gen36's drives | C1 |
| 34 | v9 gen33 | 8×400 (5791) | 201,720 | 8ba1114 | 48 local, 2 sidecars; 200 min, shared with gen33's drives | C1 |
| 33 | v9 gen32 | 8×400 (5841) | 197,940 | 8ba1114 | 48 local, 2 sidecars; 252 min, shared with exp072's drives | C1 |
| 32 | v9 gen31 | 8×400 (5770) | 199,226 | 3599c01 | 48 local, 2 sidecars; 162 min | C1; MC labels 176 min (shared with exp072's drives, after the exp072 pause) |
| 31 | v9 gen30 | 8×400 (5796) | 200,468 | 3599c01 | 48 local, 2 sidecars; 192 min, shared with gen30's drives | C1; TD/drive 0.860 |
| 30 | v9 gen29 | 8×400 (5821) | 197,411 | 3599c01 | 48 local, 2 sidecars; 160 min | C1; TD/drive 0.854; MC labels via the hub 96 min |
| 29 | v9 gen28 | 8×400 (5829) | 202,556 | 3599c01 (runtime stamp 8d4f862) | 48 local, 2 sidecars; 164 min | C1; TD/drive 0.861; MC labels via the hub 100 min |
| 28 | v9 gen27 | 8×400 (5758) | 197,775 | 630c8d4 | 48 local, 2 sidecars; 220 min, shared with gen27's drives | C1 |
| 27 | v9 gen26 | 8×400 (5839) | 203,732 | 6b5899a (runtime stamp 630c8d4) | 48 local, 2 sidecars; 164 min | C1; TD/drive 0.857; MC labels via the hub 99 min; the window for exp071 |
| 26 | v9 gen25 | 8×400 (5806) | 197,712 | 6b5899a (runtime stamp 7efb5ad) | 48 local, 2 sidecars; 160 min | C1; TD/drive 0.862; MC labels via the hub 96 min |
| 25 | v9 gen24 | 8×400 (5826) | 200,044 | 6b5899a | 48 local, 2 sidecars; 169 min, shared with gen24's drives | **first corpus under C1** (`cfgs/chance_all_partial_gen.toml`: corrected roll models + partial chance backup); TD/drive 0.858 |
| 24 | v9 gen23 | 8×400 (5818) | 194,240 | b13f105 (runtime stamp 6b5899a) | 48 local, 2 sidecars + laptop part; 147 min | the **last corpus under `gumbel16_f1000_gen`** (the C1 switch came at its end); MC labels via the hub 94 min |
| 23 | v9 gen22 | 8×400 (5802) | 191,500 | b13f105 (runtime stamp e259aeb) | 48 local, 2 sidecars + laptop 10; 142 min, shared with exp070 A1/B1 | TD/drive 0.855; MC labels via the hub 71 min (shared with exp070) |
| 22 | v9 gen21 | 8×400 (5795) | 197,046 | b13f105 (runtime stamp of a later docs commit) | 48 local, 2 sidecars + laptop 10; 137 min, shared with gen21's drives | TD/drive 0.848 |
| 21 | v9 gen20 | 8×400 (5777) | 195,606 | 3bb2b12 (runtime stamp; game code b13f105) | 48 local, 2 sidecars + laptop 10; 104 min | TD/drive 0.848; MC labels via the hub 48 min |
| 20 | v9 gen19 | 8×400 (5772) | 195,368 | 44b1109 (runtime stamp of a docs commit; game code b13f105) | 48 local, 2 sidecars + laptop 10; **104 min** | TD/drive 0.860; MC labels via the hub 49 min |
| 19 | v9 gen18 | 8×400 (5832) | 197,735 | b13f105 | 48 local, 2 sidecars + laptop 10; 133 min, shared with gen18's drives | TD/drive 0.862 |
| 18 | v9 gen17 | 8×400 (5772) | 194,844 | b13f105 (plan 062; chance toggles off) | 48 local, 2 sidecars + laptop 10; **108 min** | TD/drive 0.856; **first MC labels via the hub** (plan 062): the local label worker was refused (commit mismatch, see plan 062), so the laptop labelled it all on tract (95 min) |
| 17 | v9 gen16 | 8×400 (5792) | 201,135 | c98e13e (chance toggles off) | 48 local, 2 sidecars; 188 min | TD/drive 0.857 |
| 16 | v9 gen15 | 8×400 (5762) | 194,699 | 71827c9 (relaunch e423b43) | 48 local, 2 sidecars; 223 min, shared with gen15's drive benchmark | TD/drive 0.854 |
| 15 | v9 gen14 | 8×400 (5761) | 197,877 | 71827c9 | 48 local, 2 sidecars; 181 min | TD/drive 0.853 |
| 14 | v9 gen13 | 8×400 (5795) | 199,445 | 71827c9 | 48 local, 2 sidecars; 181 min | TD/drive 0.851 |
| 13 | v9 gen12 (back on τ=100) | 8×400 (5814) | 198,002 | 71827c9 | 48 local, 2 sidecars; 282 min, shared throughout with gen12's drive benchmark | TD/drive 0.851 |
| 12 | v9 gen11 (last τ=50 net) | 8×400 (5816) | 198,405 | f75eda5 | 48 local, 2 sidecars, laptop 10; 109 min | |
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
| `runs/value_bench/v9_20x11_val.jsonl` | **the 20x11 value benchmark** (plan 065, `VALUE_BENCH_2`): 3023 states from the v9 loop's gen44 held-out shards (4, 7) on 20x11/7 (1349), 20x10/7 (825) and 18x11/7 (849); MC truth = 48 policy-only playouts under gen43 | `GEN=gen44 BOARDS=20x11,20x10,18x11 BOARD_SIZE_W=20 BOARD_SIZE_H=11 BOARD_PLAYERS=7 scripts/value_bench_build_v9.sh runs/loopmix16x9v9 models/az_v7/bbnet_mix16x9v9_gen43.onnx runs/value_bench/v9_20x11_val.jsonl` (2026-10-19, 33d66aa) | `override-audit --all --decisions 3000 --playouts 48 --seed 58100`; gen44's val shards are never trained on, so it is clean from gen44 on; built with `target/20x11` |
| `runs/loopmix16x9v9/positions/contested_20x11_gen43.json` | the 20x11/7 contested drive set (plan 065 §4): 142 of 300 candidates (`positions --board 20x11/7 --count 300 --seed-base 72000000 --min-turns-left 4`) with an attacker TD rate in 0.25-0.75 over 4 playouts each | screened 2026-10-18/19 at cf83459 by v9 gen43 (96x8 + heads) vs itself, C1, 1000 descents, on the loop's hub | attacker TD rate over all candidates 0.681 |
| `runs/loopmix16x9g/positions/contested_{14x7,16x9}_gen04g.json` | the contested drive start positions every drive match uses (plan 051) | screened 2026-10-03 at a3c041d with the Gumbel loop's gen04 | 500 biased random seeds per board (the file stores seeds and bias, so the engine rebuilds the positions under the current rules) |
| `runs/value_bench/{g05_gen06,d1k_gen01}.jsonl` | pre-v9 value benchmarks (plan 056) | 2026-10-05 | not readable by v9 code |

## Pre-v9 corpora (not usable)

Every corpus before 2026-10-06 is unreadable by the current code: schema v9's action layout
(per-player setup, `UseSkill`), 16 new skills, `StripBall`. Runs: `runs/loopmix16x9g056` (plan 056
§7, MC labels), `runs/loopmix16x9g` (Gumbel, plan 053), `runs/loopmix16x9d1k`, `runs/loopmix16x9vl0`,
`runs/loopmix16x9`, `runs/az14x7v6`; their settings are in the plans and memory notes.
