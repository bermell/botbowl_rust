# Experiment registry — question, setup, result, conclusion

Newest first. Each entry: the question, how to reproduce it (commit, script, settings), what came
out and what was decided. Full write-ups live in the plans; head-to-head numbers are also in
[MATCHES.md](MATCHES.md), per-net numbers in [NETS.md](NETS.md), corpora in [DATA.md](DATA.md).
**Keep this current:** add the entry when an experiment starts, the result and conclusion when it
ends (see the root CLAUDE.md).

## v9 rules (2026-10-06 on)

### Hub and `prepare` at cluster scale, fake workers (plan 069 item 6), run 2026-10-23

- **Question:** does the hub keep up with thousands of streams (generate trajectories, 32-sample label items), and how do `prepare` and the trainer's loader scale with a 10-100x corpus?
- **Setup:** commits a6b6d87 (before) and 7d7f487 (hub fix), 20x11/7 build in `target/hubscale`; `scripts/hub_scale.sh OUT` (`botbowl-fake-worker`: real protocol, canned results — 48 heuristic 20x11 games, ~370 KB a line — sleeps for games; hub on ports 14100+); `scripts/prepare_scale.sh OUT CORPUS "1 3 10"` on a 1,800-game heuristic 20x11-mix corpus (`dataset --seed 70000 --mcts-iters 200 --size-centre 150`). Box shared with a GPU tuning run: CPU numbers approximate.
- **Result:** tables in plan 069 item 6. Before the fix, 4,030 streams at 200 games/s gave dispatch p99 1.18 s and ~600 games/s overloaded the hub; after, p99 38 ms and ~1,250 games/s ceiling. Labels: ~0.5 ms per item at 1,000 items/s; the hub holds ~0.18 GB RSS per GB of label input and parses ~1.6 s/GB at submit. `prepare` linear, ~12k samples/s, flat memory; the loader falls 4x (0 workers) once `spatial.npy` outgrows the page cache.
- **Conclusion:** per-result inflate/peek moved off the hub's lock (7d7f487); realistic cluster loads (67-270 games/s) are well inside the ceiling. Open: shard writes still under the lock (watch on Lustre); training data on flash tier / NVMe / page cache, and loader workers sized on the H100.

### <a id="exp081"></a>exp081: bf16 / fp16 inference numerics vs fp32, offline on CPU (plan 069 item 4), run 2026-10-22

- **Question:** the H100 binds generation (exp080), and tensor-core precision is worth ~4-7× throughput over fp32 — how much do reduced-precision forwards change the net's outputs and decisions?
- **Setup:** gen48 (`models/az_v7/bbnet_mix16x9v9_gen48.pt`), eval mode, on gen48's held-out shard 4 (`prepare --policy-target cq --tau 100`, 11,726 samples over 21 boards, the 20x11 mix); fp32 vs `torch.autocast("cpu", bfloat16 | float16)`; script `scripts/exp081_precision_study.py` (`train/.venv/bin/python scripts/exp081_precision_study.py NET.pt PREPARED_DIR`), output `runs/exp081/result.txt`. CPU autocast approximates GPU tensor-core kernels (fp32 accumulation there), so this bounds rather than reproduces the H100's numbers.
- **Result:**

  | | value \|Δv\| mean / max | value MSE vs MC label | policy KL(fp32‖·) | top-1 move changes | … of them at a near-tie (top-2 log-margin < 0.1) |
  |---|---|---|---|---|---|
  | bf16 | 0.0020 / 0.016 | 0.06722 → 0.06723 | 8.1e-5 | 0.74% (87) | 85 of 87 |
  | fp16 | 0.00026 / 0.0025 | 0.06722 → 0.06722 | 1e-6 | 0.05% (6) | 6 of 6 |

  For scale: the search at 1000 descents disagrees with itself at 4-6% of decisions (exp079), and the value bench resolves ~0.003 in RMS.
- **Conclusion:** both are far below the search's own noise; **fp16 is ~8× closer to fp32 than bf16 at the same H100 tensor-core rate**, with no overflow on this net — prefer fp16 for inference (bf16 for training is fine). Confirm on the cluster with the real kernels: value benches, net check and paired drives fp32-bot vs fp16-bot.

### <a id="exp080"></a>exp080: CPU per descent and per forward, profiled and cut (plan 069 item 1), run 2026-10-21/22

- **Question:** on a GH200 (72 Neoverse V2 cores per H100), does the CPU (search + encoder + batcher) or the GPU bind generation and labelling, and how much search CPU can be cut without changing games?
- **Setup:** branch `worktree-agent-a58c85de114fdbc68` (ef820ec … 7bf76b8, not merged; review pending). 20x11/7 capacity, `cfgs/chance_all_partial_gen.toml`, 1000 descents, random-start `--next-drive`; evaluators gen48 in-process (1 stream, cuDNN deterministic), a new debug `--nn-backend null` (value 0, uniform priors, encoder runs) and the heuristic; `perf stat` user+kernel instructions on one pinned CPU (counting after libtorch's load), LBR call-graph split; µs at a measured 5.2 ± 0.5 instructions/ns (IPC 1.24-1.39 at ~3.9 GHz). Scripts `scripts/perf_search_bench.sh` (new knobs), `perf_label_bench.sh`, `exp080_cpu_matrix.sh`; outputs `runs/exp080/`.
- **Result, generation per forward (20x11, real net):** batch-independent CPU **1.11M instructions (214 µs) → 0.68M (130 µs)** after five game-identical cuts (search 0.788M → 0.553M, encoder 0.263M → 0.065M, client 0.06M) + ~0.2M per batch for the batcher/driver. 0.613 forwards per descent, 471 per decision; the forward memo serves 49% of evaluation calls. Per descent −33% on 16x9, 20x11 and the mix; null net −45.5%, heuristic −29%. Same trajectory hashes before/after (heuristic, null, real net on three boards, labels). `-C target-cpu=native` another −5.2% (not committed; use `neoverse-v2` on Grace).
- **Result, mc-label:** 186 forwards per sample (23 per playout); batch-independent CPU per forward 0.65M → 0.33M (64 µs); generation/labelling CPU per forward ≈ 2.0.
- **Hot spots (before):** pathfinder 48% (move offerings 29%, the encoder recomputing the same paths 19%), node teardown 8%, SipHash 7%. Next identical cuts: pathfinder arena (−10-15%), bulk teardown (−5-8%), f32 straight into the batch slot. Game-changing proposals: a faster hasher (−6-8%, pairs with determinism), more streams per process.
- **Streams:** instructions per descent flat 1→96 streams (null net); IPC 1.26 → 1.13; ~16 MB RSS per 20x11 stream. **Anomaly:** in-process with the GPU, task-clock per forward 870-1106 µs at an effective 0.9-1.3 GHz — threads scheduled without retiring instructions; explains most of the loop's apparent "1.3 ms CPU per forward"; unexplained (driver blocking vs SMT contention with the live loop) — measure on Grace.
- **Projection (one V2 core ≈ this box's core at 0.7-1.2×, no SMT, batches ≥64, ~6 cores for batcher/driver/hub, no anomaly):** 66 cores × 7.7k forwards/s ≈ **500k positions/s** (300-700k) vs the H100's ~40k/s fp32 or 150-300k bf16 → **the GPU binds** (~10× at fp32, ~2× at bf16); if the anomaly holds on Grace, CPU ≈ 100k/s and binds against bf16. ~1,000 streams (~16 GB of trees) to fill an H100.
- **Conclusion (provisional):** precision and batch size on the H100 are worth more than further search cuts; measure the anomaly first on Grace.

### <a id="exp079"></a>exp079: divergence-gated drives (plan 068), run 2026-10-21

- **Question (the user, 2026-10-21):** can the drive benchmark play out only the drives where the two nets take different moves, to save compute as the reference moves to a close net (gen48)?
- **Setup:** branch `worktree-agent-a250a0f5d8ca950d1` (e7407b0..4f4c40e, not merged; review pending). Lockstep measurement (`botbowl-ui drive-agree`, both bots asked at every decision of a shared trajectory) on 40 positions of `contested_20x11_gen43`, 1000 descents, C1, in-process; validation on `contested_16x9_gen04g` through the real `eval --drive-gate`. Outputs `runs/exp079/`.
- **Result:** pairs already share their dice, so a pair with identical play scores exactly 0.5 (9 of 9 identical pairs in the null control). But **a net disagrees with itself at 1000 descents**: per-decision agreement gen48 vs gen48 0.937 ± 0.010 (shipped preset; 0.959 with the deterministic tie-break preset), so 88-90% of its own pairs diverge on 20x11; gen47 vs gen48 0.933, gen45 vs gen48 0.905. Bare-policy agreement does not predict search agreement (gen48 vs itself: policy never diverges in 40/40, the search does in 35/40); a 64-descent prescreen caught 1 of 5. Null control gen48 vs gen48 gated: s_div 0.485 ± 0.061 (26 pairs); gen45 vs gen48: p_div 0.875, s_div 0.439 ± 0.033, all-pairs 0.446 ± 0.029 (56 pairs), H0 at 36 pairs all-pairs vs 39 gated. Self-disagreement sources: Gumbel-root ties in HashMap order (fixed for deterministic `tie_break`; `cfgs/chance_all_partial_det.toml`) and recon_mcts backprop order at equal depth (not fixed). **Found on the way: the NN forward memo was keyed by state only, so a thread shared by two nets could serve one net's forward to the other** (keyed per evaluator on the branch). **Impact on master: none measured** — after every search `MctsBot::summarise` reads the searching bot's own root on the caller's thread, so the other bot is never asked about the cached state: an audited gen48-vs-gen45 drive rung (24 drives, 1,607 searches, 1.71 M memo probes) had 832,593 same-net hits and **0 cross-net hits**; the bug only bites the new `drive-agree` lockstep tool and the web server's display readout. Past MATCHES rows stand. Merge via `review-integration-067-068` (4b892a9: plan 067 = v21, plan 068 = v22; 835 tests pass) at the next STOP.
- **Conclusion:** the gate is correct but buys no compute on this search (`report` mode is free and informative; `sprt` mode only rescales the hypothesis). The real lever is search determinism (backprop order), a search change with its own gate.

### <a id="exp078"></a>exp078: 8 vs 16 MC-label playouts (plan 067), planned 2026-10-19

- **Question:** is the value head limited by label noise? Estimated from gen44: an 8-playout label's noise variance ≈ 0.049 (0.036-0.059 for P(score) 0.6-0.85) against a val value MSE of 0.080, leaving ≈ 0.031 (RMS 0.18) for the net, which matches the value bench (RMS 0.178/0.186 vs 48-playout truth). Best checkpoints land early (gen44: step 8000, epoch 0). Doubling playouts costs ~+3.7 h per generation on this box; doubling the games for the same noise cut costs ~+9 h.
- **Setup (planned, the user approved 2026-10-19):** at the next plateau, top the window's labels up from 8 to 16 playouts (exact: playouts 8..15 of each sample's own seed stream, `mc-label --top-up 8`, plan 067; both arms select checkpoints on the same 16-playout val labels; `scripts/exp078_label_topup.sh`, ~15-17 h with the loop stopped), train two nets from the same parent with the loop's recipe (arm A 8, arm B 16), with a model-selection rule that is fair across label noise. Judge on both value benches, net check, and optionally paired drives. The top-up also measures the 8-playout label noise directly (Var(a−b)/2 over two independent 8-playout means).
- **Measured label noise (2026-10-19, plan 067 branch bf220f7…6b2b50f, `mc-label --top-up 8` under gen43 on tract, 63 trajectories / 3,891 samples of `gen44/mc/shard4.jsonl`, `scripts/label_noise.py`):** Var(a−b)/2 of two independent 8-playout means = **0.0408 ± 0.0041 per sample, 0.0364 ± 0.0044 per-drive weighted** (as val_value weights), i.e. ~47% of gen44's val_value 0.077; by board 20x11/7 0.052, 18x11/7 0.047, 20x10/7 0.044, 18x9/6 0.038, 20x9/7 0.028; prior − added offset −0.0023 ± 0.0040 (none). Cross-check on the 324 states shared with `v9_20x11_val`: 0.0525 ± 0.0053 vs the bench's own mc_se²·48/8 0.0534 ± 0.0022. A 16-playout label halves it to ~0.018.
- **Decision rule:** if arm B wins the 20x11 value bench clearly (paired, beyond its SE) without losing the net check, the loop moves to 16 playouts.

### <a id="exp077"></a>exp077: in-process libtorch inference vs the Python sidecar (plan 066), run 2026-10-19

- **Question:** can an in-process Rust batch queue running libtorch replace `scripts/nn_server.py` with **bit-identical** outputs, and what does it buy on this box (the HPC case: the sidecar's ~1.5 ms of Python per batch would cap an H100)?
- **Setup:** branch `worktree-agent-a2dfb9b346660884c` (1f1ab20 … 9a0b987 on e0985e6), feature `torch` (C++ shim against `train/.venv`'s torch 2.13.0+cu126, cuDNN 91002), net `bbnet_mix16x9v9_gen44` exported with `python -m bbnn.export_ts` and optimised at load with `torch::jit::optimize_for_inference` (the sidecar's `--jit auto` route); canvas 13x22, capacity 20x11/7. Gates: `scripts/parity066.sh` (A deterministic cuDNN, B production `benchmark=True`; 120,000 states from gen44 shards 0-7 over 22 boards, 6,959 batches across all 12 buckets, policy and value-only), `scripts/gatec066.sh` (C: one stream, deterministic, `dataset` 3 random-start drives `--next-drive` C1 1000 descents + `mc-label`). Throughput: `perf_gen_steady.sh` 20x11 mix, C1, GPU exclusive (loop paused), ~10-14 min windows; `perf_label.sh` 48 trajectories × 8 playouts. Outputs `runs/exp077/`.
- **Result, parity:** trace without `optimize_for_inference` differs (96,042 / 120,000 values, max 3.1e-6); the optimised route: **0 differing bits over 455,228,416 output floats** in A and in B, for in-process vs sidecar, sidecar vs sidecar (the floor is 0 on this box), in-process vs itself and lane 1 vs lane 0. C: mc-label byte-identical; dataset identical in every sample, visit, value, tree block and outcome, only the telemetry counters `meta.extra.eq_checks`/`eq_rejects` (hash-bucket collision checks, a function of std `RandomState` seeding) differ.
- **Result, throughput (this box, GTX 1060):** GPU samples/s 2,810 (2 sidecars, 48 streams, mean batch 11.8) → **3,150 in-process (+12%, mean batch 23.3)**; 64 streams 3,038, 96 streams 3,199 (RSS 11.5 GB, 1.3 GB left): keep 48. RSS −1.9 GB (sidecars gone). mc-label 532 → **608 samples/min (+14%)**, byte-identical labels. Decisions/min 137 → 202 is over different seed sets in a short window; the GPU-samples number is the measure.
- **Review (2026-10-19, 3957986 … f453407):** fixed a 5th-net-loaded-kills-the-first bug (lane/capture streams came from libtorch's 32-stream pool), the worker's 16 MiB websocket frame limit (v20 model frames carry the `.ts`), a lost wake-up on a dying queue, and gate C now also compares without the hash-collision counters (0 other differences). Parity re-run after the fixes (12,000 states, det + prod): 0 differing bits in all 10 comparisons. `cargo test --workspace` 820 passed. Open: the in-process registry never evicts (~0.4-0.6 GB GPU per extra net). Merged c99a21f.
- **Conclusion:** passes the bit-parity gate; merged; then the switch (plan 066 phase 5: `.ts` beside every ONNX in `train_loop.sh`, `--features torch` build, hub protocol v20) is the user's call.

### <a id="exp076"></a>exp076: the loop moves to 20x11/7 (plan 065), planned 2026-10-18
- **Switched 2026-10-18 18:14 (cf83459):** the gen44 16x9 generate died at 2481/3200 games when a review build filled the disk (FATAL 18:11); the loop was relaunched straight onto `scripts/launch_plan065.sh`, so **gen44 is the first 20x11/7 generation** (champion gen43, 96x8 + heads), 150 games a shard, canvas 13x22, GPU 100% at ~2850 forwards/s. The 20x11 screen (300 candidates x 4, gen43 vs itself) runs on the hub alongside; the 20x11 value bench is built from gen44's held-out shards (`GEN=gen44`).
- **gen44 (2026-10-19):** 2098 records (8×150) in 506 min; MC labels 319 min at 96 label streams, which held 9.9 GB and swapped the 16 GB box (the background net check of gen43 was killed, no number) → 48 label streams from gen45 (f816254). Trained 70 min, best at step 8000 (epoch 0). Value bench (16x9 series) 0.178, −1.1% vs gen43 (the run's best); **20x11 bench (`v9_20x11_val`, frozen from gen44's val shards with gen43) 0.186 vs gen43's 0.192, −3.0%**, bias +0.035 → +0.018. Relaunched 09:15 at 33d66aa with both drive sets (16x9 + `contested_20x11_gen43`); gen45 is the first benchmark generation on 20x11.

- **Question:** does the v9 loop keep improving when its boards grow from ≤16x9/6 to a mix centred on
  20x11/7 (the user, 2026-10-18: "bigger boards are the direction")? 7 players turn the kickoff table on.
- **Setup:** `scripts/launch_plan065.sh` continues `runs/loopmix16x9v9` from the champion at the switch
  (planned: gen44, the 96x8 + heads lineage), first 20x11 corpus gen45. Capacity 20x11/7
  (`target/20x11`), sizes `--size-centre 220 --size-temperature 0.3 --size-floor 0.2 --size-aspect 1.5-2.8
  --size-max-area 220 --size-min-area 70 --cells-per-player 26` (20x11/7 11.2%, 7-player boards 50.9%,
  16x9/6 4.7%), sidecar canvas 13x22, `GAMES_PER_SHARD` per plan 065 §5; everything else as
  `launch_plan058.sh` (C1, MC labels ×8, cq τ 100, value weight 0.36, lr 5e-5, `--freeze-bn`).
  Commit: the merge of plan 065 (capacity-portable `GameState` load, kickoff Time-out fix).
- **Measure:** drives every 3 generations vs `bbnet_mix16x9d1k_gen04_v9` on `contested_16x9_gen04g` (the
  curve so far) and on a 20x11 contested set screened with the switch champion (plan 065 §4); value bench
  v9 (series) + `v9_20x11_val` (`VALUE_BENCH_2`, frozen from gen45's held-out shards); net check on
  16x9 + 20x11 decisions; per-board TD rate in `status.md`.
- **Decision:** keep going while the 16x9 drive curve holds (no drop beyond its noise vs gen42/45) and the
  20x11 value bench improves; if the 16x9 curve drops, raise the floor or lower the centre (plan 042's
  dilution answer), not abandon the size.

### <a id="exp075"></a>exp075: adversarial probes of gen36 (64x6) and gen39 (96x8 + heads), run 2026-10-16/17

- **Question:** where are the value/policy net and the search wrong in ways averaged benchmarks hide —
  skills that lower their owner's value, mirror images valued differently, dominated edits that move
  the value the wrong way, a lone player whose best square depends on where he starts? (plan 064)
- **Setup:** `botbowl-ui probe <suite>` (plan 064 §1, tract/CPU only, 2 threads, nice 19),
  `scripts/exp075_probes.sh runs/exp075 SUITE models/az_v7/bbnet_mix16x9v9_gen{36,39}.onnx`, seed 64000,
  board 16x9/6 (and 14x7/4 for symmetry, skills, dominance). Base states: 150-300 decisions from
  `runs/loopmix16x9v9/gen39/shard4.jsonl` (val shard; replayed) + 150-300 positions of
  `runs/loopmix16x9g/positions/contested_16x9_gen04g.json` (14x7: `contested_14x7_gen04g`). Searches:
  `cfgs/chance_all_partial.toml` at 250 / 1000 / 4000 descents. Commits per suite in each
  `*.summary.json` header: symmetry e6a198d, lone mover / stability 8aa7c25, skills / dominance /
  hill-climb 880fd51. Value bench: `scripts/exp075_vbench.sh runs/value_bench/v9_gen01_val.jsonl ...`
  (`value-bench --symmetrise`, e6a198d). Noise floor `tau` = p99 of `|ΔV|` for skills with no rules.
- **Result (gen36 → gen39):**
  - **Skills:** pooled beneficial edits violate beyond `tau` 2.0% → 1.7% (placebo 0.5%). Worst
    skills: Wrestle 7.8% → 8.3%, Jump Up 11.8% → 4.8%, Guard 1.9% → 2.5%, Catch 0.9% → 2.3%. AV+1
    lowers an own player's side 85% → 86% of the time (small, consistent). Never-fielded skills are
    invisible (placebo-level).
  - **Symmetry:** y-flip `|ΔV|` mean 0.029 → **0.043**, max 0.20 → 0.35; policy argmax not the
    mirrored move 10.2% → 8.8%; colour swap 0 exactly; a used player's movement moves V by 0.027 →
    0.022 (max 0.23). Test-time y-flip averaging: value bench RMS **−1.2% (gen36) / −1.9% (gen39)**,
    paired, > 4 σ.
  - **Dominance (beyond `tau`):** knock down an opponent 4.7% → 4.0% (worst −0.21 → −0.30), carrier
    one square on in open field 12% → 9%, add an own player 3.8% → 1.1%; violations 2-3x more common
    at activation decisions than at turn starts.
  - **Hill-climb:** 80 mutations raise the worst skill violation from 0.029 → 0.071 (gen36) and
    0.044 → 0.109 (gen39); 48% → 60% of climbs end beyond `tau`.
  - **Lone mover (12 scenarios, 140 starts):** policy keeps one destination 62.5% → 66%, search@1000
    53.5% → 51%; mostly a frontier effect (the wanted square is out of reach; moves use the full MA),
    but with the mode within 5 squares still only ~42-54%, and the agents GFI in quiet positions
    16-25% of starts.
  - **Stability (40 decisions):** 1000 vs 4000 descents pick different moves 40% → 20% (median Q gap
    0.06 / 0.09); a repeat at 1000 differs 2.5% → 10%.
- **Conclusion:** failures are skill- and context-specific (rare-event skills, the mirror, irrelevant
  inputs, activation context), not capacity: the 96x8 fixes some (Jump Up, adding players) and worsens
  others (mirror, Catch, Guard). Proposed (plan 064 §7, none adopted): symmetrised inference, an
  invariance loss, rare-skill credit (paired regulariser or event-centred positions), decision context in
  the encoding, lone mover v2, and a standing `probe_check.sh`.

### <a id="exp074"></a>exp074: the 96x8 + heads lineage generates (trial), started 2026-10-15

- **Question:** does a 96x8 + bigger-heads net, once it generates its own data (its own prior in the cq
  targets, its own search), pull ahead of the plateaued 64x6 lineage? The shadow comparisons (plan 063)
  cannot answer it: the shadow's policy target embeds the 64x6 generator's prior (the user, 2026-10-15).
- **Setup:** the loop stops after gen37's shadow step; `champion.txt` / `latest_net.txt` are pointed at
  `models/az_v7/shadow96h_gen37.{onnx,pt}` and the loop relaunches with the shadow step off
  (`launch_plan058.sh`). From gen38 the loop's nets (`bbnet_mix16x9v9_genNN`) are 96x8 + heads,
  warm-started from the previous one; everything else unchanged (C1 search, 400x8 games, MC labels,
  cq τ 100, value weight 0.36). Reference: the 64x6 **gen36** (the best of the plateau), frozen.
- **What actually happened (2026-10-16 01:25):** the STOP was placed a few seconds after the loop passed its last check, so gen38 was *generated* by the 64x6 gen37. The swap lands before gen38's mc-label: gen38 is labelled by and fine-tuned from shadow96h_gen37 (on 64x6-generated data, like a shadow step), and **gen39 is the first corpus the 96x8 generates**. First fair read: gen42.
- **Measure:** every 3 generations (gen39, gen42), drives of the 96x8 net vs 64x6 gen36, C1 both sides,
  1000 descents, 300 fixed pairs per board; plus the loop's own benches. Generation ~2x slower.
- **Result (2026-10-18):** gen42 (96x8 + heads, 3 generations on its own data) vs the 64x6 gen36, C1
  both sides, 300 fixed pairs: equal descents (1000 v 1000) 14x7 0.506 ± 0.013, 16x9 0.517 ± 0.012;
  **equal compute (1000 v 2000) 16x9 0.512 ± 0.015**. Value bench 0.179 vs the 64x6 line's best 0.192.
  **Conclusion:** the 96x8 is at least as strong as the 64x6 at equal compute (slightly ahead, not
  significant) with a clearly better value head; it pays its inference cost but does not yet win by it.
  Kept as the generator for the move to 20x11/7 (plan 065), where its larger receptive field and
  capacity should matter more.
- **Decision:** keep the 96x8 lineage if it pulls clearly ahead of gen36 within 3-4 generations;
  otherwise go back to the 64x6 lineage (gen36/37) — the only loss is time.

### <a id="exp073"></a>exp073: the 96x8 nets fine-tuned in the loop's regime, 2026-10-14/15

- **Question:** exp072's scratch 96x8 nets fit gen10-31's average target. Fine-tuned on exactly the
  window and recipe the loop's gen33 used (gen31-33, MC labels, cq τ 100, lr 5e-5, 3 epochs,
  `--freeze-bn`, value weight 0.36 per-drive), does a 96x8 start beat the 64x6 lineage?
- **Setup:** `scripts/exp073_bignet_finetune.sh` (commit in `runs/exp073/status.md`): arms A96x8 and
  B96x8h from `runs/exp072/<arm>/net.pt`; scored on gen33's held-out shards (absorption/test), the value
  bench paired vs gen33, and drives vs gen33 under C1 at 1000 descents, **300 fixed pairs per board**
  (no SPRT). The loop pauses after gen34's generate for the ~2 h of fine-tunes and is relaunched by the
  script; the drives share its hub.
- **Result, single-net (2026-10-15 00:38, 457120f):** fine-tunes took 39-40 min each (the 64x6 one ~27).

  | on gen33's held-out shards / the value bench | loop gen33 (64x6) | A96x8 fine-tuned | B96x8h fine-tuned |
  |---|---|---|---|
  | KL(target‖net) | **0.053** | 0.084 | 0.085 |
  | log P(played) / P(played) | **−0.695 / 0.681** | −0.715 / 0.674 | −0.716 / 0.673 |
  | value MSE | 0.0785 | 0.0760 | **0.0743** |
  | value bench RMS (paired vs gen33) | 0.193 | 0.189 (−2.2%) | **0.184 (−4.3%, ~3 SE)** |

  One fine-tune makes the 96x8 value heads clearly better than gen33's (B: −4.3% on the bench, −5.4%
  test MSE), but closes only part of the policy gap (KL 0.104 → 0.085 vs 0.053).
- **Drives vs gen33 (2026-10-15, C1 both sides, 1000 descents, 300 fixed pairs per board):** A96x8
  14x7 0.500 ± 0.012, 16x9 0.509 ± 0.014; B96x8h 14x7 0.507 ± 0.013, 16x9 0.515 ± 0.014 (+1.1 SE).
- **Conclusion:** after one fine-tune the 96x8 nets play even with gen33 (B slightly ahead on 16x9, not
  significant) at ~2x the inference cost: not adopted yet. The user (2026-10-15): the 96x8 + heads net's
  benches have not converged (gen33's had); keep fine-tuning it on the loop's data as a **shadow
  lineage** (train_loop.sh section 3b, `SHADOW_INIT`, models `shadow96h_genNN`) and pit the two again
  once its benches flatten.
- **Adoption rule:** an arm that beats gen33 on drives (both boards, or one with the other level) and is
  not worse on the value bench / test value MSE becomes the loop's net (the user's call on the ~2x
  generation slowdown).

### <a id="exp072"></a>exp072: a bigger net from scratch on the clean v9 corpora (plan 063), started 2026-10-13

- **Status:** started 2026-10-13 08:35 at e647d02 (`RELAUNCH=1 scripts/exp072_bigger_net.sh`, the loop stopped by `scripts/stop_after.sh "gen31 train done"`); the loop relaunches after the GPU phases and the drives share its hub. The user chose 96x8 and asked to test bigger heads (2026-10-13).
- **Result, training and single-net numbers (2026-10-14 09:38, e647d02):**

  | | 64x6 scratch (C) | 96x8 (A) | 96x8 + heads (B) | loop gen31 (64x6 fine-tune lineage) |
  |---|---|---|---|---|
  | train time | 204 min | 523 min | 556 min | — |
  | val (gen10-30 held-out) policy / value | 0.7717 / 0.0871 | 0.7614 / 0.0816 | **0.7607 / 0.0803** | — |
  | test (gen31 shards 4,7) KL(target‖net) | 0.117 | 0.105 | 0.104 | **0.055** |
  | test log P(played) / P(played) | −0.749 / 0.653 | −0.739 / 0.659 | −0.736 / 0.659 | **−0.701 / 0.680** |
  | test value MSE | 0.0899 | 0.0840 | 0.0830 | **0.0815** |
  | value bench RMS (paired vs gen31) | 0.219 (+14.1%) | 0.196 (+2.1%) | 0.196 (+2.2%) | **0.192** |
  | GPU µs per sample at batch 12 / 24 | 141 / 116 | 263 / 223 | 276 / 235 | 133 / 106 |

  At equal training (scratch on gen10-31) the 96x8 trunk beats 64x6 on every number (value MSE −6%,
  test KL −10%, value bench 0.219 → 0.196) and the bigger heads add a little (value −1.6%). But the
  loop's gen31 — 64x6 fine-tuned generation after generation on the most recent window — beats every
  scratch arm on gen31's own held-out shards (KL 0.055 vs 0.104): the scratch nets fit the average of
  gen10-31's targets, not the current ones. Inference: 96x8 costs ~2.0x per sample on the GPU (less
  than the 2.7x CPU estimate). Drives vs gen31 (C1, 1000 descents) follow on the loop's hub.
- **Drives vs gen31 (2026-10-14, C1 both sides, 1000 descents, SPRT 0.5:0.55 on the loop's hub):**
  C64x6 14x7 0.463 ± 0.029 (H0, 16 pairs), 16x9 0.502 ± 0.022 (H0, 113); A96x8 0.500 ± 0.032 (H0, 38),
  0.508 ± 0.018 (H0, 196); B96x8h 0.471 ± 0.021 (H0, 33), 0.508 ± 0.018 (H0, 188). Every scratch arm
  plays about even with gen31 despite its much worse fit to the current targets (the 14x7 SPRTs stopped
  far too early to read). **Conclusion:** at equal training 96x8 is the better net; the fair question is
  a 96x8 trained in the loop's regime → exp073.
- **Question:** is the 64x6 net the ceiling now? Does a 96x8 trunk, with or without bigger heads,
  trained from scratch on every clean v9 corpus beat the loop's gen31 in play — and does a 64x6
  trained the same way (the control) already do so?
- **Setup:** `scripts/exp072_bigger_net.sh` (commit in `runs/exp072/status.md`). Data: `runs/loopmix16x9v9`
  gen10-31 MC-labelled shards, cq τ=100, value blend 1.0; train = shards 0-3,5,6 of gen10-31 (~2.66 M),
  val = 4,7 of gen10-30 (~0.84 M, checkpoint selection), test = 4,7 of gen31 (~40 k). Arms `C64x6`
  (today's shape), `A96x8` (96x8, today's heads), `B96x8h` (96x8 + `--value-conv 64 --value-pool
  meanmax --value-hidden 128 --policy-conv 96`). Recipe: batch 256, Adam, lr 1e-3, 2000 warmup steps
  then cosine to 0, 10 epochs, val every 5000 steps, restore on val_policy + val_value, value weight
  0.36, per-drive value weight, BN live, 6 loader workers, seed 72. Scored by: test-set absorption vs
  gen31 (`absorb_probe.py`), the value bench (`v9_gen01_val.jsonl`) paired vs gen31, the sidecar's
  forward cost (`nn_server.py --bench`), and paired contested drives vs gen31 under C1 both sides
  (`cfgs/chance_all_partial.toml`, 1000 descents, both position sets, SPRT 0.5:0.55, cap 800/board).
- **Cost (CPU-measured, relative to 64x6):** forward 2.7× (A) / 2.8-2.9× (B), train step 2.5× / 2.7×,
  tract 2.5× / 2.6×; MACs 2.86× / 3.03×. GPU estimate: C ~2.7 h, A ~7.3 h, B ~7.8 h of training.
- **Decision rule (plan 063 §7):** H1 on both boards or H1 on one with the other level; no >1%
  regression on the value bench or test value MSE; A over B unless B is better by >1 SE; if C64x6
  passes too, adopt C first (no throughput cost) and require the 96x8 net to beat it. A 96x8 net
  makes a loop generation ~2.3× longer: adoption is the user's call.
- **Prior:** plan 032 #9 (2026-09-08, old rules): 96x8 from scratch on 2.2 M samples, 0.508 ± 0.043 vs
  its 64x6 twin in 120 games — no gain at that data size and that resolution.

### MC labels through the hub: equivalence and overhead (plan 062), 2026-10-10

- **Question:** does `botbowl-hub job label` (plan 062, protocol v19) write what
  `botbowl-ui mc-label` writes, and what does the hub cost?
- **Setup:** plan 062 at cf4e08c (merged to master as 44ed2f0), release build, default 28x17/11 board, fixture net
  `botbowl-nn/tests/fixtures/tiny.onnx` on tract, 2 threads either way. The corpus is 30
  trajectories, 2008 samples, two shards, one next-drive pair: the first 30 lines of `dataset
  --mode random-start --games 300 --seed 6000 --mcts-iters 4 --evaluator heuristic --next-drive
  --max-steps 2000`. 8 playouts, seed 7. The hub run is a test hub on :13339 with one
  `botbowl-worker --parallel-games 2` and `--chunk-samples 32` (80 items). The box was running the
  live loop (load 12–17).
- **Result:** local 218 s, hub 225 s (+3%, within noise). Both shards byte-identical. An item ships
  ~2% of its line's size on this board (616 KB → 14 KB zstd; ~7.5% on the loop's 16x9 corpus) and
  returns 4 bytes a sample.
- **Conclusion:** the hub adds no meaningful cost. Turning it on in the loop (`LABEL_VIA_HUB=1`)
  is a throughput question for the GPU box plus the laptop, to be read off the first generation's
  phase minutes against the local ~95.

### <a id="exp071"></a>exp071: the train step (rebalance, learning rate), run 2026-10-12

- **Question:** the loop's fine-tune barely moves its own validation loss (plan 058, gen07-22) and the
  policy absorbs ~+0.001 P(played) per generation. Does rebalancing the losses (the gradient probe:
  policy ~3x value at the trunk at τ=100, 4.2x at τ=50) or a larger step move the net more without
  costing the value head?
- **Setup:** `scripts/exp071_train_step.sh` (commit in the status file). Parent v9 gen26, the window the
  loop's gen27 trains on (gen25-27, all C1 corpora, MC labels), the loop's recipe with one change per
  arm: `tau50_vw36` (cq τ=50, value weight 0.36), `tau100_vw36` (value weight 0.36), `lr1e4` (lr 1e-4).
  Control: the loop's own gen27. The loop pauses for the arms (~2-2.5 h of GPU) and the script
  relaunches it. Scored by: the trainer's val movement, absorption on gen27's held-out shards (cq-100
  target for all), the value bench paired vs gen26.
- **Result (2026-10-12 08:52-10:47, commit 630c8d4):**

  | arm | best step | value bench vs gen26 | Δ log P(played) | Δ P(played) | Δ top-1 | Δ val MSE (gen27 held-out) |
  |---|---|---|---|---|---|---|
  | loop gen27 (control: τ 100, vw 0.25, lr 5e-5) | 10000 (ep 0) | −0.3% | −0.0026 | −0.0002 | −0.0003 | −0.0006 |
  | `tau50_vw36` | 7000 (ep 0) | **+1.2%** (worse) | +0.0022 | −0.0021 | −0.0001 | −0.0007 |
  | **`tau100_vw36`** | 29000 (ep 2) | **−1.5%** | −0.0026 | **+0.0011** | −0.0013 | −0.0008 |
  | `lr1e4` | 1000 (ep 0) | +0.6% | −0.0042 | −0.0016 | −0.0014 | +0.0011 |

  Within-run val still barely moves in every arm (val_policy step 250 → final: +0.0005 to +0.0011).
- **Conclusion:** **value weight 0.36 at τ=100 is better** (value bench −1.5% against the control's
  −0.3% on the same step, ~3 SE apart; the only arm with P(played) up; it kept improving into epoch 2).
  Adopted by the loop from gen28's training (`launch_plan058.sh`, `VALUE_WEIGHT=0.36`). **The rebalance
  does not rescue τ=50:** its value head still loses 1.2%, so τ=50's cost is the sharper target itself
  (it amplifies sub-0.1-TD Q noise), not gradient competition. **A larger step (lr 1e-4) is worse on
  every number:** the bottleneck is not step size. Next levers: the target's noise (sharpen only where
  the Q gap beats its uncertainty), net capacity.

### <a id="exp070"></a>exp070: the chance model (plan 061), run 2026-10-11, adopted

- **Status:** A1 + B1 started 2026-10-11 06:23 at the loop's plateau (gen21 drives not above gen18 on either board, value bench flat over gen19-21): `EXTERNAL_HUB=http://127.0.0.1:13337 NET=models/az_v7/bbnet_mix16x9v9_gen21.onnx ARMS="A1 B1" scripts/exp070_chance_model.sh`, on the loop's hub alongside gen22-23 (helper workers share it); gen21 both sides, cap 800 drives per board.
- **A1 result (2026-10-11 08:54):** catch-aware bounces under today's complete backup lose on both boards: 14x7 0.474 ± 0.012, 16x9 0.476 ± 0.012 (400 pairs each, ~2 SE below 0.5; the not-worse SPRT 0.47:0.5 stayed undecided at the cap). Not adopted alone. Consistent with the arm-0 telemetry: the wider bounce node eats budget in dice under `complete` (main line reaching the opponent's turn 73% → 43%). Revisit under the winning backup in C1.
- **B1 result (2026-10-11 09:21):** the partial chance backup wins 14x7, 0.552 ± 0.016 (H1, 121 pairs); 16x9 stopped at H0 after only 18 pairs (0.430 ± 0.045), too few to read. Follow-up B1b: 16x9 alone, 300 fixed pairs, no SPRT, seed 1: **0.517 ± 0.014** (not worse, +1.2 SE). Verdict: partial wins 14x7 and holds 16x9; it is the B winner for C1 (B2-B4 not run yet).
- **C1 vs the B winner (2026-10-11 11:30):** every roll model on top of the partial backup vs partial alone: 14x7 0.490 ± 0.015 (H0, 42 pairs), 16x9 0.506 ± 0.016 (H0, 151 pairs): the corrected dice are neutral once the backup no longer charges for width (they cost ~0.025 under `complete`, A1).
- **C1 vs today's control (2026-10-11 12:02):** every roll model + partial beats `gumbel16_f1000` on both boards: 14x7 0.567 ± 0.026 (H1, 60 pairs), 16x9 0.566 ± 0.027 (H1, 83 pairs). Remaining adoption check: `net_check.sh` under `cfgs/chance_all_partial.toml` (gen22 on gen23, paired with the loop's own control check on the same decisions), running.
- **Net check under C1 (2026-10-11 13:30):** `scripts/net_check.sh models/az_v7/bbnet_mix16x9v9_gen22.onnx runs/loopmix16x9v9/gen23 runs/exp070/net_check_c1/gen22_on_gen23 cfgs/chance_all_partial.toml` (same seeded decisions as the loop's control check): +0.001/+0.003/+0.006/+0.011 at 64/250/1000/4000, MONOTONE, slope 0.70, against the control's +0.001/+0.002/+0.006/+0.008, slope 0.61.
- **Conclusion: adopted.** The loop generates with `cfgs/chance_all_partial_gen.toml` and benchmarks (both sides) and net-checks with `cfgs/chance_all_partial.toml` from the relaunch after gen24's generate (first C1 corpus: gen25). Not run: B2-B4 (mass, sampled, widen), A3 alone, A2 (passes). Lesson: the corrected dice only pay once the backup stops charging for chance width; the partial backup is the win (14x7 0.552, 16x9 0.517 alone).
- **Question:** the search still scripts three rolls.
  - A bounce onto a player is dropped, although the engine makes a standing player try to catch
    it.
  - An inaccurate pass's scatter goes up x3, and a wildly inaccurate pass's deviate goes 1 up.
  - The throw-in takes the shortest in-bounds throw.

  Do exact (grouped) models of these help? And does any alternative to "a chance node has no value
  until every outcome is scored" (`partial`, `mass` 0.9, `sampled`, `widen`) beat the shipped
  `complete` backup?
- **Setup:** `scripts/exp070_chance_model.sh`; code at the plan 061 branch (3bbcfc8..1364275,
  hub protocol v18).
  - **Bots:** v9 gen13 (`models/az_v7/bbnet_mix16x9v9_gen13.onnx`) on both sides, 1000 descents.
    Each arm is a `cfgs/chance_*.toml` against `cfgs/gumbel16_f1000.toml`.
  - **Drives:** paired contested drives on `runs/loopmix16x9g/positions/contested_{14x7,16x9}_gen04g.json`,
    at most 800 per board.
  - **SPRT:** A1-A3 (roll models) non-inferiority `0.47:0.5`; B1-B4 (backup modes) `0.5:0.55`.
    C1 (every roll model under the B winner) runs `0.5:0.55` against the winner, then `0.47:0.5`
    against the control.
  - **CPU:** `scripts/perf_search_bench.sh` per arm.
- **Result (arm 0, telemetry only):**
  - **Real games.** `botbowl-ui roll-census` (16x9 build at the branch) over
    `runs/loopmix16x9v9/gen13/shard{0,1}.jsonl`: 1449 drives, 0 diverged. Per drive: 1.30 live
    bounces, 0.18 throw-ins, 0.45 kickoff deviates, **0.001 passes**. Live-bounce landing mass:
    71.4% empty, 17.4% standing player, 3.2% downed player, 8.1% out. 21% of it is what the shipped
    model drops.
  - **In search.** gen13 on tract, 1000 descents, 6 drives per arm (seed 61000, 14x7/4 + 16x9/6,
    one thread), per searched decision.
    - Shipped: 36 bounce nodes (5.8 outcomes) and 7 throw-ins; 23% of chance backups withheld;
      main line reaches the opponent's turn 73%.
    - All roll models, complete backup: bounce passes ×3, 34% withheld, main line 43%.
    - All roll models, partial backup: 0% withheld; own decisions per line 3.85 → 4.39, chance
      share 45% → 38%, exact-outcome leaves 0.37% → 1.85%.
    - Wall time ~2.6 s per decision in every arm (net-bound).
- **Conclusion:** pending the drives. A2 (passes) is deferred: the loop's bots do not pass.

### Gradient balance: policy vs value at the shared trunk (plan 059 follow-up), 2026-10-09

- **Question:** how do the policy and value losses split the gradient into the shared 64x6 trunk,
  how does the cq τ change it, and does a decision's number of legal actions change its weight?
- **Setup:** `scripts/grad_balance_probe.py models/az_v7/bbnet_mix16x9v9_gen12.pt <dirs>` (CPU,
  BN in eval mode = `--freeze-bn`), the loop's loss exactly (policy CE mean over samples + 0.25 ×
  per-drive-weighted value MSE). Data: gen13's held-out shard 4 (MC-labelled), prepared with
  `target/16x9/release/prepare` (built at 71827c9) at `--policy-target cq --tau 100|50|20
  --value-blend 1.0`: 20,285 samples.
- **Result:**

  | τ | KL(target‖net) | trunk grad norm, policy : 0.25·value | cosine |
  |---|---|---|---|
  | 100 | 0.070 | 0.455 : 0.155 = **2.9** | +0.02 |
  | 50 | 0.144 | 0.646 : 0.155 = **4.2** | +0.03 |
  | 20 | 0.367 | 1.023 : 0.155 = **6.6** | +0.03 |

  Policy CE is per decision (one cross-entropy over the legal set, averaged over samples), so a
  2-action decision weighs as much as a 100-action one; its gradient (|p − π| per sample) is
  smallest on 2-action decisions (22% of samples, 0.049 at τ=100) and largest at 3-20 actions
  (0.11). Optimiser: Adam (no weight decay, no clipping).
- **Conclusion:** the policy already supplies ~3/4 of the trunk gradient at τ=100; sharpening τ
  only grows it. The two gradients are near-orthogonal, so they do not cancel; they compete
  through Adam's per-parameter normaliser (the value term's effective step in the trunk shrinks
  ~28% at τ=50, ~54% at τ=20) and the trunk's capacity. Consistent with exp069 (τ 30 and Gumbel σ
  lose the value gain) and the τ=50 loop (gen11 value bench +1.1%, net-check gain fell).

### <a id="exp069"></a>exp069 — policy targets under the mean backup (plan 059), 2026-10-08

- **Question:** the loop's policy absorbs little per generation (Δ P(played) +0.001..+0.004). Does
  a sharper target (cq τ 50, 30, Gumbel σ) absorb more without hurting the value head?
- **Setup:** `scripts/exp069_targets_policy_only.sh` (script at 81b5842; `prepare` binary built at
  43a45c6, `botbowl-ui` at 3c25109; trainer `train/` unchanged across those). Parent v9 gen06; the
  window the loop's gen07 trained on (gen05-07, MC-labelled, train shards 0-3/5/6, val 4/7); the
  loop's recipe (lr 5e-5, 3 epochs, `--freeze-bn --eval-at 250,500`, select on combined,
  `--value-weight 0.25 --per-drive-value-weight`, `--value-blend 1.0`), only the policy target
  changed. Scored by: absorption on gen07's held-out shards prepared once at cq τ=100; the v9 value
  bench paired vs gen06; policy-only drives vs gen06, 600 pairs per board.
- **Result:**

  | arm | Δ log P(played) | Δ P(played) | Δ top-1 | value bench | policy-only mean |
  |---|---|---|---|---|---|
  | loop gen07 (τ 100) | +0.003 | +0.002 | +0.002 | −1.4% | 0.500 ± 0.005 |
  | τ 100 re-run | +0.002 | +0.001 | +0.001 | −1.7% | 0.502 ± 0.005 |
  | **τ 50** | **+0.016** | −0.001 | +0.002 | **−1.4%** | **0.509 ± 0.005** |
  | τ 30 | +0.028 | −0.010 | +0.001 | ±0% | 0.496 ± 0.007 |
  | Gumbel σ (c_scale 0.1, min_range 50) | −0.050 | −0.090 | −0.012 | +1.7% (worse) | 0.502 ± 0.007 |

- **Conclusion:** the τ=100 control reproduces to ~0.001, so the quick numbers are low-noise.
  Sharper τ raises log P(played) but costs mean P(played) and, through the shared trunk, the value
  head (τ 30 loses all of the value gain, Gumbel σ goes backwards). τ=50 keeps the value gain and
  is the only arm above its parent in policy-only play (+1.7 SE: a lead). **Adopted by the user:
  the loop trains on cq τ=50 from gen09.** Re-read via policy-only lineage at gen11-12.
- **Read-out (2026-10-09, plan 059 §7):** three τ=50 generations moved the bare policy by −0.004
  (gen11 vs gen08 0.496 ± 0.006) against +0.007 for the last three τ=100 steps (gen08 vs gen05
  0.507 ± 0.006): the one-step lead did not replicate. Recommendation: back to τ=100.
- **Side finding:** one-step policy-only drives are blunt (the argmax moves in ~0.2% of
  decisions); compare across several generations instead.

### Policy-only lineage (plan 059 §5), 2026-10-08

- **Question:** does the loop's bare policy improve over generations at all (τ=100)?
- **Setup:** `runs/exp069/po_lineage/run.sh`, botbowl-ui at 3c25109, policy-only both sides,
  600 pairs per board.
- **Result:** gen07 vs gen04 0.512 ± 0.006; gen07 vs the init net 0.518 ± 0.007 (MATCHES.md).
- **Conclusion:** yes, ~0.003-0.004 per generation at τ=100; invisible in a single step.

### 14x5 edge-bounce chains (plan 060 §6), 2026-10-08

- **Question:** why do 14x5 searches in gen09 average 63 plies (93% chance) when every other board
  averages ~9?
- **Setup:** `botbowl-ui/examples/deep_search_probe.rs` replays a corpus sample from its seed and
  dumps the deepest line (e.g. seed 19000196 drive 1 sample 7); 5 corpus positions at 1000
  descents, heuristic and gen08 (tract); a 24-drive 14x5 heuristic corpus (seed 777) before/after;
  `scripts/perf_search_bench.sh` on 16x9.
- **Result:** the search's scripted throw-in (`roll_outcomes::throw_in_outcome`, the shortest
  throw, 2D6 = 2) is `2 / scatter_divisor 3 = 0` squares on a 5-wide axis: the ball lands where it
  was thrown from, bounces out again, forever (not a DAG cycle: `bounce_squares` grows, so states
  never repeat). Fix: the shortest throw that leaves the origin square (f8aa794). 14x5: depth 49.7
  → 6.8 plies, chance 94% → 55%, turn-ending reach 51% → 83%, 98 → 37 ms per decision; the
  searches chose the same moves on the 5 probes. 16x9 games byte-identical. Also fixed: the debug
  teardown assertion, `Node::move_root` left a second root edge to the new root (526726a).
- **Conclusion:** a real search-model bug, live from the gen10 relaunch; gen01-09's 14x5 samples
  carry it (DATA.md). Only boards with a 3-5 square narrow axis play differently.

### Tree statistics, first live reading (plan 060), 2026-10-08

- **Question:** how deep does the search see, and do its lines reach the opponent's turn?
- **Setup:** `runs/plan060/budget_ladder.sh`: botbowl-ui built at 759e5b7 (`target/tree-16x9`),
  v9 gen06 on tract, `dataset --mode random-start --next-drive --games 12 --seed 60000
  --bot-config cfgs/gumbel16_f1000_gen.toml --parallel-games 3` + the loop's size flags, at
  `--mcts-iters` 250 / 1000 / 4000; `scripts/tree_stats.py` on each.
- **Result:** mean leaf depth 6.5 / 8.6 / 10.8 plies; share of descents reaching the opponent's
  turn 52% / 61% / 74% (87% at 1000 for the turn-ending decision); main line reaches it 58% / 73% /
  86%; ~half of every line is dice; ~3 own decisions per line at any budget.
- **Conclusion:** healthy on plan 060 §3's terms (depth and reach grow with budget; turn-ending
  decisions see the opponent's reply). Extra budget buys opponent-turn and dice depth, not deeper
  own planning. Open: separate a turnover-ended opponent turn from a played-out one.

### Generation throughput after the search CPU cuts (plan 058 §9), 2026-10-07 night

- **Question:** how many generation streams, and how many GPU sidecars, now that the search's CPU
  cost fell 76% (§7) and `GameState` 4x (§8)?
- **Setup:** `scripts/perf_gen_steady.sh OUT models/az_v7/bbnet_mix16x9v9_gen05.onnx "<streams>"
  1` (`SIDECARS=N` for N sidecars): the loop's generation with `--next-drive`, warm-up skipped,
  decisions counted in a 10 min (5076090) or 8 min (96fb083) window.
- **Result:** one sidecar saturates near 8k samples/s (GPU ~80%); two sidecars at 48 streams fill
  the GPU (8.55k/s, +9%); 64 streams add nothing; 48 streams use 4.4 GB on 96fb083 (6.5 GB before).
  In the loop gen06 made 1766 decisions/min, but ~40% of that was the laptop (10 streams on its
  own CPU): without it (gen08-09) the box makes ~1080/min, the benchmark's rate (plan 058 §9).
- **Conclusion:** generation is GPU-bound; the loop runs 48 streams over two sidecars and spends
  the speedup on data (400 games per shard).

### Forced moves in the corpus (plan 058 §6), 2026-10-07

- **Question:** how much search went into decisions with one action left after pruning?
- **Setup:** count of samples with < 2 children in v9 gen04 (the last corpus from the old bot).
- **Result:** 30,133 of 150,520 samples (20.0%); MoveAction 59% of them, Kickoff 100% forced.
- **Conclusion:** the bot plays them without search and `prepare` drops them (3dc4902, 2afd0db);
  gen04's window shrank 21%.

### Stream count before the CPU cuts (plan 058 §2), 2026-10-06

- **Setup:** `scripts/perf_gen_bench.sh`, 4 games per thread (the first pass, games ≈ threads, was
  wrong: tail effect).
- **Result:** CPU-bound from ~36-48 streams (17-20 games/min at 48); two sidecars +7%; mc-label
  per-sample work 2x (683fb34).
- **Conclusion:** 36 streams then (memory-capped), later superseded by §9.

## Pre-v9 rules (pointers; details in the plans)

| exp | plan | question | conclusion |
|---|---|---|---|
| exp068 | 056 | do MC-labelled values (arm F) play better? | F's search beats A's 0.533 ± 0.009 and its own policy 0.582 ± 0.011 at 1000 descents: **adopted** |
| exp067 | 056 | value labels: blend 0.5 / 1.0, TD(λ) 0.8 / 0.95, value weight 1.0, MC-averaged | MC-averaged (F): value bench RMS −15.5%, bias ~0; TD(λ) only removes bias |
| exp066 | 055 | does more search beat the bare policy, under the mean backup? | 0.525 / 0.542 / 0.590 at 250 / 1000 / 4000 descents: the criterion holds (mean backup hardcoded, ce4eda1) |
| exp065 | 055 | override audit: are the search's overrides of the prior right? | value head RMS 0.26-0.28 TD vs MC truth; overrides on Q gaps < 0.1 TD realise nothing |
| exp063 | 054 | train step: cq τ 50 vs 100 at lr 5e-5 | τ 100 at 5e-5 passed the gate (0.531), τ 50 0.510; search did not beat its policy (E5 0.517) |
| exp059 | 049/050 | policy targets (PUCT era) and WDL value head | no target beats cq τ 100; WDL step 1 fails |
| exp056 | 049 | τ and search budget on drives | see plan 049 |
| exp055 | 049 | why the vl0 loop regressed (τ, budget) | τ 100 recovers 14x7; aborted partial |
| exp054 | 049 | virtual loss 0 vs 30 | vl0 wins 0.686; the virtual-loss leak fixed (5e29d89) |
| exp048 | 048 | exploration A/B | see plan 048 |
