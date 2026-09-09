# nn_server: where the inference time went, and the event-loop rewrite

**Status: DONE (2026-09-09).** Merged to master (`116bee8`), `botbowl-nn/CLAUDE.md` updated.
Requested by the user during plan 032 after the live sidecar's stats line read
`mean_batch=1.61 batch=777us (stage 47 + fwd 701 + post 29) queue=840us/sample` while
serving six 1000-iteration games — ~1.6 ms per NN request for a 0.48 M-param net on a 16x9 board.

## Diagnosis

Measured against the old `scripts/nn_server.py` with d7 (64x6), 14x7 tier, CUDA events:

| component of the ~1.6 ms | µs | evidence |
|---|---|---|
| GPU kernels for a batch of 1–4 (jit-frozen CUDA graph, replay only) | **~300** | b=1 294, b=2 327, b=4 303, b=8 393, b=16 648. 24 cuDNN winograd launches + 2 sgemm: real work at hopeless occupancy, not launch overhead. **This is the floor.** |
| separate H2D×2 + sliced D2H launches | ~70 device + ~90 host | start→done 366–396 vs replay-only 304 |
| host in the batcher cycle: spinning `.cpu()` sync, ATen dispatch, GIL hand-offs among 12+ connection threads, per-request Python | ~400 | live `batch=777` vs ~300 of GPU |
| queue: an arrival waits for the whole in-progress serial cycle | ~840 | live `queue=840us/sample` |
| server CPU | ~1 core of 4 | 86–112 % |

`mean_batch` 1.6 is not a window-policy defect: only six clients exist and each spends about half its
cycle in the engine, so at most ~3 are ever waiting; greedy (`max_wait_us=0`) is right.

Rejected after measurement: **fp16** (795 µs at b=1 — Pascal), **channels_last** (551), **CPU torch**
(4.9 ms eager / 3.0 ms jit 2 threads), **tract in-process** (1.9–3.0 ms), capturing the copies inside the
graph (no gain), a **Rust-native server** (host side is now ~150 µs/batch; the 300 µs floor is the GPU).
The GPU sidecar is the right architecture even at mean batch 1.6; the serving loop was the problem.

## Change (`scripts/nn_server.py`; wire protocol, CLI, canary, `--bench`, `--loadgen` unchanged)

- One `selectors` loop, no per-connection threads, no `Queue`, no GIL hand-offs. Per-connection
  preallocated buffer + `recv_into`, incremental parser.
- **Depth-2 GPU pipeline:** `launch` enqueues H2D → graph replay → D2H + a CUDA event; results are
  answered one iteration later while the next batch runs. Just-in-time launch (`JIT_S` = 250 µs before
  the device goes idle) so arrivals during a batch leave together — recovers the batch sizes the serial
  design got for free (mean 1.53 vs 1.40 greedy at 6 clients).
- Coalesced pinned input buffer per slot → one H2D (launch host cost 140 → 50 µs; device 370 → 320).
- Completion wait = `select` sleep until ETA − 120 µs, then `event.query()` spin. Two traps found on the
  way: `epoll_wait` rounds sub-millisecond timeouts up to 1 ms (every sleep was 1 ms, a flat 1.36 ms round
  trip) — `SelectSelector` wakes ~65 µs late instead; and without `cudnn.benchmark=True` the b=1 graph bakes
  in a 4 ms cuDNN algorithm.
- `fwd` in the stats line is now device time from CUDA events.
- Rust: `botbowl-nn/examples/nn_bench.rs` (round-trip benchmark through `RemoteClient`),
  `NnEvaluator::value_only_raw`, `remote.rs` reads a value-only response in one syscall.
- Tests: `train/tests/test_nn_server.py` (CPU, real socket: canary, value/policy, split frames, model_id
  mismatch → EOF, bad magic, 6-client batch invariance); `cargo test -p botbowl-nn` live tests pass
  against the new server. Also fixed a pre-existing 14x7-only failure,
  `encode::tests::mirror_consistency_us_present_hits_canonical_squares` (literal (10,8) off the small board).

## Measured (idle box, 2026-09-09 16:40, two rounds, `nn_bench --iters 3000`, zero think time)

| clients | old: median / p90 / forwards/s / server CPU | new | Δ |
|---|---|---|---|
| 1 | 474 µs / 515 / 2100 / 61 % | 498 / 594 / 1970 / 50 % | none (p90 slightly worse: the spin/select wake) |
| 2 | 805 / 864 / 2465 / 69 % | 694 / 704 / 2880 / 60 % | −14 % latency, +17 % throughput |
| **6** (production) | **1370 / 1464 / 4420 / 77 %** | **914 / 1197 / 5990 / 65 %** | **−33 % latency, +36 % throughput** |
| tract in-process, 1 thread | 1878 µs | | |

The gain is concurrency-dependent: the pipeline overlaps one client's readback with another's compute,
so a lone client sees nothing and six see 1.36×. Real games interleave engine time between requests, so
in production the effect sits between the 2- and 6-client rows. The old server's idle-box 1.37 ms at six
clients vs the 1.6 ms seen live is the games' engine work contending for the same cores.

Not the 2× hoped for from the isolated pieces (~330 µs device + ~150 host) — the remaining gap is the
select/spin wake (~65 µs), the client-side syscalls, and the fact that six clients' requests still
serialise on one GPU stream. Knobs worth a sweep only with a full-game benchmark: `SPIN_S`, `JIT_S`, `DEPTH`.

## Consequences

- Generation and eval get ~1.4× more NN throughput at the production shape and ~12 points of a core back;
  no change to any result — the numerics are the same jit-frozen graph.
- A bigger net is now more expensive in the one place that matters: the GPU floor scales with the net
  (96x8 ≈ 2.5× the kernels), which is a second argument against plan 032 #9's capacity arm.
- The 300 µs floor is cuDNN on a 16x9 board; only a different GPU or a fused/smaller net moves it.
