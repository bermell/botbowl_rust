#!/usr/bin/env python3
"""Plan 069 item 6: scale-test the hub with fake workers (stdlib only; scripts/hub_scale.sh builds
the binaries and calls this).

Each stage starts a fresh hub on its own port (never 13337), one `botbowl-fake-worker` process with
CONNS connections x STREAMS streams (thousands of streams from one box: games are sleeps, results
canned corpus lines), submits one generate or label job sized to keep every stream busy for about
DURATION seconds, and samples the hub once a second: CPU (% of one core), RSS, threads. The fake
worker reports, from the worker side of the socket, results/s, MB/s each way, and the dispatch
latency (a task's last result sent -> the next task arriving on that connection: the hub's whole
per-result path). A stage's verdict is the achieved result rate against the offered one
(streams / secs per unit): below ~95% in steady state, or a dispatch p99 that grows with the stream
count, is the hub falling behind.

Stages are `kind:streams:secs:duration[:conns][:discard]`, kind generate|label; conns defaults to
streams / 64 (a GPU worker's 64 streams per connection); `discard` truncates the hub's shard files
every 2 s (generate only, append mode) so a high-rate stage does not fill the disk.

    scripts/hub_scale.py --hub-bin H --fake-bin F --canned C.jsonl --out DIR [--stages ...]
"""

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

DEFAULT_STAGES = [
    # near-realistic per-unit times (a 20x11 generate game ~60 s on a stream, a label item ~15 s;
    # generate at 20 s so each stream runs three 4-game tasks within the stage)
    "generate:100:20:240",
    "generate:1000:20:240",
    "generate:4000:20:240",
    "label:100:15:180",
    "label:1000:15:180",
    "label:4000:15:180",
    # compressed time: where does the hub saturate?
    "generate:4000:4:60::discard",
    "generate:4000:1:60::discard",
    "label:4000:4:45",
    # many thin workers (tract boxes, CPU-partition nodes): per-connection cost
    "generate:1000:10:120:1000",
]


def parse_stage(s):
    f = s.split(":")
    kind, streams, secs, duration = f[0], int(f[1]), float(f[2]), float(f[3])
    # Default: GPU-worker-shaped connections of 64 streams; the stream count is rounded to fill them
    # (1000 -> 16 x 64 = 1024, 4000 -> 62 x 64 = 3968).
    conns = int(f[4]) if len(f) > 4 and f[4] else max(1, round(streams / 64))
    per_conn = max(1, round(streams / conns))
    discard = len(f) > 5 and f[5] == "discard"
    assert kind in ("generate", "label"), s
    return dict(spec=s, kind=kind, streams=conns * per_conn, secs=secs, duration=duration, conns=conns,
                per_conn=per_conn, discard=discard)


def token(path):
    with open(path) as f:
        return f.read().strip()


def api(port, tok, path):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def proc_sample(pid):
    """(cpu seconds, rss MB, threads) of a live process, or None."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            st = f.read().rsplit(")", 1)[1].split()
        cpu = (int(st[11]) + int(st[12])) / os.sysconf("SC_CLK_TCK")
        rss = threads = 0
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
        return cpu, rss, threads
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def canned_games(path):
    """The canned file as games: (lines, samples), a next-drive record kept with its first drive."""
    games = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            v = json.loads(line)
            n = len(v.get("samples", []))
            follow = (v.get("meta", {}).get("extra", {}) or {}).get("drive") == "2"
            if follow and games:
                games[-1][0].append(line.rstrip("\n"))
                games[-1][1].append(n)
            else:
                games.append(([line.rstrip("\n")], [n]))
    return games


def write_label_inputs(games, items_wanted, chunk, shards, d):
    """Replicate the canned games over `shards` shard files until they hold ~items_wanted label
    items of at most `chunk` samples. Returns (paths, items, trajectories, bytes)."""
    per_game = [sum(max(1, math.ceil(n / chunk)) for n in ns) for _, ns in games]
    reps = max(1, math.ceil(items_wanted / max(1, sum(per_game))))
    paths = [os.path.join(d, f"in{k}.jsonl") for k in range(shards)]
    fs = [open(p, "w") for p in paths]
    items = trajs = 0
    i = 0
    for _ in range(reps):
        for (lines, _), n in zip(games, per_game):
            f = fs[i % shards]
            for line in lines:
                f.write(line + "\n")
            items += n
            trajs += len(lines)
            i += 1
    for f in fs:
        f.close()
    return paths, items, trajs, sum(os.path.getsize(p) for p in paths)


def pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(round((len(v) - 1) * q)))]


def run_stage(a, st, idx, tok, games, mean_samples):
    name = f"{idx:02d}-{st['kind']}-{st['streams']}x{st['secs']:g}s" + (f"-c{st['conns']}" if st["conns"] != max(1, math.ceil(st["streams"] / 64)) else "")
    d = os.path.join(a.out, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "noindex"))
    port = a.port + idx
    print(f"== stage {name}: {st['spec']}", flush=True)
    res = dict(stage=name, **{k: st[k] for k in ("kind", "streams", "secs", "duration", "conns", "per_conn", "discard")})

    # Inputs first (the label job's corpus), so their writing is not in the hub's numbers.
    units = math.ceil(st["streams"] * st["duration"] / st["secs"])
    if st["kind"] == "generate":
        # Generate tasks are 4 games (`job generate --batch` default): whole tasks per stream, and
        # shards of equal size, so no stream idles for want of a batch.
        tasks = max(1, round(st["duration"] / (st["secs"] * 4)))
        units = st["streams"] * tasks * 4
        units = math.ceil(units / 32) * 32
    if st["kind"] == "label":
        inputs, items, trajs, nbytes = write_label_inputs(games, units, a.chunk_samples, 8, d)
        res.update(label_items=items, label_trajectories=trajs, label_input_mb=nbytes / 1e6)
        units = items
    hub_log = open(os.path.join(d, "hub.log"), "w")
    hub = subprocess.Popen([a.hub_bin, "serve", "--bind", f"127.0.0.1:{port}", "--no-play",
                            "--model-index-dir", os.path.join(d, "noindex"), "--rate-interval-secs", "60"],
                           cwd=d, stdout=hub_log, stderr=subprocess.STDOUT)
    fake = None
    stop_janitor = threading.Event()
    try:
        for _ in range(100):
            try:
                api(port, tok, "/api/status")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("hub did not come up")
        base = proc_sample(hub.pid)
        fake_out = open(os.path.join(d, "fake.jsonl"), "w")
        t_conn = time.time()
        fake = subprocess.Popen([a.fake_bin, "--hub", f"ws://127.0.0.1:{port}/ws", "--connections", str(st["conns"]),
                                 "--streams", str(st["per_conn"]), "--gen-secs", str(st["secs"]),
                                 "--label-secs", str(st["secs"]), "--canned", a.canned, "--report-every", "5",
                                 "--name-prefix", "fake"],
                                stdout=fake_out, stderr=open(os.path.join(d, "fake.log"), "w"))
        while True:
            s = api(port, tok, "/api/status")
            if len(s["workers"]) >= st["conns"]:
                break
            if time.time() - t_conn > 300:
                raise RuntimeError(f"only {len(s['workers'])}/{st['conns']} connections after 300 s")
            time.sleep(0.5)
        res["connect_all_secs"] = time.time() - t_conn

        # Submit.
        if st["kind"] == "generate":
            per_shard = math.ceil(units / 8)
            res["games"] = per_shard * 8
            cmd = [a.hub_bin, "job", "generate", "--hub", f"http://127.0.0.1:{port}", "--mode", "random-start",
                   "--games", str(per_shard), "--shards", "0 1 2 3 4 5 6 7", "--evaluator", "heuristic",
                   "--mcts-iters", "1000", "--seed-base", "1", "--out-dir", os.path.join(d, "gen"),
                   "--label", name]
            if not st["discard"]:
                cmd.append("--truncate")
        else:
            model = os.path.join(d, "fake.onnx")
            with open(model, "wb") as f:
                f.write(os.urandom(int(a.model_mb * 1e6)))
            cmd = [a.hub_bin, "job", "label", "--hub", f"http://127.0.0.1:{port}", "--corpus", *inputs,
                   "--model", model, "--playouts", "8", "--seed", "1", "--chunk-samples", str(a.chunk_samples),
                   "--out-dir", os.path.join(d, "mc"), "--label", name]
        t_submit = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"submit failed: {out.stderr.strip()}")
        job = int(out.stdout.strip().split()[-1])
        t0 = time.time()
        res["submit_secs"] = t0 - t_submit

        if st["discard"]:
            def janitor():
                gen = os.path.join(d, "gen")
                while not stop_janitor.wait(2):
                    for fn in os.listdir(gen) if os.path.isdir(gen) else []:
                        try:
                            os.truncate(os.path.join(gen, fn), 0)
                        except OSError:
                            pass
            threading.Thread(target=janitor, daemon=True).start()

        samples = []
        state = "Running"
        last_cpu = (time.time(), base[0] if base else 0.0)
        deadline = t0 + st["duration"] * 4 + 600
        while time.time() < deadline:
            time.sleep(1)
            ps = proc_sample(hub.pid)
            now = time.time()
            if ps:
                cpu_pct = 100 * (ps[0] - last_cpu[1]) / max(1e-6, now - last_cpu[0])
                last_cpu = (now, ps[0])
                samples.append(dict(t=now - t0, cpu_pct=cpu_pct, rss_mb=ps[1], threads=ps[2]))
            if int(now - t0) % 2 == 0:
                # An overloaded hub answers its API late (the API takes the same lock as every
                # result): count it, as `job --wait` would retry, and keep going.
                try:
                    js = api(port, tok, f"/api/jobs/{job}")
                except Exception:
                    res["api_timeouts"] = res.get("api_timeouts", 0) + 1
                    continue
                state = js["state"] if isinstance(js["state"], str) else json.dumps(js["state"])
                if state != "Running":
                    break
        res["state"] = state
        res["wall_secs"] = time.time() - t0
        with open(os.path.join(d, "hub_samples.json"), "w") as f:
            json.dump(samples, f)
    finally:
        stop_janitor.set()
        if fake and fake.poll() is None:
            fake.send_signal(signal.SIGINT)
            try:
                fake.wait(30)
            except subprocess.TimeoutExpired:
                fake.kill()
        hub.send_signal(signal.SIGINT)
        try:
            hub.wait(30)
        except subprocess.TimeoutExpired:
            hub.kill()
        hub_log.close()

    # The fake worker's view.
    reports = []
    with open(os.path.join(d, "fake.jsonl")) as f:
        for line in f:
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    final = next((r for r in reversed(reports) if r.get("final")), {})
    steady = [r for r in reports if not r.get("final") and r["t"] > res["connect_all_secs"] + st["secs"] * 1.5
              and r["t"] < res["connect_all_secs"] + res["wall_secs"] - st["secs"] * 1.5]
    offered = st["streams"] / st["secs"]
    rate = pct([r["results_per_s"] for r in steady], 0.5)
    hot = [s for s in samples if s["t"] > st["secs"]] or samples
    res.update(
        units=units,
        offered_per_s=offered,
        achieved_per_s=rate,
        achieved_share=(rate / offered) if rate else None,
        ideal_secs=units * st["secs"] / st["streams"],
        efficiency=(units * st["secs"] / st["streams"]) / res["wall_secs"] if res.get("wall_secs") else None,
        hub_cpu_pct_mean=sum(s["cpu_pct"] for s in hot) / len(hot) if hot else None,
        hub_cpu_pct_p95=pct([s["cpu_pct"] for s in hot], 0.95),
        hub_rss_mb_max=max((s["rss_mb"] for s in samples), default=None),
        hub_threads=max((s["threads"] for s in samples), default=None),
        dispatch_ms=final.get("dispatch_ms"),
        dispatch_p99_worst_interval=max((r["dispatch_ms"].get("p99", 0) for r in steady if r["dispatch_ms"].get("n")), default=None),
        connect_ms=final.get("connect_ms"),
        mb_recv_per_s=pct([r["mb_recv_per_s"] for r in steady], 0.5),
        mb_sent_per_s=pct([r["mb_sent_per_s"] for r in steady], 0.5),
        reconnects=final.get("reconnects"),
        mean_samples_per_game=mean_samples,
    )
    if st["kind"] == "generate" and not st["discard"]:
        n = 0
        for fn in os.listdir(os.path.join(d, "gen")):
            with open(os.path.join(d, "gen", fn), "rb") as f:
                n += sum(1 for _ in f)
        res["lines_written"] = n
    if not a.keep:
        for sub in ("gen", "mc"):
            shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
        for fn in os.listdir(d):
            if fn.startswith("in") and fn.endswith(".jsonl") or fn == "fake.onnx":
                os.remove(os.path.join(d, fn))
    print("   " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                              if k in ("state", "wall_secs", "ideal_secs", "achieved_share", "hub_cpu_pct_mean",
                                       "hub_rss_mb_max", "submit_secs")}), flush=True)
    print("   dispatch_ms " + json.dumps(res["dispatch_ms"]), flush=True)
    return res


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hub-bin", required=True)
    ap.add_argument("--fake-bin", required=True)
    ap.add_argument("--canned", required=True, help="corpus lines (the fake's results and the label input)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=14000, help="first stage's port; stage i uses port+i")
    ap.add_argument("--token-file", default=os.path.expanduser("~/.config/botbowl/hub.token"))
    ap.add_argument("--stages", nargs="*", default=DEFAULT_STAGES)
    ap.add_argument("--chunk-samples", type=int, default=32)
    ap.add_argument("--model-mb", type=float, default=12.0, help="size of the net shipped to every connection")
    ap.add_argument("--min-free-gb", type=float, default=70.0)
    ap.add_argument("--keep", action="store_true", help="keep each stage's corpus files")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    assert a.port != 13337 and not (a.port <= 13337 < a.port + len(a.stages)), "port 13337 is the user's hub"
    tok = token(a.token_file)
    games = canned_games(a.canned)
    mean_samples = sum(sum(ns) for _, ns in games) / len(games)
    print(f"canned: {len(games)} games, {mean_samples:.0f} samples per game", flush=True)
    results = []
    for i, spec in enumerate(a.stages):
        free = shutil.disk_usage(a.out).free / 1e9
        if free < a.min_free_gb:
            print(f"!! {free:.0f} GB free < {a.min_free_gb:.0f}: stopping before stage {spec}", flush=True)
            break
        try:
            results.append(run_stage(a, parse_stage(spec), i, tok, games, mean_samples))
        except Exception as e:  # one failed stage should not lose the others
            print(f"!! stage {spec} failed: {e}", flush=True)
            results.append(dict(stage=spec, state=f"error: {e}"))
            if not a.keep:  # a failed stage's corpus files are GBs: never leave them behind
                for d in os.listdir(a.out):
                    for sub in ("gen", "mc"):
                        shutil.rmtree(os.path.join(a.out, d, sub), ignore_errors=True)
                    if os.path.isdir(os.path.join(a.out, d)):
                        for fn in os.listdir(os.path.join(a.out, d)):
                            if fn.startswith("in") and fn.endswith(".jsonl"):
                                os.remove(os.path.join(a.out, d, fn))
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(results, f, indent=1)

    rows = ["| stage | conns x streams | s/unit | units | state | wall s | ideal s | achieved/offered | hub CPU % mean / p95 | hub RSS MB | dispatch ms p50 / p99 / max | worst-interval p99 | MB/s hub->fake / fake->hub | submit s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        dm = r.get("dispatch_ms") or {}
        rows.append("| {} | {} x {} | {} | {} | {} | {} | {} | {} | {} / {} | {} | {} / {} / {} | {} | {} / {} | {} |".format(
            r.get("stage"), r.get("conns", "-"), r.get("per_conn", "-"), fmt(r.get("secs"), 0), r.get("units", "-"),
            r.get("state"), fmt(r.get("wall_secs"), 0), fmt(r.get("ideal_secs"), 0), fmt(r.get("achieved_share"), 3),
            fmt(r.get("hub_cpu_pct_mean"), 0), fmt(r.get("hub_cpu_pct_p95"), 0), fmt(r.get("hub_rss_mb_max"), 0),
            fmt(dm.get("p50")), fmt(dm.get("p99")), fmt(dm.get("max")), fmt(r.get("dispatch_p99_worst_interval")),
            fmt(r.get("mb_recv_per_s")), fmt(r.get("mb_sent_per_s")), fmt(r.get("submit_secs"), 2)))
    table = "\n".join(rows)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(f"# hub scale test ({time.strftime('%Y-%m-%d %H:%M')})\n\ncanned: {a.canned} ({len(games)} games, "
                f"{mean_samples:.0f} samples each)\n\n{table}\n")
    print(table)


if __name__ == "__main__":
    sys.exit(main())
