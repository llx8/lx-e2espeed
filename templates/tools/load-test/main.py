"""Load-test the delivered service image on this machine's GPU, beside the baseline image, before the merge: how well
the algorithm uses this server. Production's worker count is operations' choice, made from this curve.

For each image, the worker count climbs through --workers (default 1,2,4,8,12,16; the image's config file mounted with
that count, WORKER_CONFIG) until the throughput stops growing (under 5% over the previous count, or errors, or GPU
memory near the card); at each count the closed-loop clients hold as many as the workers and twice that, each level for
--seconds, sampling throughput, latency, errors, responses that differ from the same request's answer at one client,
GPU (utilization, memory, power, SM clock, memory per process), the container's CPU cores, RAM, busiest process (a
worker pinned near 1.0 core is CPU-bound), busiest host core, worker restarts (new PIDs), and the MPS client count.
Then the delivered image holds the top level at its peak count for --soak seconds (memory at the start against the
end), and each container's log is scanned for errors, warnings, and a missing MPS daemon. The verdicts of
rules/wrap-up.md follow: the peak count, what saturates there (GPU utilization or power, a worker near 1.0 core, all
host cores) or nothing (a cause to find before the merge), GPU memory against the one-worker peak times the count and
across levels, the soak, restarts, and the log. Refuses to start while another process uses the GPU; kills the
container when MemAvailable falls under 20% of RAM. About 35 to 40 minutes.

Run, from the project directory, under a small cap (it covers this client; the container has its own, LX_MEM):
  <py> lx-work/tools/load-test/main.py --image IMAGE [--baseline-image IMAGE] [--workers 1,2,4,8,12,16]
       [--seconds 60] [--soak 600] [--status-only] [--out FILE]
Prints one line per image, worker count, and level, one comparison line per level, and the verdicts; the full result
goes to --out as JSON.
"""
import argparse
import json
import multiprocessing
import os
import queue as queue_errors
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

LX_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LX_WORK))


def adapter():
    """lx-work/adapter/pipeline.py, imported only where needed: the client processes re-import this file (spawn) and
    must not load the model's libraries."""
    from adapter import pipeline
    return pipeline

# TODO at wrap-up: from the service's config (PROJECT.md, Deployment).
PORT = 0         # TODO: the port the framework listens on inside the image (config.toml listen_addr)
PATH_URL = "/"   # TODO if the service answers on another path
WORKER_CONFIG = ("", "worker_num")  # TODO: (the config file's path inside the image, its worker-count key)
HOST_PORT = 18100
SWEEP = [1, 2, 4, 8, 12, 16]  # worker counts, climbed until the throughput stops growing
GROWTH = 1.05                  # a count whose throughput is under this times the previous count's ends the climb
GPU_SATURATED = (85, 0.9)  # utilization %, or share of the power cap, above which the GPU counts as the saturated side
CORE_SATURATED = 0.9       # a worker process above this many cores is CPU-bound
MEMORY_SLACK = 1.2         # GPU memory above this times the expected peak, or growing by this across levels, needs a cause
CLIENTS_PER_PROCESS = 4    # client threads per client process: the client runs in several processes so it does not set the pace
CLIENT_SATURATED = 0.8     # a client process above this many cores makes its level invalid: the client, not the service, set the pace
DECODER_CEILING_RPS = 0    # TODO when the service decodes on a hardware engine (NVJPG, NVDEC): the decodes/s that engine reaches at
                           # the service's inputs on this GPU, measured standalone with several processes (nvidia-smi does not show
                           # its load reliably); a peak at 90% of it counts as the decoder saturating. 0: no hardware decoder.


def requests_for(project):
    """[(label, kwargs for requests.post)] in the framework's request format: real production inputs."""
    # TODO: for example, every request of every scenario posted as the framework's demo client does:
    # ad = adapter()
    # return [(f"{name}#{i}", {"files": {"file": data}}) for name in ad.SCENARIOS
    #         for i, data in enumerate(ad.load_scenario(project, name).items)]
    raise NotImplementedError("TODO")


def sh(*args, timeout=60):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def gpu_busy():
    out = sh("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader").stdout.strip()
    return out.splitlines() if out else []


def cgroup_files(pid):
    """(cpu usage file, its unit in ns, memory file, procs file) of the container's cgroup, v1 or v2."""
    lines = [line.split(":", 2) for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()]
    v2 = [path for _, ctl, path in lines if ctl == ""]
    if Path("/sys/fs/cgroup/cgroup.controllers").exists() and v2:
        base = Path("/sys/fs/cgroup" + v2[0])
        return base / "cpu.stat", 1000, base / "memory.current", base / "cgroup.procs"
    path = {c: p for _, ctls, p in lines for c in ctls.split(",")}
    cpu = next(Path(f"/sys/fs/cgroup/{d}{path['cpuacct']}/cpuacct.usage") for d in ("cpuacct", "cpu,cpuacct")
               if Path(f"/sys/fs/cgroup/{d}{path['cpuacct']}").is_dir())
    mem = Path(f"/sys/fs/cgroup/memory{path['memory']}")
    return cpu, 1, mem / "memory.usage_in_bytes", mem / "cgroup.procs"


def read_cpu_ns(path, unit):
    text = path.read_text()
    if unit == 1:
        return int(text)
    return int(next(line.split()[1] for line in text.splitlines() if line.startswith("usage_usec"))) * unit


def proc_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, IndexError, ValueError):
        return None


def host_cores():
    return [list(map(int, line.split()[1:8])) for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("cpu") and line[3].isdigit()]


class Sampler(threading.Thread):
    """GPU every 500 ms (nvidia-smi), container CPU, RAM, and processes, and host cores every second."""

    def __init__(self, name, pid):
        super().__init__(daemon=True)
        self.name, self.files, self.stop = name, cgroup_files(pid), threading.Event()
        self.gpu, self.cores, self.ram, self.proc_max, self.host_max = [], [], [], [], []
        self.new_pids, self.apps, self.seen = [], [], None  # worker restarts (PIDs new since the start), GPU memory per process
        self.smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm",
                                     "--format=csv,noheader,nounits", "-lms", "500"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self.read_smi, daemon=True).start()
        self.smi_apps = subprocess.Popen(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
                                          "-lms", "1000"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self.read_apps, daemon=True).start()

    def read_smi(self):
        for line in self.smi.stdout:
            try:
                self.gpu.append([float(v) for v in line.split(",")])
            except ValueError:
                pass

    def read_apps(self):
        """Per-process GPU memory; nvidia-smi prints one line per process per sample and nothing between samples, so a
        sample is cut where the pid repeats or the timestamp of the lines changes (one second apart)."""
        current, pids = {}, set()
        for line in self.smi_apps.stdout:
            try:
                pid, mib = line.split(",")
                pid, mib = int(pid), float(mib)
            except ValueError:
                continue
            if pid in pids:
                self.apps.append(current)
                current, pids = {}, set()
            current[pid] = mib
            pids.add(pid)

    def run(self):
        cpu, unit, mem, procs = self.files
        total = int(next(line.split()[1] for line in open("/proc/meminfo") if line.startswith("MemTotal")))
        hz = os.sysconf("SC_CLK_TCK")
        last_ns, last_t, last_host = read_cpu_ns(cpu, unit), time.time(), host_cores()
        last_ticks = {}
        while not self.stop.wait(1.0):
            now, ns, host = time.time(), read_cpu_ns(cpu, unit), host_cores()
            dt = now - last_t
            self.cores.append((ns - last_ns) / 1e9 / dt)
            self.ram.append(int(mem.read_text()) / 2 ** 30)
            busy = [1 - (b[3] + b[4] - a[3] - a[4]) / max(1, sum(b) - sum(a)) for a, b in zip(last_host, host)]
            self.host_max.append(max(busy) * 100)
            ticks = {int(p): proc_ticks(int(p)) for p in procs.read_text().split()}
            if self.seen is None:
                self.seen = set(ticks)
            self.new_pids.append(len(set(ticks) - self.seen))
            self.seen |= set(ticks)
            per = [(t - last_ticks[p]) / hz / dt for p, t in ticks.items() if t is not None and last_ticks.get(p) is not None]
            self.proc_max.append(max(per, default=0.0))
            last_ns, last_t, last_host, last_ticks = ns, now, host, ticks
            avail = int(next(line.split()[1] for line in open("/proc/meminfo") if line.startswith("MemAvailable")))
            if avail * 100 < 20 * total:
                print(f"load-test: MemAvailable {avail} kB under 20% of RAM; killing {self.name}", file=sys.stderr)
                sh("docker", "kill", self.name)

    def window(self, t0):
        """Summary of the samples taken since index marks t0 = (gpu, cpu) list lengths."""
        g, c = self.gpu[t0[0]:], t0[1]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        apps = self.apps[t0[2]:]
        edge = lambda xs, k: mean(xs[:k]) if len(xs) >= 2 * k else mean(xs)
        return {"gpu_util_mean": mean([x[0] for x in g]), "gpu_util_max": max([x[0] for x in g], default=0),
                "gpu_mem_mib_max": max([x[1] for x in g], default=0), "power_w_mean": mean([x[2] for x in g]),
                "sm_mhz_min": min([x[3] for x in g], default=0), "cpu_cores_mean": mean(self.cores[c:]),
                "busiest_process_cores": max(self.proc_max[c:], default=0), "busiest_host_core_pct": max(self.host_max[c:], default=0),
                "ram_gib_max": max(self.ram[c:], default=0), "worker_restarts": sum(self.new_pids[c:]),
                "process_mem_mib_max": max([max(a.values(), default=0) for a in apps], default=0),
                "gpu_processes_max": max([len(a) for a in apps], default=0),
                "gpu_mem_mib_start": edge([x[1] for x in g], 10), "gpu_mem_mib_end": edge([x[1] for x in g][::-1], 10),
                "ram_gib_start": edge(self.ram[c:], 5), "ram_gib_end": edge(self.ram[c:][::-1], 5)}

    def marks(self):
        return len(self.gpu), len(self.cores), len(self.apps)

    def close(self):
        self.stop.set()
        self.smi.terminate()
        self.smi_apps.terminate()


def config_with_workers(image, workers):
    """A copy of the image's config file with its worker-count key set to `workers`, mounted over the original."""
    path, key = WORKER_CONFIG
    if not path:
        sys.exit("load-test: fill WORKER_CONFIG at the top of this file for a worker sweep")
    text = sh("docker", "run", "--rm", "--entrypoint", "cat", image, path).stdout
    new, n = re.subn(rf"^(\s*{key}\s*=\s*)\d+", rf"\g<1>{workers}", text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"load-test: {key} not found once in {path} of {image}")
    out = LX_WORK / "local" / "load-test" / f"{image.replace('/', '_').replace(':', '_')}-{workers}{Path(path).suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(new)
    return out


def start(image, name, config=None):
    sh("docker", "rm", "-f", name)
    mount = ["-v", f"{config}:{WORKER_CONFIG[0]}:ro"] if config else []
    r = sh("docker", "run", "-d", "--name", name, "--label", f"gpulane.grant={os.environ.get('GPULANE_JOB', 'none')}",
           "--gpus", "all", f"--memory={os.environ.get('LX_MEM', '48g')}",
           f"--memory-swap={os.environ.get('LX_MEM', '48g')}", "-p", f"{HOST_PORT}:{PORT}", *mount, image, timeout=300)
    if r.returncode:
        sh("docker", "rm", "-f", name)
        sys.exit(f"load-test: cannot start {image}: {r.stderr.strip()[-300:]}")
    time.sleep(1)
    state = sh("docker", "inspect", "-f", "{{.State.Running}} {{.State.Pid}}", name).stdout.split()
    if state[:1] != ["true"]:  # it exited at once (an LFS pointer for a binary, a bad config): say why, leave nothing
        logs = sh("docker", "logs", "--tail", "20", name)
        sh("docker", "rm", "-f", name)
        sys.exit(f"load-test: {image} exited at start:\n{logs.stdout}{logs.stderr}")
    return int(state[1])


def wait_ready(url, reqs, limit=600):
    session = requests.Session()
    session.trust_env = False  # the shells here set http_proxy with an empty no_proxy: 127.0.0.1 must not go through it
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            if session.post(url, timeout=60, **reqs[0][1]).status_code == 200:
                return time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def client_process(url, reqs, expected, threads, start_at, seconds, offset, status_only, results):
    """One client process: `threads` closed-loop clients from start_at for `seconds`; puts its answers, its own CPU in
    cores, how late it started, and when it ended."""
    lock, out = threading.Lock(), []
    counter = iter(range(offset, 10 ** 9))
    deadline = start_at + seconds

    def client(_):
        session = requests.Session()
        session.trust_env = False
        while time.time() < deadline:
            with lock:
                i = next(counter) % len(reqs)
            t = time.perf_counter()
            try:
                r = session.post(url, timeout=60, **reqs[i][1])
                code, body = r.status_code, r.content
            except requests.RequestException as e:
                code, body = type(e).__name__, b""
            ms = (time.perf_counter() - t) * 1000
            with lock:
                out.append((i, code, ms, code == 200 and not status_only and expected is not None and body != expected[i]))

    late = max(0.0, time.time() - start_at)
    while time.time() < start_at:
        time.sleep(0.01)
    cpu0 = time.process_time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(client, range(threads)))
    results.put((out, (time.process_time() - cpu0) / seconds, late, time.time()))


def hold(url, reqs, clients, seconds, expected, status_only, sampler=None):
    """Closed loop: each client sends the next request as soon as the last one is answered. The clients run in
    ceil(clients / CLIENTS_PER_PROCESS) processes that start together; the sampler window covers the same time."""
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    n = -(-clients // CLIENTS_PER_PROCESS)
    start_at = time.time() + 5 + 0.3 * n  # time for the processes to start
    procs = []
    for k in range(n):
        threads = min(CLIENTS_PER_PROCESS, clients - k * CLIENTS_PER_PROCESS)
        proc = ctx.Process(target=client_process, daemon=True, args=(url, reqs, expected, threads, start_at, seconds,
                                                                     k * len(reqs) // n, status_only, results))
        proc.start()
        procs.append(proc)
    while time.time() < start_at:
        time.sleep(0.01)
    marks = sampler.marks() if sampler else None
    try:
        parts = [results.get(timeout=seconds + 180) for _ in procs]
    except queue_errors.Empty:
        sys.exit("load-test: a client process gave no result; it died or hung")
    for proc in procs:
        proc.join()
    out = [r for part in parts for r in part[0]]
    elapsed = max(part[3] for part in parts) - start_at
    ms = sorted(m for _, code, m, _ in out if code == 200)
    pick = lambda q: ms[min(len(ms) - 1, int(q * len(ms)))] if ms else 0.0
    level = {"clients": clients, "requests": len(out), "rps": len(ms) / elapsed, "p50_ms": pick(0.5), "p90_ms": pick(0.9),
             "p99_ms": pick(0.99), "errors": sum(1 for _, code, _, _ in out if code != 200),
             "mismatches": sum(1 for *_, bad in out if bad), "client_processes": n,
             "client_cores_max": max(part[1] for part in parts), "client_cores_sum": sum(part[1] for part in parts),
             "client_late_s_max": max(part[2] for part in parts)}
    if sampler:
        level.update(sampler.window(marks))
    return level


def mps_clients(name):
    """Clients of the image's MPS server(s), or None when the image runs no MPS server it can be asked about."""
    servers = sh("docker", "exec", name, "sh", "-c", "echo get_server_list | nvidia-cuda-mps-control").stdout.split()
    if not servers or not all(s.isdigit() for s in servers):
        return None
    return sum(len(sh("docker", "exec", name, "sh", "-c", f"echo get_client_list {s} | nvidia-cuda-mps-control").stdout.split())
               for s in servers)


LOG_FLAGS = re.compile(r"error|warn|traceback|exception|fallback|MPS daemon not started", re.I)


def run_image(image, label, reqs, seconds, status_only, workers, soak=0):
    """One image at one worker count through the levels (as many clients as workers, then twice that), then a soak of
    the top level for `soak` seconds, then the container log scanned for LOG_FLAGS."""
    name = f"lx-load-{label}-{os.getpid()}"
    url = f"http://127.0.0.1:{HOST_PORT}{PATH_URL}"
    label = f"{label} {workers}w"
    levels = sorted({workers, 2 * workers})
    pid = start(image, name, config_with_workers(image, workers))
    sampler = Sampler(name, pid)
    sampler.start()
    try:
        ready = wait_ready(url, reqs)
        if ready is None:
            logs = sh("docker", "logs", "--tail", "20", name).stdout + sh("docker", "logs", "--tail", "20", name).stderr
            sys.exit(f"load-test: {label} not ready after 600 s:\n{logs}")
        expected = None
        if not status_only:  # each request's answer at concurrency 1, the standard for the loaded answers
            s = requests.Session()
            s.trust_env = False
            expected = [s.post(url, timeout=60, **kw).content for _, kw in reqs]
        result = {"image": image, "workers": workers, "ready_s": ready, "levels": []}
        for clients in levels:
            hold(url, reqs, clients, min(10, seconds), None, True)  # warm-up at this level, not counted
            level = hold(url, reqs, clients, seconds, expected, status_only, sampler)
            result["levels"].append(level)
            print(line_of(label, level))
        if soak:
            clients = result["levels"][-1]["clients"]
            result["soak"] = hold(url, reqs, clients, soak, expected, status_only, sampler)
            print(line_of(f"{label} soak {soak} s", result["soak"]))
        result["mps_clients"] = mps_clients(name)
        logs = sh("docker", "logs", "--tail", "5000", name)
        result["log_flags"] = [l for l in (logs.stdout + logs.stderr).splitlines() if LOG_FLAGS.search(l)][:20]
        if result["log_flags"]:
            print(f"{label} log: {len(result['log_flags'])} flagged line(s), first: {result['log_flags'][0][:160]}")
        print(f"{label} MPS: {result['mps_clients'] if result['mps_clients'] is not None else 'no server found'} "
              f"clients for {workers} workers")
        return result
    finally:
        sampler.close()
        sh("docker", "rm", "-f", name)


def line_of(label, level):
    return (f"{label} {level['clients']:>3} clients: {level['rps']:.1f} req/s, P50 {level['p50_ms']:.1f} P90 {level['p90_ms']:.1f} P99 "
            f"{level['p99_ms']:.1f} ms, errors {level['errors']}, mismatches {level['mismatches']}, restarts {level['worker_restarts']}; GPU "
            f"{level['gpu_util_mean']:.0f}% (max {level['gpu_util_max']:.0f}), {level['power_w_mean']:.0f} W, "
            f"{level['gpu_mem_mib_max']:.0f} MiB ({level['gpu_processes_max']} processes, largest {level['process_mem_mib_max']:.0f}), "
            f"SM >= {level['sm_mhz_min']:.0f} MHz; CPU {level['cpu_cores_mean']:.1f} cores, busiest process {level['busiest_process_cores']:.2f}, "
            f"busiest host core {level['busiest_host_core_pct']:.0f}%; RAM {level['ram_gib_max']:.1f} GiB; "
            f"client {level['client_processes']} processes, busiest {level['client_cores_max']:.2f} cores")


def peak_of(runs):
    """(peak worker count, {workers: top level}) of one image's runs {workers: result}."""
    top = {w: r["levels"][-1] for w, r in runs.items()}
    return max(top, key=lambda w: top[w]["rps"]), top


def curve_of(label, runs):
    peak_w, top = peak_of(runs)
    curve = ", ".join(f"{w}w {top[w]['rps']:.1f} req/s at {top[w]['gpu_util_mean']:.0f}% GPU" for w in sorted(top))
    return f"{label} throughput peaks at {peak_w} workers ({curve})"


def verdicts(runs, power_cap_w, memory_total_mib):
    """The wrap-up rule's checks over the delivered image's runs {workers: result}: where throughput peaks, what
    saturates there, and GPU memory against the one-worker peak times the worker count and across levels."""
    peak_w, top = peak_of(runs)
    print(curve_of("delivered", runs))
    t = top[peak_w]
    sides = []
    if t["gpu_util_mean"] >= GPU_SATURATED[0] or t["power_w_mean"] >= GPU_SATURATED[1] * power_cap_w:
        sides.append(f"the GPU ({t['gpu_util_mean']:.0f}%, {t['power_w_mean']:.0f} of {power_cap_w:.0f} W)")
    if t["busiest_process_cores"] >= CORE_SATURATED:
        sides.append(f"a worker process at {t['busiest_process_cores']:.2f} cores (CPU-bound)")
    free = os.cpu_count() - t["client_cores_sum"]  # the client's own cores are not the service's
    if t["cpu_cores_mean"] >= CORE_SATURATED * free:
        sides.append(f"all {free:.0f} host cores the client leaves ({t['cpu_cores_mean']:.1f} busy)")
    if DECODER_CEILING_RPS and t["rps"] >= 0.9 * DECODER_CEILING_RPS:
        sides.append(f"the hardware decoder ({t['rps']:.0f} req/s against its {DECODER_CEILING_RPS:.0f} decodes/s alone)")
    ok = True
    if sides:
        print(f"at the peak the saturated side is {'; '.join(sides)}")
    else:
        ok = False
        print(f"FAIL at the peak no side saturates: GPU {t['gpu_util_mean']:.0f}%, {t['power_w_mean']:.0f} of {power_cap_w:.0f} W, "
              f"busiest process {t['busiest_process_cores']:.2f} cores, {t['cpu_cores_mean']:.1f} of {os.cpu_count()} cores; "
              "find the wait (nsys on a worker: locks, synchronizes, the decode queue, CPU preprocessing) before the merge")
    mem_top = t["gpu_mem_mib_max"]
    first = runs[peak_w]["levels"][0]["gpu_mem_mib_max"]
    if mem_top > memory_total_mib:
        ok = False
        print(f"FAIL GPU memory {mem_top:.0f} MiB exceeds the card ({memory_total_mib:.0f} MiB)")
    if 1 in runs:
        expected = runs[1]["levels"][0]["gpu_mem_mib_max"] * peak_w
        if mem_top > MEMORY_SLACK * expected:
            ok = False
            print(f"FAIL GPU memory {mem_top:.0f} MiB at {peak_w} workers, expected about {expected:.0f} (one worker "
                  f"{runs[1]['levels'][0]['gpu_mem_mib_max']:.0f} MiB x {peak_w}); find what allocates per worker beyond one process's peak")
    if mem_top > MEMORY_SLACK * first:
        ok = False
        print(f"FAIL GPU memory grows with the level at {peak_w} workers: {first:.0f} -> {mem_top:.0f} MiB; find the allocation that "
              "grows (pools, per-request allocations, graphs) before the merge")
    if ok:
        print(f"GPU memory {mem_top:.0f} MiB at the peak, steady across levels")
    restarts = sum(l["worker_restarts"] for r in runs.values() for l in r["levels"] + [r.get("soak")] if l)
    if restarts:
        ok = False
        print(f"FAIL {restarts} worker restart(s) (new PIDs in the container) during the delivered runs; read the container log")
    for r in runs.values():
        if r.get("soak"):
            s = r["soak"]
            grew = [f"{k} {s[k + '_start']:.1f} -> {s[k + '_end']:.1f}" for k, unit in (("gpu_mem_mib", "MiB"), ("ram_gib", "GiB"))
                    if s[k + "_end"] > MEMORY_SLACK * max(s[k + "_start"], 1e-9)]
            if grew:
                ok = False
                print(f"FAIL memory grows over the soak at {r['workers']} workers: {'; '.join(grew)}; find the allocation that grows")
            else:
                print(f"soak at {r['workers']} workers: GPU memory {s['gpu_mem_mib_start']:.0f} -> {s['gpu_mem_mib_end']:.0f} MiB, "
                      f"RAM {s['ram_gib_start']:.1f} -> {s['ram_gib_end']:.1f} GiB, errors {s['errors']}, mismatches {s['mismatches']}")
        if r.get("log_flags"):
            fatal = [l for l in r["log_flags"] if re.search(r"traceback|MPS daemon not started", l, re.I)]
            if fatal:
                ok = False
            print(f"{'FAIL' if fatal else 'WARN'} container log at {r['workers']} workers: {len(r['log_flags'])} flagged line(s): "
                  + " | ".join(l[:100] for l in r["log_flags"][:3]))
    return ok


def climb(image, label, reqs, counts, seconds, status_only, memory_total_mib):
    """Runs of one image, worker count by worker count, until the throughput stops growing."""
    runs, last = {}, None
    for workers in counts:
        runs[workers] = run_image(image, label, reqs, seconds, status_only, workers)
        top = runs[workers]["levels"][-1]
        if last is not None and top["rps"] < GROWTH * last:
            print(f"{label}: {workers} workers give {top['rps']:.1f} req/s, under {GROWTH:.2f}x the previous count's {last:.1f}; the climb ends")
            break
        if top["errors"] or top["gpu_mem_mib_max"] > 0.9 * memory_total_mib:
            print(f"{label}: {workers} workers: {top['errors']} errors, {top['gpu_mem_mib_max']:.0f} MiB of {memory_total_mib:.0f}; the climb ends")
            break
        last = top["rps"]
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--image", required=True, help="the image built from the delivered tree, as CI builds it")
    p.add_argument("--baseline-image", help="the baseline commit's image, built the same way")
    p.add_argument("--workers", help=f"worker counts to climb through, default {','.join(map(str, SWEEP))}")
    p.add_argument("--seconds", type=int, default=60, help="per level, after a warm-up of up to 10 s")
    p.add_argument("--soak", type=int, default=600, help="seconds the delivered image holds the top level at its peak count; 0 skips")
    p.add_argument("--status-only", action="store_true", help="outputs are not deterministic: check status codes only")
    p.add_argument("--smoke", action="store_true", help="a functional check on a shared GPU (start, readiness, sampling, "
                   "verdicts) before the long exclusive run: allows other GPU processes; its numbers mean nothing")
    p.add_argument("--project")
    p.add_argument("--out")
    args = p.parse_args()
    if not (PORT and WORKER_CONFIG[0]):
        sys.exit("load-test: fill PORT and WORKER_CONFIG at the top of this file first")
    if gpu_busy() and not args.smoke:
        sys.exit("load-test: the GPU is in use, measure only on a free GPU:\n" + "\n".join(gpu_busy()))
    counts = [int(x) for x in args.workers.split(",")] if args.workers else SWEEP
    power_cap, mem_total = (float(x) for x in sh("nvidia-smi", "--query-gpu=power.limit,memory.total",
                                                   "--format=csv,noheader,nounits").stdout.strip().split(","))
    reqs = requests_for(adapter().project_root(args.project))
    results = {}
    for label, image in (("baseline", args.baseline_image), ("delivered", args.image)):
        if image:
            results[label] = climb(image, label, reqs, counts, args.seconds, args.status_only, mem_total)
    peak_w, _ = peak_of(results["delivered"])
    if args.soak:  # the soak at the peak count, as its own run (the peak run's container is gone)
        soaked = run_image(args.image, "delivered", reqs, args.seconds, args.status_only, peak_w, args.soak)
        results["delivered"][peak_w]["soak"] = soaked["soak"]
        results["delivered"][peak_w]["log_flags"] = soaked["log_flags"] or results["delivered"][peak_w]["log_flags"]
    if "baseline" in results:
        print(curve_of("baseline", results["baseline"]))
        for w in sorted(set(results["baseline"]) & set(results["delivered"])):
            for a, b in zip(results["baseline"][w]["levels"], results["delivered"][w]["levels"]):
                print(f"{w}w {a['clients']:>3} clients: delivered {b['rps']:.1f} req/s vs baseline {a['rps']:.1f} "
                      f"({b['rps'] / max(a['rps'], 1e-9):.2f}x), P99 {b['p99_ms']:.1f} vs {a['p99_ms']:.1f} ms")
    ok = verdicts(results["delivered"], power_cap, mem_total)
    for label, runs in results.items():  # a level whose client process ran near a full core measured the client
        for w, r in runs.items():
            for level in r["levels"] + ([r["soak"]] if r.get("soak") else []):
                if level["client_cores_max"] >= CLIENT_SATURATED or level["client_late_s_max"] > 1:
                    ok = False
                    print(f"FAIL {label} {w}w {level['clients']} clients: invalid, a client process at "
                          f"{level['client_cores_max']:.2f} cores (late {level['client_late_s_max']:.1f} s); lower "
                          "CLIENTS_PER_PROCESS and rerun this part")
    if args.out:
        Path(args.out).write_text(json.dumps({"power_cap_w": power_cap, "memory_total_mib": mem_total, "verdict_ok": ok,
                                              "runs": {k: {str(w): r for w, r in v.items()} for k, v in results.items()}}, indent=1))


if __name__ == "__main__":
    main()
