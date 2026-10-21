"""Load-test the delivered service image on this machine's GPU, beside the baseline image, before the merge.

Starts each image as production runs it (its own CMD, so its MPS daemon and worker count), waits until it answers,
then holds each concurrency level for --seconds with closed-loop clients and samples while it runs: throughput,
latency, errors, responses that differ from the same request's answer at concurrency 1, GPU (utilization, memory,
power, SM clock), the container's CPU cores and RAM, its busiest process (a worker pinned near 1.0 core is
CPU-bound), and the busiest host core. Then it checks that the image's MPS server lists as many clients as workers.
Refuses to start while another process uses the GPU; kills the container when MemAvailable falls under 20% of RAM.
--workers sweeps the worker count (the image's config file mounted with that count, WORKER_CONFIG): one run of both
images per count, then the verdicts of rules/wrap-up.md: the count where the delivered throughput peaks, which side
saturates there (GPU utilization or power, a worker near 1.0 core, all host cores) or none (a cause to find before the
merge), and GPU memory against the one-worker peak times the worker count and across levels.

Run, from the project directory, under a small cap (it covers this client; the container has its own, LX_MEM):
  <py> lx-work/tools/load-test/main.py --image IMAGE [--baseline-image IMAGE] [--levels 1,6,12] [--seconds 60]
       [--workers 1,6,9,12] [--status-only] [--out FILE]
Prints one line per image, worker count, and level, one comparison line per level, and the verdicts; the full result
goes to --out as JSON.
"""
import argparse
import json
import os
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
from adapter import pipeline as ad  # noqa: E402

# TODO at wrap-up: from the service's config (PROJECT.md, Deployment).
PORT = 0         # TODO: the port the framework listens on inside the image (config.toml listen_addr)
WORKERS = 0      # TODO: production's worker count (config.toml worker_num), for the MPS client check
PATH_URL = "/"   # TODO if the service answers on another path
WORKER_CONFIG = ("", "worker_num")  # TODO for --workers: (the config file's path inside the image, its worker-count key)
HOST_PORT = 18100
GPU_SATURATED = (85, 0.9)  # utilization %, or share of the power cap, above which the GPU counts as the saturated side
CORE_SATURATED = 0.9       # a worker process above this many cores is CPU-bound
MEMORY_SLACK = 1.2         # GPU memory above this times the expected peak, or growing by this across levels, needs a cause


def requests_for(project):
    """[(label, kwargs for requests.post)] in the framework's request format: real production inputs."""
    # TODO: for example, every request of every scenario posted as the framework's demo client does:
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
        self.smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm",
                                     "--format=csv,noheader,nounits", "-lms", "500"], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self.read_smi, daemon=True).start()

    def read_smi(self):
        for line in self.smi.stdout:
            try:
                self.gpu.append([float(v) for v in line.split(",")])
            except ValueError:
                pass

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
        return {"gpu_util_mean": mean([x[0] for x in g]), "gpu_util_max": max([x[0] for x in g], default=0),
                "gpu_mem_mib_max": max([x[1] for x in g], default=0), "power_w_mean": mean([x[2] for x in g]),
                "sm_mhz_min": min([x[3] for x in g], default=0), "cpu_cores_mean": mean(self.cores[c:]),
                "busiest_process_cores": max(self.proc_max[c:], default=0), "busiest_host_core_pct": max(self.host_max[c:], default=0),
                "ram_gib_max": max(self.ram[c:], default=0)}

    def marks(self):
        return len(self.gpu), len(self.cores)

    def close(self):
        self.stop.set()
        self.smi.terminate()


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
    r = sh("docker", "run", "-d", "--name", name, "--gpus", "all", f"--memory={os.environ.get('LX_MEM', '48g')}",
           f"--memory-swap={os.environ.get('LX_MEM', '48g')}", "-p", f"{HOST_PORT}:{PORT}", *mount, image, timeout=300)
    if r.returncode:
        sys.exit(f"load-test: cannot start {image}: {r.stderr.strip()[-300:]}")
    return int(sh("docker", "inspect", "-f", "{{.State.Pid}}", name).stdout.strip())


def wait_ready(url, reqs, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            if requests.post(url, timeout=60, **reqs[0][1]).status_code == 200:
                return time.time() - t0
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def hold(url, reqs, clients, seconds, expected, status_only):
    """Closed loop: each client sends the next request as soon as the last one is answered."""
    local, lock, out = threading.local(), threading.Lock(), []
    deadline, counter = time.time() + seconds, iter(range(10 ** 9))

    def client(_):
        local.s = requests.Session()
        local.s.trust_env = False
        while time.time() < deadline:
            with lock:
                i = next(counter) % len(reqs)
            t = time.perf_counter()
            try:
                r = local.s.post(url, timeout=60, **reqs[i][1])
                code, body = r.status_code, r.content
            except requests.RequestException as e:
                code, body = type(e).__name__, b""
            ms = (time.perf_counter() - t) * 1000
            with lock:
                out.append((i, code, ms, code == 200 and not status_only and expected is not None and body != expected[i]))

    t0 = time.time()
    with ThreadPoolExecutor(clients) as ex:
        list(ex.map(client, range(clients)))
    elapsed = time.time() - t0
    ms = sorted(m for _, code, m, _ in out if code == 200)
    pick = lambda q: ms[min(len(ms) - 1, int(q * len(ms)))] if ms else 0.0
    return {"clients": clients, "requests": len(out), "rps": len(ms) / elapsed, "p50_ms": pick(0.5), "p99_ms": pick(0.99),
            "errors": sum(1 for _, code, _, _ in out if code != 200), "mismatches": sum(1 for *_, bad in out if bad)}


def mps_clients(name):
    """Clients of the image's MPS server(s), or None when the image runs no MPS server it can be asked about."""
    servers = sh("docker", "exec", name, "sh", "-c", "echo get_server_list | nvidia-cuda-mps-control").stdout.split()
    if not servers or not all(s.isdigit() for s in servers):
        return None
    return sum(len(sh("docker", "exec", name, "sh", "-c", f"echo get_client_list {s} | nvidia-cuda-mps-control").stdout.split())
               for s in servers)


def run_image(image, label, reqs, levels, seconds, status_only, workers=None):
    """One image at one worker count (None: as the image is configured) through every level."""
    name = f"lx-load-{label}-{os.getpid()}"
    url = f"http://127.0.0.1:{HOST_PORT}{PATH_URL}"
    label = f"{label} {workers}w" if workers else label
    workers = workers or WORKERS
    pid = start(image, name, config_with_workers(image, workers) if workers != WORKERS or WORKER_CONFIG[0] else None)
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
        for clients in levels or sorted({1, workers, 2 * workers}):
            hold(url, reqs, clients, min(10, seconds), None, True)  # warm-up at this level, not counted
            marks = sampler.marks()
            level = hold(url, reqs, clients, seconds, expected, status_only)
            level.update(sampler.window(marks))
            result["levels"].append(level)
            print(f"{label} {clients:>3} clients: {level['rps']:.1f} req/s, P50 {level['p50_ms']:.1f} ms, P99 "
                  f"{level['p99_ms']:.1f} ms, errors {level['errors']}, mismatches {level['mismatches']}; GPU "
                  f"{level['gpu_util_mean']:.0f}% (max {level['gpu_util_max']:.0f}), {level['power_w_mean']:.0f} W, "
                  f"{level['gpu_mem_mib_max']:.0f} MiB, SM >= {level['sm_mhz_min']:.0f} MHz; CPU "
                  f"{level['cpu_cores_mean']:.1f} cores, busiest process {level['busiest_process_cores']:.2f}, "
                  f"busiest host core {level['busiest_host_core_pct']:.0f}%; RAM {level['ram_gib_max']:.1f} GiB")
        result["mps_clients"] = mps_clients(name)
        print(f"{label} MPS: {result['mps_clients'] if result['mps_clients'] is not None else 'no server found'} "
              f"clients for {workers} workers")
        return result
    finally:
        sampler.close()
        sh("docker", "rm", "-f", name)


def verdicts(runs, power_cap_w, memory_total_mib):
    """The wrap-up rule's checks over the delivered image's runs {workers: result}: where throughput peaks, what
    saturates there, and GPU memory against the one-worker peak times the worker count and across levels."""
    top = {w: r["levels"][-1] for w, r in runs.items()}
    peak_w = max(top, key=lambda w: top[w]["rps"])
    curve = ", ".join(f"{w}w {top[w]['rps']:.1f} req/s at {top[w]['gpu_util_mean']:.0f}% GPU" for w in sorted(top))
    print(f"delivered throughput peaks at {peak_w} workers ({curve})")
    t = top[peak_w]
    sides = []
    if t["gpu_util_mean"] >= GPU_SATURATED[0] or t["power_w_mean"] >= GPU_SATURATED[1] * power_cap_w:
        sides.append(f"the GPU ({t['gpu_util_mean']:.0f}%, {t['power_w_mean']:.0f} of {power_cap_w:.0f} W)")
    if t["busiest_process_cores"] >= CORE_SATURATED:
        sides.append(f"a worker process at {t['busiest_process_cores']:.2f} cores (CPU-bound)")
    if t["cpu_cores_mean"] >= CORE_SATURATED * os.cpu_count():
        sides.append(f"all {os.cpu_count()} host cores ({t['cpu_cores_mean']:.1f} busy)")
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
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--image", required=True, help="the image built from the delivered tree, as CI builds it")
    p.add_argument("--baseline-image", help="the baseline commit's image, built the same way")
    p.add_argument("--levels", help="client counts, default 1, workers, 2x workers")
    p.add_argument("--workers", help="worker counts to sweep (WORKER_CONFIG), e.g. 1,<production>,<1.5x>,<2x>; default the production count")
    p.add_argument("--seconds", type=int, default=60, help="per level, after a warm-up of up to 10 s")
    p.add_argument("--status-only", action="store_true", help="outputs are not deterministic: check status codes only")
    p.add_argument("--project")
    p.add_argument("--out")
    args = p.parse_args()
    if not (PORT and WORKERS):
        sys.exit("load-test: fill PORT and WORKERS at the top of this file first")
    if gpu_busy():
        sys.exit("load-test: the GPU is in use, measure only on a free GPU:\n" + "\n".join(gpu_busy()))
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    counts = [int(x) for x in args.workers.split(",")] if args.workers else [WORKERS]
    power_cap, mem_total = (float(x) for x in sh("nvidia-smi", "--query-gpu=power.limit,memory.total",
                                                   "--format=csv,noheader,nounits").stdout.strip().split(","))
    reqs = requests_for(ad.project_root(args.project))
    results = {}
    for workers in counts:
        for label, image in (("baseline", args.baseline_image), ("delivered", args.image)):
            if image:
                results.setdefault(label, {})[workers] = run_image(image, label, reqs, levels, args.seconds, args.status_only, workers)
        if "baseline" in results:
            for a, b in zip(results["baseline"][workers]["levels"], results["delivered"][workers]["levels"]):
                print(f"{workers}w {a['clients']:>3} clients: delivered {b['rps']:.1f} req/s vs baseline {a['rps']:.1f} "
                      f"({b['rps'] / max(a['rps'], 1e-9):.2f}x), P99 {b['p99_ms']:.1f} vs {a['p99_ms']:.1f} ms")
    ok = verdicts(results["delivered"], power_cap, mem_total)
    if args.out:
        Path(args.out).write_text(json.dumps({"power_cap_w": power_cap, "memory_total_mib": mem_total, "verdict_ok": ok,
                                              "runs": {k: {str(w): r for w, r in v.items()} for k, v in results.items()}}, indent=1))


if __name__ == "__main__":
    main()
