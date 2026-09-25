"""Bench the pipeline end to end: cold start, warm latency per scenario, GPU state, stage breakdown, A/B.

Usage: python main.py [--project DIR] [--candidate FILE] [--scenario NAME ...] [--warmup N] [--repeat N]
                      [--stages] [--out FILE]
       python main.py --ab FILE [--rounds N] [--noise MS]   # alternate baseline and candidate processes
       python main.py --ab baseline [--rounds N]           # A/A: the project's code on both sides
Everything project-specific lives in lx-work/adapter/pipeline.py (see its contract).
Prints at most 10 summary lines; the full result goes to --out as JSON.
"""
import argparse
import contextlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

START = time.perf_counter()
LX_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LX_WORK))
import adapter.pipeline as adapter  # noqa: E402


class Timed:
    """Callable proxy that adds its wall time to totals[stage], synchronizing the GPU around it when the
    adapter says the stage runs asynchronous GPU work; other attributes pass through."""

    def __init__(self, inner, totals, stage):
        self._inner, self._totals, self._stage = inner, totals, stage

    def __call__(self, *args, **kwargs):
        adapter.sync()
        t0 = time.perf_counter()
        try:
            return self._inner(*args, **kwargs)
        finally:
            adapter.sync()
            self._totals[self._stage] += time.perf_counter() - t0

    def __getattr__(self, name):
        return getattr(self._inner, name)


def wrap_stages(pipeline):
    totals = dict.fromkeys(adapter.STAGES, 0.0)
    for stage, paths in adapter.STAGES.items():
        for path in paths:
            *parents, attr = path.split(".")
            owner = pipeline
            for part in parents:
                owner = getattr(owner, part)
            setattr(owner, attr, Timed(getattr(owner, attr), totals, stage))
    return totals


@contextlib.contextmanager
def quiet(log_path):
    """Send stdout and stderr to a log at the file-descriptor level: native libraries such as TensorRT
    write past sys.stdout and would flood the summary."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log_path, "a") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [line.split(", ") for line in out.strip().splitlines()]


def gpu_state():
    name, driver, clock, temp, power = smi("gpu=name,driver_version,clocks.sm,temperature.gpu,power.draw")[0]
    return {"gpu": name, "driver": driver, "sm_clock_mhz": int(clock), "temp_c": int(temp), "power_w": float(power)}


def process_gpu_mib():
    for pid, used in smi("compute-apps=pid,used_memory"):
        if int(pid) == os.getpid():
            return int(used)
    return None


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def run_scenario(pipeline, args, warmup, repeat, totals):
    for _ in range(warmup):
        adapter.call(pipeline, adapter.prepare(args))
    adapter.reset_peak_memory()
    samples, stages, out = [], [], None
    for _ in range(repeat):
        prepared = adapter.prepare(args)  # fresh inputs, outside the timer
        if totals is not None:
            totals.update(dict.fromkeys(totals, 0.0))
        adapter.sync()
        t0 = time.perf_counter()
        out = adapter.call(pipeline, prepared)
        adapter.sync()
        total = (time.perf_counter() - t0) * 1000
        samples.append(total)
        if totals is not None:
            row = {k: v * 1000 for k, v in totals.items()}
            row["other"] = total - sum(row.values())
            stages.append(row)
    result = {"median": statistics.median(samples), "p90": percentile(samples, 90), "p99": percentile(samples, 99),
              "spread_p10_p90": percentile(samples, 90) - percentile(samples, 10), "samples": samples,
              "memory": adapter.peak_memory(), "summary": adapter.describe(out)}
    if stages:
        result["stages_median_ms"] = {k: statistics.median(r[k] for r in stages) for k in stages[0]}
    return result


def ab(candidate, rounds, noise, scenarios, passthrough):
    """Alternate fresh baseline and candidate processes, one process per scenario: state a scenario leaves
    behind (allocator, caches) can shift the next scenario by several milliseconds. With candidate "baseline"
    both sides run the project's code (A/A), and each scenario's largest difference within a round is its
    noise floor."""
    aa = candidate == "baseline"
    floors = {name: noise if noise is not None else adapter.NOISE_MS.get(name) for name in scenarios}
    if not aa and None in floors.values():
        sys.exit(f"error: no noise floor for {[n for n, f in floors.items() if f is None]}; "
                 "measure it with --ab baseline and set adapter.NOISE_MS")
    medians = {name: {"baseline": [], "candidate": []} for name in scenarios}
    clocks = set()
    for i in range(rounds):
        for name in scenarios:
            for side in ("baseline", "candidate")[::1 if i % 2 == 0 else -1]:  # ABBA cancels any effect of running second
                out = LX_WORK / "local" / "logs" / f"ab-{side}-{name}-{i}.json"
                cmd = [sys.executable, __file__, "--out", str(out), "--scenario", name, *passthrough]
                if side == "candidate" and not aa:
                    cmd += ["--candidate", candidate]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                r = json.loads(out.read_text())
                medians[name][side].append(r["scenarios"][name]["median"])
                clocks |= {r["env"]["sm_clock_mhz"], r["env"]["sm_clock_after_mhz"]}
    print(f"{'A/A' if aa else 'A/B'} over {rounds} rounds, one fresh process per scenario and side; "
          f"SM clocks {sorted(clocks)} MHz")
    for name, m in medians.items():
        b, c = statistics.median(m["baseline"]), statistics.median(m["candidate"])
        if aa:
            floor = max(abs(x - y) for x, y in zip(m["baseline"], m["candidate"]))
            print(f"{name:14s} sides {b:8.2f} {c:8.2f} ms  noise floor {floor:.2f} ms ({100 * floor / b:.1f}%)")
            continue
        n = floors[name]
        verdict = "gain" if b - c > n else ("loss" if c - b > n else "inside noise")
        if rounds < 3:
            verdict = "no verdict under 3 rounds"
        print(f"{name:14s} baseline {b:8.2f} candidate {c:8.2f} ms  {c - b:+7.2f} ({100 * (c - b) / b:+.1f}%)  "
              f"{verdict} (noise {n:g})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project")
    ap.add_argument("--candidate")
    ap.add_argument("--scenario", nargs="*", default=list(adapter.SCENARIOS))
    ap.add_argument("--warmup", type=int, default=adapter.WARMUP)
    ap.add_argument("--repeat", type=int, default=adapter.REPEAT)
    ap.add_argument("--stages", action="store_true", help="add a per-stage breakdown (stage mode)")
    ap.add_argument("--out", default=str(LX_WORK / "local" / "logs" / "bench-last.json"))
    ap.add_argument("--ab", metavar="CANDIDATE", help='a candidate file, or "baseline" for an A/A run')
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--noise", type=float, help="one noise floor in ms for every scenario, instead of adapter.NOISE_MS")
    a = ap.parse_args()
    if a.ab:
        passthrough = ["--warmup", str(a.warmup), "--repeat", str(a.repeat)]
        return ab(a.ab, a.rounds, a.noise, a.scenario, passthrough + (["--project", a.project] if a.project else []))
    project = adapter.project_root(a.project)
    before = gpu_state()
    t_init = time.perf_counter()
    import_ms = (t_init - START) * 1000
    with quiet(LX_WORK / "local" / "logs" / "bench-init.log"):
        pipeline = adapter.load_pipeline(project, a.candidate)
    adapter.sync()
    init_ms = (time.perf_counter() - t_init) * 1000
    inputs = {name: adapter.load_scenario(project, name) for name in a.scenario}
    t_first = time.perf_counter()
    with quiet(LX_WORK / "local" / "logs" / "bench-init.log"):
        adapter.call(pipeline, adapter.prepare(inputs[a.scenario[0]]))
    adapter.sync()
    first_ms = (time.perf_counter() - t_first) * 1000
    cold_ms = import_ms + init_ms + first_ms  # reading test inputs is not part of cold start
    totals = wrap_stages(pipeline) if a.stages else None
    commit = subprocess.run(["git", "-C", str(project), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    result = {"env": {**before, "python": platform.python_version(), "argv": sys.argv, "project_commit": commit,
                      "candidate": a.candidate, **adapter.versions()},
              "cold": {"total_ms": cold_ms, "import_ms": import_ms, "init_ms": init_ms, "first_call_ms": first_ms},
              "scenarios": {}}
    with quiet(LX_WORK / "local" / "logs" / "bench-run.log"):
        for name, args in inputs.items():
            result["scenarios"][name] = run_scenario(pipeline, args, a.warmup, a.repeat, totals)
    result["gpu_mib_process"] = process_gpu_mib()
    result["env"]["sm_clock_after_mhz"] = gpu_state()["sm_clock_mhz"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"cold start {cold_ms:.0f} ms (init {init_ms:.0f}, first call {first_ms:.0f}); process GPU memory "
          f"{result['gpu_mib_process']} MiB; SM clock {before['sm_clock_mhz']} -> {result['env']['sm_clock_after_mhz']} MHz")
    for name, r in result["scenarios"].items():
        line = (f"{name:14s} {r['summary']} median {r['median']:.2f} ms p90 {r['p90']:.2f} p99 {r['p99']:.2f} "
                f"spread {r['spread_p10_p90']:.2f} {r['memory']}")
        if "stages_median_ms" in r:
            line += " | " + " ".join(f"{k} {v:.1f}" for k, v in r["stages_median_ms"].items())
        print(line)
    print(f"full result: {a.out}")


if __name__ == "__main__":
    main()
