"""Bench the pipeline end to end: cold start, warm latency per scenario, GPU state, stage breakdown, A/B.

Usage: python main.py [--project DIR] [--candidate FILE] [--scenario NAME ...] [--warmup N] [--repeat N]
                      [--stages] [--out FILE]
       python main.py --ab FILE [--rounds N] [--noise MS] [--out FILE]  # alternate the project tree (best) and a candidate
       python main.py --ab best [--rounds N]               # A/A: the project tree on both sides
With --project, the side other than the candidate is that tree, for example a copy from exp.sh export.
Everything project-specific lives in lx-work/adapter/pipeline.py (see its contract); adapter.BASELINE names the
baseline's bench JSON files, printed beside every result as the baseline set, a descriptive comparison.
Prints at most 10 summary lines; the full result goes to --out as JSON. An A/B writes every round of both sides
there; its per-round files in local/logs/ are scratch that the next A/B overwrites.
"""
import argparse
import contextlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
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


@contextlib.contextmanager
def gpu_sampler(period_ms=500):
    """Sample SM clock, temperature, and power while the block runs; the yielded dict is filled on exit.
    Readings before and after alone miss throttling in between."""
    summary = {}
    with tempfile.TemporaryFile("w+") as log:
        proc = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                                 "--format=csv,noheader,nounits", "-lms", str(period_ms)],
                                stdout=log, stderr=subprocess.DEVNULL)
        try:
            yield summary
        finally:
            proc.terminate()
            proc.wait()
            log.seek(0)
            rows = []
            for line in log:
                parts = line.strip().split(", ")
                try:
                    if len(parts) == 3:
                        rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
                except ValueError:
                    pass  # "[N/A]", or a line cut off by terminate
            summary["samples"] = len(rows)
            if rows:
                clocks, temps, powers = zip(*rows)
                summary.update(sm_clock_min_mhz=min(clocks), sm_clock_max_mhz=max(clocks), temp_max_c=max(temps),
                               power_mean_w=round(statistics.mean(powers), 1), rows=rows)


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
    p99 = percentile(samples, 99) if len(samples) >= 100 else None  # fewer samples cannot resolve a 1% tail
    result = {"median": statistics.median(samples), "p90": percentile(samples, 90), "p99": p99, "max": max(samples),
              "spread_p10_p90": percentile(samples, 90) - percentile(samples, 10), "samples": samples,
              "memory": adapter.peak_memory(), "summary": adapter.describe(out)}
    if stages:
        result["stages_median_ms"] = {k: statistics.median(r[k] for r in stages) for k in stages[0]}
    return result


def round_row(run, name):
    r = run["scenarios"][name]
    return {"median": r["median"], "p90": r["p90"], "p99": r["p99"], "max": r["max"], "memory": r["memory"],
            "gpu_mib_process": run["gpu_mib_process"], "project_commit": run["env"]["project_commit"]}


def combine(runs, name):
    """One set from several runs of a scenario: the median of each number across the runs, as the A/B judges
    medians. Memory texts that differ between runs are all shown."""
    rows = [round_row(run, name) for run in runs]
    p99s = [r["p99"] for r in rows]
    mibs = [r["gpu_mib_process"] for r in rows if r["gpu_mib_process"] is not None]
    return {"runs": len(rows), "median": statistics.median(r["median"] for r in rows),
            "p90": statistics.median(r["p90"] for r in rows),
            "p99": None if None in p99s else statistics.median(p99s), "max": statistics.median(r["max"] for r in rows),
            "memory": " | ".join(sorted({str(r["memory"]) for r in rows})),
            "gpu_mib_process": statistics.median_low(mibs) if mibs else None}


def load_baseline():
    """The baseline's bench runs (PROJECT.md, Metric) from adapter.BASELINE, a glob relative to lx-work that
    onboarding step 2 sets. They come from another session, and possibly another environment, so a change
    against them is descriptive, never a verdict."""
    if not hasattr(adapter, "BASELINE"):
        sys.exit("error: the adapter has no BASELINE; add it as in lx-e2espeed/templates/tools/adapter/")
    if not adapter.BASELINE:
        return None
    files = sorted(LX_WORK.glob(adapter.BASELINE))
    if not files:
        sys.exit(f"error: adapter.BASELINE {adapter.BASELINE!r} matches no file in lx-work")
    runs = [json.loads(f.read_text()) for f in files]
    commits = ",".join(sorted({run["env"]["project_commit"] for run in runs}))
    where = f"{len(runs)} runs of {commits} in {files[0].parent.relative_to(LX_WORK)}, descriptive"
    return {"files": [str(f.relative_to(LX_WORK)) for f in files], "where": where, "runs": runs}


def baseline_set(baseline, name):
    if baseline is None or any(name not in run["scenarios"] for run in baseline["runs"]):
        return None
    return combine(baseline["runs"], name)


def set_line(name, label, s):
    tail = f"p99 {s['p99']:.2f}" if s["p99"] is not None else f"max {s['max']:.2f}"
    mib = f", process {s['gpu_mib_process']} MiB" if s["gpu_mib_process"] is not None else ""
    return f"{name:14s} {label:9s} median {s['median']:.2f} ms p90 {s['p90']:.2f} {tail} | {s['memory']}{mib}"


def baseline_line(name, baseline, base):
    if baseline is None:
        return f"{name:14s} baseline  not set (adapter.BASELINE, onboarding step 2)"
    if base is None:
        return f"{name:14s} baseline  not recorded in {baseline['where']}"
    return set_line(name, "baseline", base) + f" | {baseline['where']}"


def change(c, b):
    return f"{c - b:+.2f} ({100 * (c - b) / b:+.1f}%)"


def ab(candidate, rounds, noise, scenarios, passthrough, project, out_path):
    """Alternate fresh processes of the project tree (best, or the --project tree) and the candidate, one process
    per scenario: state a scenario leaves behind (allocator, caches) can shift the next scenario by several
    milliseconds. With candidate "best" both sides run the project tree (A/A), and each scenario's largest
    difference within a round is its noise floor. An A/B prints three sets per scenario: baseline, best (before),
    and the candidate (after)."""
    aa = candidate == "best"
    label = Path(project).name if project else "best"
    floors = {name: noise if noise is not None else adapter.NOISE_MS.get(name) for name in scenarios}
    if not aa and None in floors.values():
        sys.exit(f"error: no noise floor for {[n for n, f in floors.items() if f is None]}; "
                 "measure it with --ab best and set adapter.NOISE_MS")
    baseline = None if aa else load_baseline()
    runs = {name: {"base": [], "candidate": []} for name in scenarios}
    clocks = set()
    for i in range(rounds):
        for name in scenarios:
            for side in ("base", "candidate")[::1 if i % 2 == 0 else -1]:  # ABBA cancels any effect of running second
                out = LX_WORK / "local" / "logs" / f"ab-{side}-{name}-{i}.json"
                cmd = [sys.executable, __file__, "--out", str(out), "--scenario", name, *passthrough]
                if side == "candidate" and not aa:
                    cmd += ["--candidate", candidate]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                r = json.loads(out.read_text())
                runs[name][side].append(r)
                env, gpu = r["env"], r["env"]["gpu_during_run"]
                clocks |= {env["sm_clock_mhz"], env["sm_clock_after_mhz"]}
                if gpu["samples"]:
                    clocks |= {gpu["sm_clock_min_mhz"], gpu["sm_clock_max_mhz"]}
    result = {"mode": "A/A" if aa else "A/B", "label": label, "candidate": candidate, "rounds": rounds,
              "sm_clock_mhz": [min(clocks), max(clocks)], "baseline_files": baseline["files"] if baseline else [],
              "scenarios": {}}
    sides = label if aa else f"{label} and {candidate}"
    print(f"{result['mode']} of {sides} over {rounds} rounds, one fresh process per scenario and side; "
          f"SM clock {min(clocks)}-{max(clocks)} MHz across all of them; full result {out_path}")
    for name, sides in runs.items():
        b, c = combine(sides["base"], name), combine(sides["candidate"], name)
        entry = result["scenarios"][name] = {
            "base": {**b, "rounds": [round_row(r, name) for r in sides["base"]]},
            "candidate": {**c, "rounds": [round_row(r, name) for r in sides["candidate"]]}}
        if aa:
            medians = [(x["median"], y["median"]) for x, y in zip(entry["base"]["rounds"], entry["candidate"]["rounds"])]
            entry["noise_floor_ms"] = floor = max(abs(x - y) for x, y in medians)
            print(f"{name:14s} sides {b['median']:8.2f} {c['median']:8.2f} ms  noise floor {floor:.2f} ms "
                  f"({100 * floor / b['median']:.1f}%)")
            continue
        n = floors[name]
        verdict = "gain" if b["median"] - c["median"] > n else ("loss" if c["median"] - b["median"] > n else "inside noise")
        if rounds < 3:
            verdict = "no verdict under 3 rounds"
        entry.update(noise_ms=n, verdict=verdict)
        base = entry["baseline"] = baseline_set(baseline, name)
        best_vs = f" | vs baseline {change(b['median'], base['median'])}" if base else ""
        cand_vs = f", vs baseline {change(c['median'], base['median'])}" if base else ""
        print(baseline_line(name, baseline, base))
        print(set_line(name, label, b) + best_vs)
        print(set_line(name, "candidate", c) + f" | vs {label} {change(c['median'], b['median'])} {verdict} "
              f"(noise {n:g}){cand_vs}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project")
    ap.add_argument("--candidate")
    ap.add_argument("--scenario", nargs="*", default=list(adapter.SCENARIOS))
    ap.add_argument("--warmup", type=int, default=adapter.WARMUP)
    ap.add_argument("--repeat", type=int, default=adapter.REPEAT)
    ap.add_argument("--stages", action="store_true", help="add a per-stage breakdown (stage mode)")
    ap.add_argument("--out", help="full result as JSON (default lx-work/local/logs/bench-last.json, or ab-last.json "
                                  "for an A/B); give each A/B its own, which the next one would overwrite")
    ap.add_argument("--ab", metavar="CANDIDATE", help='a candidate file, or "best" for an A/A run')
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--noise", type=float, help="one noise floor in ms for every scenario, instead of adapter.NOISE_MS")
    a = ap.parse_args()
    a.out = a.out or str(LX_WORK / "local" / "logs" / ("ab-last.json" if a.ab else "bench-last.json"))
    if a.ab:
        passthrough = ["--warmup", str(a.warmup), "--repeat", str(a.repeat)]
        return ab(a.ab, a.rounds, a.noise, a.scenario, passthrough + (["--project", a.project] if a.project else []),
                  a.project, a.out)
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
    with gpu_sampler() as gpu, quiet(LX_WORK / "local" / "logs" / "bench-run.log"):
        for name, args in inputs.items():
            result["scenarios"][name] = run_scenario(pipeline, args, a.warmup, a.repeat, totals)
    result["gpu_mib_process"] = process_gpu_mib()
    result["env"]["sm_clock_after_mhz"] = after = gpu_state()["sm_clock_mhz"]
    result["env"]["gpu_during_run"] = gpu
    baseline = load_baseline()  # after the timed part, so reading its files stays out of cold start
    bases = {name: baseline_set(baseline, name) for name in result["scenarios"]}
    result["baseline"] = {"files": baseline["files"], "scenarios": bases} if baseline else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"cold start {cold_ms:.0f} ms (init {init_ms:.0f}, first call {first_ms:.0f}); process GPU memory "
          f"{result['gpu_mib_process']} MiB")
    if gpu["samples"]:
        print(f"GPU during the run: SM clock {gpu['sm_clock_min_mhz']}-{gpu['sm_clock_max_mhz']} MHz (before "
              f"{before['sm_clock_mhz']}, after {after}), max {gpu['temp_max_c']} C, mean {gpu['power_mean_w']} W, "
              f"{gpu['samples']} samples")
    else:
        print(f"no GPU samples during the run; SM clock before {before['sm_clock_mhz']}, after {after} MHz")
    for name, r in result["scenarios"].items():
        tail = f"p99 {r['p99']:.2f}" if r["p99"] is not None else f"max {r['max']:.2f}"
        line = (f"{name:14s} {r['summary']} median {r['median']:.2f} ms p90 {r['p90']:.2f} {tail} "
                f"spread {r['spread_p10_p90']:.2f} {r['memory']}")
        if "stages_median_ms" in r:
            line += " | " + " ".join(f"{k} {v:.1f}" for k, v in r["stages_median_ms"].items())
        print(line)
        base = bases[name]
        print(baseline_line(name, baseline, base) + (f"; this run {change(r['median'], base['median'])}" if base else ""))
    print(f"full result: {a.out}")


if __name__ == "__main__":
    main()
