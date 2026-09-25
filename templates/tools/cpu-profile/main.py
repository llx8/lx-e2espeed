"""Attribute CPU time per function with cProfile, for steps written inline that stage mode cannot wrap.

Usage: python main.py [--project DIR] [--candidate FILE] [--scenario NAME] [--repeat N] [--top N]
Prints the top functions by own time per call; the full stats go to lx-work/local/profiles/.
GPU work shows up only as the CPU time spent launching or waiting for it.
"""
import argparse
import contextlib
import cProfile
import os
import pstats
import sys
from pathlib import Path

LX_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LX_WORK))
import adapter.pipeline as adapter  # noqa: E402


@contextlib.contextmanager
def quiet():
    log = LX_WORK / "local" / "logs" / "cpu-profile.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    saved = os.dup(1), os.dup(2)
    with open(log, "a") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project")
    ap.add_argument("--candidate")
    ap.add_argument("--scenario", default=adapter.SCENARIOS[0])
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    project = adapter.project_root(a.project)
    inputs = adapter.load_scenario(project, a.scenario)
    with quiet():
        pipeline = adapter.load_pipeline(project, a.candidate)
        for _ in range(3):
            adapter.call(pipeline, adapter.prepare(inputs))
    calls = [adapter.prepare(inputs) for _ in range(a.repeat)]
    profiler = cProfile.Profile()
    with quiet():
        adapter.sync()
        profiler.enable()
        for prepared in calls:
            adapter.call(pipeline, prepared)
        adapter.sync()
        profiler.disable()
    out = LX_WORK / "local" / "profiles" / f"cpu-{a.scenario}.pstats"
    out.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(out)
    stats = pstats.Stats(profiler)
    rows = sorted(stats.stats.items(), key=lambda kv: -kv[1][2])[:a.top]
    for (file, line, func), (_, ncalls, tottime, _, _) in rows:
        where = f"{Path(file).name}:{line}" if line else "builtin"
        print(f"{1000 * tottime / a.repeat:8.2f} ms/call  {ncalls // a.repeat:5d} calls  {func}  ({where})")
    print(f"full stats: {out}")


if __name__ == "__main__":
    main()
