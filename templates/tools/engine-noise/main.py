"""Measure how far rebuilding the engine from the same model moves the outputs: the bound for L1.

Usage: python main.py [--project DIR] [--builds N] [--out FILE]
An engine build picks tactics by timing, so two builds of one ONNX can differ (card env/engine-build-variance);
a change whose difference stays inside this spread is L1. Needs in the adapter: ENGINE, the path of the engine the
project ships or builds; build_engine(project, out), one build into the file `out` exactly as the project builds
its engine; and load_pipeline(project, candidate=None, engine=None) that loads the given engine.
Builds N engines into lx-work/local/engines/, one child process each, then runs every verify scenario through the
project's engine twice and through each rebuild once, one fresh process per run, and compares each run's
adapter.EXACT and adapter.GRADED arrays with the first run of the project's engine. Prints one line per run and the
largest rebuild difference per graded array; full data to --out. Under the memory cap: a build can take several GB.
"""
import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LX_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LX_WORK))
import adapter.pipeline as adapter  # noqa: E402

LOG = LX_WORK / "local" / "logs" / "engine-noise.log"
SCENARIOS = getattr(adapter, "VERIFY_SCENARIOS", adapter.SCENARIOS)


@contextlib.contextmanager
def quiet():
    """Builders and engines log straight to the file descriptors; keep them out of the summary."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(LOG, "a") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def worker(project, engine, out):
    """Child: every scenario through one engine, as verify runs it; arrays saved as <scenario>/<name>."""
    arrays = {}
    with quiet():
        pipeline = adapter.load_pipeline(project, engine=engine)
        for name in SCENARIOS:
            inputs = adapter.load_scenario(project, name)
            if hasattr(adapter, "verify_call"):
                got = adapter.verify_call(pipeline, inputs)
            else:
                got = adapter.call(pipeline, adapter.prepare(inputs))
            adapter.sync()
            arrays.update({f"{name}/{k}": v for k, v in adapter.to_arrays(got).items()})
    np.savez(out, **arrays)


def child(args):
    with open(LOG, "a") as f:
        subprocess.run([sys.executable, __file__, *args], check=True, stdout=f, stderr=subprocess.STDOUT)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(ref, got):
    rows = {}
    for name in SCENARIOS:
        row = {}
        for key in adapter.EXACT:
            r, g = ref[f"{name}/{key}"], got[f"{name}/{key}"]
            same = r.shape == g.shape
            row[key] = {"changed": int(np.count_nonzero(r != g)) if same else int(r.size), "size": int(r.size),
                        "shape_differs": not same}
        for key in adapter.GRADED:
            r, g = ref[f"{name}/{key}"].astype(np.float64), got[f"{name}/{key}"].astype(np.float64)
            diff = np.abs(r - g) if r.shape == g.shape else np.array([np.inf])
            row[key] = {"max": float(diff.max()) if diff.size else 0.0, "changed": int(np.count_nonzero(diff)),
                        "size": int(r.size)}
        rows[name] = row
    return rows


def cell(key, v):
    if "max" in v:
        return f"{key} max {v['max']:.3g} changed {v['changed']}/{v['size']}"
    if v["shape_differs"]:
        return f"{key} shape differs"
    return f"{key} changed {v['changed']}/{v['size']}"


def line(rows):
    return "; ".join(f"{name}: " + ", ".join(cell(k, v) for k, v in row.items()) for name, row in rows.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project")
    ap.add_argument("--builds", type=int, default=3)
    ap.add_argument("--out", default=str(LX_WORK / "local" / "logs" / "engine-noise.json"))
    ap.add_argument("--build", help=argparse.SUPPRESS)
    ap.add_argument("--worker", nargs=2, metavar=("ENGINE", "NPZ"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    project = adapter.project_root(a.project)
    if a.build:
        with quiet():
            return adapter.build_engine(project, a.build)
    if a.worker:
        return worker(project, *a.worker)

    LOG.write_text("")
    engines = LX_WORK / "local" / "engines"
    engines.mkdir(parents=True, exist_ok=True)
    own = Path(adapter.ENGINE)
    if not own.is_absolute():
        own = project / own
    runs, builds = [("project", own), ("project-again", own)], []
    for i in range(a.builds):
        path = engines / f"rebuild-{i}{own.suffix}"
        t0 = time.perf_counter()
        child(["--build", str(path), "--project", str(project)])
        builds.append({"engine": str(path), "build_s": round(time.perf_counter() - t0, 1)})
        runs.append((f"rebuild-{i}", path))
    results, ref, own_sha = [], None, sha256(own)
    for label, path in runs:
        npz = engines / f"{label}.npz"
        child(["--worker", str(path), str(npz), "--project", str(project)])
        got = dict(np.load(npz))
        ref = ref or got
        rows = compare(ref, got)
        same = sha256(path) == own_sha
        results.append({"run": label, "engine": str(path), "same_bytes_as_project": same, "scenarios": rows})
        print(f"{label:14s} {'same bytes' if same else 'other bytes':11s} {line(rows)}")
    rebuilds = [r for r in results if r["run"].startswith("rebuild")]
    bound = {key: max((s[key]["max"] for r in rebuilds for s in r["scenarios"].values()), default=None)
             for key in adapter.GRADED}
    flips = sum(s[key]["changed"] for r in rebuilds for s in r["scenarios"].values() for key in adapter.EXACT)
    Path(a.out).write_text(json.dumps({"project_engine": str(own), "project_sha256": own_sha, "builds": builds,
                                       "runs": results, "rebuild_max": bound, "versions": adapter.versions()},
                                      indent=1))
    print(f"largest rebuild difference over {len(rebuilds)} rebuilds and {len(SCENARIOS)} scenarios: "
          + ", ".join(f"{k} {v:.6g}" for k, v in bound.items() if v is not None)
          + f"; EXACT changes {flips} (build times {', '.join(str(b['build_s']) for b in builds)} s); full result {a.out}")


if __name__ == "__main__":
    main()
