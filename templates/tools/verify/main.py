"""Verify the pipeline's outputs against frozen references and grade the difference.

Usage: python main.py [--project DIR] [--candidate FILE | --saved DIR] [--against DIR] [--save DIR] [--self-test] [--out JSON]
       python main.py --freeze [--force] [--reference DIR]
  --freeze     run the project's own code and write <reference>/<scenario>.npz
  --reference  directory of the frozen references, default lx-work/baseline/reference; a scratch directory lets a
               trial run (onboarding step 4) freeze and verify without touching the baseline
  --against    compare with a run saved by --save instead of the references: an A/A, an engine rebuild, or the
               natural differences of a change the user is to judge
  --save       also write this run's arrays and their source (commit, candidate) to DIR/<scenario>.npz
  --saved      grade a run saved by --save instead of running a pipeline
  --self-test  change one EXACT and one GRADED value in memory and confirm verify reports each
  --out        every scenario's metrics as JSON, default lx-work/local/logs/verify-last.json
Arrays named in adapter.EXACT must be equal, after the optional adapter.tolerate(ref, got) has set back the
differences the accuracy standard allows (returning the arrays and a note); arrays in adapter.GRADED get a level
from their largest difference in adapter.LEVEL_METRIC ("max" absolute difference, the default, or "rel_l2" =
||got - ref|| / ||ref||) against adapter.LEVELS and must stay within adapter.ACCEPTED and adapter.criterion().
Scenarios: adapter.VERIFY_SCENARIOS when defined (verify may cover more inputs than the bench times), else
adapter.SCENARIOS. A scenario runs through the optional adapter.verify_call(pipeline, inputs), for example one call
per input of the scenario, else one adapter.call. Exit 1 on any failure.
"""
import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LX_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LX_WORK))
import adapter.pipeline as adapter  # noqa: E402

REF_DIR = LX_WORK / "baseline" / "reference"
SCENARIOS = getattr(adapter, "VERIFY_SCENARIOS", adapter.SCENARIOS)
LEVEL_METRIC = getattr(adapter, "LEVEL_METRIC", "max")


@contextlib.contextmanager
def quiet():
    """Native libraries log straight to the file descriptors; keep them out of the summary."""
    log = LX_WORK / "local" / "logs" / "verify.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log, "a") as f:
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


def level_of(worst):
    return next((name for limit, name in adapter.LEVELS if worst <= limit), "L3")


def metrics_of(ref, got):
    r = ref.astype(np.float64)
    diff = np.abs(r - got.astype(np.float64))
    mse = float(np.mean(diff ** 2))
    ref_norm, diff_norm = float(np.linalg.norm(r)), float(np.linalg.norm(diff))
    return {"max": float(diff.max()), "mae": float(diff.mean()),
            "psnr": math.inf if mse == 0 else 10 * math.log10(adapter.PEAK ** 2 / mse),
            "rel_l2": diff_norm / ref_norm if ref_norm else (0.0 if diff_norm == 0 else math.inf),
            "changed_pct": 100 * np.count_nonzero(diff) / diff.size,
            "over1_pct": 100 * np.count_nonzero(diff > 1) / diff.size}


def grade(ref, got):
    """Return (ok, text, metrics) for one scenario."""
    got, note = adapter.tolerate(ref, got) if hasattr(adapter, "tolerate") else (got, "")
    for key in adapter.EXACT:
        if ref[key].shape != got[key].shape:
            return False, f"{key} shape {got[key].shape} != {ref[key].shape}", {}
        if not np.array_equal(ref[key], got[key]):
            return False, f"{key} differs at {np.count_nonzero(ref[key] != got[key])} of {ref[key].size}", {}
    parts, worst, metrics = [], 0, {}
    for key in adapter.GRADED:
        if ref[key].shape != got[key].shape:
            return False, f"{key} shape {got[key].shape} != {ref[key].shape}", {}
        if not ref[key].size:
            continue
        m = metrics[key] = metrics_of(ref[key], got[key])
        worst = max(worst, m[LEVEL_METRIC])
        parts.append(f"{key} max {m['max']:.3g} rel_l2 {m['rel_l2']:.1e} mae {m['mae']:.4g} psnr {m['psnr']:.2f} "
                     f"changed {m['changed_pct']:.3f}% >1 {m['over1_pct']:.3f}%")
    level = level_of(worst)
    ok = int(level[1]) <= int(adapter.ACCEPTED[1]) and adapter.criterion(metrics)
    text = f"{level} " + ("; ".join(parts) if parts else "(nothing graded)") + (f"; {note}" if note else "")
    return ok, text, metrics


def self_test(ref, got):
    """Change one value of the first non-empty EXACT array and of the first non-empty GRADED array, in memory,
    and check that grade() fails each. Returns (arrays tried, misses, text)."""
    results = []
    for names in (adapter.EXACT, adapter.GRADED):
        key = next((k for k in names if got[k].size), None)
        if key is None:
            continue
        bad = {**got, key: got[key].copy()}
        flat = bad[key].reshape(-1)
        if flat.dtype == bool:
            flat[0] = not flat[0]
        else:
            flat[0] = flat[0] + 100 if flat[0] < adapter.PEAK - 100 else flat[0] - 100
        results.append((key, not grade(ref, bad)[0]))
    text = ", ".join(f"{key} {'caught' if caught else 'MISSED'}" for key, caught in results)
    return len(results), sum(not caught for _, caught in results), text or "nothing to change, every array is empty"


def run(pipeline, project, scenario):
    with quiet():
        inputs = adapter.load_scenario(project, scenario)
        if hasattr(adapter, "verify_call"):
            out = adapter.verify_call(pipeline, inputs)
        else:
            out = adapter.call(pipeline, adapter.prepare(inputs))
        adapter.sync()
    return adapter.to_arrays(out)


def source(project, candidate):
    """Where a run came from, saved next to its arrays so a later --against names it."""
    git = ["git", "-C", str(project)]
    commit = subprocess.run(git + ["rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return {"source": f"candidate {candidate}" if candidate else "project",
            "commit": commit + (" with local changes" if dirty else ""),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": adapter.versions()}


def save(path, arrays, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load(path):
    if not path.is_file():
        sys.exit(f"error: no {path}; freeze the references (onboarding step 5), or name a saved run")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays.pop("meta"))) if "meta" in arrays else {}
    return arrays, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project")
    ap.add_argument("--candidate")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite frozen references (a user decision)")
    ap.add_argument("--reference", default=str(REF_DIR), help="directory of the frozen references")
    ap.add_argument("--against", help="directory of a run saved by --save, instead of the references")
    ap.add_argument("--save", help="also write this run to this directory")
    ap.add_argument("--saved", help="grade a run saved by --save instead of running a pipeline")
    ap.add_argument("--out", default=str(LX_WORK / "local" / "logs" / "verify-last.json"))
    a = ap.parse_args()
    project = adapter.project_root(a.project)
    ref_dir = Path(a.against or a.reference)
    if a.freeze and (a.candidate or a.against or a.saved):
        sys.exit("error: references come from the project's own code; --freeze takes no candidate or saved run")
    if a.saved and (a.candidate or a.save):
        sys.exit("error: --saved grades a saved run; it takes no candidate or --save")
    if a.freeze and any(ref_dir.glob("*.npz")) and not a.force:
        # After landing, "the project's own code" may already be optimized; freezing it would hide every change.
        sys.exit(f"error: references were frozen in {ref_dir}; overwriting them needs the user (--force, recorded in DECISIONS.md)")
    if not a.saved:
        with quiet():
            pipeline = adapter.load_pipeline(project, a.candidate)
        meta = source(project, a.candidate)
    if a.freeze:
        for scenario in SCENARIOS:
            out = run(pipeline, project, scenario)
            save(ref_dir / f"{scenario}.npz", out, meta)
            print(f"froze {scenario}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))
        print(f"references: {len(SCENARIOS)} scenarios in {ref_dir}")
        return
    failures = tried = 0
    report = {"against": str(ref_dir), "scenarios": {}}
    for scenario in SCENARIOS:
        if a.saved:
            got, meta = load(Path(a.saved) / f"{scenario}.npz")
            meta = {**meta, "source": f"saved run {a.saved} ({meta.get('source', 'unknown source')})"}
        else:
            got = run(pipeline, project, scenario)
            if a.save:
                save(Path(a.save) / f"{scenario}.npz", got, meta)
        ref, ref_meta = load(ref_dir / f"{scenario}.npz")
        ok, text, metrics = grade(ref, got)
        failures += not ok
        print(f"{scenario:14s} {'pass' if ok else 'FAIL'} {text}")
        report["scenarios"][scenario] = {"ok": ok, "text": text, "metrics": metrics, "run": meta, "against_run": ref_meta}
        if a.self_test and not ok:
            print("  self-test: skipped; it proves nothing on a scenario that already fails")
        elif a.self_test:
            n, misses, text = self_test(ref, got)
            tried, failures = tried + n, failures + misses
            print(f"  self-test: {text}")
    if a.self_test and not tried and not failures:
        # Scenarios with empty outputs are expected, but in a passing run one array somewhere must prove the check.
        print("self-test: no value was changed in any scenario, so nothing shows that verify can fail")
        failures += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("verify: " + ("pass" if not failures else f"{failures} failure(s)")
          + f" ({len(SCENARIOS)} scenarios against {ref_dir}; details {a.out})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
