"""Verify the pipeline's outputs against frozen references and grade the difference.

Usage: python main.py [--project DIR] [--candidate FILE] [--freeze] [--self-test]
  --freeze     run the project's own code and write lx-work/baseline/reference/<scenario>.npz
  --self-test  change one EXACT and one GRADED value in memory and confirm verify reports each
Arrays named in adapter.EXACT must be equal; arrays in adapter.GRADED get a level from their largest
absolute difference (adapter.LEVELS) and must stay within adapter.ACCEPTED and adapter.criterion().
Exit 1 on any failure.
"""
import argparse
import contextlib
import math
import os
import sys
from pathlib import Path

import numpy as np

LX_WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LX_WORK))
import adapter.pipeline as adapter  # noqa: E402

REF_DIR = LX_WORK / "baseline" / "reference"


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


def grade(ref, got):
    """Return (ok, text) for one scenario."""
    for key in adapter.EXACT:
        if ref[key].shape != got[key].shape or not np.array_equal(ref[key], got[key]):
            return False, f"{key} differs"
    parts, worst, metrics = [], 0, {}
    for key in adapter.GRADED:
        if ref[key].shape != got[key].shape:
            return False, f"{key} shape {got[key].shape} != {ref[key].shape}"
        if not ref[key].size:
            continue
        diff = np.abs(ref[key].astype(np.float64) - got[key].astype(np.float64))
        mse = float(np.mean(diff ** 2))
        m = {"max": float(diff.max()), "mae": float(diff.mean()),
             "psnr": math.inf if mse == 0 else 10 * math.log10(adapter.PEAK ** 2 / mse),
             "changed_pct": 100 * np.count_nonzero(diff) / diff.size,
             "over1_pct": 100 * np.count_nonzero(diff > 1) / diff.size}
        metrics[key] = m
        worst = max(worst, m["max"])
        parts.append(f"{key} max {m['max']:g} mae {m['mae']:.4f} psnr {m['psnr']:.2f} "
                     f"changed {m['changed_pct']:.3f}% >1 {m['over1_pct']:.3f}%")
    level = level_of(worst)
    ok = int(level[1]) <= int(adapter.ACCEPTED[1]) and adapter.criterion(metrics)
    return ok, f"{level} " + ("; ".join(parts) if parts else "(nothing graded)")


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
        out = adapter.call(pipeline, adapter.prepare(adapter.load_scenario(project, scenario)))
        adapter.sync()
    return adapter.to_arrays(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project")
    ap.add_argument("--candidate")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite frozen references (a user decision)")
    a = ap.parse_args()
    project = adapter.project_root(a.project)
    if a.freeze and a.candidate:
        sys.exit("error: references come from the project's own code; drop --candidate")
    if a.freeze and any(REF_DIR.glob("*.npz")) and not a.force:
        # After landing, "the project's own code" may already be optimized; freezing it would hide every change.
        sys.exit("error: references were frozen in onboarding step 5; overwriting them needs the user (--force, recorded in DECISIONS.md)")
    with quiet():
        pipeline = adapter.load_pipeline(project, a.candidate)
    if a.freeze:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        for scenario in adapter.SCENARIOS:
            out = run(pipeline, project, scenario)
            np.savez_compressed(REF_DIR / f"{scenario}.npz", **out)
            print(f"froze {scenario}: " + ", ".join(f"{k} {v.shape}" for k, v in out.items()))
        return
    failures = tried = 0
    for scenario in adapter.SCENARIOS:
        ref = dict(np.load(REF_DIR / f"{scenario}.npz"))
        got = run(pipeline, project, scenario)
        ok, text = grade(ref, got)
        failures += not ok
        print(f"{scenario:14s} {'pass' if ok else 'FAIL'} {text}")
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
    print("verify: " + ("pass" if not failures else f"{failures} failure(s)"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
