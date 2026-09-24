#!/usr/bin/env python3
"""Compare matching tensors from two NPZ captures and rank divergence."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    reference = np.load(args.reference)
    candidate = np.load(args.candidate)
    common = sorted(set(reference.files) & set(candidate.files))
    if not common:
        raise RuntimeError("NPZ files have no matching tensor names")

    failed = False
    for name in common:
        expected = reference[name]
        actual = candidate[name]
        if expected.shape != actual.shape:
            print(f"FAIL {name}: shape {expected.shape} != {actual.shape}")
            failed = True
            continue
        expected_f = expected.astype(np.float64, copy=False)
        actual_f = actual.astype(np.float64, copy=False)
        diff = np.abs(expected_f - actual_f)
        finite = np.isfinite(actual_f).all()
        close = np.allclose(expected_f, actual_f, atol=args.atol, rtol=args.rtol)
        denom = np.linalg.norm(expected_f.ravel()) * np.linalg.norm(actual_f.ravel())
        cosine = float(np.dot(expected_f.ravel(), actual_f.ravel()) / denom) if denom else float("nan")
        status = "PASS" if finite and close else "FAIL"
        print(
            f"{status} {name}: max={diff.max():.8g} mean={diff.mean():.8g} "
            f"cosine={cosine:.10g} finite={finite}"
        )
        failed |= status == "FAIL"

    missing_reference = sorted(set(candidate.files) - set(reference.files))
    missing_candidate = sorted(set(reference.files) - set(candidate.files))
    if missing_reference or missing_candidate:
        print(f"only_candidate={missing_reference}")
        print(f"only_reference={missing_candidate}")
        failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
