#!/usr/bin/env python3
"""Interleaved shape x mode memory stress test.

See ../references/mixed-workload-validation.md. Import `run_matrix` and pass a
callable that performs one real end-to-end request; this module owns only the
interleaving, the measurement discipline, and the pass/fail rules.

Deliberately does NOT insert gc/empty_cache between requests: the point is to
observe the residue the production request path actually leaves behind.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Sequence

import torch

MIB = 1024**2


def run_matrix(
    run_request: Callable[[str, Any], Any],
    modes: Sequence[str],
    shapes: dict[str, Any],
    *,
    seed: int = 0,
    warmup_shape: str | None = None,
    validate: Callable[[Any, Any], str | None] | None = None,
) -> dict:
    """Interleave every (mode, shape) pair in one process and measure each request.

    run_request(mode, payload) -> output. Must be a complete service request.
    validate(output, payload) -> None when acceptable, else a failure string.
    Returns a summary dict; prints a per-request table.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory validation")

    warm_key = warmup_shape or next(iter(shapes))
    for mode in modes:
        run_request(mode, shapes[warm_key])
    torch.cuda.synchronize()

    plan = [(m, s) for m in modes for s in shapes]
    random.Random(seed).shuffle(plan)

    print(f"{len(shapes)} shapes x {len(modes)} modes = {len(plan)} requests, seed={seed}")
    print(f"{'#':>3} {'mode':<20}{'shape':<14}{'ms':>9}{'peak_a':>10}{'peak_r':>10}"
          f"{'end_a':>9}{'end_r':>9}  status")
    print("-" * 96)

    rows, failures = [], []
    for index, (mode, shape_key) in enumerate(plan, 1):
        payload = shapes[shape_key]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        status = "ok"
        try:
            output = run_request(mode, payload)
            if validate is not None:
                problem = validate(output, payload)
                if problem:
                    status = problem
                    failures.append((index, mode, shape_key, problem))
            del output
        except torch.cuda.OutOfMemoryError:
            status = "OOM"
            failures.append((index, mode, shape_key, status))
        except Exception as exc:  # noqa: BLE001 - a failed request is a result
            status = f"{type(exc).__name__}"
            failures.append((index, mode, shape_key, f"{status}: {exc}"))
        torch.cuda.synchronize()

        row = {
            "index": index,
            "mode": mode,
            "shape": shape_key,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "end_allocated": torch.cuda.memory_allocated(),
            "end_reserved": torch.cuda.memory_reserved(),
            "status": status,
        }
        rows.append(row)
        print(f"{index:>3} {mode:<20}{shape_key:<14}{row['latency_ms']:>9.0f}"
              f"{row['peak_allocated']/MIB:>9.0f}M{row['peak_reserved']/MIB:>9.0f}M"
              f"{row['end_allocated']/MIB:>8.0f}M{row['end_reserved']/MIB:>8.0f}M  {status}")

    alloc_drift = rows[-1]["end_allocated"] - rows[0]["end_allocated"]
    res_drift = rows[-1]["end_reserved"] - rows[0]["end_reserved"]
    peak_reserved = max(r["peak_reserved"] for r in rows)

    print("-" * 96)
    print(f"peak reserved (worst request) : {peak_reserved/MIB:.0f} MiB "
          f"({max(rows, key=lambda r: r['peak_reserved'])['mode']} @ "
          f"{max(rows, key=lambda r: r['peak_reserved'])['shape']})")
    print(f"end allocated drift : {alloc_drift/MIB:+.0f} MiB   <- upward means retained tensors")
    print(f"end reserved  drift : {res_drift/MIB:+.0f} MiB   <- upward with flat allocated means fragmentation")

    # Residual per mode: the floor a request starts from depends on the previous mode.
    by_mode: dict[str, int] = {}
    for row in rows:
        by_mode[row["mode"]] = max(by_mode.get(row["mode"], 0), row["end_allocated"])
    floor = min(by_mode.values())
    residual = {m: v - floor for m, v in by_mode.items() if v > floor}
    if residual:
        print("per-mode resident residual (raises the floor for whichever request follows):")
        for mode, extra in sorted(residual.items(), key=lambda kv: -kv[1]):
            print(f"  {mode:<24}{extra/MIB:>8.0f} MiB")

    print(f"failures: {failures if failures else 'none'}")
    return {
        "rows": rows,
        "peak_reserved": peak_reserved,
        "allocated_drift": alloc_drift,
        "reserved_drift": res_drift,
        "per_mode_residual": residual,
        "failures": failures,
    }
