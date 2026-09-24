import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np
import torch
from kornia.morphology import erosion as kornia_erosion
OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))

from large_kernel_eroder import LargeKernelEroder


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _ellipse_kernel(kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.float32)
    return torch.from_numpy(kernel_np).to(device=device, dtype=dtype)


def _build_mask(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, height, width), dtype=dtype, device=device)
    pad_y = max(8, height // 8)
    pad_x = max(8, width // 8)
    mask[:, pad_y:height - pad_y, pad_x:width - pad_x] = 1.0
    return mask


def _run_direct(mask: torch.Tensor, kernel_size: int, engine: str) -> torch.Tensor:
    kernel = _ellipse_kernel(kernel_size, mask.device, mask.dtype)
    return kornia_erosion(mask.unsqueeze(0), kernel, engine=engine).squeeze(0)


def _run_two_pass(mask: torch.Tensor, kernel_size: int, mode: str, engine: str) -> tuple[torch.Tensor, int]:
    if mode == "half":
        inner = _odd((kernel_size + 1) // 2)
    elif mode == "sqrt":
        inner = _odd(int(round(math.sqrt(kernel_size))))
    else:
        raise ValueError(f"Unsupported two-pass mode: {mode}")

    out = _run_direct(mask, inner, engine=engine)
    out = _run_direct(out, inner, engine=engine)
    return out, inner


@dataclass
class CaseResult:
    name: str
    status: str
    engine: str
    kernel_size: int
    inner_kernel: int | None
    max_kernel_per_pass: int | None
    height: int
    width: int
    dtype: str
    elapsed_ms: float | None
    peak_alloc_mib: float | None
    peak_reserved_mib: float | None
    sum_value: float | None
    diff_ratio_vs_ref: float | None
    error: str | None


def _measure_case(
    name: str,
    fn,
    mask: torch.Tensor,
    reference: torch.Tensor | None,
    engine: str,
    kernel_size: int,
    inner_kernel: int | None,
    max_kernel_per_pass: int | None,
) -> CaseResult:
    device = mask.device
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    try:
        start = perf_counter()
        output = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (perf_counter() - start) * 1000.0
        peak_alloc_mib = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None
        peak_reserved_mib = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)) if device.type == "cuda" else None
        diff_ratio = None
        if reference is not None:
            diff_ratio = float((output != reference).float().mean().item())
        return CaseResult(
            name=name,
            status="ok",
            engine=engine,
            kernel_size=kernel_size,
            inner_kernel=inner_kernel,
            max_kernel_per_pass=max_kernel_per_pass,
            height=mask.shape[-2],
            width=mask.shape[-1],
            dtype=str(mask.dtype).replace("torch.", ""),
            elapsed_ms=elapsed_ms,
            peak_alloc_mib=peak_alloc_mib,
            peak_reserved_mib=peak_reserved_mib,
            sum_value=float(output.sum().item()),
            diff_ratio_vs_ref=diff_ratio,
            error=None,
        )
    except RuntimeError as exc:
        message = str(exc)
        status = "oom" if "out of memory" in message.lower() else "error"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return CaseResult(
            name=name,
            status=status,
            engine=engine,
            kernel_size=kernel_size,
            inner_kernel=inner_kernel,
            max_kernel_per_pass=max_kernel_per_pass,
            height=mask.shape[-2],
            width=mask.shape[-1],
            dtype=str(mask.dtype).replace("torch.", ""),
            elapsed_ms=None,
            peak_alloc_mib=None,
            peak_reserved_mib=None,
            sum_value=None,
            diff_ratio_vs_ref=None,
            error=message.splitlines()[0],
        )


def _single_case(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    mask = _build_mask(args.height, args.width, device, dtype)

    reference = None
    if args.reference_engine:
        reference = _run_direct(mask, args.kernel_size, args.reference_engine)

    max_kernel_per_pass = None
    if args.mode == "direct":
        inner_kernel = None
        fn = lambda: _run_direct(mask, args.kernel_size, args.engine)
    elif args.mode == "split":
        eroder = LargeKernelEroder(
            max_kernel_per_pass=args.max_kernel_per_pass,
            ellipse_engine=args.engine,
        )
        split_kernels = eroder.split_odd_kernel_by_radius(args.kernel_size, args.max_kernel_per_pass)
        inner_kernel = split_kernels[0] if len(split_kernels) == 1 else None
        max_kernel_per_pass = args.max_kernel_per_pass
        fn = lambda: eroder.erode(
            mask,
            args.kernel_size,
            device=device,
            kernel_shape=args.kernel_shape,
            max_kernel_per_pass=args.max_kernel_per_pass,
        )
    else:
        inner_kernel = _odd((args.kernel_size + 1) // 2) if args.mode == "half" else _odd(int(round(math.sqrt(args.kernel_size))))
        fn = lambda: _run_two_pass(mask, args.kernel_size, args.mode, args.engine)[0]

    result = _measure_case(
        name=args.name,
        fn=fn,
        mask=mask,
        reference=reference,
        engine=args.engine,
        kernel_size=args.kernel_size,
        inner_kernel=inner_kernel,
        max_kernel_per_pass=max_kernel_per_pass,
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0


def _spawn_case(script_path: str, case: dict, timeout_sec: float | None) -> dict:
    cmd = [
        sys.executable,
        script_path,
        "--single",
        "--name", case["name"],
        "--height", str(case["height"]),
        "--width", str(case["width"]),
        "--kernel-size", str(case["kernel_size"]),
        "--engine", case["engine"],
        "--mode", case["mode"],
        "--kernel-shape", case.get("kernel_shape", "ellipse"),
        "--dtype", case["dtype"],
        "--device", case["device"],
    ]
    if case.get("max_kernel_per_pass") is not None:
        cmd.extend(["--max-kernel-per-pass", str(case["max_kernel_per_pass"])])
    if case.get("reference_engine"):
        cmd.extend(["--reference-engine", case["reference_engine"]])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        return {
            "name": case["name"],
            "status": "timeout",
            "engine": case["engine"],
            "kernel_size": case["kernel_size"],
            "inner_kernel": None,
            "max_kernel_per_pass": case.get("max_kernel_per_pass"),
            "height": case["height"],
            "width": case["width"],
            "dtype": case["dtype"],
            "elapsed_ms": None,
            "peak_alloc_mib": None,
            "peak_reserved_mib": None,
            "sum_value": None,
            "diff_ratio_vs_ref": None,
            "error": f"timeout after {timeout_sec}s",
            "returncode": None,
        }
    payload = None
    if proc.stdout.strip():
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if payload is None:
        payload = {
            "name": case["name"],
            "status": "subprocess_error",
            "engine": case["engine"],
            "kernel_size": case["kernel_size"],
            "inner_kernel": None,
            "max_kernel_per_pass": case.get("max_kernel_per_pass"),
            "height": case["height"],
            "width": case["width"],
            "dtype": case["dtype"],
            "elapsed_ms": None,
            "peak_alloc_mib": None,
            "peak_reserved_mib": None,
            "sum_value": None,
            "diff_ratio_vs_ref": None,
            "error": (proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}")[:400],
        }
    payload["returncode"] = proc.returncode
    return payload


def _default_cases(device: str, dtype: str) -> list[dict]:
    cases: list[dict] = []

    quality_cases = [
        (256, 256, 61),
        (256, 256, 121),
    ]
    for height, width, kernel_size in quality_cases:
        cases.append({
            "name": f"quality-direct-conv-{height}x{width}-k{kernel_size}",
            "height": height,
            "width": width,
            "kernel_size": kernel_size,
            "engine": "convolution",
            "mode": "direct",
            "dtype": dtype,
            "device": device,
        })
        cases.append({
            "name": f"quality-half-conv-{height}x{width}-k{kernel_size}",
            "height": height,
            "width": width,
            "kernel_size": kernel_size,
            "engine": "convolution",
            "mode": "half",
            "dtype": dtype,
            "device": device,
            "reference_engine": "convolution",
        })
        cases.append({
            "name": f"quality-sqrt-conv-{height}x{width}-k{kernel_size}",
            "height": height,
            "width": width,
            "kernel_size": kernel_size,
            "engine": "convolution",
            "mode": "sqrt",
            "dtype": dtype,
            "device": device,
            "reference_engine": "convolution",
        })

    perf_cases = [
        (512, 512, 121),
        (1024, 1024, 121),
        (1024, 1024, 241),
        (2048, 2048, 61),
        (2048, 2048, 121),
    ]
    for height, width, kernel_size in perf_cases:
        cases.append({
            "name": f"perf-unfold-{height}x{width}-k{kernel_size}",
            "height": height,
            "width": width,
            "kernel_size": kernel_size,
            "engine": "unfold",
            "mode": "direct",
            "dtype": dtype,
            "device": device,
        })
        cases.append({
            "name": f"perf-conv-{height}x{width}-k{kernel_size}",
            "height": height,
            "width": width,
            "kernel_size": kernel_size,
            "engine": "convolution",
            "mode": "direct",
            "dtype": dtype,
            "device": device,
        })
        cases.append({
            "name": f"perf-half-conv-{height}x{width}-k{kernel_size}",
            "height": height,
            "width": width,
            "kernel_size": kernel_size,
            "engine": "convolution",
            "mode": "half",
            "dtype": dtype,
            "device": device,
        })
        cases.append({
            "name": f"perf-sqrt-conv-{height}x{width}-k{kernel_size}",
            "height": height,
            "width": width,
            "kernel_size": kernel_size,
            "engine": "convolution",
            "mode": "sqrt",
            "dtype": dtype,
            "device": device,
        })

    return cases


def _pass_cap_cases(device: str, dtype: str) -> list[dict]:
    cases: list[dict] = []
    caps = [33, 49, 65, 81, 97]
    workloads = [
        (256, 256, 121),
        (512, 512, 121),
    ]
    for height, width, kernel_size in workloads:
        for cap in caps:
            cases.append({
                "name": f"split-cap{cap}-{height}x{width}-k{kernel_size}",
                "height": height,
                "width": width,
                "kernel_size": kernel_size,
                "kernel_shape": "ellipse",
                "engine": "convolution",
                "mode": "split",
                "max_kernel_per_pass": cap,
                "dtype": dtype,
                "device": device,
                "reference_engine": "convolution" if height == 256 else None,
            })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark large-kernel Kornia erosion on GPU.")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--kernel-size", type=int, default=121)
    parser.add_argument("--kernel-shape", choices=["rect", "ellipse"], default="ellipse")
    parser.add_argument("--engine", choices=["unfold", "convolution"], default="convolution")
    parser.add_argument("--mode", choices=["direct", "half", "sqrt", "split"], default="direct")
    parser.add_argument("--reference-engine", choices=["unfold", "convolution"], default=None)
    parser.add_argument("--max-kernel-per-pass", type=int, default=65)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--suite", choices=["default", "pass-cap"], default="default")
    args = parser.parse_args()

    if args.single:
        return _single_case(args)

    script_path = os.path.abspath(__file__)
    summary = []
    if args.suite == "default":
        cases = _default_cases(device=args.device, dtype=args.dtype)
    else:
        cases = _pass_cap_cases(device=args.device, dtype=args.dtype)

    for case in cases:
        result = _spawn_case(script_path, case, timeout_sec=args.timeout_sec)
        summary.append(result)
        print(json.dumps(result, ensure_ascii=False))

    print()
    print("=== Summary ===")
    for item in summary:
        extra = []
        if item.get("inner_kernel") is not None:
            extra.append(f"inner={item['inner_kernel']}")
        if item.get("max_kernel_per_pass") is not None:
            extra.append(f"cap={item['max_kernel_per_pass']}")
        if item.get("elapsed_ms") is not None:
            extra.append(f"time={item['elapsed_ms']:.2f}ms")
        if item.get("peak_alloc_mib") is not None:
            extra.append(f"alloc={item['peak_alloc_mib']:.1f}MiB")
        if item.get("diff_ratio_vs_ref") is not None:
            extra.append(f"diff={item['diff_ratio_vs_ref']:.6f}")
        if item.get("error"):
            extra.append(f"error={item['error']}")
        print(f"{item['name']}: {item['status']} {' '.join(extra)}".rstrip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
