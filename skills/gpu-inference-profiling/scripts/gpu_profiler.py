#!/usr/bin/env python3
"""
================================================================================
GPU inference profiler - reference template
================================================================================

Purpose: a reference template that shows how to write a GPU inference profiler.
         Follow its structure and adapt a copy inside the project.

Features:
- GPU hardware facts (memory, utilization, temperature, power)
- inference latency (avg/p50/p95/p99)
- memory use
- detailed analysis with the PyTorch Profiler
- results exported as a Chrome trace

Adapt for each project:
1. Model loading: the project's loading logic
2. Inputs: the real model input format
3. Inference call: the project's inference interface
4. Metrics: add project-specific metrics where needed

Example (original usage, for reference only):
    python gpu_profiler.py --model-path model.pt --input-shape 1,3,224,224

In practice:
1. Copy this file into the project's tools.
2. Adapt model loading and inference.
3. Tune the profiling parameters.
4. Hook it into the project's test flow.

Key functions:
- get_gpu_info(): GPU hardware facts
- profile_model(): PyTorch Profiler analysis
- measure_latency(): latency distribution

================================================================================
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GPUInfo:
    """GPU facts"""
    name: str
    memory_total: float  # GB
    memory_used: float   # GB
    memory_free: float   # GB
    utilization: float   # %
    temperature: float   # °C
    power_draw: float    # W
    cuda_version: str
    driver_version: str


@dataclass
class ProfilingResult:
    """Profiling result"""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_qps: float
    gpu_utilization: float
    memory_allocated_gb: float
    memory_reserved_gb: float


def get_gpu_info() -> Optional[GPUInfo]:
    """Read GPU facts"""
    if not torch.cuda.is_available():
        return None

    try:
        # Use nvidia-smi for details
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,'
             'utilization.gpu,temperature.gpu,power.draw,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            parts = [p.strip() for p in parts]

            return GPUInfo(
                name=parts[0],
                memory_total=float(parts[1]) / 1024,  # MB -> GB
                memory_used=float(parts[2]) / 1024,
                memory_free=float(parts[3]) / 1024,
                utilization=float(parts[4]) if parts[4] != '[N/A]' else 0,
                temperature=float(parts[5]) if parts[5] != '[N/A]' else 0,
                power_draw=float(parts[6]) if parts[6] != '[N/A]' else 0,
                cuda_version=torch.version.cuda or 'N/A',
                driver_version=parts[7]
            )
    except Exception as e:
        print(f"Warning: Failed to get GPU info via nvidia-smi: {e}")

    # Use what PyTorch reports instead
    return GPUInfo(
        name=torch.cuda.get_device_name(0),
        memory_total=torch.cuda.get_device_properties(0).total_memory / 1024**3,
        memory_used=torch.cuda.memory_allocated(0) / 1024**3,
        memory_free=(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3,
        utilization=0,
        temperature=0,
        power_draw=0,
        cuda_version=torch.version.cuda or 'N/A',
        driver_version='N/A'
    )


def load_model(model_path: str, model_type: str, device: str) -> Tuple[Any, Any]:
    """
    Load the model

    Returns:
        (model, example_input)
    """
    if model_type == 'torchscript':
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        # The input shape must be given
        return model, None

    elif model_type == 'pytorch':
        # Assume a state_dict
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict):
            raise ValueError("Loaded a state_dict. Please provide a full model or use --model-type torchscript")
        model.eval()
        return model, None

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_example_input(input_shape: str, device: str) -> torch.Tensor:
    """Create sample inputs"""
    shape = [int(x) for x in input_shape.split(',')]
    return torch.randn(*shape, device=device)


def profile_model(
    model: Any,
    input_data: Any,
    warmup: int = 50,
    iterations: int = 200,
    device: str = 'cuda'
) -> ProfilingResult:
    """
    Profile the model
    """
    model.eval()

    # Prepare inputs
    if isinstance(input_data, dict):
        # HuggingFace-style inputs
        forward_fn = lambda: model(**input_data)
    else:
        forward_fn = lambda: model(input_data)

    # Warm up
    print(f"Warming up ({warmup} iterations)...")
    with torch.inference_mode():
        for _ in range(warmup):
            _ = forward_fn()
    torch.cuda.synchronize()

    # Reset memory statistics
    torch.cuda.reset_peak_memory_stats()

    # Collect latency samples
    print(f"Profiling ({iterations} iterations)...")
    latencies = []

    with torch.inference_mode():
        for i in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = forward_fn()
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{iterations}")

    # Compute statistics
    latencies.sort()
    n = len(latencies)

    avg_latency = sum(latencies) / n
    p50_latency = latencies[int(n * 0.50)]
    p95_latency = latencies[int(n * 0.95)]
    p99_latency = latencies[int(n * 0.99)]
    min_latency = latencies[0]
    max_latency = latencies[-1]

    # Compute throughput (batch_size=1 when unknown)
    batch_size = 1
    if isinstance(input_data, torch.Tensor):
        batch_size = input_data.shape[0]
    elif isinstance(input_data, dict) and 'input_ids' in input_data:
        batch_size = input_data['input_ids'].shape[0]

    throughput = batch_size / (avg_latency / 1000)  # QPS

    # Read GPU utilization (approximate)
    gpu_info = get_gpu_info()
    gpu_utilization = gpu_info.utilization if gpu_info else 0

    return ProfilingResult(
        avg_latency_ms=avg_latency,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
        min_latency_ms=min_latency,
        max_latency_ms=max_latency,
        throughput_qps=throughput,
        gpu_utilization=gpu_utilization,
        memory_allocated_gb=torch.cuda.memory_allocated() / 1024**3,
        memory_reserved_gb=torch.cuda.memory_reserved() / 1024**3
    )


def run_torch_profiler(
    model: Any,
    input_data: Any,
    output_dir: Path,
    iterations: int = 10
) -> str:
    """Run the PyTorch Profiler and export a trace"""
    from torch.profiler import profile, ProfilerActivity, record_function

    trace_path = output_dir / "trace.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.inference_mode():
            for _ in range(iterations):
                with record_function("inference"):
                    if isinstance(input_data, dict):
                        _ = model(**input_data)
                    else:
                        _ = model(input_data)

    # Export a Chrome trace
    prof.export_chrome_trace(str(trace_path))

    # Print the table
    print("\n" + "=" * 80)
    print("PyTorch Profiler Results (Top 20 by CUDA time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    return str(trace_path)


def generate_report(
    gpu_info: Optional[GPUInfo],
    result: ProfilingResult,
    model_path: str,
    output_path: Path
) -> None:
    """Build the analysis report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "gpu_info": None,
        "profiling_result": {
            "latency": {
                "avg_ms": round(result.avg_latency_ms, 3),
                "p50_ms": round(result.p50_latency_ms, 3),
                "p95_ms": round(result.p95_latency_ms, 3),
                "p99_ms": round(result.p99_latency_ms, 3),
                "min_ms": round(result.min_latency_ms, 3),
                "max_ms": round(result.max_latency_ms, 3),
            },
            "throughput_qps": round(result.throughput_qps, 2),
            "memory": {
                "allocated_gb": round(result.memory_allocated_gb, 3),
                "reserved_gb": round(result.memory_reserved_gb, 3),
            }
        },
        "analysis": {
            "bottleneck": "",
            "recommendations": []
        }
    }

    if gpu_info:
        report["environment"]["cuda_version"] = gpu_info.cuda_version
        report["environment"]["driver_version"] = gpu_info.driver_version
        report["gpu_info"] = {
            "name": gpu_info.name,
            "memory_total_gb": round(gpu_info.memory_total, 2),
            "memory_used_gb": round(gpu_info.memory_used, 2),
            "utilization_percent": gpu_info.utilization,
            "temperature_c": gpu_info.temperature,
            "power_draw_w": gpu_info.power_draw,
        }

    # Classify the bottleneck
    recommendations = []

    if gpu_info and gpu_info.utilization < 30:
        report["analysis"]["bottleneck"] = "IO/CPU bound"
        recommendations.extend([
            "check whether data loading is the bottleneck",
            "use asynchronous transfers (pin_memory, non_blocking) where the project allows them",
        ])
    elif gpu_info and gpu_info.utilization < 60:
        report["analysis"]["bottleneck"] = "Partially GPU bound"
        recommendations.extend([
            "optimize the data pipeline",
            "look for frequent CPU-GPU synchronization",
            "if an eager hotspot remains material, list torch.compile as a candidate that needs explicit user approval",
        ])
    else:
        report["analysis"]["bottleneck"] = "Compute bound"
        recommendations.extend([
            "consider reduced precision (FP16/INT8), with the user's approval",
            "try a TensorRT conversion",
            "check for more efficient operators",
        ])

    # Memory advice
    if gpu_info and result.memory_allocated_gb > gpu_info.memory_total * 0.8:
        recommendations.append("memory use is high: capture a peak snapshot and inspect live temporaries")

    # Latency advice
    if result.p99_latency_ms > result.avg_latency_ms * 2:
        recommendations.append("P99 latency is high: look for GC or other jitter")

    report["analysis"]["recommendations"] = recommendations

    # Write the file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {output_path}")


def print_summary(gpu_info: Optional[GPUInfo], result: ProfilingResult) -> None:
    """Print the result summary"""
    print("\n" + "=" * 60)
    print("GPU Inference Profiling Summary")
    print("=" * 60)

    if gpu_info:
        print(f"\n[GPU Information]")
        print(f"  Name:           {gpu_info.name}")
        print(f"  Memory:         {gpu_info.memory_used:.1f} / {gpu_info.memory_total:.1f} GB")
        print(f"  Utilization:    {gpu_info.utilization:.1f}%")
        print(f"  Temperature:    {gpu_info.temperature:.0f}°C")
        print(f"  CUDA Version:   {gpu_info.cuda_version}")

    print(f"\n[Latency]")
    print(f"  Average:        {result.avg_latency_ms:.2f} ms")
    print(f"  P50:            {result.p50_latency_ms:.2f} ms")
    print(f"  P95:            {result.p95_latency_ms:.2f} ms")
    print(f"  P99:            {result.p99_latency_ms:.2f} ms")
    print(f"  Min/Max:        {result.min_latency_ms:.2f} / {result.max_latency_ms:.2f} ms")

    print(f"\n[Throughput]")
    print(f"  QPS:            {result.throughput_qps:.1f}")

    print(f"\n[Memory]")
    print(f"  Allocated:      {result.memory_allocated_gb:.3f} GB")
    print(f"  Reserved:       {result.memory_reserved_gb:.3f} GB")

    # Quick analysis
    print(f"\n[Quick Analysis]")
    if gpu_info:
        if gpu_info.utilization < 30:
            print("  ⚠️  GPU utilization is low (<30%). Likely IO/CPU bound.")
            print("     → Inspect data loading, transfers, and synchronization.")
        elif gpu_info.utilization < 60:
            print("  ℹ️  GPU utilization is moderate (30-60%). Room for optimization.")
            print("     → Inspect dataflow first; torch.compile requires explicit approval.")
        else:
            print("  ✓  GPU utilization is good (>60%). Compute bound.")
            print("     → Consider quantization or TensorRT for further optimization.")


def main():
    parser = argparse.ArgumentParser(
        description="GPU inference profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-path model.pt --input-shape 1,3,224,224
  %(prog)s --model-path model.pt --export-trace --export-report
        """
    )

    parser.add_argument('--model-path', required=True, help='model path')
    parser.add_argument('--model-type', choices=['torchscript', 'pytorch'],
                        default='torchscript', help='model type (default: torchscript)')
    parser.add_argument('--input-shape', help='input shape, comma-separated (for example 1,3,224,224)')
    parser.add_argument('--warmup', type=int, default=50, help='warmup iterations (default: 50)')
    parser.add_argument('--iterations', type=int, default=200, help='measured iterations (default: 200)')
    parser.add_argument('--device', default='cuda', help='device (default: cuda)')
    parser.add_argument('--export-trace', action='store_true', help='export a PyTorch Profiler trace')
    parser.add_argument('--export-report', action='store_true', help='export a JSON report')
    parser.add_argument('--output-dir', default='./profiling_output', help='output directory')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Error: CUDA is not available. Please check your GPU setup.")
        sys.exit(1)

    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read GPU facts
    print("Collecting GPU information...")
    gpu_info = get_gpu_info()

    # Load the model
    print(f"Loading model from: {args.model_path}")
    try:
        model, example_input = load_model(args.model_path, args.model_type, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Create inputs
    if example_input is None:
        if not args.input_shape:
            print("Error: --input-shape is required for this model type")
            sys.exit(1)
        example_input = create_example_input(args.input_shape, args.device)

    # Run profiling
    result = profile_model(
        model, example_input,
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device
    )

    # Print the summary
    print_summary(gpu_info, result)

    # Export the trace
    if args.export_trace:
        print("\nRunning PyTorch Profiler...")
        trace_path = run_torch_profiler(model, example_input, output_dir)
        print(f"Trace exported to: {trace_path}")

    # Export the report
    if args.export_report:
        report_path = output_dir / "profiling_report.json"
        generate_report(gpu_info, result, args.model_path, report_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
