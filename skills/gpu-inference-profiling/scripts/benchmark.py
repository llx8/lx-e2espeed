#!/usr/bin/env python3
"""
================================================================================
GPU inference benchmark - reference template
================================================================================

Purpose: a reference template that shows how to write a standard GPU inference
         benchmark. Follow its structure and adapt a copy inside the project.

Features:
- latency across batch sizes
- performance across precisions (FP32/FP16/BF16)
- latency distribution (avg/p50/p95/p99/min/max)
- memory use and throughput
- results exported as CSV/JSON

Adapt for each project:
1. load_model(): the project's model loading
2. input_shape: the real model input
3. preprocessing, if the model needs it
4. batch_sizes: the batch range that matters in production
5. warmup_runs / test_runs: warmup and measured repetitions

Example (original usage, for reference only):
    python benchmark.py --model-path model.pt --input-shape 3,224,224 --batch-sizes 1,2,4,8

In practice:
1. Copy this file into the project's tools.
2. Replace load_model() with the project's loading logic.
3. Generate inputs that match the real input format.
4. Add preprocessing/postprocessing timing where needed.

================================================================================
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Result of one benchmark run"""
    batch_size: int
    precision: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_qps: float
    memory_allocated_mb: float
    memory_reserved_mb: float


def load_model(model_path: str, device: str = 'cuda') -> nn.Module:
    """Load the model"""
    if model_path.endswith('.pt') or model_path.endswith('.pth'):
        model = torch.jit.load(model_path, map_location=device)
    else:
        model = torch.load(model_path, map_location=device)

    model.eval()
    return model


def convert_precision(model: nn.Module, precision: str) -> nn.Module:
    """Convert the model precision"""
    if precision == 'fp32':
        return model.float()
    elif precision == 'fp16':
        return model.half()
    elif precision == 'bf16':
        return model.to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown precision: {precision}")


def create_input(
    input_shape: List[int],
    batch_size: int,
    precision: str,
    device: str
) -> torch.Tensor:
    """Create test inputs"""
    shape = [batch_size] + input_shape

    dtype = torch.float32
    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16

    return torch.randn(*shape, dtype=dtype, device=device)


def benchmark_single(
    model: nn.Module,
    input_tensor: torch.Tensor,
    batch_size: int,
    precision: str,
    warmup: int = 50,
    iterations: int = 200
) -> BenchmarkResult:
    """Run one benchmark"""

    # Warm up
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Reset memory statistics
    torch.cuda.reset_peak_memory_stats()

    # Measure
    latencies = []
    with torch.inference_mode():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(input_tensor)
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

    # Compute statistics
    latencies.sort()
    n = len(latencies)

    return BenchmarkResult(
        batch_size=batch_size,
        precision=precision,
        avg_latency_ms=sum(latencies) / n,
        p50_latency_ms=latencies[int(n * 0.50)],
        p95_latency_ms=latencies[int(n * 0.95)],
        p99_latency_ms=latencies[int(n * 0.99)],
        min_latency_ms=latencies[0],
        max_latency_ms=latencies[-1],
        throughput_qps=batch_size / (sum(latencies) / n / 1000),
        memory_allocated_mb=torch.cuda.memory_allocated() / 1024**2,
        memory_reserved_mb=torch.cuda.memory_reserved() / 1024**2
    )


def run_benchmark(
    model_path: str,
    input_shape: List[int],
    batch_sizes: List[int],
    precisions: List[str],
    warmup: int,
    iterations: int,
    device: str
) -> List[BenchmarkResult]:
    """Run the full benchmark"""

    results = []

    for precision in precisions:
        print(f"\n{'='*60}")
        print(f"Testing precision: {precision}")
        print('='*60)

        # Load and convert the model
        try:
            model = load_model(model_path, device)
            model = convert_precision(model, precision)
        except Exception as e:
            print(f"  Error loading model with {precision}: {e}")
            continue

        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")

            try:
                # Free GPU memory
                torch.cuda.empty_cache()

                # Create inputs
                input_tensor = create_input(input_shape, batch_size, precision, device)

                # Run the benchmark
                result = benchmark_single(
                    model, input_tensor, batch_size, precision,
                    warmup=warmup, iterations=iterations
                )

                results.append(result)

                # Print results
                print(f"    Latency: {result.avg_latency_ms:.2f}ms (p99: {result.p99_latency_ms:.2f}ms)")
                print(f"    Throughput: {result.throughput_qps:.1f} QPS")
                print(f"    Memory: {result.memory_allocated_mb:.0f} MB")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    OOM at batch size {batch_size}")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e

        # Clean up
        del model
        torch.cuda.empty_cache()

    return results


def print_summary_table(results: List[BenchmarkResult]) -> None:
    """Print the summary table"""

    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 100)
    print("Benchmark Results Summary")
    print("=" * 100)

    # Group by precision
    precisions = sorted(set(r.precision for r in results))

    for precision in precisions:
        print(f"\n[{precision.upper()}]")
        print("-" * 90)
        print(f"{'Batch':>6} | {'Avg(ms)':>8} | {'P50(ms)':>8} | {'P95(ms)':>8} | "
              f"{'P99(ms)':>8} | {'QPS':>10} | {'Memory(MB)':>10}")
        print("-" * 90)

        precision_results = [r for r in results if r.precision == precision]
        for r in sorted(precision_results, key=lambda x: x.batch_size):
            print(f"{r.batch_size:>6} | {r.avg_latency_ms:>8.2f} | {r.p50_latency_ms:>8.2f} | "
                  f"{r.p95_latency_ms:>8.2f} | {r.p99_latency_ms:>8.2f} | "
                  f"{r.throughput_qps:>10.1f} | {r.memory_allocated_mb:>10.0f}")


def find_optimal_batch_size(results: List[BenchmarkResult]) -> Dict[str, Tuple[int, float]]:
    """Find the best batch size per precision (by throughput)"""

    optimal = {}
    precisions = set(r.precision for r in results)

    for precision in precisions:
        precision_results = [r for r in results if r.precision == precision]
        if precision_results:
            best = max(precision_results, key=lambda x: x.throughput_qps)
            optimal[precision] = (best.batch_size, best.throughput_qps)

    return optimal


def export_csv(results: List[BenchmarkResult], output_path: Path) -> None:
    """Export results to CSV"""

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'batch_size', 'precision', 'avg_latency_ms', 'p50_latency_ms',
            'p95_latency_ms', 'p99_latency_ms', 'min_latency_ms', 'max_latency_ms',
            'throughput_qps', 'memory_allocated_mb', 'memory_reserved_mb'
        ])

        for r in results:
            writer.writerow([
                r.batch_size, r.precision, f"{r.avg_latency_ms:.3f}",
                f"{r.p50_latency_ms:.3f}", f"{r.p95_latency_ms:.3f}",
                f"{r.p99_latency_ms:.3f}", f"{r.min_latency_ms:.3f}",
                f"{r.max_latency_ms:.3f}", f"{r.throughput_qps:.2f}",
                f"{r.memory_allocated_mb:.1f}", f"{r.memory_reserved_mb:.1f}"
            ])

    print(f"\nResults exported to: {output_path}")


def export_json(results: List[BenchmarkResult], model_path: str, output_path: Path) -> None:
    """Export results to JSON"""

    data = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "results": [
            {
                "batch_size": r.batch_size,
                "precision": r.precision,
                "latency": {
                    "avg_ms": round(r.avg_latency_ms, 3),
                    "p50_ms": round(r.p50_latency_ms, 3),
                    "p95_ms": round(r.p95_latency_ms, 3),
                    "p99_ms": round(r.p99_latency_ms, 3),
                },
                "throughput_qps": round(r.throughput_qps, 2),
                "memory_mb": round(r.memory_allocated_mb, 1),
            }
            for r in results
        ],
        "optimal_config": find_optimal_batch_size(results)
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results exported to: {output_path}")


def compare_with_baseline(
    results: List[BenchmarkResult],
    baseline_path: Optional[str]
) -> None:
    """Compare with the baseline"""

    if not baseline_path:
        return

    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load baseline: {e}")
        return

    print("\n" + "=" * 60)
    print("Comparison with Baseline")
    print("=" * 60)

    baseline_results = {
        (r['batch_size'], r['precision']): r
        for r in baseline.get('results', [])
    }

    for r in results:
        key = (r.batch_size, r.precision)
        if key in baseline_results:
            br = baseline_results[key]
            latency_diff = (r.avg_latency_ms - br['latency']['avg_ms']) / br['latency']['avg_ms'] * 100
            qps_diff = (r.throughput_qps - br['throughput_qps']) / br['throughput_qps'] * 100

            status = "✓" if qps_diff > 0 else "✗"
            print(f"{status} Batch {r.batch_size} ({r.precision}): "
                  f"Latency {latency_diff:+.1f}%, Throughput {qps_diff:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="GPU inference benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model-path', required=True, help='model path')
    parser.add_argument('--input-shape', required=True,
                        help='input shape without batch, comma-separated (for example 3,224,224)')
    parser.add_argument('--batch-sizes', default='1,2,4,8,16,32',
                        help='batch sizes to test, comma-separated (default: 1,2,4,8,16,32)')
    parser.add_argument('--precisions', default='fp32',
                        help='precisions to test, comma-separated (default: fp32; choices: fp32,fp16,bf16)')
    parser.add_argument('--warmup', type=int, default=50, help='warmup iterations (default: 50)')
    parser.add_argument('--iterations', type=int, default=200, help='measured iterations (default: 200)')
    parser.add_argument('--device', default='cuda', help='device (default: cuda)')
    parser.add_argument('--export-csv', action='store_true', help='export results as CSV')
    parser.add_argument('--export-json', action='store_true', help='export results as JSON')
    parser.add_argument('--output-dir', default='./benchmark_output', help='output directory')
    parser.add_argument('--baseline', help='baseline JSON file to compare against')
    parser.add_argument('--full-test', action='store_true',
                        help='full test (all precisions, more batch sizes)')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        sys.exit(1)

    # Parse arguments
    input_shape = [int(x) for x in args.input_shape.split(',')]

    if args.full_test:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        precisions = ['fp32', 'fp16', 'bf16']
    else:
        batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
        precisions = [x.strip() for x in args.precisions.split(',')]

    # Print the configuration
    print("=" * 60)
    print("GPU Inference Benchmark")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Input shape:  {input_shape}")
    print(f"Batch sizes:  {batch_sizes}")
    print(f"Precisions:   {precisions}")
    print(f"Warmup:       {args.warmup}")
    print(f"Iterations:   {args.iterations}")

    if torch.cuda.is_available():
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        print(f"CUDA:         {torch.version.cuda}")

    # Run the benchmark
    results = run_benchmark(
        args.model_path,
        input_shape,
        batch_sizes,
        precisions,
        args.warmup,
        args.iterations,
        args.device
    )

    # Print the summary
    print_summary_table(results)

    # Print the best configuration
    optimal = find_optimal_batch_size(results)
    if optimal:
        print("\n[Optimal Configuration]")
        for precision, (batch, qps) in optimal.items():
            print(f"  {precision}: batch_size={batch}, throughput={qps:.1f} QPS")

    # Create the output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export results
    if args.export_csv:
        export_csv(results, output_dir / "benchmark_results.csv")

    if args.export_json:
        export_json(results, args.model_path, output_dir / "benchmark_results.json")

    # Compare with the baseline
    compare_with_baseline(results, args.baseline)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
