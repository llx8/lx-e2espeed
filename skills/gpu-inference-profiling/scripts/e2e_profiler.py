#!/usr/bin/env python3
"""
================================================================================
End-to-end inference profiler - reference template
================================================================================

Purpose: a reference template that shows how to write an end-to-end inference
         profiler. Follow its structure and adapt a copy inside the project.

Features:
- StageProfiler: per-stage timer for preprocessing, inference, and postprocessing
- local pipeline analysis
- remote service load testing
- latency distribution and throughput

Adapt for each project:
1. Stages: name them after the project's real pipeline.
2. Inference call: replace with the project's inference logic.
3. Inputs: match the real input format.
4. Remote interface: adjust the request format when testing a service.

Example (original usage, for reference only):
    # local analysis
    python e2e_profiler.py --mode local --config config.yaml

    # remote load test
    python e2e_profiler.py --mode remote --service-url http://localhost:8000/predict

In practice:
1. Copy the StageProfiler class into the project.
2. Wrap each pipeline stage in profiler.stage("stage name").
3. Adjust the report format to the project's needs.

Key classes:
- StageProfiler: the core timer, reusable as is
- StageStats: the statistics record

================================================================================
"""

import argparse
import asyncio
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# Stage Profiler
# ============================================================

@dataclass
class StageStats:
    """Per-stage statistics"""
    times: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else 0

    @property
    def p50(self) -> float:
        return statistics.median(self.times) if self.times else 0

    @property
    def p95(self) -> float:
        if len(self.times) < 2:
            return self.times[0] if self.times else 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99(self) -> float:
        if len(self.times) < 2:
            return self.times[0] if self.times else 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]


class StageProfiler:
    """Pipeline stage profiler"""

    def __init__(self, cuda_sync: bool = True):
        self.stages: Dict[str, StageStats] = {}
        self.cuda_sync = cuda_sync
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str):
        """Record the duration of one stage"""
        if self.cuda_sync:
            self._cuda_sync()

        start = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda_sync:
                self._cuda_sync()

            elapsed = (time.perf_counter() - start) * 1000  # ms

            with self._lock:
                if name not in self.stages:
                    self.stages[name] = StageStats()
                self.stages[name].times.append(elapsed)

    def _cuda_sync(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    def report(self) -> str:
        """Build the analysis report"""
        if not self.stages:
            return "No stages recorded."

        total_time = sum(s.mean * s.count for s in self.stages.values())
        total_count = max(s.count for s in self.stages.values()) if self.stages else 0

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("Stage Profiling Report")
        lines.append("=" * 80)
        lines.append(
            f"{'Stage':<25} {'Count':>8} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'%':>6}"
        )
        lines.append("-" * 80)

        # Sort by mean duration
        sorted_stages = sorted(
            self.stages.items(),
            key=lambda x: x[1].mean,
            reverse=True
        )

        for name, stats in sorted_stages:
            pct = (stats.mean / (total_time / total_count) * 100) if total_time > 0 else 0
            lines.append(
                f"{name:<25} {stats.count:>8} {stats.mean:>9.2f}ms "
                f"{stats.p50:>9.2f}ms {stats.p95:>9.2f}ms {stats.p99:>9.2f}ms {pct:>5.1f}%"
            )

        lines.append("-" * 80)
        e2e_mean = total_time / total_count if total_count > 0 else 0
        lines.append(f"{'End-to-End':<25} {total_count:>8} {e2e_mean:>9.2f}ms")
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Export as a dict"""
        return {
            name: {
                "count": stats.count,
                "mean_ms": round(stats.mean, 3),
                "p50_ms": round(stats.p50, 3),
                "p95_ms": round(stats.p95, 3),
                "p99_ms": round(stats.p99, 3),
            }
            for name, stats in self.stages.items()
        }

    def reset(self):
        """Reset all records"""
        with self._lock:
            self.stages.clear()


# ============================================================
# Resource Monitor
# ============================================================

class ResourceMonitor:
    """Background resource monitor"""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.running = False
        self.thread = None
        self.data = {
            "timestamps": [],
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_util": [],
            "gpu_memory_used": [],
            "gpu_memory_total": [],
        }

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor_loop(self):
        try:
            import psutil
        except ImportError:
            print("Warning: psutil not installed, CPU/memory monitoring disabled")
            psutil = None

        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            pynvml = None
            gpu_handle = None

        while self.running:
            self.data["timestamps"].append(time.time())

            # CPU and memory
            if psutil:
                self.data["cpu_percent"].append(psutil.cpu_percent())
                self.data["memory_percent"].append(psutil.virtual_memory().percent)

            # GPU
            if pynvml and gpu_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    self.data["gpu_util"].append(util.gpu)
                    self.data["gpu_memory_used"].append(mem.used / 1024**3)  # GB
                    self.data["gpu_memory_total"].append(mem.total / 1024**3)
                except:
                    pass

            time.sleep(self.interval)

    def report(self) -> str:
        """Build the resource usage report"""
        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("Resource Usage Summary")
        lines.append("=" * 50)

        if self.data["cpu_percent"]:
            cpu_avg = statistics.mean(self.data["cpu_percent"])
            cpu_max = max(self.data["cpu_percent"])
            lines.append(f"CPU:        {cpu_avg:.1f}% avg, {cpu_max:.1f}% max")

        if self.data["memory_percent"]:
            mem_avg = statistics.mean(self.data["memory_percent"])
            mem_max = max(self.data["memory_percent"])
            lines.append(f"Memory:     {mem_avg:.1f}% avg, {mem_max:.1f}% max")

        if self.data["gpu_util"]:
            gpu_avg = statistics.mean(self.data["gpu_util"])
            gpu_max = max(self.data["gpu_util"])
            lines.append(f"GPU Util:   {gpu_avg:.1f}% avg, {gpu_max:.1f}% max")

        if self.data["gpu_memory_used"]:
            gmem_avg = statistics.mean(self.data["gpu_memory_used"])
            gmem_max = max(self.data["gpu_memory_used"])
            gmem_total = self.data["gpu_memory_total"][0] if self.data["gpu_memory_total"] else 0
            lines.append(f"GPU Memory: {gmem_avg:.1f}GB avg, {gmem_max:.1f}GB max / {gmem_total:.1f}GB")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "cpu_percent_avg": statistics.mean(self.data["cpu_percent"]) if self.data["cpu_percent"] else 0,
            "cpu_percent_max": max(self.data["cpu_percent"]) if self.data["cpu_percent"] else 0,
            "memory_percent_avg": statistics.mean(self.data["memory_percent"]) if self.data["memory_percent"] else 0,
            "gpu_util_avg": statistics.mean(self.data["gpu_util"]) if self.data["gpu_util"] else 0,
            "gpu_util_max": max(self.data["gpu_util"]) if self.data["gpu_util"] else 0,
            "gpu_memory_used_avg_gb": statistics.mean(self.data["gpu_memory_used"]) if self.data["gpu_memory_used"] else 0,
        }


# ============================================================
# Remote Service Profiler
# ============================================================

@dataclass
class RequestResult:
    """Result of one request"""
    success: bool
    latency_ms: float
    status_code: int = 0
    error: str = ""


class RemoteServiceProfiler:
    """Remote service profiler"""

    def __init__(self, service_url: str):
        self.service_url = service_url
        self.results: List[RequestResult] = []

    def send_request(self, data: bytes, timeout: float = 30) -> RequestResult:
        """Send one request"""
        import urllib.request
        import urllib.error

        start = time.perf_counter()
        try:
            req = urllib.request.Request(
                self.service_url,
                data=data,
                headers={"Content-Type": "application/octet-stream"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                _ = response.read()
                latency = (time.perf_counter() - start) * 1000
                return RequestResult(True, latency, response.status)
        except urllib.error.HTTPError as e:
            latency = (time.perf_counter() - start) * 1000
            return RequestResult(False, latency, e.code, str(e))
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return RequestResult(False, latency, 0, str(e))

    def run_benchmark(
        self,
        data: bytes,
        num_requests: int = 100,
        concurrency: int = 1,
        warmup: int = 10
    ) -> Dict:
        """Run the benchmark"""
        print(f"Warming up ({warmup} requests)...")
        for _ in range(warmup):
            self.send_request(data)

        print(f"Running benchmark ({num_requests} requests, {concurrency} concurrent)...")
        self.results = []

        start_time = time.perf_counter()

        if concurrency == 1:
            # Sequential
            for i in range(num_requests):
                result = self.send_request(data)
                self.results.append(result)
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{num_requests}")
        else:
            # Concurrent
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(self.send_request, data) for _ in range(num_requests)]
                for i, future in enumerate(futures):
                    self.results.append(future.result())
                    if (i + 1) % 20 == 0:
                        print(f"  Progress: {i + 1}/{num_requests}")

        total_time = time.perf_counter() - start_time

        return self._analyze_results(total_time)

    def _analyze_results(self, total_time: float) -> Dict:
        """Analyze the results"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if not successful:
            return {"error": "All requests failed", "failures": len(failed)}

        latencies = sorted([r.latency_ms for r in successful])
        n = len(latencies)

        return {
            "total_requests": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_s": round(total_time, 2),
            "throughput_qps": round(len(successful) / total_time, 2),
            "latency": {
                "min_ms": round(latencies[0], 2),
                "max_ms": round(latencies[-1], 2),
                "mean_ms": round(statistics.mean(latencies), 2),
                "p50_ms": round(latencies[int(n * 0.5)], 2),
                "p95_ms": round(latencies[int(n * 0.95)], 2),
                "p99_ms": round(latencies[int(n * 0.99)], 2),
            }
        }

    def report(self, results: Dict) -> str:
        """Build the report"""
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("Remote Service Benchmark Report")
        lines.append("=" * 60)
        lines.append(f"Service URL: {self.service_url}")
        lines.append(f"Total Requests: {results['total_requests']}")
        lines.append(f"Successful: {results['successful']}")
        lines.append(f"Failed: {results['failed']}")
        lines.append(f"Total Time: {results['total_time_s']}s")
        lines.append(f"Throughput: {results['throughput_qps']} QPS")
        lines.append("")
        lines.append("Latency:")
        lat = results['latency']
        lines.append(f"  Min:  {lat['min_ms']}ms")
        lines.append(f"  Mean: {lat['mean_ms']}ms")
        lines.append(f"  P50:  {lat['p50_ms']}ms")
        lines.append(f"  P95:  {lat['p95_ms']}ms")
        lines.append(f"  P99:  {lat['p99_ms']}ms")
        lines.append(f"  Max:  {lat['max_ms']}ms")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# Local Pipeline Profiler
# ============================================================

def profile_local_pipeline(
    pipeline_fn: Callable,
    input_data: Any,
    iterations: int = 100,
    warmup: int = 20
) -> Tuple[StageProfiler, ResourceMonitor]:
    """
    Profile a local inference pipeline

    Args:
        pipeline_fn: inference function taking (input_data, profiler)
        input_data: input data
        iterations: measured iterations
        warmup: warmup iterations
    """
    profiler = StageProfiler()
    monitor = ResourceMonitor()

    # Warm up
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        pipeline_fn(input_data, profiler=None)

    # Discard warmup records
    profiler.reset()

    # Start monitoring
    monitor.start()

    # Measured run
    print(f"Profiling ({iterations} iterations)...")
    for i in range(iterations):
        pipeline_fn(input_data, profiler=profiler)
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{iterations}")

    # Stop monitoring
    monitor.stop()

    return profiler, monitor


# ============================================================
# Report Generator
# ============================================================

def generate_report(
    stage_profiler: Optional[StageProfiler],
    resource_monitor: Optional[ResourceMonitor],
    remote_results: Optional[Dict],
    output_path: Optional[str] = None
) -> Dict:
    """Build the full report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "stages": stage_profiler.to_dict() if stage_profiler else {},
        "resources": resource_monitor.to_dict() if resource_monitor else {},
        "remote_benchmark": remote_results or {},
    }

    # Print the report
    if stage_profiler:
        print(stage_profiler.report())

    if resource_monitor:
        print(resource_monitor.report())

    # Save JSON
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to: {output_path}")

    return report


# ============================================================
# Example Pipeline
# ============================================================

def example_pipeline(input_data: Any, profiler: Optional[StageProfiler] = None):
    """Sample inference pipeline (for the demo)"""
    import time

    if profiler is None:
        # Without profiling, run directly
        time.sleep(0.01)  # simulated preprocessing
        time.sleep(0.05)  # simulated inference
        time.sleep(0.01)  # simulated postprocessing
        return {"result": "ok"}

    with profiler.stage("preprocess"):
        time.sleep(0.01)  # simulated preprocessing

    with profiler.stage("inference"):
        time.sleep(0.05)  # simulated inference

    with profiler.stage("postprocess"):
        time.sleep(0.01)  # simulated postprocessing

    return {"result": "ok"}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end inference service profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # profile a remote service
  %(prog)s --mode remote --service-url http://localhost:8000/predict --input-file test.jpg

  # load test
  %(prog)s --mode remote --service-url http://localhost:8000/predict --input-file test.jpg -c 10 -n 1000

  # run the demo
  %(prog)s --mode demo
        """
    )

    parser.add_argument('--mode', choices=['local', 'remote', 'demo'],
                        default='demo', help='analysis mode')
    parser.add_argument('--service-url', help='remote service URL')
    parser.add_argument('--input-file', help='input file path')
    parser.add_argument('-n', '--requests', type=int, default=100,
                        help='number of requests (default: 100)')
    parser.add_argument('-c', '--concurrency', type=int, default=1,
                        help='concurrency (default: 1)')
    parser.add_argument('--warmup', type=int, default=20,
                        help='warmup iterations (default: 20)')
    parser.add_argument('--output', help='output JSON path')

    args = parser.parse_args()

    if args.mode == 'demo':
        # Demo mode
        print("Running demo pipeline profiling...")
        profiler, monitor = profile_local_pipeline(
            example_pipeline,
            input_data=None,
            iterations=50,
            warmup=10
        )
        generate_report(profiler, monitor, None, args.output)

    elif args.mode == 'remote':
        if not args.service_url:
            print("Error: --service-url is required for remote mode")
            sys.exit(1)

        # Read the input data
        if args.input_file:
            input_data = Path(args.input_file).read_bytes()
        else:
            input_data = b"test data"

        # Remote service analysis
        profiler = RemoteServiceProfiler(args.service_url)
        monitor = ResourceMonitor()

        monitor.start()
        results = profiler.run_benchmark(
            input_data,
            num_requests=args.requests,
            concurrency=args.concurrency,
            warmup=args.warmup
        )
        monitor.stop()

        print(profiler.report(results))
        print(monitor.report())

        if args.output:
            report = {
                "timestamp": datetime.now().isoformat(),
                "service_url": args.service_url,
                "benchmark": results,
                "resources": monitor.to_dict(),
            }
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(f"\nReport saved to: {args.output}")

    elif args.mode == 'local':
        print("Local mode requires custom pipeline integration.")
        print("See the example_pipeline() function for reference.")
        print("\nExample usage in your code:")
        print("""
from e2e_profiler import StageProfiler, ResourceMonitor, profile_local_pipeline

def my_pipeline(input_data, profiler=None):
    if profiler:
        with profiler.stage("preprocess"):
            tensor = preprocess(input_data)
        with profiler.stage("inference"):
            output = model(tensor)
        with profiler.stage("postprocess"):
            result = postprocess(output)
    else:
        # ... normal execution ...
    return result

profiler, monitor = profile_local_pipeline(my_pipeline, input_data)
print(profiler.report())
print(monitor.report())
        """)


if __name__ == "__main__":
    main()
