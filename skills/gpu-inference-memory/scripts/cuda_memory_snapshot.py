#!/usr/bin/env python3
"""Record CUDA allocation history and dump a pickle snapshot."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def start_cuda_memory_history() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; cannot record CUDA memory history")
    torch.cuda.memory._record_memory_history()


def dump_cuda_memory_snapshot(snapshot_path: str = "cuda_memory_snapshot.pickle") -> str:
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(snapshot_path)
    resolved = str(Path(snapshot_path).resolve())
    print(f"CUDA memory snapshot saved: {resolved}")
    print("Tell the user which input and E2E iteration produced this snapshot.")
    return resolved


def run_with_oom_snapshot(
    run: Callable[[], Any],
    snapshot_path: str = "cuda_oom_snapshot.pickle",
) -> Any:
    """Run one real E2E call and preserve allocator history if it OOMs."""
    start_cuda_memory_history()
    try:
        return run()
    except torch.cuda.OutOfMemoryError:
        dump_cuda_memory_snapshot(snapshot_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump current CUDA memory history after project code enabled recording."
    )
    parser.add_argument("output", nargs="?", default="cuda_memory_snapshot.pickle")
    args = parser.parse_args()
    dump_cuda_memory_snapshot(args.output)


if __name__ == "__main__":
    main()
