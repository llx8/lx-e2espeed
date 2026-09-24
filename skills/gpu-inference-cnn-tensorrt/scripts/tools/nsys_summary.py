#!/usr/bin/env python3
"""Export high-signal CUDA and NVTX summaries from an Nsight Systems report."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPORTS = ("cuda_gpu_kern_sum", "cuda_api_sum", "nvtx_sum")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help=".nsys-rep file")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if shutil.which("nsys") is None:
        raise RuntimeError("nsys is not available on PATH")
    if not args.report.is_file():
        raise FileNotFoundError(args.report)

    output_dir = args.output_dir or args.report.parent / f"{args.report.stem}_stats"
    output_dir.mkdir(parents=True, exist_ok=True)

    for report_name in REPORTS:
        output_path = output_dir / f"{report_name}.csv"
        result = subprocess.run(
            [
                "nsys",
                "stats",
                "--report",
                report_name,
                "--format",
                "csv",
                "--force-export=true",
                str(args.report),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        output_path.write_text(result.stdout, encoding="utf-8")
        print(output_path)


if __name__ == "__main__":
    main()
