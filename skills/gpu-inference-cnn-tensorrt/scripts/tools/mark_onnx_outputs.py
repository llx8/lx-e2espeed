#!/usr/bin/env python3
"""Expose selected ONNX node outputs for precision bisection."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--node-index", type=int, action="append", default=[])
    parser.add_argument("--tensor", action="append", default=[])
    args = parser.parse_args()

    if not args.node_index and not args.tensor:
        raise ValueError("Provide at least one --node-index or --tensor")

    model = shape_inference.infer_shapes(onnx.load(args.input))
    graph = model.graph
    value_info = {
        item.name: item
        for item in [*graph.input, *graph.output, *graph.value_info]
    }
    requested = list(args.tensor)
    for index in args.node_index:
        if index < 0 or index >= len(graph.node):
            raise IndexError(f"node index {index} outside [0, {len(graph.node)})")
        requested.extend(graph.node[index].output)

    existing = {item.name for item in graph.output}
    for name in dict.fromkeys(requested):
        if name in existing:
            continue
        info = value_info.get(name)
        if info is None:
            raise KeyError(f"No inferred value_info for tensor: {name}")
        graph.output.append(info)
        existing.add(name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"saved {args.output} with {len(graph.output)} graph outputs")


if __name__ == "__main__":
    main()
