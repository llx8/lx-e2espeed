#!/usr/bin/env python3
"""Inspect a prebuilt ColorAdjustor TRT engine: list IO tensors + profile shape
ranges, dump layer information, and optionally run it (for nsys to trace the
real attention kernels)."""
import argparse
import sys

import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--run", action="store_true", help="run one inference at max shape")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--layers", action="store_true", help="dump layer info JSON")
    ap.add_argument("--plugin", action="append", default=[], help="plugin .so to CDLL-load first")
    args = ap.parse_args()

    import ctypes
    for so in args.plugin:
        ctypes.CDLL(so)
    trt.init_libnvinfer_plugins(LOGGER, "")
    runtime = trt.Runtime(LOGGER)
    with open(args.plan, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {args.plan}")

    print(f"=== IO tensors: {args.plan.split('/')[-1]} ===")
    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        is_in = mode == trt.TensorIOMode.INPUT
        print(f"  [{'IN ' if is_in else 'OUT'}] {name}  shape={tuple(shape)}  dtype={dtype}")
        if is_in:
            mn = engine.get_tensor_profile_value(0, name, trt.TensorProfileValue.MIN) if False else None
        io.append((name, is_in, tuple(shape), dtype))

    # profile shape ranges for dynamic inputs
    print("=== profile[0] shape ranges (inputs) ===")
    for name, is_in, shape, dtype in io:
        if not is_in:
            continue
        try:
            mn = engine.get_tensor_profile_shape(name, 0)
            print(f"  {name}: min={tuple(mn[0])} opt={tuple(mn[1])} max={tuple(mn[2])}")
        except Exception as exc:
            print(f"  {name}: (no profile) {exc}")

    if args.layers:
        ctx = engine.create_execution_context()
        insp = engine.create_engine_inspector()
        insp.execution_context = ctx
        info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        print("=== layer info JSON ===")
        print(info[:20000])

    if args.run:
        ctx = engine.create_execution_context()
        # set every dynamic input to its MAX profile shape
        tensors = {}
        for name, is_in, shape, dtype in io:
            tdtype = torch.float16 if dtype == trt.DataType.HALF else torch.float32
            if is_in:
                mx = engine.get_tensor_profile_shape(name, 0)[2]
                rshape = tuple(mx)
                ctx.set_input_shape(name, rshape)
                t = torch.randn(rshape, dtype=tdtype, device="cuda")
                tensors[name] = t
                ctx.set_tensor_address(name, t.data_ptr())
        ctx.infer_shapes()
        for name, is_in, shape, dtype in io:
            if not is_in:
                rshape = tuple(ctx.get_tensor_shape(name))
                tdtype = torch.float16 if dtype == trt.DataType.HALF else torch.float32
                t = torch.empty(rshape, dtype=tdtype, device="cuda")
                tensors[name] = t
                ctx.set_tensor_address(name, t.data_ptr())
        for _ in range(2):
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("tensorrt.engine.steady_state")
        for _ in range(args.iters):
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(f"ran {args.iters} iters at max shape")


if __name__ == "__main__":
    main()
