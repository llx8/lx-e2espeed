"""GPU checks of ../trt_plugin_export.py: export each wrapper, parse the ONNX with the plugin library loaded, build
an FP16 engine, deserialize it, run it, and compare with the eager module; also count the reformat layers around
each plugin, the number the A/B is about.

Usage: python test_trt_plugin_engine.py libtrt_plugins.so   (a GPU; torch, onnx, and tensorrt installed)
The CPU checks are in test_trt_plugin_export.py and run first. Peak host memory about 3 GB (two small builds).
Passed 2026-10-12 with torch 2.11, TensorRT 10.16, and a delivered library on an RTX 5090: 29 checks.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("tpe", Path(__file__).resolve().parents[1] / "trt_plugin_export.py")
tpe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpe)

ok = fail = 0
LOGGER = trt.Logger(trt.Logger.ERROR)


def check(cond, what):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("ok   " if cond else "FAIL ") + what)


def build(onnx_bytes):
    """FP16 engine from ONNX bytes, or (None, parser errors)."""
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_bytes):
        return None, "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    plan = builder.build_serialized_network(network, config)
    return plan, "" if plan else "build returned no plan"


def run(plan, inputs):
    """Deserialize the plan in a new runtime and run it on CUDA copies of the inputs; return the first output."""
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    feeds = iter(inputs)
    out = None
    keep = []
    for name in names:
        dtype = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t = next(feeds).to("cuda", dtype).contiguous()
        else:
            t = out = torch.empty(tuple(engine.get_tensor_shape(name)), dtype=dtype, device="cuda")
        keep.append(t)
        context.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.current_stream()
    check(context.execute_async_v3(stream.cuda_stream), "the engine runs")
    stream.synchronize()
    layers = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))["Layers"]
    names = [layer if isinstance(layer, str) else layer.get("Name", "") for layer in layers]
    return out.float().cpu(), names


if len(sys.argv) != 2:
    sys.exit(__doc__)
tpe.load_plugin_library(sys.argv[1])
torch.manual_seed(0)
import test_trt_plugin_export as cpu  # noqa: E402  (its CPU checks run first and build the reference modules)

ok, fail = ok + cpu.ok, fail + cpu.fail
print("-- GPU engine checks")

# GroupNorm: the CPU test's network, already swapped to the plugin; the eager reference is nn.GroupNorm.
x = cpu.x
buf = io.BytesIO()
torch.onnx.export(cpu.net, (x,), buf, opset_version=17, custom_opsets=tpe.PLUGIN_OPSETS, dynamo=False)
plan, err = build(buf.getvalue())
check(plan is not None, f"GroupNorm plugin engine builds {err}")
if plan:
    got, names = run(plan, [x])
    e = float((got - cpu.ref).abs().max())
    check(e < 5e-2, f"GroupNorm engine matches nn.GroupNorm in FP16 (max {e:.2e})")
    reformats = sum("Reformatting CopyNode" in n for n in names)
    plugin = [n for n in names if tpe.GROUPNORM_OP in n and "Reformatting" not in n]
    check(len(plugin) == 2, f"2 GroupNorm plugin layers in the plan ({len(names)} layers, {reformats} reformats; "
                            "the FP32 network inputs and outputs bring some)")

# MSDA: the CPU test's wrapped module; the eager reference is Deformable-DETR's forward.
buf = io.BytesIO()
torch.onnx.export(cpu.Wrap(cpu.holder.attn), (cpu.query, cpu.refp, cpu.src), buf, opset_version=17,
                  custom_opsets=tpe.PLUGIN_OPSETS, dynamo=False)
plan, err = build(buf.getvalue())
check(plan is not None, f"MSDA plugin engine builds {err}")
if plan:
    want = cpu.attn(cpu.query, cpu.refp, cpu.src, cpu.shapes, cpu.start)
    got, names = run(plan, [cpu.query, cpu.refp, cpu.src])
    e = float((got - want).abs().max())
    check(e < 5e-2, f"MSDA engine matches Deformable-DETR's forward in FP16 (max {e:.2e})")
    reformats = sum("Reformatting CopyNode" in n for n in names)
    plugin = [n for n in names if tpe.MSDA_FULL_OP in n and "Reformatting" not in n]
    check(len(plugin) == 1, f"1 MSDA plugin layer in the plan ({len(names)} layers, {reformats} reformats)")
print(f"passed {ok}, failed {fail}")
sys.exit(1 if fail else 0)
