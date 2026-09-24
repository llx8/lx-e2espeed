import os
import pathlib
from typing import Callable, Dict, List, Optional, Tuple

import onnx
import torch
from onnx import TensorProto, helper
from onnx.checker import ValidationError
from onnxsim import simplify
from torch.nn import Module

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - optional runtime dependency
    trt = None


DYNAMIC_DIM = -1
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if trt is not None else None
DEFAULT_SHARED_CONTEXT_MEMORY_BYTES = 4_680_843_264


class TensorRTContextMemoryPool:
    """A single CUDA buffer reused by non-concurrent TensorRT contexts."""

    def __init__(self, device: Optional[torch.device] = None, initial_size_bytes: int = 0):
        self.device = torch.device(device) if device is not None else torch.device("cuda")
        self.buffer: Optional[torch.Tensor] = None
        self.size_bytes = 0
        if initial_size_bytes > 0:
            self.ensure(initial_size_bytes)

    def ensure(self, size_bytes: int) -> int:
        size_bytes = int(size_bytes)
        if size_bytes <= 0:
            raise ValueError(f"TensorRT context memory size must be positive, got {size_bytes}")
        if self.buffer is None or self.size_bytes < size_bytes:
            if self.buffer is not None:
                self.buffer = None
                self.size_bytes = 0
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            self.buffer = torch.empty((size_bytes,), dtype=torch.uint8, device=self.device)
            self.size_bytes = size_bytes
        return self.buffer.data_ptr()


_SHARED_CONTEXT_MEMORY_POOLS: Dict[str, TensorRTContextMemoryPool] = {}


def _shared_context_memory_initial_size() -> int:
    return int(
        os.environ.get(
            "GPU_INFERENCE_TRT_SHARED_CONTEXT_MEMORY_BYTES",
            str(DEFAULT_SHARED_CONTEXT_MEMORY_BYTES),
        )
    )


def get_shared_context_memory_pool(device: Optional[torch.device] = None) -> TensorRTContextMemoryPool:
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT shared context memory requires CUDA.")

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

    key = str(device)
    pool = _SHARED_CONTEXT_MEMORY_POOLS.get(key)
    if pool is None:
        pool = TensorRTContextMemoryPool(
            device=device,
            initial_size_bytes=_shared_context_memory_initial_size(),
        )
        _SHARED_CONTEXT_MEMORY_POOLS[key] = pool
    return pool


def _shared_context_memory_enabled() -> bool:
    return bool(int(os.environ.get("GPU_INFERENCE_TRT_SHARED_CONTEXT_MEMORY", "1")))


class PythonTorchTensorRTModule(Module):
    """Minimal TensorRT module wrapper with a torch-like forward API."""

    def __init__(
        self,
        engine_path: str,
        input_names: List[str],
        output_names: List[str],
        context_memory_pool: Optional[TensorRTContextMemoryPool] = None,
    ):
        super().__init__()
        if trt is None:
            raise ImportError("TensorRT is not installed.")

        self.engine_path = engine_path
        self.input_names = input_names
        self.output_names = output_names
        self.context_memory_pool = context_memory_pool

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        if self.context_memory_pool is None:
            self.context = self.engine.create_execution_context()
        else:
            self.context = self.engine.create_execution_context_without_device_memory()
        if self.context is None:
            raise RuntimeError(
                f"Failed to create TensorRT execution context for engine: {engine_path}. "
                "This usually means CUDA memory allocation failed during context initialization."
            )

        self.input_dtypes = [
            torch.float16 if self.engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
            for name in self.input_names
        ]
        self.output_dtypes = [
            torch.float16 if self.engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
            for name in self.output_names
        ]

    def forward(self, *inputs: torch.Tensor):
        contiguous_inputs = [tensor.contiguous() for tensor in inputs]
        assert len(contiguous_inputs) == len(self.input_names), (
            f"Wrong number of inputs: expected {len(self.input_names)}, got {len(contiguous_inputs)}"
        )

        for index, name in enumerate(self.input_names):
            tensor = contiguous_inputs[index]
            if not tensor.is_cuda:
                tensor = tensor.cuda()
                contiguous_inputs[index] = tensor
            if tensor.dtype != self.input_dtypes[index]:
                raise TypeError(
                    f"Dtype mismatch for input {name}: expected {self.input_dtypes[index]}, got {tensor.dtype}"
                )

            requested_shape = tuple(tensor.shape)
            ok = self.context.set_input_shape(name, requested_shape)
            if ok is False:
                raise ValueError(
                    f"TensorRT rejected input shape {requested_shape} for input {name}."
                )
            actual_shape = tuple(self.context.get_tensor_shape(name))
            if actual_shape != requested_shape:
                raise ValueError(
                    f"TensorRT did not accept requested shape {requested_shape} for input {name}; "
                    f"runtime shape is {actual_shape}."
                )
            self.context.set_tensor_address(name, tensor.data_ptr())

        uninferred = self.context.infer_shapes()
        if uninferred:
            raise RuntimeError(f"TensorRT could not infer shapes for inputs: {uninferred}")

        if self.context_memory_pool is not None:
            required_memory = int(self.context.update_device_memory_size_for_shapes())
            if required_memory <= 0:
                required_memory = int(getattr(self.engine, "device_memory_size_v2", self.engine.device_memory_size))
            memory_ptr = self.context_memory_pool.ensure(required_memory)
            self._last_context_memory_size = required_memory
            self.context.set_device_memory(memory_ptr, required_memory)

        outputs = []
        current_device = torch.cuda.current_device()
        for index, name in enumerate(self.output_names):
            shape = tuple(self.context.get_tensor_shape(name))
            if DYNAMIC_DIM in shape:
                raise ValueError(f"Dynamic output shape is not supported at runtime for output {name}: {shape}")
            output = torch.empty(shape, dtype=self.output_dtypes[index], device=current_device)
            self.context.set_tensor_address(name, output.data_ptr())
            outputs.append(output)

        with torch.autograd.profiler.record_function("PythonTorchTensorRTModule:TensorRTRuntime"):
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def export_onnx(
    model: Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    onnx_path: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    dynamic_shapes: Optional[Dict[str, Dict[int, object]]] = None,
    opset_version: int = 17,
    dynamo_export: bool = False,
    force_rebuild: bool = False,
    simplify_model: bool = True,
    autograd_inlining: bool = True,
) -> str:
    if os.path.exists(onnx_path) and not force_rebuild:
        return onnx_path

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamic_shapes=dynamic_shapes,
            dynamo=dynamo_export,
            external_data=False,
            autograd_inlining=autograd_inlining,
        )

    if simplify_model:
        onnx_model = onnx.load(onnx_path)
        onnx_model_simplified, check = simplify(onnx_model)
        if not check:
            raise RuntimeError(f"Simplified ONNX model could not be validated: {onnx_path}")
        onnx.save(onnx_model_simplified, onnx_path)
    return onnx_path


def cast_decomposed_attention_to_bf16(
    onnx_path: str,
    output_path: Optional[str] = None,
    force_rewrite: bool = False,
) -> str:
    """Run decomposed MatMul -> Softmax -> MatMul attention in BF16.

    Legacy torch.onnx export lowers scaled-dot-product attention into separate
    QK MatMul, Softmax, and AV MatMul nodes. TensorRT FP16 can overflow there
    on large profiles, so this inserts casts around only those subgraphs.
    """
    output_path = output_path or onnx_path
    if os.path.exists(output_path) and not force_rewrite:
        return output_path

    model = onnx.load(onnx_path)
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    targets = {}
    patched = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or not softmax.input or not softmax.output:
            continue
        qk = producer.get(softmax.input[0])
        av_consumers = consumers.get(softmax.output[0], [])
        if qk is None or qk.op_type != "MatMul":
            continue
        if len(av_consumers) != 1 or av_consumers[0].op_type != "MatMul":
            continue
        av = av_consumers[0]
        targets[id(qk)] = ("qk", patched, softmax, av)
        targets[id(softmax)] = ("softmax", patched, qk, av)
        targets[id(av)] = ("av", patched, qk, softmax)
        patched += 1

    if patched == 0:
        raise RuntimeError(f"No decomposed attention subgraphs found to patch: {onnx_path}")

    new_nodes = []
    for node in model.graph.node:
        target = targets.get(id(node))
        if target is None:
            new_nodes.append(node)
            continue

        role, index, _a, _b = target
        safe_name = node.name.replace("/", "_").strip("_") or f"attention_{index}_{role}"

        if role == "qk":
            bf16_inputs = []
            for input_index, input_name in enumerate(node.input):
                cast_output = f"{input_name}_{safe_name}_bf16"
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[input_name],
                        outputs=[cast_output],
                        name=f"{safe_name}_input{input_index}_cast_bf16",
                        to=TensorProto.BFLOAT16,
                    )
                )
                bf16_inputs.append(cast_output)
            original_output = node.output[0]
            node.input[:] = bf16_inputs
            node.output[0] = f"{original_output}_{safe_name}_bf16"
            new_nodes.append(node)
        elif role == "softmax":
            qk = targets[id(node)][2]
            node.input[0] = qk.output[0]
            original_output = node.output[0]
            node.output[0] = f"{original_output}_{safe_name}_bf16"
            new_nodes.append(node)
        elif role == "av":
            softmax = targets[id(node)][3]
            bf16_inputs = [softmax.output[0]]
            for input_index, input_name in enumerate(list(node.input)[1:], start=1):
                cast_output = f"{input_name}_{safe_name}_bf16"
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[input_name],
                        outputs=[cast_output],
                        name=f"{safe_name}_input{input_index}_cast_bf16",
                        to=TensorProto.BFLOAT16,
                    )
                )
                bf16_inputs.append(cast_output)
            original_output = node.output[0]
            bf16_output = f"{original_output}_{safe_name}_bf16"
            node.input[:] = bf16_inputs
            node.output[0] = bf16_output
            new_nodes.append(node)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[bf16_output],
                    outputs=[original_output],
                    name=f"{safe_name}_output_cast_fp16",
                    to=TensorProto.FLOAT16,
                )
            )

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    try:
        onnx.checker.check_model(model)
    except ValidationError as exc:
        if "SeedVRGroupNormPlugin" not in str(exc):
            raise
        print("Attention BF16 rewrite: skipped ONNX checker for SeedVRGroupNormPlugin custom op")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model, output_path)
    print(f"Attention BF16 rewrite: patched decomposed attention={patched}")
    return output_path


def replace_decomposed_attention_with_plugin(
    onnx_path: str,
    output_path: Optional[str] = None,
    sm_scale: float = 1.0,
    force_rewrite: bool = False,
) -> str:
    """Fuse decomposed MatMul -> Softmax -> MatMul attention into ColorAttnD16Int8Plugin.

    Detects the same triplets as ``cast_decomposed_attention_to_bf16`` (a Softmax
    fed by a MatMul whose single consumer is a MatMul) and replaces each with one
    custom plugin node taking [Q, K^T, V] = [QK.in0, QK.in1, AV.in1] and producing
    AV.out. The softmax scale is already folded into the kept Q/K Mul nodes, so the
    plugin's ``sm_scale`` defaults to 1.0. head_dim is 16 for every attention in
    ColorAdjustor, so the blanket pattern match is safe.
    """
    output_path = output_path or onnx_path
    if os.path.exists(output_path) and not force_rewrite:
        return output_path

    model = onnx.load(onnx_path)
    producer = {output: node for node in model.graph.node for output in node.output}
    consumers: Dict[str, List] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    skip = set()       # ids of QK / Softmax nodes to drop
    av_to_plugin = {}  # id(AV node) -> replacement plugin node
    patched = 0
    for softmax in model.graph.node:
        if softmax.op_type != "Softmax" or not softmax.input or not softmax.output:
            continue
        qk = producer.get(softmax.input[0])
        av_consumers = consumers.get(softmax.output[0], [])
        if qk is None or qk.op_type != "MatMul" or len(qk.input) != 2:
            continue
        if len(av_consumers) != 1 or av_consumers[0].op_type != "MatMul":
            continue
        av = av_consumers[0]
        if len(av.input) != 2 or not av.output:
            continue

        query = qk.input[0]
        key_t = qk.input[1]
        value = av.input[1]
        safe_name = (softmax.name.replace("/", "_").strip("_") or f"attention_{patched}")
        plugin = helper.make_node(
            "ColorAttnD16Int8Plugin",
            inputs=[query, key_t, value],
            outputs=[av.output[0]],
            name=f"{safe_name}_ColorAttnD16Int8Plugin",
        )
        plugin.domain = ""
        plugin.attribute.append(helper.make_attribute("sm_scale", float(sm_scale)))
        av_to_plugin[id(av)] = plugin
        skip.add(id(qk))
        skip.add(id(softmax))
        patched += 1

    if patched == 0:
        raise RuntimeError(f"No decomposed attention subgraphs found to patch: {onnx_path}")

    # Emit each plugin in the AV node's slot so the graph stays topologically sorted.
    new_nodes = []
    for node in model.graph.node:
        if id(node) in skip:
            continue
        replacement = av_to_plugin.get(id(node))
        new_nodes.append(replacement if replacement is not None else node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model, output_path)
    print(f"Attention plugin rewrite: fused decomposed attention={patched} into ColorAttnD16Int8Plugin")
    return output_path


_COLOR_ATTN_PLUGIN_LIB = (
    pathlib.Path(__file__).resolve().parent
    / ".."
    / "plugins"
    / "color_attn_d16"
    / "build"
    / "libcolor_attn_d16_plugin.so"
)
_loaded_plugin_libs: set = set()


def load_color_attn_plugin_lib(lib_path: Optional[str] = None) -> str:
    """ctypes-load the ColorAttnD16Int8Plugin .so so its creator registers with TensorRT.

    Idempotent; must be called before both build_engine and engine deserialization for
    any graph containing the plugin node.
    """
    import ctypes

    path = os.path.abspath(str(lib_path or _COLOR_ATTN_PLUGIN_LIB))
    if path in _loaded_plugin_libs:
        return path
    if not os.path.exists(path):
        raise FileNotFoundError(f"ColorAttnD16 plugin library not found: {path}")
    ctypes.CDLL(path)
    _loaded_plugin_libs.add(path)
    return path


def replace_omnisr_window_attention_with_plugin(
    onnx_path: str,
    output_path: Optional[str] = None,
    sm_scale: float = 1.0,
    force_rewrite: bool = False,
    cast_io_to_fp16: bool = False,
) -> str:
    """Fuse OmniSR window attention (Einsum QK -> Add rel_pos_bias -> Softmax -> Einsum AV)
    into ColorAttnD16Int8Plugin with the optional bias input.

    Pattern:
      Einsum("bhid,bhjd->bhij")(q, k) -> Add(., bias[h,i,j]) -> Softmax -> Einsum("bhij,bhjd->bhid")(., v)
    The softmax scale is pre-folded into q by the kept Mul, so sm_scale defaults to 1.
    K is wired straight in with kt_transposed=0 (the plugin reads raw [B,H,Nkv,16] K via
    strides), so no graph-side Transpose is added. With a plain fp16 export the q/k/v/bias
    tensors are already fp16 and are wired directly; set ``cast_io_to_fp16`` only for graphs
    whose attention runs in fp32 (legacy fp32-patched export).
    Channel_Attention softmaxes (MatMul -> Mul temperature -> Softmax) do not match and are
    left untouched.
    """
    output_path = output_path or onnx_path
    if os.path.exists(output_path) and not force_rewrite:
        return output_path

    model = onnx.load(onnx_path)
    graph = model.graph
    producer = {output: node for node in graph.node for output in node.output}
    consumers: Dict[str, List] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    def einsum_equation(node) -> str:
        for attr in node.attribute:
            if attr.name == "equation":
                return attr.s.decode("utf-8").replace(" ", "")
        return ""

    skip = set()          # ids of QK-Einsum / Add / Softmax nodes to drop
    av_replacements = {}  # id(AV node) -> list of replacement nodes
    patched = 0
    for softmax in graph.node:
        if softmax.op_type != "Softmax" or not softmax.input or not softmax.output:
            continue
        add = producer.get(softmax.input[0])
        if add is None or add.op_type != "Add" or len(add.input) != 2:
            continue
        # One Add operand is the QK Einsum, the other is the rel_pos_bias tensor.
        qk = producer.get(add.input[0])
        bias_name = add.input[1]
        if qk is None or qk.op_type != "Einsum":
            qk = producer.get(add.input[1])
            bias_name = add.input[0]
        if qk is None or qk.op_type != "Einsum" or einsum_equation(qk) != "bhid,bhjd->bhij":
            continue
        av_consumers = consumers.get(softmax.output[0], [])
        if len(av_consumers) != 1:
            continue
        av = av_consumers[0]
        if av.op_type != "Einsum" or einsum_equation(av) != "bhij,bhjd->bhid" or not av.output:
            continue
        # The QK output and Softmax output must feed nothing else.
        if len(consumers.get(qk.output[0], [])) != 1 or len(consumers.get(add.output[0], [])) != 1:
            continue

        query, key = qk.input[0], qk.input[1]
        value = av.input[1]
        safe_name = (softmax.name.replace("/", "_").strip("_") or f"omnisr_attention_{patched}")

        nodes = []

        def cast_fp16(tensor_name: str, tag: str) -> str:
            if not cast_io_to_fp16:
                return tensor_name
            out = f"{tensor_name}_{safe_name}_{tag}_fp16"
            nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[tensor_name],
                    outputs=[out],
                    name=f"{safe_name}_{tag}_cast_fp16",
                    to=TensorProto.FLOAT16,
                )
            )
            return out

        q_in = cast_fp16(query, "q")
        k_in = cast_fp16(key, "k")  # raw [B,H,Nkv,16]; plugin reads it via strides (kt_transposed=0)
        v_in = cast_fp16(value, "v")
        bias_in = cast_fp16(bias_name, "bias")
        plugin_out = f"{av.output[0]}_{safe_name}_fp16" if cast_io_to_fp16 else av.output[0]
        plugin = helper.make_node(
            "ColorAttnD16Int8Plugin",
            inputs=[q_in, k_in, v_in, bias_in],
            outputs=[plugin_out],
            name=f"{safe_name}_ColorAttnD16Int8Plugin",
        )
        plugin.domain = ""
        plugin.attribute.append(helper.make_attribute("sm_scale", float(sm_scale)))
        plugin.attribute.append(helper.make_attribute("has_bias", 1))
        plugin.attribute.append(helper.make_attribute("kt_transposed", 0))
        nodes.append(plugin)
        if cast_io_to_fp16:
            nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[plugin_out],
                    outputs=[av.output[0]],
                    name=f"{safe_name}_output_cast_fp32",
                    to=TensorProto.FLOAT,
                )
            )

        av_replacements[id(av)] = nodes
        skip.add(id(qk))
        skip.add(id(add))
        skip.add(id(softmax))
        patched += 1

    if patched == 0:
        raise RuntimeError(f"No OmniSR window attention subgraphs found to patch: {onnx_path}")

    new_nodes = []
    for node in graph.node:
        if id(node) in skip:
            continue
        replacement = av_replacements.get(id(node))
        if replacement is not None:
            new_nodes.extend(replacement)
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    onnx.save(model, output_path)
    print(f"OmniSR attention plugin rewrite: fused window attention={patched} into ColorAttnD16Int8Plugin")
    return output_path


def normalize_trt_plugin_domains(onnx_path: str, output_path: Optional[str] = None) -> str:
    """Strip the custom domain from torch.onnx-exported TRT plugin nodes.

    torch.onnx custom symbolics must emit ops under a namespace (we use
    ``trt::``); the TRT ONNX parser resolves plugin creators registered in the
    default namespace, so rewrite those nodes to domain "" and drop the now
    unused opset import.
    """
    output_path = output_path or onnx_path
    model = onnx.load(onnx_path)
    changed = False
    for node in model.graph.node:
        if node.domain and node.op_type.endswith("Plugin"):
            node.domain = ""
            changed = True
    if changed:
        kept = [op for op in model.opset_import if op.domain != "trt"]
        del model.opset_import[:]
        model.opset_import.extend(kept)
        onnx.save(model, output_path)
    return output_path


def build_engine(
    onnx_path: str,
    engine_path: str,
    profile_input_shape: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
    precision: str = "fp16",
    force_rebuild: bool = False,
    optimization_level: int = 3,
    workspace_size: Optional[int] = None,
) -> str:
    if trt is None:
        raise ImportError("TensorRT is not installed.")

    if os.path.exists(engine_path) and not force_rebuild:
        return engine_path

    pathlib.Path(os.path.dirname(engine_path) or ".").mkdir(parents=True, exist_ok=True)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    with (
        trt.Builder(TRT_LOGGER) as builder,
        builder.create_network(network_flags) as network,
        trt.OnnxParser(network, TRT_LOGGER) as parser,
    ):
        if not parser.parse_from_file(onnx_path):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.builder_optimization_level = optimization_level
        if hasattr(trt, "ProfilingVerbosity"):
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if workspace_size is not None:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size))

        if profile_input_shape:
            profile = builder.create_optimization_profile()
            for name, shape_triplet in profile_input_shape.items():
                min_shape, opt_shape, max_shape = shape_triplet
                profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine.")

        with open(engine_path, "wb") as f:
            f.write(engine)

    return engine_path


def load_engine(
    onnx_path: str,
    engine_path: str,
    profile_input_shape: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
    precision: str = "fp16",
    force_rebuild: bool = False,
    optimization_level: int = 3,
    workspace_size: Optional[int] = None,
    context_memory_pool: Optional[TensorRTContextMemoryPool] = None,
) -> PythonTorchTensorRTModule:
    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        profile_input_shape=profile_input_shape,
        precision=precision,
        force_rebuild=force_rebuild,
        optimization_level=optimization_level,
        workspace_size=workspace_size,
    )
    onnx_model = onnx.load(onnx_path)
    input_names = [item.name for item in onnx_model.graph.input]
    output_names = [item.name for item in onnx_model.graph.output]
    if context_memory_pool is None and _shared_context_memory_enabled():
        context_memory_pool = get_shared_context_memory_pool()
    return PythonTorchTensorRTModule(
        engine_path,
        input_names,
        output_names,
        context_memory_pool=context_memory_pool,
    )


def load_engine_from_pytorch(
    model: Module,
    dummy_inputs: Tuple[torch.Tensor, ...],
    engine_name: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    profile_input_shape: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
    dynamic_shapes: Optional[Dict[str, Dict[int, object]]] = None,
    opset_version: int = 17,
    precision: str = "fp16",
    force_rebuild: bool = False,
    optimization_level: int = 3,
    cache_dir: Optional[str] = None,
    dynamo_export: bool = False,
    simplify_model: bool = True,
    workspace_size: Optional[int] = None,
    onnx_postprocess: Optional[Callable[[str], str]] = None,
    engine_suffix: Optional[str] = None,
    context_memory_pool: Optional[TensorRTContextMemoryPool] = None,
) -> PythonTorchTensorRTModule:
    cache_root = cache_dir or os.path.join(os.path.dirname(os.path.realpath(__file__)), "engines")
    os.makedirs(cache_root, exist_ok=True)

    onnx_path = os.path.join(cache_root, f"{engine_name}.onnx")
    plan_suffix = engine_suffix or precision
    engine_path = os.path.join(cache_root, f"{engine_name}_{plan_suffix}.plan")

    export_onnx(
        model=model,
        dummy_inputs=dummy_inputs,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset_version,
        dynamo_export=dynamo_export,
        force_rebuild=force_rebuild,
        simplify_model=simplify_model,
    )
    engine_onnx_path = onnx_postprocess(onnx_path) if onnx_postprocess is not None else onnx_path

    return load_engine(
        onnx_path=engine_onnx_path,
        engine_path=engine_path,
        profile_input_shape=profile_input_shape,
        precision=precision,
        force_rebuild=force_rebuild,
        optimization_level=optimization_level,
        workspace_size=workspace_size,
        context_memory_pool=context_memory_pool,
    )
