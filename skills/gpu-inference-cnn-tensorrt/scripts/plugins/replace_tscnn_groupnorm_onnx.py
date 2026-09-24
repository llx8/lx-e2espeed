import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _attr(node: onnx.NodeProto, name: str, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def _consumers(nodes: list[onnx.NodeProto]) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for name in node.input:
            out.setdefault(name, []).append(node)
    return out


def _producers(nodes: list[onnx.NodeProto]) -> dict[str, onnx.NodeProto]:
    out: dict[str, onnx.NodeProto] = {}
    for node in nodes:
        for name in node.output:
            out[name] = node
    return out


def _prune_unreachable_nodes(model: onnx.ModelProto) -> int:
    required = {value.name for value in model.graph.output}
    kept: list[onnx.NodeProto] = []
    for node in reversed(model.graph.node):
        if any(output in required for output in node.output):
            kept.append(node)
            required.update(name for name in node.input if name)
    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    return removed


def _constant_array(
    name: str,
    producers: dict[str, onnx.NodeProto],
    initializers: dict[str, np.ndarray],
) -> np.ndarray | None:
    if name in initializers:
        return initializers[name]
    node = producers.get(name)
    if node is None or node.op_type != "Constant":
        return None
    value = _attr(node, "value")
    if value is None:
        return None
    return numpy_helper.to_array(value)


def _constant_input(
    node: onnx.NodeProto,
    data_input: str,
    producers: dict[str, onnx.NodeProto],
    initializers: dict[str, np.ndarray],
) -> np.ndarray | None:
    for name in node.input:
        if name == data_input:
            continue
        value = _constant_array(name, producers, initializers)
        if value is not None:
            return value
    return None


def _non_shape_consumers(consumers: dict[str, list[onnx.NodeProto]], name: str) -> list[onnx.NodeProto]:
    return [node for node in consumers.get(name, []) if node.op_type != "Shape"]


def _single_consumer(consumers: dict[str, list[onnx.NodeProto]], name: str, op_type: str) -> onnx.NodeProto | None:
    items = consumers.get(name, [])
    if len(items) != 1:
        return None
    node = items[0]
    if node.op_type != op_type:
        return None
    return node


def _single_non_shape_consumer(
    consumers: dict[str, list[onnx.NodeProto]],
    name: str,
    op_type: str,
) -> onnx.NodeProto | None:
    items = _non_shape_consumers(consumers, name)
    if len(items) != 1:
        return None
    node = items[0]
    if node.op_type != op_type:
        return None
    return node


def _match_silu_after_add(
    add: onnx.NodeProto,
    consumers: dict[str, list[onnx.NodeProto]],
) -> list[onnx.NodeProto] | None:
    if len(add.output) != 1:
        return None
    users = _non_shape_consumers(consumers, add.output[0])
    sigmoid = next((node for node in users if node.op_type == "Sigmoid"), None)
    mul = next((node for node in users if node.op_type == "Mul"), None)
    if sigmoid is None or mul is None or len(sigmoid.output) != 1 or len(mul.output) != 1:
        return None
    if set(mul.input) != {add.output[0], sigmoid.output[0]}:
        return None
    if consumers.get(sigmoid.output[0], []) != [mul]:
        return None
    return [sigmoid, mul]


@dataclass(frozen=True)
class _ExpandedGnMatch:
    input_name: str
    output_name: str
    groups: int
    eps: float
    weight: np.ndarray
    bias: np.ndarray
    skip_names: set[str]
    fused_silu: bool


def _match_expanded_groupnorm(
    node: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
    initializers: dict[str, np.ndarray],
    fuse_silu: bool,
) -> _ExpandedGnMatch | None:
    if node.op_type != "InstanceNormalization" or len(node.input) != 3 or len(node.output) != 1:
        return None
    pre_reshape = producers.get(node.input[0])
    if pre_reshape is None or pre_reshape.op_type != "Reshape" or len(pre_reshape.input) < 2:
        return None

    shape = _constant_array(pre_reshape.input[1], producers, initializers)
    if shape is None or shape.size < 2:
        return None
    groups = int(shape.reshape(-1)[1])

    scale = _constant_array(node.input[1], producers, initializers)
    bias = _constant_array(node.input[2], producers, initializers)
    if scale is None or bias is None:
        return None
    scale = scale.astype(np.float32).reshape(-1)
    bias = bias.astype(np.float32).reshape(-1)
    if scale.size != groups or bias.size != groups:
        return None
    if not np.allclose(scale, 1.0) or not np.allclose(bias, 0.0):
        return None

    post_reshape = _single_consumer(consumers, node.output[0], "Reshape")
    if post_reshape is None or len(post_reshape.output) != 1:
        return None
    mul = _single_consumer(consumers, post_reshape.output[0], "Mul")
    if mul is None or len(mul.output) != 1:
        return None
    add = _single_consumer(consumers, mul.output[0], "Add")
    if add is None or len(add.output) != 1:
        return None

    weight = _constant_input(mul, post_reshape.output[0], producers, initializers)
    beta = _constant_input(add, mul.output[0], producers, initializers)
    if weight is None or beta is None:
        return None
    weight = weight.astype(np.float32).reshape(-1)
    beta = beta.astype(np.float32).reshape(-1)
    if weight.size % groups != 0 or beta.size != weight.size:
        return None

    output_name = add.output[0]
    skip_names = {pre_reshape.name, node.name, post_reshape.name, mul.name, add.name}
    fused_silu = False
    matched_silu = _match_silu_after_add(add, consumers) if fuse_silu else None
    if matched_silu is not None:
        output_name = matched_silu[-1].output[0]
        skip_names.update(matched.name for matched in matched_silu)
        fused_silu = True

    return _ExpandedGnMatch(
        input_name=pre_reshape.input[0],
        output_name=output_name,
        groups=groups,
        eps=float(_attr(node, "epsilon", 1e-6)),
        weight=weight,
        bias=beta,
        skip_names=skip_names,
        fused_silu=fused_silu,
    )


def _make_plugin_attrs(match: _ExpandedGnMatch, block_pix: int):
    return [
        helper.make_attribute("groups", int(match.groups)),
        helper.make_attribute("eps", float(match.eps)),
        helper.make_attribute("block_pix", int(block_pix)),
        helper.make_attribute("weight", match.weight.astype(np.float32).reshape(-1)),
        helper.make_attribute("bias", match.bias.astype(np.float32).reshape(-1)),
    ]


def replace_tscnn_groupnorm(
    input_path: str,
    output_path: str,
    block_pix: int,
    fuse_silu: bool = True,
    only_last: bool = False,
) -> tuple[int, int, int]:
    model = onnx.load(input_path)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    nodes = list(model.graph.node)
    producers = _producers(nodes)
    consumers = _consumers(nodes)
    target_names: set[str] | None = None
    if only_last:
        groupnorms = [node for node in nodes if node.op_type == "InstanceNormalization"]
        if not groupnorms:
            raise ValueError(f"No InstanceNormalization nodes found in {input_path}")
        target_names = {groupnorms[-1].name}

    skip: set[str] = set()
    replaced = 0
    fused = 0
    new_nodes: list[onnx.NodeProto] = []

    for node in nodes:
        if node.name in skip:
            continue
        if node.op_type != "InstanceNormalization" or (target_names is not None and node.name not in target_names):
            new_nodes.append(node)
            continue
        matched = _match_expanded_groupnorm(node, producers, consumers, initializers, fuse_silu)
        if matched is None:
            new_nodes.append(node)
            continue
        plugin = helper.make_node(
            "TSCNNGroupNormSiluPlugin" if matched.fused_silu else "TSCNNGroupNormPlugin",
            [matched.input_name],
            [matched.output_name],
            name=f"{node.name}_TSCNNGroupNormPlugin",
        )
        plugin.attribute.extend(_make_plugin_attrs(matched, block_pix))
        skip.update(matched.skip_names)
        replaced += 1
        fused += int(matched.fused_silu)
        new_nodes.append(plugin)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    removed = _prune_unreachable_nodes(model)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return replaced, fused, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--block-pix", type=int, default=4096)
    parser.add_argument("--no-fuse-silu", action="store_true")
    parser.add_argument("--only-last", action="store_true")
    args = parser.parse_args()

    replaced, fused, removed = replace_tscnn_groupnorm(
        args.input,
        args.output,
        block_pix=args.block_pix,
        fuse_silu=not args.no_fuse_silu,
        only_last=args.only_last,
    )
    print(f"replaced {replaced} expanded GroupNorm nodes; fused {fused} SiLU patterns; pruned {removed} nodes")


if __name__ == "__main__":
    main()
