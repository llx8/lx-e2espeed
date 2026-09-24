import json

import numpy as np
import torch
import triton
import triton.language as tl

try:
    import cupy as cp
    import tensorrt as trt
except ImportError:  # pragma: no cover - optional runtime dependency
    cp = None
    trt = None


def _volume(dims) -> int:
    total = 1
    for dim in dims:
        total *= int(dim)
    return total


@triton.jit
def _gn_partial_nchw_kernel(
    x,
    partial_sum,
    partial_sumsq,
    hw: tl.constexpr,
    group_ch: tl.constexpr,
    group_ch_block: tl.constexpr,
    block_pix: tl.constexpr,
):
    group = tl.program_id(0)
    pix_block = tl.program_id(1)
    pix = pix_block * block_pix + tl.arange(0, block_pix)
    ch = tl.arange(0, group_ch_block)
    offs = (group * group_ch + ch[:, None]) * hw + pix[None, :]
    mask = (pix[None, :] < hw) & (ch[:, None] < group_ch)
    vals = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    vals = tl.where(mask, vals, 0.0)
    s = tl.sum(tl.sum(vals, axis=1), axis=0)
    ss = tl.sum(tl.sum(vals * vals, axis=1), axis=0)
    n_blocks = tl.cdiv(hw, block_pix)
    tl.store(partial_sum + group * n_blocks + pix_block, s)
    tl.store(partial_sumsq + group * n_blocks + pix_block, ss)


@triton.jit
def _gn_stats_kernel(
    partial_sum,
    partial_sumsq,
    mean,
    invstd,
    n_blocks: tl.constexpr,
    elems_per_group: tl.constexpr,
    eps: tl.constexpr,
    block_n: tl.constexpr,
):
    group = tl.program_id(0)
    offs = tl.arange(0, block_n)
    mask = offs < n_blocks
    s = tl.load(partial_sum + group * n_blocks + offs, mask=mask, other=0.0)
    ss = tl.load(partial_sumsq + group * n_blocks + offs, mask=mask, other=0.0)
    total = tl.sum(s, axis=0)
    total_sq = tl.sum(ss, axis=0)
    m = total / elems_per_group
    var = total_sq / elems_per_group - m * m
    tl.store(mean + group, m)
    tl.store(invstd + group, tl.rsqrt(var + eps))


@triton.jit
def _gn_apply_nchw_kernel(
    x,
    weight,
    bias,
    mean,
    invstd,
    y,
    hw: tl.constexpr,
    group_ch: tl.constexpr,
    group_ch_block: tl.constexpr,
    block_pix: tl.constexpr,
    silu: tl.constexpr,
):
    group = tl.program_id(0)
    pix_block = tl.program_id(1)
    pix = pix_block * block_pix + tl.arange(0, block_pix)
    ch = tl.arange(0, group_ch_block)
    cidx = group * group_ch + ch
    offs = cidx[:, None] * hw + pix[None, :]
    mask = (pix[None, :] < hw) & (ch[:, None] < group_ch)
    vals = tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight + cidx).to(tl.float32)
    b = tl.load(bias + cidx).to(tl.float32)
    m = tl.load(mean + group)
    r = tl.load(invstd + group)
    out = (vals - m) * r * w[:, None] + b[:, None]
    if silu:
        out = out / (1.0 + tl.exp(-out))
    tl.store(y + offs, out, mask=mask)


def _launch_groupnorm_nchw(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    groups: int,
    eps: float,
    block_pix: int,
    silu: bool,
) -> None:
    if x.ndim not in (3, 4):
        raise ValueError(f"TSCNNGroupNorm plugin expects NCHW or NC(HW) input, got {tuple(x.shape)}")
    channels = int(x.shape[1])
    if channels % groups != 0:
        raise ValueError(f"channels={channels} must be divisible by groups={groups}")

    batch = int(x.shape[0])
    group_ch = channels // groups
    hw = int(x.shape[2] if x.ndim == 3 else x.shape[2] * x.shape[3])
    n_blocks = triton.cdiv(hw, block_pix)
    group_ch_block = triton.next_power_of_2(group_ch)
    partial_sum = torch.empty((batch, groups, n_blocks), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty_like(partial_sum)
    mean = torch.empty((batch, groups), device=x.device, dtype=torch.float32)
    invstd = torch.empty_like(mean)

    for batch_index in range(batch):
        x_b = x[batch_index : batch_index + 1]
        y_b = y[batch_index : batch_index + 1]
        _gn_partial_nchw_kernel[(groups, n_blocks)](
            x_b,
            partial_sum[batch_index],
            partial_sumsq[batch_index],
            hw,
            group_ch,
            group_ch_block,
            block_pix,
            num_warps=8,
        )
        block_n = triton.next_power_of_2(n_blocks)
        _gn_stats_kernel[(groups,)](
            partial_sum[batch_index],
            partial_sumsq[batch_index],
            mean[batch_index],
            invstd[batch_index],
            n_blocks,
            hw * group_ch,
            eps,
            block_n,
            num_warps=8,
        )
        _gn_apply_nchw_kernel[(groups, n_blocks)](
            x_b,
            weight,
            bias,
            mean[batch_index],
            invstd[batch_index],
            y_b,
            hw,
            group_ch,
            group_ch_block,
            block_pix,
            silu,
            num_warps=8,
        )


class _TSCNNGroupNormBase(trt.IPluginV2DynamicExt if trt is not None else object):
    PLUGIN_TYPE = ""
    FUSE_SILU = False

    def __init__(self, fc=None):
        if trt is not None:
            trt.IPluginV2DynamicExt.__init__(self)
        self.num_outputs = 1
        self.plugin_namespace = ""
        self.plugin_type = type(self).PLUGIN_TYPE
        self.plugin_version = "1"
        self.groups = 32
        self.eps = 1e-6
        self.block_pix = 4096
        self.weight = np.array([], dtype=np.float32)
        self.bias = np.array([], dtype=np.float32)
        self._weight_t = None
        self._bias_t = None

        if fc is not None:
            for field in fc:
                if field.name == "groups":
                    self.groups = int(np.array(field.data, dtype=np.int32).reshape(-1)[0])
                elif field.name == "eps":
                    self.eps = float(np.array(field.data, dtype=np.float32).reshape(-1)[0])
                elif field.name == "block_pix":
                    self.block_pix = int(np.array(field.data, dtype=np.int32).reshape(-1)[0])
                elif field.name == "weight":
                    self.weight = np.array(field.data, dtype=np.float32).reshape(-1).copy()
                elif field.name == "bias":
                    self.bias = np.array(field.data, dtype=np.float32).reshape(-1).copy()

    def get_output_datatype(self, index, input_types):
        return input_types[0]

    def get_output_dimensions(self, output_index, inputs, exprBuilder):
        return trt.DimsExprs(inputs[0])

    def serialize(self):
        return json.dumps(
            {
                "groups": self.groups,
                "eps": self.eps,
                "block_pix": self.block_pix,
                "weight": self.weight.tolist(),
                "bias": self.bias.tolist(),
            }
        ).encode("utf-8")

    def configure_plugin(self, inp, out):
        pass

    def supports_format_combination(self, pos, in_out, num_inputs):
        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0:
            return desc.type == trt.DataType.HALF
        if pos == 1:
            return desc.type == in_out[0].type
        return False

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        inp_dtype = trt.nptype(input_desc[0].type)
        in_shape = tuple(int(dim) for dim in input_desc[0].dims)
        out_shape = tuple(int(dim) for dim in output_desc[0].dims)

        x_mem = cp.cuda.UnownedMemory(inputs[0], _volume(in_shape) * cp.dtype(inp_dtype).itemsize, self)
        y_mem = cp.cuda.UnownedMemory(outputs[0], _volume(out_shape) * cp.dtype(inp_dtype).itemsize, self)
        x_cp = cp.ndarray(in_shape, dtype=inp_dtype, memptr=cp.cuda.MemoryPointer(x_mem, 0))
        y_cp = cp.ndarray(out_shape, dtype=inp_dtype, memptr=cp.cuda.MemoryPointer(y_mem, 0))
        x_t = torch.as_tensor(x_cp, device="cuda")
        y_t = torch.as_tensor(y_cp, device="cuda")

        if self._weight_t is None or self._weight_t.device != x_t.device:
            self._weight_t = torch.as_tensor(self.weight, device=x_t.device, dtype=torch.float32)
            self._bias_t = torch.as_tensor(self.bias, device=x_t.device, dtype=torch.float32)

        _launch_groupnorm_nchw(
            x_t,
            y_t,
            self._weight_t,
            self._bias_t,
            self.groups,
            self.eps,
            self.block_pix,
            self.FUSE_SILU,
        )
        return 0

    def clone(self):
        cloned = type(self)()
        cloned.__dict__.update(self.__dict__)
        cloned._weight_t = None
        cloned._bias_t = None
        return cloned


class TSCNNGroupNormPlugin(_TSCNNGroupNormBase):
    PLUGIN_TYPE = "TSCNNGroupNormPlugin"
    FUSE_SILU = False


class TSCNNGroupNormSiluPlugin(_TSCNNGroupNormBase):
    PLUGIN_TYPE = "TSCNNGroupNormSiluPlugin"
    FUSE_SILU = True


class _TSCNNGroupNormCreatorBase(trt.IPluginCreator if trt is not None else object):
    PLUGIN_CLASS = None
    PLUGIN_NAME = ""

    def __init__(self):
        if trt is not None:
            trt.IPluginCreator.__init__(self)
        self.name = self.PLUGIN_NAME
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [
                trt.PluginField("groups", np.array([], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("eps", np.array([], dtype=np.float32), trt.PluginFieldType.FLOAT32),
                trt.PluginField("block_pix", np.array([], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("weight", np.array([], dtype=np.float32), trt.PluginFieldType.FLOAT32),
                trt.PluginField("bias", np.array([], dtype=np.float32), trt.PluginFieldType.FLOAT32),
            ]
        )

    def create_plugin(self, name, fc):
        return self.PLUGIN_CLASS(fc)

    def deserialize_plugin(self, name, data):
        state = json.loads(data.decode("utf-8"))
        plugin = self.PLUGIN_CLASS()
        plugin.groups = int(state["groups"])
        plugin.eps = float(state["eps"])
        plugin.block_pix = int(state["block_pix"])
        plugin.weight = np.array(state["weight"], dtype=np.float32)
        plugin.bias = np.array(state["bias"], dtype=np.float32)
        return plugin


class TSCNNGroupNormPluginCreator(_TSCNNGroupNormCreatorBase):
    PLUGIN_CLASS = TSCNNGroupNormPlugin
    PLUGIN_NAME = "TSCNNGroupNormPlugin"


class TSCNNGroupNormSiluPluginCreator(_TSCNNGroupNormCreatorBase):
    PLUGIN_CLASS = TSCNNGroupNormSiluPlugin
    PLUGIN_NAME = "TSCNNGroupNormSiluPlugin"


_REGISTERED = False


def register_tscnn_groupnorm_silu_plugin() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    if trt is None:
        raise ImportError("TensorRT is required to register TSCNN GroupNorm plugins.")
    registry = trt.get_plugin_registry()
    registry.register_creator(TSCNNGroupNormPluginCreator(), "")
    registry.register_creator(TSCNNGroupNormSiluPluginCreator(), "")
    _REGISTERED = True
