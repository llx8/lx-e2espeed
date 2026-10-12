"""Load a compiled TensorRT plugin library, and export PyTorch modules as its plugin nodes.

Template, translated from a delivered implementation (2026-10). Copy it into the target repo next to its engine code,
set PLUGIN_LIBRARY and PLUGINS, and check every op name, attribute, input order, and constraint against the plugin
library's own headers and README before use: the plugin library defines them, this file only follows them.

- load_plugin_library(): call before building or deserializing a plan that uses the plugins. Fails fast on a missing
  file, a Git LFS pointer, and a creator missing from TensorRT's registry.
- use_groupnorm_plugin(model): every nn.GroupNorm becomes a block-reduce GroupNorm plugin node (no grid barrier, so
  no MPS deadlock). Call it on the model to export for the FP16 engine only.
- use_msda_full_plugin(model, module_type): every multi-scale deformable attention module (Deformable-DETR layout:
  value_proj, sampling_offsets, attention_weights, output_proj) becomes value_proj, one plugin node that does both
  linears, the softmax, the sampling locations, and the sampling, then output_proj.
- PLUGIN_OPSETS: pass as torch.onnx.export(..., custom_opsets=PLUGIN_OPSETS), with the TorchScript exporter
  (dynamo=False): symbolic() is how the plugin nodes get into the graph.

Each wrapper's eager forward is a PyTorch reference of the plugin, so the swapped model can be compared with the
original before export: they agree to float rounding, which proves the weight layout.
"""
import ctypes
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn.functional as F
from torch.autograd import Function

PLUGIN_LIBRARY = Path(__file__).resolve().parent / "engines" / "libtrt_plugins.so"  # TODO: the shipped library
# Any domain works: TensorRT's ONNX parser finds a plugin by op name and version only.
PLUGIN_DOMAIN = "trt_plugins"
PLUGIN_OPSETS = {PLUGIN_DOMAIN: 1}
GROUPNORM_OP = "HWCGroupNormPlugin_TRT"
MSDA_FULL_OP = "MultiscaleDeformableAttnFullPlugin_TRT"
PLUGINS = [(GROUPNORM_OP, "1"), (MSDA_FULL_OP, "1")]  # TODO: (name, version) the library registers and the plan uses
LFS_POINTER = b"version https://git-lfs"

_loaded = set()


def load_plugin_library(path=PLUGIN_LIBRARY, plugins=PLUGINS):
    """Load the library once and check that TensorRT's registry has every plugin.

    The library has no rpath: libnvinfer, libcudart, and libcublas must already be loaded, which importing tensorrt
    and torch does, so this module imports both first.
    """
    path = Path(path)
    if path in _loaded:
        return
    if not path.is_file():
        raise FileNotFoundError(f"no TensorRT plugin library at {path}: it ships by Git LFS (run `git lfs pull` in "
                                "the repo), or rebuild it with the repo's plugin build script")
    with open(path, "rb") as f:
        if f.read(len(LFS_POINTER)) == LFS_POINTER:
            raise RuntimeError(f"{path} is a Git LFS pointer, not the library: run `git lfs pull` in the repo")
    # RTLD_GLOBAL: the library's REGISTER_TENSORRT_PLUGIN registers its creators in TensorRT's global registry.
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    registry = trt.get_plugin_registry()
    missing = [f"{name} v{version}" for name, version in plugins if registry.get_creator(name, version, "") is None]
    if missing:
        raise RuntimeError(f"after loading {path}, TensorRT's plugin registry lacks {', '.join(missing)}")
    _loaded.add(path)


def _swap(model, match, make):
    """Replace every child module that match() accepts with make(child), in place; return the count."""
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if match(child):
                setattr(parent, name, make(name, child))
                count += 1
    return count


# Plugin constants (gamma, beta) reach the autograd.Function as an integer key: a tensor argument would become a
# graph input, and the parser would feed it through extra copy layers in every block. Registered outside the trace.
_CONSTS = {}


def _register_const(value):
    _CONSTS[len(_CONSTS)] = value
    return len(_CONSTS) - 1


class GroupNormPluginFunction(Function):
    """F.group_norm; exported as the GroupNorm plugin with gamma and beta as float-list attributes."""

    @staticmethod
    def forward(ctx, x, key, num_groups, eps):
        gamma, beta = _CONSTS[key]
        return F.group_norm(x, num_groups, gamma.to(x.dtype), beta.to(x.dtype), eps)

    @staticmethod
    def symbolic(g, x, key, num_groups, eps):
        gamma, beta = _CONSTS[key]
        out = g.op(f"{PLUGIN_DOMAIN}::{GROUPNORM_OP}", x, num_groups_i=num_groups, eps_f=eps, relu_i=0,
                   gamma_f=gamma.float().cpu().tolist(), beta_f=beta.float().cpu().tolist())
        out.setType(x.type())
        return out


class GroupNormPlugin(torch.nn.Module):
    def __init__(self, gn):
        super().__init__()
        # The original module is not kept as a child, or a second swap would wrap it again.
        self.num_groups, self.eps = gn.num_groups, float(gn.eps)
        self.key = _register_const((gn.weight.detach().clone(), gn.bias.detach().clone()))

    def forward(self, x):
        return GroupNormPluginFunction.apply(x, self.key, self.num_groups, self.eps)


def use_groupnorm_plugin(model):
    """Swap every affine nn.GroupNorm for the plugin; return the count, which the caller should assert."""
    def match(m):
        if isinstance(m, torch.nn.GroupNorm) and not m.affine:
            raise ValueError("the GroupNorm plugin takes gamma and beta; this GroupNorm has affine=False")
        return isinstance(m, torch.nn.GroupNorm)
    return _swap(model, match, lambda name, gn: GroupNormPlugin(gn))


def ms_deform_attn_reference(value, spatial_shapes, locations, weights):
    """Multi-scale deformable attention sampling in PyTorch (the public grid_sample formulation).

    value [N, S, H, Dh]; spatial_shapes [L, 2] as (h, w); locations [N, Q, H, L, P, 2] as (x, y) in [0, 1];
    weights [N, Q, H, L, P]. Returns [N, Q, H * Dh].
    """
    n, _, heads, dh = value.shape
    _, q, _, levels, points, _ = locations.shape
    sizes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
    per_level = value.split([h * w for h, w in sizes], dim=1)
    grids = 2 * locations - 1
    samples = []
    for level, (h, w) in enumerate(sizes):
        v = per_level[level].flatten(2).transpose(1, 2).reshape(n * heads, dh, h, w)
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        samples.append(F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False))
    weights = weights.transpose(1, 2).reshape(n * heads, 1, q, levels * points)
    out = (torch.stack(samples, dim=-2).flatten(-2) * weights).sum(-1).view(n, heads * dh, q)
    return out.transpose(1, 2).contiguous()


class MSDAFullPluginFunction(Function):
    """Both linears, softmax, sampling locations, and sampling; exported as the Full MSDA plugin.

    value [N, S, D] after value_proj; reference_points [N, Q, L, 2]; query [N, Q, D]; w_cat [D, 3*H*L*P] and b_cat
    are the sampling-offset and attention-weight linears concatenated, offset columns first. Returns [N, Q, D].
    """

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, reference_points, query, w_cat, b_cat, heads, points):
        n, s, d = value.shape
        q, levels = query.shape[1], spatial_shapes.shape[0]
        n_off = heads * levels * points * 2
        lin = query @ w_cat.to(query.dtype) + b_cat.to(query.dtype)
        offsets = lin[..., :n_off].reshape(n, q, heads, levels, points, 2)
        weights = F.softmax(lin[..., n_off:].reshape(n, q, heads, levels * points), -1)
        weights = weights.reshape(n, q, heads, levels, points)
        normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).to(query.dtype)
        locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
        return ms_deform_attn_reference(value.view(n, s, heads, d // heads), spatial_shapes, locations, weights)

    @staticmethod
    def symbolic(g, value, spatial_shapes, level_start_index, reference_points, query, w_cat, b_cat, heads, points):
        # The plugin takes int32 shapes and start indices (6 is ONNX INT32).
        spatial_shapes = g.op("Cast", spatial_shapes, to_i=6)
        level_start_index = g.op("Cast", level_start_index, to_i=6)
        out = g.op(f"{PLUGIN_DOMAIN}::{MSDA_FULL_OP}", value, spatial_shapes, level_start_index, reference_points,
                   query, w_cat, b_cat, num_heads_i=heads)
        out.setType(query.type())
        return out


class MSDAFullPlugin(torch.nn.Module):
    """value_proj, padding mask, and output_proj as before; the middle is one plugin node."""

    def __init__(self, attn):
        super().__init__()
        self.heads, self.points = attn.n_heads, attn.n_points
        self.value_proj, self.output_proj = attn.value_proj, attn.output_proj
        so, aw = attn.sampling_offsets, attn.attention_weights
        # The plugin's GEMM is FP16: store the weights in FP16, so the plan holds no conversion.
        self.register_buffer("w_cat", torch.cat([so.weight.t(), aw.weight.t()], 1).detach().half().contiguous())
        self.register_buffer("b_cat", torch.cat([so.bias, aw.bias]).detach().half())

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        if reference_points.shape[-1] != 2:
            # Box reference points ([..., 4]) and model-specific layouts need their own reshape before this call.
            raise ValueError(f"the MSDA plugin takes [N, Q, L, 2] reference points, got {tuple(reference_points.shape)}")
        out = MSDAFullPluginFunction.apply(value, input_spatial_shapes, input_level_start_index, reference_points,
                                           query, self.w_cat, self.b_cat, self.heads, self.points)
        return self.output_proj(out)


def use_msda_full_plugin(model, module_type, head_dim=32, levels=(3, 4), points=4, heads=(8,)):
    """Swap every module_type instance for the plugin wrapper; return the count.

    The defaults are the constraints of the library this template was written for (FP16, batch 1, 8 heads of 32
    channels, 3 or 4 levels, 4 points); take the real ones from its sources. That library's creator ignores the
    num_heads attribute and builds 8 heads, so another head count fails only at enqueue. A module outside them raises.
    """
    def make(name, m):
        if (m.d_model // m.n_heads != head_dim or m.n_heads not in heads or m.n_levels not in levels
                or m.n_points != points):
            raise ValueError(f"{name}: the MSDA plugin does not support d_model={m.d_model} n_heads={m.n_heads} "
                             f"n_levels={m.n_levels} n_points={m.n_points}")
        return MSDAFullPlugin(m)
    return _swap(model, lambda m: isinstance(m, module_type), make)
