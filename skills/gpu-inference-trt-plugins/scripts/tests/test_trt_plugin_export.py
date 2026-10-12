"""CPU checks of ../trt_plugin_export.py: eager wrappers against independent references, the exported plugin nodes,
and the loader's failure paths. With a compiled plugin library as the argument, also loads it for real.

Usage: python test_trt_plugin_export.py [libtrt_plugins.so]   (torch, onnx, and tensorrt installed; no GPU needed)
Passed 2026-10-12 with torch 2.11, onnx 1.21, TensorRT 10.16, and a delivered library: 20 checks.
"""
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("tpe", Path(__file__).resolve().parents[1] / "trt_plugin_export.py")
tpe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpe)

ok = fail = 0


def naive_sampling(value, shapes, loc, w):
    """Bilinear sampling point by point, zero padding, align_corners=False: independent of grid_sample."""
    n, _, heads, dh = value.shape
    _, q, _, levels, points, _ = loc.shape
    out = torch.zeros(n, q, heads, dh, dtype=value.dtype)
    starts = [0] + (shapes[:, 0] * shapes[:, 1]).cumsum(0).tolist()
    for b, i, h, lv, p in torch.cartesian_prod(*map(torch.arange, (n, q, heads, levels, points))).tolist():
        hh, ww = shapes[lv].tolist()
        x, y = loc[b, i, h, lv, p, 0] * ww - 0.5, loc[b, i, h, lv, p, 1] * hh - 0.5
        x0, y0 = int(torch.floor(x)), int(torch.floor(y))
        for yy, xx in ((y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)):
            if 0 <= yy < hh and 0 <= xx < ww:
                weight = (1 - abs(x - xx)) * (1 - abs(y - yy))
                out[b, i, h] += w[b, i, h, lv, p] * weight * value[b, starts[lv] + yy * ww + xx, h]
    return out.flatten(2)


def check(cond, what):
    global ok, fail
    ok, fail = ok + bool(cond), fail + (not cond)
    print(("ok   " if cond else "FAIL ") + what)


class RefMSDA(torch.nn.Module):
    """Deformable-DETR's MSDeformAttn forward, [N, Q, L, 2] reference points."""

    def __init__(self, d_model=64, n_levels=3, n_heads=2, n_points=4):
        super().__init__()
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = torch.nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = torch.nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = torch.nn.Linear(d_model, d_model)
        self.output_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, shapes, start, mask=None):
        n, q, _ = query.shape
        value = self.value_proj(input_flatten)
        if mask is not None:
            value = value.masked_fill(mask[..., None], 0.0)
        value = value.view(n, -1, self.n_heads, self.d_model // self.n_heads)
        off = self.sampling_offsets(query).view(n, q, self.n_heads, self.n_levels, self.n_points, 2)
        w = F.softmax(self.attention_weights(query).view(n, q, self.n_heads, -1), -1)
        w = w.view(n, q, self.n_heads, self.n_levels, self.n_points)
        norm = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
        loc = reference_points[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
        return self.output_proj(naive_sampling(value, shapes, loc, w))


torch.manual_seed(0)

# GroupNorm: eager equality and the exported node.
net = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.GroupNorm(32, 64), torch.nn.SiLU(),
                          torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.GroupNorm(32, 64))
for m in net.modules():
    if isinstance(m, torch.nn.GroupNorm):
        torch.nn.init.normal_(m.weight)
        torch.nn.init.normal_(m.bias)
x = torch.randn(1, 3, 16, 20)
ref = net(x)
check(tpe.use_groupnorm_plugin(net) == 2, "use_groupnorm_plugin swaps 2 GroupNorms")
check(tpe.use_groupnorm_plugin(net) == 0, "a second swap finds none")
check(torch.allclose(net(x), ref, atol=1e-5), f"eager GroupNorm wrapper equals nn.GroupNorm (max {float((net(x) - ref).abs().max()):.2e})")
buf = io.BytesIO()
torch.onnx.export(net, (x,), buf, opset_version=17, custom_opsets=tpe.PLUGIN_OPSETS, dynamo=False)
model = onnx.load_from_string(buf.getvalue())
nodes = [n for n in model.graph.node if n.op_type == tpe.GROUPNORM_OP]
check(len(nodes) == 2 and all(n.domain == tpe.PLUGIN_DOMAIN for n in nodes), "2 GroupNorm plugin nodes in the plugin domain")
attrs = {a.name: a for a in nodes[0].attribute}
check(sorted(attrs) == ["beta", "eps", "gamma", "num_groups", "relu"] and attrs["num_groups"].i == 32
      and len(attrs["gamma"].floats) == 64, "GroupNorm node attributes num_groups, eps, relu, gamma[64], beta[64]")
check([i.name for i in model.graph.input] == ["input.1"] or len(model.graph.input) == 1, "no constant became a graph input")
check(len(nodes[0].input) == 1, "the GroupNorm node takes only the activation")
check(any(o.domain == tpe.PLUGIN_DOMAIN for o in model.opset_import), "the plugin domain is in opset_import")
try:
    tpe.use_groupnorm_plugin(torch.nn.Sequential(torch.nn.GroupNorm(2, 4, affine=False)))
    check(False, "affine=False GroupNorm raises")
except ValueError:
    check(True, "affine=False GroupNorm raises")

# MSDA: eager equality with Deformable-DETR's forward, and the exported node.
shapes = torch.tensor([[8, 10], [4, 5], [2, 3]])
start = torch.cat([torch.zeros(1, dtype=torch.long), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
s = int((shapes[:, 0] * shapes[:, 1]).sum())
attn = RefMSDA()
for p in attn.parameters():
    torch.nn.init.normal_(p, std=0.2)
holder = torch.nn.Module()
holder.attn = attn
query, src = torch.randn(1, 7, 64), torch.randn(1, s, 64)
refp = torch.rand(1, 7, 3, 2)
mask = torch.zeros(1, s, dtype=torch.bool)
mask[0, -3:] = True
want = attn(query, refp, src, shapes, start, mask)
check(tpe.use_msda_full_plugin(holder, RefMSDA, head_dim=32, levels=(3,), points=4) == 1, "use_msda_full_plugin swaps 1 module")
got = holder.attn(query, refp, src, shapes, start, mask)
err = float((got - want).abs().max())
check(err < 2e-2, f"eager MSDA wrapper (FP16 weights) matches Deformable-DETR's forward (max {err:.2e})")
attn32 = RefMSDA()
attn32.load_state_dict(attn.state_dict())
holder32 = torch.nn.Module()
holder32.attn = attn32
tpe.use_msda_full_plugin(holder32, RefMSDA, head_dim=32, levels=(3,), points=4)
holder32.attn.w_cat = torch.cat([attn.sampling_offsets.weight.t(), attn.attention_weights.weight.t()], 1).detach()
holder32.attn.b_cat = torch.cat([attn.sampling_offsets.bias, attn.attention_weights.bias]).detach()
err32 = float((holder32.attn(query, refp, src, shapes, start, mask) - want).abs().max())
check(err32 < 1e-5, f"with FP32 weights the wrapper equals it to rounding (max {err32:.2e}): weight layout right")
try:
    tpe.use_msda_full_plugin(torch.nn.Sequential(RefMSDA(n_levels=2)), RefMSDA, levels=(3, 4))
    check(False, "an unsupported level count raises")
except ValueError:
    check(True, "an unsupported level count raises")
try:
    holder.attn(query, torch.rand(1, 7, 3, 4), src, shapes, start)
    check(False, "box reference points raise")
except ValueError:
    check(True, "box reference points raise")


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, query, refp, src):
        return self.m(query, refp, src, shapes, start)


buf = io.BytesIO()
torch.onnx.export(Wrap(holder.attn), (query, refp, src), buf, opset_version=17, custom_opsets=tpe.PLUGIN_OPSETS,
                  dynamo=False)
model = onnx.load_from_string(buf.getvalue())
nodes = [n for n in model.graph.node if n.op_type == tpe.MSDA_FULL_OP]
check(len(nodes) == 1 and len(nodes[0].input) == 7 and nodes[0].attribute[0].name == "num_heads"
      and nodes[0].attribute[0].i == 2, "one MSDA Full node, 7 inputs, num_heads=2")
casts = {n.output[0]: n.attribute[0].i for n in model.graph.node if n.op_type == "Cast"}
inits = {t.name: t.data_type for t in model.graph.initializer}
consts = {n.output[0]: n.attribute[0].t.data_type for n in model.graph.node if n.op_type == "Constant"}
check(all(6 in (casts.get(nodes[0].input[i]), inits.get(nodes[0].input[i]), consts.get(nodes[0].input[i]))
          for i in (1, 2)), "spatial_shapes and level_start_index reach the node as INT32 (Cast, folded if constant)")

# Loader failure paths, and a real library if given.
with tempfile.TemporaryDirectory() as d:
    try:
        tpe.load_plugin_library(Path(d) / "missing.so")
        check(False, "a missing library raises FileNotFoundError")
    except FileNotFoundError:
        check(True, "a missing library raises FileNotFoundError")
    pointer = Path(d) / "pointer.so"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    try:
        tpe.load_plugin_library(pointer)
        check(False, "an LFS pointer raises RuntimeError")
    except RuntimeError as e:
        check("LFS pointer" in str(e), "an LFS pointer raises RuntimeError")
if len(sys.argv) > 1:
    tpe.load_plugin_library(sys.argv[1])
    check(True, "the real library loads and registers both creators")
    with tempfile.TemporaryDirectory() as d:
        link = Path(d) / "again.so"
        link.symlink_to(Path(sys.argv[1]).resolve())
        try:
            tpe.load_plugin_library(link, plugins=[("NoSuchPlugin_TRT", "1")])
            check(False, "a creator missing from the registry raises")
        except RuntimeError as e:
            check("lacks NoSuchPlugin_TRT" in str(e), "a creator missing from the registry raises")
print(f"passed {ok}, failed {fail}")
sys.exit(1 if fail else 0)
