---
name: gpu-inference-trt-plugins
description: "Reuse a tested TensorRT plugin library before writing one: GroupNorm without a grid barrier (MPS deadlock), fused multi-scale deformable attention, IPluginV3 versions of built-in plugins; vendoring, loading, ONNX export, checks."
---

# Reuse TensorRT plugins

## When
- The ONNX has GroupNormalization or InstanceNormalization and production runs MPS (assume it does), or Myelin normalization kernels are slow.
- Multi-scale deformable attention (the Deformable-DETR family) is a large share of the engine, or its built-in plugin sits between reformat layers.
- Any built-in plugin in TensorRT 10.11 or later: check whether a version 2 (IPluginV3) exists.
- Another measured hot op that a library on this machine already covers (window attention, ECA, padding, heads).

## 1. Look for an existing plugin first
- Read `lx-e2espeed/local/site.md` when it exists: it lists this machine's tested plugin libraries and the repos that integrated them. Otherwise ask the user whether a colleague has one. A new plugin is the last resort.
- The library's README and headers are the authority for each plugin's op name, version, attributes, input order, dtypes, formats, and constraints; this skill gives only the method and what was measured.
- Name the colleague who wrote the plugins wherever their gain is reported (the user's wish, 2026-10).

## Plugins seen
Measured 2026-10 with TensorRT 10.16 on an RTX 5090 (sm_120), FP16 engines, unless noted.

| Plugin | Replaces | Constraints in the library seen | Effect |
|---|---|---|---|
| block-reduce GroupNorm (LINEAR and HWC8) | ONNX GroupNormalization, which Myelin lowers to grid-barrier kernels | FP16; gamma, beta as attributes | grid-barrier kernels 1 to 0; latency +0.07 ms, inside noise (the plugin splits the Myelin graph); the HWC8 variant removed 37 reformats (0.96 ms) on sm_86, where convolutions pick HWC8 |
| full deformable attention: both linears as one cuBLAS GEMM, then softmax, locations, and sampling in one kernel | the middle of the attention module, between value_proj and output_proj | FP16, batch 1, 32 channels per head, 3 or 4 levels, 4 points | -0.43 ms (-4.5%) over the V3 built-in; process memory +270 MiB (one cuBLAS handle per instance, 12 instances); cold start +0.19 s; needs libcublas at run time |
| built-in MultiscaleDeformableAttnPlugin_TRT version 2 (IPluginV3), by the ONNX node attribute `plugin_version="2"` | the same plugin's version 1 (IPluginV2DynamicExt) | same kernel | 956 to 634 layers, 48 reformats to 0, engine 8.76 to 7.37 ms (-14.8%); the plan lost its auxiliary streams, giving back 0.73 ms of overlap |

The library seen also has window attention, LayerNorm with shift and partition (Swin), ECA, reflection padding, and segmentation heads; those were measured only in the project that wrote them (Swin hybrid -12% on Ampere, -1.4% on the RTX 5090).

## 2. Integrate
One experiment per plugin, in the repo that builds the engine.
1. Vendor: copy the plugin sources unchanged (for example into `plugins/src/`) and name the source repo, commit, and license in the build script's header. Nothing imports the library's repo at run time.
2. Build script: `nvcc -std=c++17 -O3 -Xcompiler -fPIC -shared -cudart shared`, one `-gencode arch=compute_XX,code=sm_XX` per production compute capability, the headers of the runtime's TensorRT version (pip TensorRT ships none: use the TensorRT OSS release of that version), linking `-l:libnvinfer.so.10 -lcudart` plus `-lcublas` for GEMM plugins, no rpath. Write the library next to the plan, never over a production binary.
3. Loader and export: copy `scripts/trt_plugin_export.py`, set its library path and plugin list, and check its op names and attributes against the headers. Call `load_plugin_library()` before building and before deserializing; swap modules only in the model exported for the FP16 engine, assert the count, and pass `custom_opsets=PLUGIN_OPSETS` to the TorchScript exporter.
4. Reference: an FP16-only plugin stays out of the FP32 reference build, which keeps the built-in op.
5. Ship the library with the plan (Git LFS in the same repo, with `.gitattributes` and an un-ignore line), and say in the repo's readme: load it before the plan, rebuild it before the plan, its compute capabilities, CUDA and TensorRT majors, and runtime libraries.

## 3. Check before the A/B
- ONNX op counts: `python -c "import collections, onnx, sys; print(collections.Counter(n.op_type for n in onnx.load(sys.argv[1], load_external_data=False).graph.node))" model.onnx`. Expect the plugin count and no GroupNormalization left.
- The plan: the `gpu-inference-mps-hang` scanner, `--names` then `--all`, must report 0 grid-barrier kernels. The library: `cuobjdump -sass lib.so | awk '/Function :/{f=$3; s=y=0} /LDG.E.STRONG.GPU/{s=1} /YIELD/{y=1} s && y && f {print f; f=""}'` must print nothing.
- Reformats: count layers named `Reformatting CopyNode` next to the plugin in `engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)`.
- In the production image: `readelf -d lib.so | grep NEEDED` resolves, the plan deserializes, and one run equals the local run (L0).
- Under MPS: the partition test in the `gpu-inference-mps-hang` skill.
- After editing the template: `python scripts/tests/test_trt_plugin_export.py [lib.so]` (CPU, no GPU needed).

## Pitfalls
- Constants such as gamma, beta, and weights go in as attributes or initializers, never as graph inputs: an input cost a copy layer per block (+0.35 ms in one network). The template passes them by integer key for this reason.
- Keep the plugin's tensor 3-D where the library expects it: 4-D and 5-D views brought reformats (+1.8 ms).
- `supportsFormatCombination` must force the output format to equal the input's, or TensorRT inserts reformats.
- Each instance of a cuBLAS plugin calls `cublasCreate` in `attachToContext`: about 22 MiB per instance. Sharing one handle is a change to the library; ask its author.
- Swapping out an op the exporter could not see through can unblock TorchScript constant folding: slice bounds computed from a CUDA tensor became CUDA constants and `torch.onnx.export` segfaulted reading them. Pass the shapes as a Python list to the code that slices.
- An ONNX cached by file name keeps the old plugin or version until deleted: export fresh into a temporary directory.
- The plan and the library bind to the compute capabilities they were built for: a plan built on a CI runner's GPU does not load on production's, and a production GPU learned late (sm_86, not sm_120) meant rebuilding both. Name files by compute capability and confirm production's GPUs during onboarding.
- A cuBLASLt GEMM can contain CUTLASS serial split-K code (LDG.E.STRONG.GPU with YIELD) that the fixed shapes never run (0 of 12 launches). Each block waits only for an earlier one, so it is not the grid barrier; recorded so a SASS hit there is understood, and rechecked after a cuBLAS upgrade.

## Writing a new plugin
Only for a measured hotspot no library covers, and with the user's go-ahead: IPluginV3, reductions within a block (no grid barrier), the format rule above, and constants as attributes. The `gpu-inference-cnn-tensorrt` skill has a Triton-backed GroupNorm plugin and a D=16 attention plugin as worked examples; the playbook card `trt-plugin-build-hygiene` covers headers and build directories.
