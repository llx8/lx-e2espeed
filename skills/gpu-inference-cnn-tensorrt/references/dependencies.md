# TensorRT Source Dependencies

| Capability | Required dependencies |
| --- | --- |
| Shared runtime and ONNX rewrites | PyTorch, ONNX, ONNX Simplifier, TensorRT Python bindings |
| TSCNN GroupNorm+SiLU plugin | TensorRT Python bindings, Triton language/runtime, PyTorch CUDA |
| ColorAttn D16 plugin | CMake, C++17, CUDA toolkit, TensorRT headers/library, SM120 GPU |
| Static VAE example | Diffusers, PyTorch, shared runtime dependencies |
| BiRefNet deformable-conv ONNX export | torchvision and `deform_conv2d_onnx_exporter==1.2.0`; PyTorch 2.9 requires the documented HALF mapping |
| Nsight analysis | Nsight Systems CLI (`nsys`) and NVTX-enabled workload |
| Precision tools | ONNX and NumPy; backend-specific capture code remains project-owned |

Build the ColorAttn plugin out of tree:

```bash
cmake -S scripts/plugins/color_attn_d16 -B /tmp/color-attn-build \
  -DTRT_LIB=/path/to/tensorrt/lib \
  -DTRT_INCLUDE=/path/to/tensorrt/include \
  -DCUDA_INC_DIR=/path/to/cuda/include \
  -DCUDA_LIB_DIR=/path/to/cuda
cmake --build /tmp/color-attn-build -j
```

Do not commit temporary build directories. After validating the target GPU, CUDA, and TensorRT ABI, copy the shared libraries production loads into the project's agreed directory and commit them; load them through the project's path convention, never from another project or `/tmp`.
