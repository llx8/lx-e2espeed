# GPU Image Operation Routing

## PyTorch first

Use `torch.nn.functional.interpolate`, `grid_sample`, `pad`, pooling, separable/grouped `conv2d`, tensor slicing, and pointwise operations when they express the reference semantics. Rectangular dilation/erosion can use max pooling (`dilation(x)` or `-max_pool2d(-x)` for erosion).

Avoid `unfold` for very large images or kernels when it materializes a large patch tensor.

## Kornia

Use Kornia for Gaussian/median filters, morphology with shaped kernels, geometry, color conversion, and differentiable image operations already resident in PyTorch CUDA tensors. Match `border_type`, kernel shape, sigma, normalization, and engine. Kornia `unfold` morphology can have extreme peak memory for large kernels; prefer convolution, pooling, split passes, or overlap tiles after measurement.

The bundled `large_kernel_eroder.py` uses exact radius composition for rectangular flat kernels. Its repeated ellipse erosion is a discrete approximation and requires mask regression. The bundled `large_kernel_gaussian_blur.py` composes sigma across smaller separable passes; it is an approximation of a single OpenCV auto-sigma kernel and must be measured for quality and speed.

When product tolerance permits soft-mask approximation, benchmark repeated `avg_pool2d` box blurs before implementing a large Gaussian kernel. Choose two or three odd box widths whose combined variance approximates the target sigma, keep the mask in FP16, and validate final mask coverage, ROI boxes, routing, and final images. Prefer this mature path when an exact large Gaussian costs material latency but later resize or blending makes its pixel-level distinction immaterial.

## cuCIM and CuPy

Use cuCIM for mature CUDA-native image/scikit-image operations not provided efficiently by PyTorch/Kornia. `scripts/ops/cucim_distance.py` demonstrates Euclidean distance transform through DLPack and a CuPy external stream so the tensor stays on GPU and follows the active PyTorch stream.

Do not convert through NumPy. Preserve ownership until the consumer finishes and validate stream ordering. Install cuCIM/CuPy versions compatible with the active CUDA environment.

## Triton Lanczos3

Use `scripts/ops/triton_lanczos_resize.py` only when profiling proves Lanczos resize is material and PIL-style Lanczos3 semantics are required. It accepts contiguous CUDA BCHW FP16/FP32, uses half-pixel coordinates, widens support for antialiased downscale, and performs horizontal and vertical passes.

Its temporary buffer is proportional to `B*C*input_height*output_width`; include that allocation in peak-memory testing for maximum images. Use overlap/strip processing if that temporary becomes live at the peak.

## Large-image boundary

Route full-resolution morphology, blur, median, resize, mask, and blend operations to `gpu-inference-memory` when they need ordinary/overlap tiling or maximum-input validation.
