# Bit-exact OpenCV warpAffine on the GPU

## When
The CPU reference crops, aligns, or letterboxes with `cv2.warpAffine` on uint8 images (INTER_LINEAR, BORDER_CONSTANT 0), and the GPU path should produce the same bytes. A float bilinear (`grid_sample`) differs by up to one level per pixel, which in one project moved the outputs from L1 to L2. The exact replica ran as fast as the float one (1.93 ms both), and the user picked it; two projects ship one (2026-10).

## Bundled op
`scripts/ops/opencv_warp_affine_u8.py` replicates both majors byte for byte (uint8, INTER_LINEAR, BORDER_CONSTANT; any affine; plain output or fused with the channel swap, CHW, and the lookup below), and `scripts/tests/test_opencv_warp_affine.py` checks it against the installed cv2: 4.10.0 and 5.0.0, 320 cases each, exact. OpenCV 5.x interpolates in float32 and rounds the coordinate differently in a row's SIMD body and its scalar tail; the op's docstring has the arithmetic. The two majors differ in 12.9% of the output bytes, so `opencv_major` follows the production image.

## OpenCV 4.10's arithmetic
The 8U path (imgwarp.cpp) does not interpolate in float. The arithmetic is version-specific: replicate the OpenCV of the production image, not the local one.
1. Without `WARP_INVERSE_MAP`, invert M in double: D = 1 / (M0 M4 - M1 M3) (0 if singular); M0' = M4 D, M4' = M0 D, M1' = -M1 D, M3' = -M3 D; M2' = -M0' M2 - M1' M5, M5' = -M3' M2 - M4' M5.
2. Per output pixel (x, y), in double, multiply then add, rounded half to even: adelta = rint(M0' x 1024), bdelta = rint(M3' x 1024), X0 = rint((M1' y + M2') 1024) + 16, Y0 = rint((M4' y + M5') 1024) + 16.
3. X = (X0 + adelta) >> 5, Y = (Y0 + bdelta) >> 5. The taps start at (X >> 5, Y >> 5), saturated to int16; the fractions fx = X & 31, fy = Y & 31 are 1/32 pixel.
4. Weights in 15 bits: (32 - fy)(32 - fx) 32, (32 - fy) fx 32, fy (32 - fx) 32, fy fx 32. At fx = fy = 0 OpenCV's table holds 32767, 0, 0, 1; both pick the first tap for any uint8 value.
5. Taps outside the image read 0; the result is (sum + 16384) >> 15, saturated to uint8.
OpenCV's SIMD paths do the same integer arithmetic. The products in steps 1 and 2 are rounded separately, so the replica must not fuse them: launch Triton kernels with `enable_fp_fusion=False`.

## Two kernel shapes
- Any affine (rotation, similarity alignment of faces): one Triton kernel over output pixels, matrix entries passed as `tl.float64`, `libdevice.rint` for the roundings.
- Scale and translation only (letterbox): rows and columns separate. For each input size compute one column table and one row table (two source indices and two 5-bit weights each) once, cache them on the GPU, and the warp becomes a gather and an integer weighted sum.

## Fuse what follows
Write the engine input in the same kernel: the channel swap, the HWC to CHW layout, and the normalize as a (3, 256) lookup that the CPU path's own normalize function computes over all 256 values, so the lookup equals it bit for bit.

## Check
- Compare the engine input bytes, not only the final outputs, against the production image's OpenCV on real inputs, borders included.
- Triton in the CUDA 13 images seen: set `TRITON_LIBCUDA_PATH` to the directory of the libcuda already loaded (read `/proc/self/maps`) before the first launch. Triton's `ldconfig -p` probe segfaulted on glibc 2.32 once `LD_LIBRARY_PATH` reached 128 characters.
- A cold Triton cache compiles at the first call: warm it at Init with a synthetic input.
