---
tier: on-demand
read_when: profiling shows the symptom below
---
# Per-element tiny matrix products as elementwise sums

- Category: compute
- Applies to: PyTorch code with einsum or bmm over per-pixel or per-point matrices of 2 or 3 rows (Jacobians, J^T W J, J^T r), eager or captured in CUDA graphs; any GPU, most on the L20

## Symptom
A kernel profile shows cuBLAS batched GEMM or GEMV kernels (gemmSN, gemvx, sgemm 128x64 or 128x128) inside per-element math. cuBLAS picks them per GPU: one J^T J einsum ran as a 128x128 SGEMM taking 14.9 ms a request on the L20, while the RTX 5090 picked a 128x32 kernel at 0.26 ms, so the development GPU hid it.

## Technique
Replace each per-element product over an inner dimension of 2 or 3 by a broadcast product summed over that dimension (two elementwise kernels), and the weighted sums over all elements of J^T J and J^T r by one weighted elementwise reduction. FP32 throughout; check against the original on real inputs (relative error about 1e-7). Probe the kernels on the production GPU, since another GPU may hide the slow choice.
One matmul over the flattened element dimension, (w J)^T J, was as fast on the L20, but it keeps a cuBLAS kernel whose choice differs per GPU.

## Effect
L20: -15.1 ms (-48%) for the Hessian and gradient sums, then -3.4 ms (-21%) for the Jacobians, on a request of 31 ms at the start; RTX 5090 -2.9 and -1.9 ms. Outputs L0 to L1 (summation order). Inside a TensorRT engine of the same solver, reverting it together with on-device constants cost +0.48 ms (L20) and +1.66 ms (RTX 5090).

## Side effects
None measured: memory equal or lower, quality and solver step counts unchanged.

## Does not apply
- None measured yet; einsums off the production path (unused branches) were left as they were.

## Evidence
- ✓ 2026-10: camera-calibration CNN-transformer service with an iterative least-squares refinement, batch 1, about 150k pixels: per-pixel J^T J and J^T r einsums as one weighted reduction: L20 31.2 -> 16.1 ms (-48.4%, three runs each), RTX 5090 A/B -2.88 ms (-17.6%), L0 to L1.
- ✓ 2026-10: the same, the Jacobian einsums (2x2, 1x2 by 2x2, 1x2 by 2x3 per pixel) as broadcast sums: L20 A/B 16.12 -> 12.71 ms (-21.1%), solver kernel time 6.14 -> 2.50 ms; RTX 5090 -1.87 ms (-13.9%), L0 to L1.
