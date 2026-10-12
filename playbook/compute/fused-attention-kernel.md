---
tier: on-demand
read_when: profiling shows the symptom below
---
# Fused attention kernel tuning

- Category: compute
- Applies to: a hand-written fused attention kernel (CUDA with mma PTX or WMMA); measured on sm_86 and sm_120

## Symptom
A custom attention kernel is the largest kernel, or one is about to be written; `ncu` shows shared-memory bank conflicts, local-memory spills, or an issue-bound profile (`not_selected` the top stall, no pipe saturated).

## Technique
In the order of the gains measured:
- Keep P in registers: the QK `m16n8k16` accumulators of two adjacent 8-key tiles are exactly the A operand of a PV `mma.m16n8k16`, so pack them to half2 and feed PV with no shared-memory round trip; reduce the row max and sum with `__shfl_xor_sync` over the 4 lanes of a row. PV must then be FP16 or BF16, not FP8.
- Remove bank conflicts by layout: compute the row stride against the 32 banks (256 B gave 16-way conflicts, 272 B 2-way), and prefer a staging layout that makes `ldmatrix.trans` conflict-free over padding that costs occupancy. Count them with `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`.
- Never cap registers into spills: `--maxrregcount=64` spilled and cost 2.5x; occupancy bought with fewer registers did not pay back in either kernel.
- An issue-bound kernel gains only from fewer instructions: a template for full tiles (`nkv % BN == 0`) compiles out the per-key masks (-11%), and the dequant scale and softmax shift fuse into one FFMA (-9%).
- Issue each load next to the MMA that consumes it instead of hoisting them all; the scheduler hides its latency there.

## Effect
Head dim 16 on sm_120: 1.45x to 2.86x per shape over SageAttention padded to 64; the network 1.21x over TensorRT's own attention. Masked attention on sm_86: 35.6x over the first kernel, 1.61x over Triton, 1.91x over cuDNN SDPA.

## Side effects
Weeks of kernel work. Fragment and register layouts differed from the documentation on both GPUs: verify the mapping by experiment. The kernel is tied to one architecture and its shapes.

## Does not apply
- exp2 by polynomial emulation (FA4 style), even for half the keys: +12% to +19% on the issue-bound d=16 kernel, since one MUFU.EX2 is cheaper; `ex2.approx.f16x2` is not accelerated on consumer sm_120 (+5%).
- Relayout of P by warp shuffles instead of choosing the MMA shape: 2x slower.
- Double buffering or 8 warps that grow shared memory enough to drop blocks per SM: equal or slower.
- Block skipping for random sparsity: at 75% random sparsity almost no 64-key tile is fully masked, removing the skip changed nothing, and dense flash-attn without a mask stayed 1.16x faster. Skipping needs block-structured masks.
- Software-pipelined ldmatrix, occupancy via sub-block softmax, hand strength reduction of addresses (identical SASS): no gain or slower.
- A softmax shift frozen from the first key tile instead of the running max: -15% on synthetic data, but it overflowed FP32 on real inputs, and the clamp that made it safe ran as slow as the running max.
- `__launch_bounds__` with a minimum of 6 blocks per SM: 12 registers spilled, +3.5%.

## Evidence
- ✓ 2026-09: image color-transfer UNet, D=16 attention over up to 147k keys, INT8 QK and FP16 PV TensorRT plugin, RTX 5090: 1.45x to 2.86x per shape, network 208 -> 171 ms.
- ✓ 2026-09: masked attention kernel study, B=16 H=12 N=512 D=64, 75% random sparsity, RTX 3080: 16.6 -> 0.466 ms, 1.61x over Triton.
