// Host-side launch API for the d=16 INT8-QK + FP8-PV fused attention used by the
// ColorAdjustor TensorRT plugin. Pure CUDA (no TensorRT / no torch) so it can be
// reused by both the plugin and a standalone numerical test harness.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace color_attn_d16
{

constexpr int kHeadDim = 16;

// dtype tags for the bf16/fp16 IO tensors.
enum DType
{
    kHalf = 0,
    kBF16 = 1,
};

// Bytes of scratch the launcher needs for one (B,H,Nq,Nkv) problem. Caller owns
// the allocation and passes it back to launchColorAttnD16.
size_t workspaceSize(int batch, int heads, int nq, int nkv);

// Fused attention over Q [B,H,Nq,16], K^T [B,H,16,Nkv], V [B,H,Nkv,16] producing
// O [B,H,Nq,16], all in `dtype` (kHalf/kBF16) and contiguous (row-major / kLINEAR).
//
// K is supplied already transposed to [..,16,Nkv] (the diffusers QK layout). The
// softmax scale is assumed pre-folded into Q/K (smScale defaults to 1). smooth_k
// (per-channel K mean centering) is applied internally and needs no output
// correction because it only shifts each row's logits by a constant.
//
// `bias` is an optional additive attention bias applied to the logits before
// softmax (OmniSR rel_pos_bias): [H, Nq, Nkv] in the same dtype as Q/K/V,
// broadcast over the batch dim. Pass nullptr for no bias.
//
// `ktTransposed` selects the K input layout: true = K^T [B,H,16,Nkv] (diffusers
// decomposed-attention layout), false = raw K [B,H,Nkv,16] (einsum bhid,bhjd
// layout, avoids a graph-side Transpose). K is only read by the strided
// prepasses, so this is just a stride swap.
//
// Returns cudaSuccess on success.
cudaError_t launchColorAttnD16(
    const void* q,
    const void* kt,
    const void* v,
    const void* bias,
    void* o,
    int batch,
    int heads,
    int nq,
    int nkv,
    int dtype,
    float smScale,
    bool ktTransposed,
    void* workspace,
    size_t workspaceBytes,
    cudaStream_t stream);

} // namespace color_attn_d16
