// d=16 INT8-QK (m16n8k16.s8.s8 + smooth_k) + FP8-E4M3-PV fused flash attention.
//
// Generalized from tools/bench_d16_attention_cuda_e2e.py to strided multi-head
// [B,H,N,16] bf16/fp16 IO so it can back the ColorAdjustor TensorRT plugin.
//
// IO contract (matches the diffusers decomposed-attention subgraph we replace):
//   Q   : [B, H, Nq, 16]   contiguous
//   K^T : [B, H, 16, Nkv]  contiguous (K already transposed for the QK MatMul)
//   V   : [B, H, Nkv, 16]  contiguous
//   O   : [B, H, Nq, 16]   contiguous
// The softmax scale is pre-folded into Q/K by the kept graph Muls, so smScale=1.

#include "color_attn_d16_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace color_attn_d16
{
namespace
{

constexpr int kD = kHeadDim;     // 16
constexpr float kNegInf = -3.4028234663852886e38f;

__device__ __forceinline__ float load_as_float(const void* ptr, int64_t idx, int dtype)
{
    if (dtype == kHalf)
    {
        return __half2float(reinterpret_cast<const half*>(ptr)[idx]);
    }
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
}

__device__ __forceinline__ void store_as_dtype(void* ptr, int64_t idx, float v, int dtype)
{
    if (dtype == kHalf)
    {
        reinterpret_cast<half*>(ptr)[idx] = __float2half_rn(v);
    }
    else
    {
        reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16_rn(v);
    }
}

// Pack two softmax probabilities into one 32-bit register as a half2/bf162 pair, in the
// element type the PV m16n8k16 MMA consumes (kHalf -> .f16.f16, else -> .bf16.bf16).
template <bool kHalf>
__device__ __forceinline__ uint32_t pack_p2(float a, float b)
{
    if constexpr (kHalf)
    {
        const __half2 h = __floats2half2_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&h);
    }
    else
    {
        const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&h);
    }
}

// exp2 for the softmax logit. p is produced in fp32, so it overflows fp32 (arg >= 128) -> inf -> NaN
// regardless of the P storage type. The fp16 path uses a running max (arg <= 0, never overflows) so it
// needs no clamp. The bf16 frozen-shift path freezes M at the tile-0 max, so a later tile's score can
// beat M by >128 log2 -> fp32 overflow; clamp the arg to keep p and Sum(p) finite. 100 keeps p<=2^100,
// so Sum over up to ~2^18 keys stays < 2^128 (fp32 max). Only keys that beat the tile-0 max by >100
// log2 saturate -- they already dominate the row, so the softmax stays accurate where it matters.
// (A cvt.satfinite pack was tried to move this off the FMNMX pipe: it saturates only to bf16max=2^127,
// which is too high -- the fp32 Sum(p)/PV accumulation over ~2^18 keys then overflows -> NaN. The clamp
// ceiling must be low (2^100), and only fminf can pick the ceiling, so the FMNMX stays.)
template <bool kClamp>
__device__ __forceinline__ float exp2_logit(float arg)
{
    if constexpr (kClamp)
    {
        return exp2f(fminf(arg, 100.0f));
    }
    else
    {
        return exp2f(arg);
    }
}

// ldmatrix.x2: one instruction loads two 8x8 b16 matrices (= two 8-key x 16-int8
// tiles) into the fragment layout the m16n8k16 B operand expects. For a d=16 row
// (16 bytes) this is bank-conflict-free without any swizzle. Lane L (<16) supplies
// the smem address of row L; r0 holds keys [base..base+7], r1 holds [base+8..+15].
__device__ __forceinline__ void ldmatrix_x2_u8(uint32_t& r0, uint32_t& r1, const uint8_t* smem_row)
{
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

// ldmatrix.x2.trans for an fp16/bf16 V tile stored [key][d] (= [contraction][free] row-major).
// The on-the-fly transpose yields the m16n8k16 B operand {b0=keys 0-7 half, b1=keys 8-15 half} for
// a 16-key block, reading V straight from the cp.async staging with NO explicit transpose pass.
// This is conflict-free only because V is staged de-interleaved into two d-half planes [nn][key][8]:
// each key's 16-B d-half makes consecutive keys 4 banks apart, so the 8 keys of an ldmatrix tile
// span all 32 banks. (A tight [key][d16] staging strides keys 8 banks apart -> 2-way conflict, the
// reason the naive no-transpose load was slower; d=16 is too narrow to fix that with a within-row
// swizzle the way TRT's d=64 MHA does, so we split the d-halves instead.)
__device__ __forceinline__ void ldmatrix_x2_trans_b16(uint32_t& r0, uint32_t& r1, const uint16_t* smem)
{
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

// Async copy global->shared (no register staging). src_bytes < 16/4 zero-fills the rest, and
// src_bytes==0 issues no global read (safe predication for out-of-range keys, which are masked
// downstream by the softmax validity check anyway).
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, int src_bytes)
{
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_4(void* smem, const void* gmem, int src_bytes)
{
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ uint32_t pack_quant4_s8(float v0, float v1, float v2, float v3, float s)
{
    const int q0 = max(-127, min(127, __float2int_rn(v0 / s)));
    const int q1 = max(-127, min(127, __float2int_rn(v1 / s)));
    const int q2 = max(-127, min(127, __float2int_rn(v2 / s)));
    const int q3 = max(-127, min(127, __float2int_rn(v3 / s)));
    return static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q0)))
        | (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q1))) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q2))) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q3))) << 24);
}

// Inline Q quantization for the fused kernel. The four lanes in a row-group own
// dims {0..3}, {4..7}, {8..11}, {12..15}; reduce amax within that group and
// return the same packed int8 fragment layout previously loaded from q16.
__device__ __forceinline__ uint32_t load_quant_q_frag(
    const void* q, int64_t base, bool valid, int dtype, float* qscale)
{
    const int kk = 4 * (threadIdx.x & 3);
    const float v0 = valid ? load_as_float(q, base + kk + 0, dtype) : 0.0f;
    const float v1 = valid ? load_as_float(q, base + kk + 1, dtype) : 0.0f;
    const float v2 = valid ? load_as_float(q, base + kk + 2, dtype) : 0.0f;
    const float v3 = valid ? load_as_float(q, base + kk + 3, dtype) : 0.0f;
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float s = fmaxf(amax / 127.0f, 1.0e-6f);
    *qscale = valid ? s : 0.0f;
    return valid ? pack_quant4_s8(v0, v1, v2, v3, s) : 0u;
}

// ---- Prepasses (one (b,h) per blockIdx.y) ------------------------------------------------

// smooth_k: per-channel K mean over the key axis. One block per (channel d, b, h).
// K^T element (d, kv) lives at base + d*sKd + kv*sKn (kv contiguous).
__global__ void kMeanKernel(const void* __restrict__ kt, float* __restrict__ km, int nkv, int heads,
    int bhTotal, int64_t sB, int64_t sH, int64_t sKd, int64_t sKn, int dtype)
{
    int d = blockIdx.x;
    // bh split across grid y*z: OmniSR window batches push B*H past the 65535 grid.y limit.
    int bh = blockIdx.y + gridDim.y * blockIdx.z;
    if (bh >= bhTotal)
    {
        return;
    }
    int b = bh / heads;
    int h = bh % heads;
    int64_t base = static_cast<int64_t>(b) * sB + static_cast<int64_t>(h) * sH + static_cast<int64_t>(d) * sKd;
    float sum = 0.0f;
    for (int kv = threadIdx.x; kv < nkv; kv += blockDim.x)
    {
        sum += load_as_float(kt, base + static_cast<int64_t>(kv) * sKn, dtype);
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        km[static_cast<int64_t>(bh) * kD + d] = sh[0] / static_cast<float>(nkv);
    }
}

// K signed int8 with smooth_k centering + per-block (kBN-key) scale (sage-style granularity).
__global__ void quantKKernel(const void* __restrict__ kt, const float* __restrict__ km,
    uint8_t* __restrict__ ku16, float* __restrict__ ks, int nkv, int heads, int bhTotal, int64_t sB, int64_t sH,
    int64_t sKd, int64_t sKn, int dtype)
{
    // One block (kBN threads) per kBN-key block -> a single per-block K scale (sage parity:
    // per-block-of-64 quant). The attn kernel streams keys in kBN tiles, so each tile maps to
    // exactly one scale -> ks collapses from per-key to one scalar per tile.
    const int blk = blockIdx.x;
    const int bh = blockIdx.y + gridDim.y * blockIdx.z;  // bh split across y*z (grid.y limit)
    if (bh >= bhTotal)
    {
        return;
    }
    const int kv = blk * blockDim.x + threadIdx.x;  // blockDim.x == kBN
    const int b = bh / heads;
    const int h = bh % heads;
    const float* kmh = km + static_cast<int64_t>(bh) * kD;
    const bool valid = kv < nkv;
    int64_t base = static_cast<int64_t>(b) * sB + static_cast<int64_t>(h) * sH + static_cast<int64_t>(kv) * sKn;
    float kc[kD];
    float amax = 0.0f;
#pragma unroll
    for (int d = 0; d < kD; ++d)
    {
        float v = valid ? load_as_float(kt, base + static_cast<int64_t>(d) * sKd, dtype) - kmh[d] : 0.0f;
        kc[d] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    // Block-wide max over the kBN keys in this block.
    __shared__ float sh[256];
    sh[threadIdx.x] = amax;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float s = fmaxf(sh[0] / 127.0f, 1.0e-6f);
    if (valid)
    {
        int64_t outb = (static_cast<int64_t>(bh) * nkv + kv) * kD;
#pragma unroll
        for (int d = 0; d < kD; ++d)
        {
            int qi = max(-127, min(127, __float2int_rn(kc[d] / s)));
            ku16[outb + d] = static_cast<uint8_t>(static_cast<int8_t>(qi));
        }
    }
    if (threadIdx.x == 0)
    {
        ks[static_cast<int64_t>(bh) * gridDim.x + blk] = s;
    }
}

// (V prepasses removed: V is now streamed raw fp16/bf16 directly into the PV m16n8k16 MMA.)

// ---- Fused attention -----------------------------------------------------------------------
// One block owns kBlockM = kWarps*16 query rows (warp w -> rows [w*16, w*16+16)). Each streamed
// kBN-key tile is loaded into shared memory ONCE per block (cooperatively by all warps) and
// reused by every warp, so global K/V traffic scales with Nq/kBlockM instead of Nq/16.
// QK: m16n8k16.s8.s8 (signed, no offset). PV: m16n8k32.f32.e4m3.e4m3. Online softmax;
// score / probability tiles are per-warp, the K / V tiles are block-shared.
constexpr int kWarps = 4;
constexpr int kBN = 64;                 // streamed keys per tile (kBN/8 m16n8k16 QK sub-tiles)
constexpr int kQT = 2;                  // query 16-row tiles handled per warp (arithmetic intensity)
constexpr int kBlockM = kWarps * 16 * kQT;  // query rows per block
constexpr int kNT = kBN / 8;            // number of m16n8k16 QK sub-tiles
constexpr float kLog2e = 1.4426950408889634f;

// Cooperatively prefetch one kBN-key K / V-staging tile into the given smem buffers.
// (K scale is now one scalar per kBN tile -> read directly from global, not staged.)
// K is int8 (16 B/key); V is raw fp16/bf16 streamed straight from the input tensor (no fp8 quant
// prepass). Each V key row is 16 elems x 2 B = 32 B = two cp.async.16, but we de-interleave the two
// d-halves into separate planes so ldmatrix.x2.trans can read V conflict-free without a transpose
// pass: v_buf is [nn=2][key=kBN][d8], i.e. d0-7 of key k at k*8, d8-15 at kBN*8 + k*8. Each key's
// 16-B d-half puts consecutive keys 4 banks apart -> the 8 keys of one ldmatrix tile cover all 32
// banks. (A tight [key][d16] staging strides keys 8 banks apart -> 2-way conflict on the trans read.)
__device__ __forceinline__ void cp_async_load_kv(uint8_t* ku_buf, uint16_t* v_buf,
    const uint8_t* ku16p, const uint8_t* vp, int kbase, int n)
{
    for (int key_local = threadIdx.x; key_local < kBN; key_local += blockDim.x)
    {
        const int gkey = kbase + key_local;
        const int sb = gkey < n ? 16 : 0;
        cp_async_16(&ku_buf[key_local * 16], &ku16p[static_cast<int64_t>(gkey) * 16], sb);
        cp_async_16(&v_buf[key_local * 8], &vp[static_cast<int64_t>(gkey) * 32], sb);          // d0-7  -> plane 0
        cp_async_16(&v_buf[kBN * 8 + key_local * 8], &vp[static_cast<int64_t>(gkey) * 32 + 16], sb); // d8-15 -> plane 1
    }
}

// kFullKeys=true is dispatched when nkv % kBN == 0 (every key tile is exactly full, which holds for
// all real ColorAdjustor shapes). It compiles out the per-key (key_base+c0 < n) bounds masking
// — ISETP + FSEL per logit — which is pure waste on an issue-bound kernel. Ragged nkv uses false.
// kHasBias adds an additive attention bias (OmniSR rel_pos_bias) to the logits before softmax:
// bias [H, Nq, Nkv] in the IO dtype, broadcast over batch. The bias is folded into the raw QK
// dots right after the MMA (logit_log2 = dot*ql + bias*log2e), which forfeits the deferred-dequant
// trick (max/exp2 then run at scale 1) but keeps the rest of the pipeline unchanged. OmniSR runs
// tiny windows (Nq=Nkv=64, a single key tile), so the extra per-key global bias loads are cheap.
template <bool kFullKeys, bool kHalf, bool kHasBias>
__global__ void __launch_bounds__(kWarps * 32, 5) fusedAttnKernel(const void* __restrict__ q,
    const uint8_t* __restrict__ ku16, const float* __restrict__ ks,
    const void* __restrict__ v, const void* __restrict__ bias, void* __restrict__ o, int dtype, int m,
    int n, int heads, int bhTotal, float sm_scale, int64_t sOB, int64_t sOH, int64_t sON)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;  // 0..kWarps-1
    const int group = lane >> 2;        // 0..7 -> rows {group, group+8}
    const int tid = lane & 3;           // 0..3 -> cols {2*tid, 2*tid+1}
    const int bh = blockIdx.y + gridDim.y * blockIdx.z;  // bh split across y*z (grid.y limit)
    const int block_row_base = blockIdx.x * kBlockM;
    if (bh >= bhTotal || block_row_base >= m)
    {
        return;
    }
    const int warp_row_base = block_row_base + warp * 16 * kQT;
    const int b = bh / heads;
    const int h = bh % heads;

    const uint8_t* ku16p = ku16 + static_cast<int64_t>(bh) * n * kD;
    const int64_t qbase_bh = static_cast<int64_t>(bh) * m * kD;
    // Raw V (contiguous [B,H,Nkv,16] fp16/bf16) streamed directly: 2 bytes per element.
    const uint8_t* vp = reinterpret_cast<const uint8_t*>(v) + static_cast<int64_t>(bh) * n * kD * 2;
    const float* ksp = ks + static_cast<int64_t>(bh) * ((n + kBN - 1) / kBN);  // per-block K scales
    // Bias is [H, Nq, Nkv] shared across the batch dim (OmniSR windows all use the same table).
    const int64_t bias_base = kHasBias ? static_cast<int64_t>(h) * m * n : 0;
    const int64_t obase = static_cast<int64_t>(b) * sOB + static_cast<int64_t>(h) * sOH;

    __shared__ __align__(16) uint8_t ku_smem[2][kBN * 16];      // double-buffered K tile [key][d] (int8)
    __shared__ __align__(16) uint16_t vstage[2][kBN * 16];      // double-buffered V staging [nn][key][d8] (fp16/bf16)
    // No p_smem: the QK m16n8k16 output fragment (two adjacent 8-key sub-tiles) already lands in the
    // exact PV m16n8k16 A-operand layout, so P feeds PV with no smem round-trip.

    // Each lane owns two query rows per 16-row tile: {group, group+8}. A warp covers kQT tiles.
    int row_lo[kQT], row_hi[kQT];
    uint32_t qa0[kQT], qa1[kQT];
    float qsc_lo[kQT], qsc_hi[kQT];
    float m_lo[kQT], m_hi[kQT], l_lo[kQT], l_hi[kQT];
    float og[kQT][8];  // O accumulator in PV C-layout, per query tile
#pragma unroll
    for (int qt = 0; qt < kQT; ++qt)
    {
        const int rb = warp_row_base + qt * 16;
        row_lo[qt] = rb + group;
        row_hi[qt] = rb + group + 8;
        float qs_lo = 0.0f, qs_hi = 0.0f;
        qa0[qt] = load_quant_q_frag(q, qbase_bh + static_cast<int64_t>(row_lo[qt]) * kD, row_lo[qt] < m, dtype, &qs_lo);
        qa1[qt] = load_quant_q_frag(q, qbase_bh + static_cast<int64_t>(row_hi[qt]) * kD, row_hi[qt] < m, dtype, &qs_hi);
        qsc_lo[qt] = qs_lo * sm_scale * kLog2e;
        qsc_hi[qt] = qs_hi * sm_scale * kLog2e;
        m_lo[qt] = kNegInf;
        m_hi[qt] = kNegInf;
        l_lo[qt] = 0.0f;
        l_hi[qt] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            og[qt][i] = 0.0f;
        }
    }

    const int nTiles = (n + kBN - 1) / kBN;
    // Prologue: kick off the async copy of tile 0 into buffer 0.
    cp_async_load_kv(ku_smem[0], vstage[0], ku16p, vp, 0, n);
    cp_async_commit();

    for (int it = 0; it < nTiles; ++it)
    {
        const int cur = it & 1;
        const int key_base = it * kBN;
        // Prefetch the next tile into the other buffer so its global load overlaps this tile's
        // compute, then wait until the current tile's copy has landed.
        if (it + 1 < nTiles)
        {
            cp_async_load_kv(ku_smem[1 - cur], vstage[1 - cur], ku16p, vp, (it + 1) * kBN, n);
            cp_async_commit();
            cp_async_wait<1>();
        }
        else
        {
            cp_async_wait<0>();
        }
        // No transpose pass: V is staged de-interleaved as [nn][key][d8], read directly by
        // ldmatrix.x2.trans in the PV loop. One barrier makes the cross-warp cp.async tiles visible.
        __syncthreads();  // ku_smem[cur] + vstage[cur] visible for cross-warp/lane ldmatrix

        const float ks_tile = ksp[it];  // single per-block K scale for all keys in this tile
        // Load K B-fragments once via ldmatrix.x2 (B operand is query-independent -> reused for all qt).
        uint32_t rkb[kNT];
#pragma unroll
        for (int kb = 0; kb < kBN; kb += 16)
        {
            ldmatrix_x2_u8(rkb[kb / 8], rkb[kb / 8 + 1], &ku_smem[cur][(kb + (lane & 15)) * kD]);
        }

#pragma unroll
        for (int qt = 0; qt < kQT; ++qt)
        {
            // QK -> registers (no score smem); dequant straight into log2-domain logits.
            // Per-block K scale folds into the per-query factor once (no per-key ks reads/muls).
            const float ql = qsc_lo[qt] * ks_tile;
            const float qh = qsc_hi[qt] * ks_tile;
            // With bias the dequant + bias fold happens right after the MMA (dl holds log2-domain
            // logits already), so the deferred max/exp2 scale collapses to 1.
            const float sl = kHasBias ? 1.0f : ql;
            const float sh = kHasBias ? 1.0f : qh;
            // Raw int->float QK dot products. The ×ql dequant is *deferred* to the loop-2 FFMA
            // (since ql,qh > 0, the tile row-max of d·ql equals ql·max(d) -> scale once per row,
            // not per key). Ragged invalid keys carry a finite -1e30 sentinel so the loop-2 FFMA
            // underflows them to exp2(-inf)=0 with no per-key branch (and no 0·inf NaN).
            float dl0[kNT], dl1[kNT], dh0[kNT], dh1[kNT];
            float lm_lo = kNegInf, lm_hi = kNegInf;
#pragma unroll
            for (int t = 0; t < kNT; ++t)
            {
                const uint32_t b0 = rkb[t];
                int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
                    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                    : "r"(qa0[qt]), "r"(qa1[qt]), "r"(b0));
                dl0[t] = static_cast<float>(d0);
                dl1[t] = static_cast<float>(d1);
                dh0[t] = static_cast<float>(d2);
                dh1[t] = static_cast<float>(d3);
                if constexpr (kHasBias)
                {
                    const int c0 = key_base + t * 8 + 2 * tid;
                    const bool k0ok = kFullKeys || (c0 < n);
                    const bool k1ok = kFullKeys || (c0 + 1 < n);
                    const int64_t bb_lo = bias_base + static_cast<int64_t>(row_lo[qt]) * n;
                    const int64_t bb_hi = bias_base + static_cast<int64_t>(row_hi[qt]) * n;
                    const bool row_lo_ok = row_lo[qt] < m;
                    const bool row_hi_ok = row_hi[qt] < m;
                    const float b_lo0 = (row_lo_ok && k0ok) ? load_as_float(bias, bb_lo + c0, dtype) * kLog2e : 0.0f;
                    const float b_lo1 = (row_lo_ok && k1ok) ? load_as_float(bias, bb_lo + c0 + 1, dtype) * kLog2e : 0.0f;
                    const float b_hi0 = (row_hi_ok && k0ok) ? load_as_float(bias, bb_hi + c0, dtype) * kLog2e : 0.0f;
                    const float b_hi1 = (row_hi_ok && k1ok) ? load_as_float(bias, bb_hi + c0 + 1, dtype) * kLog2e : 0.0f;
                    dl0[t] = fmaf(dl0[t], ql, b_lo0);
                    dl1[t] = fmaf(dl1[t], ql, b_lo1);
                    dh0[t] = fmaf(dh0[t], qh, b_hi0);
                    dh1[t] = fmaf(dh1[t], qh, b_hi1);
                }
                if (!kFullKeys)
                {
                    const int c0 = t * 8 + 2 * tid;
                    if ((key_base + c0) >= n)
                    {
                        dl0[t] = -1e30f;
                        dh0[t] = -1e30f;
                    }
                    if ((key_base + c0 + 1) >= n)
                    {
                        dl1[t] = -1e30f;
                        dh1[t] = -1e30f;
                    }
                }
                if (kHalf || it == 0)  // fp16 needs the max every tile; bf16 freezes tile-0's
                {
                    lm_lo = fmaxf(lm_lo, fmaxf(dl0[t], dl1[t]));
                    lm_hi = fmaxf(lm_hi, fmaxf(dh0[t], dh1[t]));
                }
            }
            // Softmax shift. Two regimes by P dtype:
            //  - bf16 (route (C)): compute the row-max ONCE on tile 0 and *freeze* it as a constant
            //    shift M for every tile. M cancels in O = Σpv/Σp, so a constant M needs no per-tile
            //    rescale — tiles 1..N skip the max shfl-reduction, the rescale exp2, and the og*=al
            //    (the bulk of the online-softmax scaffolding, paid on each of down1's 2304 tiles).
            //    Safe only for bf16: its fp32-like exponent range (2^±126) cannot overflow even when a
            //    later tile's score exceeds the frozen tile-0 max.
            //  - fp16: a frozen shift is UNSAFE — a later tile's score can beat the tile-0 max by >16
            //    log2, p = exp2(score − M) overflows fp16 P (max 2^16) → inf → NaN. The running max +
            //    per-tile rescale keeps the shift current (args ≤ 0, no overflow). Note this rescale IS
            //    fp16's overflow guard: a frozen-M + exp2-arg clamp was tried and *lost* (the per-key
            //    clamp adds as much math-pipe work as the rescale it removes; see README).
            float nm_lo, nm_hi, al_lo = 1.0f, al_hi = 1.0f;
            if constexpr (kHalf)
            {
                lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, 1));
                lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, 2));
                lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, 1));
                lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, 2));
                lm_lo *= sl;
                lm_hi *= sh;
                nm_lo = fmaxf(m_lo[qt], lm_lo);
                nm_hi = fmaxf(m_hi[qt], lm_hi);
                al_lo = m_lo[qt] > kNegInf ? exp2f(m_lo[qt] - nm_lo) : 0.0f;
                al_hi = m_hi[qt] > kNegInf ? exp2f(m_hi[qt] - nm_hi) : 0.0f;
                og[qt][0] *= al_lo;
                og[qt][1] *= al_lo;
                og[qt][4] *= al_lo;
                og[qt][5] *= al_lo;
                og[qt][2] *= al_hi;
                og[qt][3] *= al_hi;
                og[qt][6] *= al_hi;
                og[qt][7] *= al_hi;
            }
            else if (it == 0)
            {
                lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, 1));
                lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, 2));
                lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, 1));
                lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, 2));
                nm_lo = lm_lo * sl;  // raw-dot tile-0 max -> log2 domain (already log2 when biased)
                nm_hi = lm_hi * sh;
                m_lo[qt] = nm_lo;    // freeze the fixed shift for all later tiles
                m_hi[qt] = nm_hi;
            }
            else
            {
                nm_lo = m_lo[qt];    // reuse frozen shift; no max, no rescale
                nm_hi = m_hi[qt];
            }

            // p = exp2(logit - m), packed into the PV A operand and fed straight to the MMA with no
            // smem round-trip. KEY INSIGHT: the QK m16n8k16 output fragment for two adjacent 8-key
            // sub-tiles {t0=2b, t1=2b+1} is *already* the m16n8k16 A operand of the 16-key block b:
            //   a0 = P[q=group  ][block keys 2tid,2tid+1]  (sub-tile t0, keys 0-7)
            //   a1 = P[q=group+8][block keys 2tid,2tid+1]
            //   a2 = P[q=group  ][block keys 2tid+8,+9]   (sub-tile t1, keys 8-15)
            //   a3 = P[q=group+8][block keys 2tid+8,+9]
            // all this thread's own registers. V (B operand) is trans-loaded per block. The f16/bf16 MMA
            // accumulates directly into the f32 og (no fp8 P, no two-level f16 partial, no STS/LDSM).
            float ls_lo = 0.0f, ls_hi = 0.0f;
#pragma unroll
            for (int b16 = 0; b16 < kNT / 2; ++b16)
            {
                const int co = b16 * 16;
                const int t0 = 2 * b16;
                const int t1 = t0 + 1;
                // Single FFMA per key: exp2(dot·ql − m). Invalid ragged keys -> exp2(-inf)=0.
                // bf16 freezes M -> arg can overflow fp32 -> clamp (kClamp=!kHalf); fp16 running max -> none.
                const float p0_lo0 = exp2_logit<!kHalf>(fmaf(dl0[t0], sl, -nm_lo));
                const float p0_lo1 = exp2_logit<!kHalf>(fmaf(dl1[t0], sl, -nm_lo));
                const float p0_hi0 = exp2_logit<!kHalf>(fmaf(dh0[t0], sh, -nm_hi));
                const float p0_hi1 = exp2_logit<!kHalf>(fmaf(dh1[t0], sh, -nm_hi));
                const float p1_lo0 = exp2_logit<!kHalf>(fmaf(dl0[t1], sl, -nm_lo));
                const float p1_lo1 = exp2_logit<!kHalf>(fmaf(dl1[t1], sl, -nm_lo));
                const float p1_hi0 = exp2_logit<!kHalf>(fmaf(dh0[t1], sh, -nm_hi));
                const float p1_hi1 = exp2_logit<!kHalf>(fmaf(dh1[t1], sh, -nm_hi));
                ls_lo += p0_lo0 + p0_lo1 + p1_lo0 + p1_lo1;
                ls_hi += p0_hi0 + p0_hi1 + p1_hi0 + p1_hi1;
                const uint32_t a0 = pack_p2<kHalf>(p0_lo0, p0_lo1);
                const uint32_t a1 = pack_p2<kHalf>(p0_hi0, p0_hi1);
                const uint32_t a2 = pack_p2<kHalf>(p1_lo0, p1_lo1);
                const uint32_t a3 = pack_p2<kHalf>(p1_hi0, p1_hi1);
#pragma unroll
                for (int nn = 0; nn < 2; ++nn)
                {
                    // V (B operand) loaded per (block, d-half) via ldmatrix.x2.trans straight from the
                    // de-interleaved staging [nn][key][d8]: lane L<16 points at key (co+L)'s d-half in
                    // plane nn; the transpose produces {b0=keys 0-7, b1=keys 8-15} of the B operand with
                    // no explicit transpose pass and no bank conflicts (16-B key stride = 4 banks).
                    uint32_t b0, b1;
                    ldmatrix_x2_trans_b16(b0, b1, &vstage[cur][nn * kBN * 8 + (co + (lane & 15)) * 8]);
                    if constexpr (kHalf)
                    {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(og[qt][nn * 4 + 0]), "+f"(og[qt][nn * 4 + 1]),
                              "+f"(og[qt][nn * 4 + 2]), "+f"(og[qt][nn * 4 + 3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    }
                    else
                    {
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                            : "+f"(og[qt][nn * 4 + 0]), "+f"(og[qt][nn * 4 + 1]),
                              "+f"(og[qt][nn * 4 + 2]), "+f"(og[qt][nn * 4 + 3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    }
                }
            }
            // Row-sum reduction across the 4 tid lanes within each group.
            ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, 1);
            ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, 2);
            ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, 1);
            ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, 2);
            if constexpr (kHalf)
            {
                l_lo[qt] = l_lo[qt] * al_lo + ls_lo;  // fp16: rescale denominator by running-max correction
                l_hi[qt] = l_hi[qt] * al_hi + ls_hi;
                m_lo[qt] = nm_lo;
                m_hi[qt] = nm_hi;
            }
            else
            {
                l_lo[qt] += ls_lo;  // bf16: frozen shift -> no per-tile denominator rescale
                l_hi[qt] += ls_hi;
            }
        }
        // All reads of ku_smem[cur] / vstage[cur] done before the next iteration's prefetch
        // overwrites the reused buffer.
        __syncthreads();
    }

    // Epilogue: O = (sum_k P_k V_k) / l = og / l. V is no longer mean-shifted / scaled (raw fp16/bf16
    // operand straight into the PV MMA), so the old * Vscale + Vmean correction is gone.
#pragma unroll
    for (int qt = 0; qt < kQT; ++qt)
    {
        const float inv_l_lo = row_lo[qt] < m ? 1.0f / l_lo[qt] : 0.0f;
        const float inv_l_hi = row_hi[qt] < m ? 1.0f / l_hi[qt] : 0.0f;
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
        {
            const int d0c = nn * 8 + 2 * tid;
            const int d1c = d0c + 1;
            if (row_lo[qt] < m)
            {
                store_as_dtype(o, obase + static_cast<int64_t>(row_lo[qt]) * sON + d0c, og[qt][nn * 4 + 0] * inv_l_lo, dtype);
                store_as_dtype(o, obase + static_cast<int64_t>(row_lo[qt]) * sON + d1c, og[qt][nn * 4 + 1] * inv_l_lo, dtype);
            }
            if (row_hi[qt] < m)
            {
                store_as_dtype(o, obase + static_cast<int64_t>(row_hi[qt]) * sON + d0c, og[qt][nn * 4 + 2] * inv_l_hi, dtype);
                store_as_dtype(o, obase + static_cast<int64_t>(row_hi[qt]) * sON + d1c, og[qt][nn * 4 + 3] * inv_l_hi, dtype);
            }
        }
    }
}

// ---- Small-window fused path (nq,nkv <= 64: OmniSR 8x8 windows) ----------------------------
// The general path pays two standalone K prepass launches (mean + quant) and a kBlockM=128
// block granularity. For OmniSR that means 131072 (b,h) windows of 64x64 each: the prepass
// blocks are launch-bound and half of every attention block runs empty rows. This kernel
// instead gives each (b,h) window ONE block (bh rides gridDim.x, no y*z split needed), fuses
// the K mean/centering/quant inline in smem, and — since the whole key range is a single
// tile — computes the exact row max up front, so the online-softmax scaffolding (running max,
// rescale, frozen-shift clamp) disappears entirely for both fp16 and bf16.
constexpr int kSN = 64;           // max keys (one tile)
constexpr int kSWarps = 4;        // 4 warps x 16 rows = 64 query rows
constexpr int kSNT = kSN / 8;     // m16n8k16 QK sub-tiles

template <bool kFullKeys, bool kHalf, bool kHasBias>
__global__ void __launch_bounds__(kSWarps * 32) fusedAttnSmallKernel(const void* __restrict__ q,
    const void* __restrict__ kt, const void* __restrict__ v, const void* __restrict__ bias,
    void* __restrict__ o, int dtype, int m, int n, int heads, float sm_scale, int64_t sKH,
    int64_t sKd, int64_t sKn)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = lane >> 2;
    const int tid = lane & 3;
    const int bh = blockIdx.x;
    const int h = bh % heads;

    const int64_t qbase = static_cast<int64_t>(bh) * m * kD;
    const int64_t kbase = static_cast<int64_t>(bh) * sKH;  // contiguous [B,H,..] -> bh*H-stride
    const uint8_t* vp = reinterpret_cast<const uint8_t*>(v) + static_cast<int64_t>(bh) * n * kD * 2;
    const int64_t bias_base = kHasBias ? static_cast<int64_t>(h) * m * n : 0;
    const int64_t obase = static_cast<int64_t>(bh) * m * kD;

    __shared__ float kraw[kSN * kD];                       // centered-K staging (fp32)
    __shared__ float kstat[kSN];                           // channel means (16) then per-key amax (64)
    __shared__ __align__(16) uint8_t ku_smem[kSN * kD];    // int8 K tile [key][d]
    __shared__ __align__(16) uint16_t vstage[kSN * kD];    // V staging, de-interleaved [nn][key][d8]
    __shared__ float ksc;                                  // single per-tile K scale

    // Kick off the async V copy first so it overlaps the K mean/quant work below.
    for (int key = threadIdx.x; key < kSN; key += blockDim.x)
    {
        const int sb = key < n ? 16 : 0;
        cp_async_16(&vstage[key * 8], &vp[static_cast<int64_t>(key) * 32], sb);           // d0-7  -> plane 0
        cp_async_16(&vstage[kSN * 8 + key * 8], &vp[static_cast<int64_t>(key) * 32 + 16], sb); // d8-15 -> plane 1
    }
    cp_async_commit();

    // K raw load: thread t (<64) owns key t (strided gather works for both K^T and raw K layouts).
    if (threadIdx.x < kSN)
    {
        const int key = threadIdx.x;
        const bool valid = key < n;
#pragma unroll
        for (int d = 0; d < kD; ++d)
        {
            kraw[key * kD + d]
                = valid ? load_as_float(kt, kbase + static_cast<int64_t>(d) * sKd + static_cast<int64_t>(key) * sKn, dtype)
                        : 0.0f;
        }
    }
    __syncthreads();
    // smooth_k channel means (over the valid keys only).
    if (threadIdx.x < kD)
    {
        float s = 0.0f;
        for (int key = 0; key < n; ++key)
        {
            s += kraw[key * kD + threadIdx.x];
        }
        kstat[threadIdx.x] = s / static_cast<float>(n);
    }
    __syncthreads();
    // Center + per-key amax; then one block-wide amax -> single per-tile K scale (kBN==kSN parity
    // with the general path's per-64-key-block scale).
    float kmean_reg[kD];
#pragma unroll
    for (int d = 0; d < kD; ++d)
    {
        kmean_reg[d] = kstat[d];
    }
    __syncthreads();  // kstat re-used for per-key amax below
    if (threadIdx.x < kSN)
    {
        const int key = threadIdx.x;
        float amax = 0.0f;
#pragma unroll
        for (int d = 0; d < kD; ++d)
        {
            const float c = kraw[key * kD + d] - kmean_reg[d];
            kraw[key * kD + d] = c;
            amax = fmaxf(amax, fabsf(c));
        }
        kstat[key] = amax;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float amax = 0.0f;
#pragma unroll
        for (int key = 0; key < kSN; ++key)
        {
            amax = fmaxf(amax, kstat[key]);
        }
        ksc = fmaxf(amax / 127.0f, 1.0e-6f);
    }
    __syncthreads();
    if (threadIdx.x < kSN)
    {
        const int key = threadIdx.x;
        const float s = ksc;
#pragma unroll
        for (int d = 0; d < kD; d += 4)
        {
            reinterpret_cast<uint32_t*>(ku_smem)[(key * kD + d) / 4] = pack_quant4_s8(
                kraw[key * kD + d], kraw[key * kD + d + 1], kraw[key * kD + d + 2], kraw[key * kD + d + 3], s);
        }
    }

    // Per-lane Q rows {base+group, base+group+8} of this warp's 16-row pair... (kQT collapses to 1:
    // 4 warps x 16 rows cover all 64 queries).
    const int row_lo = warp * 16 + group;
    const int row_hi = row_lo + 8;
    float qs_lo = 0.0f, qs_hi = 0.0f;
    const uint32_t qa0 = load_quant_q_frag(q, qbase + static_cast<int64_t>(row_lo) * kD, row_lo < m, dtype, &qs_lo);
    const uint32_t qa1 = load_quant_q_frag(q, qbase + static_cast<int64_t>(row_hi) * kD, row_hi < m, dtype, &qs_hi);

    cp_async_wait<0>();
    __syncthreads();  // ku_smem + vstage visible for cross-lane ldmatrix

    const float ql = qs_lo * sm_scale * kLog2e * ksc;
    const float qh = qs_hi * sm_scale * kLog2e * ksc;
    const float sl = kHasBias ? 1.0f : ql;
    const float sh = kHasBias ? 1.0f : qh;

    uint32_t rkb[kSNT];
#pragma unroll
    for (int kb = 0; kb < kSN; kb += 16)
    {
        ldmatrix_x2_u8(rkb[kb / 8], rkb[kb / 8 + 1], &ku_smem[(kb + (lane & 15)) * kD]);
    }

    float dl0[kSNT], dl1[kSNT], dh0[kSNT], dh1[kSNT];
    float lm_lo = kNegInf, lm_hi = kNegInf;
#pragma unroll
    for (int t = 0; t < kSNT; ++t)
    {
        const uint32_t b0 = rkb[t];
        int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(qa0), "r"(qa1), "r"(b0));
        dl0[t] = static_cast<float>(d0);
        dl1[t] = static_cast<float>(d1);
        dh0[t] = static_cast<float>(d2);
        dh1[t] = static_cast<float>(d3);
        if constexpr (kHasBias)
        {
            const int c0 = t * 8 + 2 * tid;
            const bool k0ok = kFullKeys || (c0 < n);
            const bool k1ok = kFullKeys || (c0 + 1 < n);
            const int64_t bb_lo = bias_base + static_cast<int64_t>(row_lo) * n;
            const int64_t bb_hi = bias_base + static_cast<int64_t>(row_hi) * n;
            const bool rlok = row_lo < m;
            const bool rhok = row_hi < m;
            const float b_lo0 = (rlok && k0ok) ? load_as_float(bias, bb_lo + c0, dtype) * kLog2e : 0.0f;
            const float b_lo1 = (rlok && k1ok) ? load_as_float(bias, bb_lo + c0 + 1, dtype) * kLog2e : 0.0f;
            const float b_hi0 = (rhok && k0ok) ? load_as_float(bias, bb_hi + c0, dtype) * kLog2e : 0.0f;
            const float b_hi1 = (rhok && k1ok) ? load_as_float(bias, bb_hi + c0 + 1, dtype) * kLog2e : 0.0f;
            dl0[t] = fmaf(dl0[t], ql, b_lo0);
            dl1[t] = fmaf(dl1[t], ql, b_lo1);
            dh0[t] = fmaf(dh0[t], qh, b_hi0);
            dh1[t] = fmaf(dh1[t], qh, b_hi1);
        }
        if (!kFullKeys)
        {
            const int c0 = t * 8 + 2 * tid;
            if (c0 >= n)
            {
                dl0[t] = -1e30f;
                dh0[t] = -1e30f;
            }
            if (c0 + 1 >= n)
            {
                dl1[t] = -1e30f;
                dh1[t] = -1e30f;
            }
        }
        lm_lo = fmaxf(lm_lo, fmaxf(dl0[t], dl1[t]));
        lm_hi = fmaxf(lm_hi, fmaxf(dh0[t], dh1[t]));
    }
    // Exact row max over the single tile (args to exp2 are then <= 0: no clamp, no rescale).
    lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, 1));
    lm_lo = fmaxf(lm_lo, __shfl_xor_sync(0xffffffffu, lm_lo, 2));
    lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, 1));
    lm_hi = fmaxf(lm_hi, __shfl_xor_sync(0xffffffffu, lm_hi, 2));
    const float nm_lo = lm_lo * sl;
    const float nm_hi = lm_hi * sh;

    float og[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        og[i] = 0.0f;
    }
    float ls_lo = 0.0f, ls_hi = 0.0f;
#pragma unroll
    for (int b16 = 0; b16 < kSNT / 2; ++b16)
    {
        const int co = b16 * 16;
        const int t0 = 2 * b16;
        const int t1 = t0 + 1;
        const float p0_lo0 = exp2f(fmaf(dl0[t0], sl, -nm_lo));
        const float p0_lo1 = exp2f(fmaf(dl1[t0], sl, -nm_lo));
        const float p0_hi0 = exp2f(fmaf(dh0[t0], sh, -nm_hi));
        const float p0_hi1 = exp2f(fmaf(dh1[t0], sh, -nm_hi));
        const float p1_lo0 = exp2f(fmaf(dl0[t1], sl, -nm_lo));
        const float p1_lo1 = exp2f(fmaf(dl1[t1], sl, -nm_lo));
        const float p1_hi0 = exp2f(fmaf(dh0[t1], sh, -nm_hi));
        const float p1_hi1 = exp2f(fmaf(dh1[t1], sh, -nm_hi));
        ls_lo += p0_lo0 + p0_lo1 + p1_lo0 + p1_lo1;
        ls_hi += p0_hi0 + p0_hi1 + p1_hi0 + p1_hi1;
        const uint32_t a0 = pack_p2<kHalf>(p0_lo0, p0_lo1);
        const uint32_t a1 = pack_p2<kHalf>(p0_hi0, p0_hi1);
        const uint32_t a2 = pack_p2<kHalf>(p1_lo0, p1_lo1);
        const uint32_t a3 = pack_p2<kHalf>(p1_hi0, p1_hi1);
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
        {
            uint32_t b0, b1;
            ldmatrix_x2_trans_b16(b0, b1, &vstage[nn * kSN * 8 + (co + (lane & 15)) * 8]);
            if constexpr (kHalf)
            {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(og[nn * 4 + 0]), "+f"(og[nn * 4 + 1]), "+f"(og[nn * 4 + 2]), "+f"(og[nn * 4 + 3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
            else
            {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(og[nn * 4 + 0]), "+f"(og[nn * 4 + 1]), "+f"(og[nn * 4 + 2]), "+f"(og[nn * 4 + 3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
    }
    ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, 1);
    ls_lo += __shfl_xor_sync(0xffffffffu, ls_lo, 2);
    ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, 1);
    ls_hi += __shfl_xor_sync(0xffffffffu, ls_hi, 2);

    const float inv_l_lo = row_lo < m ? 1.0f / ls_lo : 0.0f;
    const float inv_l_hi = row_hi < m ? 1.0f / ls_hi : 0.0f;
#pragma unroll
    for (int nn = 0; nn < 2; ++nn)
    {
        const int d0c = nn * 8 + 2 * tid;
        const int d1c = d0c + 1;
        if (row_lo < m)
        {
            store_as_dtype(o, obase + static_cast<int64_t>(row_lo) * kD + d0c, og[nn * 4 + 0] * inv_l_lo, dtype);
            store_as_dtype(o, obase + static_cast<int64_t>(row_lo) * kD + d1c, og[nn * 4 + 1] * inv_l_lo, dtype);
        }
        if (row_hi < m)
        {
            store_as_dtype(o, obase + static_cast<int64_t>(row_hi) * kD + d0c, og[nn * 4 + 2] * inv_l_hi, dtype);
            store_as_dtype(o, obase + static_cast<int64_t>(row_hi) * kD + d1c, og[nn * 4 + 3] * inv_l_hi, dtype);
        }
    }
}

// ---- Workspace layout --------------------------------------------------------------------

inline size_t alignUp(size_t x, size_t a)
{
    return (x + a - 1) / a * a;
}

struct Workspace
{
    uint8_t* ku16;
    float* ks;
    float* km;
    size_t total;
};

Workspace planWorkspace(char* base, int batch, int heads, int nq, int nkv)
{
    const int64_t bh = static_cast<int64_t>(batch) * heads;
    const int64_t nkvElems = bh * nkv;
    const int64_t statElems = bh * kD;
    const int64_t nkBlocks = bh * ((nkv + kBN - 1) / kBN);  // per-block K scales
    Workspace ws{};
    size_t off = 0;
    auto take = [&](size_t bytes) -> char* {
        char* p = base ? base + off : nullptr;
        off = alignUp(off + bytes, 256);
        return p;
    };
    ws.ku16 = reinterpret_cast<uint8_t*>(take(static_cast<size_t>(nkvElems) * kD * sizeof(uint8_t)));
    ws.ks = reinterpret_cast<float*>(take(static_cast<size_t>(nkBlocks) * sizeof(float)));
    ws.km = reinterpret_cast<float*>(take(static_cast<size_t>(statElems) * sizeof(float)));
    ws.total = off;
    return ws;
}

} // namespace

size_t workspaceSize(int batch, int heads, int nq, int nkv)
{
    return planWorkspace(nullptr, batch, heads, nq, nkv).total;
}

cudaError_t launchColorAttnD16(const void* q, const void* kt, const void* v, const void* bias, void* o,
    int batch, int heads, int nq, int nkv, int dtype, float smScale, bool ktTransposed, void* workspace,
    size_t workspaceBytes, cudaStream_t stream)
{
    if (batch <= 0 || heads <= 0 || nq <= 0 || nkv <= 0)
    {
        return cudaErrorInvalidValue;
    }
    const int bh = batch * heads;

    // Small-window fast path (OmniSR 8x8 windows): one block per (b,h), K prepass fused inline,
    // exact single-tile softmax. Needs no workspace.
    if (nq <= kSN && nkv <= kSN)
    {
        const int64_t sKH = static_cast<int64_t>(kD) * nkv;
        const int64_t sKd = ktTransposed ? nkv : 1;
        const int64_t sKn = ktTransposed ? 1 : kD;
        const bool isHalfS = dtype == kHalf;
        const bool fullKeysS = nkv == kSN;
        const bool hasBiasS = bias != nullptr;
        dim3 grid(static_cast<unsigned>(bh));
#define COLOR_ATTN_D16_SMALL_LAUNCH(FK, HF, HB)                                                     \
    fusedAttnSmallKernel<FK, HF, HB><<<grid, kSWarps * 32, 0, stream>>>(                            \
        q, kt, v, bias, o, dtype, nq, nkv, heads, smScale, sKH, sKd, sKn)
        if (fullKeysS)
        {
            if (isHalfS)
                hasBiasS ? COLOR_ATTN_D16_SMALL_LAUNCH(true, true, true) : COLOR_ATTN_D16_SMALL_LAUNCH(true, true, false);
            else
                hasBiasS ? COLOR_ATTN_D16_SMALL_LAUNCH(true, false, true) : COLOR_ATTN_D16_SMALL_LAUNCH(true, false, false);
        }
        else
        {
            if (isHalfS)
                hasBiasS ? COLOR_ATTN_D16_SMALL_LAUNCH(false, true, true) : COLOR_ATTN_D16_SMALL_LAUNCH(false, true, false);
            else
                hasBiasS ? COLOR_ATTN_D16_SMALL_LAUNCH(false, false, true) : COLOR_ATTN_D16_SMALL_LAUNCH(false, false, false);
        }
#undef COLOR_ATTN_D16_SMALL_LAUNCH
        return cudaPeekAtLastError();
    }

    Workspace ws = planWorkspace(static_cast<char*>(workspace), batch, heads, nq, nkv);
    if (workspace == nullptr || workspaceBytes < ws.total)
    {
        return cudaErrorInvalidValue;
    }
    // Contiguous strides (in elements) for the [B,H,*,*] tensors. K is read only by the strided
    // prepasses, so the K^T [B,H,16,Nkv] vs raw K [B,H,Nkv,16] layouts differ only in (sKd, sKn).
    const int64_t kSB = static_cast<int64_t>(heads) * kD * nkv;
    const int64_t kSH = static_cast<int64_t>(kD) * nkv;
    const int64_t kSKd = ktTransposed ? nkv : 1;
    const int64_t kSKn = ktTransposed ? 1 : kD;
    const int64_t oSB = static_cast<int64_t>(heads) * nq * kD;
    const int64_t oSH = static_cast<int64_t>(nq) * kD;
    const int64_t oSN = kD;

    const int kThreads = 256;

    // bh can exceed the 65535 grid.y limit (OmniSR: B = thousands of 8x8 windows), so split it
    // across grid y*z; kernels recombine bh = blockIdx.y + gridDim.y * blockIdx.z.
    const unsigned bhY = static_cast<unsigned>(bh > 65535 ? 65535 : bh);
    const unsigned bhZ = static_cast<unsigned>((bh + bhY - 1) / bhY);

    dim3 statGrid(static_cast<unsigned>(kD), bhY, bhZ);
    kMeanKernel<<<statGrid, kThreads, 0, stream>>>(kt, ws.km, nkv, heads, bh, kSB, kSH, kSKd, kSKn, dtype);

    // One block of kBN threads per kBN-key block -> single per-block K scale.
    dim3 kGrid(static_cast<unsigned>((nkv + kBN - 1) / kBN), bhY, bhZ);
    quantKKernel<<<kGrid, kBN, 0, stream>>>(kt, ws.km, ws.ku16, ws.ks, nkv, heads, bh, kSB, kSH, kSKd, kSKn, dtype);

    // V needs no prepass: it is streamed raw (fp16/bf16) straight into the PV m16n8k16 MMA.

    dim3 attnGrid(static_cast<unsigned>((nq + kBlockM - 1) / kBlockM), bhY, bhZ);
    const bool isHalf = dtype == kHalf;
    const bool fullKeys = nkv % kBN == 0;
    const bool hasBias = bias != nullptr;
#define COLOR_ATTN_D16_LAUNCH(FK, HF, HB)                                                                    \
    fusedAttnKernel<FK, HF, HB><<<attnGrid, kWarps * 32, 0, stream>>>(                                       \
        q, ws.ku16, ws.ks, v, bias, o, dtype, nq, nkv, heads, bh, smScale, oSB, oSH, oSN)
    if (fullKeys)
    {
        if (isHalf)
            hasBias ? COLOR_ATTN_D16_LAUNCH(true, true, true) : COLOR_ATTN_D16_LAUNCH(true, true, false);
        else
            hasBias ? COLOR_ATTN_D16_LAUNCH(true, false, true) : COLOR_ATTN_D16_LAUNCH(true, false, false);
    }
    else
    {
        if (isHalf)
            hasBias ? COLOR_ATTN_D16_LAUNCH(false, true, true) : COLOR_ATTN_D16_LAUNCH(false, true, false);
        else
            hasBias ? COLOR_ATTN_D16_LAUNCH(false, false, true) : COLOR_ATTN_D16_LAUNCH(false, false, false);
    }
#undef COLOR_ATTN_D16_LAUNCH

    return cudaPeekAtLastError();
}

} // namespace color_attn_d16
