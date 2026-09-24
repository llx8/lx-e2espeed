// Thin torch binding around launchColorAttnD16 for standalone numerical testing.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "color_attn_d16_kernels.h"

torch::Tensor color_attn_d16_forward(torch::Tensor q, torch::Tensor kt, torch::Tensor v, double sm_scale,
    c10::optional<torch::Tensor> bias_opt = c10::nullopt)
{
    TORCH_CHECK(q.is_cuda() && kt.is_cuda() && v.is_cuda(), "q/kt/v must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && kt.dim() == 4 && v.dim() == 4, "q/kt/v must be [B,H,N,*] 4D");
    q = q.contiguous();
    kt = kt.contiguous();
    v = v.contiguous();
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int Nq = static_cast<int>(q.size(2));
    const int Nkv = static_cast<int>(v.size(2));
    TORCH_CHECK(q.size(3) == 16 && kt.size(2) == 16 && v.size(3) == 16, "head_dim must be 16");
    TORCH_CHECK(kt.size(3) == Nkv, "K^T last dim must equal Nkv");
    TORCH_CHECK(q.scalar_type() == kt.scalar_type() && q.scalar_type() == v.scalar_type(), "dtype mismatch");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "q/kt/v must be fp16 or bf16");

    torch::Tensor bias;
    const void* biasPtr = nullptr;
    if (bias_opt.has_value() && bias_opt->defined())
    {
        bias = bias_opt->contiguous();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.scalar_type() == q.scalar_type(), "bias dtype must match q");
        if (bias.dim() == 4)
        {
            TORCH_CHECK(bias.size(0) == 1, "4D bias must have leading dim 1");
            bias = bias.squeeze(0);
        }
        TORCH_CHECK(bias.dim() == 3 && bias.size(0) == H && bias.size(1) == Nq && bias.size(2) == Nkv,
            "bias must be [H,Nq,Nkv]");
        biasPtr = bias.data_ptr();
    }

    const int tag = q.scalar_type() == at::kHalf ? color_attn_d16::kHalf : color_attn_d16::kBF16;
    auto o = torch::empty({B, H, Nq, 16}, q.options());
    const size_t wsBytes = color_attn_d16::workspaceSize(B, H, Nq, Nkv);
    auto ws = torch::empty({static_cast<int64_t>(wsBytes)},
        torch::TensorOptions().device(q.device()).dtype(torch::kUInt8));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = color_attn_d16::launchColorAttnD16(q.data_ptr(), kt.data_ptr(), v.data_ptr(), biasPtr,
        o.data_ptr(), B, H, Nq, Nkv, tag, static_cast<float>(sm_scale), true, ws.data_ptr(), wsBytes, stream);
    TORCH_CHECK(err == cudaSuccess, "launchColorAttnD16 failed: ", cudaGetErrorString(err));
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("color_attn_d16", &color_attn_d16_forward,
        "d=16 INT8-QK + FP8-PV fused attention (Q,K^T,V,sm_scale[,bias])",
        pybind11::arg("q"), pybind11::arg("kt"), pybind11::arg("v"), pybind11::arg("sm_scale"),
        pybind11::arg("bias") = pybind11::none());
}
