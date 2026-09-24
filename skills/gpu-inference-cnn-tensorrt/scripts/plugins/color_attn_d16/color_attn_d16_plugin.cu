// ColorAttnD16Int8 TensorRT plugin: replaces the diffusers decomposed d=16
// attention subgraph (QK MatMul -> Softmax -> AV MatMul) with a single fused
// INT8-QK (m16n8k16.s8.u8 + smooth_k) + FP8-E4M3-PV flash-attention kernel.
//
// Inputs (all the SAME float type, fp16 or bf16, kLINEAR):
//   0: Q    [B, H, Nq, 16]
//   1: K^T  [B, H, 16, Nkv]   (K already transposed by the kept graph Transpose)
//   2: V    [B, H, Nkv, 16]
//   3: bias [H, Nq, Nkv] or [1, H, Nq, Nkv]  (OPTIONAL additive pre-softmax bias, e.g.
//      OmniSR rel_pos_bias; broadcast over batch)
// Output:
//   0: O   [B, H, Nq, 16]    (= Q dims)
//
// The softmax scale is pre-folded into Q/K by the kept graph Muls, so the
// default sm_scale is 1.0.

#include "color_attn_d16_kernels.h"

#include "NvInfer.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace color_attn_d16_plugin
{
constexpr char const* kPLUGIN_NAME{"ColorAttnD16Int8Plugin"};
constexpr char const* kPLUGIN_VERSION{"1"};
constexpr int32_t kHEAD_DIM = color_attn_d16::kHeadDim;

bool dimMatches(int32_t actual, int32_t expected, bool allowDynamic) noexcept
{
    return (allowDynamic && actual < 0) || actual == expected;
}

bool dimsEqual(int32_t a, int32_t b, bool allowDynamic) noexcept
{
    return (allowDynamic && (a < 0 || b < 0)) || a == b;
}

bool validateAttentionDims(
    Dims const& qd, Dims const& kd, Dims const& vd, bool ktTransposed, bool allowDynamic) noexcept
{
    if (qd.nbDims != 4 || kd.nbDims != 4 || vd.nbDims != 4)
    {
        return false;
    }

    // Q: [B,H,Nq,16], V: [B,H,Nkv,16], K: [B,H,16,Nkv] (ktTransposed) or [B,H,Nkv,16] (raw).
    int const kDimD = ktTransposed ? 2 : 3;
    int const kDimN = ktTransposed ? 3 : 2;
    if (!dimMatches(qd.d[3], kHEAD_DIM, allowDynamic) || !dimMatches(kd.d[kDimD], kHEAD_DIM, allowDynamic)
        || !dimMatches(vd.d[3], kHEAD_DIM, allowDynamic))
    {
        return false;
    }
    if (!dimsEqual(qd.d[0], kd.d[0], allowDynamic) || !dimsEqual(qd.d[0], vd.d[0], allowDynamic)
        || !dimsEqual(qd.d[1], kd.d[1], allowDynamic) || !dimsEqual(qd.d[1], vd.d[1], allowDynamic))
    {
        return false;
    }
    return dimsEqual(kd.d[kDimN], vd.d[2], allowDynamic);
}

// bias: [H, Nq, Nkv] (3D) or [1, H, Nq, Nkv] (4D with leading 1).
bool validateBiasDims(Dims const& bd, Dims const& qd, Dims const& vd, bool allowDynamic) noexcept
{
    int off = 0;
    if (bd.nbDims == 4)
    {
        if (!dimMatches(bd.d[0], 1, allowDynamic))
        {
            return false;
        }
        off = 1;
    }
    else if (bd.nbDims != 3)
    {
        return false;
    }
    return dimsEqual(bd.d[off + 0], qd.d[1], allowDynamic) && dimsEqual(bd.d[off + 1], qd.d[2], allowDynamic)
        && dimsEqual(bd.d[off + 2], vd.d[2], allowDynamic);
}

class ColorAttnD16Int8Plugin : public IPluginV2DynamicExt
{
public:
    explicit ColorAttnD16Int8Plugin(float smScale, bool hasBias = false, bool ktTransposed = true)
        : mSmScale(smScale)
        , mHasBias(hasBias)
        , mKtTransposed(ktTransposed)
    {
    }

    ColorAttnD16Int8Plugin(void const* data, size_t length)
    {
        auto const* bytes = static_cast<uint8_t const*>(data);
        if (length >= sizeof(float))
        {
            std::memcpy(&mSmScale, bytes, sizeof(float));
        }
        if (length >= sizeof(float) + 1)
        {
            mHasBias = bytes[sizeof(float)] != 0;
        }
        if (length >= sizeof(float) + 2)
        {
            mKtTransposed = bytes[sizeof(float) + 1] != 0;
        }
    }

    int32_t expectedNbInputs() const noexcept
    {
        return mHasBias ? 4 : 3;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    DimsExprs getOutputDimensions(
        int32_t index, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder&) noexcept override
    {
        if (index != 0 || (nbInputs != 3 && nbInputs != 4))
        {
            return DimsExprs{};
        }
        return DimsExprs(inputs[0]); // O has the same shape as Q.
    }

    int32_t initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override {}

    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const*,
        int32_t) const noexcept override
    {
        if (inputs == nullptr || (nbInputs != 3 && nbInputs != 4)
            || !validateAttentionDims(inputs[0].dims, inputs[1].dims, inputs[2].dims, mKtTransposed,
                /*allowDynamic=*/false))
        {
            return 0;
        }
        if (nbInputs == 4 && !validateBiasDims(inputs[3].dims, inputs[0].dims, inputs[2].dims, /*allowDynamic=*/false))
        {
            return 0;
        }
        int32_t const batch = static_cast<int32_t>(inputs[0].dims.d[0]);
        int32_t const heads = static_cast<int32_t>(inputs[0].dims.d[1]);
        int32_t const nq = static_cast<int32_t>(inputs[0].dims.d[2]);
        int32_t const nkv = static_cast<int32_t>(inputs[2].dims.d[2]);
        return color_attn_d16::workspaceSize(batch, heads, nq, nkv);
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        Dims const& qd = inputDesc[0].dims; // [B, H, Nq, 16]
        Dims const& kd = inputDesc[1].dims; // [B, H, 16, Nkv] or [B, H, Nkv, 16]
        Dims const& vd = inputDesc[2].dims; // [B, H, Nkv, 16]
        if (!validateAttentionDims(qd, kd, vd, mKtTransposed, /*allowDynamic=*/false))
        {
            return 1;
        }
        if (mHasBias && !validateBiasDims(inputDesc[3].dims, qd, vd, /*allowDynamic=*/false))
        {
            return 1;
        }
        int32_t const batch = static_cast<int32_t>(qd.d[0]);
        int32_t const heads = static_cast<int32_t>(qd.d[1]);
        int32_t const nq = static_cast<int32_t>(qd.d[2]);
        int32_t const nkv = static_cast<int32_t>(vd.d[2]);

        int dtypeTag = inputDesc[0].type == DataType::kHALF ? color_attn_d16::kHalf : color_attn_d16::kBF16;

        void const* biasPtr = mHasBias ? inputs[3] : nullptr;
        size_t const wsBytes = color_attn_d16::workspaceSize(batch, heads, nq, nkv);
        cudaError_t err = color_attn_d16::launchColorAttnD16(inputs[0], inputs[1], inputs[2], biasPtr, outputs[0],
            batch, heads, nq, nkv, dtypeTag, mSmScale, mKtTransposed, workspace, wsBytes, stream);
        if (err != cudaSuccess)
        {
            std::cerr << "[ColorAttnD16Int8Plugin] launch failed: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        return 0;
    }

    size_t getSerializationSize() const noexcept override
    {
        return sizeof(float) + 2;
    }

    void serialize(void* buffer) const noexcept override
    {
        auto* bytes = static_cast<uint8_t*>(buffer);
        std::memcpy(bytes, &mSmScale, sizeof(float));
        bytes[sizeof(float)] = mHasBias ? 1 : 0;
        bytes[sizeof(float) + 1] = mKtTransposed ? 1 : 0;
    }

    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || (nbInputs != 3 && nbInputs != 4) || nbOutputs != 1 || pos < 0
            || pos >= nbInputs + 1)
        {
            return false;
        }
        if (inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }
        // All operands share one float type (fp16 or bf16), matched to input 0.
        bool typeOk = inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kBF16;
        return typeOk && inOut[pos].type == inOut[0].type;
    }

    char const* getPluginType() const noexcept override
    {
        return kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPLUGIN_VERSION;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        auto* plugin = new ColorAttnD16Int8Plugin(mSmScale, mHasBias, mKtTransposed);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    void destroy() noexcept override
    {
        delete this;
    }

    void setPluginNamespace(char const* ns) noexcept override
    {
        mNamespace = ns ? ns : "";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    DataType getOutputDataType(int32_t, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        return (inputTypes != nullptr && (nbInputs == 3 || nbInputs == 4)) ? inputTypes[0] : DataType::kHALF;
    }

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const*,
        int32_t nbOutputs) noexcept override
    {
        if (in == nullptr || (nbInputs != 3 && nbInputs != 4) || nbOutputs != 1)
        {
            std::cerr << "[ColorAttnD16Int8Plugin] expected 3 or 4 inputs and 1 output" << std::endl;
            return;
        }
        mHasBias = nbInputs == 4;
        if (!validateAttentionDims(in[0].desc.dims, in[1].desc.dims, in[2].desc.dims, mKtTransposed,
                /*allowDynamic=*/true)
            || !validateAttentionDims(in[0].min, in[1].min, in[2].min, mKtTransposed, /*allowDynamic=*/false)
            || !validateAttentionDims(in[0].max, in[1].max, in[2].max, mKtTransposed, /*allowDynamic=*/false))
        {
            std::cerr << "[ColorAttnD16Int8Plugin] unsupported input shapes; expected Q=[B,H,Nq,16], "
                         "K=[B,H,16,Nkv] (kt_transposed=1) or [B,H,Nkv,16] (kt_transposed=0), V=[B,H,Nkv,16]"
                      << std::endl;
        }
        if (mHasBias
            && (!validateBiasDims(in[3].desc.dims, in[0].desc.dims, in[2].desc.dims, /*allowDynamic=*/true)
                || !validateBiasDims(in[3].min, in[0].min, in[2].min, /*allowDynamic=*/false)
                || !validateBiasDims(in[3].max, in[0].max, in[2].max, /*allowDynamic=*/false)))
        {
            std::cerr << "[ColorAttnD16Int8Plugin] unsupported bias shape; expected [H,Nq,Nkv] or [1,H,Nq,Nkv]"
                      << std::endl;
        }
    }

private:
    float mSmScale{1.0F};
    bool mHasBias{false};
    bool mKtTransposed{true};
    std::string mNamespace;
};

class ColorAttnD16Int8PluginCreator : public IPluginCreator
{
public:
    ColorAttnD16Int8PluginCreator()
    {
        mAttrs.emplace_back(PluginField("sm_scale", nullptr, PluginFieldType::kFLOAT32, 1));
        mAttrs.emplace_back(PluginField("has_bias", nullptr, PluginFieldType::kINT32, 1));
        mAttrs.emplace_back(PluginField("kt_transposed", nullptr, PluginFieldType::kINT32, 1));
        mFC.nbFields = static_cast<int32_t>(mAttrs.size());
        mFC.fields = mAttrs.data();
    }

    char const* getPluginName() const noexcept override
    {
        return kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPLUGIN_VERSION;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV2* createPlugin(char const*, PluginFieldCollection const* fc) noexcept override
    {
        float smScale = 1.0F;
        bool hasBias = false;
        bool ktTransposed = true;
        if (fc != nullptr)
        {
            for (int32_t i = 0; i < fc->nbFields; ++i)
            {
                if (fc->fields[i].name != nullptr && std::strcmp(fc->fields[i].name, "sm_scale") == 0
                    && fc->fields[i].type == PluginFieldType::kFLOAT32 && fc->fields[i].length >= 1)
                {
                    smScale = *static_cast<float const*>(fc->fields[i].data);
                }
                if (fc->fields[i].name != nullptr && std::strcmp(fc->fields[i].name, "has_bias") == 0
                    && fc->fields[i].type == PluginFieldType::kINT32 && fc->fields[i].length >= 1)
                {
                    hasBias = *static_cast<int32_t const*>(fc->fields[i].data) != 0;
                }
                if (fc->fields[i].name != nullptr && std::strcmp(fc->fields[i].name, "kt_transposed") == 0
                    && fc->fields[i].type == PluginFieldType::kINT32 && fc->fields[i].length >= 1)
                {
                    ktTransposed = *static_cast<int32_t const*>(fc->fields[i].data) != 0;
                }
            }
        }
        auto* plugin = new ColorAttnD16Int8Plugin(smScale, hasBias, ktTransposed);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    IPluginV2* deserializePlugin(char const*, void const* data, size_t length) noexcept override
    {
        auto* plugin = new ColorAttnD16Int8Plugin(data, length);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    void setPluginNamespace(char const* ns) noexcept override
    {
        mNamespace = ns ? ns : "";
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    PluginFieldCollection mFC{};
    std::vector<PluginField> mAttrs;
    std::string mNamespace;
};
} // namespace color_attn_d16_plugin

using ColorAttnD16Int8PluginCreator = color_attn_d16_plugin::ColorAttnD16Int8PluginCreator;
REGISTER_TENSORRT_PLUGIN(ColorAttnD16Int8PluginCreator);
