#pragma once

#include "aexrt.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace aexrt {

class Buffer;

class Device {
public:
    explicit Device(uint32_t adapter_index = 0) : handle_(aexrt_d3d12_create_device(adapter_index)) {
        if (!handle_) {
            throw std::runtime_error("failed to create AEXRT D3D12 device");
        }
    }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Device&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Device& operator=(Device&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Device() {
        reset();
    }

    AexrtDevice* get() const { return handle_; }

    AexrtD3D12Capabilities capabilities() const {
        AexrtD3D12Capabilities result{};
        if (!aexrt_d3d12_get_capabilities(handle_, &result)) {
            throw std::runtime_error("failed to query AEXRT D3D12 capabilities");
        }
        return result;
    }

private:
    void reset() {
        if (handle_) {
            aexrt_d3d12_destroy_device(handle_);
            handle_ = nullptr;
        }
    }

    AexrtDevice* handle_ = nullptr;
};

class Buffer {
public:
    Buffer() = default;
    explicit Buffer(AexrtBuffer* handle) : handle_(handle) {
        if (!handle_) {
            throw std::runtime_error("failed to create AEXRT buffer");
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Buffer() {
        reset();
    }

    AexrtBuffer* get() const { return handle_; }

private:
    void reset() {
        if (handle_) {
            aexrt_d3d12_destroy_buffer(handle_);
            handle_ = nullptr;
        }
    }

    AexrtBuffer* handle_ = nullptr;
};

class PreparedDispatch {
public:
    PreparedDispatch() = default;
    explicit PreparedDispatch(AexrtPreparedDispatch* handle) : handle_(handle) {
        if (!handle_) {
            throw std::runtime_error("failed to prepare AEXRT dispatch");
        }
    }

    PreparedDispatch(const PreparedDispatch&) = delete;
    PreparedDispatch& operator=(const PreparedDispatch&) = delete;

    PreparedDispatch(PreparedDispatch&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    PreparedDispatch& operator=(PreparedDispatch&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~PreparedDispatch() {
        reset();
    }

    AexrtPreparedDispatch* get() const { return handle_; }

private:
    void reset() {
        if (handle_) {
            aexrt_d3d12_destroy_prepared(handle_);
            handle_ = nullptr;
        }
    }

    AexrtPreparedDispatch* handle_ = nullptr;
};

class Graph {
public:
    Graph() = default;
    explicit Graph(AexrtCompiledGraph* handle) : handle_(handle) {
        if (!handle_) {
            throw std::runtime_error("failed to compile AEXRT graph");
        }
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Graph& operator=(Graph&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~Graph() {
        reset();
    }

    AexrtCompiledGraph* get() const { return handle_; }
    uint32_t dispatch_count() const { return aexrt_graph_dispatch_count(handle_); }
    uint32_t buffer_count() const { return aexrt_graph_buffer_count(handle_); }

private:
    void reset() {
        if (handle_) {
            aexrt_destroy_graph(handle_);
            handle_ = nullptr;
        }
    }

    AexrtCompiledGraph* handle_ = nullptr;
};

class YoloModel {
public:
    YoloModel() = default;
    explicit YoloModel(AexrtYoloModel* handle) : handle_(handle) {
        if (!handle_) {
            throw std::runtime_error("failed to compile AEXRT YOLO package");
        }
    }

    YoloModel(const YoloModel&) = delete;
    YoloModel& operator=(const YoloModel&) = delete;

    YoloModel(YoloModel&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    YoloModel& operator=(YoloModel&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~YoloModel() {
        reset();
    }

    AexrtYoloModel* get() const { return handle_; }
    uint64_t input_element_count() const { return aexrt_yolo_input_element_count(handle_); }
    uint32_t class_count() const { return aexrt_yolo_class_count(handle_); }
    uint32_t anchor_count() const { return aexrt_yolo_anchor_count(handle_); }
    uint32_t channel_count() const { return aexrt_yolo_channel_count(handle_); }
    AexrtYoloOutputLayout output_layout() const { return static_cast<AexrtYoloOutputLayout>(aexrt_yolo_output_layout(handle_)); }
    bool has_objectness() const { return aexrt_yolo_has_objectness(handle_) != 0; }
    AexrtYoloPackageMode package_mode() const { return static_cast<AexrtYoloPackageMode>(aexrt_yolo_package_mode(handle_)); }
    uint32_t graph_node_count() const { return aexrt_yolo_graph_node_count(handle_); }
    uint32_t graph_value_count() const { return aexrt_yolo_graph_value_count(handle_); }
    uint32_t constant_count() const { return aexrt_yolo_constant_count(handle_); }
    uint32_t prepared_command_count() const { return aexrt_yolo_prepared_command_count(handle_); }
    uint32_t supported_prepared_command_count() const { return aexrt_yolo_supported_prepared_command_count(handle_); }
    uint32_t unsupported_prepared_command_count() const { return aexrt_yolo_unsupported_prepared_command_count(handle_); }
    uint32_t prepared_skipped_command_count() const { return aexrt_yolo_prepared_skipped_command_count(handle_); }
    uint32_t late_concat_conv1x1_fusion_count() const { return aexrt_yolo_late_concat_conv1x1_fusion_count(handle_); }
    uint32_t concat_residual_conv1x1_fusion_count() const { return aexrt_yolo_concat_residual_conv1x1_fusion_count(handle_); }
    uint32_t paired_conv3x3_fusion_count() const { return aexrt_yolo_paired_conv3x3_fusion_count(handle_); }
    uint32_t c2f_bottleneck_superblock_count() const { return aexrt_yolo_c2f_bottleneck_superblock_count(handle_); }
    uint32_t c2f_tail_residual_fusion_count() const { return aexrt_yolo_c2f_tail_residual_fusion_count(handle_); }
    bool head_fusion_enabled() const { return aexrt_yolo_head_fusion_enabled(handle_) != 0; }
    bool head_final_conv_fusion_enabled() const { return aexrt_yolo_head_final_conv_fusion_enabled(handle_) != 0; }
    uint32_t tileflow_conv1x1_count() const { return aexrt_yolo_tileflow_conv1x1_count(handle_); }
    uint32_t tileflow_3x3_spatial_count() const { return aexrt_yolo_tileflow_3x3_spatial_count(handle_); }
    uint32_t tileflow_3x3_pack4_count() const { return aexrt_yolo_tileflow_3x3_pack4_count(handle_); }
    uint32_t tileflow_3x3_pack8_count() const { return aexrt_yolo_tileflow_3x3_pack8_count(handle_); }
    uint32_t tileflow_3x3_implicit_gemm_count() const { return aexrt_yolo_tileflow_3x3_implicit_gemm_count(handle_); }
    uint32_t tileflow_3x3_implicit_gemm_40x40_count() const { return aexrt_yolo_tileflow_3x3_implicit_gemm_40x40_count(handle_); }
    uint32_t tileflow_3x3_implicit_gemm_20x20_count() const { return aexrt_yolo_tileflow_3x3_implicit_gemm_20x20_count(handle_); }
    uint32_t tileflow_3x3_implicit_gemm_10x10_count() const { return aexrt_yolo_tileflow_3x3_implicit_gemm_10x10_count(handle_); }
    uint32_t tileflow_3x3_exact_40x40_64x64_count() const { return aexrt_yolo_tileflow_3x3_exact_40x40_64x64_count(handle_); }
    uint32_t tileflow_3x3_exact_20x20_64x64_count() const { return aexrt_yolo_tileflow_3x3_exact_20x20_64x64_count(handle_); }
    uint32_t tileflow_3x3_exact_10x10_128x128_count() const { return aexrt_yolo_tileflow_3x3_exact_10x10_128x128_count(handle_); }
    uint32_t tileflow_3x3_exact_20x20_128x64_count() const { return aexrt_yolo_tileflow_3x3_exact_20x20_128x64_count(handle_); }
    uint32_t tileflow_3x3_exact_10x10_256x64_count() const { return aexrt_yolo_tileflow_3x3_exact_10x10_256x64_count(handle_); }
    uint32_t tileflow_native_fp16_count() const { return aexrt_yolo_tileflow_native_fp16_count(handle_); }
    bool executable() const { return aexrt_yolo_is_executable(handle_) != 0; }
    AexrtYoloRunTiming last_run_timing() const {
        AexrtYoloRunTiming result{};
        result.struct_size = sizeof(result);
        if (!aexrt_yolo_get_last_run_timing(handle_, &result)) {
            result.valid = 0;
        }
        return result;
    }
    uint32_t profile_event_count() const { return aexrt_yolo_profile_event_count(handle_); }
    const char* profile_event_label(uint32_t index) const { return aexrt_yolo_profile_event_label(handle_, index); }
    double profile_event_ms(uint32_t index) const { return aexrt_yolo_profile_event_ms(handle_, index); }
    /* 排空 N-buffer 流水线并回收最后一帧检测结果（同步模式下为空操作）。 */
    bool flush_pipeline(std::vector<AexrtYoloDetection>& detections) {
        detections.resize(64);
        uint32_t count = 0;
        if (!aexrt_yolo_flush_pipeline(handle_, detections.data(), 64, &count)) {
            detections.clear();
            return false;
        }
        detections.resize(count);
        return true;
    }

private:
    void reset() {
        if (handle_) {
            aexrt_yolo_destroy(handle_);
            handle_ = nullptr;
        }
    }

    AexrtYoloModel* handle_ = nullptr;
};

inline Buffer upload_float32(Device& device, const std::vector<float>& data) {
    return Buffer(aexrt_d3d12_upload_float32(device.get(), data.data(), static_cast<uint64_t>(data.size())));
}

inline Buffer allocate_float32_uav(Device& device, uint64_t element_count) {
    return Buffer(aexrt_d3d12_allocate_float32_uav(device.get(), element_count));
}

inline Buffer create_float32_view(Device& device, Buffer& buffer, uint64_t element_offset, uint64_t element_count) {
    return Buffer(aexrt_d3d12_create_float32_view(device.get(), buffer.get(), element_offset, element_count));
}

inline void relu_float32(Device& device, Buffer& input, Buffer& output, uint64_t element_count) {
    if (!aexrt_d3d12_relu_float32(device.get(), input.get(), output.get(), element_count)) {
        throw std::runtime_error("AEXRT ReLU dispatch failed");
    }
}

inline void conv2d_silu_float32(Device& device, Buffer& input, Buffer& weight, Buffer& bias, Buffer& output, const AexrtConv2DDesc& desc) {
    if (!aexrt_d3d12_conv2d_silu_float32(device.get(), input.get(), weight.get(), bias.get(), output.get(), &desc)) {
        throw std::runtime_error("AEXRT Conv2D+SiLU dispatch failed");
    }
}

inline void conv2d_float32(Device& device, Buffer& input, Buffer& weight, Buffer& bias, Buffer& output, const AexrtConv2DDesc& desc) {
    if (!aexrt_d3d12_conv2d_float32(device.get(), input.get(), weight.get(), bias.get(), output.get(), &desc)) {
        throw std::runtime_error("AEXRT Conv2D dispatch failed");
    }
}

inline void concat_conv1x1_float32(
    Device& device,
    const std::vector<AexrtBuffer*>& inputs,
    const std::vector<uint32_t>& input_channels,
    Buffer& weight,
    Buffer& bias,
    Buffer& output,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    uint32_t out_channels,
    bool silu = false) {
    if (inputs.size() != input_channels.size()) {
        throw std::runtime_error("AEXRT concat-conv inputs/channels size mismatch");
    }
    if (!aexrt_d3d12_concat_conv1x1_float32(
            device.get(),
            inputs.data(),
            input_channels.data(),
            static_cast<uint32_t>(inputs.size()),
            weight.get(),
            bias.get(),
            output.get(),
            batch,
            height,
            width,
            out_channels,
            silu ? 1u : 0u)) {
        throw std::runtime_error("AEXRT Concat+1x1Conv dispatch failed");
    }
}

inline PreparedDispatch prepare_relu_float32(Device& device, Buffer& input, Buffer& output, uint64_t element_count) {
    return PreparedDispatch(aexrt_d3d12_prepare_relu_float32(device.get(), input.get(), output.get(), element_count));
}

inline PreparedDispatch prepare_conv2d_silu_float32(Device& device, Buffer& input, Buffer& weight, Buffer& bias, Buffer& output, const AexrtConv2DDesc& desc) {
    return PreparedDispatch(aexrt_d3d12_prepare_conv2d_silu_float32(device.get(), input.get(), weight.get(), bias.get(), output.get(), &desc));
}

inline PreparedDispatch prepare_conv2d_silu_upload_float32(Device& device, Buffer& weight, Buffer& bias, Buffer& output, const AexrtConv2DDesc& desc, uint32_t ring_size = 2) {
    return PreparedDispatch(aexrt_d3d12_prepare_conv2d_silu_upload_float32(device.get(), weight.get(), bias.get(), output.get(), &desc, ring_size));
}

inline void execute(Device& device, PreparedDispatch& dispatch) {
    if (!aexrt_d3d12_execute_prepared(device.get(), dispatch.get())) {
        throw std::runtime_error("AEXRT prepared dispatch failed");
    }
}

inline void execute_upload(Device& device, PreparedDispatch& dispatch, const std::vector<float>& input) {
    if (!aexrt_d3d12_execute_prepared_upload_float32(device.get(), dispatch.get(), input.data(), static_cast<uint64_t>(input.size()))) {
        throw std::runtime_error("AEXRT prepared upload dispatch failed");
    }
}

inline std::vector<float> download_float32(Device& device, Buffer& buffer, uint64_t element_count) {
    std::vector<float> out(static_cast<size_t>(element_count));
    if (!aexrt_d3d12_download_float32(device.get(), buffer.get(), out.data(), element_count)) {
        throw std::runtime_error("AEXRT download failed");
    }
    return out;
}

inline Graph compile_relu_graph(Device& device, uint64_t element_count) {
    AexrtNodeDesc nodes[] = {
        {AEXRT_OP_RELU_FLOAT32, 0, 0, 1},
    };
    AexrtGraphDesc desc{};
    desc.element_count = element_count;
    desc.input_count = 1;
    desc.node_count = 1;
    desc.nodes = nodes;
    desc.output_value = 1;
    return Graph(aexrt_compile_graph(device.get(), &desc));
}

inline Graph compile_relu_gelu_graph(Device& device, uint64_t element_count) {
    AexrtNodeDesc nodes[] = {
        {AEXRT_OP_RELU_FLOAT32, 0, 0, 1},
        {AEXRT_OP_GELU_FLOAT32, 1, 0, 2},
    };
    AexrtGraphDesc desc{};
    desc.element_count = element_count;
    desc.input_count = 1;
    desc.node_count = 2;
    desc.nodes = nodes;
    desc.output_value = 2;
    return Graph(aexrt_compile_graph(device.get(), &desc));
}

inline Graph compile_add_relu_graph(Device& device, uint64_t element_count) {
    AexrtNodeDesc nodes[] = {
        {AEXRT_OP_ADD_FLOAT32, 0, 1, 2},
        {AEXRT_OP_RELU_FLOAT32, 2, 0, 3},
    };
    AexrtGraphDesc desc{};
    desc.element_count = element_count;
    desc.input_count = 2;
    desc.node_count = 2;
    desc.nodes = nodes;
    desc.output_value = 3;
    return Graph(aexrt_compile_graph(device.get(), &desc));
}

inline Graph load_graph_json(Device& device, const std::string& path) {
    return Graph(aexrt_load_graph_json(device.get(), path.c_str()));
}

inline YoloModel compile_yolo_from_package(Device& device, const std::string& path) {
    return YoloModel(aexrt_yolo_compile_from_package(device.get(), path.c_str()));
}

inline YoloModel load_engine(Device& device, const std::string& path) {
    return YoloModel(aexrt_yolo_load_engine(device.get(), path.c_str()));
}

inline YoloModel compile_yolo_from_onnx(Device& device, const std::string& path) {
    return YoloModel(aexrt_yolo_compile_from_onnx(device.get(), path.c_str()));
}

inline std::vector<float> run(Device& device, Graph& graph, const std::vector<float>& input) {
    std::vector<float> output(input.size());
    if (!aexrt_run(device.get(), graph.get(), input.data(), output.data())) {
        throw std::runtime_error("AEXRT graph run failed");
    }
    return output;
}

inline std::vector<float> run(Device& device, Graph& graph, const std::vector<float>& input0, const std::vector<float>& input1) {
    if (input0.size() != input1.size()) {
        throw std::runtime_error("AEXRT graph inputs must have the same size");
    }
    std::vector<float> output(input0.size());
    if (!aexrt_run2(device.get(), graph.get(), input0.data(), input1.data(), output.data())) {
        throw std::runtime_error("AEXRT graph run2 failed");
    }
    return output;
}

inline std::vector<float> run(Device& device, Graph& graph, const std::vector<std::vector<float>>& inputs) {
    if (inputs.empty()) {
        throw std::runtime_error("AEXRT graph requires at least one input");
    }
    const size_t size = inputs[0].size();
    std::vector<const float*> ptrs;
    ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (input.size() != size) {
            throw std::runtime_error("AEXRT graph inputs must have the same size");
        }
        ptrs.push_back(input.data());
    }
    std::vector<float> output(size);
    if (!aexrt_run_n(device.get(), graph.get(), ptrs.data(), static_cast<uint32_t>(ptrs.size()), output.data())) {
        throw std::runtime_error("AEXRT graph run_n failed");
    }
    return output;
}

inline std::vector<AexrtYoloDetection> run_yolo(
    Device& device,
    YoloModel& model,
    const std::vector<float>& input,
    uint32_t max_detections = 100) {
    std::vector<AexrtYoloDetection> detections(max_detections);
    uint32_t count = 0;
    if (!aexrt_yolo_run(
            device.get(),
            model.get(),
            input.data(),
            static_cast<uint64_t>(input.size()),
            detections.data(),
            max_detections,
            &count)) {
        throw std::runtime_error("AEXRT YOLO run failed");
    }
    detections.resize(count);
    return detections;
}

}  // namespace aexrt
