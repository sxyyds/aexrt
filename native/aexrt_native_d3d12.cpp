#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <string>
#include <mutex>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr const char* kDeviceCapsule = "aexrt_native_d3d12.Device";
constexpr const char* kBufferCapsule = "aexrt_native_d3d12.Buffer";
constexpr const char* kReluDispatchCapsule = "aexrt_native_d3d12.ReluDispatch";
constexpr const char* kConvSiluDispatchCapsule = "aexrt_native_d3d12.ConvSiluDispatch";
constexpr const char* kConvSiluUploadDispatchCapsule = "aexrt_native_d3d12.ConvSiluUploadDispatch";
constexpr const char* kConvSiluChainUploadDispatchCapsule = "aexrt_native_d3d12.ConvSiluChainUploadDispatch";
constexpr const char* kPreparedGraphCapsule = "aexrt_native_d3d12.PreparedGraph";

struct PreparedGraphHandle;

struct DeviceContext {
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> relu_root_signature;
    ComPtr<ID3D12PipelineState> relu_pso;
    ComPtr<ID3D12RootSignature> conv_root_signature;
    ComPtr<ID3D12PipelineState> conv_silu_pso;
    ComPtr<ID3D12PipelineState> conv_linear_pso;
    ComPtr<ID3D12PipelineState> conv1x1_silu_pso;
    ComPtr<ID3D12PipelineState> conv1x1_linear_pso;
    ComPtr<ID3D12PipelineState> conv3x3_silu_pso;
    ComPtr<ID3D12PipelineState> conv3x3_winograd_silu_pso;
    ComPtr<ID3D12PipelineState> conv3x3_winograd_packed_silu_pso;
    ComPtr<ID3D12PipelineState> conv3x3_winograd_packed_oc4_silu_pso;
    ComPtr<ID3D12RootSignature> concat_conv_root_signature;
    ComPtr<ID3D12PipelineState> concat_conv1x1_silu_pso;
    ComPtr<ID3D12PipelineState> concat_conv1x1_linear_pso;
    ComPtr<ID3D12PipelineState> concat_conv1x1_fp16_silu_pso;
    ComPtr<ID3D12PipelineState> concat_conv1x1_fp16_linear_pso;
    ComPtr<ID3D12RootSignature> concat_conv_int8_root_signature;
    ComPtr<ID3D12PipelineState> concat_conv1x1_int8_silu_pso;
    ComPtr<ID3D12PipelineState> concat_conv1x1_int8_linear_pso;
    ComPtr<ID3D12RootSignature> concat_residual_conv_root_signature;
    ComPtr<ID3D12PipelineState> concat_residual_conv1x1_silu_pso;
    ComPtr<ID3D12PipelineState> concat_residual_conv1x1_linear_pso;
    ComPtr<ID3D12PipelineState> concat_residual_conv1x1_tiled_silu_pso;
    ComPtr<ID3D12RootSignature> c2f_bottleneck_root_signature;
    ComPtr<ID3D12PipelineState> c2f_bottleneck_tiled_pso;
    ComPtr<ID3D12RootSignature> sppf_tail_root_signature;
    ComPtr<ID3D12PipelineState> sppf_tail_silu_pso;
    ComPtr<ID3D12PipelineState> sppf_tail_linear_pso;
    ComPtr<ID3D12RootSignature> unary_root_signature;
    ComPtr<ID3D12PipelineState> unary_pso;
    ComPtr<ID3D12RootSignature> binary_root_signature;
    ComPtr<ID3D12PipelineState> binary_pso;
    ComPtr<ID3D12RootSignature> slice_root_signature;
    ComPtr<ID3D12PipelineState> slice_pso;
    ComPtr<ID3D12RootSignature> concat_root_signature;
    ComPtr<ID3D12PipelineState> concat_pso;
    ComPtr<ID3D12RootSignature> resize_root_signature;
    ComPtr<ID3D12PipelineState> resize_pso;
    ComPtr<ID3D12RootSignature> maxpool_root_signature;
    ComPtr<ID3D12PipelineState> maxpool_pso;
    ComPtr<ID3D12RootSignature> transpose_root_signature;
    ComPtr<ID3D12PipelineState> transpose_pso;
    ComPtr<ID3D12RootSignature> softmax_root_signature;
    ComPtr<ID3D12PipelineState> softmax_pso;
    ComPtr<ID3D12RootSignature> dfl_root_signature;
    ComPtr<ID3D12PipelineState> dfl_project_pso;
    ComPtr<ID3D12RootSignature> yolo_root_signature;
    ComPtr<ID3D12PipelineState> yolo_decode_filter_pso;
    ComPtr<ID3D12RootSignature> yolo_head_root_signature;
    ComPtr<ID3D12PipelineState> yolo_head_decode_pso;
    ComPtr<ID3D12RootSignature> yolo_nms_root_signature;
    ComPtr<ID3D12PipelineState> yolo_nms_mark_pso;
    ComPtr<ID3D12RootSignature> yolo_topk_root_signature;
    ComPtr<ID3D12PipelineState> yolo_topk_pso;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
    uint32_t adapter_index = 0;
    bool batch_active = false;
    bool prepared_recording = false;
    PreparedGraphHandle* active_prepared_graph = nullptr;
    ComPtr<ID3D12CommandAllocator> saved_allocator;
    ComPtr<ID3D12GraphicsCommandList> saved_list;
    std::vector<ComPtr<ID3D12DescriptorHeap>> batch_heaps;
    std::vector<ComPtr<ID3D12Resource>> batch_resources;
    std::mutex mutex;

    ~DeviceContext() {
        if (fence_event) {
            CloseHandle(fence_event);
        }
    }
};

struct BufferHandle {
    DeviceContext* owner = nullptr;
    ComPtr<ID3D12Resource> resource;
    uint64_t nbytes = 0;
    uint64_t element_offset = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    std::string label;
    PyObject* parent_capsule = nullptr;
};

struct ReluDispatchHandle {
    DeviceContext* owner = nullptr;
    BufferHandle* input = nullptr;
    BufferHandle* output = nullptr;
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    uint64_t element_count = 0;
};

struct Conv2DDesc {
    uint32_t batch = 0;
    uint32_t in_channels = 0;
    uint32_t in_h = 0;
    uint32_t in_w = 0;
    uint32_t out_channels = 0;
    uint32_t out_h = 0;
    uint32_t out_w = 0;
    uint32_t kernel_h = 0;
    uint32_t kernel_w = 0;
    uint32_t stride_h = 0;
    uint32_t stride_w = 0;
    uint32_t pad_top = 0;
    uint32_t pad_left = 0;
    uint32_t dilation_h = 0;
    uint32_t dilation_w = 0;
    uint32_t groups = 0;
};

struct ConvSiluDispatchHandle {
    DeviceContext* owner = nullptr;
    BufferHandle* input = nullptr;
    BufferHandle* weight = nullptr;
    BufferHandle* bias = nullptr;
    BufferHandle* output = nullptr;
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    Conv2DDesc desc{};
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    uint64_t element_count = 0;
};

struct ConvSiluUploadRingSlot {
    ComPtr<ID3D12Resource> upload;
    void* mapped = nullptr;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
};

struct ConvSiluUploadDispatchHandle {
    DeviceContext* owner = nullptr;
    BufferHandle* input = nullptr;
    BufferHandle* weight = nullptr;
    BufferHandle* bias = nullptr;
    BufferHandle* output = nullptr;
    PyObject* device_capsule = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    Conv2DDesc desc{};
    uint64_t input_nbytes = 0;
    uint64_t element_count = 0;
    uint32_t next_slot = 0;
    std::vector<ConvSiluUploadRingSlot> slots;
};

struct ConvSiluChainSlot {
    ComPtr<ID3D12Resource> upload;
    void* mapped = nullptr;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
};

struct ConvSiluChainUploadDispatchHandle {
    DeviceContext* owner = nullptr;
    BufferHandle* input = nullptr;
    BufferHandle* output = nullptr;
    PyObject* device_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    std::vector<BufferHandle*> weights;
    std::vector<BufferHandle*> biases;
    std::vector<BufferHandle*> block_outputs;
    std::vector<BufferHandle*> owned_buffers;
    std::vector<PyObject*> weight_capsules;
    std::vector<PyObject*> bias_capsules;
    std::vector<Conv2DDesc> descs;
    uint64_t input_nbytes = 0;
    uint32_t next_slot = 0;
    std::vector<ConvSiluChainSlot> slots;
};

struct PreparedGraphHandle {
    DeviceContext* owner = nullptr;
    PyObject* device_capsule = nullptr;
    ComPtr<ID3D12CommandAllocator> command_allocator;
    ComPtr<ID3D12GraphicsCommandList> command_list;
    std::vector<ComPtr<ID3D12DescriptorHeap>> descriptor_heaps;
    std::vector<ComPtr<ID3D12Resource>> resources;
};

static std::wstring utf8_to_wide(const char* text) {
    if (!text || !*text) {
        return L"";
    }
    int needed = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (needed <= 0) {
        return L"";
    }
    std::wstring out(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, out.data(), needed);
    return out;
}

static std::string wide_to_utf8(const WCHAR* text) {
    if (!text || !*text) {
        return "";
    }
    int needed = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        return "";
    }
    std::string out(static_cast<size_t>(needed - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), needed, nullptr, nullptr);
    return out;
}

static std::string hresult_hex(HRESULT hr) {
    char buf[32];
    snprintf(buf, sizeof(buf), "0x%08X", static_cast<unsigned int>(hr));
    return std::string(buf);
}

static PyObject* raise_hr(const char* where, HRESULT hr) {
    PyErr_Format(PyExc_RuntimeError, "%s failed with HRESULT %s", where, hresult_hex(hr).c_str());
    return nullptr;
}

static bool get_hardware_adapter(uint32_t requested, ComPtr<IDXGIAdapter1>* out) {
    ComPtr<IDXGIFactory6> factory6;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory6));
    if (FAILED(hr)) {
        return false;
    }

    uint32_t hardware_index = 0;
    for (uint32_t i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = factory6->EnumAdapterByGpuPreference(
            i,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&adapter));
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr)) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        if (hardware_index == requested) {
            *out = adapter;
            return true;
        }
        hardware_index += 1;
    }

    ComPtr<IDXGIFactory1> factory1;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory1));
    if (FAILED(hr)) {
        return false;
    }

    hardware_index = 0;
    for (uint32_t i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        hr = factory1->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        if (FAILED(hr)) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        if (hardware_index == requested) {
            *out = adapter;
            return true;
        }
        hardware_index += 1;
    }
    return false;
}

static bool aexrt_pyd_env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return !(value[0] == '0' && value[1] == '\0');
}

static bool aexrt_pyd_env_has_value(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

/* WARP 软件适配器（CI / 无独立 GPU 机器的冒烟路径），与 aexrt_d3d12_runtime.cpp 保持一致。 */
static bool get_warp_adapter(ComPtr<IDXGIAdapter1>* out) {
    ComPtr<IDXGIFactory4> factory4;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory4));
    if (FAILED(hr)) {
        return false;
    }
    ComPtr<IDXGIAdapter1> warp;
    hr = factory4->EnumWarpAdapter(IID_PPV_ARGS(&warp));
    if (FAILED(hr)) {
        return false;
    }
    *out = warp;
    return true;
}

/* 原始 DXGI 适配器序号（含软件适配器）。 */
static bool get_adapter_at_raw_index(uint32_t raw_index, ComPtr<IDXGIAdapter1>* out) {
    ComPtr<IDXGIFactory1> factory1;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory1));
    if (FAILED(hr)) {
        return false;
    }
    ComPtr<IDXGIAdapter1> adapter;
    hr = factory1->EnumAdapters1(raw_index, &adapter);
    if (FAILED(hr)) {
        return false;
    }
    *out = adapter;
    return true;
}

/*
 * 与 aexrt_d3d12_runtime.cpp 相同的选择优先级：
 *   1. AEXRT_NATIVE_D3D12_WARP=1       -> WARP 软件设备
 *   2. AEXRT_NATIVE_D3D12_ADAPTER=<n>  -> 原始 DXGI 序号（软件适配器计入）
 *   3. adapter_index                   -> 纯硬件序号（原有行为）
 */
static bool get_adapter_for_device(uint32_t adapter_index, ComPtr<IDXGIAdapter1>* out) {
    if (aexrt_pyd_env_flag("AEXRT_NATIVE_D3D12_WARP")) {
        return get_warp_adapter(out);
    }
    if (aexrt_pyd_env_has_value("AEXRT_NATIVE_D3D12_ADAPTER")) {
        const char* raw = std::getenv("AEXRT_NATIVE_D3D12_ADAPTER");
        const long parsed = raw != nullptr ? std::strtol(raw, nullptr, 10) : 0;
        if (parsed >= 0) {
            return get_adapter_at_raw_index(static_cast<uint32_t>(parsed), out);
        }
    }
    return get_hardware_adapter(adapter_index, out);
}

static D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = type;
    props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask = 1;
    props.VisibleNodeMask = 1;
    return props;
}

static D3D12_RESOURCE_DESC buffer_desc(uint64_t nbytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = nbytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    return desc;
}

static HRESULT create_committed_buffer(
    ID3D12Device* device,
    D3D12_HEAP_TYPE heap_type,
    D3D12_RESOURCE_STATES initial_state,
    uint64_t nbytes,
    D3D12_RESOURCE_FLAGS flags,
    ID3D12Resource** resource) {
    auto heap = heap_properties(heap_type);
    auto desc = buffer_desc(nbytes);
    desc.Flags = flags;
    return device->CreateCommittedResource(
        &heap,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initial_state,
        nullptr,
        IID_PPV_ARGS(resource));
}

static HRESULT create_committed_buffer(
    ID3D12Device* device,
    D3D12_HEAP_TYPE heap_type,
    D3D12_RESOURCE_STATES initial_state,
    uint64_t nbytes,
    ID3D12Resource** resource) {
    return create_committed_buffer(device, heap_type, initial_state, nbytes, D3D12_RESOURCE_FLAG_NONE, resource);
}

static HRESULT begin_commands(DeviceContext* ctx) {
    HRESULT hr = ctx->allocator->Reset();
    if (FAILED(hr)) {
        return hr;
    }
    return ctx->list->Reset(ctx->allocator.Get(), nullptr);
}

static HRESULT finish_commands(DeviceContext* ctx) {
    HRESULT hr = ctx->list->Close();
    if (FAILED(hr)) {
        return hr;
    }
    ID3D12CommandList* lists[] = {ctx->list.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);

    const uint64_t value = ++ctx->fence_value;
    hr = ctx->queue->Signal(ctx->fence.Get(), value);
    if (FAILED(hr)) {
        return hr;
    }
    if (ctx->fence->GetCompletedValue() < value) {
        hr = ctx->fence->SetEventOnCompletion(value, ctx->fence_event);
        if (FAILED(hr)) {
            return hr;
        }
        WaitForSingleObject(ctx->fence_event, INFINITE);
    }
    return S_OK;
}

static HRESULT signal_and_wait(DeviceContext* ctx) {
    const uint64_t value = ++ctx->fence_value;
    HRESULT hr = ctx->queue->Signal(ctx->fence.Get(), value);
    if (FAILED(hr)) {
        return hr;
    }
    if (ctx->fence->GetCompletedValue() < value) {
        hr = ctx->fence->SetEventOnCompletion(value, ctx->fence_event);
        if (FAILED(hr)) {
            return hr;
        }
        WaitForSingleObject(ctx->fence_event, INFINITE);
    }
    return S_OK;
}

static HRESULT begin_or_join_commands(DeviceContext* ctx, bool* owns_submission) {
    if (ctx->batch_active) {
        *owns_submission = false;
        return S_OK;
    }
    *owns_submission = true;
    return begin_commands(ctx);
}

static HRESULT finish_if_owned(DeviceContext* ctx, bool owns_submission) {
    if (!owns_submission) {
        return S_OK;
    }
    return finish_commands(ctx);
}

static void keep_descriptor_heap_alive(DeviceContext* ctx, const ComPtr<ID3D12DescriptorHeap>& heap) {
    if (ctx->batch_active && heap) {
        ctx->batch_heaps.push_back(heap);
    }
}

static void keep_resource_alive(DeviceContext* ctx, const ComPtr<ID3D12Resource>& resource) {
    if (ctx->batch_active && resource) {
        ctx->batch_resources.push_back(resource);
    }
}

static void transition_if_needed(
    ID3D12GraphicsCommandList* list,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after) {
    if (before == after) {
        return;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

static void uav_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.UAV.pResource = resource;
    list->ResourceBarrier(1, &barrier);
}

static DeviceContext* get_device(PyObject* capsule) {
    return reinterpret_cast<DeviceContext*>(PyCapsule_GetPointer(capsule, kDeviceCapsule));
}

static BufferHandle* get_buffer(PyObject* capsule) {
    return reinterpret_cast<BufferHandle*>(PyCapsule_GetPointer(capsule, kBufferCapsule));
}

static ReluDispatchHandle* get_relu_dispatch(PyObject* capsule) {
    return reinterpret_cast<ReluDispatchHandle*>(PyCapsule_GetPointer(capsule, kReluDispatchCapsule));
}

static ConvSiluDispatchHandle* get_conv_silu_dispatch(PyObject* capsule) {
    return reinterpret_cast<ConvSiluDispatchHandle*>(PyCapsule_GetPointer(capsule, kConvSiluDispatchCapsule));
}

static ConvSiluUploadDispatchHandle* get_conv_silu_upload_dispatch(PyObject* capsule) {
    return reinterpret_cast<ConvSiluUploadDispatchHandle*>(PyCapsule_GetPointer(capsule, kConvSiluUploadDispatchCapsule));
}

static ConvSiluChainUploadDispatchHandle* get_conv_silu_chain_upload_dispatch(PyObject* capsule) {
    return reinterpret_cast<ConvSiluChainUploadDispatchHandle*>(PyCapsule_GetPointer(capsule, kConvSiluChainUploadDispatchCapsule));
}

static PreparedGraphHandle* get_prepared_graph(PyObject* capsule) {
    return reinterpret_cast<PreparedGraphHandle*>(PyCapsule_GetPointer(capsule, kPreparedGraphCapsule));
}

static void destroy_device(PyObject* capsule) {
    auto* ctx = reinterpret_cast<DeviceContext*>(PyCapsule_GetPointer(capsule, kDeviceCapsule));
    delete ctx;
}

static void destroy_buffer(PyObject* capsule) {
    auto* buffer = reinterpret_cast<BufferHandle*>(PyCapsule_GetPointer(capsule, kBufferCapsule));
    if (buffer) {
        Py_XDECREF(buffer->parent_capsule);
    }
    delete buffer;
}

static void destroy_relu_dispatch(PyObject* capsule) {
    auto* dispatch = reinterpret_cast<ReluDispatchHandle*>(PyCapsule_GetPointer(capsule, kReluDispatchCapsule));
    if (dispatch) {
        Py_XDECREF(dispatch->device_capsule);
        Py_XDECREF(dispatch->input_capsule);
        Py_XDECREF(dispatch->output_capsule);
    }
    delete dispatch;
}

static void destroy_conv_silu_dispatch(PyObject* capsule) {
    auto* dispatch = reinterpret_cast<ConvSiluDispatchHandle*>(PyCapsule_GetPointer(capsule, kConvSiluDispatchCapsule));
    if (dispatch) {
        Py_XDECREF(dispatch->device_capsule);
        Py_XDECREF(dispatch->input_capsule);
        Py_XDECREF(dispatch->weight_capsule);
        Py_XDECREF(dispatch->bias_capsule);
        Py_XDECREF(dispatch->output_capsule);
    }
    delete dispatch;
}

static void destroy_conv_silu_upload_dispatch(PyObject* capsule) {
    auto* dispatch = reinterpret_cast<ConvSiluUploadDispatchHandle*>(PyCapsule_GetPointer(capsule, kConvSiluUploadDispatchCapsule));
    if (dispatch) {
        D3D12_RANGE empty_range{0, 0};
        for (auto& slot : dispatch->slots) {
            if (slot.mapped && slot.upload) {
                slot.upload->Unmap(0, &empty_range);
                slot.mapped = nullptr;
            }
        }
        delete dispatch->input;
        Py_XDECREF(dispatch->device_capsule);
        Py_XDECREF(dispatch->weight_capsule);
        Py_XDECREF(dispatch->bias_capsule);
        Py_XDECREF(dispatch->output_capsule);
    }
    delete dispatch;
}

static void destroy_conv_silu_chain_upload_dispatch(PyObject* capsule) {
    auto* dispatch = reinterpret_cast<ConvSiluChainUploadDispatchHandle*>(PyCapsule_GetPointer(capsule, kConvSiluChainUploadDispatchCapsule));
    if (dispatch) {
        D3D12_RANGE empty_range{0, 0};
        for (auto& slot : dispatch->slots) {
            if (slot.mapped && slot.upload) {
                slot.upload->Unmap(0, &empty_range);
                slot.mapped = nullptr;
            }
        }
        for (auto* buffer : dispatch->owned_buffers) {
            delete buffer;
        }
        Py_XDECREF(dispatch->device_capsule);
        Py_XDECREF(dispatch->output_capsule);
        for (auto* obj : dispatch->weight_capsules) {
            Py_XDECREF(obj);
        }
        for (auto* obj : dispatch->bias_capsules) {
            Py_XDECREF(obj);
        }
    }
    delete dispatch;
}

static void destroy_prepared_graph(PyObject* capsule) {
    auto* graph = reinterpret_cast<PreparedGraphHandle*>(PyCapsule_GetPointer(capsule, kPreparedGraphCapsule));
    if (graph) {
        Py_XDECREF(graph->device_capsule);
    }
    delete graph;
}

static HRESULT create_relu_descriptor_heap(
    DeviceContext* ctx,
    BufferHandle* input,
    BufferHandle* output,
    uint64_t element_count,
    ID3D12DescriptorHeap** out_heap) {
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 2;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    HRESULT hr = ctx->device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptor_heap));
    if (FAILED(hr)) {
        return hr;
    }
    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE srv_cpu = descriptor_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE uav_cpu = srv_cpu;
    uav_cpu.ptr += descriptor_size;
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
    uav_gpu.ptr += descriptor_size;

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_desc.Buffer.FirstElement = input->element_offset;
    srv_desc.Buffer.NumElements = static_cast<UINT>(element_count);
    srv_desc.Buffer.StructureByteStride = sizeof(float);
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    ctx->device->CreateShaderResourceView(input->resource.Get(), &srv_desc, srv_cpu);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.FirstElement = output->element_offset;
    uav_desc.Buffer.NumElements = static_cast<UINT>(element_count);
    uav_desc.Buffer.StructureByteStride = sizeof(float);
    uav_desc.Buffer.CounterOffsetInBytes = 0;
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    ctx->device->CreateUnorderedAccessView(output->resource.Get(), nullptr, &uav_desc, uav_cpu);

    *out_heap = descriptor_heap.Detach();
    return S_OK;
}

static HRESULT execute_relu_descriptor_heap(
    DeviceContext* ctx,
    BufferHandle* input,
    BufferHandle* output,
    uint64_t element_count,
    ID3D12DescriptorHeap* descriptor_heap) {
    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
    uav_gpu.ptr += descriptor_size;

    std::lock_guard<std::mutex> lock(ctx->mutex);
    HRESULT hr = begin_commands(ctx);
    if (FAILED(hr)) {
        return hr;
    }

    auto input_before = input->state;
    transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    input->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    ID3D12DescriptorHeap* heaps[] = {descriptor_heap};
    ctx->list->SetDescriptorHeaps(1, heaps);
    ctx->list->SetComputeRootSignature(ctx->relu_root_signature.Get());
    ctx->list->SetPipelineState(ctx->relu_pso.Get());
    ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
    ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
    UINT constants[1] = {static_cast<UINT>(element_count)};
    ctx->list->SetComputeRoot32BitConstants(2, 1, constants, 0);
    ctx->list->Dispatch(static_cast<UINT>((element_count + 255) / 256), 1, 1);

    transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_COMMON);
    output->state = D3D12_RESOURCE_STATE_COMMON;
    transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, input_before);
    input->state = input_before;
    return finish_commands(ctx);
}

static HRESULT dispatch_relu_into(DeviceContext* ctx, BufferHandle* input, BufferHandle* output, uint64_t element_count) {
    ComPtr<ID3D12DescriptorHeap> descriptor_heap;
    HRESULT hr = create_relu_descriptor_heap(ctx, input, output, element_count, &descriptor_heap);
    if (FAILED(hr)) {
        return hr;
    }
    return execute_relu_descriptor_heap(ctx, input, output, element_count, descriptor_heap.Get());
}

static HRESULT record_prepared_relu_dispatch(DeviceContext* ctx, ReluDispatchHandle* dispatch) {
    HRESULT hr = ctx->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&dispatch->command_allocator));
    if (FAILED(hr)) {
        return hr;
    }

    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        dispatch->command_allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&dispatch->command_list));
    if (FAILED(hr)) {
        return hr;
    }

    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = dispatch->descriptor_heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
    uav_gpu.ptr += descriptor_size;

    auto input_before = dispatch->input->state;
    auto output_before = dispatch->output->state;
    transition_if_needed(
        dispatch->command_list.Get(),
        dispatch->input->resource.Get(),
        input_before,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition_if_needed(
        dispatch->command_list.Get(),
        dispatch->output->resource.Get(),
        output_before,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* heaps[] = {dispatch->descriptor_heap.Get()};
    dispatch->command_list->SetDescriptorHeaps(1, heaps);
    dispatch->command_list->SetComputeRootSignature(ctx->relu_root_signature.Get());
    dispatch->command_list->SetPipelineState(ctx->relu_pso.Get());
    dispatch->command_list->SetComputeRootDescriptorTable(0, srv_gpu);
    dispatch->command_list->SetComputeRootDescriptorTable(1, uav_gpu);
    UINT constants[1] = {static_cast<UINT>(dispatch->element_count)};
    dispatch->command_list->SetComputeRoot32BitConstants(2, 1, constants, 0);
    dispatch->command_list->Dispatch(static_cast<UINT>((dispatch->element_count + 255) / 256), 1, 1);

    transition_if_needed(
        dispatch->command_list.Get(),
        dispatch->output->resource.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        output_before);
    transition_if_needed(
        dispatch->command_list.Get(),
        dispatch->input->resource.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        input_before);

    return dispatch->command_list->Close();
}

static HRESULT execute_prepared_relu_dispatch(DeviceContext* ctx, ReluDispatchHandle* dispatch) {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ID3D12CommandList* lists[] = {dispatch->command_list.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);
    return signal_and_wait(ctx);
}

static HRESULT ensure_relu_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->relu_root_signature && ctx->relu_pso) {
        return S_OK;
    }

    const char* shader_source = R"(
StructuredBuffer<float> aexrt_input : register(t0);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConstants : register(b0) {
    uint element_count;
};

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < element_count) {
        float x = aexrt_input[i];
        aexrt_output[i] = max(x, 0.0);
    }
}
)";

    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(
        shader_source,
        strlen(shader_source),
        "aexrt_relu_float32",
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &shader,
        &errors);
    if (FAILED(hr)) {
        if (errors && error) {
            *error = static_cast<const char*>(errors->GetBufferPointer());
        }
        return hr;
    }

    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 1;
    srv_range.BaseShaderRegister = 0;
    srv_range.RegisterSpace = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 1;
    uav_range.BaseShaderRegister = 0;
    uav_range.RegisterSpace = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv_range;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav_range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.RegisterSpace = 0;
    params[2].Constants.Num32BitValues = 1;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    root_desc.NumStaticSamplers = 0;
    root_desc.pStaticSamplers = nullptr;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_errors;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_errors);
    if (FAILED(hr)) {
        if (root_errors && error) {
            *error = static_cast<const char*>(root_errors->GetBufferPointer());
        }
        return hr;
    }

    hr = ctx->device->CreateRootSignature(
        0,
        root_blob->GetBufferPointer(),
        root_blob->GetBufferSize(),
        IID_PPV_ARGS(&ctx->relu_root_signature));
    if (FAILED(hr)) {
        return hr;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = ctx->relu_root_signature.Get();
    pso_desc.CS.pShaderBytecode = shader->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader->GetBufferSize();
    return ctx->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&ctx->relu_pso));
}

static uint64_t conv_output_elements(const Conv2DDesc& desc) {
    return uint64_t(desc.batch) * desc.out_channels * desc.out_h * desc.out_w;
}

static bool is_conv1x1_fast_path(const Conv2DDesc& desc) {
    return desc.kernel_h == 1 && desc.kernel_w == 1 &&
        desc.stride_h == 1 && desc.stride_w == 1 &&
        desc.pad_top == 0 && desc.pad_left == 0 &&
        desc.dilation_h == 1 && desc.dilation_w == 1 &&
        desc.groups == 1 &&
        desc.in_h == desc.out_h && desc.in_w == desc.out_w;
}

static bool is_conv3x3_tiled_fast_path(const Conv2DDesc& desc) {
    return desc.kernel_h == 3 && desc.kernel_w == 3 &&
        desc.stride_h == 1 && desc.stride_w == 1 &&
        desc.pad_top == 1 && desc.pad_left == 1 &&
        desc.dilation_h == 1 && desc.dilation_w == 1 &&
        desc.groups == 1 &&
        desc.in_h == desc.out_h && desc.in_w == desc.out_w;
}

static bool native_winograd_enabled() {
    const char* value = std::getenv("AEXRT_NATIVE_D3D12_WINOGRAD");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static bool native_winograd_packed_enabled() {
    const char* value = std::getenv("AEXRT_NATIVE_D3D12_WINOGRAD_PACKED");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static bool native_winograd_oc4_enabled() {
    const char* value = std::getenv("AEXRT_NATIVE_D3D12_WINOGRAD_OC4");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static uint64_t buffer_float_elements(const BufferHandle* buffer) {
    return buffer ? buffer->nbytes / sizeof(float) : 0;
}

static uint64_t conv3x3_winograd_packed_weight_elements(const Conv2DDesc& desc) {
    return uint64_t(desc.out_channels) * desc.in_channels * 16;
}

static bool has_conv3x3_winograd_packed_weights(BufferHandle* weight, const Conv2DDesc& desc) {
    return buffer_float_elements(weight) >= conv3x3_winograd_packed_weight_elements(desc);
}

static bool use_conv3x3_winograd_packed(BufferHandle* weight, const Conv2DDesc& desc) {
    return is_conv3x3_tiled_fast_path(desc) &&
        native_winograd_packed_enabled() &&
        has_conv3x3_winograd_packed_weights(weight, desc);
}

static bool use_conv3x3_winograd_packed_oc4(BufferHandle* weight, const Conv2DDesc& desc) {
    return use_conv3x3_winograd_packed(weight, desc) && native_winograd_oc4_enabled();
}

static bool validate_conv_silu_args(
    DeviceContext* ctx,
    BufferHandle* input,
    BufferHandle* weight,
    BufferHandle* bias,
    BufferHandle* output,
    const Conv2DDesc& desc,
    const char** reason) {
    if (!ctx || !input || !weight || !bias || !output) {
        if (reason) *reason = "device/input/weight/bias/output is null";
        return false;
    }
    if (input->owner != ctx || weight->owner != ctx || bias->owner != ctx || output->owner != ctx) {
        if (reason) *reason = "buffer belongs to a different device";
        return false;
    }
    if (desc.batch == 0 || desc.in_channels == 0 || desc.out_channels == 0 || desc.in_h == 0 || desc.in_w == 0 ||
        desc.out_h == 0 || desc.out_w == 0 || desc.kernel_h == 0 || desc.kernel_w == 0 || desc.groups == 0 ||
        desc.stride_h == 0 || desc.stride_w == 0 || desc.dilation_h == 0 || desc.dilation_w == 0) {
        if (reason) *reason = "invalid Conv2D shape/stride/dilation/group";
        return false;
    }
    if (desc.in_channels % desc.groups != 0 || desc.out_channels % desc.groups != 0) {
        if (reason) *reason = "channels must be divisible by groups";
        return false;
    }
    const uint64_t input_elements = uint64_t(desc.batch) * desc.in_channels * desc.in_h * desc.in_w;
    const uint64_t weight_elements = uint64_t(desc.out_channels) * (desc.in_channels / desc.groups) * desc.kernel_h * desc.kernel_w;
    const uint64_t bias_elements = desc.out_channels;
    const uint64_t output_elements = conv_output_elements(desc);
    if (input->nbytes < input_elements * sizeof(float) ||
        weight->nbytes < weight_elements * sizeof(float) ||
        bias->nbytes < bias_elements * sizeof(float) ||
        output->nbytes < output_elements * sizeof(float)) {
        if (reason) *reason = "buffer is smaller than Conv2D descriptor requires";
        return false;
    }
    return true;
}

static HRESULT ensure_conv_silu_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->conv_root_signature && ctx->conv_silu_pso) {
        return S_OK;
    }

    const char* shader_source = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch;
    uint in_channels;
    uint in_h;
    uint in_w;
    uint out_channels;
    uint out_h;
    uint out_w;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint pad_top;
    uint pad_left;
    uint dilation_h;
    uint dilation_w;
    uint groups;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) {
        return;
    }

    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t = t / out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;

    uint oc_per_group = out_channels / groups;
    uint ic_per_group = in_channels / groups;
    uint group_id = oc / oc_per_group;
    uint ic_base = group_id * ic_per_group;

    float acc = aexrt_bias[oc];
    for (uint icg = 0; icg < ic_per_group; ++icg) {
        uint ic = ic_base + icg;
        for (uint kh = 0; kh < kernel_h; ++kh) {
            int ih = int(oh * stride_h + kh * dilation_h) - int(pad_top);
            if (ih < 0 || ih >= int(in_h)) {
                continue;
            }
            for (uint kw = 0; kw < kernel_w; ++kw) {
                int iw = int(ow * stride_w + kw * dilation_w) - int(pad_left);
                if (iw < 0 || iw >= int(in_w)) {
                    continue;
                }
                uint input_idx = ((n * in_channels + ic) * in_h + uint(ih)) * in_w + uint(iw);
                uint weight_idx = ((oc * ic_per_group + icg) * kernel_h + kh) * kernel_w + kw;
                acc += aexrt_input[input_idx] * aexrt_weight[weight_idx];
            }
        }
    }

    float sig = 1.0 / (1.0 + exp(-acc));
    aexrt_output[idx] = acc * sig;
}
)";

    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(
        shader_source,
        strlen(shader_source),
        "aexrt_conv2d_silu_float32",
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &shader,
        &errors);
    if (FAILED(hr)) {
        if (errors && error) {
            *error = static_cast<const char*>(errors->GetBufferPointer());
        }
        return hr;
    }

    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 3;
    srv_range.BaseShaderRegister = 0;
    srv_range.RegisterSpace = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 1;
    uav_range.BaseShaderRegister = 0;
    uav_range.RegisterSpace = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv_range;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav_range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.RegisterSpace = 0;
    params[2].Constants.Num32BitValues = 16;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_errors;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_errors);
    if (FAILED(hr)) {
        if (root_errors && error) {
            *error = static_cast<const char*>(root_errors->GetBufferPointer());
        }
        return hr;
    }

    hr = ctx->device->CreateRootSignature(
        0,
        root_blob->GetBufferPointer(),
        root_blob->GetBufferSize(),
        IID_PPV_ARGS(&ctx->conv_root_signature));
    if (FAILED(hr)) {
        return hr;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = ctx->conv_root_signature.Get();
    pso_desc.CS.pShaderBytecode = shader->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader->GetBufferSize();
    return ctx->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&ctx->conv_silu_pso));
}

static HRESULT ensure_conv_linear_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->conv_root_signature && ctx->conv_linear_pso) {
        return S_OK;
    }
    if (!ctx->conv_root_signature) {
        HRESULT hr = ensure_conv_silu_pipeline(ctx, error);
        if (FAILED(hr)) {
            return hr;
        }
    }

    const char* shader_source = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch; uint in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w;
    uint pad_top; uint pad_left; uint dilation_h; uint dilation_w; uint groups;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint oc_per_group = out_channels / groups;
    uint ic_per_group = in_channels / groups;
    uint group_id = oc / oc_per_group;
    uint ic_base = group_id * ic_per_group;
    float acc = aexrt_bias[oc];
    for (uint icg = 0; icg < ic_per_group; ++icg) {
        uint ic = ic_base + icg;
        for (uint kh = 0; kh < kernel_h; ++kh) {
            int ih = int(oh * stride_h + kh * dilation_h) - int(pad_top);
            if (ih < 0 || ih >= int(in_h)) continue;
            for (uint kw = 0; kw < kernel_w; ++kw) {
                int iw = int(ow * stride_w + kw * dilation_w) - int(pad_left);
                if (iw < 0 || iw >= int(in_w)) continue;
                uint input_idx = ((n * in_channels + ic) * in_h + uint(ih)) * in_w + uint(iw);
                uint weight_idx = ((oc * ic_per_group + icg) * kernel_h + kh) * kernel_w + kw;
                acc += aexrt_input[input_idx] * aexrt_weight[weight_idx];
            }
        }
    }
    aexrt_output[idx] = acc;
}
)";
    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(shader_source, strlen(shader_source), "aexrt_conv2d_linear_float32", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader, &errors);
    if (FAILED(hr)) {
        if (errors && error) *error = static_cast<const char*>(errors->GetBufferPointer());
        return hr;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = ctx->conv_root_signature.Get();
    pso_desc.CS.pShaderBytecode = shader->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader->GetBufferSize();
    return ctx->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&ctx->conv_linear_pso));
}

static HRESULT compile_compute_pso(
    DeviceContext* ctx,
    ID3D12RootSignature* root,
    const char* source,
    const char* name,
    ID3D12PipelineState** pso,
    std::string* error);

static HRESULT ensure_conv1x1_pipeline(DeviceContext* ctx, bool silu, std::string* error) {
    ComPtr<ID3D12PipelineState>& target = silu ? ctx->conv1x1_silu_pso : ctx->conv1x1_linear_pso;
    if (ctx->conv_root_signature && target) {
        return S_OK;
    }
    if (!ctx->conv_root_signature) {
        HRESULT hr = ensure_conv_silu_pipeline(ctx, error);
        if (FAILED(hr)) return hr;
    }

    const char* shader_source_silu = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch; uint in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w;
    uint pad_top; uint pad_left; uint dilation_h; uint dilation_w; uint groups;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    float acc = aexrt_bias[oc];
    uint wbase = oc * in_channels;
    uint ibase = (n * in_channels) * in_h * in_w + spatial;
    for (uint ic = 0; ic < in_channels; ++ic) {
        acc += aexrt_input[ibase + ic * in_h * in_w] * aexrt_weight[wbase + ic];
    }
    float sig = 1.0 / (1.0 + exp(-acc));
    aexrt_output[idx] = acc * sig;
}
)";
    const char* shader_source_linear = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch; uint in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w;
    uint pad_top; uint pad_left; uint dilation_h; uint dilation_w; uint groups;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    float acc = aexrt_bias[oc];
    uint wbase = oc * in_channels;
    uint ibase = (n * in_channels) * in_h * in_w + spatial;
    for (uint ic = 0; ic < in_channels; ++ic) {
        acc += aexrt_input[ibase + ic * in_h * in_w] * aexrt_weight[wbase + ic];
    }
    aexrt_output[idx] = acc;
}
)";
    const char* source = silu ? shader_source_silu : shader_source_linear;
    const char* name = silu ? "aexrt_conv1x1_silu_float32" : "aexrt_conv1x1_linear_float32";
    return compile_compute_pso(ctx, ctx->conv_root_signature.Get(), source, name, target.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_conv3x3_silu_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->conv_root_signature && ctx->conv3x3_silu_pso) {
        return S_OK;
    }
    if (!ctx->conv_root_signature) {
        HRESULT hr = ensure_conv_silu_pipeline(ctx, error);
        if (FAILED(hr)) return hr;
    }

    const char* shader_source = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch; uint in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w;
    uint pad_top; uint pad_left; uint dilation_h; uint dilation_w; uint groups;
};
groupshared float tile[18 * 18];
[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_tid : SV_GroupThreadID) {
    uint tile_x = group_id.x * 16;
    uint tile_y = group_id.y * 16;
    uint ow = tile_x + group_tid.x;
    uint oh = tile_y + group_tid.y;
    uint ocn = group_id.z;
    uint oc = ocn % out_channels;
    uint n = ocn / out_channels;
    uint local_index = group_tid.y * 16 + group_tid.x;

    float acc = aexrt_bias[oc];
    for (uint ic = 0; ic < in_channels; ++ic) {
        for (uint e = local_index; e < 18 * 18; e += 16 * 16) {
            uint ty = e / 18;
            uint tx = e - ty * 18;
            int ih = int(tile_y + ty) - 1;
            int iw = int(tile_x + tx) - 1;
            float v = 0.0;
            if (n < batch && ih >= 0 && ih < int(in_h) && iw >= 0 && iw < int(in_w)) {
                uint input_idx = ((n * in_channels + ic) * in_h + uint(ih)) * in_w + uint(iw);
                v = aexrt_input[input_idx];
            }
            tile[e] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        if (n < batch && ow < out_w && oh < out_h) {
            uint wbase = (oc * in_channels + ic) * 9;
            uint center = (group_tid.y + 1) * 18 + (group_tid.x + 1);
            acc += tile[center - 19] * aexrt_weight[wbase + 0];
            acc += tile[center - 18] * aexrt_weight[wbase + 1];
            acc += tile[center - 17] * aexrt_weight[wbase + 2];
            acc += tile[center - 1] * aexrt_weight[wbase + 3];
            acc += tile[center] * aexrt_weight[wbase + 4];
            acc += tile[center + 1] * aexrt_weight[wbase + 5];
            acc += tile[center + 17] * aexrt_weight[wbase + 6];
            acc += tile[center + 18] * aexrt_weight[wbase + 7];
            acc += tile[center + 19] * aexrt_weight[wbase + 8];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (n < batch && ow < out_w && oh < out_h) {
        uint out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
        float sig = 1.0 / (1.0 + exp(-acc));
        aexrt_output[out_idx] = acc * sig;
    }
}
)";
    return compile_compute_pso(
        ctx,
        ctx->conv_root_signature.Get(),
        shader_source,
        "aexrt_conv3x3_tiled_silu_float32",
        ctx->conv3x3_silu_pso.ReleaseAndGetAddressOf(),
        error);
}

static HRESULT ensure_conv3x3_winograd_silu_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->conv_root_signature && ctx->conv3x3_winograd_silu_pso) {
        return S_OK;
    }
    if (!ctx->conv_root_signature) {
        HRESULT hr = ensure_conv_silu_pipeline(ctx, error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch; uint in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w;
    uint pad_top; uint pad_left; uint dilation_h; uint dilation_w; uint groups;
};
float input_at(uint n, uint c, int h, int w0) {
    if (h < 0 || h >= int(in_h) || w0 < 0 || w0 >= int(in_w)) return 0.0;
    return aexrt_input[((n * in_channels + c) * in_h + uint(h)) * in_w + uint(w0)];
}
[numthreads(8, 8, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_tid : SV_GroupThreadID) {
    uint tile_x = group_id.x * 8 + group_tid.x;
    uint tile_y = group_id.y * 8 + group_tid.y;
    uint ocn = group_id.z;
    uint oc = ocn % out_channels;
    uint n = ocn / out_channels;
    uint oh0 = tile_y * 2;
    uint ow0 = tile_x * 2;
    if (n >= batch || oc >= out_channels || oh0 >= out_h || ow0 >= out_w) return;

    float M[4][4];
    [unroll] for (uint i = 0; i < 4; ++i) {
        [unroll] for (uint j = 0; j < 4; ++j) {
            M[i][j] = 0.0;
        }
    }

    for (uint ic = 0; ic < in_channels; ++ic) {
        float d[4][4];
        [unroll] for (uint yy = 0; yy < 4; ++yy) {
            [unroll] for (uint xx = 0; xx < 4; ++xx) {
                d[yy][xx] = input_at(n, ic, int(oh0 + yy) - 1, int(ow0 + xx) - 1);
            }
        }
        float tv[4][4];
        [unroll] for (uint r = 0; r < 4; ++r) {
            tv[r][0] = d[r][0] - d[r][2];
            tv[r][1] = d[r][1] + d[r][2];
            tv[r][2] = -d[r][1] + d[r][2];
            tv[r][3] = d[r][1] - d[r][3];
        }
        float V[4][4];
        [unroll] for (uint c = 0; c < 4; ++c) {
            V[0][c] = tv[0][c] - tv[2][c];
            V[1][c] = tv[1][c] + tv[2][c];
            V[2][c] = -tv[1][c] + tv[2][c];
            V[3][c] = tv[1][c] - tv[3][c];
        }

        uint wbase = (oc * in_channels + ic) * 9;
        float g00 = aexrt_weight[wbase + 0];
        float g01 = aexrt_weight[wbase + 1];
        float g02 = aexrt_weight[wbase + 2];
        float g10 = aexrt_weight[wbase + 3];
        float g11 = aexrt_weight[wbase + 4];
        float g12 = aexrt_weight[wbase + 5];
        float g20 = aexrt_weight[wbase + 6];
        float g21 = aexrt_weight[wbase + 7];
        float g22 = aexrt_weight[wbase + 8];
        float tg[4][3];
        tg[0][0] = g00; tg[0][1] = g01; tg[0][2] = g02;
        tg[1][0] = 0.5 * (g00 + g10 + g20);
        tg[1][1] = 0.5 * (g01 + g11 + g21);
        tg[1][2] = 0.5 * (g02 + g12 + g22);
        tg[2][0] = 0.5 * (g00 - g10 + g20);
        tg[2][1] = 0.5 * (g01 - g11 + g21);
        tg[2][2] = 0.5 * (g02 - g12 + g22);
        tg[3][0] = g20; tg[3][1] = g21; tg[3][2] = g22;
        float U[4][4];
        [unroll] for (uint r2 = 0; r2 < 4; ++r2) {
            U[r2][0] = tg[r2][0];
            U[r2][1] = 0.5 * (tg[r2][0] + tg[r2][1] + tg[r2][2]);
            U[r2][2] = 0.5 * (tg[r2][0] - tg[r2][1] + tg[r2][2]);
            U[r2][3] = tg[r2][2];
        }
        [unroll] for (uint yy2 = 0; yy2 < 4; ++yy2) {
            [unroll] for (uint xx2 = 0; xx2 < 4; ++xx2) {
                M[yy2][xx2] += U[yy2][xx2] * V[yy2][xx2];
            }
        }
    }

    float t0[4];
    float t1[4];
    [unroll] for (uint j2 = 0; j2 < 4; ++j2) {
        t0[j2] = M[0][j2] + M[1][j2] + M[2][j2];
        t1[j2] = M[1][j2] - M[2][j2] - M[3][j2];
    }
    float y00 = t0[0] + t0[1] + t0[2] + aexrt_bias[oc];
    float y01 = t0[1] - t0[2] - t0[3] + aexrt_bias[oc];
    float y10 = t1[0] + t1[1] + t1[2] + aexrt_bias[oc];
    float y11 = t1[1] - t1[2] - t1[3] + aexrt_bias[oc];

    uint base = (n * out_channels + oc) * out_h * out_w;
    if (oh0 < out_h && ow0 < out_w) {
        float s = 1.0 / (1.0 + exp(-y00));
        aexrt_output[base + oh0 * out_w + ow0] = y00 * s;
    }
    if (oh0 < out_h && ow0 + 1 < out_w) {
        float s = 1.0 / (1.0 + exp(-y01));
        aexrt_output[base + oh0 * out_w + ow0 + 1] = y01 * s;
    }
    if (oh0 + 1 < out_h && ow0 < out_w) {
        float s = 1.0 / (1.0 + exp(-y10));
        aexrt_output[base + (oh0 + 1) * out_w + ow0] = y10 * s;
    }
    if (oh0 + 1 < out_h && ow0 + 1 < out_w) {
        float s = 1.0 / (1.0 + exp(-y11));
        aexrt_output[base + (oh0 + 1) * out_w + ow0 + 1] = y11 * s;
    }
}
)";
    return compile_compute_pso(
        ctx,
        ctx->conv_root_signature.Get(),
        shader_source,
        "aexrt_conv3x3_winograd_f2x2_silu_float32",
        ctx->conv3x3_winograd_silu_pso.ReleaseAndGetAddressOf(),
        error);
}

static HRESULT ensure_conv3x3_winograd_packed_silu_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->conv_root_signature && ctx->conv3x3_winograd_packed_silu_pso) {
        return S_OK;
    }
    if (!ctx->conv_root_signature) {
        HRESULT hr = ensure_conv_silu_pipeline(ctx, error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch; uint in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w;
    uint pad_top; uint pad_left; uint dilation_h; uint dilation_w; uint groups;
};
float input_at(uint n, uint c, int h, int w0) {
    if (h < 0 || h >= int(in_h) || w0 < 0 || w0 >= int(in_w)) return 0.0;
    return aexrt_input[((n * in_channels + c) * in_h + uint(h)) * in_w + uint(w0)];
}
[numthreads(8, 8, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_tid : SV_GroupThreadID) {
    uint tile_x = group_id.x * 8 + group_tid.x;
    uint tile_y = group_id.y * 8 + group_tid.y;
    uint ocn = group_id.z;
    uint oc = ocn % out_channels;
    uint n = ocn / out_channels;
    uint oh0 = tile_y * 2;
    uint ow0 = tile_x * 2;
    if (n >= batch || oc >= out_channels || oh0 >= out_h || ow0 >= out_w) return;

    float M[4][4];
    [unroll] for (uint i = 0; i < 4; ++i) {
        [unroll] for (uint j = 0; j < 4; ++j) {
            M[i][j] = 0.0;
        }
    }

    for (uint ic = 0; ic < in_channels; ++ic) {
        float d[4][4];
        [unroll] for (uint yy = 0; yy < 4; ++yy) {
            [unroll] for (uint xx = 0; xx < 4; ++xx) {
                d[yy][xx] = input_at(n, ic, int(oh0 + yy) - 1, int(ow0 + xx) - 1);
            }
        }
        float tv[4][4];
        [unroll] for (uint r = 0; r < 4; ++r) {
            tv[r][0] = d[r][0] - d[r][2];
            tv[r][1] = d[r][1] + d[r][2];
            tv[r][2] = -d[r][1] + d[r][2];
            tv[r][3] = d[r][1] - d[r][3];
        }
        float V[4][4];
        [unroll] for (uint c = 0; c < 4; ++c) {
            V[0][c] = tv[0][c] - tv[2][c];
            V[1][c] = tv[1][c] + tv[2][c];
            V[2][c] = -tv[1][c] + tv[2][c];
            V[3][c] = tv[1][c] - tv[3][c];
        }

        uint wbase = (oc * in_channels + ic) * 16;
        [unroll] for (uint yy2 = 0; yy2 < 4; ++yy2) {
            [unroll] for (uint xx2 = 0; xx2 < 4; ++xx2) {
                M[yy2][xx2] += aexrt_weight[wbase + yy2 * 4 + xx2] * V[yy2][xx2];
            }
        }
    }

    float t0[4];
    float t1[4];
    [unroll] for (uint j2 = 0; j2 < 4; ++j2) {
        t0[j2] = M[0][j2] + M[1][j2] + M[2][j2];
        t1[j2] = M[1][j2] - M[2][j2] - M[3][j2];
    }
    float y00 = t0[0] + t0[1] + t0[2] + aexrt_bias[oc];
    float y01 = t0[1] - t0[2] - t0[3] + aexrt_bias[oc];
    float y10 = t1[0] + t1[1] + t1[2] + aexrt_bias[oc];
    float y11 = t1[1] - t1[2] - t1[3] + aexrt_bias[oc];

    uint base = (n * out_channels + oc) * out_h * out_w;
    if (oh0 < out_h && ow0 < out_w) {
        float s = 1.0 / (1.0 + exp(-y00));
        aexrt_output[base + oh0 * out_w + ow0] = y00 * s;
    }
    if (oh0 < out_h && ow0 + 1 < out_w) {
        float s = 1.0 / (1.0 + exp(-y01));
        aexrt_output[base + oh0 * out_w + ow0 + 1] = y01 * s;
    }
    if (oh0 + 1 < out_h && ow0 < out_w) {
        float s = 1.0 / (1.0 + exp(-y10));
        aexrt_output[base + (oh0 + 1) * out_w + ow0] = y10 * s;
    }
    if (oh0 + 1 < out_h && ow0 + 1 < out_w) {
        float s = 1.0 / (1.0 + exp(-y11));
        aexrt_output[base + (oh0 + 1) * out_w + ow0 + 1] = y11 * s;
    }
}
)";
    return compile_compute_pso(
        ctx,
        ctx->conv_root_signature.Get(),
        shader_source,
        "aexrt_conv3x3_winograd_f2x2_packed_silu_float32",
        ctx->conv3x3_winograd_packed_silu_pso.ReleaseAndGetAddressOf(),
        error);
}

static HRESULT ensure_conv3x3_winograd_packed_oc4_silu_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->conv_root_signature && ctx->conv3x3_winograd_packed_oc4_silu_pso) {
        return S_OK;
    }
    if (!ctx->conv_root_signature) {
        HRESULT hr = ensure_conv_silu_pipeline(ctx, error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source = R"(
StructuredBuffer<float> aexrt_input : register(t0);
StructuredBuffer<float> aexrt_weight : register(t1);
StructuredBuffer<float> aexrt_bias : register(t2);
RWStructuredBuffer<float> aexrt_output : register(u0);
cbuffer AexrtConv2DConstants : register(b0) {
    uint batch; uint in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w;
    uint pad_top; uint pad_left; uint dilation_h; uint dilation_w; uint groups;
};
float input_at(uint n, uint c, int h, int w0) {
    if (h < 0 || h >= int(in_h) || w0 < 0 || w0 >= int(in_w)) return 0.0;
    return aexrt_input[((n * in_channels + c) * in_h + uint(h)) * in_w + uint(w0)];
}
void write_silu4(uint n, uint oc, uint oh0, uint ow0, float y00, float y01, float y10, float y11) {
    if (oc >= out_channels) return;
    uint base = (n * out_channels + oc) * out_h * out_w;
    if (oh0 < out_h && ow0 < out_w) {
        float s = 1.0 / (1.0 + exp(-y00));
        aexrt_output[base + oh0 * out_w + ow0] = y00 * s;
    }
    if (oh0 < out_h && ow0 + 1 < out_w) {
        float s = 1.0 / (1.0 + exp(-y01));
        aexrt_output[base + oh0 * out_w + ow0 + 1] = y01 * s;
    }
    if (oh0 + 1 < out_h && ow0 < out_w) {
        float s = 1.0 / (1.0 + exp(-y10));
        aexrt_output[base + (oh0 + 1) * out_w + ow0] = y10 * s;
    }
    if (oh0 + 1 < out_h && ow0 + 1 < out_w) {
        float s = 1.0 / (1.0 + exp(-y11));
        aexrt_output[base + (oh0 + 1) * out_w + ow0 + 1] = y11 * s;
    }
}
[numthreads(8, 8, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_tid : SV_GroupThreadID) {
    uint oc_groups = (out_channels + 3) / 4;
    uint tile_x = group_id.x * 8 + group_tid.x;
    uint tile_y = group_id.y * 8 + group_tid.y;
    uint ocg_n = group_id.z;
    uint oc_base = (ocg_n % oc_groups) * 4;
    uint n = ocg_n / oc_groups;
    uint oh0 = tile_y * 2;
    uint ow0 = tile_x * 2;
    if (n >= batch || oh0 >= out_h || ow0 >= out_w) return;

    float M0[4][4];
    float M1[4][4];
    float M2[4][4];
    float M3[4][4];
    [unroll] for (uint i = 0; i < 4; ++i) {
        [unroll] for (uint j = 0; j < 4; ++j) {
            M0[i][j] = 0.0;
            M1[i][j] = 0.0;
            M2[i][j] = 0.0;
            M3[i][j] = 0.0;
        }
    }

    for (uint ic = 0; ic < in_channels; ++ic) {
        float d[4][4];
        [unroll] for (uint yy = 0; yy < 4; ++yy) {
            [unroll] for (uint xx = 0; xx < 4; ++xx) {
                d[yy][xx] = input_at(n, ic, int(oh0 + yy) - 1, int(ow0 + xx) - 1);
            }
        }
        float tv[4][4];
        [unroll] for (uint r = 0; r < 4; ++r) {
            tv[r][0] = d[r][0] - d[r][2];
            tv[r][1] = d[r][1] + d[r][2];
            tv[r][2] = -d[r][1] + d[r][2];
            tv[r][3] = d[r][1] - d[r][3];
        }
        float V[4][4];
        [unroll] for (uint c = 0; c < 4; ++c) {
            V[0][c] = tv[0][c] - tv[2][c];
            V[1][c] = tv[1][c] + tv[2][c];
            V[2][c] = -tv[1][c] + tv[2][c];
            V[3][c] = tv[1][c] - tv[3][c];
        }

        uint wbase0 = ((oc_base + 0) * in_channels + ic) * 16;
        uint wbase1 = ((oc_base + 1) * in_channels + ic) * 16;
        uint wbase2 = ((oc_base + 2) * in_channels + ic) * 16;
        uint wbase3 = ((oc_base + 3) * in_channels + ic) * 16;
        [unroll] for (uint yy2 = 0; yy2 < 4; ++yy2) {
            [unroll] for (uint xx2 = 0; xx2 < 4; ++xx2) {
                float v = V[yy2][xx2];
                uint p = yy2 * 4 + xx2;
                M0[yy2][xx2] += aexrt_weight[wbase0 + p] * v;
                if (oc_base + 1 < out_channels) M1[yy2][xx2] += aexrt_weight[wbase1 + p] * v;
                if (oc_base + 2 < out_channels) M2[yy2][xx2] += aexrt_weight[wbase2 + p] * v;
                if (oc_base + 3 < out_channels) M3[yy2][xx2] += aexrt_weight[wbase3 + p] * v;
            }
        }
    }

    float t00[4]; float t01[4];
    float t10[4]; float t11[4];
    float t20[4]; float t21[4];
    float t30[4]; float t31[4];
    [unroll] for (uint j2 = 0; j2 < 4; ++j2) {
        t00[j2] = M0[0][j2] + M0[1][j2] + M0[2][j2];
        t01[j2] = M0[1][j2] - M0[2][j2] - M0[3][j2];
        t10[j2] = M1[0][j2] + M1[1][j2] + M1[2][j2];
        t11[j2] = M1[1][j2] - M1[2][j2] - M1[3][j2];
        t20[j2] = M2[0][j2] + M2[1][j2] + M2[2][j2];
        t21[j2] = M2[1][j2] - M2[2][j2] - M2[3][j2];
        t30[j2] = M3[0][j2] + M3[1][j2] + M3[2][j2];
        t31[j2] = M3[1][j2] - M3[2][j2] - M3[3][j2];
    }
    write_silu4(n, oc_base + 0, oh0, ow0,
        t00[0] + t00[1] + t00[2] + aexrt_bias[oc_base + 0],
        t00[1] - t00[2] - t00[3] + aexrt_bias[oc_base + 0],
        t01[0] + t01[1] + t01[2] + aexrt_bias[oc_base + 0],
        t01[1] - t01[2] - t01[3] + aexrt_bias[oc_base + 0]);
    if (oc_base + 1 < out_channels) {
        write_silu4(n, oc_base + 1, oh0, ow0,
            t10[0] + t10[1] + t10[2] + aexrt_bias[oc_base + 1],
            t10[1] - t10[2] - t10[3] + aexrt_bias[oc_base + 1],
            t11[0] + t11[1] + t11[2] + aexrt_bias[oc_base + 1],
            t11[1] - t11[2] - t11[3] + aexrt_bias[oc_base + 1]);
    }
    if (oc_base + 2 < out_channels) {
        write_silu4(n, oc_base + 2, oh0, ow0,
            t20[0] + t20[1] + t20[2] + aexrt_bias[oc_base + 2],
            t20[1] - t20[2] - t20[3] + aexrt_bias[oc_base + 2],
            t21[0] + t21[1] + t21[2] + aexrt_bias[oc_base + 2],
            t21[1] - t21[2] - t21[3] + aexrt_bias[oc_base + 2]);
    }
    if (oc_base + 3 < out_channels) {
        write_silu4(n, oc_base + 3, oh0, ow0,
            t30[0] + t30[1] + t30[2] + aexrt_bias[oc_base + 3],
            t30[1] - t30[2] - t30[3] + aexrt_bias[oc_base + 3],
            t31[0] + t31[1] + t31[2] + aexrt_bias[oc_base + 3],
            t31[1] - t31[2] - t31[3] + aexrt_bias[oc_base + 3]);
    }
}
)";
    return compile_compute_pso(
        ctx,
        ctx->conv_root_signature.Get(),
        shader_source,
        "aexrt_conv3x3_winograd_f2x2_packed_oc4_silu_float32",
        ctx->conv3x3_winograd_packed_oc4_silu_pso.ReleaseAndGetAddressOf(),
        error);
}

static HRESULT ensure_preferred_conv3x3_silu_pipeline(DeviceContext* ctx, std::string* error) {
    if (native_winograd_packed_enabled()) {
        HRESULT hr = S_OK;
        if (native_winograd_oc4_enabled()) {
            hr = ensure_conv3x3_winograd_packed_oc4_silu_pipeline(ctx, error);
            if (FAILED(hr)) return hr;
        }
        hr = ensure_conv3x3_winograd_packed_silu_pipeline(ctx, error);
        if (FAILED(hr)) return hr;
        return ensure_conv3x3_silu_pipeline(ctx, error);
    }
    if (native_winograd_enabled()) {
        return ensure_conv3x3_winograd_silu_pipeline(ctx, error);
    }
    return ensure_conv3x3_silu_pipeline(ctx, error);
}

static HRESULT make_two_table_root(
    DeviceContext* ctx,
    UINT srv_count,
    UINT uav_count,
    UINT constant_count,
    ID3D12RootSignature** root,
    std::string* error) {
    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = srv_count;
    srv_range.BaseShaderRegister = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = uav_count;
    uav_range.BaseShaderRegister = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv_range;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav_range;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = constant_count;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_errors;
    HRESULT hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_errors);
    if (FAILED(hr)) {
        if (root_errors && error) *error = static_cast<const char*>(root_errors->GetBufferPointer());
        return hr;
    }
    return ctx->device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(root));
}

static HRESULT compile_compute_pso(
    DeviceContext* ctx,
    ID3D12RootSignature* root,
    const char* source,
    const char* name,
    ID3D12PipelineState** pso,
    std::string* error) {
    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(source, strlen(source), name, nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader, &errors);
    if (FAILED(hr)) {
        if (errors && error) *error = static_cast<const char*>(errors->GetBufferPointer());
        return hr;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root;
    pso_desc.CS.pShaderBytecode = shader->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader->GetBufferSize();
    return ctx->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(pso));
}

static HRESULT ensure_unary_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->unary_root_signature && ctx->unary_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 1, 1, 2, ctx->unary_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) { uint count; uint op; };
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= count) return;
    float v = x[i];
    if (op == 1u) v = max(v, 0.0);
    else if (op == 2u) v = 1.0 / (1.0 + exp(-v));
    else if (op == 3u) v = tanh(v);
    else if (op == 4u) v = 0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
    y[i] = v;
}
)";
    return compile_compute_pso(ctx, ctx->unary_root_signature.Get(), shader_source, "aexrt_unary_float32", ctx->unary_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_binary_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->binary_root_signature && ctx->binary_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 2, 1, 24, ctx->binary_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> a : register(t0);
StructuredBuffer<float> b : register(t1);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint count; uint op; uint rank; uint _pad0;
    uint out0; uint out1; uint out2; uint out3;
    uint a0; uint a1; uint a2; uint a3;
    uint b0; uint b1; uint b2; uint b3;
    uint astr0; uint astr1; uint astr2; uint astr3;
    uint bstr0; uint bstr1; uint bstr2; uint bstr3;
};
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count) return;
    uint d3 = idx % out3; uint t = idx / out3;
    uint d2 = t % out2; t /= out2;
    uint d1 = t % out1; uint d0 = t / out1;
    uint ia = (a0 == 1u ? 0u : d0) * astr0 + (a1 == 1u ? 0u : d1) * astr1 + (a2 == 1u ? 0u : d2) * astr2 + (a3 == 1u ? 0u : d3) * astr3;
    uint ib = (b0 == 1u ? 0u : d0) * bstr0 + (b1 == 1u ? 0u : d1) * bstr1 + (b2 == 1u ? 0u : d2) * bstr2 + (b3 == 1u ? 0u : d3) * bstr3;
    float av = a[ia];
    float bv = b[ib];
    float outv = av + bv;
    if (op == 1u) outv = av - bv;
    else if (op == 2u) outv = av * bv;
    else if (op == 3u) outv = av / bv;
    y[idx] = outv;
}
)";
    return compile_compute_pso(ctx, ctx->binary_root_signature.Get(), shader_source, "aexrt_binary_broadcast_float32", ctx->binary_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_slice_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->slice_root_signature && ctx->slice_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 1, 1, 14, ctx->slice_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint count; uint rank; uint _p0; uint _p1;
    uint in0; uint in1; uint in2; uint in3;
    uint out0; uint out1; uint out2; uint out3;
    uint axis; uint start;
};
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count) return;
    uint d3 = idx % out3; uint t = idx / out3;
    uint d2 = t % out2; t /= out2;
    uint d1 = t % out1; uint d0 = t / out1;
    if (axis == 0u) d0 += start;
    else if (axis == 1u) d1 += start;
    else if (axis == 2u) d2 += start;
    else d3 += start;
    y[idx] = x[((d0 * in1 + d1) * in2 + d2) * in3 + d3];
}
)";
    return compile_compute_pso(ctx, ctx->slice_root_signature.Get(), shader_source, "aexrt_slice_float32", ctx->slice_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_transpose_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->transpose_root_signature && ctx->transpose_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 1, 1, 17, ctx->transpose_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint count; uint rank; uint _p0; uint _p1;
    uint in0; uint in1; uint in2; uint in3;
    uint out0; uint out1; uint out2; uint out3;
    uint p0; uint p1; uint p2; uint p3; uint _p2;
};
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count) return;
    uint od[4];
    od[3] = idx % out3; uint t = idx / out3;
    od[2] = t % out2; t /= out2;
    od[1] = t % out1; od[0] = t / out1;
    uint id[4] = {0u, 0u, 0u, 0u};
    uint p[4] = {p0, p1, p2, p3};
    [unroll] for (uint i = 0; i < 4u; ++i) id[p[i]] = od[i];
    y[idx] = x[((id[0] * in1 + id[1]) * in2 + id[2]) * in3 + id[3]];
}
)";
    return compile_compute_pso(ctx, ctx->transpose_root_signature.Get(), shader_source, "aexrt_transpose_float32", ctx->transpose_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_resize_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->resize_root_signature && ctx->resize_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 1, 1, 6, ctx->resize_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) { uint n; uint c; uint in_h; uint in_w; uint out_h; uint out_w; };
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = n * c * out_h * out_w;
    if (idx >= total) return;
    uint ox = idx % out_w; uint t = idx / out_w;
    uint oy = t % out_h; t /= out_h;
    uint ch = t % c; uint bn = t / c;
    uint iy = min((oy * in_h) / out_h, in_h - 1u);
    uint ix = min((ox * in_w) / out_w, in_w - 1u);
    y[idx] = x[((bn * c + ch) * in_h + iy) * in_w + ix];
}
)";
    return compile_compute_pso(ctx, ctx->resize_root_signature.Get(), shader_source, "aexrt_resize_nearest_float32", ctx->resize_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_maxpool_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->maxpool_root_signature && ctx->maxpool_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 1, 1, 13, ctx->maxpool_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint n; uint c; uint in_h; uint in_w; uint out_h; uint out_w;
    uint kernel_h; uint kernel_w; uint stride_h; uint stride_w; uint pad_top; uint pad_left; uint dilation;
};
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = n * c * out_h * out_w;
    if (idx >= total) return;
    uint ox = idx % out_w; uint t = idx / out_w;
    uint oy = t % out_h; t /= out_h;
    uint ch = t % c; uint bn = t / c;
    float m = -3.402823466e+38;
    for (uint ky = 0; ky < kernel_h; ++ky) {
        int iy = int(oy * stride_h + ky * dilation) - int(pad_top);
        if (iy < 0 || iy >= int(in_h)) continue;
        for (uint kx = 0; kx < kernel_w; ++kx) {
            int ix = int(ox * stride_w + kx * dilation) - int(pad_left);
            if (ix < 0 || ix >= int(in_w)) continue;
            m = max(m, x[((bn * c + ch) * in_h + uint(iy)) * in_w + uint(ix)]);
        }
    }
    y[idx] = m;
}
)";
    return compile_compute_pso(ctx, ctx->maxpool_root_signature.Get(), shader_source, "aexrt_maxpool2d_float32", ctx->maxpool_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_softmax_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->softmax_root_signature && ctx->softmax_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 1, 1, 6, ctx->softmax_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) { uint n; uint c; uint h; uint w; uint axis; uint groups; };
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint g = tid.x;
    if (g >= groups || axis != 1u) return;
    uint ww = g % w; uint t = g / w;
    uint hh = t % h; uint bn = t / h;
    float m = -3.402823466e+38;
    for (uint cc = 0; cc < c; ++cc) {
        m = max(m, x[((bn * c + cc) * h + hh) * w + ww]);
    }
    float s = 0.0;
    for (uint cc2 = 0; cc2 < c; ++cc2) {
        s += exp(x[((bn * c + cc2) * h + hh) * w + ww] - m);
    }
    for (uint cc3 = 0; cc3 < c; ++cc3) {
        uint idx = ((bn * c + cc3) * h + hh) * w + ww;
        y[idx] = exp(x[idx] - m) / s;
    }
}
)";
    return compile_compute_pso(ctx, ctx->softmax_root_signature.Get(), shader_source, "aexrt_softmax_axis1_float32", ctx->softmax_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_concat_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->concat_root_signature && ctx->concat_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 8, 1, 18, ctx->concat_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint count; uint rank; uint axis; uint parts;
    uint out0; uint out1; uint out2; uint out3;
    uint s0; uint s1; uint s2; uint s3; uint s4; uint s5; uint s6; uint s7;
    uint _p0; uint _p1;
};
float read_part(uint part, uint idx) {
    if (part == 0u) return x0[idx];
    if (part == 1u) return x1[idx];
    if (part == 2u) return x2[idx];
    if (part == 3u) return x3[idx];
    if (part == 4u) return x4[idx];
    if (part == 5u) return x5[idx];
    if (part == 6u) return x6[idx];
    return x7[idx];
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count) return;
    uint d3 = idx % out3; uint t = idx / out3;
    uint d2 = t % out2; t /= out2;
    uint d1 = t % out1; uint d0 = t / out1;
    uint dims[4] = {d0, d1, d2, d3};
    uint sizes[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    uint coord = dims[axis];
    uint part = 0u;
    uint offset = 0u;
    [unroll] for (uint i = 0; i < 8u; ++i) {
        if (i >= parts) break;
        uint next = offset + sizes[i];
        if (coord < next) {
            part = i;
            break;
        }
        offset = next;
    }
    dims[axis] = coord - offset;
    uint pshape0 = out0;
    uint pshape1 = out1;
    uint pshape2 = out2;
    uint pshape3 = out3;
    if (axis == 0u) pshape0 = sizes[part];
    else if (axis == 1u) pshape1 = sizes[part];
    else if (axis == 2u) pshape2 = sizes[part];
    else pshape3 = sizes[part];
    uint src_idx = ((dims[0] * pshape1 + dims[1]) * pshape2 + dims[2]) * pshape3 + dims[3];
    y[idx] = read_part(part, src_idx);
}
)";
    return compile_compute_pso(ctx, ctx->concat_root_signature.Get(), shader_source, "aexrt_concat_float32", ctx->concat_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_dfl_project_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->dfl_root_signature && ctx->dfl_project_pso) return S_OK;
    HRESULT hr = make_two_table_root(ctx, 1, 1, 4, ctx->dfl_root_signature.ReleaseAndGetAddressOf(), error);
    if (FAILED(hr)) return hr;
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) { uint bins; uint box_dims; uint anchors; uint count; };
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count) return;
    uint a = idx % anchors;
    uint d = (idx / anchors) % box_dims;
    float acc = 0.0;
    for (uint b = 0; b < bins; ++b) {
        acc += x[(b * box_dims + d) * anchors + a] * float(b);
    }
    y[idx] = acc;
}
)";
    return compile_compute_pso(ctx, ctx->dfl_root_signature.Get(), shader_source, "aexrt_dfl_project_float32", ctx->dfl_project_pso.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_concat_conv1x1_pipeline(DeviceContext* ctx, bool silu, std::string* error) {
    ComPtr<ID3D12PipelineState>& target = silu ? ctx->concat_conv1x1_silu_pso : ctx->concat_conv1x1_linear_pso;
    if (ctx->concat_conv_root_signature && target) {
        return S_OK;
    }
    if (!ctx->concat_conv_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 10, 1, 16, ctx->concat_conv_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source_silu = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<float> weight : register(t8);
StructuredBuffer<float> bias : register(t9);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
};
float read_input(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float acc = bias[oc];
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            acc += read_input(source, n, ic, spatial) * weight[oc * total_in_channels + global_c + ic];
        }
        global_c += count;
    }
    float sig = 1.0 / (1.0 + exp(-acc));
    y[idx] = acc * sig;
}
)";
    const char* shader_source_linear = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<float> weight : register(t8);
StructuredBuffer<float> bias : register(t9);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
};
float read_input(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float acc = bias[oc];
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            acc += read_input(source, n, ic, spatial) * weight[oc * total_in_channels + global_c + ic];
        }
        global_c += count;
    }
    y[idx] = acc;
}
)";
    const char* source = silu ? shader_source_silu : shader_source_linear;
    const char* name = silu ? "aexrt_concat_conv1x1_silu_float32" : "aexrt_concat_conv1x1_linear_float32";
    return compile_compute_pso(ctx, ctx->concat_conv_root_signature.Get(), source, name, target.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_concat_conv1x1_fp16_pipeline(DeviceContext* ctx, bool silu, std::string* error) {
    ComPtr<ID3D12PipelineState>& target = silu ? ctx->concat_conv1x1_fp16_silu_pso : ctx->concat_conv1x1_fp16_linear_pso;
    if (ctx->concat_conv_root_signature && target) {
        return S_OK;
    }
    if (!ctx->concat_conv_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 10, 1, 16, ctx->concat_conv_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source_silu = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<uint> weight : register(t8);
StructuredBuffer<float> bias : register(t9);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
};
float read_input(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
float read_weight(uint index) {
    uint packed = weight[index >> 1];
    uint h = ((index & 1u) == 0u) ? (packed & 0xffffu) : (packed >> 16);
    return f16tof32(h);
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float acc = bias[oc];
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            acc += read_input(source, n, ic, spatial) * read_weight(oc * total_in_channels + global_c + ic);
        }
        global_c += count;
    }
    float sig = 1.0 / (1.0 + exp(-acc));
    y[idx] = acc * sig;
}
)";
    const char* shader_source_linear = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<uint> weight : register(t8);
StructuredBuffer<float> bias : register(t9);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
};
float read_input(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
float read_weight(uint index) {
    uint packed = weight[index >> 1];
    uint h = ((index & 1u) == 0u) ? (packed & 0xffffu) : (packed >> 16);
    return f16tof32(h);
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float acc = bias[oc];
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            acc += read_input(source, n, ic, spatial) * read_weight(oc * total_in_channels + global_c + ic);
        }
        global_c += count;
    }
    y[idx] = acc;
}
)";
    const char* source = silu ? shader_source_silu : shader_source_linear;
    const char* name = silu ? "aexrt_concat_conv1x1_silu_fp16_weight" : "aexrt_concat_conv1x1_linear_fp16_weight";
    return compile_compute_pso(ctx, ctx->concat_conv_root_signature.Get(), source, name, target.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_concat_conv1x1_int8_pipeline(DeviceContext* ctx, bool silu, std::string* error) {
    ComPtr<ID3D12PipelineState>& target = silu ? ctx->concat_conv1x1_int8_silu_pso : ctx->concat_conv1x1_int8_linear_pso;
    if (ctx->concat_conv_int8_root_signature && target) {
        return S_OK;
    }
    if (!ctx->concat_conv_int8_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 11, 1, 20, ctx->concat_conv_int8_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source_silu = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<uint> weight : register(t8);
StructuredBuffer<float> scale : register(t9);
StructuredBuffer<float> bias : register(t10);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
    uint act_inv_scale_bits; uint _p0; uint _p1; uint _p2;
};
float read_input(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
int signed_byte(uint packed, uint lane) {
    uint b = (packed >> (lane * 8u)) & 0xffu;
    return b >= 128u ? int(b) - 256 : int(b);
}
int read_weight(uint index) {
    return signed_byte(weight[index >> 2], index & 3u);
}
int quant_act(float v, float inv_scale) {
    float qf = round(v * inv_scale);
    qf = min(127.0, max(-128.0, qf));
    return int(qf);
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float inv_scale = asfloat(act_inv_scale_bits);
    int acc_i = 0;
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            int qa = quant_act(read_input(source, n, ic, spatial), inv_scale);
            int qw = read_weight(oc * total_in_channels + global_c + ic);
            acc_i += qa * qw;
        }
        global_c += count;
    }
    float acc = bias[oc] + float(acc_i) * scale[oc];
    float sig = 1.0 / (1.0 + exp(-acc));
    y[idx] = acc * sig;
}
)";
    const char* shader_source_linear = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<uint> weight : register(t8);
StructuredBuffer<float> scale : register(t9);
StructuredBuffer<float> bias : register(t10);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
    uint act_inv_scale_bits; uint _p0; uint _p1; uint _p2;
};
float read_input(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
int signed_byte(uint packed, uint lane) {
    uint b = (packed >> (lane * 8u)) & 0xffu;
    return b >= 128u ? int(b) - 256 : int(b);
}
int read_weight(uint index) {
    return signed_byte(weight[index >> 2], index & 3u);
}
int quant_act(float v, float inv_scale) {
    float qf = round(v * inv_scale);
    qf = min(127.0, max(-128.0, qf));
    return int(qf);
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float inv_scale = asfloat(act_inv_scale_bits);
    int acc_i = 0;
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            int qa = quant_act(read_input(source, n, ic, spatial), inv_scale);
            int qw = read_weight(oc * total_in_channels + global_c + ic);
            acc_i += qa * qw;
        }
        global_c += count;
    }
    y[idx] = bias[oc] + float(acc_i) * scale[oc];
}
)";
    const char* source = silu ? shader_source_silu : shader_source_linear;
    const char* name = silu ? "aexrt_concat_conv1x1_silu_int8_activation_weight" : "aexrt_concat_conv1x1_linear_int8_activation_weight";
    return compile_compute_pso(ctx, ctx->concat_conv_int8_root_signature.Get(), source, name, target.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_concat_residual_conv1x1_pipeline(DeviceContext* ctx, bool silu, std::string* error) {
    ComPtr<ID3D12PipelineState>& target = silu ? ctx->concat_residual_conv1x1_silu_pso : ctx->concat_residual_conv1x1_linear_pso;
    if (ctx->concat_residual_conv_root_signature && target) {
        return S_OK;
    }
    if (!ctx->concat_residual_conv_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 18, 1, 24, ctx->concat_residual_conv_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source_silu = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<float> r0 : register(t8);
StructuredBuffer<float> r1 : register(t9);
StructuredBuffer<float> r2 : register(t10);
StructuredBuffer<float> r3 : register(t11);
StructuredBuffer<float> r4 : register(t12);
StructuredBuffer<float> r5 : register(t13);
StructuredBuffer<float> r6 : register(t14);
StructuredBuffer<float> r7 : register(t15);
StructuredBuffer<float> weight : register(t16);
StructuredBuffer<float> bias : register(t17);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
    uint q0; uint q1; uint q2; uint q3; uint q4; uint q5; uint q6; uint q7;
};
float read_primary(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
float read_residual(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return r0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return r1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return r2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return r3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return r4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return r5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return r6[(n * c6 + local_c) * in_h * in_w + spatial];
    return r7[(n * c7 + local_c) * in_h * in_w + spatial];
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    uint residuals[8] = {q0, q1, q2, q3, q4, q5, q6, q7};
    float acc = bias[oc];
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            float v = read_primary(source, n, ic, spatial);
            if (residuals[source] != 0) {
                v += read_residual(source, n, ic, spatial);
            }
            acc += v * weight[oc * total_in_channels + global_c + ic];
        }
        global_c += count;
    }
    float sig = 1.0 / (1.0 + exp(-acc));
    y[idx] = acc * sig;
}
)";
    const char* shader_source_linear = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<float> r0 : register(t8);
StructuredBuffer<float> r1 : register(t9);
StructuredBuffer<float> r2 : register(t10);
StructuredBuffer<float> r3 : register(t11);
StructuredBuffer<float> r4 : register(t12);
StructuredBuffer<float> r5 : register(t13);
StructuredBuffer<float> r6 : register(t14);
StructuredBuffer<float> r7 : register(t15);
StructuredBuffer<float> weight : register(t16);
StructuredBuffer<float> bias : register(t17);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
    uint q0; uint q1; uint q2; uint q3; uint q4; uint q5; uint q6; uint q7;
};
float read_primary(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
float read_residual(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return r0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return r1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return r2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return r3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return r4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return r5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return r6[(n * c6 + local_c) * in_h * in_w + spatial];
    return r7[(n * c7 + local_c) * in_h * in_w + spatial];
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;
    uint ow = idx % out_w;
    uint t = idx / out_w;
    uint oh = t % out_h;
    t /= out_h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint spatial = oh * in_w + ow;
    uint channels[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    uint residuals[8] = {q0, q1, q2, q3, q4, q5, q6, q7};
    float acc = bias[oc];
    uint global_c = 0;
    [unroll]
    for (uint source = 0; source < 8; ++source) {
        uint count = channels[source];
        if (source >= input_count) count = 0;
        for (uint ic = 0; ic < count; ++ic) {
            float v = read_primary(source, n, ic, spatial);
            if (residuals[source] != 0) {
                v += read_residual(source, n, ic, spatial);
            }
            acc += v * weight[oc * total_in_channels + global_c + ic];
        }
        global_c += count;
    }
    y[idx] = acc;
}
)";
    const char* source = silu ? shader_source_silu : shader_source_linear;
    const char* name = silu ? "aexrt_concat_residual_conv1x1_silu_float32" : "aexrt_concat_residual_conv1x1_linear_float32";
    return compile_compute_pso(ctx, ctx->concat_residual_conv_root_signature.Get(), source, name, target.ReleaseAndGetAddressOf(), error);
}

static HRESULT ensure_concat_residual_conv1x1_tiled_silu_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->concat_residual_conv_root_signature && ctx->concat_residual_conv1x1_tiled_silu_pso) {
        return S_OK;
    }
    if (!ctx->concat_residual_conv_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 18, 1, 24, ctx->concat_residual_conv_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source = R"(
StructuredBuffer<float> x0 : register(t0);
StructuredBuffer<float> x1 : register(t1);
StructuredBuffer<float> x2 : register(t2);
StructuredBuffer<float> x3 : register(t3);
StructuredBuffer<float> x4 : register(t4);
StructuredBuffer<float> x5 : register(t5);
StructuredBuffer<float> x6 : register(t6);
StructuredBuffer<float> x7 : register(t7);
StructuredBuffer<float> r0 : register(t8);
StructuredBuffer<float> r1 : register(t9);
StructuredBuffer<float> r2 : register(t10);
StructuredBuffer<float> r3 : register(t11);
StructuredBuffer<float> r4 : register(t12);
StructuredBuffer<float> r5 : register(t13);
StructuredBuffer<float> r6 : register(t14);
StructuredBuffer<float> r7 : register(t15);
StructuredBuffer<float> weight : register(t16);
StructuredBuffer<float> bias : register(t17);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint total_in_channels; uint in_h; uint in_w;
    uint out_channels; uint out_h; uint out_w; uint input_count;
    uint c0; uint c1; uint c2; uint c3; uint c4; uint c5; uint c6; uint c7;
    uint q0; uint q1; uint q2; uint q3; uint q4; uint q5; uint q6; uint q7;
};
groupshared float tile_x[32 * 16];
groupshared float tile_w[8 * 32];
float read_primary(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return x0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return x1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return x2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return x3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return x4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return x5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return x6[(n * c6 + local_c) * in_h * in_w + spatial];
    return x7[(n * c7 + local_c) * in_h * in_w + spatial];
}
float read_residual(uint source, uint n, uint local_c, uint spatial) {
    if (source == 0) return r0[(n * c0 + local_c) * in_h * in_w + spatial];
    if (source == 1) return r1[(n * c1 + local_c) * in_h * in_w + spatial];
    if (source == 2) return r2[(n * c2 + local_c) * in_h * in_w + spatial];
    if (source == 3) return r3[(n * c3 + local_c) * in_h * in_w + spatial];
    if (source == 4) return r4[(n * c4 + local_c) * in_h * in_w + spatial];
    if (source == 5) return r5[(n * c5 + local_c) * in_h * in_w + spatial];
    if (source == 6) return r6[(n * c6 + local_c) * in_h * in_w + spatial];
    return r7[(n * c7 + local_c) * in_h * in_w + spatial];
}
float read_logical(uint n, uint global_c, uint spatial) {
    uint base = 0;
    if (global_c < base + c0) {
        float v = read_primary(0, n, global_c - base, spatial);
        if (q0 != 0) v += read_residual(0, n, global_c - base, spatial);
        return v;
    }
    base += c0;
    if (global_c < base + c1) {
        float v = read_primary(1, n, global_c - base, spatial);
        if (q1 != 0) v += read_residual(1, n, global_c - base, spatial);
        return v;
    }
    base += c1;
    if (global_c < base + c2) {
        float v = read_primary(2, n, global_c - base, spatial);
        if (q2 != 0) v += read_residual(2, n, global_c - base, spatial);
        return v;
    }
    base += c2;
    if (global_c < base + c3) {
        float v = read_primary(3, n, global_c - base, spatial);
        if (q3 != 0) v += read_residual(3, n, global_c - base, spatial);
        return v;
    }
    base += c3;
    if (global_c < base + c4) {
        float v = read_primary(4, n, global_c - base, spatial);
        if (q4 != 0) v += read_residual(4, n, global_c - base, spatial);
        return v;
    }
    base += c4;
    if (global_c < base + c5) {
        float v = read_primary(5, n, global_c - base, spatial);
        if (q5 != 0) v += read_residual(5, n, global_c - base, spatial);
        return v;
    }
    base += c5;
    if (global_c < base + c6) {
        float v = read_primary(6, n, global_c - base, spatial);
        if (q6 != 0) v += read_residual(6, n, global_c - base, spatial);
        return v;
    }
    base += c6;
    float v = read_primary(7, n, global_c - base, spatial);
    if (q7 != 0) v += read_residual(7, n, global_c - base, spatial);
    return v;
}
[numthreads(16, 8, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_tid : SV_GroupThreadID) {
    uint spatial_count = out_h * out_w;
    uint spatial = group_id.x * 16 + group_tid.x;
    uint oc = group_id.y * 8 + group_tid.y;
    uint n = group_id.z;
    uint local = group_tid.y * 16 + group_tid.x;
    float acc = 0.0;
    for (uint k0 = 0; k0 < total_in_channels; k0 += 32) {
        for (uint e = local; e < 32 * 16; e += 16 * 8) {
            uint k = e / 16;
            uint s = e - k * 16;
            uint gc = k0 + k;
            uint sp = group_id.x * 16 + s;
            float xv = 0.0;
            if (n < batch && gc < total_in_channels && sp < spatial_count) {
                xv = read_logical(n, gc, sp);
            }
            tile_x[e] = xv;
        }
        for (uint e2 = local; e2 < 8 * 32; e2 += 16 * 8) {
            uint row = e2 / 32;
            uint k = e2 - row * 32;
            uint out_c = group_id.y * 8 + row;
            uint gc = k0 + k;
            float wv = 0.0;
            if (out_c < out_channels && gc < total_in_channels) {
                wv = weight[out_c * total_in_channels + gc];
            }
            tile_w[e2] = wv;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint k = 0; k < 32; ++k) {
            acc += tile_x[k * 16 + group_tid.x] * tile_w[group_tid.y * 32 + k];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (n < batch && oc < out_channels && spatial < spatial_count) {
        acc += bias[oc];
        float sig = 1.0 / (1.0 + exp(-acc));
        y[(n * out_channels + oc) * out_h * out_w + spatial] = acc * sig;
    }
}
)";
    return compile_compute_pso(
        ctx,
        ctx->concat_residual_conv_root_signature.Get(),
        shader_source,
        "aexrt_concat_residual_conv1x1_tiled_silu_float32",
        ctx->concat_residual_conv1x1_tiled_silu_pso.ReleaseAndGetAddressOf(),
        error);
}

static HRESULT ensure_c2f_bottleneck_tiled_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->c2f_bottleneck_root_signature && ctx->c2f_bottleneck_tiled_pso) {
        return S_OK;
    }
    if (!ctx->c2f_bottleneck_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 5, 1, 8, ctx->c2f_bottleneck_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source = R"(
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> w1 : register(t1);
StructuredBuffer<float> b1 : register(t2);
StructuredBuffer<float> w2 : register(t3);
StructuredBuffer<float> b2 : register(t4);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint in_channels; uint h; uint w;
    uint mid_channels; uint out_channels; uint total; uint reserved0;
};
groupshared float first_tile[10 * 10];
groupshared float second_w[16 * 9];
[numthreads(8, 8, 4)]
void main(uint3 group_id : SV_GroupID, uint3 group_tid : SV_GroupThreadID) {
    uint oc_tiles = (out_channels + 15) / 16;
    uint n = group_id.z / oc_tiles;
    uint oc0 = (group_id.z - n * oc_tiles) * 16 + group_tid.z;
    uint oc1 = oc0 + 4;
    uint oc2 = oc0 + 8;
    uint oc3 = oc0 + 12;
    uint oh = group_id.y * 8 + group_tid.y;
    uint ow = group_id.x * 8 + group_tid.x;
    uint local = (group_tid.z * 8 + group_tid.y) * 8 + group_tid.x;
    float acc0 = oc0 < out_channels ? b2[oc0] : 0.0;
    float acc1 = oc1 < out_channels ? b2[oc1] : 0.0;
    float acc2 = oc2 < out_channels ? b2[oc2] : 0.0;
    float acc3 = oc3 < out_channels ? b2[oc3] : 0.0;
    for (uint mc = 0; mc < mid_channels; ++mc) {
        for (uint ew = local; ew < 16 * 9; ew += 8 * 8 * 4) {
            uint row = ew / 9;
            uint kk = ew - row * 9;
            uint out_c = (group_id.z - n * oc_tiles) * 16 + row;
            float wv = 0.0;
            if (out_c < out_channels) {
                wv = w2[(out_c * mid_channels + mc) * 9 + kk];
            }
            second_w[ew] = wv;
        }
        for (uint e = local; e < 100; e += 8 * 8 * 4) {
            uint ty = e / 10;
            uint tx = e - ty * 10;
            int ih0 = int(group_id.y * 8 + ty) - 1;
            int iw0 = int(group_id.x * 8 + tx) - 1;
            float v = 0.0;
            if (n < batch && ih0 >= 0 && ih0 < int(h) && iw0 >= 0 && iw0 < int(w)) {
                v = b1[mc];
                for (uint ic = 0; ic < in_channels; ++ic) {
                    for (uint ky = 0; ky < 3; ++ky) {
                        int ih1 = ih0 + int(ky) - 1;
                        if (ih1 < 0 || ih1 >= int(h)) continue;
                        for (uint kx = 0; kx < 3; ++kx) {
                            int iw1 = iw0 + int(kx) - 1;
                            if (iw1 < 0 || iw1 >= int(w)) continue;
                            uint input_idx = ((n * in_channels + ic) * h + uint(ih1)) * w + uint(iw1);
                            uint weight_idx = ((mc * in_channels + ic) * 3 + ky) * 3 + kx;
                            v += x[input_idx] * w1[weight_idx];
                        }
                    }
                }
                float sig = 1.0 / (1.0 + exp(-v));
                v = v * sig;
            }
            first_tile[e] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        if (n < batch && oh < h && ow < w) {
            uint wbase0 = (oc0 * mid_channels + mc) * 9;
            uint wbase1 = (oc1 * mid_channels + mc) * 9;
            uint wbase2 = (oc2 * mid_channels + mc) * 9;
            uint wbase3 = (oc3 * mid_channels + mc) * 9;
            uint center = group_tid.y * 10 + group_tid.x;
            float v0 = first_tile[center];
            float v1 = first_tile[center + 1];
            float v2 = first_tile[center + 2];
            float v3 = first_tile[center + 10];
            float v4 = first_tile[center + 11];
            float v5 = first_tile[center + 12];
            float v6 = first_tile[center + 20];
            float v7 = first_tile[center + 21];
            float v8 = first_tile[center + 22];
            if (oc0 < out_channels) {
                acc0 += v0 * second_w[group_tid.z * 9 + 0];
                acc0 += v1 * second_w[group_tid.z * 9 + 1];
                acc0 += v2 * second_w[group_tid.z * 9 + 2];
                acc0 += v3 * second_w[group_tid.z * 9 + 3];
                acc0 += v4 * second_w[group_tid.z * 9 + 4];
                acc0 += v5 * second_w[group_tid.z * 9 + 5];
                acc0 += v6 * second_w[group_tid.z * 9 + 6];
                acc0 += v7 * second_w[group_tid.z * 9 + 7];
                acc0 += v8 * second_w[group_tid.z * 9 + 8];
            }
            if (oc1 < out_channels) {
                acc1 += v0 * second_w[(group_tid.z + 4) * 9 + 0];
                acc1 += v1 * second_w[(group_tid.z + 4) * 9 + 1];
                acc1 += v2 * second_w[(group_tid.z + 4) * 9 + 2];
                acc1 += v3 * second_w[(group_tid.z + 4) * 9 + 3];
                acc1 += v4 * second_w[(group_tid.z + 4) * 9 + 4];
                acc1 += v5 * second_w[(group_tid.z + 4) * 9 + 5];
                acc1 += v6 * second_w[(group_tid.z + 4) * 9 + 6];
                acc1 += v7 * second_w[(group_tid.z + 4) * 9 + 7];
                acc1 += v8 * second_w[(group_tid.z + 4) * 9 + 8];
            }
            if (oc2 < out_channels) {
                acc2 += v0 * second_w[(group_tid.z + 8) * 9 + 0];
                acc2 += v1 * second_w[(group_tid.z + 8) * 9 + 1];
                acc2 += v2 * second_w[(group_tid.z + 8) * 9 + 2];
                acc2 += v3 * second_w[(group_tid.z + 8) * 9 + 3];
                acc2 += v4 * second_w[(group_tid.z + 8) * 9 + 4];
                acc2 += v5 * second_w[(group_tid.z + 8) * 9 + 5];
                acc2 += v6 * second_w[(group_tid.z + 8) * 9 + 6];
                acc2 += v7 * second_w[(group_tid.z + 8) * 9 + 7];
                acc2 += v8 * second_w[(group_tid.z + 8) * 9 + 8];
            }
            if (oc3 < out_channels) {
                acc3 += v0 * second_w[(group_tid.z + 12) * 9 + 0];
                acc3 += v1 * second_w[(group_tid.z + 12) * 9 + 1];
                acc3 += v2 * second_w[(group_tid.z + 12) * 9 + 2];
                acc3 += v3 * second_w[(group_tid.z + 12) * 9 + 3];
                acc3 += v4 * second_w[(group_tid.z + 12) * 9 + 4];
                acc3 += v5 * second_w[(group_tid.z + 12) * 9 + 5];
                acc3 += v6 * second_w[(group_tid.z + 12) * 9 + 6];
                acc3 += v7 * second_w[(group_tid.z + 12) * 9 + 7];
                acc3 += v8 * second_w[(group_tid.z + 12) * 9 + 8];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (n < batch && oh < h && ow < w) {
        if (oc0 < out_channels) {
            float sig0 = 1.0 / (1.0 + exp(-acc0));
            uint idx0 = ((n * out_channels + oc0) * h + oh) * w + ow;
            y[idx0] = acc0 * sig0 + x[idx0];
        }
        if (oc1 < out_channels) {
            float sig1 = 1.0 / (1.0 + exp(-acc1));
            uint idx1 = ((n * out_channels + oc1) * h + oh) * w + ow;
            y[idx1] = acc1 * sig1 + x[idx1];
        }
        if (oc2 < out_channels) {
            float sig2 = 1.0 / (1.0 + exp(-acc2));
            uint idx2 = ((n * out_channels + oc2) * h + oh) * w + ow;
            y[idx2] = acc2 * sig2 + x[idx2];
        }
        if (oc3 < out_channels) {
            float sig3 = 1.0 / (1.0 + exp(-acc3));
            uint idx3 = ((n * out_channels + oc3) * h + oh) * w + ow;
            y[idx3] = acc3 * sig3 + x[idx3];
        }
    }
}
)";
    return compile_compute_pso(
        ctx,
        ctx->c2f_bottleneck_root_signature.Get(),
        shader_source,
        "aexrt_c2f_bottleneck_tiled_float32",
        ctx->c2f_bottleneck_tiled_pso.ReleaseAndGetAddressOf(),
        error);
}

static HRESULT ensure_sppf_tail_pipeline(DeviceContext* ctx, bool silu, std::string* error) {
    ComPtr<ID3D12PipelineState>& target = silu ? ctx->sppf_tail_silu_pso : ctx->sppf_tail_linear_pso;
    if (ctx->sppf_tail_root_signature && target) {
        return S_OK;
    }
    if (!ctx->sppf_tail_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 3, 1, 8, ctx->sppf_tail_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }
    const char* shader_source_silu = R"(
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> weight : register(t1);
StructuredBuffer<float> bias : register(t2);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint in_channels; uint h; uint w;
    uint out_channels; uint total; uint _p0; uint _p1;
};
float input_at(uint n, uint c, int yy, int xx) {
    if (yy < 0 || yy >= int(h) || xx < 0 || xx >= int(w)) return -3.402823466e+38;
    return x[((n * in_channels + c) * h + uint(yy)) * w + uint(xx)];
}
float max_radius(uint n, uint c, uint oy, uint ox, int radius) {
    float m = -3.402823466e+38;
    for (int dy = -6; dy <= 6; ++dy) {
        if (dy < -radius || dy > radius) continue;
        int yy = int(oy) + dy;
        for (int dx = -6; dx <= 6; ++dx) {
            if (dx < -radius || dx > radius) continue;
            m = max(m, input_at(n, c, yy, int(ox) + dx));
        }
    }
    return m;
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= total) return;
    uint ox = idx % w;
    uint t = idx / w;
    uint oy = t % h;
    t /= h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    uint plane = h * w;
    float acc = bias[oc];
    uint wbase = oc * in_channels * 4u;
    for (uint ic = 0; ic < in_channels; ++ic) {
        float v0 = x[((n * in_channels + ic) * h + oy) * w + ox];
        float v1 = max_radius(n, ic, oy, ox, 2);
        float v2 = max_radius(n, ic, oy, ox, 4);
        float v3 = max_radius(n, ic, oy, ox, 6);
        acc += v0 * weight[wbase + ic];
        acc += v1 * weight[wbase + in_channels + ic];
        acc += v2 * weight[wbase + in_channels * 2u + ic];
        acc += v3 * weight[wbase + in_channels * 3u + ic];
    }
    float sig = 1.0 / (1.0 + exp(-acc));
    y[idx] = acc * sig;
}
)";
    const char* shader_source_linear = R"(
StructuredBuffer<float> x : register(t0);
StructuredBuffer<float> weight : register(t1);
StructuredBuffer<float> bias : register(t2);
RWStructuredBuffer<float> y : register(u0);
cbuffer C : register(b0) {
    uint batch; uint in_channels; uint h; uint w;
    uint out_channels; uint total; uint _p0; uint _p1;
};
float input_at(uint n, uint c, int yy, int xx) {
    if (yy < 0 || yy >= int(h) || xx < 0 || xx >= int(w)) return -3.402823466e+38;
    return x[((n * in_channels + c) * h + uint(yy)) * w + uint(xx)];
}
float max_radius(uint n, uint c, uint oy, uint ox, int radius) {
    float m = -3.402823466e+38;
    for (int dy = -6; dy <= 6; ++dy) {
        if (dy < -radius || dy > radius) continue;
        int yy = int(oy) + dy;
        for (int dx = -6; dx <= 6; ++dx) {
            if (dx < -radius || dx > radius) continue;
            m = max(m, input_at(n, c, yy, int(ox) + dx));
        }
    }
    return m;
}
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= total) return;
    uint ox = idx % w;
    uint t = idx / w;
    uint oy = t % h;
    t /= h;
    uint oc = t % out_channels;
    uint n = t / out_channels;
    float acc = bias[oc];
    uint wbase = oc * in_channels * 4u;
    for (uint ic = 0; ic < in_channels; ++ic) {
        float v0 = x[((n * in_channels + ic) * h + oy) * w + ox];
        float v1 = max_radius(n, ic, oy, ox, 2);
        float v2 = max_radius(n, ic, oy, ox, 4);
        float v3 = max_radius(n, ic, oy, ox, 6);
        acc += v0 * weight[wbase + ic];
        acc += v1 * weight[wbase + in_channels + ic];
        acc += v2 * weight[wbase + in_channels * 2u + ic];
        acc += v3 * weight[wbase + in_channels * 3u + ic];
    }
    y[idx] = acc;
}
)";
    const char* source = silu ? shader_source_silu : shader_source_linear;
    const char* name = silu ? "aexrt_sppf_tail_conv1x1_silu_float32" : "aexrt_sppf_tail_conv1x1_linear_float32";
    return compile_compute_pso(ctx, ctx->sppf_tail_root_signature.Get(), source, name, target.ReleaseAndGetAddressOf(), error);
}

static void create_float_srv(DeviceContext* ctx, ID3D12Resource* resource, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = 0;
    srv.Buffer.NumElements = elements;
    srv.Buffer.StructureByteStride = sizeof(float);
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    ctx->device->CreateShaderResourceView(resource, &srv, handle);
}

static void create_float_srv(DeviceContext* ctx, BufferHandle* buffer, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = buffer->element_offset;
    srv.Buffer.NumElements = elements;
    srv.Buffer.StructureByteStride = sizeof(float);
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    ctx->device->CreateShaderResourceView(buffer->resource.Get(), &srv, handle);
}

static void create_uint_srv(DeviceContext* ctx, BufferHandle* buffer, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = buffer->element_offset;
    srv.Buffer.NumElements = elements;
    srv.Buffer.StructureByteStride = sizeof(uint32_t);
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    ctx->device->CreateShaderResourceView(buffer->resource.Get(), &srv, handle);
}

static void create_float_uav(DeviceContext* ctx, ID3D12Resource* resource, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = elements;
    uav.Buffer.StructureByteStride = sizeof(float);
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    ctx->device->CreateUnorderedAccessView(resource, nullptr, &uav, handle);
}

static void create_float_uav(DeviceContext* ctx, BufferHandle* buffer, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = buffer->element_offset;
    uav.Buffer.NumElements = elements;
    uav.Buffer.StructureByteStride = sizeof(float);
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    ctx->device->CreateUnorderedAccessView(buffer->resource.Get(), nullptr, &uav, handle);
}

static void create_uint_uav(DeviceContext* ctx, BufferHandle* buffer, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = buffer->element_offset;
    uav.Buffer.NumElements = elements;
    uav.Buffer.StructureByteStride = sizeof(uint32_t);
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    ctx->device->CreateUnorderedAccessView(buffer->resource.Get(), nullptr, &uav, handle);
}

static HRESULT create_heap(DeviceContext* ctx, UINT descriptors, ID3D12DescriptorHeap** heap_out) {
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = descriptors;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    HRESULT hr = ctx->device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) return hr;
    *heap_out = heap.Detach();
    return S_OK;
}

static bool all_owned_by(DeviceContext* ctx, const std::vector<BufferHandle*>& buffers) {
    for (auto* b : buffers) {
        if (!b || b->owner != ctx) return false;
    }
    return true;
}

static bool parse_uint_sequence(PyObject* seq_obj, std::vector<UINT>& out, size_t expected) {
    PyObject* seq = PySequence_Fast(seq_obj, "expected a sequence of unsigned integers");
    if (!seq) return false;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (expected != 0 && static_cast<size_t>(n) != expected) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "expected %zu unsigned integers, got %zd", expected, n);
        return false;
    }
    out.resize(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        unsigned long v = PyLong_AsUnsignedLong(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        out[static_cast<size_t>(i)] = static_cast<UINT>(v);
    }
    Py_DECREF(seq);
    return true;
}

static HRESULT ensure_yolo_head_decode_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->yolo_head_root_signature && ctx->yolo_head_decode_pso) {
        return S_OK;
    }
    if (!ctx->yolo_head_root_signature) {
        HRESULT hr = make_two_table_root(ctx, 6, 2, 14, ctx->yolo_head_root_signature.ReleaseAndGetAddressOf(), error);
        if (FAILED(hr)) return hr;
    }

    const char* shader_source = R"(
StructuredBuffer<float> box0 : register(t0);
StructuredBuffer<float> cls0 : register(t1);
StructuredBuffer<float> box1 : register(t2);
StructuredBuffer<float> cls1 : register(t3);
StructuredBuffer<float> box2 : register(t4);
StructuredBuffer<float> cls2 : register(t5);
RWStructuredBuffer<float> candidates : register(u0);
RWStructuredBuffer<uint> candidate_counter : register(u1);
cbuffer AexrtYoloHeadConstants : register(b0) {
    uint h0;
    uint w0;
    uint h1;
    uint w1;
    uint h2;
    uint w2;
    uint classes;
    uint max_candidates;
    float conf_threshold;
    float stride0;
    float stride1;
    float stride2;
    uint total_anchors;
    uint _pad0;
};
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}
float read_box(uint scale, uint ch, uint pos) {
    uint p0 = h0 * w0;
    uint p1 = h1 * w1;
    if (scale == 0u) return box0[ch * p0 + pos];
    if (scale == 1u) return box1[ch * p1 + pos];
    return box2[ch * (h2 * w2) + pos];
}
float read_cls(uint scale, uint ch, uint pos) {
    uint p0 = h0 * w0;
    uint p1 = h1 * w1;
    if (scale == 0u) return cls0[ch * p0 + pos];
    if (scale == 1u) return cls1[ch * p1 + pos];
    return cls2[ch * (h2 * w2) + pos];
}
float dfl(uint scale, uint dim, uint pos) {
    uint base = dim * 16u;
    float m = -3.402823e38;
    [unroll]
    for (uint b = 0u; b < 16u; ++b) {
        m = max(m, read_box(scale, base + b, pos));
    }
    float sum = 0.0;
    float acc = 0.0;
    [unroll]
    for (uint b = 0u; b < 16u; ++b) {
        float e = exp(read_box(scale, base + b, pos) - m);
        sum += e;
        acc += e * (float)b;
    }
    return acc / max(sum, 1e-20);
}
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint a = tid.x;
    if (a >= total_anchors || classes == 0u) {
        return;
    }
    uint p0 = h0 * w0;
    uint p1 = h1 * w1;
    uint scale = 0u;
    uint pos = a;
    uint h = h0;
    uint w = w0;
    float stride = stride0;
    if (a >= p0 + p1) {
        scale = 2u;
        pos = a - p0 - p1;
        h = h2;
        w = w2;
        stride = stride2;
    } else if (a >= p0) {
        scale = 1u;
        pos = a - p0;
        h = h1;
        w = w1;
        stride = stride1;
    }
    if (pos >= h * w) {
        return;
    }
    float best = sigmoid(read_cls(scale, 0u, pos));
    uint best_cls = 0u;
    for (uint c = 1u; c < classes; ++c) {
        float s = sigmoid(read_cls(scale, c, pos));
        if (s > best) {
            best = s;
            best_cls = c;
        }
    }
    if (best < conf_threshold) {
        return;
    }
    uint out_idx = 0u;
    InterlockedAdd(candidate_counter[0], 1u, out_idx);
    if (out_idx >= max_candidates) {
        return;
    }
    float ax = (float)(pos % w) + 0.5;
    float ay = (float)(pos / w) + 0.5;
    float l = dfl(scale, 0u, pos);
    float t = dfl(scale, 1u, pos);
    float r = dfl(scale, 2u, pos);
    float b = dfl(scale, 3u, pos);
    uint dst = out_idx * 6u;
    candidates[dst + 0u] = (ax - l) * stride;
    candidates[dst + 1u] = (ay - t) * stride;
    candidates[dst + 2u] = (ax + r) * stride;
    candidates[dst + 3u] = (ay + b) * stride;
    candidates[dst + 4u] = best;
    candidates[dst + 5u] = (float)best_cls;
}
)";
    return compile_compute_pso(
        ctx,
        ctx->yolo_head_root_signature.Get(),
        shader_source,
        "aexrt_yolo_head_decode_float32",
        ctx->yolo_head_decode_pso.ReleaseAndGetAddressOf(),
        error);
}

static HRESULT ensure_yolo_decode_filter_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->yolo_root_signature && ctx->yolo_decode_filter_pso) {
        return S_OK;
    }

    const char* shader_source = R"(
StructuredBuffer<float> yolo_output : register(t0);
RWStructuredBuffer<float> detections : register(u0);
RWStructuredBuffer<uint> counter : register(u1);
cbuffer AexrtYoloConstants : register(b0) {
    uint anchors;
    uint channels;
    uint classes;
    uint max_detections;
    float conf_threshold;
};
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint a = tid.x;
    if (a >= anchors || classes == 0) {
        return;
    }
    float best = yolo_output[(4u + 0u) * anchors + a];
    uint best_cls = 0;
    for (uint c = 1; c < classes; ++c) {
        float s = yolo_output[(4u + c) * anchors + a];
        if (s > best) {
            best = s;
            best_cls = c;
        }
    }
    if (best < conf_threshold) {
        return;
    }
    uint out_idx = 0;
    InterlockedAdd(counter[0], 1, out_idx);
    if (out_idx >= max_detections) {
        return;
    }
    float cx = yolo_output[0u * anchors + a];
    float cy = yolo_output[1u * anchors + a];
    float w = yolo_output[2u * anchors + a];
    float h = yolo_output[3u * anchors + a];
    uint base = out_idx * 6u;
    detections[base + 0u] = cx - 0.5 * w;
    detections[base + 1u] = cy - 0.5 * h;
    detections[base + 2u] = cx + 0.5 * w;
    detections[base + 3u] = cy + 0.5 * h;
    detections[base + 4u] = best;
    detections[base + 5u] = (float)best_cls;
}
)";

    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(shader_source, strlen(shader_source), "aexrt_yolo_decode_filter_float32", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader, &errors);
    if (FAILED(hr)) {
        if (errors && error) {
            *error = static_cast<const char*>(errors->GetBufferPointer());
        }
        return hr;
    }

    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 1;
    srv_range.BaseShaderRegister = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 2;
    uav_range.BaseShaderRegister = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv_range;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav_range;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = 5;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_errors;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_errors);
    if (FAILED(hr)) {
        if (root_errors && error) {
            *error = static_cast<const char*>(root_errors->GetBufferPointer());
        }
        return hr;
    }
    hr = ctx->device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&ctx->yolo_root_signature));
    if (FAILED(hr)) {
        return hr;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = ctx->yolo_root_signature.Get();
    pso_desc.CS.pShaderBytecode = shader->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader->GetBufferSize();
    return ctx->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&ctx->yolo_decode_filter_pso));
}

static HRESULT ensure_yolo_nms_mark_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->yolo_nms_root_signature && ctx->yolo_nms_mark_pso) {
        return S_OK;
    }
    const char* shader_source = R"(
StructuredBuffer<float> candidates : register(t0);
StructuredBuffer<uint> candidate_counter : register(t1);
RWStructuredBuffer<uint> keep_flags : register(u0);
cbuffer AexrtNmsConstants : register(b0) {
    uint max_candidates;
    float iou_threshold;
};
float box_iou(uint ia, uint ib) {
    uint a = ia * 6u;
    uint b = ib * 6u;
    float x1 = max(candidates[a + 0u], candidates[b + 0u]);
    float y1 = max(candidates[a + 1u], candidates[b + 1u]);
    float x2 = min(candidates[a + 2u], candidates[b + 2u]);
    float y2 = min(candidates[a + 3u], candidates[b + 3u]);
    float iw = max(0.0, x2 - x1);
    float ih = max(0.0, y2 - y1);
    float inter = iw * ih;
    float area_a = max(0.0, candidates[a + 2u] - candidates[a + 0u]) * max(0.0, candidates[a + 3u] - candidates[a + 1u]);
    float area_b = max(0.0, candidates[b + 2u] - candidates[b + 0u]) * max(0.0, candidates[b + 3u] - candidates[b + 1u]);
    return inter / max(area_a + area_b - inter, 1e-7);
}
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    uint count = min(candidate_counter[0], max_candidates);
    if (i >= max_candidates) {
        return;
    }
    if (i >= count) {
        keep_flags[i] = 0u;
        return;
    }
    uint base_i = i * 6u;
    float score_i = candidates[base_i + 4u];
    float cls_i = candidates[base_i + 5u];
    uint keep = 1u;
    for (uint j = 0; j < count; ++j) {
        if (j == i) {
            continue;
        }
        uint base_j = j * 6u;
        if (candidates[base_j + 5u] != cls_i) {
            continue;
        }
        float score_j = candidates[base_j + 4u];
        bool higher = (score_j > score_i) || (score_j == score_i && j < i);
        if (higher && box_iou(i, j) > iou_threshold) {
            keep = 0u;
            break;
        }
    }
    keep_flags[i] = keep;
}
)";
    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(shader_source, strlen(shader_source), "aexrt_yolo_nms_mark_float32", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader, &errors);
    if (FAILED(hr)) {
        if (errors && error) *error = static_cast<const char*>(errors->GetBufferPointer());
        return hr;
    }
    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 2;
    srv_range.BaseShaderRegister = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 1;
    uav_range.BaseShaderRegister = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv_range;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav_range;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = 2;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_errors;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_errors);
    if (FAILED(hr)) {
        if (root_errors && error) *error = static_cast<const char*>(root_errors->GetBufferPointer());
        return hr;
    }
    hr = ctx->device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&ctx->yolo_nms_root_signature));
    if (FAILED(hr)) return hr;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = ctx->yolo_nms_root_signature.Get();
    pso_desc.CS.pShaderBytecode = shader->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader->GetBufferSize();
    return ctx->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&ctx->yolo_nms_mark_pso));
}

static HRESULT ensure_yolo_topk_pipeline(DeviceContext* ctx, std::string* error) {
    if (ctx->yolo_topk_root_signature && ctx->yolo_topk_pso) {
        return S_OK;
    }
    const char* shader_source = R"(
StructuredBuffer<float> candidates : register(t0);
StructuredBuffer<uint> candidate_counter : register(t1);
StructuredBuffer<uint> keep_flags : register(t2);
RWStructuredBuffer<float> final_detections : register(u0);
RWStructuredBuffer<uint> final_counter : register(u1);
cbuffer AexrtTopKConstants : register(b0) {
    uint max_candidates;
    uint max_detections;
};
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    uint count = min(candidate_counter[0], max_candidates);
    if (i >= count || keep_flags[i] == 0u) {
        return;
    }
    uint base_i = i * 6u;
    float score_i = candidates[base_i + 4u];
    uint rank = 0u;
    for (uint j = 0; j < count; ++j) {
        if (keep_flags[j] == 0u || j == i) {
            continue;
        }
        float score_j = candidates[j * 6u + 4u];
        if (score_j > score_i || (score_j == score_i && j < i)) {
            rank += 1u;
        }
    }
    uint capped = min(rank + 1u, max_detections);
    InterlockedMax(final_counter[0], capped);
    if (rank >= max_detections) {
        return;
    }
    uint dst = rank * 6u;
    final_detections[dst + 0u] = candidates[base_i + 0u];
    final_detections[dst + 1u] = candidates[base_i + 1u];
    final_detections[dst + 2u] = candidates[base_i + 2u];
    final_detections[dst + 3u] = candidates[base_i + 3u];
    final_detections[dst + 4u] = candidates[base_i + 4u];
    final_detections[dst + 5u] = candidates[base_i + 5u];
}
)";
    ComPtr<ID3DBlob> shader;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(shader_source, strlen(shader_source), "aexrt_yolo_topk_float32", nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader, &errors);
    if (FAILED(hr)) {
        if (errors && error) *error = static_cast<const char*>(errors->GetBufferPointer());
        return hr;
    }
    D3D12_DESCRIPTOR_RANGE srv_range{};
    srv_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv_range.NumDescriptors = 3;
    srv_range.BaseShaderRegister = 0;
    srv_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_DESCRIPTOR_RANGE uav_range{};
    uav_range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav_range.NumDescriptors = 2;
    uav_range.BaseShaderRegister = 0;
    uav_range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv_range;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav_range;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = 2;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_errors;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_errors);
    if (FAILED(hr)) {
        if (root_errors && error) *error = static_cast<const char*>(root_errors->GetBufferPointer());
        return hr;
    }
    hr = ctx->device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&ctx->yolo_topk_root_signature));
    if (FAILED(hr)) return hr;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = ctx->yolo_topk_root_signature.Get();
    pso_desc.CS.pShaderBytecode = shader->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader->GetBufferSize();
    return ctx->device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&ctx->yolo_topk_pso));
}

static HRESULT create_conv_silu_descriptor_heap(
    DeviceContext* ctx,
    BufferHandle* input,
    BufferHandle* weight,
    BufferHandle* bias,
    BufferHandle* output,
    const Conv2DDesc& desc,
    ID3D12DescriptorHeap** out_heap) {
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 4;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    HRESULT hr = ctx->device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        return hr;
    }

    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = input->element_offset;
    srv.Buffer.StructureByteStride = sizeof(float);
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    srv.Buffer.NumElements = desc.batch * desc.in_channels * desc.in_h * desc.in_w;
    ctx->device->CreateShaderResourceView(input->resource.Get(), &srv, cursor);
    cursor.ptr += descriptor_size;

    srv.Buffer.FirstElement = weight->element_offset;
    srv.Buffer.NumElements = static_cast<UINT>(buffer_float_elements(weight));
    ctx->device->CreateShaderResourceView(weight->resource.Get(), &srv, cursor);
    cursor.ptr += descriptor_size;

    srv.Buffer.FirstElement = bias->element_offset;
    srv.Buffer.NumElements = desc.out_channels;
    ctx->device->CreateShaderResourceView(bias->resource.Get(), &srv, cursor);
    cursor.ptr += descriptor_size;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = output->element_offset;
    uav.Buffer.NumElements = static_cast<UINT>(conv_output_elements(desc));
    uav.Buffer.StructureByteStride = sizeof(float);
    uav.Buffer.CounterOffsetInBytes = 0;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    ctx->device->CreateUnorderedAccessView(output->resource.Get(), nullptr, &uav, cursor);

    *out_heap = heap.Detach();
    return S_OK;
}

static void write_conv_silu_descriptors(
    DeviceContext* ctx,
    ID3D12DescriptorHeap* heap,
    UINT base_descriptor,
    BufferHandle* input,
    BufferHandle* weight,
    BufferHandle* bias,
    BufferHandle* output,
    const Conv2DDesc& desc) {
    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    cursor.ptr += static_cast<SIZE_T>(descriptor_size) * base_descriptor;

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = input->element_offset;
    srv.Buffer.StructureByteStride = sizeof(float);
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    srv.Buffer.NumElements = desc.batch * desc.in_channels * desc.in_h * desc.in_w;
    ctx->device->CreateShaderResourceView(input->resource.Get(), &srv, cursor);
    cursor.ptr += descriptor_size;

    srv.Buffer.FirstElement = weight->element_offset;
    srv.Buffer.NumElements = static_cast<UINT>(buffer_float_elements(weight));
    ctx->device->CreateShaderResourceView(weight->resource.Get(), &srv, cursor);
    cursor.ptr += descriptor_size;

    srv.Buffer.FirstElement = bias->element_offset;
    srv.Buffer.NumElements = desc.out_channels;
    ctx->device->CreateShaderResourceView(bias->resource.Get(), &srv, cursor);
    cursor.ptr += descriptor_size;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = output->element_offset;
    uav.Buffer.NumElements = static_cast<UINT>(conv_output_elements(desc));
    uav.Buffer.StructureByteStride = sizeof(float);
    uav.Buffer.CounterOffsetInBytes = 0;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    ctx->device->CreateUnorderedAccessView(output->resource.Get(), nullptr, &uav, cursor);
}

static void conv_constants(const Conv2DDesc& desc, UINT constants[16]) {
    constants[0] = desc.batch;
    constants[1] = desc.in_channels;
    constants[2] = desc.in_h;
    constants[3] = desc.in_w;
    constants[4] = desc.out_channels;
    constants[5] = desc.out_h;
    constants[6] = desc.out_w;
    constants[7] = desc.kernel_h;
    constants[8] = desc.kernel_w;
    constants[9] = desc.stride_h;
    constants[10] = desc.stride_w;
    constants[11] = desc.pad_top;
    constants[12] = desc.pad_left;
    constants[13] = desc.dilation_h;
    constants[14] = desc.dilation_w;
    constants[15] = desc.groups;
}

static HRESULT record_conv_silu_commands(
    DeviceContext* ctx,
    ID3D12GraphicsCommandList* list,
    ID3D12DescriptorHeap* heap,
    BufferHandle* input,
    BufferHandle* weight,
    BufferHandle* bias,
    BufferHandle* output,
    const Conv2DDesc& desc) {
    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
    uav_gpu.ptr += descriptor_size * 3;

    transition_if_needed(list, input->resource.Get(), input->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition_if_needed(list, weight->resource.Get(), weight->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition_if_needed(list, bias->resource.Get(), bias->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition_if_needed(list, output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* heaps[] = {heap};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(ctx->conv_root_signature.Get());
    ID3D12PipelineState* pso = ctx->conv_silu_pso.Get();
    if (is_conv1x1_fast_path(desc) && ctx->conv1x1_silu_pso) {
        pso = ctx->conv1x1_silu_pso.Get();
    } else if (use_conv3x3_winograd_packed_oc4(weight, desc) && ctx->conv3x3_winograd_packed_oc4_silu_pso) {
        pso = ctx->conv3x3_winograd_packed_oc4_silu_pso.Get();
    } else if (use_conv3x3_winograd_packed(weight, desc) && ctx->conv3x3_winograd_packed_silu_pso) {
        pso = ctx->conv3x3_winograd_packed_silu_pso.Get();
    } else if (is_conv3x3_tiled_fast_path(desc) && native_winograd_enabled() && ctx->conv3x3_winograd_silu_pso) {
        pso = ctx->conv3x3_winograd_silu_pso.Get();
    } else if (is_conv3x3_tiled_fast_path(desc) && ctx->conv3x3_silu_pso) {
        pso = ctx->conv3x3_silu_pso.Get();
    }
    list->SetPipelineState(pso);
    list->SetComputeRootDescriptorTable(0, srv_gpu);
    list->SetComputeRootDescriptorTable(1, uav_gpu);
    UINT constants[16]{};
    conv_constants(desc, constants);
    list->SetComputeRoot32BitConstants(2, 16, constants, 0);
    if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_winograd_packed_oc4_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * ((desc.out_channels + 3) / 4));
    } else if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_winograd_packed_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * desc.out_channels);
    } else if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_winograd_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * desc.out_channels);
    } else if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * desc.out_channels);
    } else {
        list->Dispatch(static_cast<UINT>((conv_output_elements(desc) + 255) / 256), 1, 1);
    }

    transition_if_needed(list, output->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output->state);
    transition_if_needed(list, bias->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, bias->state);
    transition_if_needed(list, weight->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, weight->state);
    transition_if_needed(list, input->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, input->state);
    return S_OK;
}

static HRESULT record_conv_silu_commands_at(
    DeviceContext* ctx,
    ID3D12GraphicsCommandList* list,
    ID3D12DescriptorHeap* heap,
    UINT base_descriptor,
    BufferHandle* input,
    BufferHandle* weight,
    BufferHandle* bias,
    BufferHandle* output,
    const Conv2DDesc& desc) {
    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
    srv_gpu.ptr += static_cast<UINT64>(descriptor_size) * base_descriptor;
    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
    uav_gpu.ptr += descriptor_size * 3;

    transition_if_needed(list, input->resource.Get(), input->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition_if_needed(list, weight->resource.Get(), weight->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition_if_needed(list, bias->resource.Get(), bias->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition_if_needed(list, output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    list->SetComputeRootSignature(ctx->conv_root_signature.Get());
    ID3D12PipelineState* pso = ctx->conv_silu_pso.Get();
    if (is_conv1x1_fast_path(desc) && ctx->conv1x1_silu_pso) {
        pso = ctx->conv1x1_silu_pso.Get();
    } else if (use_conv3x3_winograd_packed_oc4(weight, desc) && ctx->conv3x3_winograd_packed_oc4_silu_pso) {
        pso = ctx->conv3x3_winograd_packed_oc4_silu_pso.Get();
    } else if (use_conv3x3_winograd_packed(weight, desc) && ctx->conv3x3_winograd_packed_silu_pso) {
        pso = ctx->conv3x3_winograd_packed_silu_pso.Get();
    } else if (is_conv3x3_tiled_fast_path(desc) && native_winograd_enabled() && ctx->conv3x3_winograd_silu_pso) {
        pso = ctx->conv3x3_winograd_silu_pso.Get();
    } else if (is_conv3x3_tiled_fast_path(desc) && ctx->conv3x3_silu_pso) {
        pso = ctx->conv3x3_silu_pso.Get();
    }
    list->SetPipelineState(pso);
    list->SetComputeRootDescriptorTable(0, srv_gpu);
    list->SetComputeRootDescriptorTable(1, uav_gpu);
    UINT constants[16]{};
    conv_constants(desc, constants);
    list->SetComputeRoot32BitConstants(2, 16, constants, 0);
    if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_winograd_packed_oc4_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * ((desc.out_channels + 3) / 4));
    } else if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_winograd_packed_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * desc.out_channels);
    } else if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_winograd_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * desc.out_channels);
    } else if (is_conv3x3_tiled_fast_path(desc) && pso == ctx->conv3x3_silu_pso.Get()) {
        list->Dispatch(static_cast<UINT>((desc.out_w + 15) / 16), static_cast<UINT>((desc.out_h + 15) / 16), desc.batch * desc.out_channels);
    } else {
        list->Dispatch(static_cast<UINT>((conv_output_elements(desc) + 255) / 256), 1, 1);
    }

    transition_if_needed(list, output->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output->state);
    transition_if_needed(list, bias->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, bias->state);
    transition_if_needed(list, weight->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, weight->state);
    transition_if_needed(list, input->resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, input->state);
    return S_OK;
}

static HRESULT dispatch_conv_silu_into(
    DeviceContext* ctx,
    BufferHandle* input,
    BufferHandle* weight,
    BufferHandle* bias,
    BufferHandle* output,
    const Conv2DDesc& desc) {
    ComPtr<ID3D12DescriptorHeap> heap;
    HRESULT hr = create_conv_silu_descriptor_heap(ctx, input, weight, bias, output, desc, &heap);
    if (FAILED(hr)) {
        return hr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    bool owns_submission = false;
    hr = begin_or_join_commands(ctx, &owns_submission);
    if (FAILED(hr)) {
        return hr;
    }
    hr = record_conv_silu_commands(ctx, ctx->list.Get(), heap.Get(), input, weight, bias, output, desc);
    if (FAILED(hr)) {
        return hr;
    }
    keep_descriptor_heap_alive(ctx, heap);
    return finish_if_owned(ctx, owns_submission);
}

static HRESULT record_prepared_conv_silu_dispatch(DeviceContext* ctx, ConvSiluDispatchHandle* dispatch) {
    HRESULT hr = ctx->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&dispatch->command_allocator));
    if (FAILED(hr)) {
        return hr;
    }
    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        dispatch->command_allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&dispatch->command_list));
    if (FAILED(hr)) {
        return hr;
    }
    hr = record_conv_silu_commands(
        ctx,
        dispatch->command_list.Get(),
        dispatch->descriptor_heap.Get(),
        dispatch->input,
        dispatch->weight,
        dispatch->bias,
        dispatch->output,
        dispatch->desc);
    if (FAILED(hr)) {
        return hr;
    }
    return dispatch->command_list->Close();
}

static HRESULT record_upload_conv_silu_commands(
    DeviceContext* ctx,
    ConvSiluUploadDispatchHandle* dispatch,
    ConvSiluUploadRingSlot* slot) {
    HRESULT hr = ctx->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&slot->command_allocator));
    if (FAILED(hr)) {
        return hr;
    }
    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        slot->command_allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&slot->command_list));
    if (FAILED(hr)) {
        return hr;
    }

    transition_if_needed(
        slot->command_list.Get(),
        dispatch->input->resource.Get(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST);
    slot->command_list->CopyBufferRegion(
        dispatch->input->resource.Get(),
        0,
        slot->upload.Get(),
        0,
        dispatch->input_nbytes);
    transition_if_needed(
        slot->command_list.Get(),
        dispatch->input->resource.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON);

    hr = record_conv_silu_commands(
        ctx,
        slot->command_list.Get(),
        slot->descriptor_heap.Get(),
        dispatch->input,
        dispatch->weight,
        dispatch->bias,
        dispatch->output,
        dispatch->desc);
    if (FAILED(hr)) {
        return hr;
    }
    return slot->command_list->Close();
}

static HRESULT record_upload_conv_silu_chain_commands(
    DeviceContext* ctx,
    ConvSiluChainUploadDispatchHandle* dispatch,
    ConvSiluChainSlot* slot) {
    HRESULT hr = ctx->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&slot->command_allocator));
    if (FAILED(hr)) {
        return hr;
    }
    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        slot->command_allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&slot->command_list));
    if (FAILED(hr)) {
        return hr;
    }

    transition_if_needed(
        slot->command_list.Get(),
        dispatch->input->resource.Get(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST);
    slot->command_list->CopyBufferRegion(
        dispatch->input->resource.Get(),
        0,
        slot->upload.Get(),
        0,
        dispatch->input_nbytes);
    transition_if_needed(
        slot->command_list.Get(),
        dispatch->input->resource.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON);

    BufferHandle* current_input = dispatch->input;
    ID3D12DescriptorHeap* heaps[] = {slot->descriptor_heap.Get()};
    slot->command_list->SetDescriptorHeaps(1, heaps);
    for (size_t i = 0; i < dispatch->descs.size(); ++i) {
        hr = record_conv_silu_commands_at(
            ctx,
            slot->command_list.Get(),
            slot->descriptor_heap.Get(),
            static_cast<UINT>(i * 4),
            current_input,
            dispatch->weights[i],
            dispatch->biases[i],
            dispatch->block_outputs[i],
            dispatch->descs[i]);
        if (FAILED(hr)) {
            return hr;
        }
        current_input = dispatch->block_outputs[i];
    }

    return slot->command_list->Close();
}

static HRESULT execute_prepared_conv_silu_dispatch(DeviceContext* ctx, ConvSiluDispatchHandle* dispatch) {
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ID3D12CommandList* lists[] = {dispatch->command_list.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);
    return signal_and_wait(ctx);
}

static PyObject* py_probe_device(PyObject*, PyObject* args) {
    unsigned int index = 0;
    if (!PyArg_ParseTuple(args, "|I", &index)) {
        return nullptr;
    }

    ComPtr<IDXGIAdapter1> adapter;
    if (!get_adapter_for_device(index, &adapter)) {
        PyErr_SetString(PyExc_RuntimeError, "no hardware DXGI adapter found");
        return nullptr;
    }

    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr);
    if (FAILED(hr)) {
        return raise_hr("D3D12CreateDevice(probe)", hr);
    }

    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    auto name = wide_to_utf8(desc.Description);
    return Py_BuildValue(
        "{s:s,s:k,s:k,s:k,s:I,s:K}",
        "name", name.c_str(),
        "vendor_id", static_cast<unsigned long>(desc.VendorId),
        "device_id", static_cast<unsigned long>(desc.DeviceId),
        "subsys_id", static_cast<unsigned long>(desc.SubSysId),
        "revision", static_cast<unsigned int>(desc.Revision),
        "dedicated_video_memory", static_cast<unsigned long long>(desc.DedicatedVideoMemory));
}

static PyObject* py_create_device(PyObject*, PyObject* args) {
    unsigned int index = 0;
    if (!PyArg_ParseTuple(args, "|I", &index)) {
        return nullptr;
    }

    ComPtr<IDXGIAdapter1> adapter;
    if (!get_adapter_for_device(index, &adapter)) {
        PyErr_SetString(PyExc_RuntimeError, "no hardware DXGI adapter found");
        return nullptr;
    }

    auto* ctx = new DeviceContext();
    ctx->adapter = adapter;
    ctx->adapter_index = index;

    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx->device));
    if (FAILED(hr)) {
        delete ctx;
        return raise_hr("D3D12CreateDevice", hr);
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queue_desc.NodeMask = 0;
    hr = ctx->device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&ctx->queue));
    if (FAILED(hr)) {
        delete ctx;
        return raise_hr("CreateCommandQueue", hr);
    }

    hr = ctx->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ctx->allocator));
    if (FAILED(hr)) {
        delete ctx;
        return raise_hr("CreateCommandAllocator", hr);
    }

    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        ctx->allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&ctx->list));
    if (FAILED(hr)) {
        delete ctx;
        return raise_hr("CreateCommandList", hr);
    }
    ctx->list->Close();

    hr = ctx->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->fence));
    if (FAILED(hr)) {
        delete ctx;
        return raise_hr("CreateFence", hr);
    }
    ctx->fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!ctx->fence_event) {
        delete ctx;
        PyErr_SetString(PyExc_RuntimeError, "CreateEvent failed");
        return nullptr;
    }

    return PyCapsule_New(ctx, kDeviceCapsule, destroy_device);
}

static PyObject* py_device_info(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &device_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }

    DXGI_ADAPTER_DESC1 desc{};
    ctx->adapter->GetDesc1(&desc);
    auto name = wide_to_utf8(desc.Description);
    return Py_BuildValue(
        "{s:s,s:k,s:k,s:k,s:I,s:K}",
        "name", name.c_str(),
        "vendor_id", static_cast<unsigned long>(desc.VendorId),
        "device_id", static_cast<unsigned long>(desc.DeviceId),
        "subsys_id", static_cast<unsigned long>(desc.SubSysId),
        "revision", static_cast<unsigned int>(desc.Revision),
        "dedicated_video_memory", static_cast<unsigned long long>(desc.DedicatedVideoMemory));
}

static PyObject* py_allocate_buffer(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    unsigned long long nbytes = 0;
    const char* label = "";
    if (!PyArg_ParseTuple(args, "OK|s", &device_capsule, &nbytes, &label)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }
    if (nbytes == 0) {
        PyErr_SetString(PyExc_ValueError, "nbytes must be positive");
        return nullptr;
    }

    auto* buffer = new BufferHandle();
    buffer->owner = ctx;
    buffer->nbytes = nbytes;
    buffer->label = label ? label : "";
    buffer->state = D3D12_RESOURCE_STATE_COMMON;

    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        buffer->state,
        nbytes,
        &buffer->resource);
    if (FAILED(hr)) {
        delete buffer;
        return raise_hr("CreateCommittedResource(default buffer)", hr);
    }
    auto wide_label = utf8_to_wide(buffer->label.c_str());
    if (!wide_label.empty()) {
        buffer->resource->SetName(wide_label.c_str());
    }
    return PyCapsule_New(buffer, kBufferCapsule, destroy_buffer);
}

static PyObject* py_allocate_uav_buffer(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    unsigned long long nbytes = 0;
    const char* label = "";
    if (!PyArg_ParseTuple(args, "OK|s", &device_capsule, &nbytes, &label)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }
    if (nbytes == 0) {
        PyErr_SetString(PyExc_ValueError, "nbytes must be positive");
        return nullptr;
    }

    auto* buffer = new BufferHandle();
    buffer->owner = ctx;
    buffer->nbytes = nbytes;
    buffer->label = label ? label : "";
    buffer->state = D3D12_RESOURCE_STATE_COMMON;

    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        buffer->state,
        nbytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        &buffer->resource);
    if (FAILED(hr)) {
        delete buffer;
        return raise_hr("CreateCommittedResource(UAV buffer)", hr);
    }
    auto wide_label = utf8_to_wide(buffer->label.c_str());
    if (!wide_label.empty()) {
        buffer->resource->SetName(wide_label.c_str());
    }
    return PyCapsule_New(buffer, kBufferCapsule, destroy_buffer);
}

static PyObject* py_upload_buffer(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    Py_buffer source{};
    const char* label = "";
    if (!PyArg_ParseTuple(args, "Oy*|s", &device_capsule, &source, &label)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        PyBuffer_Release(&source);
        return nullptr;
    }
    if (source.len <= 0) {
        PyBuffer_Release(&source);
        PyErr_SetString(PyExc_ValueError, "upload source must be non-empty");
        return nullptr;
    }

    auto* buffer = new BufferHandle();
    buffer->owner = ctx;
    buffer->nbytes = static_cast<uint64_t>(source.len);
    buffer->label = label ? label : "";
    buffer->state = D3D12_RESOURCE_STATE_COMMON;

    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        buffer->state,
        buffer->nbytes,
        &buffer->resource);
    if (FAILED(hr)) {
        PyBuffer_Release(&source);
        delete buffer;
        return raise_hr("CreateCommittedResource(upload target)", hr);
    }

    ComPtr<ID3D12Resource> upload;
    hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        buffer->nbytes,
        &upload);
    if (FAILED(hr)) {
        PyBuffer_Release(&source);
        delete buffer;
        return raise_hr("CreateCommittedResource(upload heap)", hr);
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, 0};
    hr = upload->Map(0, &read_range, &mapped);
    if (FAILED(hr)) {
        PyBuffer_Release(&source);
        delete buffer;
        return raise_hr("Map(upload heap)", hr);
    }
    memcpy(mapped, source.buf, static_cast<size_t>(source.len));
    upload->Unmap(0, nullptr);
    PyBuffer_Release(&source);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, D3D12_RESOURCE_STATE_COPY_DEST);
            buffer->state = D3D12_RESOURCE_STATE_COPY_DEST;
            ctx->list->CopyBufferRegion(buffer->resource.Get(), 0, upload.Get(), 0, buffer->nbytes);
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, D3D12_RESOURCE_STATE_COMMON);
            buffer->state = D3D12_RESOURCE_STATE_COMMON;
            hr = finish_commands(ctx);
        }
    }
    if (FAILED(hr)) {
        delete buffer;
        return raise_hr("upload command submission", hr);
    }

    auto wide_label = utf8_to_wide(buffer->label.c_str());
    if (!wide_label.empty()) {
        buffer->resource->SetName(wide_label.c_str());
    }
    return PyCapsule_New(buffer, kBufferCapsule, destroy_buffer);
}

static PyObject* py_upload_buffer_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* buffer_capsule = nullptr;
    Py_buffer source{};
    if (!PyArg_ParseTuple(args, "OOy*", &device_capsule, &buffer_capsule, &source)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* buffer = get_buffer(buffer_capsule);
    if (!ctx || !buffer) {
        PyBuffer_Release(&source);
        return nullptr;
    }
    if (buffer->owner != ctx) {
        PyBuffer_Release(&source);
        PyErr_SetString(PyExc_ValueError, "buffer belongs to a different device");
        return nullptr;
    }
    if (source.len <= 0 || static_cast<uint64_t>(source.len) > buffer->nbytes) {
        PyBuffer_Release(&source);
        PyErr_SetString(PyExc_ValueError, "upload source is empty or exceeds target buffer size");
        return nullptr;
    }
    const uint64_t upload_nbytes = static_cast<uint64_t>(source.len);

    ComPtr<ID3D12Resource> upload;
    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        upload_nbytes,
        &upload);
    if (FAILED(hr)) {
        PyBuffer_Release(&source);
        return raise_hr("CreateCommittedResource(upload heap)", hr);
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, 0};
    hr = upload->Map(0, &read_range, &mapped);
    if (FAILED(hr)) {
        PyBuffer_Release(&source);
        return raise_hr("Map(upload heap)", hr);
    }
    memcpy(mapped, source.buf, static_cast<size_t>(source.len));
    upload->Unmap(0, nullptr);
    PyBuffer_Release(&source);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, D3D12_RESOURCE_STATE_COPY_DEST);
            auto before_copy = buffer->state;
            buffer->state = D3D12_RESOURCE_STATE_COPY_DEST;
            ctx->list->CopyBufferRegion(buffer->resource.Get(), buffer->element_offset * sizeof(float), upload.Get(), 0, upload_nbytes);
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, before_copy);
            buffer->state = before_copy;
            hr = finish_commands(ctx);
        }
    }
    if (FAILED(hr)) {
        return raise_hr("upload_into command submission", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_create_buffer_view(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* buffer_capsule = nullptr;
    unsigned long long element_offset = 0;
    unsigned long long element_count = 0;
    const char* label = "";
    if (!PyArg_ParseTuple(args, "OOKK|s", &device_capsule, &buffer_capsule, &element_offset, &element_count, &label)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* parent = get_buffer(buffer_capsule);
    if (!ctx || !parent) {
        return nullptr;
    }
    if (parent->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "buffer belongs to a different device");
        return nullptr;
    }
    if (element_count == 0) {
        PyErr_SetString(PyExc_ValueError, "view element_count must be positive");
        return nullptr;
    }
    const uint64_t byte_offset = element_offset * sizeof(float);
    const uint64_t view_nbytes = element_count * sizeof(float);
    if (byte_offset > parent->nbytes || view_nbytes > parent->nbytes - byte_offset) {
        PyErr_SetString(PyExc_ValueError, "buffer view exceeds parent buffer");
        return nullptr;
    }
    auto* view = new BufferHandle();
    view->owner = ctx;
    view->resource = parent->resource;
    view->nbytes = view_nbytes;
    view->element_offset = parent->element_offset + element_offset;
    view->state = parent->state;
    view->label = label ? label : "";
    Py_INCREF(buffer_capsule);
    view->parent_capsule = buffer_capsule;
    return PyCapsule_New(view, kBufferCapsule, destroy_buffer);
}

static PyObject* py_download_buffer(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* buffer_capsule = nullptr;
    unsigned long long nbytes = 0;
    if (!PyArg_ParseTuple(args, "OOK", &device_capsule, &buffer_capsule, &nbytes)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* buffer = get_buffer(buffer_capsule);
    if (!ctx || !buffer) {
        return nullptr;
    }
    if (buffer->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "buffer belongs to a different device");
        return nullptr;
    }
    if (nbytes > buffer->nbytes) {
        PyErr_SetString(PyExc_ValueError, "download size exceeds buffer size");
        return nullptr;
    }

    ComPtr<ID3D12Resource> readback;
    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nbytes,
        &readback);
    if (FAILED(hr)) {
        return raise_hr("CreateCommittedResource(readback heap)", hr);
    }

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            auto before_copy = buffer->state;
            buffer->state = D3D12_RESOURCE_STATE_COPY_SOURCE;
            ctx->list->CopyBufferRegion(readback.Get(), 0, buffer->resource.Get(), buffer->element_offset * sizeof(float), nbytes);
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, before_copy);
            buffer->state = before_copy;
            hr = finish_commands(ctx);
        }
    }
    if (FAILED(hr)) {
        return raise_hr("download command submission", hr);
    }

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(nbytes)};
    hr = readback->Map(0, &read_range, &mapped);
    if (FAILED(hr)) {
        return raise_hr("Map(readback heap)", hr);
    }
    PyObject* out = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(mapped), static_cast<Py_ssize_t>(nbytes));
    D3D12_RANGE write_range{0, 0};
    readback->Unmap(0, &write_range);
    return out;
}

static PyObject* py_buffer_info(PyObject*, PyObject* args) {
    PyObject* buffer_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &buffer_capsule)) {
        return nullptr;
    }
    BufferHandle* buffer = get_buffer(buffer_capsule);
    if (!buffer) {
        return nullptr;
    }
    return Py_BuildValue(
        "{s:K,s:K,s:s,s:I}",
        "nbytes", static_cast<unsigned long long>(buffer->nbytes),
        "element_offset", static_cast<unsigned long long>(buffer->element_offset),
        "label", buffer->label.c_str(),
        "state", static_cast<unsigned int>(buffer->state));
}

static PyObject* py_dispatch_relu_float32(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    unsigned long long element_count = 0;
    if (!PyArg_ParseTuple(args, "OOK", &device_capsule, &input_capsule, &element_count)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    if (!ctx || !input) {
        return nullptr;
    }
    if (input->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "input buffer belongs to a different device");
        return nullptr;
    }
    if (element_count == 0) {
        PyErr_SetString(PyExc_ValueError, "element_count must be positive");
        return nullptr;
    }
    const uint64_t nbytes = element_count * sizeof(float);
    if (nbytes > input->nbytes) {
        PyErr_SetString(PyExc_ValueError, "element_count exceeds input buffer size");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_relu_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "ReLU pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_relu_pipeline", hr);
    }

    auto* output = new BufferHandle();
    output->owner = ctx;
    output->nbytes = nbytes;
    output->label = "aexrt_relu_float32_output";
    output->state = D3D12_RESOURCE_STATE_COMMON;
    hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        output->state,
        nbytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        &output->resource);
    if (FAILED(hr)) {
        delete output;
        return raise_hr("CreateCommittedResource(relu output)", hr);
    }

    hr = dispatch_relu_into(ctx, input, output, element_count);
    if (FAILED(hr)) {
        delete output;
        return raise_hr("relu dispatch command submission", hr);
    }

    return PyCapsule_New(output, kBufferCapsule, destroy_buffer);
}

static PyObject* py_dispatch_relu_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    unsigned long long element_count = 0;
    if (!PyArg_ParseTuple(args, "OOOK", &device_capsule, &input_capsule, &output_capsule, &element_count)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    if (!ctx || !input || !output) {
        return nullptr;
    }
    if (input->owner != ctx || output->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "input/output buffer belongs to a different device");
        return nullptr;
    }
    if (element_count == 0) {
        PyErr_SetString(PyExc_ValueError, "element_count must be positive");
        return nullptr;
    }
    const uint64_t nbytes = element_count * sizeof(float);
    if (nbytes > input->nbytes || nbytes > output->nbytes) {
        PyErr_SetString(PyExc_ValueError, "element_count exceeds input/output buffer size");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_relu_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "ReLU pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_relu_pipeline", hr);
    }

    hr = dispatch_relu_into(ctx, input, output, element_count);
    if (FAILED(hr)) {
        return raise_hr("relu dispatch_into command submission", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_prepare_relu_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    unsigned long long element_count = 0;
    if (!PyArg_ParseTuple(args, "OOOK", &device_capsule, &input_capsule, &output_capsule, &element_count)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    if (!ctx || !input || !output) {
        return nullptr;
    }
    if (input->owner != ctx || output->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "input/output buffer belongs to a different device");
        return nullptr;
    }
    if (element_count == 0) {
        PyErr_SetString(PyExc_ValueError, "element_count must be positive");
        return nullptr;
    }
    const uint64_t nbytes = element_count * sizeof(float);
    if (nbytes > input->nbytes || nbytes > output->nbytes) {
        PyErr_SetString(PyExc_ValueError, "element_count exceeds input/output buffer size");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_relu_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "ReLU pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_relu_pipeline", hr);
    }

    auto* dispatch = new ReluDispatchHandle();
    dispatch->owner = ctx;
    dispatch->input = input;
    dispatch->output = output;
    dispatch->element_count = element_count;
    Py_INCREF(device_capsule);
    Py_INCREF(input_capsule);
    Py_INCREF(output_capsule);
    dispatch->device_capsule = device_capsule;
    dispatch->input_capsule = input_capsule;
    dispatch->output_capsule = output_capsule;

    hr = create_relu_descriptor_heap(
        ctx,
        input,
        output,
        element_count,
        dispatch->descriptor_heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
        Py_DECREF(device_capsule);
        Py_DECREF(input_capsule);
        Py_DECREF(output_capsule);
        delete dispatch;
        return raise_hr("create_relu_descriptor_heap", hr);
    }

    hr = record_prepared_relu_dispatch(ctx, dispatch);
    if (FAILED(hr)) {
        Py_DECREF(device_capsule);
        Py_DECREF(input_capsule);
        Py_DECREF(output_capsule);
        delete dispatch;
        return raise_hr("record_prepared_relu_dispatch", hr);
    }

    return PyCapsule_New(dispatch, kReluDispatchCapsule, destroy_relu_dispatch);
}

static PyObject* py_execute_relu_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* dispatch_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &device_capsule, &dispatch_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    ReluDispatchHandle* dispatch = get_relu_dispatch(dispatch_capsule);
    if (!ctx || !dispatch) {
        return nullptr;
    }
    if (dispatch->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "prepared dispatch belongs to a different device");
        return nullptr;
    }

    HRESULT hr = execute_prepared_relu_dispatch(ctx, dispatch);
    if (FAILED(hr)) {
        return raise_hr("execute_relu_float32_dispatch", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_conv2d_silu_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    Conv2DDesc desc{};
    if (!PyArg_ParseTuple(
            args,
            "OOOOO(IIIIIIIIIIIIIIII)",
            &device_capsule,
            &input_capsule,
            &weight_capsule,
            &bias_capsule,
            &output_capsule,
            &desc.batch,
            &desc.in_channels,
            &desc.in_h,
            &desc.in_w,
            &desc.out_channels,
            &desc.out_h,
            &desc.out_w,
            &desc.kernel_h,
            &desc.kernel_w,
            &desc.stride_h,
            &desc.stride_w,
            &desc.pad_top,
            &desc.pad_left,
            &desc.dilation_h,
            &desc.dilation_w,
            &desc.groups)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    const char* reason = nullptr;
    if (!validate_conv_silu_args(ctx, input, weight, bias, output, desc, &reason)) {
        PyErr_SetString(PyExc_ValueError, reason ? reason : "invalid Conv2D+SiLU arguments");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_conv_silu_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr) && is_conv1x1_fast_path(desc)) {
        hr = ensure_conv1x1_pipeline(ctx, true, &pipeline_error);
    }
    if (SUCCEEDED(hr) && is_conv3x3_tiled_fast_path(desc)) {
        hr = ensure_preferred_conv3x3_silu_pipeline(ctx, &pipeline_error);
    }
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "Conv2D+SiLU pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_conv_silu_pipeline", hr);
    }

    hr = dispatch_conv_silu_into(ctx, input, weight, bias, output, desc);
    if (FAILED(hr)) {
        return raise_hr("conv2d_silu dispatch_into command submission", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_prepare_conv2d_silu_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    Conv2DDesc desc{};
    if (!PyArg_ParseTuple(
            args,
            "OOOOO(IIIIIIIIIIIIIIII)",
            &device_capsule,
            &input_capsule,
            &weight_capsule,
            &bias_capsule,
            &output_capsule,
            &desc.batch,
            &desc.in_channels,
            &desc.in_h,
            &desc.in_w,
            &desc.out_channels,
            &desc.out_h,
            &desc.out_w,
            &desc.kernel_h,
            &desc.kernel_w,
            &desc.stride_h,
            &desc.stride_w,
            &desc.pad_top,
            &desc.pad_left,
            &desc.dilation_h,
            &desc.dilation_w,
            &desc.groups)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    const char* reason = nullptr;
    if (!validate_conv_silu_args(ctx, input, weight, bias, output, desc, &reason)) {
        PyErr_SetString(PyExc_ValueError, reason ? reason : "invalid Conv2D+SiLU arguments");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_conv_silu_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr) && is_conv1x1_fast_path(desc)) {
        hr = ensure_conv1x1_pipeline(ctx, true, &pipeline_error);
    }
    if (SUCCEEDED(hr) && is_conv3x3_tiled_fast_path(desc)) {
        hr = ensure_preferred_conv3x3_silu_pipeline(ctx, &pipeline_error);
    }
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "Conv2D+SiLU pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_conv_silu_pipeline", hr);
    }

    auto* dispatch = new ConvSiluDispatchHandle();
    dispatch->owner = ctx;
    dispatch->input = input;
    dispatch->weight = weight;
    dispatch->bias = bias;
    dispatch->output = output;
    dispatch->desc = desc;
    dispatch->element_count = conv_output_elements(desc);
    Py_INCREF(device_capsule);
    Py_INCREF(input_capsule);
    Py_INCREF(weight_capsule);
    Py_INCREF(bias_capsule);
    Py_INCREF(output_capsule);
    dispatch->device_capsule = device_capsule;
    dispatch->input_capsule = input_capsule;
    dispatch->weight_capsule = weight_capsule;
    dispatch->bias_capsule = bias_capsule;
    dispatch->output_capsule = output_capsule;

    hr = create_conv_silu_descriptor_heap(
        ctx,
        input,
        weight,
        bias,
        output,
        desc,
        dispatch->descriptor_heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) {
        Py_DECREF(device_capsule);
        Py_DECREF(input_capsule);
        Py_DECREF(weight_capsule);
        Py_DECREF(bias_capsule);
        Py_DECREF(output_capsule);
        delete dispatch;
        return raise_hr("create_conv_silu_descriptor_heap", hr);
    }

    hr = record_prepared_conv_silu_dispatch(ctx, dispatch);
    if (FAILED(hr)) {
        Py_DECREF(device_capsule);
        Py_DECREF(input_capsule);
        Py_DECREF(weight_capsule);
        Py_DECREF(bias_capsule);
        Py_DECREF(output_capsule);
        delete dispatch;
        return raise_hr("record_prepared_conv_silu_dispatch", hr);
    }

    return PyCapsule_New(dispatch, kConvSiluDispatchCapsule, destroy_conv_silu_dispatch);
}

static PyObject* py_execute_conv2d_silu_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* dispatch_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &device_capsule, &dispatch_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    ConvSiluDispatchHandle* dispatch = get_conv_silu_dispatch(dispatch_capsule);
    if (!ctx || !dispatch) {
        return nullptr;
    }
    if (dispatch->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "prepared Conv2D+SiLU dispatch belongs to a different device");
        return nullptr;
    }

    HRESULT hr = execute_prepared_conv_silu_dispatch(ctx, dispatch);
    if (FAILED(hr)) {
        return raise_hr("execute_conv2d_silu_float32_dispatch", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_prepare_conv2d_silu_upload_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    Conv2DDesc desc{};
    unsigned int ring_size = 2;
    if (!PyArg_ParseTuple(
            args,
            "OOOO(IIIIIIIIIIIIIIII)|I",
            &device_capsule,
            &weight_capsule,
            &bias_capsule,
            &output_capsule,
            &desc.batch,
            &desc.in_channels,
            &desc.in_h,
            &desc.in_w,
            &desc.out_channels,
            &desc.out_h,
            &desc.out_w,
            &desc.kernel_h,
            &desc.kernel_w,
            &desc.stride_h,
            &desc.stride_w,
            &desc.pad_top,
            &desc.pad_left,
            &desc.dilation_h,
            &desc.dilation_w,
            &desc.groups,
            &ring_size)) {
        return nullptr;
    }
    if (ring_size == 0 || ring_size > 8) {
        PyErr_SetString(PyExc_ValueError, "ring_size must be in [1, 8]");
        return nullptr;
    }

    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    if (!ctx || !weight || !bias || !output) {
        return nullptr;
    }

    const uint64_t input_nbytes = uint64_t(desc.batch) * desc.in_channels * desc.in_h * desc.in_w * sizeof(float);
    auto* input = new BufferHandle();
    input->owner = ctx;
    input->nbytes = input_nbytes;
    input->label = "aexrt_conv_silu_upload_input";
    input->state = D3D12_RESOURCE_STATE_COMMON;
    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        input->state,
        input->nbytes,
        &input->resource);
    if (FAILED(hr)) {
        delete input;
        return raise_hr("CreateCommittedResource(upload fused input)", hr);
    }

    const char* reason = nullptr;
    if (!validate_conv_silu_args(ctx, input, weight, bias, output, desc, &reason)) {
        delete input;
        PyErr_SetString(PyExc_ValueError, reason ? reason : "invalid Conv2D+SiLU upload dispatch arguments");
        return nullptr;
    }

    std::string pipeline_error;
    hr = ensure_conv_silu_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr) && is_conv1x1_fast_path(desc)) {
        hr = ensure_conv1x1_pipeline(ctx, true, &pipeline_error);
    }
    if (SUCCEEDED(hr) && is_conv3x3_tiled_fast_path(desc)) {
        hr = ensure_preferred_conv3x3_silu_pipeline(ctx, &pipeline_error);
    }
    if (FAILED(hr)) {
        delete input;
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "Conv2D+SiLU pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_conv_silu_pipeline", hr);
    }

    auto* dispatch = new ConvSiluUploadDispatchHandle();
    dispatch->owner = ctx;
    dispatch->input = input;
    dispatch->weight = weight;
    dispatch->bias = bias;
    dispatch->output = output;
    dispatch->desc = desc;
    dispatch->input_nbytes = input_nbytes;
    dispatch->element_count = conv_output_elements(desc);
    dispatch->slots.resize(ring_size);
    Py_INCREF(device_capsule);
    Py_INCREF(weight_capsule);
    Py_INCREF(bias_capsule);
    Py_INCREF(output_capsule);
    dispatch->device_capsule = device_capsule;
    dispatch->weight_capsule = weight_capsule;
    dispatch->bias_capsule = bias_capsule;
    dispatch->output_capsule = output_capsule;

    auto cleanup = [&]() {
        D3D12_RANGE empty_range{0, 0};
        for (auto& slot : dispatch->slots) {
            if (slot.mapped && slot.upload) {
                slot.upload->Unmap(0, &empty_range);
                slot.mapped = nullptr;
            }
        }
        delete dispatch->input;
        Py_DECREF(device_capsule);
        Py_DECREF(weight_capsule);
        Py_DECREF(bias_capsule);
        Py_DECREF(output_capsule);
        delete dispatch;
    };

    for (uint32_t i = 0; i < ring_size; ++i) {
        auto& slot = dispatch->slots[i];
        hr = create_committed_buffer(
            ctx->device.Get(),
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            input_nbytes,
            &slot.upload);
        if (FAILED(hr)) {
            cleanup();
            return raise_hr("CreateCommittedResource(persistent upload slot)", hr);
        }
        D3D12_RANGE read_range{0, 0};
        hr = slot.upload->Map(0, &read_range, &slot.mapped);
        if (FAILED(hr)) {
            cleanup();
            return raise_hr("Map(persistent upload slot)", hr);
        }
        hr = create_conv_silu_descriptor_heap(
            ctx,
            input,
            weight,
            bias,
            output,
            desc,
            slot.descriptor_heap.ReleaseAndGetAddressOf());
        if (FAILED(hr)) {
            cleanup();
            return raise_hr("create_conv_silu_descriptor_heap(upload slot)", hr);
        }
        hr = record_upload_conv_silu_commands(ctx, dispatch, &slot);
        if (FAILED(hr)) {
            cleanup();
            return raise_hr("record_upload_conv_silu_commands", hr);
        }
    }

    return PyCapsule_New(dispatch, kConvSiluUploadDispatchCapsule, destroy_conv_silu_upload_dispatch);
}

static PyObject* py_execute_conv2d_silu_upload_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* dispatch_capsule = nullptr;
    Py_buffer source{};
    if (!PyArg_ParseTuple(args, "OOy*", &device_capsule, &dispatch_capsule, &source)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    ConvSiluUploadDispatchHandle* dispatch = get_conv_silu_upload_dispatch(dispatch_capsule);
    if (!ctx || !dispatch) {
        PyBuffer_Release(&source);
        return nullptr;
    }
    if (dispatch->owner != ctx) {
        PyBuffer_Release(&source);
        PyErr_SetString(PyExc_ValueError, "prepared Conv2D+SiLU upload dispatch belongs to a different device");
        return nullptr;
    }
    if (source.len <= 0 || static_cast<uint64_t>(source.len) > dispatch->input_nbytes) {
        PyBuffer_Release(&source);
        PyErr_SetString(PyExc_ValueError, "input upload is empty or exceeds prepared input size");
        return nullptr;
    }
    ConvSiluUploadRingSlot& slot = dispatch->slots[dispatch->next_slot % dispatch->slots.size()];
    std::memcpy(slot.mapped, source.buf, static_cast<size_t>(source.len));
    PyBuffer_Release(&source);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ID3D12CommandList* lists[] = {slot.command_list.Get()};
        ctx->queue->ExecuteCommandLists(1, lists);
        HRESULT hr = signal_and_wait(ctx);
        if (FAILED(hr)) {
            return raise_hr("execute_conv2d_silu_upload_float32_dispatch", hr);
        }
    }
    dispatch->next_slot = (dispatch->next_slot + 1) % static_cast<uint32_t>(dispatch->slots.size());
    Py_RETURN_NONE;
}

static bool parse_conv_desc_object(PyObject* obj, Conv2DDesc* desc) {
    PyObject* seq = PySequence_Fast(obj, "Conv2D descriptor must be a sequence");
    if (!seq) {
        return false;
    }
    if (PySequence_Fast_GET_SIZE(seq) != 16) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "Conv2D descriptor must have 16 values");
        return false;
    }
    uint32_t* fields[] = {
        &desc->batch,
        &desc->in_channels,
        &desc->in_h,
        &desc->in_w,
        &desc->out_channels,
        &desc->out_h,
        &desc->out_w,
        &desc->kernel_h,
        &desc->kernel_w,
        &desc->stride_h,
        &desc->stride_w,
        &desc->pad_top,
        &desc->pad_left,
        &desc->dilation_h,
        &desc->dilation_w,
        &desc->groups,
    };
    for (Py_ssize_t i = 0; i < 16; ++i) {
        unsigned long value = PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return false;
        }
        *fields[i] = static_cast<uint32_t>(value);
    }
    Py_DECREF(seq);
    return true;
}

static PyObject* py_prepare_conv2d_silu_chain_upload_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* weights_obj = nullptr;
    PyObject* biases_obj = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* descs_obj = nullptr;
    unsigned int ring_size = 2;
    if (!PyArg_ParseTuple(
            args,
            "OOOOO|I",
            &device_capsule,
            &weights_obj,
            &biases_obj,
            &output_capsule,
            &descs_obj,
            &ring_size)) {
        return nullptr;
    }
    if (ring_size == 0 || ring_size > 8) {
        PyErr_SetString(PyExc_ValueError, "ring_size must be in [1, 8]");
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* final_output = get_buffer(output_capsule);
    if (!ctx || !final_output) {
        return nullptr;
    }

    PyObject* weights_seq = PySequence_Fast(weights_obj, "weights must be a sequence of buffer capsules");
    if (!weights_seq) return nullptr;
    PyObject* biases_seq = PySequence_Fast(biases_obj, "biases must be a sequence of buffer capsules");
    if (!biases_seq) { Py_DECREF(weights_seq); return nullptr; }
    PyObject* descs_seq = PySequence_Fast(descs_obj, "descs must be a sequence of Conv2D descriptors");
    if (!descs_seq) { Py_DECREF(weights_seq); Py_DECREF(biases_seq); return nullptr; }

    const Py_ssize_t block_count = PySequence_Fast_GET_SIZE(descs_seq);
    if (block_count <= 0 ||
        PySequence_Fast_GET_SIZE(weights_seq) != block_count ||
        PySequence_Fast_GET_SIZE(biases_seq) != block_count) {
        Py_DECREF(weights_seq);
        Py_DECREF(biases_seq);
        Py_DECREF(descs_seq);
        PyErr_SetString(PyExc_ValueError, "weights, biases, and descs must have the same positive length");
        return nullptr;
    }

    std::vector<Conv2DDesc> descs(static_cast<size_t>(block_count));
    std::vector<BufferHandle*> weights(static_cast<size_t>(block_count), nullptr);
    std::vector<BufferHandle*> biases(static_cast<size_t>(block_count), nullptr);
    for (Py_ssize_t i = 0; i < block_count; ++i) {
        if (!parse_conv_desc_object(PySequence_Fast_GET_ITEM(descs_seq, i), &descs[static_cast<size_t>(i)])) {
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            return nullptr;
        }
        weights[static_cast<size_t>(i)] = get_buffer(PySequence_Fast_GET_ITEM(weights_seq, i));
        biases[static_cast<size_t>(i)] = get_buffer(PySequence_Fast_GET_ITEM(biases_seq, i));
        if (!weights[static_cast<size_t>(i)] || !biases[static_cast<size_t>(i)]) {
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            return nullptr;
        }
    }

    std::string pipeline_error;
    HRESULT hr = ensure_conv_silu_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr)) {
        for (const auto& desc : descs) {
            if (is_conv1x1_fast_path(desc)) {
                hr = ensure_conv1x1_pipeline(ctx, true, &pipeline_error);
                if (FAILED(hr)) {
                    break;
                }
            }
            if (is_conv3x3_tiled_fast_path(desc)) {
                hr = ensure_preferred_conv3x3_silu_pipeline(ctx, &pipeline_error);
                if (FAILED(hr)) {
                    break;
                }
            }
        }
    }
    if (FAILED(hr)) {
        Py_DECREF(weights_seq);
        Py_DECREF(biases_seq);
        Py_DECREF(descs_seq);
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "Conv2D+SiLU pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_conv_silu_pipeline", hr);
    }

    auto* dispatch = new ConvSiluChainUploadDispatchHandle();
    dispatch->owner = ctx;
    dispatch->output = final_output;
    dispatch->weights = weights;
    dispatch->biases = biases;
    dispatch->descs = descs;
    dispatch->input_nbytes = uint64_t(descs[0].batch) * descs[0].in_channels * descs[0].in_h * descs[0].in_w * sizeof(float);
    Py_INCREF(device_capsule);
    Py_INCREF(output_capsule);
    dispatch->device_capsule = device_capsule;
    dispatch->output_capsule = output_capsule;
    dispatch->weight_capsules.reserve(static_cast<size_t>(block_count));
    dispatch->bias_capsules.reserve(static_cast<size_t>(block_count));
    for (Py_ssize_t i = 0; i < block_count; ++i) {
        PyObject* w_obj = PySequence_Fast_GET_ITEM(weights_seq, i);
        PyObject* b_obj = PySequence_Fast_GET_ITEM(biases_seq, i);
        Py_INCREF(w_obj);
        Py_INCREF(b_obj);
        dispatch->weight_capsules.push_back(w_obj);
        dispatch->bias_capsules.push_back(b_obj);
    }

    auto cleanup = [&]() {
        D3D12_RANGE empty_range{0, 0};
        for (auto& slot : dispatch->slots) {
            if (slot.mapped && slot.upload) {
                slot.upload->Unmap(0, &empty_range);
                slot.mapped = nullptr;
            }
        }
        for (auto* buffer : dispatch->owned_buffers) {
            delete buffer;
        }
        Py_DECREF(device_capsule);
        Py_DECREF(output_capsule);
        for (auto* obj : dispatch->weight_capsules) Py_DECREF(obj);
        for (auto* obj : dispatch->bias_capsules) Py_DECREF(obj);
        delete dispatch;
    };

    auto* input = new BufferHandle();
    input->owner = ctx;
    input->nbytes = dispatch->input_nbytes;
    input->label = "aexrt_conv_silu_chain_input";
    input->state = D3D12_RESOURCE_STATE_COMMON;
    hr = create_committed_buffer(ctx->device.Get(), D3D12_HEAP_TYPE_DEFAULT, input->state, input->nbytes, &input->resource);
    if (FAILED(hr)) {
        delete input;
        cleanup();
        Py_DECREF(weights_seq);
        Py_DECREF(biases_seq);
        Py_DECREF(descs_seq);
        return raise_hr("CreateCommittedResource(chain input)", hr);
    }
    dispatch->input = input;
    dispatch->owned_buffers.push_back(input);

    dispatch->block_outputs.resize(static_cast<size_t>(block_count), nullptr);
    const uint32_t arena_count = block_count > 1 ? (block_count == 2 ? 1u : 2u) : 0u;
    uint64_t arena_nbytes[2] = {0, 0};
    for (Py_ssize_t i = 0; i < block_count - 1; ++i) {
        uint32_t arena_index = static_cast<uint32_t>(i % 2);
        uint64_t needed = conv_output_elements(descs[static_cast<size_t>(i)]) * sizeof(float);
        if (needed > arena_nbytes[arena_index]) {
            arena_nbytes[arena_index] = needed;
        }
    }
    std::vector<BufferHandle*> arenas(arena_count, nullptr);
    for (uint32_t i = 0; i < arena_count; ++i) {
        auto* arena = new BufferHandle();
        arena->owner = ctx;
        arena->nbytes = arena_nbytes[i];
        arena->label = "aexrt_conv_silu_chain_arena";
        arena->state = D3D12_RESOURCE_STATE_COMMON;
        hr = create_committed_buffer(
            ctx->device.Get(),
            D3D12_HEAP_TYPE_DEFAULT,
            arena->state,
            arena->nbytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            &arena->resource);
        if (FAILED(hr)) {
            delete arena;
            cleanup();
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            return raise_hr("CreateCommittedResource(chain arena)", hr);
        }
        arenas[i] = arena;
        dispatch->owned_buffers.push_back(arena);
    }

    BufferHandle* current_input = input;
    for (Py_ssize_t i = 0; i < block_count; ++i) {
        const auto& desc = descs[static_cast<size_t>(i)];
        BufferHandle* output = i == block_count - 1 ? final_output : arenas[static_cast<size_t>(i % 2)];
        const char* reason = nullptr;
        if (!validate_conv_silu_args(ctx, current_input, weights[static_cast<size_t>(i)], biases[static_cast<size_t>(i)], output, desc, &reason)) {
            cleanup();
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            PyErr_SetString(PyExc_ValueError, reason ? reason : "invalid Conv2D+SiLU chain block");
            return nullptr;
        }
        dispatch->block_outputs[static_cast<size_t>(i)] = output;
        current_input = output;
    }

    dispatch->slots.resize(ring_size);
    for (uint32_t slot_index = 0; slot_index < ring_size; ++slot_index) {
        auto& slot = dispatch->slots[slot_index];
        hr = create_committed_buffer(ctx->device.Get(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, dispatch->input_nbytes, &slot.upload);
        if (FAILED(hr)) {
            cleanup();
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            return raise_hr("CreateCommittedResource(chain upload slot)", hr);
        }
        D3D12_RANGE read_range{0, 0};
        hr = slot.upload->Map(0, &read_range, &slot.mapped);
        if (FAILED(hr)) {
            cleanup();
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            return raise_hr("Map(chain upload slot)", hr);
        }
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = static_cast<UINT>(block_count * 4);
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = ctx->device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&slot.descriptor_heap));
        if (FAILED(hr)) {
            cleanup();
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            return raise_hr("CreateDescriptorHeap(chain packed descriptors)", hr);
        }
        current_input = input;
        for (Py_ssize_t i = 0; i < block_count; ++i) {
            write_conv_silu_descriptors(
                ctx,
                slot.descriptor_heap.Get(),
                static_cast<UINT>(i * 4),
                current_input,
                weights[static_cast<size_t>(i)],
                biases[static_cast<size_t>(i)],
                dispatch->block_outputs[static_cast<size_t>(i)],
                descs[static_cast<size_t>(i)]);
            current_input = dispatch->block_outputs[static_cast<size_t>(i)];
        }
        hr = record_upload_conv_silu_chain_commands(ctx, dispatch, &slot);
        if (FAILED(hr)) {
            cleanup();
            Py_DECREF(weights_seq);
            Py_DECREF(biases_seq);
            Py_DECREF(descs_seq);
            return raise_hr("record_upload_conv_silu_chain_commands", hr);
        }
    }

    Py_DECREF(weights_seq);
    Py_DECREF(biases_seq);
    Py_DECREF(descs_seq);
    return PyCapsule_New(dispatch, kConvSiluChainUploadDispatchCapsule, destroy_conv_silu_chain_upload_dispatch);
}

static PyObject* py_execute_conv2d_silu_chain_upload_float32_dispatch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* dispatch_capsule = nullptr;
    Py_buffer source{};
    if (!PyArg_ParseTuple(args, "OOy*", &device_capsule, &dispatch_capsule, &source)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    ConvSiluChainUploadDispatchHandle* dispatch = get_conv_silu_chain_upload_dispatch(dispatch_capsule);
    if (!ctx || !dispatch) {
        PyBuffer_Release(&source);
        return nullptr;
    }
    if (dispatch->owner != ctx) {
        PyBuffer_Release(&source);
        PyErr_SetString(PyExc_ValueError, "prepared Conv2D+SiLU chain upload dispatch belongs to a different device");
        return nullptr;
    }
    if (source.len <= 0 || static_cast<uint64_t>(source.len) > dispatch->input_nbytes) {
        PyBuffer_Release(&source);
        PyErr_SetString(PyExc_ValueError, "input upload is empty or exceeds prepared chain input size");
        return nullptr;
    }
    ConvSiluChainSlot& slot = dispatch->slots[dispatch->next_slot % dispatch->slots.size()];
    std::memcpy(slot.mapped, source.buf, static_cast<size_t>(source.len));
    PyBuffer_Release(&source);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ID3D12CommandList* lists[] = {slot.command_list.Get()};
        ctx->queue->ExecuteCommandLists(1, lists);
        HRESULT hr = signal_and_wait(ctx);
        if (FAILED(hr)) {
            return raise_hr("execute_conv2d_silu_chain_upload_float32_dispatch", hr);
        }
    }
    dispatch->next_slot = (dispatch->next_slot + 1) % static_cast<uint32_t>(dispatch->slots.size());
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_yolo_decode_filter_float32(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* yolo_capsule = nullptr;
    PyObject* detections_capsule = nullptr;
    PyObject* counter_capsule = nullptr;
    unsigned int anchors = 0;
    unsigned int channels = 0;
    unsigned int classes = 0;
    unsigned int max_detections = 0;
    float conf_threshold = 0.25f;
    if (!PyArg_ParseTuple(
            args,
            "OOOOIIIIf",
            &device_capsule,
            &yolo_capsule,
            &detections_capsule,
            &counter_capsule,
            &anchors,
            &channels,
            &classes,
            &max_detections,
            &conf_threshold)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* yolo = get_buffer(yolo_capsule);
    BufferHandle* detections = get_buffer(detections_capsule);
    BufferHandle* counter = get_buffer(counter_capsule);
    if (!ctx || !yolo || !detections || !counter) {
        return nullptr;
    }
    if (yolo->owner != ctx || detections->owner != ctx || counter->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "YOLO buffers belong to a different device");
        return nullptr;
    }
    if (anchors == 0 || channels < 5 || classes == 0 || channels < classes + 4 || max_detections == 0) {
        PyErr_SetString(PyExc_ValueError, "invalid YOLO decode/filter dimensions");
        return nullptr;
    }
    if (yolo->nbytes < uint64_t(anchors) * channels * sizeof(float) ||
        detections->nbytes < uint64_t(max_detections) * 6 * sizeof(float) ||
        counter->nbytes < sizeof(uint32_t)) {
        PyErr_SetString(PyExc_ValueError, "YOLO decode/filter buffers are too small");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_yolo_decode_filter_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "YOLO decode/filter pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_yolo_decode_filter_pipeline", hr);
    }

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 3;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = ctx->device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        return raise_hr("CreateDescriptorHeap(yolo)", hr);
    }
    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = anchors * channels;
    srv.Buffer.StructureByteStride = sizeof(float);
    ctx->device->CreateShaderResourceView(yolo->resource.Get(), &srv, cursor);
    cursor.ptr += descriptor_size;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = max_detections * 6;
    uav.Buffer.StructureByteStride = sizeof(float);
    ctx->device->CreateUnorderedAccessView(detections->resource.Get(), nullptr, &uav, cursor);
    cursor.ptr += descriptor_size;

    uav.Buffer.NumElements = 1;
    uav.Buffer.StructureByteStride = sizeof(uint32_t);
    ctx->device->CreateUnorderedAccessView(counter->resource.Get(), nullptr, &uav, cursor);

    ComPtr<ID3D12Resource> zero_upload;
    hr = create_committed_buffer(ctx->device.Get(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, sizeof(uint32_t), &zero_upload);
    if (FAILED(hr)) {
        return raise_hr("CreateCommittedResource(yolo zero upload)", hr);
    }
    uint32_t* mapped_zero = nullptr;
    D3D12_RANGE read_range{0, 0};
    hr = zero_upload->Map(0, &read_range, reinterpret_cast<void**>(&mapped_zero));
    if (FAILED(hr)) {
        return raise_hr("Map(yolo zero upload)", hr);
    }
    *mapped_zero = 0;
    zero_upload->Unmap(0, nullptr);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            auto yolo_before = yolo->state;
            auto det_before = detections->state;
            auto counter_before = counter->state;

            transition_if_needed(ctx->list.Get(), counter->resource.Get(), counter->state, D3D12_RESOURCE_STATE_COPY_DEST);
            counter->state = D3D12_RESOURCE_STATE_COPY_DEST;
            ctx->list->CopyBufferRegion(counter->resource.Get(), 0, zero_upload.Get(), 0, sizeof(uint32_t));
            transition_if_needed(ctx->list.Get(), yolo->resource.Get(), yolo->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), detections->resource.Get(), detections->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition_if_needed(ctx->list.Get(), counter->resource.Get(), counter->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            yolo->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            detections->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            counter->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size;
            ctx->list->SetDescriptorHeaps(1, heaps);
            ctx->list->SetComputeRootSignature(ctx->yolo_root_signature.Get());
            ctx->list->SetPipelineState(ctx->yolo_decode_filter_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT constants[5] = {anchors, channels, classes, max_detections, 0};
            std::memcpy(&constants[4], &conf_threshold, sizeof(float));
            ctx->list->SetComputeRoot32BitConstants(2, 5, constants, 0);
            ctx->list->Dispatch((anchors + 255) / 256, 1, 1);

            transition_if_needed(ctx->list.Get(), counter->resource.Get(), counter->state, counter_before);
            transition_if_needed(ctx->list.Get(), detections->resource.Get(), detections->state, det_before);
            transition_if_needed(ctx->list.Get(), yolo->resource.Get(), yolo->state, yolo_before);
            counter->state = counter_before;
            detections->state = det_before;
            yolo->state = yolo_before;
            keep_descriptor_heap_alive(ctx, heap);
            keep_resource_alive(ctx, zero_upload);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) {
        return raise_hr("YOLO decode/filter command submission", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_yolo_decode_nms_float32(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* yolo_capsule = nullptr;
    PyObject* candidate_capsule = nullptr;
    PyObject* candidate_counter_capsule = nullptr;
    PyObject* keep_capsule = nullptr;
    PyObject* final_capsule = nullptr;
    PyObject* final_counter_capsule = nullptr;
    unsigned int anchors = 0;
    unsigned int channels = 0;
    unsigned int classes = 0;
    unsigned int max_candidates = 0;
    unsigned int max_detections = 0;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOIIIIIff",
            &device_capsule,
            &yolo_capsule,
            &candidate_capsule,
            &candidate_counter_capsule,
            &keep_capsule,
            &final_capsule,
            &final_counter_capsule,
            &anchors,
            &channels,
            &classes,
            &max_candidates,
            &max_detections,
            &conf_threshold,
            &iou_threshold)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* yolo = get_buffer(yolo_capsule);
    BufferHandle* candidates = get_buffer(candidate_capsule);
    BufferHandle* candidate_counter = get_buffer(candidate_counter_capsule);
    BufferHandle* keep = get_buffer(keep_capsule);
    BufferHandle* final_detections = get_buffer(final_capsule);
    BufferHandle* final_counter = get_buffer(final_counter_capsule);
    if (!ctx || !yolo || !candidates || !candidate_counter || !keep || !final_detections || !final_counter) {
        return nullptr;
    }
    if (yolo->owner != ctx || candidates->owner != ctx || candidate_counter->owner != ctx || keep->owner != ctx ||
        final_detections->owner != ctx || final_counter->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "YOLO NMS buffers belong to a different device");
        return nullptr;
    }
    if (anchors == 0 || channels < 5 || classes == 0 || channels < classes + 4 || max_candidates == 0 || max_detections == 0) {
        PyErr_SetString(PyExc_ValueError, "invalid YOLO NMS dimensions");
        return nullptr;
    }
    if (yolo->nbytes < uint64_t(anchors) * channels * sizeof(float) ||
        candidates->nbytes < uint64_t(max_candidates) * 6 * sizeof(float) ||
        candidate_counter->nbytes < sizeof(uint32_t) ||
        keep->nbytes < uint64_t(max_candidates) * sizeof(uint32_t) ||
        final_detections->nbytes < uint64_t(max_detections) * 6 * sizeof(float) ||
        final_counter->nbytes < sizeof(uint32_t)) {
        PyErr_SetString(PyExc_ValueError, "YOLO NMS buffers are too small");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_yolo_decode_filter_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr)) hr = ensure_yolo_nms_mark_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr)) hr = ensure_yolo_topk_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "YOLO NMS pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_yolo_nms_pipelines", hr);
    }

    ComPtr<ID3D12Resource> zero_upload;
    hr = create_committed_buffer(ctx->device.Get(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, sizeof(uint32_t) * 2, &zero_upload);
    if (FAILED(hr)) return raise_hr("CreateCommittedResource(yolo nms zero upload)", hr);
    uint32_t* mapped_zero = nullptr;
    D3D12_RANGE read_range{0, 0};
    hr = zero_upload->Map(0, &read_range, reinterpret_cast<void**>(&mapped_zero));
    if (FAILED(hr)) return raise_hr("Map(yolo nms zero upload)", hr);
    mapped_zero[0] = 0;
    mapped_zero[1] = 0;
    zero_upload->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC decode_heap_desc{};
    decode_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    decode_heap_desc.NumDescriptors = 3;
    decode_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> decode_heap;
    hr = ctx->device->CreateDescriptorHeap(&decode_heap_desc, IID_PPV_ARGS(&decode_heap));
    if (FAILED(hr)) return raise_hr("CreateDescriptorHeap(yolo decode)", hr);

    D3D12_DESCRIPTOR_HEAP_DESC nms_heap_desc{};
    nms_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    nms_heap_desc.NumDescriptors = 3;
    nms_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> nms_heap;
    hr = ctx->device->CreateDescriptorHeap(&nms_heap_desc, IID_PPV_ARGS(&nms_heap));
    if (FAILED(hr)) return raise_hr("CreateDescriptorHeap(yolo nms)", hr);

    D3D12_DESCRIPTOR_HEAP_DESC topk_heap_desc{};
    topk_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    topk_heap_desc.NumDescriptors = 5;
    topk_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> topk_heap;
    hr = ctx->device->CreateDescriptorHeap(&topk_heap_desc, IID_PPV_ARGS(&topk_heap));
    if (FAILED(hr)) return raise_hr("CreateDescriptorHeap(yolo topk)", hr);

    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto make_float_srv = [&](ID3D12Resource* resource, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_UNKNOWN;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = elements;
        srv.Buffer.StructureByteStride = sizeof(float);
        ctx->device->CreateShaderResourceView(resource, &srv, handle);
    };
    auto make_uint_srv = [&](ID3D12Resource* resource, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_UNKNOWN;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = elements;
        srv.Buffer.StructureByteStride = sizeof(uint32_t);
        ctx->device->CreateShaderResourceView(resource, &srv, handle);
    };
    auto make_float_uav = [&](ID3D12Resource* resource, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = elements;
        uav.Buffer.StructureByteStride = sizeof(float);
        ctx->device->CreateUnorderedAccessView(resource, nullptr, &uav, handle);
    };
    auto make_uint_uav = [&](ID3D12Resource* resource, UINT elements, D3D12_CPU_DESCRIPTOR_HANDLE handle) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = elements;
        uav.Buffer.StructureByteStride = sizeof(uint32_t);
        ctx->device->CreateUnorderedAccessView(resource, nullptr, &uav, handle);
    };

    D3D12_CPU_DESCRIPTOR_HANDLE cursor = decode_heap->GetCPUDescriptorHandleForHeapStart();
    make_float_srv(yolo->resource.Get(), anchors * channels, cursor);
    cursor.ptr += descriptor_size;
    make_float_uav(candidates->resource.Get(), max_candidates * 6, cursor);
    cursor.ptr += descriptor_size;
    make_uint_uav(candidate_counter->resource.Get(), 1, cursor);

    cursor = nms_heap->GetCPUDescriptorHandleForHeapStart();
    make_float_srv(candidates->resource.Get(), max_candidates * 6, cursor);
    cursor.ptr += descriptor_size;
    make_uint_srv(candidate_counter->resource.Get(), 1, cursor);
    cursor.ptr += descriptor_size;
    make_uint_uav(keep->resource.Get(), max_candidates, cursor);

    cursor = topk_heap->GetCPUDescriptorHandleForHeapStart();
    make_float_srv(candidates->resource.Get(), max_candidates * 6, cursor);
    cursor.ptr += descriptor_size;
    make_uint_srv(candidate_counter->resource.Get(), 1, cursor);
    cursor.ptr += descriptor_size;
    make_uint_srv(keep->resource.Get(), max_candidates, cursor);
    cursor.ptr += descriptor_size;
    make_float_uav(final_detections->resource.Get(), max_detections * 6, cursor);
    cursor.ptr += descriptor_size;
    make_uint_uav(final_counter->resource.Get(), 1, cursor);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            auto yolo_before = yolo->state;
            auto candidates_before = candidates->state;
            auto candidate_counter_before = candidate_counter->state;
            auto keep_before = keep->state;
            auto final_before = final_detections->state;
            auto final_counter_before = final_counter->state;

            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, D3D12_RESOURCE_STATE_COPY_DEST);
            transition_if_needed(ctx->list.Get(), final_counter->resource.Get(), final_counter->state, D3D12_RESOURCE_STATE_COPY_DEST);
            candidate_counter->state = D3D12_RESOURCE_STATE_COPY_DEST;
            final_counter->state = D3D12_RESOURCE_STATE_COPY_DEST;
            ctx->list->CopyBufferRegion(candidate_counter->resource.Get(), 0, zero_upload.Get(), 0, sizeof(uint32_t));
            ctx->list->CopyBufferRegion(final_counter->resource.Get(), 0, zero_upload.Get(), sizeof(uint32_t), sizeof(uint32_t));

            transition_if_needed(ctx->list.Get(), yolo->resource.Get(), yolo->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), candidates->resource.Get(), candidates->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            yolo->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            candidates->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            candidate_counter->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

            ID3D12DescriptorHeap* heaps[] = {decode_heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = decode_heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size;
            ctx->list->SetComputeRootSignature(ctx->yolo_root_signature.Get());
            ctx->list->SetPipelineState(ctx->yolo_decode_filter_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT decode_constants[5] = {anchors, channels, classes, max_candidates, 0};
            std::memcpy(&decode_constants[4], &conf_threshold, sizeof(float));
            ctx->list->SetComputeRoot32BitConstants(2, 5, decode_constants, 0);
            ctx->list->Dispatch((anchors + 255) / 256, 1, 1);
            uav_barrier(ctx->list.Get(), candidates->resource.Get());
            uav_barrier(ctx->list.Get(), candidate_counter->resource.Get());

            transition_if_needed(ctx->list.Get(), candidates->resource.Get(), candidates->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), keep->resource.Get(), keep->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            candidates->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            candidate_counter->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            keep->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            heaps[0] = nms_heap.Get();
            ctx->list->SetDescriptorHeaps(1, heaps);
            srv_gpu = nms_heap->GetGPUDescriptorHandleForHeapStart();
            uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 2;
            ctx->list->SetComputeRootSignature(ctx->yolo_nms_root_signature.Get());
            ctx->list->SetPipelineState(ctx->yolo_nms_mark_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT nms_constants[2] = {max_candidates, 0};
            std::memcpy(&nms_constants[1], &iou_threshold, sizeof(float));
            ctx->list->SetComputeRoot32BitConstants(2, 2, nms_constants, 0);
            ctx->list->Dispatch((max_candidates + 127) / 128, 1, 1);
            uav_barrier(ctx->list.Get(), keep->resource.Get());

            transition_if_needed(ctx->list.Get(), keep->resource.Get(), keep->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), final_detections->resource.Get(), final_detections->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition_if_needed(ctx->list.Get(), final_counter->resource.Get(), final_counter->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            keep->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            final_detections->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            final_counter->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            heaps[0] = topk_heap.Get();
            ctx->list->SetDescriptorHeaps(1, heaps);
            srv_gpu = topk_heap->GetGPUDescriptorHandleForHeapStart();
            uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 3;
            ctx->list->SetComputeRootSignature(ctx->yolo_topk_root_signature.Get());
            ctx->list->SetPipelineState(ctx->yolo_topk_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT topk_constants[2] = {max_candidates, max_detections};
            ctx->list->SetComputeRoot32BitConstants(2, 2, topk_constants, 0);
            ctx->list->Dispatch((max_candidates + 127) / 128, 1, 1);

            transition_if_needed(ctx->list.Get(), final_counter->resource.Get(), final_counter->state, final_counter_before);
            transition_if_needed(ctx->list.Get(), final_detections->resource.Get(), final_detections->state, final_before);
            transition_if_needed(ctx->list.Get(), keep->resource.Get(), keep->state, keep_before);
            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, candidate_counter_before);
            transition_if_needed(ctx->list.Get(), candidates->resource.Get(), candidates->state, candidates_before);
            transition_if_needed(ctx->list.Get(), yolo->resource.Get(), yolo->state, yolo_before);
            final_counter->state = final_counter_before;
            final_detections->state = final_before;
            keep->state = keep_before;
            candidate_counter->state = candidate_counter_before;
            candidates->state = candidates_before;
            yolo->state = yolo_before;
            keep_descriptor_heap_alive(ctx, decode_heap);
            keep_descriptor_heap_alive(ctx, nms_heap);
            keep_descriptor_heap_alive(ctx, topk_heap);
            keep_resource_alive(ctx, zero_upload);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) {
        return raise_hr("YOLO decode/NMS/topK command submission", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_yolo_head_decode_nms_float32(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* box0_capsule = nullptr;
    PyObject* cls0_capsule = nullptr;
    PyObject* box1_capsule = nullptr;
    PyObject* cls1_capsule = nullptr;
    PyObject* box2_capsule = nullptr;
    PyObject* cls2_capsule = nullptr;
    PyObject* candidate_capsule = nullptr;
    PyObject* candidate_counter_capsule = nullptr;
    PyObject* keep_capsule = nullptr;
    PyObject* final_capsule = nullptr;
    PyObject* final_counter_capsule = nullptr;
    unsigned int h0 = 0, w0 = 0, h1 = 0, w1 = 0, h2 = 0, w2 = 0;
    unsigned int classes = 0;
    unsigned int max_candidates = 0;
    unsigned int max_detections = 0;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    float stride0 = 8.0f;
    float stride1 = 16.0f;
    float stride2 = 32.0f;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOOIIIIIIIIIfffff",
            &device_capsule,
            &box0_capsule,
            &cls0_capsule,
            &box1_capsule,
            &cls1_capsule,
            &box2_capsule,
            &cls2_capsule,
            &candidate_capsule,
            &candidate_counter_capsule,
            &keep_capsule,
            &final_capsule,
            &final_counter_capsule,
            &h0,
            &w0,
            &h1,
            &w1,
            &h2,
            &w2,
            &classes,
            &max_candidates,
            &max_detections,
            &conf_threshold,
            &iou_threshold,
            &stride0,
            &stride1,
            &stride2)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* box0 = get_buffer(box0_capsule);
    BufferHandle* cls0 = get_buffer(cls0_capsule);
    BufferHandle* box1 = get_buffer(box1_capsule);
    BufferHandle* cls1 = get_buffer(cls1_capsule);
    BufferHandle* box2 = get_buffer(box2_capsule);
    BufferHandle* cls2 = get_buffer(cls2_capsule);
    BufferHandle* candidates = get_buffer(candidate_capsule);
    BufferHandle* candidate_counter = get_buffer(candidate_counter_capsule);
    BufferHandle* keep = get_buffer(keep_capsule);
    BufferHandle* final_detections = get_buffer(final_capsule);
    BufferHandle* final_counter = get_buffer(final_counter_capsule);
    std::vector<BufferHandle*> inputs = {box0, cls0, box1, cls1, box2, cls2};
    if (!ctx || !box0 || !cls0 || !box1 || !cls1 || !box2 || !cls2 || !candidates || !candidate_counter || !keep || !final_detections || !final_counter) {
        return nullptr;
    }
    std::vector<BufferHandle*> owned = inputs;
    owned.insert(owned.end(), {candidates, candidate_counter, keep, final_detections, final_counter});
    if (!all_owned_by(ctx, owned)) {
        PyErr_SetString(PyExc_ValueError, "YOLO head NMS buffers belong to a different device");
        return nullptr;
    }
    const uint64_t p0 = uint64_t(h0) * w0;
    const uint64_t p1 = uint64_t(h1) * w1;
    const uint64_t p2 = uint64_t(h2) * w2;
    const uint64_t total_anchors = p0 + p1 + p2;
    if (p0 == 0 || p1 == 0 || p2 == 0 || total_anchors > UINT32_MAX || classes == 0 || max_candidates == 0 || max_detections == 0) {
        PyErr_SetString(PyExc_ValueError, "invalid YOLO head NMS dimensions");
        return nullptr;
    }
    if (box0->nbytes < p0 * 64 * sizeof(float) || box1->nbytes < p1 * 64 * sizeof(float) || box2->nbytes < p2 * 64 * sizeof(float) ||
        cls0->nbytes < p0 * classes * sizeof(float) || cls1->nbytes < p1 * classes * sizeof(float) || cls2->nbytes < p2 * classes * sizeof(float) ||
        candidates->nbytes < uint64_t(max_candidates) * 6 * sizeof(float) ||
        candidate_counter->nbytes < sizeof(uint32_t) ||
        keep->nbytes < uint64_t(max_candidates) * sizeof(uint32_t) ||
        final_detections->nbytes < uint64_t(max_detections) * 6 * sizeof(float) ||
        final_counter->nbytes < sizeof(uint32_t)) {
        PyErr_SetString(PyExc_ValueError, "YOLO head NMS buffers are too small");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_yolo_head_decode_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr)) hr = ensure_yolo_nms_mark_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr)) hr = ensure_yolo_topk_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "YOLO head NMS pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_yolo_head_nms_pipelines", hr);
    }

    ComPtr<ID3D12Resource> zero_upload;
    hr = create_committed_buffer(ctx->device.Get(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, sizeof(uint32_t) * 2, &zero_upload);
    if (FAILED(hr)) return raise_hr("CreateCommittedResource(yolo head zero upload)", hr);
    uint32_t* mapped_zero = nullptr;
    D3D12_RANGE read_range{0, 0};
    hr = zero_upload->Map(0, &read_range, reinterpret_cast<void**>(&mapped_zero));
    if (FAILED(hr)) return raise_hr("Map(yolo head zero upload)", hr);
    mapped_zero[0] = 0;
    mapped_zero[1] = 0;
    zero_upload->Unmap(0, nullptr);

    ComPtr<ID3D12DescriptorHeap> decode_heap;
    hr = create_heap(ctx, 8, &decode_heap);
    if (FAILED(hr)) return raise_hr("CreateDescriptorHeap(yolo head decode)", hr);
    ComPtr<ID3D12DescriptorHeap> nms_heap;
    hr = create_heap(ctx, 3, &nms_heap);
    if (FAILED(hr)) return raise_hr("CreateDescriptorHeap(yolo head nms)", hr);
    ComPtr<ID3D12DescriptorHeap> topk_heap;
    hr = create_heap(ctx, 5, &topk_heap);
    if (FAILED(hr)) return raise_hr("CreateDescriptorHeap(yolo head topk)", hr);

    const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = decode_heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, box0, static_cast<UINT>(p0 * 64), cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, cls0, static_cast<UINT>(p0 * classes), cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, box1, static_cast<UINT>(p1 * 64), cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, cls1, static_cast<UINT>(p1 * classes), cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, box2, static_cast<UINT>(p2 * 64), cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, cls2, static_cast<UINT>(p2 * classes), cursor); cursor.ptr += descriptor_size;
    create_float_uav(ctx, candidates, max_candidates * 6, cursor); cursor.ptr += descriptor_size;
    create_uint_uav(ctx, candidate_counter, 1, cursor);

    cursor = nms_heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, candidates, max_candidates * 6, cursor); cursor.ptr += descriptor_size;
    create_uint_srv(ctx, candidate_counter, 1, cursor); cursor.ptr += descriptor_size;
    create_uint_uav(ctx, keep, max_candidates, cursor);

    cursor = topk_heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, candidates, max_candidates * 6, cursor); cursor.ptr += descriptor_size;
    create_uint_srv(ctx, candidate_counter, 1, cursor); cursor.ptr += descriptor_size;
    create_uint_srv(ctx, keep, max_candidates, cursor); cursor.ptr += descriptor_size;
    create_float_uav(ctx, final_detections, max_detections * 6, cursor); cursor.ptr += descriptor_size;
    create_uint_uav(ctx, final_counter, 1, cursor);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            std::vector<D3D12_RESOURCE_STATES> input_before;
            input_before.reserve(inputs.size());
            for (auto* input : inputs) {
                input_before.push_back(input->state);
                transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                input->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            auto candidates_before = candidates->state;
            auto candidate_counter_before = candidate_counter->state;
            auto keep_before = keep->state;
            auto final_before = final_detections->state;
            auto final_counter_before = final_counter->state;

            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, D3D12_RESOURCE_STATE_COPY_DEST);
            transition_if_needed(ctx->list.Get(), final_counter->resource.Get(), final_counter->state, D3D12_RESOURCE_STATE_COPY_DEST);
            candidate_counter->state = D3D12_RESOURCE_STATE_COPY_DEST;
            final_counter->state = D3D12_RESOURCE_STATE_COPY_DEST;
            ctx->list->CopyBufferRegion(candidate_counter->resource.Get(), 0, zero_upload.Get(), 0, sizeof(uint32_t));
            ctx->list->CopyBufferRegion(final_counter->resource.Get(), 0, zero_upload.Get(), sizeof(uint32_t), sizeof(uint32_t));

            transition_if_needed(ctx->list.Get(), candidates->resource.Get(), candidates->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            candidates->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            candidate_counter->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

            ID3D12DescriptorHeap* heaps[] = {decode_heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = decode_heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 6;
            ctx->list->SetComputeRootSignature(ctx->yolo_head_root_signature.Get());
            ctx->list->SetPipelineState(ctx->yolo_head_decode_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT constants[14] = {h0, w0, h1, w1, h2, w2, classes, max_candidates, 0, 0, 0, 0, static_cast<UINT>(total_anchors), 0};
            std::memcpy(&constants[8], &conf_threshold, sizeof(float));
            std::memcpy(&constants[9], &stride0, sizeof(float));
            std::memcpy(&constants[10], &stride1, sizeof(float));
            std::memcpy(&constants[11], &stride2, sizeof(float));
            ctx->list->SetComputeRoot32BitConstants(2, 14, constants, 0);
            ctx->list->Dispatch((static_cast<UINT>(total_anchors) + 255) / 256, 1, 1);
            uav_barrier(ctx->list.Get(), candidates->resource.Get());
            uav_barrier(ctx->list.Get(), candidate_counter->resource.Get());

            transition_if_needed(ctx->list.Get(), candidates->resource.Get(), candidates->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), keep->resource.Get(), keep->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            candidates->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            candidate_counter->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            keep->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            heaps[0] = nms_heap.Get();
            ctx->list->SetDescriptorHeaps(1, heaps);
            srv_gpu = nms_heap->GetGPUDescriptorHandleForHeapStart();
            uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 2;
            ctx->list->SetComputeRootSignature(ctx->yolo_nms_root_signature.Get());
            ctx->list->SetPipelineState(ctx->yolo_nms_mark_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT nms_constants[2] = {max_candidates, 0};
            std::memcpy(&nms_constants[1], &iou_threshold, sizeof(float));
            ctx->list->SetComputeRoot32BitConstants(2, 2, nms_constants, 0);
            ctx->list->Dispatch((max_candidates + 127) / 128, 1, 1);
            uav_barrier(ctx->list.Get(), keep->resource.Get());

            transition_if_needed(ctx->list.Get(), keep->resource.Get(), keep->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), final_detections->resource.Get(), final_detections->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition_if_needed(ctx->list.Get(), final_counter->resource.Get(), final_counter->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            keep->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            final_detections->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            final_counter->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            heaps[0] = topk_heap.Get();
            ctx->list->SetDescriptorHeaps(1, heaps);
            srv_gpu = topk_heap->GetGPUDescriptorHandleForHeapStart();
            uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 3;
            ctx->list->SetComputeRootSignature(ctx->yolo_topk_root_signature.Get());
            ctx->list->SetPipelineState(ctx->yolo_topk_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT topk_constants[2] = {max_candidates, max_detections};
            ctx->list->SetComputeRoot32BitConstants(2, 2, topk_constants, 0);
            ctx->list->Dispatch((max_candidates + 127) / 128, 1, 1);

            transition_if_needed(ctx->list.Get(), final_counter->resource.Get(), final_counter->state, final_counter_before);
            transition_if_needed(ctx->list.Get(), final_detections->resource.Get(), final_detections->state, final_before);
            transition_if_needed(ctx->list.Get(), keep->resource.Get(), keep->state, keep_before);
            transition_if_needed(ctx->list.Get(), candidate_counter->resource.Get(), candidate_counter->state, candidate_counter_before);
            transition_if_needed(ctx->list.Get(), candidates->resource.Get(), candidates->state, candidates_before);
            final_counter->state = final_counter_before;
            final_detections->state = final_before;
            keep->state = keep_before;
            candidate_counter->state = candidate_counter_before;
            candidates->state = candidates_before;
            for (size_t i = 0; i < inputs.size(); ++i) {
                transition_if_needed(ctx->list.Get(), inputs[i]->resource.Get(), inputs[i]->state, input_before[i]);
                inputs[i]->state = input_before[i];
            }
            keep_descriptor_heap_alive(ctx, decode_heap);
            keep_descriptor_heap_alive(ctx, nms_heap);
            keep_descriptor_heap_alive(ctx, topk_heap);
            keep_resource_alive(ctx, zero_upload);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) {
        return raise_hr("YOLO head decode/NMS/topK command submission", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_conv2d_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    Conv2DDesc desc{};
    if (!PyArg_ParseTuple(
            args,
            "OOOOO(IIIIIIIIIIIIIIII)",
            &device_capsule,
            &input_capsule,
            &weight_capsule,
            &bias_capsule,
            &output_capsule,
            &desc.batch,
            &desc.in_channels,
            &desc.in_h,
            &desc.in_w,
            &desc.out_channels,
            &desc.out_h,
            &desc.out_w,
            &desc.kernel_h,
            &desc.kernel_w,
            &desc.stride_h,
            &desc.stride_w,
            &desc.pad_top,
            &desc.pad_left,
            &desc.dilation_h,
            &desc.dilation_w,
            &desc.groups)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    const char* reason = nullptr;
    if (!validate_conv_silu_args(ctx, input, weight, bias, output, desc, &reason)) {
        PyErr_SetString(PyExc_ValueError, reason ? reason : "invalid Conv2D arguments");
        return nullptr;
    }
    std::string pipeline_error;
    HRESULT hr = ensure_conv_linear_pipeline(ctx, &pipeline_error);
    if (SUCCEEDED(hr) && is_conv1x1_fast_path(desc)) {
        hr = ensure_conv1x1_pipeline(ctx, false, &pipeline_error);
    }
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "Conv2D pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_conv_linear_pipeline", hr);
    }
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_conv_silu_descriptor_heap(ctx, input, weight, bias, output, desc, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_conv_descriptor_heap", hr);
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            const UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 3;
            auto input_before = input->state;
            auto weight_before = weight->state;
            auto bias_before = bias->state;
            auto output_before = output->state;
            transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), weight->resource.Get(), weight->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), bias->resource.Get(), bias->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            input->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            weight->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bias->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            ctx->list->SetComputeRootSignature(ctx->conv_root_signature.Get());
            ID3D12PipelineState* pso = (is_conv1x1_fast_path(desc) && ctx->conv1x1_linear_pso) ? ctx->conv1x1_linear_pso.Get() : ctx->conv_linear_pso.Get();
            ctx->list->SetPipelineState(pso);
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT constants[16]{};
            conv_constants(desc, constants);
            ctx->list->SetComputeRoot32BitConstants(2, 16, constants, 0);
            ctx->list->Dispatch(static_cast<UINT>((conv_output_elements(desc) + 255) / 256), 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, output_before);
            transition_if_needed(ctx->list.Get(), bias->resource.Get(), bias->state, bias_before);
            transition_if_needed(ctx->list.Get(), weight->resource.Get(), weight->state, weight_before);
            transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, input_before);
            output->state = output_before;
            bias->state = bias_before;
            weight->state = weight_before;
            input->state = input_before;
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("conv2d dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_concat_conv1x1_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* inputs_obj = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    unsigned int activation = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOI", &device_capsule, &inputs_obj, &weight_capsule, &bias_capsule, &output_capsule, &constants_obj, &activation)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, 16)) return nullptr;
    PyObject* seq = PySequence_Fast(inputs_obj, "inputs must be a sequence");
    if (!seq) return nullptr;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "concat-conv supports 1..8 inputs");
        return nullptr;
    }
    std::vector<BufferHandle*> inputs;
    inputs.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        inputs.push_back(get_buffer(PySequence_Fast_GET_ITEM(seq, i)));
    }
    Py_DECREF(seq);

    std::vector<BufferHandle*> all = inputs;
    all.push_back(weight);
    all.push_back(bias);
    all.push_back(output);
    if (!ctx || !all_owned_by(ctx, all)) {
        PyErr_SetString(PyExc_ValueError, "invalid concat-conv buffers");
        return nullptr;
    }
    if (activation > 1) {
        PyErr_SetString(PyExc_ValueError, "concat-conv activation must be 0(linear) or 1(silu)");
        return nullptr;
    }

    const UINT batch = constants[0];
    const UINT total_in_channels = constants[1];
    const UINT in_h = constants[2];
    const UINT in_w = constants[3];
    const UINT out_channels = constants[4];
    const UINT out_h = constants[5];
    const UINT out_w = constants[6];
    const UINT input_count = constants[7];
    if (input_count != static_cast<UINT>(inputs.size()) || input_count == 0 || input_count > 8) {
        PyErr_SetString(PyExc_ValueError, "concat-conv input_count does not match inputs");
        return nullptr;
    }
    if (batch == 0 || total_in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 || out_h != in_h || out_w != in_w) {
        PyErr_SetString(PyExc_ValueError, "invalid concat-conv constants");
        return nullptr;
    }
    UINT summed_channels = 0;
    for (UINT i = 0; i < input_count; ++i) {
        const UINT channels = constants[8 + i];
        summed_channels += channels;
        const uint64_t required = uint64_t(batch) * channels * in_h * in_w * sizeof(float);
        if (channels == 0 || inputs[i]->nbytes < required) {
            PyErr_SetString(PyExc_ValueError, "concat-conv input buffer is smaller than constants require");
            return nullptr;
        }
    }
    const uint64_t weight_elements = uint64_t(out_channels) * total_in_channels;
    const uint64_t float_weight_bytes = weight_elements * sizeof(float);
    const uint64_t fp16_packed_weight_bytes = ((weight_elements + 1) / 2) * sizeof(uint32_t);
    const bool fp16_packed_weight = weight->nbytes >= fp16_packed_weight_bytes && weight->nbytes < float_weight_bytes;
    if (summed_channels != total_in_channels ||
        weight->nbytes < (fp16_packed_weight ? fp16_packed_weight_bytes : float_weight_bytes) ||
        bias->nbytes < uint64_t(out_channels) * sizeof(float) ||
        output->nbytes < uint64_t(batch) * out_channels * out_h * out_w * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "concat-conv buffer sizes do not match constants");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = fp16_packed_weight
        ? ensure_concat_conv1x1_fp16_pipeline(ctx, activation != 0, &pipeline_error)
        : ensure_concat_conv1x1_pipeline(ctx, activation != 0, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "Concat+1x1Conv pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_concat_conv1x1_pipeline", hr);
    }

    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 11, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_concat_conv_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 8; ++i) {
        const UINT source_index = i < input_count ? i : input_count - 1;
        BufferHandle* src = inputs[source_index];
        const UINT channels = constants[8 + source_index];
        create_float_srv(ctx, src, batch * channels * in_h * in_w, cursor);
        cursor.ptr += descriptor_size;
    }
    if (fp16_packed_weight) {
        create_uint_srv(ctx, weight, static_cast<UINT>((weight_elements + 1) / 2), cursor);
    } else {
        create_float_srv(ctx, weight, static_cast<UINT>(weight_elements), cursor);
    }
    cursor.ptr += descriptor_size;
    create_float_srv(ctx, bias, out_channels, cursor);
    cursor.ptr += descriptor_size;
    create_float_uav(ctx, output, batch * out_channels * out_h * out_w, cursor);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            std::vector<ID3D12Resource*> unique_resources;
            std::vector<D3D12_RESOURCE_STATES> unique_before;
            auto transition_unique = [&](BufferHandle* buffer) {
                ID3D12Resource* resource = buffer->resource.Get();
                for (auto* existing : unique_resources) {
                    if (existing == resource) {
                        buffer->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                        return;
                    }
                }
                unique_resources.push_back(resource);
                unique_before.push_back(buffer->state);
                transition_if_needed(ctx->list.Get(), resource, buffer->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                buffer->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            };
            for (auto* src : inputs) transition_unique(src);
            transition_unique(weight);
            transition_unique(bias);
            auto ob = output->state;
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 10;
            ctx->list->SetComputeRootSignature(ctx->concat_conv_root_signature.Get());
            if (fp16_packed_weight) {
                ctx->list->SetPipelineState(activation != 0 ? ctx->concat_conv1x1_fp16_silu_pso.Get() : ctx->concat_conv1x1_fp16_linear_pso.Get());
            } else {
                ctx->list->SetPipelineState(activation != 0 ? ctx->concat_conv1x1_silu_pso.Get() : ctx->concat_conv1x1_linear_pso.Get());
            }
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, 16, constants.data(), 0);
            const UINT count = batch * out_channels * out_h * out_w;
            ctx->list->Dispatch((count + 255) / 256, 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            output->state = ob;
            for (size_t i = 0; i < unique_resources.size(); ++i) {
                transition_if_needed(ctx->list.Get(), unique_resources[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, unique_before[i]);
            }
            for (auto* src : inputs) {
                for (size_t i = 0; i < unique_resources.size(); ++i) {
                    if (src->resource.Get() == unique_resources[i]) {
                        src->state = unique_before[i];
                        break;
                    }
                }
            }
            for (size_t i = 0; i < unique_resources.size(); ++i) {
                if (weight->resource.Get() == unique_resources[i]) weight->state = unique_before[i];
                if (bias->resource.Get() == unique_resources[i]) bias->state = unique_before[i];
            }
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("concat-conv dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_concat_conv1x1_int8_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* inputs_obj = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* scale_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    unsigned int activation = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOI", &device_capsule, &inputs_obj, &weight_capsule, &scale_capsule, &bias_capsule, &output_capsule, &constants_obj, &activation)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* scale = get_buffer(scale_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, 20)) return nullptr;
    PyObject* seq = PySequence_Fast(inputs_obj, "inputs must be a sequence");
    if (!seq) return nullptr;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "int8 concat-conv supports 1..8 inputs");
        return nullptr;
    }
    std::vector<BufferHandle*> inputs;
    inputs.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        inputs.push_back(get_buffer(PySequence_Fast_GET_ITEM(seq, i)));
    }
    Py_DECREF(seq);

    std::vector<BufferHandle*> all = inputs;
    all.push_back(weight);
    all.push_back(scale);
    all.push_back(bias);
    all.push_back(output);
    if (!ctx || !all_owned_by(ctx, all)) {
        PyErr_SetString(PyExc_ValueError, "invalid int8 concat-conv buffers");
        return nullptr;
    }
    if (activation > 1) {
        PyErr_SetString(PyExc_ValueError, "int8 concat-conv activation must be 0(linear) or 1(silu)");
        return nullptr;
    }

    const UINT batch = constants[0];
    const UINT total_in_channels = constants[1];
    const UINT in_h = constants[2];
    const UINT in_w = constants[3];
    const UINT out_channels = constants[4];
    const UINT out_h = constants[5];
    const UINT out_w = constants[6];
    const UINT input_count = constants[7];
    if (input_count != static_cast<UINT>(inputs.size()) || input_count == 0 || input_count > 8) {
        PyErr_SetString(PyExc_ValueError, "int8 concat-conv input_count does not match inputs");
        return nullptr;
    }
    if (batch == 0 || total_in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 || out_h != in_h || out_w != in_w) {
        PyErr_SetString(PyExc_ValueError, "invalid int8 concat-conv constants");
        return nullptr;
    }
    UINT summed_channels = 0;
    for (UINT i = 0; i < input_count; ++i) {
        const UINT channels = constants[8 + i];
        summed_channels += channels;
        const uint64_t required = uint64_t(batch) * channels * in_h * in_w * sizeof(float);
        if (channels == 0 || inputs[i]->nbytes < required) {
            PyErr_SetString(PyExc_ValueError, "int8 concat-conv input buffer is smaller than constants require");
            return nullptr;
        }
    }
    const uint64_t weight_elements = uint64_t(out_channels) * total_in_channels;
    const uint64_t packed_weight_bytes = ((weight_elements + 3) / 4) * sizeof(uint32_t);
    if (summed_channels != total_in_channels ||
        weight->nbytes < packed_weight_bytes ||
        scale->nbytes < uint64_t(out_channels) * sizeof(float) ||
        bias->nbytes < uint64_t(out_channels) * sizeof(float) ||
        output->nbytes < uint64_t(batch) * out_channels * out_h * out_w * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "int8 concat-conv buffer sizes do not match constants");
        return nullptr;
    }

    std::string pipeline_error;
    HRESULT hr = ensure_concat_conv1x1_int8_pipeline(ctx, activation != 0, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "INT8 Concat+1x1Conv pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_concat_conv1x1_int8_pipeline", hr);
    }

    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 12, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_int8_concat_conv_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 8; ++i) {
        const UINT source_index = i < input_count ? i : input_count - 1;
        BufferHandle* src = inputs[source_index];
        const UINT channels = constants[8 + source_index];
        create_float_srv(ctx, src, batch * channels * in_h * in_w, cursor);
        cursor.ptr += descriptor_size;
    }
    create_uint_srv(ctx, weight, static_cast<UINT>((weight_elements + 3) / 4), cursor);
    cursor.ptr += descriptor_size;
    create_float_srv(ctx, scale, out_channels, cursor);
    cursor.ptr += descriptor_size;
    create_float_srv(ctx, bias, out_channels, cursor);
    cursor.ptr += descriptor_size;
    create_float_uav(ctx, output, batch * out_channels * out_h * out_w, cursor);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            std::vector<ID3D12Resource*> unique_resources;
            std::vector<D3D12_RESOURCE_STATES> unique_before;
            auto transition_unique = [&](BufferHandle* buffer) {
                ID3D12Resource* resource = buffer->resource.Get();
                for (auto* existing : unique_resources) {
                    if (existing == resource) {
                        buffer->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                        return;
                    }
                }
                unique_resources.push_back(resource);
                unique_before.push_back(buffer->state);
                transition_if_needed(ctx->list.Get(), resource, buffer->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                buffer->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            };
            for (auto* src : inputs) transition_unique(src);
            transition_unique(weight);
            transition_unique(scale);
            transition_unique(bias);
            auto ob = output->state;
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 11;
            ctx->list->SetComputeRootSignature(ctx->concat_conv_int8_root_signature.Get());
            ctx->list->SetPipelineState(activation != 0 ? ctx->concat_conv1x1_int8_silu_pso.Get() : ctx->concat_conv1x1_int8_linear_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, 20, constants.data(), 0);
            const UINT count = batch * out_channels * out_h * out_w;
            ctx->list->Dispatch((count + 255) / 256, 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            output->state = ob;
            for (size_t i = 0; i < unique_resources.size(); ++i) {
                transition_if_needed(ctx->list.Get(), unique_resources[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, unique_before[i]);
            }
            for (auto* src : inputs) {
                for (size_t i = 0; i < unique_resources.size(); ++i) {
                    if (src->resource.Get() == unique_resources[i]) {
                        src->state = unique_before[i];
                        break;
                    }
                }
            }
            for (size_t i = 0; i < unique_resources.size(); ++i) {
                if (weight->resource.Get() == unique_resources[i]) weight->state = unique_before[i];
                if (scale->resource.Get() == unique_resources[i]) scale->state = unique_before[i];
                if (bias->resource.Get() == unique_resources[i]) bias->state = unique_before[i];
            }
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("int8 concat-conv dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_concat_residual_conv1x1_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* inputs_obj = nullptr;
    PyObject* residuals_obj = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    unsigned int activation = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOI", &device_capsule, &inputs_obj, &residuals_obj, &weight_capsule, &bias_capsule, &output_capsule, &constants_obj, &activation)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, 24)) return nullptr;
    PyObject* input_seq = PySequence_Fast(inputs_obj, "inputs must be a sequence");
    if (!input_seq) return nullptr;
    PyObject* residual_seq = PySequence_Fast(residuals_obj, "residuals must be a sequence");
    if (!residual_seq) {
        Py_DECREF(input_seq);
        return nullptr;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(input_seq);
    if (n <= 0 || n > 8 || PySequence_Fast_GET_SIZE(residual_seq) != n) {
        Py_DECREF(input_seq);
        Py_DECREF(residual_seq);
        PyErr_SetString(PyExc_ValueError, "concat-residual-conv supports matching 1..8 inputs/residuals");
        return nullptr;
    }
    std::vector<BufferHandle*> inputs;
    std::vector<BufferHandle*> residuals;
    inputs.reserve(static_cast<size_t>(n));
    residuals.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        inputs.push_back(get_buffer(PySequence_Fast_GET_ITEM(input_seq, i)));
        residuals.push_back(get_buffer(PySequence_Fast_GET_ITEM(residual_seq, i)));
    }
    Py_DECREF(input_seq);
    Py_DECREF(residual_seq);

    std::vector<BufferHandle*> all = inputs;
    all.insert(all.end(), residuals.begin(), residuals.end());
    all.push_back(weight);
    all.push_back(bias);
    all.push_back(output);
    if (!ctx || !all_owned_by(ctx, all)) {
        PyErr_SetString(PyExc_ValueError, "invalid concat-residual-conv buffers");
        return nullptr;
    }
    if (activation > 1) {
        PyErr_SetString(PyExc_ValueError, "concat-residual-conv activation must be 0(linear) or 1(silu)");
        return nullptr;
    }

    const UINT batch = constants[0];
    const UINT total_in_channels = constants[1];
    const UINT in_h = constants[2];
    const UINT in_w = constants[3];
    const UINT out_channels = constants[4];
    const UINT out_h = constants[5];
    const UINT out_w = constants[6];
    const UINT input_count = constants[7];
    if (input_count != static_cast<UINT>(inputs.size()) || input_count == 0 || input_count > 8) {
        PyErr_SetString(PyExc_ValueError, "concat-residual-conv input_count does not match inputs");
        return nullptr;
    }
    if (batch == 0 || total_in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 || out_h != in_h || out_w != in_w) {
        PyErr_SetString(PyExc_ValueError, "invalid concat-residual-conv constants");
        return nullptr;
    }
    UINT summed_channels = 0;
    for (UINT i = 0; i < input_count; ++i) {
        const UINT channels = constants[8 + i];
        const UINT has_residual = constants[16 + i];
        summed_channels += channels;
        const uint64_t required = uint64_t(batch) * channels * in_h * in_w * sizeof(float);
        if (channels == 0 || inputs[i]->nbytes < required || (has_residual && residuals[i]->nbytes < required)) {
            PyErr_SetString(PyExc_ValueError, "concat-residual-conv input/residual buffer is smaller than constants require");
            return nullptr;
        }
    }
    if (summed_channels != total_in_channels ||
        weight->nbytes < uint64_t(out_channels) * total_in_channels * sizeof(float) ||
        bias->nbytes < uint64_t(out_channels) * sizeof(float) ||
        output->nbytes < uint64_t(batch) * out_channels * out_h * out_w * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "concat-residual-conv buffer sizes do not match constants");
        return nullptr;
    }

    const bool tiled_silu = activation != 0 && std::getenv("AEXRT_NATIVE_D3D12_C2F_SUPERBLOCK_TILED") != nullptr;
    std::string pipeline_error;
    HRESULT hr = tiled_silu
        ? ensure_concat_residual_conv1x1_tiled_silu_pipeline(ctx, &pipeline_error)
        : ensure_concat_residual_conv1x1_pipeline(ctx, activation != 0, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "Concat+Residual+1x1Conv pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_concat_residual_conv1x1_pipeline", hr);
    }

    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 19, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_concat_residual_conv_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 8; ++i) {
        const UINT source_index = i < input_count ? i : input_count - 1;
        BufferHandle* src = inputs[source_index];
        const UINT channels = constants[8 + source_index];
        create_float_srv(ctx, src, batch * channels * in_h * in_w, cursor);
        cursor.ptr += descriptor_size;
    }
    for (UINT i = 0; i < 8; ++i) {
        const UINT source_index = i < input_count ? i : input_count - 1;
        BufferHandle* src = residuals[source_index];
        const UINT channels = constants[8 + source_index];
        create_float_srv(ctx, src, batch * channels * in_h * in_w, cursor);
        cursor.ptr += descriptor_size;
    }
    create_float_srv(ctx, weight, out_channels * total_in_channels, cursor);
    cursor.ptr += descriptor_size;
    create_float_srv(ctx, bias, out_channels, cursor);
    cursor.ptr += descriptor_size;
    create_float_uav(ctx, output, batch * out_channels * out_h * out_w, cursor);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            std::vector<ID3D12Resource*> unique_resources;
            std::vector<D3D12_RESOURCE_STATES> unique_before;
            auto transition_unique = [&](BufferHandle* buffer) {
                ID3D12Resource* resource = buffer->resource.Get();
                for (auto* existing : unique_resources) {
                    if (existing == resource) {
                        buffer->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                        return;
                    }
                }
                unique_resources.push_back(resource);
                unique_before.push_back(buffer->state);
                transition_if_needed(ctx->list.Get(), resource, buffer->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                buffer->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            };
            for (auto* src : inputs) transition_unique(src);
            for (auto* src : residuals) transition_unique(src);
            transition_unique(weight);
            transition_unique(bias);
            auto ob = output->state;
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 18;
            ctx->list->SetComputeRootSignature(ctx->concat_residual_conv_root_signature.Get());
            ctx->list->SetPipelineState(tiled_silu ? ctx->concat_residual_conv1x1_tiled_silu_pso.Get() : (activation != 0 ? ctx->concat_residual_conv1x1_silu_pso.Get() : ctx->concat_residual_conv1x1_linear_pso.Get()));
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, 24, constants.data(), 0);
            const UINT count = batch * out_channels * out_h * out_w;
            if (tiled_silu) {
                ctx->list->Dispatch((out_h * out_w + 15) / 16, (out_channels + 7) / 8, batch);
            } else {
                ctx->list->Dispatch((count + 255) / 256, 1, 1);
            }
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            output->state = ob;
            for (size_t i = 0; i < unique_resources.size(); ++i) {
                transition_if_needed(ctx->list.Get(), unique_resources[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, unique_before[i]);
            }
            auto restore = [&](BufferHandle* buffer) {
                for (size_t i = 0; i < unique_resources.size(); ++i) {
                    if (buffer->resource.Get() == unique_resources[i]) {
                        buffer->state = unique_before[i];
                        return;
                    }
                }
            };
            for (auto* src : inputs) restore(src);
            for (auto* src : residuals) restore(src);
            restore(weight);
            restore(bias);
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("concat-residual-conv dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_c2f_bottleneck_tiled_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* w1_capsule = nullptr;
    PyObject* b1_capsule = nullptr;
    PyObject* w2_capsule = nullptr;
    PyObject* b2_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &device_capsule, &input_capsule, &w1_capsule, &b1_capsule, &w2_capsule, &b2_capsule, &output_capsule, &constants_obj)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* w1 = get_buffer(w1_capsule);
    BufferHandle* b1 = get_buffer(b1_capsule);
    BufferHandle* w2 = get_buffer(w2_capsule);
    BufferHandle* b2 = get_buffer(b2_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, 8)) return nullptr;
    if (!ctx || !all_owned_by(ctx, {input, w1, b1, w2, b2, output})) {
        PyErr_SetString(PyExc_ValueError, "invalid C2f bottleneck buffers");
        return nullptr;
    }
    const UINT batch = constants[0];
    const UINT in_channels = constants[1];
    const UINT h = constants[2];
    const UINT w = constants[3];
    const UINT mid_channels = constants[4];
    const UINT out_channels = constants[5];
    const UINT total = constants[6];
    if (batch == 0 || in_channels == 0 || h == 0 || w == 0 || mid_channels == 0 || out_channels == 0 ||
        total != batch * out_channels * h * w || in_channels != out_channels) {
        PyErr_SetString(PyExc_ValueError, "invalid C2f bottleneck constants");
        return nullptr;
    }
    if (input->nbytes < uint64_t(batch) * in_channels * h * w * sizeof(float) ||
        w1->nbytes < uint64_t(mid_channels) * in_channels * 9 * sizeof(float) ||
        b1->nbytes < uint64_t(mid_channels) * sizeof(float) ||
        w2->nbytes < uint64_t(out_channels) * mid_channels * 9 * sizeof(float) ||
        b2->nbytes < uint64_t(out_channels) * sizeof(float) ||
        output->nbytes < uint64_t(total) * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "C2f bottleneck buffer sizes do not match constants");
        return nullptr;
    }
    std::string pipeline_error;
    HRESULT hr = ensure_c2f_bottleneck_tiled_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "C2f bottleneck tiled pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_c2f_bottleneck_tiled_pipeline", hr);
    }
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 6, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_c2f_bottleneck_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, input, batch * in_channels * h * w, cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, w1, mid_channels * in_channels * 9, cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, b1, mid_channels, cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, w2, out_channels * mid_channels * 9, cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, b2, out_channels, cursor); cursor.ptr += descriptor_size;
    create_float_uav(ctx, output, total, cursor);
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            std::vector<BufferHandle*> srvs = {input, w1, b1, w2, b2};
            std::vector<D3D12_RESOURCE_STATES> before;
            before.reserve(srvs.size());
            for (auto* src : srvs) {
                before.push_back(src->state);
                transition_if_needed(ctx->list.Get(), src->resource.Get(), src->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                src->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            auto ob = output->state;
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 5;
            ctx->list->SetComputeRootSignature(ctx->c2f_bottleneck_root_signature.Get());
            ctx->list->SetPipelineState(ctx->c2f_bottleneck_tiled_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, 8, constants.data(), 0);
            ctx->list->Dispatch((w + 7) / 8, (h + 7) / 8, batch * ((out_channels + 15) / 16));
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            output->state = ob;
            for (size_t i = 0; i < srvs.size(); ++i) {
                transition_if_needed(ctx->list.Get(), srvs[i]->resource.Get(), srvs[i]->state, before[i]);
                srvs[i]->state = before[i];
            }
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("C2f bottleneck tiled dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_sppf_tail_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* weight_capsule = nullptr;
    PyObject* bias_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    unsigned int activation = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOI", &device_capsule, &input_capsule, &weight_capsule, &bias_capsule, &output_capsule, &constants_obj, &activation)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* weight = get_buffer(weight_capsule);
    BufferHandle* bias = get_buffer(bias_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, 8)) return nullptr;
    if (!ctx || !all_owned_by(ctx, {input, weight, bias, output})) {
        PyErr_SetString(PyExc_ValueError, "invalid SPPF tail buffers");
        return nullptr;
    }
    if (activation > 1) {
        PyErr_SetString(PyExc_ValueError, "SPPF tail activation must be 0(linear) or 1(silu)");
        return nullptr;
    }
    const UINT batch = constants[0];
    const UINT in_channels = constants[1];
    const UINT h = constants[2];
    const UINT w = constants[3];
    const UINT out_channels = constants[4];
    const UINT total = constants[5];
    if (batch == 0 || in_channels == 0 || h == 0 || w == 0 || out_channels == 0 ||
        total != batch * out_channels * h * w ||
        input->nbytes < uint64_t(batch) * in_channels * h * w * sizeof(float) ||
        weight->nbytes < uint64_t(out_channels) * in_channels * 4 * sizeof(float) ||
        bias->nbytes < uint64_t(out_channels) * sizeof(float) ||
        output->nbytes < uint64_t(total) * sizeof(float)) {
        PyErr_SetString(PyExc_ValueError, "SPPF tail buffer sizes do not match constants");
        return nullptr;
    }
    std::string pipeline_error;
    HRESULT hr = ensure_sppf_tail_pipeline(ctx, activation != 0, &pipeline_error);
    if (FAILED(hr)) {
        if (!pipeline_error.empty()) {
            PyErr_Format(PyExc_RuntimeError, "SPPF tail pipeline creation failed: %s", pipeline_error.c_str());
            return nullptr;
        }
        return raise_hr("ensure_sppf_tail_pipeline", hr);
    }
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 4, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_sppf_tail_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, input, batch * in_channels * h * w, cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, weight, out_channels * in_channels * 4, cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, bias, out_channels, cursor); cursor.ptr += descriptor_size;
    create_float_uav(ctx, output, total, cursor);
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            std::vector<BufferHandle*> srvs = {input, weight, bias};
            std::vector<D3D12_RESOURCE_STATES> before;
            before.reserve(srvs.size());
            for (auto* src : srvs) {
                before.push_back(src->state);
                transition_if_needed(ctx->list.Get(), src->resource.Get(), src->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                src->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            auto ob = output->state;
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu;
            uav_gpu.ptr += descriptor_size * 3;
            ctx->list->SetComputeRootSignature(ctx->sppf_tail_root_signature.Get());
            ctx->list->SetPipelineState(activation != 0 ? ctx->sppf_tail_silu_pso.Get() : ctx->sppf_tail_linear_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, 8, constants.data(), 0);
            ctx->list->Dispatch((total + 255) / 256, 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            output->state = ob;
            for (size_t i = 0; i < srvs.size(); ++i) {
                transition_if_needed(ctx->list.Get(), srvs[i]->resource.Get(), srvs[i]->state, before[i]);
                srvs[i]->state = before[i];
            }
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("SPPF tail dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_unary_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    unsigned int element_count = 0;
    unsigned int op = 0;
    if (!PyArg_ParseTuple(args, "OOOII", &device_capsule, &input_capsule, &output_capsule, &element_count, &op)) return nullptr;
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    if (!ctx || !input || !output || input->owner != ctx || output->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "invalid unary buffers");
        return nullptr;
    }
    if (input->nbytes < uint64_t(element_count) * 4 || output->nbytes < uint64_t(element_count) * 4) {
        PyErr_SetString(PyExc_ValueError, "unary buffer is too small");
        return nullptr;
    }
    std::string pipeline_error;
    HRESULT hr = ensure_unary_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) return raise_hr("ensure_unary_pipeline", hr);
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 2, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_unary_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, input, element_count, cursor);
    cursor.ptr += descriptor_size;
    create_float_uav(ctx, output, element_count, cursor);
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            auto ib = input->state, ob = output->state;
            transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            input->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu; uav_gpu.ptr += descriptor_size;
            ctx->list->SetComputeRootSignature(ctx->unary_root_signature.Get());
            ctx->list->SetPipelineState(ctx->unary_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            UINT constants[2] = {element_count, op};
            ctx->list->SetComputeRoot32BitConstants(2, 2, constants, 0);
            ctx->list->Dispatch((element_count + 255) / 256, 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, ib);
            output->state = ob; input->state = ib;
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("unary dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_binary_broadcast_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* a_capsule = nullptr;
    PyObject* b_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOOOO", &device_capsule, &a_capsule, &b_capsule, &output_capsule, &constants_obj)) return nullptr;
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* a = get_buffer(a_capsule);
    BufferHandle* b = get_buffer(b_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, 24)) return nullptr;
    if (!ctx || !all_owned_by(ctx, {a, b, output})) {
        PyErr_SetString(PyExc_ValueError, "invalid binary buffers");
        return nullptr;
    }
    UINT count = constants[0];
    std::string pipeline_error;
    HRESULT hr = ensure_binary_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) return raise_hr("ensure_binary_pipeline", hr);
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 3, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_binary_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, a, static_cast<UINT>(a->nbytes / 4), cursor); cursor.ptr += descriptor_size;
    create_float_srv(ctx, b, static_cast<UINT>(b->nbytes / 4), cursor); cursor.ptr += descriptor_size;
    create_float_uav(ctx, output, count, cursor);
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            auto ab = a->state, bb = b->state, ob = output->state;
            transition_if_needed(ctx->list.Get(), a->resource.Get(), a->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), b->resource.Get(), b->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            a->state = b->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu; uav_gpu.ptr += descriptor_size * 2;
            ctx->list->SetComputeRootSignature(ctx->binary_root_signature.Get());
            ctx->list->SetPipelineState(ctx->binary_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, 24, constants.data(), 0);
            ctx->list->Dispatch((count + 255) / 256, 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            transition_if_needed(ctx->list.Get(), b->resource.Get(), b->state, bb);
            transition_if_needed(ctx->list.Get(), a->resource.Get(), a->state, ab);
            output->state = ob; b->state = bb; a->state = ab;
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("binary broadcast dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_single_table_float32_into(
    PyObject*,
    PyObject* args,
    HRESULT (*ensure)(DeviceContext*, std::string*),
    ComPtr<ID3D12RootSignature> DeviceContext::*root_member,
    ComPtr<ID3D12PipelineState> DeviceContext::*pso_member,
    UINT srv_count,
    UINT constants_expected,
    const char* label) {
    PyObject* device_capsule = nullptr;
    PyObject* input_capsule = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOOO", &device_capsule, &input_capsule, &output_capsule, &constants_obj)) return nullptr;
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* input = get_buffer(input_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, constants_expected)) return nullptr;
    if (!ctx || !all_owned_by(ctx, {input, output})) {
        PyErr_SetString(PyExc_ValueError, "invalid buffers");
        return nullptr;
    }
    std::string pipeline_error;
    HRESULT hr = ensure(ctx, &pipeline_error);
    if (FAILED(hr)) return raise_hr(label, hr);
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, srv_count + 1, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_single_table_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    create_float_srv(ctx, input, static_cast<UINT>(input->nbytes / 4), cursor);
    cursor.ptr += descriptor_size * srv_count;
    create_float_uav(ctx, output, static_cast<UINT>(output->nbytes / 4), cursor);
    UINT dispatch_count = constants[0];
    if (label && std::strcmp(label, "resize") == 0) dispatch_count = constants[0] * constants[1] * constants[4] * constants[5];
    else if (label && std::strcmp(label, "maxpool") == 0) dispatch_count = constants[0] * constants[1] * constants[4] * constants[5];
    else if (label && std::strcmp(label, "softmax") == 0) dispatch_count = constants[5];
    else if (label && std::strcmp(label, "dfl_project") == 0) dispatch_count = constants[3];
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            auto ib = input->state, ob = output->state;
            transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            input->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu; uav_gpu.ptr += descriptor_size * srv_count;
            ctx->list->SetComputeRootSignature((ctx->*root_member).Get());
            ctx->list->SetPipelineState((ctx->*pso_member).Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, constants_expected, constants.data(), 0);
            ctx->list->Dispatch((dispatch_count + 255) / 256, 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            transition_if_needed(ctx->list.Get(), input->resource.Get(), input->state, ib);
            output->state = ob; input->state = ib;
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("single table dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_dispatch_slice_float32_into(PyObject* self, PyObject* args) {
    return py_dispatch_single_table_float32_into(self, args, ensure_slice_pipeline, &DeviceContext::slice_root_signature, &DeviceContext::slice_pso, 1, 14, "slice");
}

static PyObject* py_dispatch_transpose_float32_into(PyObject* self, PyObject* args) {
    return py_dispatch_single_table_float32_into(self, args, ensure_transpose_pipeline, &DeviceContext::transpose_root_signature, &DeviceContext::transpose_pso, 1, 17, "transpose");
}

static PyObject* py_dispatch_resize_nearest_float32_into(PyObject* self, PyObject* args) {
    return py_dispatch_single_table_float32_into(self, args, ensure_resize_pipeline, &DeviceContext::resize_root_signature, &DeviceContext::resize_pso, 1, 6, "resize");
}

static PyObject* py_dispatch_maxpool2d_float32_into(PyObject* self, PyObject* args) {
    return py_dispatch_single_table_float32_into(self, args, ensure_maxpool_pipeline, &DeviceContext::maxpool_root_signature, &DeviceContext::maxpool_pso, 1, 13, "maxpool");
}

static PyObject* py_dispatch_softmax_axis1_float32_into(PyObject* self, PyObject* args) {
    return py_dispatch_single_table_float32_into(self, args, ensure_softmax_pipeline, &DeviceContext::softmax_root_signature, &DeviceContext::softmax_pso, 1, 6, "softmax");
}

static PyObject* py_dispatch_dfl_project_float32_into(PyObject* self, PyObject* args) {
    return py_dispatch_single_table_float32_into(self, args, ensure_dfl_project_pipeline, &DeviceContext::dfl_root_signature, &DeviceContext::dfl_project_pso, 1, 4, "dfl_project");
}

static PyObject* py_dispatch_concat_float32_into(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* inputs_obj = nullptr;
    PyObject* output_capsule = nullptr;
    PyObject* constants_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OOOO", &device_capsule, &inputs_obj, &output_capsule, &constants_obj)) return nullptr;
    DeviceContext* ctx = get_device(device_capsule);
    BufferHandle* output = get_buffer(output_capsule);
    std::vector<UINT> constants;
    if (!parse_uint_sequence(constants_obj, constants, 18)) return nullptr;
    PyObject* seq = PySequence_Fast(inputs_obj, "inputs must be a sequence");
    if (!seq) return nullptr;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > 8) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "concat supports 1..8 inputs");
        return nullptr;
    }
    std::vector<BufferHandle*> inputs;
    for (Py_ssize_t i = 0; i < n; ++i) {
        inputs.push_back(get_buffer(PySequence_Fast_GET_ITEM(seq, i)));
    }
    Py_DECREF(seq);
    std::vector<BufferHandle*> all = inputs;
    all.push_back(output);
    if (!ctx || !all_owned_by(ctx, all)) {
        PyErr_SetString(PyExc_ValueError, "invalid concat buffers");
        return nullptr;
    }
    std::string pipeline_error;
    HRESULT hr = ensure_concat_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) return raise_hr("ensure_concat_pipeline", hr);
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = create_heap(ctx, 9, heap.ReleaseAndGetAddressOf());
    if (FAILED(hr)) return raise_hr("create_concat_heap", hr);
    UINT descriptor_size = ctx->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cursor = heap->GetCPUDescriptorHandleForHeapStart();
    for (int i = 0; i < 8; ++i) {
        BufferHandle* src = inputs[static_cast<size_t>(i < n ? i : n - 1)];
        create_float_srv(ctx, src, static_cast<UINT>(src->nbytes / 4), cursor);
        cursor.ptr += descriptor_size;
    }
    create_float_uav(ctx, output, static_cast<UINT>(output->nbytes / 4), cursor);
    UINT count = constants[0];
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        bool owns_submission = false;
        hr = begin_or_join_commands(ctx, &owns_submission);
        if (SUCCEEDED(hr)) {
            std::vector<ID3D12Resource*> unique_resources;
            std::vector<D3D12_RESOURCE_STATES> unique_before;
            for (auto* src : inputs) {
                ID3D12Resource* resource = src->resource.Get();
                bool seen = false;
                for (auto* existing : unique_resources) {
                    if (existing == resource) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) {
                    unique_resources.push_back(resource);
                    unique_before.push_back(src->state);
                    transition_if_needed(ctx->list.Get(), resource, src->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
                src->state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            }
            auto ob = output->state;
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            output->state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            ctx->list->SetDescriptorHeaps(1, heaps);
            D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = heap->GetGPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = srv_gpu; uav_gpu.ptr += descriptor_size * 8;
            ctx->list->SetComputeRootSignature(ctx->concat_root_signature.Get());
            ctx->list->SetPipelineState(ctx->concat_pso.Get());
            ctx->list->SetComputeRootDescriptorTable(0, srv_gpu);
            ctx->list->SetComputeRootDescriptorTable(1, uav_gpu);
            ctx->list->SetComputeRoot32BitConstants(2, 18, constants.data(), 0);
            ctx->list->Dispatch((count + 255) / 256, 1, 1);
            transition_if_needed(ctx->list.Get(), output->resource.Get(), output->state, ob);
            output->state = ob;
            for (size_t i = 0; i < inputs.size(); ++i) {
                D3D12_RESOURCE_STATES restored = D3D12_RESOURCE_STATE_COMMON;
                for (size_t r = 0; r < unique_resources.size(); ++r) {
                    if (unique_resources[r] == inputs[i]->resource.Get()) {
                        restored = unique_before[r];
                        break;
                    }
                }
                inputs[i]->state = restored;
            }
            for (size_t r = 0; r < unique_resources.size(); ++r) {
                transition_if_needed(ctx->list.Get(), unique_resources[r], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, unique_before[r]);
            }
            keep_descriptor_heap_alive(ctx, heap);
            hr = finish_if_owned(ctx, owns_submission);
        }
    }
    if (FAILED(hr)) return raise_hr("concat dispatch command submission", hr);
    Py_RETURN_NONE;
}

static PyObject* py_begin_batch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &device_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (ctx->batch_active) {
        PyErr_SetString(PyExc_RuntimeError, "native D3D12 batch recording is already active");
        return nullptr;
    }
    ctx->batch_heaps.clear();
    ctx->batch_resources.clear();
    HRESULT hr = begin_commands(ctx);
    if (FAILED(hr)) {
        return raise_hr("begin_batch", hr);
    }
    ctx->batch_active = true;
    Py_RETURN_NONE;
}

static PyObject* py_begin_prepared_graph(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &device_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (ctx->batch_active || ctx->prepared_recording) {
        PyErr_SetString(PyExc_RuntimeError, "native D3D12 recording scope is already active");
        return nullptr;
    }

    auto* graph = new PreparedGraphHandle();
    graph->owner = ctx;
    Py_INCREF(device_capsule);
    graph->device_capsule = device_capsule;

    HRESULT hr = ctx->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&graph->command_allocator));
    if (FAILED(hr)) {
        Py_DECREF(device_capsule);
        delete graph;
        return raise_hr("CreateCommandAllocator(prepared graph)", hr);
    }
    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        graph->command_allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&graph->command_list));
    if (FAILED(hr)) {
        Py_DECREF(device_capsule);
        delete graph;
        return raise_hr("CreateCommandList(prepared graph)", hr);
    }

    ctx->saved_allocator = ctx->allocator;
    ctx->saved_list = ctx->list;
    ctx->allocator = graph->command_allocator;
    ctx->list = graph->command_list;
    ctx->batch_heaps.clear();
    ctx->batch_resources.clear();
    ctx->batch_active = true;
    ctx->prepared_recording = true;
    ctx->active_prepared_graph = graph;
    Py_RETURN_NONE;
}

static PyObject* py_end_prepared_graph(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &device_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->prepared_recording || !ctx->active_prepared_graph) {
        PyErr_SetString(PyExc_RuntimeError, "native D3D12 prepared graph recording is not active");
        return nullptr;
    }

    PreparedGraphHandle* graph = ctx->active_prepared_graph;
    HRESULT hr = ctx->list->Close();
    graph->descriptor_heaps = std::move(ctx->batch_heaps);
    graph->resources = std::move(ctx->batch_resources);
    ctx->batch_heaps.clear();
    ctx->batch_resources.clear();
    ctx->batch_active = false;
    ctx->prepared_recording = false;
    ctx->active_prepared_graph = nullptr;
    ctx->allocator = ctx->saved_allocator;
    ctx->list = ctx->saved_list;
    ctx->saved_allocator.Reset();
    ctx->saved_list.Reset();
    if (FAILED(hr)) {
        Py_DECREF(graph->device_capsule);
        delete graph;
        return raise_hr("Close(prepared graph)", hr);
    }
    return PyCapsule_New(graph, kPreparedGraphCapsule, destroy_prepared_graph);
}

static PyObject* py_execute_prepared_graph(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    PyObject* graph_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &device_capsule, &graph_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    PreparedGraphHandle* graph = get_prepared_graph(graph_capsule);
    if (!ctx || !graph) {
        return nullptr;
    }
    if (graph->owner != ctx) {
        PyErr_SetString(PyExc_ValueError, "prepared graph belongs to a different device");
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ID3D12CommandList* lists[] = {graph->command_list.Get()};
    ctx->queue->ExecuteCommandLists(1, lists);
    HRESULT hr = signal_and_wait(ctx);
    if (FAILED(hr)) {
        return raise_hr("execute_prepared_graph", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_end_batch(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &device_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    if (!ctx->batch_active) {
        PyErr_SetString(PyExc_RuntimeError, "native D3D12 batch recording is not active");
        return nullptr;
    }
    HRESULT hr = finish_commands(ctx);
    ctx->batch_active = false;
    ctx->batch_heaps.clear();
    ctx->batch_resources.clear();
    if (FAILED(hr)) {
        return raise_hr("end_batch", hr);
    }
    Py_RETURN_NONE;
}

static PyObject* py_synchronize(PyObject*, PyObject* args) {
    PyObject* device_capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &device_capsule)) {
        return nullptr;
    }
    DeviceContext* ctx = get_device(device_capsule);
    if (!ctx) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    HRESULT hr = signal_and_wait(ctx);
    if (FAILED(hr)) {
        return raise_hr("Signal/Wait", hr);
    }
    Py_RETURN_NONE;
}

static PyMethodDef kMethods[] = {
    {"probe_device", py_probe_device, METH_VARARGS, "Probe a hardware D3D12 adapter."},
    {"create_device", py_create_device, METH_VARARGS, "Create an AEXRT native D3D12 device."},
    {"device_info", py_device_info, METH_VARARGS, "Return native D3D12 device information."},
    {"allocate_buffer", py_allocate_buffer, METH_VARARGS, "Allocate a committed default-heap buffer."},
    {"allocate_uav_buffer", py_allocate_uav_buffer, METH_VARARGS, "Allocate a committed default-heap UAV buffer."},
    {"upload_buffer", py_upload_buffer, METH_VARARGS, "Upload bytes through an upload heap into a default buffer."},
    {"upload_buffer_into", py_upload_buffer_into, METH_VARARGS, "Upload bytes into an existing default buffer."},
    {"create_buffer_view", py_create_buffer_view, METH_VARARGS, "Create a zero-copy float32 view into a native buffer."},
    {"download_buffer", py_download_buffer, METH_VARARGS, "Download bytes through a readback heap."},
    {"dispatch_relu_float32", py_dispatch_relu_float32, METH_VARARGS, "Run AEXRT native D3D12 ReLU float32 compute dispatch."},
    {"dispatch_relu_float32_into", py_dispatch_relu_float32_into, METH_VARARGS, "Run ReLU float32 into a persistent output buffer."},
    {"prepare_relu_float32_dispatch", py_prepare_relu_float32_dispatch, METH_VARARGS, "Prepare persistent descriptors for a ReLU float32 dispatch."},
    {"execute_relu_float32_dispatch", py_execute_relu_float32_dispatch, METH_VARARGS, "Execute a prepared ReLU float32 dispatch."},
    {"dispatch_conv2d_silu_float32_into", py_dispatch_conv2d_silu_float32_into, METH_VARARGS, "Run fused Conv2D+SiLU float32 into a persistent output buffer."},
    {"prepare_conv2d_silu_float32_dispatch", py_prepare_conv2d_silu_float32_dispatch, METH_VARARGS, "Prepare a fused Conv2D+SiLU float32 dispatch."},
    {"execute_conv2d_silu_float32_dispatch", py_execute_conv2d_silu_float32_dispatch, METH_VARARGS, "Execute a prepared Conv2D+SiLU float32 dispatch."},
    {"prepare_conv2d_silu_upload_float32_dispatch", py_prepare_conv2d_silu_upload_float32_dispatch, METH_VARARGS, "Prepare persistent upload-ring plus fused Conv2D+SiLU dispatch."},
    {"execute_conv2d_silu_upload_float32_dispatch", py_execute_conv2d_silu_upload_float32_dispatch, METH_VARARGS, "Upload into a persistent ring slot and execute fused Conv2D+SiLU in one queue submission."},
    {"prepare_conv2d_silu_chain_upload_float32_dispatch", py_prepare_conv2d_silu_chain_upload_float32_dispatch, METH_VARARGS, "Prepare a full Conv2D+SiLU chain command list with a persistent upload ring."},
    {"execute_conv2d_silu_chain_upload_float32_dispatch", py_execute_conv2d_silu_chain_upload_float32_dispatch, METH_VARARGS, "Upload into a persistent ring slot and execute a full Conv2D+SiLU chain in one queue submission."},
    {"dispatch_conv2d_float32_into", py_dispatch_conv2d_float32_into, METH_VARARGS, "Run native Conv2D float32 without activation into a persistent output buffer."},
    {"dispatch_concat_conv1x1_float32_into", py_dispatch_concat_conv1x1_float32_into, METH_VARARGS, "Run fused channel-Concat plus 1x1 Conv float32 into a persistent output buffer."},
    {"dispatch_concat_conv1x1_int8_float32_into", py_dispatch_concat_conv1x1_int8_float32_into, METH_VARARGS, "Run fused channel-Concat plus 1x1 Conv with int8 activation/weight quantization."},
    {"dispatch_concat_residual_conv1x1_float32_into", py_dispatch_concat_residual_conv1x1_float32_into, METH_VARARGS, "Run fused channel-Concat plus optional residual Add plus 1x1 Conv float32."},
    {"dispatch_c2f_bottleneck_tiled_float32_into", py_dispatch_c2f_bottleneck_tiled_float32_into, METH_VARARGS, "Run fused C2f 3x3->3x3 residual bottleneck as a tiled superblock."},
    {"dispatch_sppf_tail_float32_into", py_dispatch_sppf_tail_float32_into, METH_VARARGS, "Run fused SPPF MaxPool cascade plus logical concat plus 1x1 Conv."},
    {"dispatch_unary_float32_into", py_dispatch_unary_float32_into, METH_VARARGS, "Run a native unary float32 op into a persistent output buffer."},
    {"dispatch_binary_broadcast_float32_into", py_dispatch_binary_broadcast_float32_into, METH_VARARGS, "Run a native broadcast binary float32 op into a persistent output buffer."},
    {"dispatch_slice_float32_into", py_dispatch_slice_float32_into, METH_VARARGS, "Run a native rank<=4 slice copy into a persistent output buffer."},
    {"dispatch_concat_float32_into", py_dispatch_concat_float32_into, METH_VARARGS, "Run a native rank<=4 concat into a persistent output buffer."},
    {"dispatch_resize_nearest_float32_into", py_dispatch_resize_nearest_float32_into, METH_VARARGS, "Run native NCHW nearest resize into a persistent output buffer."},
    {"dispatch_maxpool2d_float32_into", py_dispatch_maxpool2d_float32_into, METH_VARARGS, "Run native NCHW maxpool2d into a persistent output buffer."},
    {"dispatch_transpose_float32_into", py_dispatch_transpose_float32_into, METH_VARARGS, "Run native rank<=4 transpose into a persistent output buffer."},
    {"dispatch_softmax_axis1_float32_into", py_dispatch_softmax_axis1_float32_into, METH_VARARGS, "Run native rank4 axis-1 softmax into a persistent output buffer."},
    {"dispatch_dfl_project_float32_into", py_dispatch_dfl_project_float32_into, METH_VARARGS, "Run native YOLOv8 DFL 16-bin projection into a persistent output buffer."},
    {"dispatch_yolo_decode_filter_float32", py_dispatch_yolo_decode_filter_float32, METH_VARARGS, "Decode and confidence-filter YOLO output into GPU detection buffers."},
    {"dispatch_yolo_decode_nms_float32", py_dispatch_yolo_decode_nms_float32, METH_VARARGS, "Decode, class-wise NMS, and topK compact YOLO output on GPU."},
    {"dispatch_yolo_head_decode_nms_float32", py_dispatch_yolo_head_decode_nms_float32, METH_VARARGS, "Decode YOLOv8 raw multi-scale head tensors with DFL, class-wise NMS, and topK on GPU."},
    {"begin_batch", py_begin_batch, METH_VARARGS, "Begin one native D3D12 batch command recording scope."},
    {"end_batch", py_end_batch, METH_VARARGS, "Submit and wait for the active native D3D12 batch command recording scope."},
    {"begin_prepared_graph", py_begin_prepared_graph, METH_VARARGS, "Begin recording a reusable native D3D12 prepared graph command list."},
    {"end_prepared_graph", py_end_prepared_graph, METH_VARARGS, "Finish recording and return a reusable native D3D12 prepared graph handle."},
    {"execute_prepared_graph", py_execute_prepared_graph, METH_VARARGS, "Execute a reusable native D3D12 prepared graph command list."},
    {"buffer_info", py_buffer_info, METH_VARARGS, "Return native D3D12 buffer metadata."},
    {"synchronize", py_synchronize, METH_VARARGS, "Wait for the device command queue fence."},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "aexrt_native_d3d12",
    "AEXRT native D3D12 HAL extension. Does not call DirectML.",
    -1,
    kMethods,
};

}  // namespace

extern "C" {

__declspec(dllexport) int aexrt_d3d12_probe() {
    ComPtr<IDXGIAdapter1> adapter;
    return get_adapter_for_device(0, &adapter) ? 1 : 0;
}

__declspec(dllexport) DeviceContext* aexrt_d3d12_create_device(uint32_t adapter_index) {
    ComPtr<IDXGIAdapter1> adapter;
    if (!get_adapter_for_device(adapter_index, &adapter)) {
        return nullptr;
    }

    auto* ctx = new DeviceContext();
    ctx->adapter = adapter;
    ctx->adapter_index = adapter_index;

    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx->device));
    if (FAILED(hr)) {
        delete ctx;
        return nullptr;
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = ctx->device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&ctx->queue));
    if (FAILED(hr)) {
        delete ctx;
        return nullptr;
    }
    hr = ctx->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ctx->allocator));
    if (FAILED(hr)) {
        delete ctx;
        return nullptr;
    }
    hr = ctx->device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        ctx->allocator.Get(),
        nullptr,
        IID_PPV_ARGS(&ctx->list));
    if (FAILED(hr)) {
        delete ctx;
        return nullptr;
    }
    ctx->list->Close();
    hr = ctx->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->fence));
    if (FAILED(hr)) {
        delete ctx;
        return nullptr;
    }
    ctx->fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!ctx->fence_event) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

__declspec(dllexport) void aexrt_d3d12_destroy_device(DeviceContext* ctx) {
    delete ctx;
}

__declspec(dllexport) BufferHandle* aexrt_d3d12_upload_float32(
    DeviceContext* ctx,
    const float* data,
    uint64_t element_count) {
    if (!ctx || !data || element_count == 0) {
        return nullptr;
    }
    const uint64_t nbytes = element_count * sizeof(float);
    auto* buffer = new BufferHandle();
    buffer->owner = ctx;
    buffer->nbytes = nbytes;
    buffer->state = D3D12_RESOURCE_STATE_COMMON;
    buffer->label = "aexrt_c_api_upload";

    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        buffer->state,
        nbytes,
        &buffer->resource);
    if (FAILED(hr)) {
        delete buffer;
        return nullptr;
    }

    ComPtr<ID3D12Resource> upload;
    hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nbytes,
        &upload);
    if (FAILED(hr)) {
        delete buffer;
        return nullptr;
    }
    void* mapped = nullptr;
    D3D12_RANGE read_range{0, 0};
    hr = upload->Map(0, &read_range, &mapped);
    if (FAILED(hr)) {
        delete buffer;
        return nullptr;
    }
    memcpy(mapped, data, static_cast<size_t>(nbytes));
    upload->Unmap(0, nullptr);

    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        hr = begin_commands(ctx);
        if (SUCCEEDED(hr)) {
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, D3D12_RESOURCE_STATE_COPY_DEST);
            buffer->state = D3D12_RESOURCE_STATE_COPY_DEST;
            ctx->list->CopyBufferRegion(buffer->resource.Get(), 0, upload.Get(), 0, nbytes);
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, D3D12_RESOURCE_STATE_COMMON);
            buffer->state = D3D12_RESOURCE_STATE_COMMON;
            hr = finish_commands(ctx);
        }
    }
    if (FAILED(hr)) {
        delete buffer;
        return nullptr;
    }
    return buffer;
}

__declspec(dllexport) BufferHandle* aexrt_d3d12_allocate_float32_uav(
    DeviceContext* ctx,
    uint64_t element_count) {
    if (!ctx || element_count == 0) {
        return nullptr;
    }
    auto* buffer = new BufferHandle();
    buffer->owner = ctx;
    buffer->nbytes = element_count * sizeof(float);
    buffer->state = D3D12_RESOURCE_STATE_COMMON;
    buffer->label = "aexrt_c_api_uav";
    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        buffer->state,
        buffer->nbytes,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        &buffer->resource);
    if (FAILED(hr)) {
        delete buffer;
        return nullptr;
    }
    return buffer;
}

__declspec(dllexport) void aexrt_d3d12_destroy_buffer(BufferHandle* buffer) {
    delete buffer;
}

__declspec(dllexport) int aexrt_d3d12_relu_float32(
    DeviceContext* ctx,
    BufferHandle* input,
    BufferHandle* output,
    uint64_t element_count) {
    if (!ctx || !input || !output) {
        return 0;
    }
    std::string pipeline_error;
    HRESULT hr = ensure_relu_pipeline(ctx, &pipeline_error);
    if (FAILED(hr)) {
        return 0;
    }
    hr = dispatch_relu_into(ctx, input, output, element_count);
    return SUCCEEDED(hr) ? 1 : 0;
}

__declspec(dllexport) int aexrt_d3d12_download_float32(
    DeviceContext* ctx,
    BufferHandle* buffer,
    float* out,
    uint64_t element_count) {
    if (!ctx || !buffer || !out || element_count == 0) {
        return 0;
    }
    const uint64_t nbytes = element_count * sizeof(float);
    if (nbytes > buffer->nbytes) {
        return 0;
    }
    ComPtr<ID3D12Resource> readback;
    HRESULT hr = create_committed_buffer(
        ctx->device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nbytes,
        &readback);
    if (FAILED(hr)) {
        return 0;
    }
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        hr = begin_commands(ctx);
        if (SUCCEEDED(hr)) {
            auto before = buffer->state;
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            buffer->state = D3D12_RESOURCE_STATE_COPY_SOURCE;
            ctx->list->CopyBufferRegion(readback.Get(), 0, buffer->resource.Get(), 0, nbytes);
            transition_if_needed(ctx->list.Get(), buffer->resource.Get(), buffer->state, before);
            buffer->state = before;
            hr = finish_commands(ctx);
        }
    }
    if (FAILED(hr)) {
        return 0;
    }
    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(nbytes)};
    hr = readback->Map(0, &read_range, &mapped);
    if (FAILED(hr)) {
        return 0;
    }
    memcpy(out, mapped, static_cast<size_t>(nbytes));
    D3D12_RANGE write_range{0, 0};
    readback->Unmap(0, &write_range);
    return 1;
}

}  // extern "C"

PyMODINIT_FUNC PyInit_aexrt_native_d3d12(void) {
    PyObject* module = PyModule_Create(&kModule);
    if (!module) {
        return nullptr;
    }
    PyModule_AddStringConstant(module, "api", "aexrt_native_d3d12");
    PyModule_AddStringConstant(module, "architecture", "AEXRT HAL + timeline fence + buffer arena");
    return module;
}
