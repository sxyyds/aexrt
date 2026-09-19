#include "../../native/aexrt.hpp"

#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <shellapi.h>
#include <wrl/client.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

static ComPtr<ID3D11Device> g_d3d_device;
static ComPtr<ID3D11DeviceContext> g_d3d_context;
static ComPtr<IDXGISwapChain> g_swap_chain;
static ComPtr<ID3D11RenderTargetView> g_main_rtv;

struct CapturedFrame {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> bgra;
};

struct Letterbox {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    uint32_t input_size = 0;
};

struct DrawBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    uint32_t class_id = 0;
};

struct PreviewTexture {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    uint32_t width = 0;
    uint32_t height = 0;
};

class DxgiScreenCapture;

struct BrowserState {
    char path[4096]{};
    int adapter_index = 0;
    int device_adapter_index = -1;
    bool loaded = false;
    bool loading = false;
    bool last_source_was_onnx = false;
    bool live_capture = false;
    double load_ms = 0.0;
    double run_ms = 0.0;
    double capture_ms = 0.0;
    double preprocess_ms = 0.0;
    double frame_ms = 0.0;
    double fps = 0.0;
    uint32_t fps_frames = 0;
    std::chrono::high_resolution_clock::time_point fps_start = std::chrono::high_resolution_clock::now();
    uint32_t last_detection_count = 0;
    uint32_t input_size = 0;
    std::string loaded_path;
    std::string status = u8"请选择 .onnx 或 .aexrt 模型。";
    std::unique_ptr<aexrt::Device> device;
    std::unique_ptr<aexrt::YoloModel> model;
    std::unique_ptr<DxgiScreenCapture> capture;
    CapturedFrame frame;
    PreviewTexture preview;
    std::vector<DrawBox> boxes;
};

static void throw_if_failed(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        std::ostringstream oss;
        oss << what << " failed, hr=0x" << std::hex << static_cast<unsigned long>(hr);
        throw std::runtime_error(oss.str());
    }
}

static std::wstring utf8_to_wide(const std::string& text) {
    if (text.empty()) return {};
    int len = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), len);
    return out;
}

static std::string wide_to_utf8(const std::wstring& text) {
    if (text.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), len, nullptr, nullptr);
    return out;
}

static bool ends_with_ci(const std::string& text, const std::string& suffix) {
    if (suffix.size() > text.size()) return false;
    for (size_t i = 0; i < suffix.size(); ++i) {
        char a = static_cast<char>(std::tolower(static_cast<unsigned char>(text[text.size() - suffix.size() + i])));
        char b = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[i])));
        if (a != b) return false;
    }
    return true;
}

static const char* yolo_layout_name(AexrtYoloOutputLayout layout) {
    if (layout == AEXRT_YOLO_OUTPUT_LAYOUT_CHANNELS_LAST) return "channels_last";
    if (layout == AEXRT_YOLO_OUTPUT_LAYOUT_CHANNELS_FIRST) return "channels_first";
    return "unknown";
}

class DxgiScreenCapture {
public:
    explicit DxgiScreenCapture(uint32_t adapter_index, uint32_t target_width = 0, uint32_t target_height = 0)
        : requested_width_(target_width), requested_height_(target_height) {
        init(adapter_index);
    }

    uint32_t width() const { return capture_width_; }
    uint32_t height() const { return capture_height_; }
    uint32_t desktop_width() const { return desktop_width_; }
    uint32_t desktop_height() const { return desktop_height_; }

    bool capture(CapturedFrame& frame, uint32_t timeout_ms = 0) {
        DXGI_OUTDUPL_FRAME_INFO frame_info{};
        ComPtr<IDXGIResource> desktop_resource;
        HRESULT hr = duplication_->AcquireNextFrame(timeout_ms, &frame_info, &desktop_resource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
        if (hr == DXGI_ERROR_ACCESS_LOST) throw std::runtime_error("DXGI desktop duplication access lost");
        throw_if_failed(hr, "AcquireNextFrame");

        ComPtr<ID3D11Texture2D> texture;
        hr = desktop_resource.As(&texture);
        if (FAILED(hr)) {
            duplication_->ReleaseFrame();
            throw_if_failed(hr, "Query frame texture");
        }

        D3D11_TEXTURE2D_DESC desc{};
        texture->GetDesc(&desc);
        update_capture_rect(desc.Width, desc.Height);
        ensure_staging(desc, capture_width_, capture_height_);
        D3D11_BOX source_box{
            capture_x_,
            capture_y_,
            0,
            capture_x_ + capture_width_,
            capture_y_ + capture_height_,
            1
        };
        context_->CopySubresourceRegion(staging_.Get(), 0, 0, 0, 0, texture.Get(), 0, &source_box);

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = context_->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            duplication_->ReleaseFrame();
            throw_if_failed(hr, "Map desktop frame");
        }

        frame.width = capture_width_;
        frame.height = capture_height_;
        frame.bgra.resize(static_cast<size_t>(frame.width) * frame.height * 4);
        const uint8_t* src = static_cast<const uint8_t*>(mapped.pData);
        const uint32_t row_bytes = frame.width * 4;
        for (uint32_t y = 0; y < frame.height; ++y) {
            std::memcpy(frame.bgra.data() + static_cast<size_t>(y) * row_bytes, src + static_cast<size_t>(y) * mapped.RowPitch, row_bytes);
        }
        context_->Unmap(staging_.Get(), 0);
        duplication_->ReleaseFrame();
        return true;
    }

private:
    void init(uint32_t adapter_index) {
        ComPtr<IDXGIFactory1> factory;
        throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");

        ComPtr<IDXGIAdapter1> adapter;
        throw_if_failed(factory->EnumAdapters1(adapter_index, &adapter), "EnumAdapters1");

        D3D_FEATURE_LEVEL feature_level{};
        D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        throw_if_failed(
            D3D11CreateDevice(
                adapter.Get(),
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                flags,
                levels,
                1,
                D3D11_SDK_VERSION,
                &device_,
                &feature_level,
                &context_),
            "D3D11CreateDevice");

        ComPtr<IDXGIOutput> output;
        throw_if_failed(adapter->EnumOutputs(0, &output), "EnumOutputs");
        DXGI_OUTPUT_DESC output_desc{};
        throw_if_failed(output->GetDesc(&output_desc), "GetDesc");
        desktop_width_ = static_cast<uint32_t>(output_desc.DesktopCoordinates.right - output_desc.DesktopCoordinates.left);
        desktop_height_ = static_cast<uint32_t>(output_desc.DesktopCoordinates.bottom - output_desc.DesktopCoordinates.top);
        update_capture_rect(desktop_width_, desktop_height_);

        ComPtr<IDXGIOutput1> output1;
        throw_if_failed(output.As(&output1), "Query IDXGIOutput1");
        throw_if_failed(output1->DuplicateOutput(device_.Get(), &duplication_), "DuplicateOutput");
    }

    void update_capture_rect(uint32_t source_width, uint32_t source_height) {
        desktop_width_ = source_width;
        desktop_height_ = source_height;
        capture_width_ = requested_width_ ? std::min(requested_width_, source_width) : source_width;
        capture_height_ = requested_height_ ? std::min(requested_height_, source_height) : source_height;
        capture_width_ = std::max(1u, capture_width_);
        capture_height_ = std::max(1u, capture_height_);
        capture_x_ = (source_width > capture_width_) ? (source_width - capture_width_) / 2 : 0;
        capture_y_ = (source_height > capture_height_) ? (source_height - capture_height_) / 2 : 0;
    }

    void ensure_staging(const D3D11_TEXTURE2D_DESC& src_desc, uint32_t width, uint32_t height) {
        if (staging_ && staging_width_ == width && staging_height_ == height) return;
        D3D11_TEXTURE2D_DESC desc = src_desc;
        desc.Width = width;
        desc.Height = height;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.Usage = D3D11_USAGE_STAGING;
        throw_if_failed(device_->CreateTexture2D(&desc, nullptr, &staging_), "Create staging texture");
        staging_width_ = width;
        staging_height_ = height;
    }

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D> staging_;
    uint32_t requested_width_ = 0;
    uint32_t requested_height_ = 0;
    uint32_t desktop_width_ = 0;
    uint32_t desktop_height_ = 0;
    uint32_t capture_x_ = 0;
    uint32_t capture_y_ = 0;
    uint32_t capture_width_ = 0;
    uint32_t capture_height_ = 0;
    uint32_t staging_width_ = 0;
    uint32_t staging_height_ = 0;
};

static uint32_t infer_square_input_size(uint64_t input_elements) {
    if (input_elements % 3 != 0) throw std::runtime_error("YOLO input is not 3-channel CHW");
    uint64_t pixels = input_elements / 3;
    uint32_t size = static_cast<uint32_t>(std::sqrt(static_cast<double>(pixels)) + 0.5);
    if (uint64_t(size) * size != pixels) throw std::runtime_error("YOLO input is not square 3xHxW");
    return size;
}

static std::vector<float> preprocess_bgra_letterbox(const CapturedFrame& frame, uint32_t input_size, Letterbox& letterbox) {
    if (frame.width == input_size && frame.height == input_size) {
        letterbox.scale = 1.0f;
        letterbox.pad_x = 0.0f;
        letterbox.pad_y = 0.0f;
        letterbox.input_size = input_size;
        const size_t plane = static_cast<size_t>(input_size) * input_size;
        std::vector<float> input(plane * 3);
        for (uint32_t y = 0; y < input_size; ++y) {
            const uint8_t* row = frame.bgra.data() + static_cast<size_t>(y) * input_size * 4;
            for (uint32_t x = 0; x < input_size; ++x) {
                const size_t idx = static_cast<size_t>(y) * input_size + x;
                const uint8_t* p = row + static_cast<size_t>(x) * 4;
                input[idx] = p[2] / 255.0f;
                input[plane + idx] = p[1] / 255.0f;
                input[plane * 2 + idx] = p[0] / 255.0f;
            }
        }
        return input;
    }
    const float fill = 114.0f / 255.0f;
    std::vector<float> input(static_cast<size_t>(3) * input_size * input_size, fill);
    const float sx = static_cast<float>(input_size) / std::max(1u, frame.width);
    const float sy = static_cast<float>(input_size) / std::max(1u, frame.height);
    letterbox.scale = std::min(sx, sy);
    const uint32_t resized_w = std::max(1u, static_cast<uint32_t>(std::round(frame.width * letterbox.scale)));
    const uint32_t resized_h = std::max(1u, static_cast<uint32_t>(std::round(frame.height * letterbox.scale)));
    letterbox.pad_x = (static_cast<float>(input_size) - resized_w) * 0.5f;
    letterbox.pad_y = (static_cast<float>(input_size) - resized_h) * 0.5f;
    letterbox.input_size = input_size;

    const int pad_x = static_cast<int>(std::round(letterbox.pad_x));
    const int pad_y = static_cast<int>(std::round(letterbox.pad_y));
    const size_t plane = static_cast<size_t>(input_size) * input_size;
    for (uint32_t dy = 0; dy < resized_h; ++dy) {
        float src_y = (static_cast<float>(dy) + 0.5f) / letterbox.scale - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        int y1 = y0 + 1;
        float fy = src_y - y0;
        y0 = std::clamp(y0, 0, static_cast<int>(frame.height) - 1);
        y1 = std::clamp(y1, 0, static_cast<int>(frame.height) - 1);
        int out_y = pad_y + static_cast<int>(dy);
        if (out_y < 0 || out_y >= static_cast<int>(input_size)) continue;

        for (uint32_t dx = 0; dx < resized_w; ++dx) {
            float src_x = (static_cast<float>(dx) + 0.5f) / letterbox.scale - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            int x1 = x0 + 1;
            float fx = src_x - x0;
            x0 = std::clamp(x0, 0, static_cast<int>(frame.width) - 1);
            x1 = std::clamp(x1, 0, static_cast<int>(frame.width) - 1);
            int out_x = pad_x + static_cast<int>(dx);
            if (out_x < 0 || out_x >= static_cast<int>(input_size)) continue;

            const uint8_t* p00 = frame.bgra.data() + (static_cast<size_t>(y0) * frame.width + x0) * 4;
            const uint8_t* p01 = frame.bgra.data() + (static_cast<size_t>(y0) * frame.width + x1) * 4;
            const uint8_t* p10 = frame.bgra.data() + (static_cast<size_t>(y1) * frame.width + x0) * 4;
            const uint8_t* p11 = frame.bgra.data() + (static_cast<size_t>(y1) * frame.width + x1) * 4;
            float wx0 = 1.0f - fx;
            float wy0 = 1.0f - fy;
            float weights[4] = {wx0 * wy0, fx * wy0, wx0 * fy, fx * fy};
            float b = p00[0] * weights[0] + p01[0] * weights[1] + p10[0] * weights[2] + p11[0] * weights[3];
            float g = p00[1] * weights[0] + p01[1] * weights[1] + p10[1] * weights[2] + p11[1] * weights[3];
            float r = p00[2] * weights[0] + p01[2] * weights[1] + p10[2] * weights[2] + p11[2] * weights[3];
            size_t out = static_cast<size_t>(out_y) * input_size + out_x;
            input[out] = r / 255.0f;
            input[plane + out] = g / 255.0f;
            input[plane * 2 + out] = b / 255.0f;
        }
    }
    return input;
}

static std::vector<DrawBox> map_detections_to_source(
    const std::vector<AexrtYoloDetection>& detections,
    const Letterbox& letterbox,
    uint32_t src_w,
    uint32_t src_h) {
    std::vector<DrawBox> boxes;
    boxes.reserve(detections.size());
    for (const auto& det : detections) {
        DrawBox b{};
        b.x1 = (det.x1 - letterbox.pad_x) / letterbox.scale;
        b.y1 = (det.y1 - letterbox.pad_y) / letterbox.scale;
        b.x2 = (det.x2 - letterbox.pad_x) / letterbox.scale;
        b.y2 = (det.y2 - letterbox.pad_y) / letterbox.scale;
        b.x1 = std::clamp(b.x1, 0.0f, static_cast<float>(src_w - 1));
        b.y1 = std::clamp(b.y1, 0.0f, static_cast<float>(src_h - 1));
        b.x2 = std::clamp(b.x2, 0.0f, static_cast<float>(src_w - 1));
        b.y2 = std::clamp(b.y2, 0.0f, static_cast<float>(src_h - 1));
        b.score = det.score;
        b.class_id = static_cast<uint32_t>(std::max(0.0f, det.class_id));
        if (b.x2 > b.x1 && b.y2 > b.y1) boxes.push_back(b);
    }
    return boxes;
}

static void update_preview_texture(PreviewTexture& preview, const CapturedFrame& frame) {
    if (!g_d3d_device || !g_d3d_context || frame.bgra.empty()) return;
    if (!preview.texture || preview.width != frame.width || preview.height != frame.height) {
        preview.texture.Reset();
        preview.srv.Reset();
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = frame.width;
        desc.Height = frame.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        throw_if_failed(g_d3d_device->CreateTexture2D(&desc, nullptr, &preview.texture), "Create preview texture");
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = desc.Format;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;
        throw_if_failed(g_d3d_device->CreateShaderResourceView(preview.texture.Get(), &srv_desc, &preview.srv), "Create preview SRV");
        preview.width = frame.width;
        preview.height = frame.height;
    }
    D3D11_MAPPED_SUBRESOURCE mapped{};
    throw_if_failed(g_d3d_context->Map(preview.texture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped), "Map preview texture");
    const uint32_t row_bytes = frame.width * 4;
    uint8_t* dst = static_cast<uint8_t*>(mapped.pData);
    for (uint32_t y = 0; y < frame.height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * mapped.RowPitch, frame.bgra.data() + static_cast<size_t>(y) * row_bytes, row_bytes);
    }
    g_d3d_context->Unmap(preview.texture.Get(), 0);
}

static void create_render_target() {
    ComPtr<ID3D11Texture2D> back_buffer;
    g_swap_chain->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
    g_d3d_device->CreateRenderTargetView(back_buffer.Get(), nullptr, &g_main_rtv);
}

static void cleanup_render_target() {
    g_main_rtv.Reset();
}

static bool create_device_d3d(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL feature_level{};
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        levels,
        2,
        D3D11_SDK_VERSION,
        &sd,
        &g_swap_chain,
        &g_d3d_device,
        &feature_level,
        &g_d3d_context);
    if (FAILED(hr)) return false;
    create_render_target();
    return true;
}

static void cleanup_device_d3d() {
    cleanup_render_target();
    g_swap_chain.Reset();
    g_d3d_context.Reset();
    g_d3d_device.Reset();
}

static LRESULT WINAPI wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) return true;
    switch (msg) {
        case WM_SIZE:
            if (wparam != SIZE_MINIMIZED && g_swap_chain) {
                cleanup_render_target();
                g_swap_chain->ResizeBuffers(0, LOWORD(lparam), HIWORD(lparam), DXGI_FORMAT_UNKNOWN, 0);
                create_render_target();
            }
            return 0;
        case WM_SYSCOMMAND:
            if ((wparam & 0xfff0) == SC_KEYMENU) return 0;
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

static bool open_model_dialog(HWND owner, char* out, size_t out_size) {
    wchar_t file_name[4096]{};
    std::wstring current = utf8_to_wide(out);
    if (!current.empty()) {
        wcsncpy_s(file_name, current.c_str(), _TRUNCATE);
    }
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFile = file_name;
    ofn.nMaxFile = static_cast<DWORD>(_countof(file_name));
    ofn.lpstrFilter = L"YOLO ONNX / AEXRT Engine\0*.onnx;*.aexrt\0ONNX 模型\0*.onnx\0AEXRT Engine\0*.aexrt\0所有文件\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameW(&ofn)) return false;
    std::string utf8 = wide_to_utf8(file_name);
    strncpy_s(out, out_size, utf8.c_str(), _TRUNCATE);
    return true;
}

static void ensure_aexrt_device(BrowserState& state) {
    if (!state.device || state.device_adapter_index != state.adapter_index) {
        state.capture.reset();
        state.preview = PreviewTexture{};
        state.live_capture = false;
        state.model.reset();
        state.loaded = false;
        state.device = std::make_unique<aexrt::Device>(static_cast<uint32_t>(std::max(0, state.adapter_index)));
        state.device_adapter_index = state.adapter_index;
    }
}

static void load_model(BrowserState& state) {
    const std::string path = state.path;
    if (path.empty()) {
        state.status = u8"请先选择模型文件。";
        return;
    }
    if (!std::filesystem::exists(std::filesystem::u8path(path))) {
        state.status = u8"文件不存在。";
        return;
    }
    state.loading = true;
    state.loaded = false;
    state.model.reset();
    try {
        ensure_aexrt_device(state);
        const bool is_onnx = ends_with_ci(path, ".onnx");
        auto t0 = std::chrono::high_resolution_clock::now();
        aexrt::YoloModel compiled = is_onnx
            ? aexrt::compile_yolo_from_onnx(*state.device, path)
            : aexrt::load_engine(*state.device, path);
        auto t1 = std::chrono::high_resolution_clock::now();
        state.load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        state.model = std::make_unique<aexrt::YoloModel>(std::move(compiled));
        state.loaded = state.model->executable();
        state.loaded_path = path;
        state.last_source_was_onnx = is_onnx;
        state.input_size = state.loaded ? infer_square_input_size(state.model->input_element_count()) : 0;
        state.status = state.loaded ? u8"模型已加载，C++ runtime 图已准备好。" : u8"模型加载了，但当前图不可执行。";
    } catch (const std::exception& ex) {
        state.status = std::string(u8"加载失败：") + ex.what();
        state.model.reset();
        state.loaded = false;
    }
    state.loading = false;
}

static void stop_live_capture(BrowserState& state) {
    state.live_capture = false;
    state.capture.reset();
}

static void start_live_capture(BrowserState& state) {
    if (!state.device || !state.model || !state.loaded) {
        state.status = u8"请先加载可执行模型。";
        return;
    }
    try {
        const uint32_t target = std::max(1u, state.input_size);
        state.capture = std::make_unique<DxgiScreenCapture>(static_cast<uint32_t>(std::max(0, state.adapter_index)), target, target);
        state.live_capture = true;
        state.fps = 0.0;
        state.fps_frames = 0;
        state.fps_start = std::chrono::high_resolution_clock::now();
        state.status = u8"DXGI 实时预览已启动，采集尺寸跟随模型输入。";
    } catch (const std::exception& ex) {
        state.live_capture = false;
        state.capture.reset();
        state.status = std::string(u8"DXGI 启动失败：") + ex.what();
    }
}

static void run_zero_input(BrowserState& state) {
    if (!state.device || !state.model || !state.loaded) {
        state.status = u8"请先加载可执行模型。";
        return;
    }
    try {
        std::vector<float> input(static_cast<size_t>(state.model->input_element_count()), 0.0f);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto detections = aexrt::run_yolo(*state.device, *state.model, input, 64);
        auto t1 = std::chrono::high_resolution_clock::now();
        state.run_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        state.last_detection_count = static_cast<uint32_t>(detections.size());
        state.status = u8"已完成一次 C++ 推理调用。";
    } catch (const std::exception& ex) {
        state.status = std::string(u8"推理失败：") + ex.what();
    }
}

static void process_live_frame(BrowserState& state) {
    if (!state.live_capture || !state.capture || !state.device || !state.model || !state.loaded) return;
    try {
        const auto frame_start = std::chrono::high_resolution_clock::now();

        const auto cap0 = std::chrono::high_resolution_clock::now();
        bool got = state.capture->capture(state.frame, 0);
        const auto cap1 = std::chrono::high_resolution_clock::now();
        if (!got) return;
        state.capture_ms = std::chrono::duration<double, std::milli>(cap1 - cap0).count();
        update_preview_texture(state.preview, state.frame);

        Letterbox letterbox{};
        const auto prep0 = std::chrono::high_resolution_clock::now();
        std::vector<float> input = preprocess_bgra_letterbox(state.frame, state.input_size, letterbox);
        const auto prep1 = std::chrono::high_resolution_clock::now();
        state.preprocess_ms = std::chrono::duration<double, std::milli>(prep1 - prep0).count();

        const auto inf0 = std::chrono::high_resolution_clock::now();
        auto detections = aexrt::run_yolo(*state.device, *state.model, input, 64);
        const auto inf1 = std::chrono::high_resolution_clock::now();
        state.run_ms = std::chrono::duration<double, std::milli>(inf1 - inf0).count();
        state.last_detection_count = static_cast<uint32_t>(detections.size());
        state.boxes = map_detections_to_source(detections, letterbox, state.frame.width, state.frame.height);

        const auto frame_end = std::chrono::high_resolution_clock::now();
        state.frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        state.fps_frames += 1;
        double elapsed = std::chrono::duration<double>(frame_end - state.fps_start).count();
        if (elapsed >= 0.5) {
            state.fps = state.fps_frames / elapsed;
            state.fps_frames = 0;
            state.fps_start = frame_end;
        }
    } catch (const std::exception& ex) {
        stop_live_capture(state);
        state.status = std::string(u8"实时预览停止：") + ex.what();
    }
}

static void write_text_report(const std::string& path, const std::string& text) {
    if (path.empty()) return;
    std::filesystem::path report_path = std::filesystem::u8path(path);
    if (report_path.has_parent_path()) {
        std::filesystem::create_directories(report_path.parent_path());
    }
    std::ofstream out(report_path, std::ios::binary | std::ios::trunc);
    out << text;
}

static int run_live_benchmark(BrowserState& state, uint32_t frame_count, const std::string& report_path) {
    std::ostringstream report;
    auto finish = [&](int code) -> int {
        const std::string text = report.str();
        if (!text.empty()) {
            std::printf("%s", text.c_str());
            std::fflush(stdout);
        }
        write_text_report(report_path, text);
        return code;
    };

    try {
        load_model(state);
        report << "status=" << state.status << "\n";
        if (!state.loaded || !state.model || !state.device) return finish(2);

        const uint32_t target = std::max(1u, state.input_size);
        DxgiScreenCapture capture(static_cast<uint32_t>(std::max(0, state.adapter_index)), target, target);
        report << "live_test source=" << state.path
               << " model_input=" << target << "x" << target
               << " capture=" << capture.width() << "x" << capture.height()
               << " desktop=" << capture.desktop_width() << "x" << capture.desktop_height()
               << " frames=" << frame_count << "\n";
        report << "graph classes=" << state.model->class_count()
               << " channels=" << state.model->channel_count()
               << " anchors=" << state.model->anchor_count()
               << " layout=" << yolo_layout_name(state.model->output_layout())
               << " objectness=" << (state.model->has_objectness() ? 1 : 0)
               << " commands=" << state.model->prepared_command_count()
               << " supported=" << state.model->supported_prepared_command_count()
               << " unsupported=" << state.model->unsupported_prepared_command_count()
               << " load_ms=" << std::fixed << std::setprecision(3) << state.load_ms << "\n";

        std::vector<float> warmup(static_cast<size_t>(state.model->input_element_count()), 0.0f);
        (void)aexrt::run_yolo(*state.device, *state.model, warmup, 1);
        report << "optimized skipped=" << state.model->prepared_skipped_command_count()
               << " late_concat_conv1x1=" << state.model->late_concat_conv1x1_fusion_count()
               << " concat_residual_cv2=" << state.model->concat_residual_conv1x1_fusion_count()
               << " c2f_bottleneck=" << state.model->c2f_bottleneck_superblock_count()
               << " c2f_tail=" << state.model->c2f_tail_residual_fusion_count()
               << " head_fusion=" << (state.model->head_fusion_enabled() ? 1 : 0)
               << " head_final_conv=" << (state.model->head_final_conv_fusion_enabled() ? 1 : 0)
               << "\n";
        report << "tileflow conv1x1=" << state.model->tileflow_conv1x1_count()
               << " spatial3x3=" << state.model->tileflow_3x3_spatial_count()
               << " pack4_3x3=" << state.model->tileflow_3x3_pack4_count()
               << " pack8_3x3=" << state.model->tileflow_3x3_pack8_count()
               << " implicit_gemm_3x3=" << state.model->tileflow_3x3_implicit_gemm_count()
               << " shape40=" << state.model->tileflow_3x3_implicit_gemm_40x40_count()
               << " shape20=" << state.model->tileflow_3x3_implicit_gemm_20x20_count()
               << " shape10=" << state.model->tileflow_3x3_implicit_gemm_10x10_count()
               << "\n";

        double capture_total = 0.0;
        double preprocess_total = 0.0;
        double infer_total = 0.0;
        double frame_total = 0.0;
        uint32_t frames = 0;
        uint32_t timeouts = 0;
        uint32_t detections = 0;
        const uint32_t max_attempts = frame_count * 20 + 100;

        CapturedFrame frame;
        for (uint32_t attempts = 0; frames < frame_count && attempts < max_attempts; ++attempts) {
            const auto frame_start = std::chrono::high_resolution_clock::now();

            const auto cap0 = std::chrono::high_resolution_clock::now();
            bool got = capture.capture(frame, 100);
            const auto cap1 = std::chrono::high_resolution_clock::now();
            if (!got) {
                ++timeouts;
                continue;
            }

            Letterbox letterbox{};
            const auto prep0 = std::chrono::high_resolution_clock::now();
            std::vector<float> input = preprocess_bgra_letterbox(frame, target, letterbox);
            const auto prep1 = std::chrono::high_resolution_clock::now();

            const auto inf0 = std::chrono::high_resolution_clock::now();
            auto boxes = aexrt::run_yolo(*state.device, *state.model, input, 64);
            const auto inf1 = std::chrono::high_resolution_clock::now();

            const auto frame_end = std::chrono::high_resolution_clock::now();
            capture_total += std::chrono::duration<double, std::milli>(cap1 - cap0).count();
            preprocess_total += std::chrono::duration<double, std::milli>(prep1 - prep0).count();
            infer_total += std::chrono::duration<double, std::milli>(inf1 - inf0).count();
            frame_total += std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            detections = static_cast<uint32_t>(boxes.size());
            ++frames;
        }

        if (frames == 0) {
            report << "live_test no frames captured, timeouts=" << timeouts << "\n";
            return finish(4);
        }

        const double inv = 1.0 / frames;
        const double avg_frame = frame_total * inv;
        const double fps = avg_frame > 0.0 ? 1000.0 / avg_frame : 0.0;
        report << "avg capture=" << capture_total * inv
               << " ms preprocess=" << preprocess_total * inv
               << " ms infer=" << infer_total * inv
               << " ms frame=" << avg_frame
               << " ms fps=" << std::setprecision(1) << fps
               << " frames=" << frames
               << " timeouts=" << timeouts
               << " last_det=" << detections << "\n";
        return finish(0);
    } catch (const std::exception& ex) {
        report << "error=" << ex.what() << "\n";
        return finish(10);
    }
}

static int run_infer_benchmark(BrowserState& state, uint32_t run_count, const std::string& report_path) {
    std::ostringstream report;
    auto finish = [&](int code) -> int {
        const std::string text = report.str();
        if (!text.empty()) {
            std::printf("%s", text.c_str());
            std::fflush(stdout);
        }
        write_text_report(report_path, text);
        return code;
    };
    try {
        load_model(state);
        report << "status=" << state.status << "\n";
        if (!state.loaded || !state.model || !state.device) return finish(2);
        std::vector<float> input(static_cast<size_t>(state.model->input_element_count()), 0.0f);
        (void)aexrt::run_yolo(*state.device, *state.model, input, 1);
        report << "infer_test source=" << state.path
               << " runs=" << run_count
               << " input_elements=" << state.model->input_element_count()
               << " classes=" << state.model->class_count()
               << " channels=" << state.model->channel_count()
               << " anchors=" << state.model->anchor_count()
               << " layout=" << yolo_layout_name(state.model->output_layout())
               << " objectness=" << (state.model->has_objectness() ? 1 : 0)
               << " commands=" << state.model->prepared_command_count()
               << " skipped=" << state.model->prepared_skipped_command_count()
               << " late_concat_conv1x1=" << state.model->late_concat_conv1x1_fusion_count()
               << " concat_residual_cv2=" << state.model->concat_residual_conv1x1_fusion_count()
               << " c2f_bottleneck=" << state.model->c2f_bottleneck_superblock_count()
               << " c2f_tail=" << state.model->c2f_tail_residual_fusion_count()
               << " head_fusion=" << (state.model->head_fusion_enabled() ? 1 : 0)
               << " head_final_conv=" << (state.model->head_final_conv_fusion_enabled() ? 1 : 0)
               << "\n";
        report << "tileflow conv1x1=" << state.model->tileflow_conv1x1_count()
               << " spatial3x3=" << state.model->tileflow_3x3_spatial_count()
               << " pack4_3x3=" << state.model->tileflow_3x3_pack4_count()
               << " pack8_3x3=" << state.model->tileflow_3x3_pack8_count()
               << " implicit_gemm_3x3=" << state.model->tileflow_3x3_implicit_gemm_count()
               << " shape40=" << state.model->tileflow_3x3_implicit_gemm_40x40_count()
               << " shape20=" << state.model->tileflow_3x3_implicit_gemm_20x20_count()
               << " shape10=" << state.model->tileflow_3x3_implicit_gemm_10x10_count()
               << "\n";

        double total = 0.0;
        uint32_t detections = 0;
        for (uint32_t i = 0; i < run_count; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto boxes = aexrt::run_yolo(*state.device, *state.model, input, 64);
            const auto t1 = std::chrono::high_resolution_clock::now();
            total += std::chrono::duration<double, std::milli>(t1 - t0).count();
            detections = static_cast<uint32_t>(boxes.size());
        }
        const double avg = run_count ? total / run_count : 0.0;
        report << "avg_infer=" << std::fixed << std::setprecision(3) << avg
               << " ms fps=" << (avg > 0.0 ? 1000.0 / avg : 0.0)
               << " runs=" << run_count
               << " last_det=" << detections << "\n";
        return finish(0);
    } catch (const std::exception& ex) {
        report << "error=" << ex.what() << "\n";
        return finish(10);
    }
}

static void add_chinese_font() {
    ImGuiIO& io = ImGui::GetIO();
    const char* candidates[] = {
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
    };
    for (const char* path : candidates) {
        if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) {
            io.Fonts->AddFontFromFileTTF(path, 18.0f, nullptr, io.Fonts->GetGlyphRangesChineseFull());
            return;
        }
    }
    io.Fonts->AddFontDefault();
}

static void draw_browser_ui(HWND hwnd, BrowserState& state) {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin(u8"AEXRT 原生 YOLO 模型浏览器", nullptr, flags);

    ImGui::TextUnformatted(u8"AEXRT 原生 C++ / D3D12 YOLO 模型浏览器");
    ImGui::Separator();

    ImGui::SetNextItemWidth(-130.0f);
    ImGui::InputText(u8"模型路径", state.path, sizeof(state.path));
    ImGui::SameLine();
    if (ImGui::Button(u8"浏览...")) {
        open_model_dialog(hwnd, state.path, sizeof(state.path));
    }

    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt(u8"适配器", &state.adapter_index);
    if (state.adapter_index < 0) state.adapter_index = 0;
    ImGui::SameLine();
    if (ImGui::Button(u8"加载模型")) {
        load_model(state);
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"推理热身")) {
        run_zero_input(state);
    }

    ImGui::Spacing();
    ImGui::TextWrapped(u8"状态：%s", state.status.c_str());
    ImGui::Separator();

    if (state.model) {
        ImGui::Text(u8"来源：%s", state.last_source_was_onnx ? u8"ONNX 直读" : u8"AEXRT Package");
        ImGui::TextWrapped(u8"已加载：%s", state.loaded_path.c_str());
        ImGui::Text(u8"加载耗时：%.3f ms", state.load_ms);
        ImGui::Text(u8"输入元素：%llu", static_cast<unsigned long long>(state.model->input_element_count()));
        ImGui::Text(u8"类别数：%u", state.model->class_count());
        ImGui::Text("Channels: %u", state.model->channel_count());
        ImGui::Text(u8"Anchor 数：%u", state.model->anchor_count());
        ImGui::Text("YOLO layout: %s", yolo_layout_name(state.model->output_layout()));
        ImGui::Text("Objectness: %s", state.model->has_objectness() ? "yes" : "no");
        ImGui::Text(u8"ONNX/图节点：%u", state.model->graph_node_count());
        ImGui::Text(u8"Replay Value：%u", state.model->graph_value_count());
        ImGui::Text(u8"常量：%u", state.model->constant_count());
        ImGui::Text(u8"Prepared Command：%u", state.model->prepared_command_count());
        ImGui::Text(u8"支持 / 不支持：%u / %u", state.model->supported_prepared_command_count(), state.model->unsupported_prepared_command_count());
        ImGui::Text("Replay optimized: skipped %u, late concat-conv %u, concat-residual %u, C2f %u/%u, head %s",
            state.model->prepared_skipped_command_count(),
            state.model->late_concat_conv1x1_fusion_count(),
            state.model->concat_residual_conv1x1_fusion_count(),
            state.model->c2f_bottleneck_superblock_count(),
            state.model->c2f_tail_residual_fusion_count(),
            state.model->head_fusion_enabled() ? "ON" : "OFF");
        ImGui::Text("TileFlow: 1x1 %u, 3x3 tile %u, pack4 %u, pack8 %u, igemm %u (40 %u/20 %u/10 %u)",
            state.model->tileflow_conv1x1_count(),
            state.model->tileflow_3x3_spatial_count(),
            state.model->tileflow_3x3_pack4_count(),
            state.model->tileflow_3x3_pack8_count(),
            state.model->tileflow_3x3_implicit_gemm_count(),
            state.model->tileflow_3x3_implicit_gemm_40x40_count(),
            state.model->tileflow_3x3_implicit_gemm_20x20_count(),
            state.model->tileflow_3x3_implicit_gemm_10x10_count());
        ImGui::Text(u8"可执行：%s", state.model->executable() ? u8"是" : u8"否");
        if (state.run_ms > 0.0) {
            ImGui::Text(u8"上次推理：%.3f ms，检测数：%u", state.run_ms, state.last_detection_count);
        }
    } else {
        ImGui::TextUnformatted(u8"尚未加载模型。");
    }

    ImGui::End();
}

static void draw_browser_ui_live(HWND hwnd, BrowserState& state) {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin(u8"AEXRT 原生 YOLO 实时预览", nullptr, flags);

    ImGui::TextUnformatted(u8"AEXRT 原生 C++ / D3D12 YOLO 实时预览");
    ImGui::Separator();

    ImGui::SetNextItemWidth(-130.0f);
    ImGui::InputText(u8"模型路径", state.path, sizeof(state.path));
    ImGui::SameLine();
    if (ImGui::Button(u8"浏览...")) {
        open_model_dialog(hwnd, state.path, sizeof(state.path));
    }

    ImGui::SetNextItemWidth(110.0f);
    ImGui::InputInt(u8"适配器", &state.adapter_index);
    if (state.adapter_index < 0) state.adapter_index = 0;
    ImGui::SameLine();
    if (ImGui::Button(u8"加载模型")) {
        load_model(state);
    }
    ImGui::SameLine();
    if (ImGui::Button(u8"推理热身")) {
        run_zero_input(state);
    }
    ImGui::SameLine();
    if (!state.live_capture) {
        if (ImGui::Button(u8"启动DXGI预览")) start_live_capture(state);
    } else {
        if (ImGui::Button(u8"停止预览")) stop_live_capture(state);
    }

    ImGui::TextWrapped(u8"状态：%s", state.status.c_str());
    ImGui::Separator();

    const float side_width = 330.0f;
    ImGui::BeginChild("metrics", ImVec2(side_width, 0.0f), true);
    ImGui::TextUnformatted(u8"实时指标");
    ImGui::Separator();
    ImGui::Text(u8"FPS：%.1f", state.fps);
    ImGui::Text(u8"推理耗时：%.3f ms", state.run_ms);
    ImGui::Text(u8"截图耗时：%.3f ms", state.capture_ms);
    ImGui::Text(u8"预处理耗时：%.3f ms", state.preprocess_ms);
    ImGui::Text(u8"整帧耗时：%.3f ms", state.frame_ms);
    ImGui::Text(u8"检测框：%u", state.last_detection_count);
    if (state.frame.width && state.frame.height) {
        ImGui::Text(u8"截图尺寸：%ux%u", state.frame.width, state.frame.height);
    }
    if (state.capture) {
        ImGui::Text(u8"桌面尺寸：%ux%u", state.capture->desktop_width(), state.capture->desktop_height());
    }

    ImGui::Spacing();
    ImGui::TextUnformatted(u8"模型");
    ImGui::Separator();
    if (state.model) {
        ImGui::Text(u8"来源：%s", state.last_source_was_onnx ? u8"ONNX直读" : u8"AEXRT包");
        ImGui::Text(u8"输入：%ux%u", state.input_size, state.input_size);
        ImGui::Text(u8"类别：%u", state.model->class_count());
        ImGui::Text("Channels: %u", state.model->channel_count());
        ImGui::Text(u8"Anchors：%u", state.model->anchor_count());
        ImGui::Text("Layout: %s", yolo_layout_name(state.model->output_layout()));
        ImGui::Text("Objectness: %s", state.model->has_objectness() ? "yes" : "no");
        ImGui::Text(u8"Commands：%u", state.model->prepared_command_count());
        ImGui::Text(u8"Unsupported：%u", state.model->unsupported_prepared_command_count());
        ImGui::Text("Skipped: %u  LateConcatConv: %u  ConcatResidual: %u",
            state.model->prepared_skipped_command_count(),
            state.model->late_concat_conv1x1_fusion_count(),
            state.model->concat_residual_conv1x1_fusion_count());
        ImGui::Text("TileFlow: 1x1 %u  3x3 pack8 %u  igemm %u (%u/%u)",
            state.model->tileflow_conv1x1_count(),
            state.model->tileflow_3x3_pack8_count(),
            state.model->tileflow_3x3_implicit_gemm_count(),
            state.model->tileflow_3x3_implicit_gemm_20x20_count(),
            state.model->tileflow_3x3_implicit_gemm_10x10_count());
        ImGui::Text(u8"加载耗时：%.3f ms", state.load_ms);
    } else {
        ImGui::TextUnformatted(u8"尚未加载模型。");
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("preview", ImVec2(0.0f, 0.0f), true);
    ImGui::TextUnformatted(u8"DXGI 截图预览");
    ImGui::Separator();
    if (state.preview.srv && state.preview.width && state.preview.height) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float scale = std::min(avail.x / static_cast<float>(state.preview.width), avail.y / static_cast<float>(state.preview.height));
        if (!(scale > 0.0f)) scale = 1.0f;
        ImVec2 image_size(static_cast<float>(state.preview.width) * scale, static_cast<float>(state.preview.height) * scale);
        ImGui::Image(reinterpret_cast<ImTextureID>(state.preview.srv.Get()), image_size);
        ImVec2 min = ImGui::GetItemRectMin();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        for (const auto& box : state.boxes) {
            ImVec2 p1(min.x + box.x1 * scale, min.y + box.y1 * scale);
            ImVec2 p2(min.x + box.x2 * scale, min.y + box.y2 * scale);
            draw->AddRect(p1, p2, IM_COL32(0, 255, 90, 255), 0.0f, 0, 2.0f);
            char label[64]{};
            std::snprintf(label, sizeof(label), "cls %u %.2f", box.class_id, box.score);
            draw->AddText(ImVec2(p1.x + 3.0f, std::max(min.y, p1.y - 18.0f)), IM_COL32(0, 255, 90, 255), label);
        }
    } else {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImGui::Dummy(ImVec2(std::max(1.0f, avail.x), std::max(240.0f, avail.y)));
        ImVec2 min = ImGui::GetItemRectMin();
        ImVec2 max = ImGui::GetItemRectMax();
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(min, max, IM_COL32(18, 20, 24, 255));
        draw->AddText(ImVec2(min.x + 18.0f, min.y + 18.0f), IM_COL32(190, 200, 210, 255), "DXGI preview not running");
    }
    ImGui::EndChild();
    ImGui::End();
}

int main(int argc, char** argv) {
    SetProcessDPIAware();
    BrowserState state;
    const char* default_model = "models/cs2V8_320.onnx";
    bool self_test = false;
    uint32_t live_test_frames = 0;
    uint32_t infer_test_runs = 0;
    std::string live_test_report = "build\\native\\imgui_live_test_report.txt";
    const char* cli_model = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--self-test") == 0) {
            self_test = true;
        } else if (std::strcmp(argv[i], "--live-test") == 0) {
            live_test_frames = 120;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                live_test_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
            }
        } else if (std::strcmp(argv[i], "--infer-test") == 0) {
            infer_test_runs = 100;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                infer_test_runs = static_cast<uint32_t>(std::stoul(argv[++i]));
            }
        } else if (std::strcmp(argv[i], "--adapter") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --adapter\n");
                return 1;
            }
            state.adapter_index = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--report") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --report\n");
                return 1;
            }
            live_test_report = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("native_imgui_yolo_browser.exe [model.onnx|model.aexrt] [--self-test] [--live-test N] [--infer-test N] [--adapter N] [--report PATH]\n");
            return 0;
        } else {
            cli_model = argv[i];
        }
    }
    if (cli_model) {
        strncpy_s(state.path, cli_model, _TRUNCATE);
    } else if (std::filesystem::exists(std::filesystem::u8path(default_model))) {
        strncpy_s(state.path, default_model, _TRUNCATE);
    }
    if (live_test_frames != 0) {
        return run_live_benchmark(state, live_test_frames, live_test_report);
    }
    if (infer_test_runs != 0) {
        return run_infer_benchmark(state, infer_test_runs, live_test_report);
    }
    if (self_test) {
        load_model(state);
        std::printf("status=%s\n", state.status.c_str());
        if (!state.loaded || !state.model) return 2;
        run_zero_input(state);
        std::printf("source=%s\n", state.last_source_was_onnx ? "onnx" : "package");
        std::printf("input=%llu classes=%u channels=%u anchors=%u layout=%s objectness=%u commands=%u unsupported=%u load_ms=%.3f run_ms=%.3f detections=%u\n",
            static_cast<unsigned long long>(state.model->input_element_count()),
            state.model->class_count(),
            state.model->channel_count(),
            state.model->anchor_count(),
            yolo_layout_name(state.model->output_layout()),
            state.model->has_objectness() ? 1u : 0u,
            state.model->prepared_command_count(),
            state.model->unsupported_prepared_command_count(),
            state.load_ms,
            state.run_ms,
            state.last_detection_count);
        std::printf("optimized skipped=%u late_concat_conv1x1=%u concat_residual_cv2=%u c2f_bottleneck=%u c2f_tail=%u head_fusion=%u head_final_conv=%u\n",
            state.model->prepared_skipped_command_count(),
            state.model->late_concat_conv1x1_fusion_count(),
            state.model->concat_residual_conv1x1_fusion_count(),
            state.model->c2f_bottleneck_superblock_count(),
            state.model->c2f_tail_residual_fusion_count(),
            state.model->head_fusion_enabled() ? 1u : 0u,
            state.model->head_final_conv_fusion_enabled() ? 1u : 0u);
        std::printf("tileflow conv1x1=%u spatial3x3=%u pack4_3x3=%u pack8_3x3=%u implicit_gemm_3x3=%u shape40=%u shape20=%u shape10=%u\n",
            state.model->tileflow_conv1x1_count(),
            state.model->tileflow_3x3_spatial_count(),
            state.model->tileflow_3x3_pack4_count(),
            state.model->tileflow_3x3_pack8_count(),
            state.model->tileflow_3x3_implicit_gemm_count(),
            state.model->tileflow_3x3_implicit_gemm_40x40_count(),
            state.model->tileflow_3x3_implicit_gemm_20x20_count(),
            state.model->tileflow_3x3_implicit_gemm_10x10_count());
        return state.loaded ? 0 : 3;
    }

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_CLASSDC;
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"AEXRTNativeYoloBrowser";
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowW(
        wc.lpszClassName,
        L"AEXRT 原生 YOLO 模型浏览器",
        WS_OVERLAPPEDWINDOW,
        100,
        100,
        1180,
        720,
        nullptr,
        nullptr,
        wc.hInstance,
        nullptr);

    if (!create_device_d3d(hwnd)) {
        cleanup_device_d3d();
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    add_chinese_font();
    ImGui::StyleColorsDark();
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_d3d_device.Get(), g_d3d_context.Get());

    bool done = false;
    while (!done) {
        MSG msg{};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        process_live_frame(state);
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        draw_browser_ui_live(hwnd, state);
        ImGui::Render();

        const float clear_color[4] = {0.08f, 0.09f, 0.10f, 1.0f};
        ID3D11RenderTargetView* rtv = g_main_rtv.Get();
        g_d3d_context->OMSetRenderTargets(1, &rtv, nullptr);
        g_d3d_context->ClearRenderTargetView(g_main_rtv.Get(), clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_swap_chain->Present(1, 0);
    }

    state.model.reset();
    state.device.reset();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    cleanup_device_d3d();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}
