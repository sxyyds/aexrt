#include "../../native/aexrt.hpp"

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

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

struct Options {
    std::string package_path = "examples\\cs2V8_320.aexrt";
    std::string onnx_path;
    uint32_t adapter_index = 0;
    uint32_t max_detections = 64;
    uint32_t frame_limit = 0;
};

static void throw_if_failed(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        std::ostringstream oss;
        oss << what << " failed, hr=0x" << std::hex << static_cast<unsigned long>(hr);
        throw std::runtime_error(oss.str());
    }
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

static std::string path_to_utf8ish(const std::filesystem::path& path) {
    return path.string();
}

static std::string resolve_package_path(const Options& options) {
    namespace fs = std::filesystem;
    if (!options.onnx_path.empty()) {
        fs::path onnx(options.onnx_path);
        std::string stem = onnx.stem().string();
        std::vector<fs::path> candidates = {
            onnx.parent_path() / (stem + ".aexrt"),
            fs::path("examples") / (stem + ".aexrt"),
            fs::path("examples\\cs2V8_320.aexrt"),
        };
        for (const auto& candidate : candidates) {
            if (!candidate.empty() && fs::exists(candidate)) {
                std::cout << "using compiled AEXRT engine: " << path_to_utf8ish(candidate) << "\n";
                return path_to_utf8ish(candidate);
            }
        }
        throw std::runtime_error("could not find a compiled .aexrt engine for the ONNX path; run aexrtc build first");
    }
    if (ends_with_ci(options.package_path, ".onnx")) {
        Options tmp = options;
        tmp.onnx_path = options.package_path;
        return resolve_package_path(tmp);
    }
    return options.package_path;
}

static void print_usage() {
    std::cout
        << "native_dxgi_yolo_overlay.exe [model.onnx|model.aexrt]\n"
        << "  --onnx PATH       Parse and compile ONNX directly in the C++ AEXRT runtime\n"
        << "  --package PATH    Load a compiled AEXRT engine directly\n"
        << "  --adapter N       DXGI/AEXRT adapter index, default 0\n"
        << "  --max-det N       Max boxes to read back, default 64\n"
        << "  --frames N        Exit after N rendered frames, default 0 means run until window closes\n";
}

static Options parse_options(int argc, char** argv) {
    Options options;
    bool positional_consumed = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--onnx") {
            options.onnx_path = need_value("--onnx");
        } else if (arg == "--package") {
            options.package_path = need_value("--package");
        } else if (arg == "--adapter") {
            options.adapter_index = static_cast<uint32_t>(std::stoul(need_value("--adapter")));
        } else if (arg == "--max-det") {
            options.max_detections = static_cast<uint32_t>(std::stoul(need_value("--max-det")));
        } else if (arg == "--frames") {
            options.frame_limit = static_cast<uint32_t>(std::stoul(need_value("--frames")));
        } else if (!positional_consumed) {
            options.package_path = arg;
            positional_consumed = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
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

    bool capture(CapturedFrame& frame, uint32_t timeout_ms = 16) {
        DXGI_OUTDUPL_FRAME_INFO frame_info{};
        ComPtr<IDXGIResource> desktop_resource;
        HRESULT hr = duplication_->AcquireNextFrame(timeout_ms, &frame_info, &desktop_resource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            return false;
        }
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            throw std::runtime_error("DXGI desktop duplication access lost");
        }
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
    if (input_elements % 3 != 0) {
        throw std::runtime_error("YOLO input is not 3-channel CHW");
    }
    uint64_t pixels = input_elements / 3;
    uint32_t size = static_cast<uint32_t>(std::sqrt(static_cast<double>(pixels)) + 0.5);
    if (uint64_t(size) * size != pixels) {
        throw std::runtime_error("YOLO input is not square 3xHxW");
    }
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
        if (b.x2 > b.x1 && b.y2 > b.y1) {
            boxes.push_back(b);
        }
    }
    return boxes;
}

static LRESULT CALLBACK overlay_wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_ERASEBKGND:
        return 1;
    default:
        return DefWindowProc(hwnd, msg, wparam, lparam);
    }
}

static HWND create_overlay_window(HINSTANCE instance, uint32_t frame_w, uint32_t frame_h) {
    const char* class_name = "AEXRT_DXGI_YOLO_OVERLAY";
    WNDCLASSA wc{};
    wc.lpfnWndProc = overlay_wnd_proc;
    wc.hInstance = instance;
    wc.lpszClassName = class_name;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    RegisterClassA(&wc);

    int target_w = static_cast<int>(std::min<uint32_t>(frame_w, 1280));
    int target_h = static_cast<int>(std::max<uint32_t>(1, uint64_t(target_w) * frame_h / std::max(1u, frame_w)));
    RECT rect{0, 0, target_w, target_h};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowExA(
        0,
        class_name,
        "AEXRT DXGI YOLO",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        nullptr,
        nullptr,
        instance,
        nullptr);
    if (!hwnd) {
        throw std::runtime_error("CreateWindowExA failed");
    }
    return hwnd;
}

static void draw_frame(HWND hwnd, const CapturedFrame& frame, const std::vector<DrawBox>& boxes) {
    RECT client{};
    GetClientRect(hwnd, &client);
    int client_w = std::max(1L, client.right - client.left);
    int client_h = std::max(1L, client.bottom - client.top);

    HDC hdc = GetDC(hwnd);
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = static_cast<LONG>(frame.width);
    bmi.bmiHeader.biHeight = -static_cast<LONG>(frame.height);
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    SetStretchBltMode(hdc, HALFTONE);
    StretchDIBits(
        hdc,
        0,
        0,
        client_w,
        client_h,
        0,
        0,
        static_cast<int>(frame.width),
        static_cast<int>(frame.height),
        frame.bgra.data(),
        &bmi,
        DIB_RGB_COLORS,
        SRCCOPY);

    HPEN pen = CreatePen(PS_SOLID, 2, RGB(0, 255, 70));
    HGDIOBJ old_pen = SelectObject(hdc, pen);
    HGDIOBJ old_brush = SelectObject(hdc, GetStockObject(HOLLOW_BRUSH));
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(0, 255, 70));

    for (const auto& box : boxes) {
        int x1 = static_cast<int>(std::round(box.x1 * client_w / std::max(1u, frame.width)));
        int y1 = static_cast<int>(std::round(box.y1 * client_h / std::max(1u, frame.height)));
        int x2 = static_cast<int>(std::round(box.x2 * client_w / std::max(1u, frame.width)));
        int y2 = static_cast<int>(std::round(box.y2 * client_h / std::max(1u, frame.height)));
        Rectangle(hdc, x1, y1, x2, y2);
        std::ostringstream label;
        label << "cls " << box.class_id << " " << std::fixed << std::setprecision(2) << box.score;
        std::string text = label.str();
        TextOutA(hdc, x1 + 3, std::max(0, y1 - 16), text.c_str(), static_cast<int>(text.size()));
    }

    SelectObject(hdc, old_brush);
    SelectObject(hdc, old_pen);
    DeleteObject(pen);
    ReleaseDC(hwnd, hdc);
}

int main(int argc, char** argv) {
    try {
        SetProcessDPIAware();
        Options options = parse_options(argc, argv);
        const bool use_onnx = !options.onnx_path.empty() || ends_with_ci(options.package_path, ".onnx");
        const std::string model_path = !options.onnx_path.empty() ? options.onnx_path : options.package_path;

        if (!aexrt_d3d12_probe()) {
            std::cerr << "no D3D12 adapter available\n";
            return 1;
        }

        aexrt::Device device(options.adapter_index);
        auto model = use_onnx ? aexrt::compile_yolo_from_onnx(device, model_path) : aexrt::load_engine(device, model_path);
        if (!model.executable() || model.package_mode() != AEXRT_YOLO_PACKAGE_NATIVE_D3D12_GRAPH) {
            std::cerr << "model is not an executable native_d3d12_graph YOLO graph\n";
            return 2;
        }

        const uint32_t input_size = infer_square_input_size(model.input_element_count());
        DxgiScreenCapture capture(options.adapter_index, input_size, input_size);
        std::cout << "AEXRT DXGI YOLO overlay\n"
                  << "  source=" << model_path << (use_onnx ? " (direct ONNX)" : " (package)") << "\n"
                  << "  capture=" << capture.width() << "x" << capture.height()
                  << " desktop=" << capture.desktop_width() << "x" << capture.desktop_height() << "\n"
                  << "  model_input=" << input_size << "x" << input_size
                  << " classes=" << model.class_count()
                  << " prepared_commands=" << model.prepared_command_count()
                  << " supported=" << model.supported_prepared_command_count()
                  << " unsupported=" << model.unsupported_prepared_command_count() << "\n";

        std::vector<float> warmup(model.input_element_count(), 0.0f);
        (void)aexrt::run_yolo(device, model, warmup, 1);
        std::cout << "  optimized skipped=" << model.prepared_skipped_command_count()
                  << " late_concat_conv1x1=" << model.late_concat_conv1x1_fusion_count()
                  << " concat_residual_cv2=" << model.concat_residual_conv1x1_fusion_count()
                  << " c2f_bottleneck=" << model.c2f_bottleneck_superblock_count()
                  << " c2f_tail=" << model.c2f_tail_residual_fusion_count()
                  << " head_fusion=" << (model.head_fusion_enabled() ? 1 : 0)
                  << " head_final_conv=" << (model.head_final_conv_fusion_enabled() ? 1 : 0)
                  << "\n";
        std::cout << "  tileflow conv1x1=" << model.tileflow_conv1x1_count()
                  << " spatial3x3=" << model.tileflow_3x3_spatial_count()
                  << " pack4_3x3=" << model.tileflow_3x3_pack4_count()
                  << " pack8_3x3=" << model.tileflow_3x3_pack8_count()
                  << " implicit_gemm_3x3=" << model.tileflow_3x3_implicit_gemm_count()
                  << " shape40=" << model.tileflow_3x3_implicit_gemm_40x40_count()
                  << " shape20=" << model.tileflow_3x3_implicit_gemm_20x20_count()
                  << " shape10=" << model.tileflow_3x3_implicit_gemm_10x10_count()
                  << "\n";

        HWND hwnd = create_overlay_window(GetModuleHandle(nullptr), capture.width(), capture.height());
        uint32_t rendered = 0;
        uint32_t frames_since_title = 0;
        double last_capture_ms = 0.0;
        double last_preprocess_ms = 0.0;
        double last_infer_ms = 0.0;
        double last_frame_ms = 0.0;
        auto title_start = std::chrono::high_resolution_clock::now();
        bool running = true;
        while (running) {
            MSG msg{};
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    running = false;
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            if (!running) break;

            auto frame_start = std::chrono::high_resolution_clock::now();
            CapturedFrame frame;
            auto cap0 = std::chrono::high_resolution_clock::now();
            if (!capture.capture(frame, 100)) {
                continue;
            }
            auto cap1 = std::chrono::high_resolution_clock::now();
            last_capture_ms = std::chrono::duration<double, std::milli>(cap1 - cap0).count();

            Letterbox letterbox{};
            auto prep0 = std::chrono::high_resolution_clock::now();
            std::vector<float> input = preprocess_bgra_letterbox(frame, input_size, letterbox);
            auto prep1 = std::chrono::high_resolution_clock::now();
            last_preprocess_ms = std::chrono::duration<double, std::milli>(prep1 - prep0).count();

            auto t0 = std::chrono::high_resolution_clock::now();
            std::vector<AexrtYoloDetection> detections = aexrt::run_yolo(device, model, input, options.max_detections);
            auto t1 = std::chrono::high_resolution_clock::now();
            last_infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::vector<DrawBox> boxes = map_detections_to_source(detections, letterbox, frame.width, frame.height);
            draw_frame(hwnd, frame, boxes);
            auto frame_end = std::chrono::high_resolution_clock::now();
            last_frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

            ++rendered;
            ++frames_since_title;
            auto now = frame_end;
            double elapsed = std::chrono::duration<double>(now - title_start).count();
            if (elapsed >= 0.5) {
                double fps = frames_since_title / elapsed;
                std::ostringstream title;
                title << "AEXRT DXGI YOLO | det=" << boxes.size()
                      << " cap=" << std::fixed << std::setprecision(2) << last_capture_ms << " ms"
                      << " prep=" << last_preprocess_ms << " ms"
                      << " infer=" << last_infer_ms << " ms"
                      << " frame=" << last_frame_ms << " ms"
                      << " fps=" << std::setprecision(1) << fps;
                SetWindowTextA(hwnd, title.str().c_str());
                frames_since_title = 0;
                title_start = now;
            }
            if (options.frame_limit != 0 && rendered >= options.frame_limit) {
                running = false;
            }
        }
        DestroyWindow(hwnd);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 10;
    }
}
