#define NOMINMAX
#include <windows.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

struct AexrtD3D12Device;
struct AexrtD3D12Buffer;

using ProbeFn = int (*)();
using CreateDeviceFn = AexrtD3D12Device* (*)(uint32_t);
using DestroyDeviceFn = void (*)(AexrtD3D12Device*);
using UploadFloat32Fn = AexrtD3D12Buffer* (*)(AexrtD3D12Device*, const float*, uint64_t);
using AllocateFloat32UavFn = AexrtD3D12Buffer* (*)(AexrtD3D12Device*, uint64_t);
using DestroyBufferFn = void (*)(AexrtD3D12Buffer*);
using ReluFloat32Fn = int (*)(AexrtD3D12Device*, AexrtD3D12Buffer*, AexrtD3D12Buffer*, uint64_t);
using DownloadFloat32Fn = int (*)(AexrtD3D12Device*, AexrtD3D12Buffer*, float*, uint64_t);

template <typename T>
T load_symbol(HMODULE module, const char* name) {
    auto ptr = reinterpret_cast<T>(GetProcAddress(module, name));
    if (!ptr) {
        std::cerr << "missing symbol: " << name << "\n";
        std::exit(2);
    }
    return ptr;
}

bool try_load_python311() {
    if (GetModuleHandleW(L"python311.dll") || LoadLibraryW(L"python311.dll")) {
        return true;
    }

    wchar_t local_app_data[MAX_PATH]{};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", local_app_data, MAX_PATH) > 0) {
        std::wstring path = std::wstring(local_app_data) + L"\\Programs\\Python\\Python311\\python311.dll";
        if (LoadLibraryW(path.c_str())) {
            return true;
        }
    }

    wchar_t program_files[MAX_PATH]{};
    if (GetEnvironmentVariableW(L"ProgramFiles", program_files, MAX_PATH) > 0) {
        std::wstring path = std::wstring(program_files) + L"\\Python311\\python311.dll";
        if (LoadLibraryW(path.c_str())) {
            return true;
        }
    }

    return false;
}

int main() {
    try_load_python311();

    const wchar_t* dll_path = L"src\\aexrt_native_d3d12.cp311-win_amd64.pyd";
    HMODULE module = LoadLibraryW(dll_path);
    if (!module) {
        DWORD err = GetLastError();
        std::cerr << "failed to load " << "src\\aexrt_native_d3d12.cp311-win_amd64.pyd" << "\n";
        std::cerr << "GetLastError: " << err << "\n";
        std::cerr << "make sure python311.dll is discoverable or run from the same Python install environment\n";
        return 1;
    }

    auto probe = load_symbol<ProbeFn>(module, "aexrt_d3d12_probe");
    auto create_device = load_symbol<CreateDeviceFn>(module, "aexrt_d3d12_create_device");
    auto destroy_device = load_symbol<DestroyDeviceFn>(module, "aexrt_d3d12_destroy_device");
    auto upload = load_symbol<UploadFloat32Fn>(module, "aexrt_d3d12_upload_float32");
    auto allocate_uav = load_symbol<AllocateFloat32UavFn>(module, "aexrt_d3d12_allocate_float32_uav");
    auto destroy_buffer = load_symbol<DestroyBufferFn>(module, "aexrt_d3d12_destroy_buffer");
    auto relu = load_symbol<ReluFloat32Fn>(module, "aexrt_d3d12_relu_float32");
    auto download = load_symbol<DownloadFloat32Fn>(module, "aexrt_d3d12_download_float32");

    if (!probe()) {
        std::cerr << "no D3D12 adapter available\n";
        return 1;
    }

    AexrtD3D12Device* device = create_device(0);
    if (!device) {
        std::cerr << "failed to create AEXRT D3D12 device\n";
        return 1;
    }

    std::vector<float> x = {-2.0f, -0.5f, 0.0f, 3.0f, 9.0f};
    std::vector<float> y(x.size(), 0.0f);
    AexrtD3D12Buffer* input = upload(device, x.data(), static_cast<uint64_t>(x.size()));
    AexrtD3D12Buffer* output = allocate_uav(device, static_cast<uint64_t>(x.size()));
    if (!input || !output) {
        std::cerr << "failed to allocate buffers\n";
        return 1;
    }

    if (!relu(device, input, output, static_cast<uint64_t>(x.size()))) {
        std::cerr << "relu dispatch failed\n";
        return 1;
    }
    if (!download(device, output, y.data(), static_cast<uint64_t>(y.size()))) {
        std::cerr << "download failed\n";
        return 1;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float expected = x[i] > 0.0f ? x[i] : 0.0f;
        max_diff = std::max(max_diff, std::fabs(y[i] - expected));
    }

    std::cout << "AEXRT C++ native ReLU max diff: " << max_diff << "\n";

    destroy_buffer(output);
    destroy_buffer(input);
    destroy_device(device);
    FreeLibrary(module);
    return max_diff == 0.0f ? 0 : 3;
}
