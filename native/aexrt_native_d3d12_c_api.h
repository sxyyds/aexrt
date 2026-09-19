#pragma once

#include <stdint.h>

#ifdef _WIN32
#define AEXRT_API __declspec(dllimport)
#else
#define AEXRT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DeviceContext AexrtD3D12Device;
typedef struct BufferHandle AexrtD3D12Buffer;

AEXRT_API int aexrt_d3d12_probe(void);
AEXRT_API AexrtD3D12Device* aexrt_d3d12_create_device(uint32_t adapter_index);
AEXRT_API void aexrt_d3d12_destroy_device(AexrtD3D12Device* device);

AEXRT_API AexrtD3D12Buffer* aexrt_d3d12_upload_float32(
    AexrtD3D12Device* device,
    const float* data,
    uint64_t element_count);

AEXRT_API AexrtD3D12Buffer* aexrt_d3d12_allocate_float32_uav(
    AexrtD3D12Device* device,
    uint64_t element_count);

AEXRT_API void aexrt_d3d12_destroy_buffer(AexrtD3D12Buffer* buffer);

AEXRT_API int aexrt_d3d12_relu_float32(
    AexrtD3D12Device* device,
    AexrtD3D12Buffer* input,
    AexrtD3D12Buffer* output,
    uint64_t element_count);

AEXRT_API int aexrt_d3d12_download_float32(
    AexrtD3D12Device* device,
    AexrtD3D12Buffer* buffer,
    float* out,
    uint64_t element_count);

#ifdef __cplusplus
}
#endif
