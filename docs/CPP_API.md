# AEXRT Native C++ API

AEXRT now has a pure native C++ runtime DLL. It does not include `Python.h`, does not require `python311.dll`, and does not load the Python extension module.

Current native C++ artifacts:

- `native/aexrt.h`: stable C ABI.
- `native/aexrt.hpp`: RAII C++ wrapper.
- `native/aexrt_d3d12_runtime.cpp`: pure C++ D3D12 runtime implementation.
- `build/native/aexrt_native_cpp.dll`: built native runtime.
- `examples/cpp/native_relu_pure.cpp`: pure C++ example.
- `examples/export_yolo_package.py`: ONNX-to-`.aexrt` compiler example.

It currently covers these native D3D12 float32 graph paths:

- `Relu`
- `Relu -> Gelu`
- `Add -> Relu`
- prepared graph command replay
- host float32 input/output

## Build

Build the pure native C++ runtime and example:

```powershell
powershell -ExecutionPolicy Bypass -File examples\cpp\build_pure_native_relu.ps1
```

Run:

```powershell
examples\cpp\bin\native_relu_pure.exe
examples\cpp\bin\native_relu_graph.exe
examples\cpp\bin\native_relu_gelu_graph.exe
examples\cpp\bin\native_add_relu_graph.exe
```

Expected:

```text
AEXRT pure C++ native ReLU max diff: 0
AEXRT C++ compiled graph ReLU max diff: 0
AEXRT C++ Relu->Gelu graph max diff: 1.19209e-07
AEXRT C++ Add->Relu graph max diff: 0
```

The older `examples/cpp/native_relu.cpp` demonstrates loading symbols from the Python extension `.pyd`, but that is now a compatibility/demo path. The first-class C++ path is `native_relu_pure.cpp` + `aexrt_native_cpp.dll`.

## Optional Python Extension Build

The Python extension still exists:

```powershell
py setup.py build_ext --inplace --force
```

That build is for Python `InferenceSession(..., backend="native_d3d12")`, not for native C++ applications.

## Exported C ABI

The declarations live in:

```text
native\aexrt.h
```

The C++ RAII wrapper lives in:

```text
native\aexrt.hpp
```

Minimal C++ usage:

```cpp
#include "aexrt.hpp"

int main() {
    aexrt::Device device(0);
    std::vector<float> x = {-2.0f, 3.0f};
    auto input = aexrt::upload_float32(device, x);
    auto output = aexrt::allocate_float32_uav(device, x.size());
    auto dispatch = aexrt::prepare_relu_float32(device, input, output, x.size());
    aexrt::execute(device, dispatch);
    auto y = aexrt::download_float32(device, output, x.size());
}
```

Multi-node graph usage:

```cpp
aexrt::Device device(0);

auto relu_gelu = aexrt::compile_relu_gelu_graph(device, x.size());
auto y0 = aexrt::run(device, relu_gelu, x);

auto add_relu = aexrt::compile_add_relu_graph(device, a.size());
auto y1 = aexrt::run(device, add_relu, a, b);
```

Current exported functions:

```cpp
int aexrt_d3d12_probe();
AexrtDevice* aexrt_d3d12_create_device(uint32_t adapter_index);
void aexrt_d3d12_destroy_device(AexrtDevice* device);

AexrtBuffer* aexrt_d3d12_upload_float32(
    AexrtDevice* device,
    const float* data,
    uint64_t element_count);

AexrtBuffer* aexrt_d3d12_allocate_float32_uav(
    AexrtDevice* device,
    uint64_t element_count);

int aexrt_d3d12_relu_float32(
    AexrtDevice* device,
    AexrtBuffer* input,
    AexrtBuffer* output,
    uint64_t element_count);

int aexrt_d3d12_download_float32(
    AexrtDevice* device,
    AexrtBuffer* buffer,
    float* out,
    uint64_t element_count);

AexrtCompiledGraph* aexrt_compile_graph(
    AexrtDevice* device,
    const AexrtGraphDesc* desc);

int aexrt_run(
    AexrtDevice* device,
    AexrtCompiledGraph* graph,
    const float* input,
    float* output);

void aexrt_destroy_graph(AexrtCompiledGraph* graph);

AexrtYoloModel* aexrt_yolo_load_engine(
    AexrtDevice* device,
    const char* engine_path);

int aexrt_yolo_run(
    AexrtDevice* device,
    AexrtYoloModel* model,
    const float* input,
    uint64_t input_element_count,
    AexrtYoloDetection* detections,
    uint32_t max_detections,
    uint32_t* out_detection_count);

void aexrt_yolo_destroy(AexrtYoloModel* model);
```

## Binary Model Engine

The first-class model deployment path is a binary `.aexrt` engine. Build it
directly from ONNX; no JSON package is generated or loaded:

```powershell
aexrtc build model.onnx -o model.aexrt
aexrtc inspect model.aexrt
```

```cpp
#include "aexrt.hpp"

aexrt::Device device(0);
auto model = aexrt::load_engine(device, "model.aexrt");
std::vector<float> input(model.input_element_count(), 0.0f);
auto detections = aexrt::run_yolo(device, model, input, 100);
```

Engine V1 carries the binary command stream, raw FP16/FP32 constants, fixed
kernel and fusion plans, packed stride-2 3x3 weights, one activation arena, and
the DXIL/PSO cache. The C++ loader validates the nine-section container and
executes the serialized choices. See
[`AEXRT_ENGINE.md`](AEXRT_ENGINE.md) for the exact wire format.

The small `AexrtGraphDesc` API and its JSON graph loader remain compatibility
interfaces for elementwise tests. They are not the model deployment workflow.
