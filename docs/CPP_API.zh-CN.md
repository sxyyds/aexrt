# AEXRT 原生 C++ API

**[English](CPP_API.md) | 简体中文**

AEXRT 提供纯原生 C++ 运行时 DLL：不包含 `Python.h`，不依赖
`python311.dll`，不加载 Python 扩展模块。

当前原生 C++ 产物：

- `native/aexrt.h`：稳定的 C ABI。
- `native/aexrt.hpp`：RAII C++ 封装。
- `native/aexrt_d3d12_runtime.cpp`：纯 C++ D3D12 运行时实现。
- `build/native/aexrt_native_cpp.dll`：构建出的原生运行时。
- `examples/cpp/native_relu_pure.cpp`：纯 C++ 示例。
- `examples/export_yolo_package.py`：ONNX → `.aexrt` 编译器示例。

当前覆盖的原生 D3D12 float32 图路径：

- `Relu`
- `Relu -> Gelu`
- `Add -> Relu`
- 预录制图命令回放
- host float32 输入/输出

## 构建

构建纯原生 C++ 运行时与示例：

```powershell
powershell -ExecutionPolicy Bypass -File examples\cpp\build_pure_native_relu.ps1
```

运行：

```powershell
examples\cpp\bin\native_relu_pure.exe
examples\cpp\bin\native_relu_graph.exe
examples\cpp\bin\native_relu_gelu_graph.exe
examples\cpp\bin\native_add_relu_graph.exe
```

预期输出：

```text
AEXRT pure C++ native ReLU max diff: 0
AEXRT C++ compiled graph ReLU max diff: 0
AEXRT C++ Relu->Gelu graph max diff: 1.19209e-07
AEXRT C++ Add->Relu graph max diff: 0
```

旧的 `examples/cpp/native_relu.cpp` 演示从 Python 扩展 `.pyd` 加载符号，
现在只是兼容/演示路径。一等公民的 C++ 路径是 `native_relu_pure.cpp` +
`aexrt_native_cpp.dll`。

## 可选的 Python 扩展构建

Python 扩展仍然存在：

```powershell
py setup.py build_ext --inplace --force
```

该构建服务于 Python 的 `InferenceSession(..., backend="native_d3d12")`，
与原生 C++ 应用无关。

## 导出的 C ABI

声明位于：

```text
native\aexrt.h
```

C++ RAII 封装位于：

```text
native\aexrt.hpp
```

最小 C++ 用法：

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

多节点图用法：

```cpp
aexrt::Device device(0);

auto relu_gelu = aexrt::compile_relu_gelu_graph(device, x.size());
auto y0 = aexrt::run(device, relu_gelu, x);

auto add_relu = aexrt::compile_add_relu_graph(device, a.size());
auto y1 = aexrt::run(device, add_relu, a, b);
```

当前导出函数：

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

## 二进制模型引擎

一等公民的模型部署路径是二进制 `.aexrt` 引擎。直接从 ONNX 构建，不产
生也不加载任何 JSON 包：

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

引擎 V1 携带二进制命令流、原始 FP16/FP32 常量、固定 kernel 与融合计
划、打包的 stride-2 3x3 权重、激活 arena 以及 DXIL/PSO 缓存。C++ 加载
器校验九段容器并执行序列化的选择。精确线格式见
[`AEXRT_ENGINE.zh-CN.md`](AEXRT_ENGINE.zh-CN.md)。

小型 `AexrtGraphDesc` API 及其 JSON 图加载器保留为 elementwise 测试的
兼容接口，不是模型部署工作流。
