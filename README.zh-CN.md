# AetherX Runtime (AEXRT)

**[English](README.md) | 简体中文**

完全自研的 D3D12 推理引擎——不用 CUDA、不用 DirectML、不依赖任何外部
ML 运行时。所有计算内核均为手写 HLSL compute shader，ONNX 模型编译为
包含完整 GPU 执行计划的 `.aexrt` 二进制引擎。

## .aexrt 二进制引擎

一等公民的原生模型工作流是二进制的，无需 JSON：

```text
model.onnx -> aexrtc -> model.aexrt -> AEXRT 原生 D3D12
```

```powershell
py -3.11 -m pip install -e .
aexrtc build models\cs2V8_320.onnx -o examples\cs2V8_320.aexrt
aexrtc inspect examples\cs2V8_320.aexrt
examples\cpp\bin\native_yolo_package.exe examples\cs2V8_320.aexrt --runs 100
```

V1 将二进制命令流、原始常量、固定 kernel 计划、打包权重和 GPU 内存
arena 存进一个带校验和的文件。线格式与 API 见
[`docs/AEXRT_ENGINE.md`](docs/AEXRT_ENGINE.md)。

## 架构

- **二进制引擎**（`.aexrt`）：命令流 + 常量 + kernel 计划 + 打包权重 +
  GPU 内存 arena，单文件带校验和
- **原生 D3D12 HAL**：设备、缓冲、队列、围栏、描述符 arena——全部
  自建，零 DirectML 依赖
- **图编译器**：ONNX → 图 IR → kernel 计划 → 物理调度计划
- **内存规划器**：生命周期着色 arena + 页感知 slot 复用
- **离线 autotuner**：按设备实测选算法，带数值门禁（fail-closed）
- **W8A8 量化**：生产者端预量化（动态激活 scale + 逐通道权重 scale，
  int8 dot4 计算）

## 核心特性

- **55+ 手写 HLSL 计算内核**：Winograd F(2x2)/F(4x4)、隐式 GEMM、
  stride2 direct、FP16 packed、int8 dot4、融合 conv+SiLU、concat+conv、
  neck lattice 链、C2F tail residual、paired conv、YOLO 头融合
- **W8A8 int8 + 精度审计**：逐层误差模拟、检测头排除、8 张真实图像
  对 ORT-CPU 真值验证
- **per-16-通道组激活 scale（pg16）**：比 per-tensor 精度减半的误差，
  与 per-tensor 同速（组内 int 链 + 组间一次浮点转换）
- **LDS 分块 GEMM 内核**（1x1/3x3/stride2）：权重/输入 tile 进
  groupshared，实测复用率 × K 深度决定成败
- **量化调度去重**：同一输入同变体只录一次量化
- **零拷贝输入**：持久映射上传缓冲直写，消除每帧 memcpy
- **N-buffer 流水线**（1-4 帧）+ 逐 dispatch GPU 时间戳剖析

## 性能（RTX 5060 Laptop GPU，320×320 YOLO 推理）

38 轮优化后（完整日志见 [docs/NATIVE_D3D12.md](docs/NATIVE_D3D12.md)）：

| 模型 | 端到端延迟 | fps | 累计 vs 基线 |
|---|---|---|---|
| cs2V8_320 | **1.08 ms** | 922 | **-63%** |
| apex10w | 1.79 ms | 558 | -68% |
| DYv11s | 2.50 ms | 400 | -62% |

引擎核心延迟（提交 + GPU + 读回，零拷贝输入）：**~1.0 ms**。

正确性：8 张真实图像全部与 ORT-CPU 真值一致（检测计数 100%，分数
偏差 ≤ 0.02），300 个单元测试全绿。

优化战役亮点：
- W8A8 int8 + per-16-通道组 scale（pg16）+ 量化调度去重
- 1x1/3x3/stride2 卷积的 LDS 分块 GEMM 内核族
- 揪出隐藏的 amax 网格塌缩瓶颈（单轮 -21~-42%）
- 零拷贝输入通路（持久映射上传缓冲）
- 诚实的负结果记录（寄存器悬崖定律、融合陷阱、时序 scale 精度门禁）

## 引擎类型

- **fp16/fp32**：标准精度，autotuner 按形状选算法
- **int8（W8A8）**：生产者端预量化，比 fp16 快 20-30%，精度经审计
  （检测头排除，全部层 ≤2% 模拟误差）

## 构建

```powershell
py setup.py build_ext --inplace
powershell -ExecutionPolicy Bypass -File examples\cpp\build_pure_native_relu.ps1
```

## 验证

```powershell
py tests\test_runtime.py
examples\cpp\bin\native_yolo_package.exe examples\cs2V8_320.aexrt --runs 100 --profile
```

## 详细文档

- [`docs/AEXRT_ENGINE.md`](docs/AEXRT_ENGINE.md)——二进制引擎线格式
- [`docs/NATIVE_D3D12.md`](docs/NATIVE_D3D12.md)——38 轮优化全日志
  （内核重写、int8 量化、LDS 分块、正确性修复、负结果及其定律）
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)——设计概览
- [`docs/CPP_API.md`](docs/CPP_API.md)——C/C++ API 参考
- [`docs/ROADMAP.md`](docs/ROADMAP.md)——原始路线图

## LLM 算子（Python 后端）

LayerNorm、GELU、Softmax、Embedding、SDPA、RoPE 可通过
`InferenceSession(graph, backend="torch")` 使用。

## 许可证

[MIT](LICENSE)
