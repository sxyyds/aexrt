# AEXRT 二进制引擎 V1

**[English](AEXRT_ENGINE.md) | 简体中文**

> 本文件是线格式规范的结构性中文版：全部章节、表格与字节布局与英文版
> 一一对应；英文版中的部分实现注记做了压缩。逐字节细节以
> [英文版](AEXRT_ENGINE.md) 为准。

`.aexrt` 是 AEXRT 原生运行时的一等公民可部署模型格式。常规工作流：

```text
model.onnx -> aexrtc -> model.aexrt -> 原生 D3D12 运行时
```

不生成也不需要 JSON 包。一个引擎包含：可执行命令流、原始常量、固定
kernel 计划与物理调度计划、打包权重、GPU 内存 arena 计划、序列化融合
组以及 shader/PSO 缓存。C++ 加载器校验容器并执行序列化的选择——引擎
即部署工件。

Python 与 C++ 都能直接加载引擎：

```python
from aexrt import NativeCppYoloModel

model = NativeCppYoloModel("examples/cs2V8_320.aexrt")
```

```cpp
#include "aexrt.hpp"

aexrt::Device device(0);
auto model = aexrt::load_engine(device, "examples\\cs2V8_320.aexrt");
```

## 容器布局

所有整数与浮点均为小端。文件头与节表之后是按 64 字节对齐的节载荷。
每节有独立 CRC32。节表内的偏移相对文件；原始常量与打包权重记录内的
偏移相对节。

64 字节文件头：

| 偏移 | 类型 | 字段 | V1 值 |
| ---: | --- | --- | --- |
| 0 | `char[8]` | magic | `AEXRTENG` |
| 8 | `u32` | 引擎版本 | `1` |
| 12 | `u32` | 头大小 | `64` |
| 16 | `u32` | 节数 | 当前编译器输出为 `11`；无第 11 节的旧引擎为 `9`/`10` |
| 20 | `u32` | flags | bit 0 必须置位 |
| 24 | `u64` | 文件大小 | 精确字节数 |
| 32 | `u64` | TOC 偏移 | `64` |
| 40 | `u64` | TOC 大小 | `节数 * 32` |
| 48 | `u8[16]` | 保留 | 零 |

每个 32 字节 TOC 项为 `<IIQQII>`：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| 节类型 | `u32` | 下表的节 ID |
| 节 flags | `u32` | V1 为零 |
| 偏移 | `u64` | 相对文件的载荷偏移 |
| 大小 | `u64` | 载荷字节数 |
| 校验和 | `u32` | 恰为载荷字节的 CRC32 |
| 保留 | `u32` | 零 |

容器 V1 要求 ID 1–9 九个基础语义节。当前编译器额外产出第 10、11 节。
缺少可选节的旧 V1 引擎仍然合法。第 11 节要求 Memory Plan V2 与第
10 节同时存在；读取器拒绝重复节并校验每一个存在的节。

| ID | 节 | 用途 |
| ---: | --- | --- |
| 1 | Manifest | 目标、精度、YOLO 元数据、计数、源哈希 |
| 2 | Values | 稠密值表与 arena 绑定 |
| 3 | Commands | 可执行二进制命令流 |
| 4 | Raw constants | 未压缩张量字节 |
| 5 | Kernel plan | 权威 kernel 与精度选择 |
| 6 | Packed weights | kernel 就绪的权重布局 |
| 7 | Memory plan | 版本化激活页、存储元数据、逐值字节区间 |
| 8 | Fusion plan | 权威逻辑融合组（kind 1–6） |
| 9 | Pipeline cache | 键控 DXIL/shader 字节码与驱动 PSO blob |
| 10 | Physical dispatch plan | 权威物理所有权、执行点、kernel、精度与稀疏逻辑命令成员 |
| 11 | Arena barrier plan | 权威逐调度页转换与重叠 UAV 复用排序 |

## Manifest

Manifest 为 124 字节：17 个 `u32`、两个 `u64`、两个 `f32`，随后是 32
字节 SHA-256 摘要。

| 索引 | 字段 | V1 值或含义 |
| ---: | --- | --- |
| 0 | manifest 版本 | `1` |
| 1 | 运行时目标 | `1` = 原生 D3D12 |
| 2 | 存储/计划精度 | `1` = FP32，`2` = FP16 |
| 3 | 模式 | `2` = 带 YOLO 后处理的原生图 |
| 4 | 输出布局 | `1` = channels first，`2` = channels last |
| 5 | objectness | `0` 或 `1` |
| 6-10 | 输出元数据 | channels、anchors、classes、max candidates、max detections |
| 11-14 | 图计数 | 节点、值、常量、命令 |
| 15-16 | 值 ID | 图输入、图输出 |

其余字段为 `input_elements: u64`、`arena_nbytes: u64`、
`conf_threshold: f32`、`iou_threshold: f32` 与 `source_sha256: u8[32]`。
从文件编译时，摘要是源 ONNX 字节的 SHA-256。

## 值与命令

Values 节以 `<II>` 开头（`count`、保留零）。每个值是 48 字节
`<IIQ4IQQ>` 记录：

```text
id, flags, elements, shape[4], arena_offset, arena_nbytes
```

值 ID 稠密且记录按 ID 排序。shape 归一化为四维。值 flags：bit 0 输入、
bit 1 常量、bit 2 arena、bit 3 alias。非 arena 值的 arena 偏移为
`UINT64_MAX`、大小为零。Memory Plan V2 下，arena 偏移相对该值在第 7 节
的页；值记录刻意不重复页 ID。第 7 节必须重复相同的偏移与大小。

Commands 节同样以 `<II>` 开头。每个变长命令有 24 字节 `<6I>` 头，随后
是 `input_count` 个输入 ID 与 `param_count` 个操作参数（均为 `u32`
数组）：

```text
kind, output, input_count, param_count, planned_kernel, precision
```

命令 kind ID：

| ID | Kind | ID | Kind |
| ---: | --- | ---: | --- |
| 1 | Conv | 8 | Resize |
| 2 | Conv + SiLU | 9 | MaxPool |
| 3 | Concat + Conv1x1 | 10 | Unary |
| 4 | View | 11 | Binary |
| 5 | Alias | 12 | MatMul |
| 6 | Slice | 13 | Transpose |
| 7 | Concat | 14 | Softmax |

> 其余章节——**Raw Constants、Kernel Plan、Packed Weights、Memory
> Arena（V2 页计划与 V1 遗留）、Fusion Plan、Physical Dispatch Plan
> V2、Arena Barrier Plan V1、Pipeline Cache、校验与演进**——的完整
> 字节级表格见
> [英文版](AEXRT_ENGINE.md#raw-constants)。结构与语义与本节相同：
> 每节以计数头开始，记录定长且按稳定键排序，全部偏移与校验和规则
> 遵循"容器布局"一节。
