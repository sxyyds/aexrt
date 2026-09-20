# 路线图

## v0.1 已完成

- Graph IR
- Numpy CPU backend
- Torch CUDA/CPU backend
- 常量折叠
- Identity 删除
- MatMul + Add + Activation 融合
- MLP / Attention / RoPE 测试
- MLP benchmark
- ONNX 常见算子导入器
- Torch CUDA Graph replay 雏形

## v0.2：Portable Execution Core

目标：先把“所有显卡都能接入”的运行时骨架做硬，而不是先绑定 CUDA。

- 结构化 backend capability：op、dtype、feature、device 信息
- 静态 execution plan：节点序列、设备信息、能力校验
- 静态 memory plan：input / constant / temporary / output、生命周期、buffer 字节估算
- temporary buffer 保守复用：仅复用生命周期不重叠的中间值
- 后端 prepare 阶段拒绝不支持的 op / dtype
- Torch / Numpy 作为第一批 capability backend
- `torch-directml` 实验桥：`backend="directml"`
- AEXRT Device / Buffer HAL：`RuntimeDeviceInfo`, `DeviceBuffer`, upload, download
- AEXRT native D3D12 入口：`backend="native_d3d12"` / `device="d3d12"`
- `backend="auto"` 默认优先 CUDA，再 DirectML 兼容桥，再 CPU；显式 D3D12 不 fallback
- 固化低延迟 benchmark，持续跟踪 CUDA Graph replay 收益

## v0.3：Native D3D12 Minimum Backend

目标：先在 Windows 上打通不依赖 DirectML 的 AEXRT 自有 GPU 后端闭环。

- D3D12 device / command queue / descriptor heap 初始化
- native buffer upload / download
- persistent constants
- `Add`, `MatMul`, `FusedLinear`, `Gelu`, `LayerNorm`, `Softmax`
- 与 Numpy / Torch 后端做数值对齐测试
- native D3D12 backend capability 上报

## v0.4：低延迟执行层

- 统一 buffer allocator
- 临时 buffer 复用
- input staging buffer
- zero-copy host/device staging where possible
- async execution queue
- shape-specialized compiled graph cache
- autotune cache：按 device / shape / dtype 选择策略

## v0.5：热路径特化

- CUDA native backend
- Vulkan compute backend
- Metal backend
- Fused LayerNorm kernel
- Fused RoPE kernel
- Fused Bias+GELU kernel
- KV-cache friendly SDPA

## v0.6：模型生态

- 完整 ONNX importer
- safetensors loader
- LLaMA/Qwen/Gemma block builder
- INT8/INT4 weight-only quantization
- FP8 path

## v1.0：生产化

- C ABI
- Python wheel
- graph serialization ABI
- telemetry-free profiler
- deterministic benchmark suite
