# Roadmap

**[中文](ROADMAP.md) | English**

## v0.1 — Done

- Graph IR
- Numpy CPU backend
- Torch CUDA/CPU backend
- Constant folding
- Identity elimination
- MatMul + Add + Activation fusion
- MLP / Attention / RoPE tests
- MLP benchmark
- ONNX importer for common ops
- CUDA Graph replay prototype

## v0.2 — Portable Execution Core

Goal: harden the "any GPU can plug in" runtime skeleton before binding to
CUDA.

- Structured backend capabilities: op, dtype, feature, device info
- Static execution plan: node order, device info, capability validation
- Static memory plan: input / constant / temporary / output, lifetimes,
  byte estimates
- Conservative temporary reuse (disjoint lifetimes only)
- Backends reject unsupported op / dtype at prepare time
- Torch / Numpy as the first capability backends
- `torch-directml` experimental bridge: `backend="directml"`
- AEXRT Device / Buffer HAL: `RuntimeDeviceInfo`, `DeviceBuffer`, upload,
  download
- AEXRT native D3D12 entry: `backend="native_d3d12"` / `device="d3d12"`
- `backend="auto"` prefers CUDA, then DirectML bridge, then CPU; explicit
  D3D12 never falls back
- Fixed low-latency benchmark tracking CUDA Graph replay gains

## v0.3 — Native D3D12 Minimum Backend

Goal: a complete AEXRT-owned GPU backend loop on Windows without DirectML.

- D3D12 device / command queue / descriptor heap init
- Native buffer upload / download
- Persistent constants
- `Add`, `MatMul`, `FusedLinear`, `Gelu`, `LayerNorm`, `Softmax`
- Numerical alignment tests vs Numpy / Torch backends
- Native D3D12 backend capability reporting

## v0.4 — Low-Latency Execution Layer

- Unified buffer allocator
- Temporary buffer reuse
- Input staging buffer
- Zero-copy host/device staging where possible
- Async execution queue
- Shape-specialized compiled-graph cache
- Autotune cache: policy keyed by device / shape / dtype

## v0.5 — Hot-Path Specialization

- CUDA native backend
- Vulkan compute backend
- Metal backend
- Fused LayerNorm kernel
- Fused RoPE kernel
- Fused Bias+GELU kernel
- KV-cache friendly SDPA

## v0.6 — Model Ecosystem

- Complete ONNX importer
- safetensors loader
- LLaMA/Qwen/Gemma block builder
- INT8/INT4 weight-only quantization
- FP8 path

## v1.0 — Production

- C ABI
- Python wheel
- Graph serialization ABI
- Telemetry-free profiler
- Deterministic benchmark suite
