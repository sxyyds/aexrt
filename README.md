# AetherX Runtime (AEXRT)

**English | [简体中文](README.zh-CN.md)**

A **fully self-developed D3D12 inference engine** — no CUDA, no DirectML, no
external ML runtime dependencies. All compute kernels are hand-written HLSL
compute shaders. ONNX models compile to a binary `.aexrt` file containing the
complete GPU execution plan.

## .aexrt Binary Engine

The first-class native model workflow is binary and does not require JSON:

```text
model.onnx -> aexrtc -> model.aexrt -> AEXRT native D3D12
```

```powershell
py -3.11 -m pip install -e .
aexrtc build models\cs2V8_320.onnx -o examples\cs2V8_320.aexrt
aexrtc inspect examples\cs2V8_320.aexrt
examples\cpp\bin\native_yolo_package.exe examples\cs2V8_320.aexrt --runs 100
```

V1 stores the binary command stream, raw constants, fixed kernel plan, packed
weights, and GPU memory arena in one checksummed file. See
[`docs/AEXRT_ENGINE.md`](docs/AEXRT_ENGINE.md) for the wire format and APIs.

## Architecture

- **Binary engine** (`.aexrt`): command stream + constants + kernel plan +
  packed weights + GPU memory arena in one checksummed file
- **Native D3D12 HAL**: device, buffer, queue, fence, descriptor arena —
  all self-built, no DirectML dependency
- **Graph compiler**: ONNX → Graph IR → kernel plan → physical dispatch plan
- **Memory planner**: lifetime-colored arena with page-aware slot reuse
- **Autotuner**: per-device algorithm selection with numeric gating
- **W8A8 quantization**: producer-side pre-quantization (dynamic per-tensor
  activation scale, per-channel weight scale, int8 dot4 compute)

### Key Features

- 55+ hand-written HLSL compute kernels (winograd, implicit GEMM, stride2
  direct, FP16 packed, int8 dot4, fused conv+silu, concat+conv, neck
  lattice chains, C2F tail residual, paired conv, YOLO head fusion)
- W8A8 int8 with accuracy audit (per-layer simulated error, detection-head
  exclusion, 8-image real-data validation vs ORT-CPU ground truth)
- Offline autotuner with fail-closed numeric gate
- N-buffer pipelining (1-4 frames)
- GPU timestamp profiling per dispatch

### Performance (RTX 5060 Laptop GPU, 320x320 YOLO inference)

After 38 optimization rounds (full log in docs/NATIVE_D3D12.md):

| Model | e2e latency | fps | cumulative vs baseline |
|---|---|---|---|
| cs2V8_320 | **1.08 ms** | 922 | **-63%** |
| apex10w | 1.79 ms | 558 | -68% |
| DYv11s | 2.50 ms | 400 | -62% |

Engine core latency (submit + GPU + readback, zero-copy input): ~1.0 ms.

Correctness: 8 real images, all engines match ORT-CPU ground truth
(detection count 100%, score delta ≤ 0.02). 300 unit tests green.

Highlights of the optimization campaign:
- W8A8 int8 with per-16-channel-group scales (pg16), quant dispatch dedup
- LDS-tiled GEMM kernels for 1x1/3x3/stride2 convolutions
- Fixed a hidden amax grid-collapse bottleneck (single-round -21~-42%)
- Zero-copy input path (persistent-mapped upload buffer)
- Honest negative results documented (register cliffs, fusion traps)

### Engine Types

- **fp16/fp32**: standard precision, autotuner selects per-shape algorithms
- **int8 (W8A8)**: producer-side pre-quantization, ~20-30% faster than fp16,
  accuracy audited (detection-head excluded, all layers ≤2% simulated error)

### Build

```powershell
py setup.py build_ext --inplace
powershell -ExecutionPolicy Bypass -File examples\cpp\build_pure_native_relu.ps1
```

### Validate

```powershell
py tests\test_runtime.py
examples\cpp\bin\native_yolo_package.exe examples\cs2V8_320.aexrt --runs 100 --profile
```

## Documentation

Bilingual (EN + 简体中文) — see the [full index](docs/README.md):

- Binary engine wire format — [EN](docs/AEXRT_ENGINE.md) / [中文](docs/AEXRT_ENGINE.zh-CN.md)
- Architecture — [EN](docs/ARCHITECTURE.en.md) / [中文](docs/ARCHITECTURE.md)
- Native C++ API — [EN](docs/CPP_API.md) / [中文](docs/CPP_API.zh-CN.md)
- 38-round optimization log — [中文 (full)](docs/NATIVE_D3D12.md) / [EN (summary)](docs/NATIVE_D3D12.en.md)
- Roadmap — [EN](docs/ROADMAP.en.md) / [中文](docs/ROADMAP.md)

## LLM Operators (Python backend)

LayerNorm, GELU, Softmax, Embedding, SDPA, RoPE available via
`InferenceSession(graph, backend="torch")`.

## License

[MIT](LICENSE)
