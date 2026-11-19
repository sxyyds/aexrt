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

### Performance (RTX 5060 Laptop GPU, 320x320)

**N-buffer throughput** (`--pipeline 4`, bounded spin fence; per-frame):

| Model | AEXRT | fps |
|---|---|---|
| cs2V8_320 | **0.95 ms** | **~1053** |
| apex10w | 1.72 ms | 581 |
| DYv11s | 2.34 ms | 427 |

**Single-frame latency** (pipeline=1, same rebuilt harness; AEXRT side uses the
persistent mapped input pointer and the bounded spin fence; 150-200 runs after
50 warm-up submits so DVFS settles):

| Model | AEXRT sync | ORT DirectML EP | ratio |
|---|---|---|---|
| cs2V8_320 | 1.23 ms | 1.28 ms | 1.04x faster |
| apex10w | 2.07 ms | 1.72 ms | 0.83x |
| DYv11s | 3.01 ms | 2.46 ms | 0.82x |

Short runs are DVFS-noisy: across repeated processes the same pairs swing by
±10% (cs2V8 1.04–1.11x, apex 0.83–0.95x, DY 0.82–0.92x). The table is the
150-run single-process result in `build/benchmarks/dml_compare_report.md`
(re-run 2026-11-18); earlier `dml_compare_zerocopy.md` figures were in the same
band. Apex/DY still trail: their GPU event sums are 1.72 / 2.49 ms against
DML end-to-end 1.72 / 2.46 ms, so the remaining gap is kernel structure
(especially 1x1/concat classes), not host overhead. Runtime algo-override
sweeps did not reproduce the in-process autotuner gains; the original engine's
per-shape kernels are already optimal for the existing kernel set.

PC runtime knobs:
- `aexrt_yolo_input_ptr()` — write the frame into the persistent upload buffer,
  then pass that pointer to `aexrt_yolo_run`; the per-frame host memcpy is skipped.
- `AEXRT_NATIVE_D3D12_SPIN_FENCE=1` — bounded (2 ms) fence spin before falling
  back to the event wait; removes the OS wake-up tail.
- `AEXRT_NATIVE_D3D12_PIPELINE_FRAMES=N` (1..4) — N-buffer throughput mode.

Correctness: 8 real images, all engines match ORT-CPU ground truth
(detection count 100%, score delta ≤ 0.02). 300 unit tests green.

Highlights of the optimization campaign:
- W8A8 int8 with per-16-channel-group scales (pg16), quant dispatch dedup
- LDS-tiled GEMM kernels for 1x1/3x3/stride2 convolutions
- int8 1x1 LTS: LDS bank-conflict padding + 4-OC/thread short-spatial variant
- int8 3x3 LTS: B-tile staged in LDS (32 oc x 9 tap x KB) + missing halo barrier fixed
- Negative results: stride-2 3x3 B-staging (occupancy-limited), concat 16-OC blocking
  (neutral on RTX 5060, kept as opt-in), 4-OC 1x1 LTS at 40x40 (model-dependent)
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

## AEXRT-M (Android / Vulkan)

The same graph/plan discipline targets phones: the `.AXM1` flat model is executed
by a Vulkan compute runtime on Dimensity 9500 (Mali-G925). Fixed-shape graphs are
recorded once, value placement is position-major with zero-copy concat/slice, and
a load-time autotuner picks the fastest conv variant on the actual device.

Host-verified on RTX 5060 (numerics vs ONNX Runtime: NRMSE <= 1e-3; GPU timestamps:
~1.6 ms for cs2V8_320 fp16). The same exporter and W8A8 contract also produce an
`.AXM3` model: 1.18 ms on RTX 5060 vs 2.11 ms fp16 (1.79x). On Mali-G925 the
shader-integer-dot throughput is below fp16 dot, so the device policy keeps fp16
(8.45 vs 5.0-5.5 ms measured): the format is shared, the execution strategy is
measured per device. `--emit-policy` persists the per-layer choice, removing the
~0.7 s startup autotune. `--pipeline 2` keeps two frame sets in flight and lifts
Mali-G925 throughput from ~121-136 to 188-192 fps with byte-identical
detections; `--stream` adds per-frame input staging (cached host buffer + one
GPU copy) and overlaps each frame's CPU NMS with the next frame's GPU work at
+2% per frame. A `--caps` probe shows Mali-G925 also exposes
`VK_KHR_cooperative_matrix` (fp16 16x32x32, int8 4x16x16); a SPIR-V generator
plus `--bench-coopmat` measures the matrix units at 19.2 TMAC/s on RTX 5060
(Mali pending), the next 2-4x lever (see `mobile/README.md`). An in-process library API (`aexrt_m_open/run/close` in
`mobile/runtime/aexrt_m.h`, built with `py mobile/tools/build.py api-host`)
exposes the same graph to an app; its detections are byte-identical to the CLI.
See [`mobile/README.md`](mobile/README.md) for the architecture, build/test
commands, negative results, and the device runbook.

```bash
py mobile/tools/build.py shaders
py mobile/tools/build.py android --api 35
py mobile/tools/build.py host
py mobile/tools/make_subsets.py
py mobile/tools/verify_host.py --heads build/tmp/heads_final
```

## Documentation

Bilingual (EN + 简体中文) — see the [full index](docs/README.md):

- Binary engine wire format — [EN](docs/AEXRT_ENGINE.md) / [中文](docs/AEXRT_ENGINE.zh-CN.md)
- Architecture — [EN](docs/ARCHITECTURE.en.md) / [中文](docs/ARCHITECTURE.md)
- Native C++ API — [EN](docs/CPP_API.md) / [中文](docs/CPP_API.zh-CN.md)
- Roadmap — [EN](docs/ROADMAP.en.md) / [中文](docs/ROADMAP.md)

## LLM Operators (Python backend)

LayerNorm, GELU, Softmax, Embedding, SDPA, RoPE available via
`InferenceSession(graph, backend="torch")`.

## License

[MIT](LICENSE)
