# Native D3D12 Optimization Log — Executive Summary (English)

**[完整中文日志](NATIVE_D3D12.md) | English summary**

The full 38-round log (kernel rewrites, quantization systems, correctness
fixes, negative results) is written in Chinese. This page summarizes what
matters in English.

## Final Results (RTX 5060 Laptop GPU, 320x320 YOLO, boost clocks)

| Model | e2e latency | vs baseline | Correctness |
|---|---|---|---|
| cs2V8_320 | **1.08 ms** | **-63%** | 8/8 images match ORT-CPU |
| apex10w | 1.79 ms | -68% | 8/8 |
| DYv11s | 2.50 ms | -62% | 8/8 |

Engine-core latency (submit + GPU + readback, zero-copy input): ~1.0 ms.
300 unit tests green throughout.

## What Was Built

- **W8A8 int8 pipeline**: producer-side pre-quantization, per-16-channel-
  group activation scales ("pg16": int chains within a group, one float
  conversion between groups — same speed as per-tensor, half the error),
  recording-time quant dispatch dedup.
- **LDS-tiled GEMM kernels** for 1x1 / 3x3 / stride-2 convolutions.
- **amax grid-collapse fix** — the quantizer's per-group workgroup mapping
  collapsed to 1-2 workgroups on low-channel/large-spatial inputs;
  re-engineering it to block-split strips + atomic max delivered
  -21~-42% in a single round.
- **Zero-copy input** via a persistent-mapped upload buffer.
- **Fusion surgeries**: unfusing C2F-tail, concat-conv and head-interior
  locks so int8 could claim the convs (each guarded by accuracy checks).
- **Dual-compute-queue machinery** (works, numerically proven, economically
  rejected — queue-switch cost exceeds parallel gain at current scale).

## Established Laws (negative results, each measured)

1. **Register cliff**: > 8 int accumulators per thread always loses on this
   GPU, regardless of source (oc-doubling, 4x unrolls, 2-position blocking —
   four independent confirmations).
2. **LDS tiling pays only when reuse x K-depth is high**: 1x1 weights (32x
   reuse) win big; 3x3 s1 (9x) neutral; stride2 (2.25x) needs K >= 64.
3. **Un-fusion trades**: materializing an intermediate + quantizing it can
   cost more than the fused kernel saves — always measure per shape.
4. **Temporal amax reuse** (frame N quantized with frame N-1 scale) fails
   the accuracy gate on unrelated images (2-4x per-layer error growth);
   safe for correlated game frames but unprovable in this harness.
5. GPU time is ~92% of wall — CPU-side work is already negligible; kernel
   efficiency and fixed overheads are the only levers.

## Timing Breakdown (zero-copy, boost state)

wall p1 ≈ fence ~1010 μs + submit ~13 μs + readback ~1 μs + harness ~46 μs.
memcpy is zero (input written directly into the mapped upload buffer —
this is also the real GPU-capture deployment path).

## Remaining Identified Levers

Producer-side amax atomics (~-25 μs, shader side already written), LTS-3x3/
s2 epilogues (~-10 μs each), temporal quantization (deployment-state safe,
~-60 μs). All documented with designs in the Chinese log.
