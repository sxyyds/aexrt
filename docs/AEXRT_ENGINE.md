# AEXRT Binary Engine V1

**English | [简体中文](AEXRT_ENGINE.zh-CN.md)**

`.aexrt` is the first-class deployable model format for the native AEXRT
runtime. The normal workflow is:

```text
model.onnx -> aexrtc -> model.aexrt -> native D3D12 runtime
```

No JSON package is generated or required. An engine contains the executable
command stream, raw constants, fixed kernel and physical-dispatch plans, packed
weights, the GPU memory arena plan, serialized fusion groups, and the shader/PSO
cache in one binary file. Current engines also carry the offline arena resource
transition and UAV-reuse barrier schedule.

## Build And Inspect

Install the local command once, then build or inspect an engine:

```powershell
py -3.11 -m pip install -e .
aexrtc build models\cs2V8_320.onnx -o examples\cs2V8_320.aexrt --precision fp16
aexrtc inspect examples\cs2V8_320.aexrt
```

The Python API exposes the same compiler:

```python
from aexrt import save_aexrt_engine_from_onnx

info = save_aexrt_engine_from_onnx(
    "models/cs2V8_320.onnx",
    "examples/cs2V8_320.aexrt",
)
```

Load the engine directly from Python or C++:

```python
from aexrt import NativeCppYoloModel

model = NativeCppYoloModel("examples/cs2V8_320.aexrt")
```

```cpp
#include "aexrt.hpp"

aexrt::Device device(0);
auto model = aexrt::load_engine(device, "examples\\cs2V8_320.aexrt");
```

## Container Layout

All integers and floats are little-endian. The file header and section table
are followed by 64-byte-aligned section payloads. Every section has its own
CRC32. Offsets in the section table are file-relative; offsets inside the raw
constant and packed-weight records are section-relative.

The 64-byte file header is:

| Offset | Type | Field | V1 value |
| ---: | --- | --- | --- |
| 0 | `char[8]` | magic | `AEXRTENG` |
| 8 | `u32` | engine version | `1` |
| 12 | `u32` | header size | `64` |
| 16 | `u32` | section count | `11` for current compiler output; `9` or `10` for legacy engines without section 11 |
| 20 | `u32` | flags | bit 0 must be set |
| 24 | `u64` | file size | exact byte length |
| 32 | `u64` | TOC offset | `64` |
| 40 | `u64` | TOC size | `section_count * 32` |
| 48 | `u8[16]` | reserved | zero |

Each 32-byte TOC entry is `<IIQQII>`:

| Field | Type | Meaning |
| --- | --- | --- |
| section type | `u32` | ID from the section table below |
| section flags | `u32` | zero in V1 |
| offset | `u64` | file-relative payload offset |
| size | `u64` | payload bytes |
| checksum | `u32` | CRC32 of exactly the payload bytes |
| reserved | `u32` | zero |

Container V1 requires the nine base semantic sections with IDs 1 through 9.
The current compiler also emits sections 10 and 11. Legacy V1 engines without
either optional section remain valid. Section 11 requires both Memory Plan V2
and section 10, while readers reject duplicate sections and validate every
section that is present.

| ID | Section | Purpose |
| ---: | --- | --- |
| 1 | Manifest | target, precision, YOLO metadata, counts, source hash |
| 2 | Values | dense value table and arena bindings |
| 3 | Commands | executable binary command stream |
| 4 | Raw constants | uncompressed tensor bytes |
| 5 | Kernel plan | authoritative kernel and precision selection |
| 6 | Packed weights | kernel-ready weight layouts |
| 7 | Memory plan | versioned activation pages, storage metadata, and per-value byte ranges |
| 8 | Fusion plan | authoritative logical fusion groups, kinds 1 through 6 |
| 9 | Pipeline cache | keyed DXIL/shader bytecode and driver PSO blobs |
| 10 | Physical dispatch plan | authoritative physical ownership, execution points, kernels, precision, and sparse logical-command membership |
| 11 | Arena barrier plan | authoritative per-dispatch page transitions and overlapping UAV-reuse ordering |

## Manifest

The manifest is 124 bytes: 17 `u32` values, two `u64` values, two `f32`
values, then a 32-byte SHA-256 digest.

| Index | Field | V1 value or meaning |
| ---: | --- | --- |
| 0 | manifest version | `1` |
| 1 | runtime target | `1` = native D3D12 |
| 2 | storage/plan precision | `1` = FP32, `2` = FP16 |
| 3 | mode | `2` = native graph with YOLO postprocess |
| 4 | output layout | `1` = channels first, `2` = channels last |
| 5 | objectness | `0` or `1` |
| 6-10 | output metadata | channels, anchors, classes, max candidates, max detections |
| 11-14 | graph counts | nodes, values, constants, commands |
| 15-16 | value IDs | graph input, graph output |

The remaining fields are `input_elements: u64`, `arena_nbytes: u64`,
`conf_threshold: f32`, `iou_threshold: f32`, and `source_sha256: u8[32]`.
When compiling from a file, the digest is the SHA-256 of the source ONNX bytes.

## Values And Commands

The values section starts with `<II>` (`count`, reserved zero). Each value is a
48-byte `<IIQ4IQQ>` record:

```text
id, flags, elements, shape[4], arena_offset, arena_nbytes
```

Value IDs are dense and records are ordered by ID. Shapes are normalized to
four dimensions. Value flags are bit 0 input, bit 1 constant, bit 2 arena, and
bit 3 alias. Non-arena values use `UINT64_MAX` as their arena offset and zero
as their arena size. Under Memory Plan V2, an arena offset is relative to the
value's page from section 7; the values record intentionally does not duplicate
the page ID. Section 7 must repeat the same offset and size.

The commands section also begins with `<II>` (`count`, reserved zero). Each
variable-length command has a 24-byte `<6I>` header followed by `input_count`
input IDs and `param_count` operation parameters, both as `u32` arrays:

```text
kind, output, input_count, param_count, planned_kernel, precision
```

Command kind IDs are:

| ID | Kind | ID | Kind |
| ---: | --- | ---: | --- |
| 1 | Conv | 8 | Resize |
| 2 | Conv + SiLU | 9 | MaxPool |
| 3 | Concat + Conv1x1 | 10 | Unary |
| 4 | View | 11 | Binary |
| 5 | Alias | 12 | MatMul |
| 6 | Slice | 13 | Transpose |
| 7 | Concat | 14 | Softmax |

Conv, Conv+SiLU, Concat+Conv1x1, and MatMul commands use the selected engine
precision. Other commands use FP32. The planned kernel and precision are
repeated in the kernel-plan section so the loader can reject an internally
inconsistent engine.

## Raw Constants

The section starts with `<II>` (`count`, reserved zero), followed by `count`
32-byte `<IIQQQ>` records:

```text
value_id, dtype, elements, data_offset, data_nbytes
```

`dtype=1` is FP32 and `dtype=2` is IEEE FP16. In an FP16 engine, Conv,
Concat+Conv1x1, and MatMul weights and biases are stored as FP16; constants
that are not consumed by those kernels remain FP32. Tensor payloads are raw
little-endian bytes, individually aligned to 64 bytes. There is no base64,
JSON, compression, or textual replay stream in an engine.

## Kernel Plan

The section header is `<II>` (`command_count`, `plan_version=1`). Every command
has one 24-byte `<6I>` plan record:

```text
command_index, kind, planned_kernel, precision, packed_value, flags
```

Bit 0 of `flags` marks the plan authoritative. Bit 1 requests DXIL for that
command's shader. Bit 2 declares an FP16 primary activation input and bit 3 an
FP16 activation output; these bits must agree with Memory Plan V2. When DXIL
is unavailable, a compatible DXBC kernel must preserve the same activation IO
types.
`packed_value` is the source weight value ID, or `UINT32_MAX` when no packed
weight is required. Conv and
Conv+SiLU kernel IDs are the stable `AexrtTileFlowConvAlgorithm` IDs declared by
the native runtime. Concat+Conv1x1 uses `0` for the generic path, `1` for the
direct pack4 path, `2` for GEMM, and `3` for native-FP16 OC8 x pos2 with packed
weights. Other V1 commands use kernel ID zero.

This section is the source of truth for engine execution. Kernel and precision
selection for `.aexrt` commands is not recomputed from runtime environment
switches. New kernels must receive stable IDs and be selected by the engine
compiler.

## Packed Weights

The section starts with `<II>` (`count`, reserved zero), followed by `count`
40-byte `<4IQQQ>` records:

```text
value_id, layout, in_channels, out_channels,
elements, data_offset, data_nbytes
```

Packed layouts are:

| ID | Storage | Logical packed shape |
| ---: | --- | --- |
| 1 | FP32 Conv3x3 OC4 | `[out_channels/4][in_channels][3][3][4]` |
| 2 | FP16 Conv3x3 OC4 | `[out_channels/4][in_channels][3][3][4]` |
| 3 | FP16 Winograd F(2x2,3x3) | `[out_channels][in_channels][4][4]` |
| 4 | FP16 Conv1x1 OC8 | `[out_channels/8][in_channels][8]` |
| 5 | FP32 Winograd F(2x2,3x3) | `[out_channels][in_channels][4][4]` |

The output-channel lane is innermost for layouts 1, 2, and 4. Layouts 3 and 5
store compiler-transformed 4x4 Winograd coefficients. Layout 5 preserves the
FP32 transform for kernels whose numerical audit rejects a second FP16
quantization. Packed payloads are aligned to 64 bytes.

## Memory Arena

Section 7 is versioned independently of the container. The current compiler
emits Memory Plan V2. The first `u32` is `2`; a different first value is parsed
as the legacy V1 alignment field.

### Memory Plan V2

The 32-byte header is `<4IQII>`:

```text
version=2, alignment, page_count, value_record_count,
total_nbytes, flags, reserved
```

`flags` must equal bit 0 (authoritative), `reserved` must be zero, and
`total_nbytes` must equal both the manifest arena size and the sum of all page
sizes. The native loader accepts at most 64 pages. Each page is one D3D12
resource and has a 24-byte `<4IQ>` record:

```text
page_id, storage_dtype, storage_layout, page_flags, nbytes
```

Page IDs are dense from zero, `page_flags` is currently `1`, and `nbytes` is a
nonzero multiple of the plan alignment. Storage enums are:

| Field | ID | Meaning |
| --- | ---: | --- |
| `storage_dtype` | 1 | FP32 |
| `storage_dtype` | 2 | IEEE FP16 |
| `storage_layout` | 1 | linear NCHW |
| `storage_layout` | 2 | blocked NCHW8 |

Blocked NCHW8 is valid only with FP16. A page is homogeneous in dtype and
layout so all views over that resource have compatible physical storage.

Page records are followed by 40-byte `<6IQQ>` value records:

```text
value_id, value_flags, page_id, storage_dtype, storage_layout,
reserved, offset, nbytes
```

The arena bit must be present in `value_flags`, the storage fields must match
the referenced page, `reserved` must be zero, and `offset` must be aligned.
The range `[offset, offset + nbytes)` must fit in the page. FP32 records require
`elements * 4` bytes and FP16 records require `elements * 2` bytes. The values
section mirrors `offset` and `nbytes`; section 7 adds the owning page and
physical storage type.

The compiler derives liveness from section 10's physical dispatch order, not
only from logical command order. It resolves inputs hidden inside a fused
record, keeps delayed inputs live until their physical consumer, rejects a
forward physical dependency, and holds the graph output through final
consumption. Page coloring prevents a dispatch output from sharing a resource
page with any of that dispatch's external inputs. Within a page, nonoverlapping
lifetimes may reuse an aligned byte slot. Best-fit versus exact-size reuse,
reuse gaps, and disabling reuse are compiler placement choices; they do not
change the V2 wire format.

`VIEW` and `ALIAS` outputs do not allocate independent slots. They resolve to
their ultimate source allocation, accumulate nested element offsets, and carry
an arena record that references the same page, dtype, and layout. Their byte
range is the typed subview of the canonical allocation, and their consumers
extend the canonical source lifetime. Alias and arena bits are both set on
these binding records. This lets the loader validate FP16 aliases before
prepared physical replay without materializing or copying the view. A
completely unreferenced declared value may still omit storage.

The native arena executor accepts FP32 or FP16 pages in linear NCHW layout.
It allocates 4-byte or 2-byte resources from the serialized dtype and creates
page-relative typed views after checking byte alignment and bounds. Blocked
NCHW8 remains reserved and is rejected until every producer and consumer has
a matching blocked-layout kernel.

### Legacy Memory Plan V1

The legacy header is `<IIQ>` (`alignment`, `value_count`, `total_nbytes`) and
is followed by 24-byte `<IIQQ>` records:

```text
value_id, value_flags, offset, nbytes
```

V1 describes one FP32 resource and has no page, dtype, or layout fields.

## Fusion Plan

The section begins with `<4I>`:

```text
plan_version=1, group_count, command_count, flags
```

Bit 0 of `flags` marks the plan authoritative. Every group is one 32-byte
`<8I>` record:

```text
kind, start_command, end_command, precision,
kernel, flags, aux0, aux1
```

Group kinds are `1` paired Conv3x3+SiLU, `2` C2F tail residual, `3` late
Concat+Conv1x1, `4` YOLO head decode, `5` YOLO final-conv head decode,
`6` stride2 Conv3x3+SiLU followed by Conv1x1+SiLU, and `9` the six-command
neck `LatticeChain C3`. Kind 5 remains reserved and is not emitted by the
current compiler. Kinds 7 and 8 are physical-only ownership forms and are not
valid in section 8.

Kind 9 owns `CONCAT -> 1x1 -> 1x1 -> 3x3`, the parallel entry `1x1`, and the
final `CONCAT_CONV1X1`. It uses kernel field `0`, precision `2`, and flags
`AUTHORITATIVE | DXIL` without FP16 activation-I/O bits. `aux0` packs
`spatial | (hidden_channels << 16)` and `aux1` packs
`entry_channels | (output_channels << 16)`. The dimensions and complete value
flow are validated offline; runtime model names and environment variables do
not select this path. One kind-9 physical record expands to a fixed two-node
micro-DAG: phase A performs virtual Concat plus reduce/body1/branch 1x1, and
phase B performs the spatial 3x3 plus final merge 1x1. Thus each of the three v5
neck C3 owners emits two GPU dispatches, reducing their 18 logical dispatches to
6 without recomputing the 1x1 stages.

Section 8 remains the logical fusion description and the compatibility source
for engines without section 10. When section 10 is present, section 8 is
cross-validated while compiling typed physical operations; recording then
follows section 10 directly.

## Physical Dispatch Plan V2

Section 10 is optional at the container level for legacy compatibility, but it
is emitted by the current compiler. Its 32-byte `<8I>` header is:

```text
plan_version=2, record_count, logical_command_count, plan_flags,
record_size=48, logical_index_count, logical_index_table_offset, reserved
```

`plan_flags` must equal bit 0 (authoritative), `reserved` must be zero,
`logical_command_count` must equal the section 3 command count, and
`logical_index_count` must also equal that command count. The index table starts
exactly at `32 + record_count * 48`, and no trailing bytes are allowed.

Each 48-byte record is `<Q10I>`:

| Field | Type | Meaning |
| --- | --- | --- |
| `stable_id` | `u64` | nonzero deterministic identity of this physical record |
| `logical_start` | `u32` | first owned logical command index |
| `logical_end` | `u32` | last owned logical command index |
| `execution_index` | `u32` | logical position at which the physical dispatch executes |
| `logical_index_offset` | `u32` | element offset in the trailing `u32` index table |
| `logical_index_count` | `u32` | number of logical commands owned by this record |
| `kernel` | `u32` | serialized kernel ID for the physical dispatch |
| `precision` | `u32` | `1` FP32 or `2` FP16 |
| `flags` | `u32` | bit 0 authoritative; bit 1 requires DXIL; bit 2 reads FP16 activation; bit 3 writes FP16 activation |
| `fusion_kind` | `u32` | `0` unfused, or one of the physical fusion kinds below |
| `reserved` | `u32` | zero |

The trailing table is the concatenation of every record's sorted logical
indices. Each logical command appears exactly once across all records. Record
offsets are contiguous, stable IDs are unique, and execution indices are
strictly increasing and must belong to their own record. For kind 0 the record
contains only its execution index; a fused record contains at least two
indices. `logical_start` and `logical_end` are the first and last table entries,
not a claim that every index between them belongs to the record. This permits a
physical dispatch to own sparse logical commands while unrelated commands in
the numeric gap remain separate.

The compiler forms `stable_id` with a personalized 64-bit BLAKE2b digest over
the physical record selection and the full signatures of its owned logical
commands, including command kind, output, inputs, and parameters. It therefore
stays stable across identical builds and changes when the physical command
signature changes.

Physical fusion kinds are:

| ID | Kind |
| ---: | --- |
| 0 | one unfused logical command |
| 1 | paired Conv3x3+SiLU |
| 2 | C2F tail residual |
| 3 | late Concat+Conv1x1 |
| 4 | YOLO head decode |
| 5 | YOLO final-conv head decode |
| 6 | stride2 Conv3x3+SiLU followed by Conv1x1+SiLU |
| 7 | `CONCAT_RESIDUAL_CV2` |
| 8 | position-owned Winograd residual + CV2 |
| 9 | six-command neck `LatticeChain C3` |

Kind 7 owns a `CONCAT_CONV1X1` command and one or more sparse `BINARY Add`
producers whose outputs feed its branches and have no other consumer. It
executes at the Concat/CV2 command index. A residual Add already owned by a C2F
tail record is excluded. The native kernel reads each selected Add's primary
and residual inputs directly, performs the residual sum while packing branch
channels, and applies CV2 1x1 (and SiLU when requested) without dispatching or
materializing those standalone Adds. Its kernel, precision, and flags come from
the owned `CONCAT_CONV1X1` plan. The runtime may apply this fusion only to Add
indices explicitly listed in the kind 7 record.

The native executor compiles section 10 into typed prepared operations and
records those operations in physical-record order. Kind 0 calls the unfused
logical command recorder; kinds 1, 2, 3, 4, 6, 7, 8, and 9 call their fused
recorders with prevalidated payloads. Matchers run only during this compile
step to prove exact ownership. Kind 5 is rejected until implemented. An
unencoded fusion remains unfused even if the legacy logical matcher could
discover it.

For kind 9, only the final `CONCAT_CONV1X1` output is materialized in the engine
arena. The five logical intermediate outputs have no Memory Plan V2 record and
create no section 11 write footprint. Phase A writes two reusable
physical-private FP32 scratch planes for body1 and branch; their UAV-to-SRV
transitions are the sole phase boundary. Phase B uses horizontally paired
Winograd tiles at 20x20 and a four-tile direct bundle at 40x40, then applies the
merge with K8 FP16 partials and FP32 block accumulation. Both scratch resources
return to COMMON at the end of the prepared command list so replayed lists have
stable before/after states. The native loader accepts the omitted logical
values only when section 10 proves that each is internal to the same validated
kind-9 owner.

A singleton kind-0 record that owns only `VIEW` or `ALIAS` remains in section
10 so logical command coverage and stable IDs stay complete. The compiler DAG
marks it `zero_dispatch`, excludes it from material producer/consumer edges,
and connects later users directly to the canonical producer. Native replay
only creates or validates the typed buffer view; no GPU dispatch is emitted.

## Arena Barrier Plan V1

Section 11 moves arena synchronization out of runtime command/value pattern
matching. Its 32-byte `<8I>` header is:

```text
plan_version=1, plan_flags, physical_dispatch_count, page_count,
record_count, record_size=40, reserved0, reserved1
```

`plan_flags` must equal bit 0 (authoritative), both reserved fields are zero,
the dispatch count must equal section 10, and the page count must equal Memory
Plan V2. No trailing bytes are allowed. Each 40-byte `<6IQQ>` record is:

```text
dispatch_index, page_id, kind, before_value_id, after_value_id,
reserved, offset, nbytes
```

Records are strictly ordered by `(dispatch_index, page_id)`, so at most one
arena action exists for a page at a physical dispatch. `reserved` is zero and
the reason range `[offset, offset + nbytes)` must be nonempty and contained by
the referenced page. Action kinds are:

| Kind | Action | Value fields |
| ---: | --- | --- |
| 1 | transition the page to non-pixel-shader SRV | `before_value_id = UINT32_MAX`; `after_value_id` is the first read witness |
| 2 | transition the page to UAV | `before_value_id = UINT32_MAX`; `after_value_id` is the first write witness |
| 3 | UAV barrier for overlapping reuse while the page remains UAV | before/after identify the prior and next write witnesses |

The compiler resolves aliases and materialized outputs on section 10's
physical timeline, then simulates each page from `COMMON`. A state transition
clears pending UAV writes. A kind 3 record is emitted only when a later write
overlaps a pending half-open byte range and the resource has remained UAV.
Disjoint writes to separate slots of one page produce no action, and a prior
UAV-to-SRV transition already provides ordering, so no redundant UAV barrier is
serialized. Page coloring separately guarantees that a dispatch never needs
one page as both SRV and UAV.

Pure alias records have no material reads or writes and therefore never create
a section 11 action. A real consumer reached through one or more aliases uses
the canonical page directly in the offline access schedule.

The native loader independently rebuilds this schedule from sections 3, 7, and
10 and requires an exact record-for-record match. Missing actions, extra
actions, wrong witnesses, wrong ranges, and dispatch/page count drift fail
closed. During prepared replay the runtime executes section 11 immediately
before each physical dispatch. Arena transition calls inside kernel recorders
then act only as consistency checks; they cannot infer or insert an unplanned
arena barrier. Non-arena scratch resources retain their local synchronization.

Legacy engines without section 11 keep the previous byte-range-aware runtime
fallback. The current compiler always writes section 11, including a valid
header when no kind 3 action is necessary.

## Pipeline Cache

The section begins with `<4I>`:

```text
cache_version=1, shader_count, pso_count, flags
```

It is followed by `dxil_count + pso_count` 32-byte `<IIQQQ>` records:

```text
kind, flags, 64-bit key, payload_offset, payload_nbytes
```

Kind `1` is compiled shader bytecode and kind `2` is the driver-provided cached
PSO blob. For shader records, bit 0 of the record flags identifies DXIL; a
clear bit identifies DXBC. Payload offsets are section-relative and payloads
are 64-byte aligned. The cache key covers shader source, entry point, target,
macros, and compile flags; PSO keys cover the shader bytecode and pipeline
flags.

`aexrtc build` performs one zero-input prepared replay by default, exports all
created shader and PSO entries, and rebuilds section 9 with fresh CRC and TOC
metadata. `--no-pipeline-cache` leaves a valid empty cache section. On load,
matching entries are consumed before compilation and `CreateComputePipelineState`.

## Validation And Evolution

The native loader rejects unsupported versions, malformed sizes or IDs,
missing mandatory sections, CRC mismatches, inconsistent command/kernel plans,
invalid fusion ranges, invalid cache records, and arena ranges outside the
declared allocation. The source hash identifies which ONNX input produced the
engine; it is not a replacement for section integrity checks.

FP16 storage metadata, fusion ownership, shader targets, and DXIL/PSO caches
are engine data. With section 10 present, runtime pattern matching cannot add
physical ownership that the engine did not serialize. Device capability or PSO
creation failure may still select a compatible fallback with the same physical
IO dtype for an authorized command, but it does not rewrite the physical plan.

Raw FP16 constants are decoded into the runtime's FP32 CPU mirror and generic
constant buffers. Packed-weight layouts 2 through 4 remain FP16 and are
consumed directly by the native Shader Model 6.2 paths selected by the kernel
plan. Kernel ID 46 uses layout 5 for the measured v5 256-to-256, 10x10
standalone Winograd layers: the compiler transforms the FP16 source weights,
stores the transformed coefficients as FP32, and selects FP32 input transform
and accumulation without runtime weight conversion. The measured activation
island stores one v5 40x40 intermediate as FP16 linear NCHW between two 1x1
SiLU commands; conversion is fused into the producer and consumer and adds no
dispatch.

The v5 1x1 plan also selects native-FP16 kernel ID 45 for the measured
128-to-64 and 256-to-64 shapes at 40x40, and the 128-to-128, 256-to-128, and
512-to-128 shapes at 20x20. Kernel ID 3 covers the measured two-branch
`(32,32)`-to-64 Concat at 80x80 and `(64,64)`-to-128 Concat at 40x40; the same
serialized command plan is inherited by physical fusion kind 7 when it owns a
residual Add.

The v11 plan selects kernel ID 33 for the measured 128-to-128, 80x80-to-40x40
stride-2 Conv3x3. Native-FP16 kernel ID 45 is selected for exact ordinary 1x1
shapes 128-to-128 at 40x40, 384-to-256 at 20x20, and 512-to-512 at 10x10.
Kernel ID 3 is selected for the 40x40 three-branch `(64,64,64)` Concat producing
128 channels, and for the 20x20 three-branch `(128,128,128)` Concat producing
256 channels only when physical fusion kind 7 owns a residual Add. The existing
512-to-128, 40x40 two-branch Concat rule remains enabled. Higher-error front
shapes and measured losing 1x1 shapes remain on their SM5 kernels. These are
exact shape and physical-ownership rules in the engine compiler, not model-ID
or environment-variable branches.

If a binary engine has both Memory Plan V2 and section 10, failure to compile
or prepare the authoritative physical replay returns an execution error. It
does not fall through to logical GPU or CPU replay, because neither fallback
can promise the serialized physical lifetimes, dtypes, and reused page slots.
