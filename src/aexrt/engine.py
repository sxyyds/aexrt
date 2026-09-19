from __future__ import annotations

import base64
import hashlib
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .barrier_v2 import (
    ARENA_BARRIER_KIND_TRANSITION_SRV,
    ARENA_BARRIER_KIND_TRANSITION_UAV,
    ARENA_BARRIER_KIND_UAV_REUSE,
    build_arena_barrier_plan,
    encode_arena_barrier_plan,
    inspect_arena_barrier_plan,
)
from .graph import Graph
from .memory_v2 import (
    LAYOUT_LINEAR_NCHW,
    build_page_colored_arena,
    build_physical_value_dag,
    encode_memory_plan_v2,
    inspect_memory_plan_v2,
)
from .yolo import build_yolo_native_d3d12_graph_package


ENGINE_MAGIC = b"AEXRTENG"
ENGINE_VERSION = 1
ENGINE_HEADER_SIZE = 64
ENGINE_TOC_ENTRY_SIZE = 32
ENGINE_ALIGNMENT = 64
ARENA_ALIGNMENT = 256

SECTION_MANIFEST = 1
SECTION_VALUES = 2
SECTION_COMMANDS = 3
SECTION_CONSTANTS = 4
SECTION_KERNEL_PLAN = 5
SECTION_PACKED_WEIGHTS = 6
SECTION_MEMORY_PLAN = 7
SECTION_FUSION_PLAN = 8
SECTION_PIPELINE_CACHE = 9
SECTION_PHYSICAL_DISPATCH_PLAN = 10
SECTION_ARENA_BARRIER_PLAN = 11
SECTION_PARALLEL_PLAN = 12

PHYSICAL_DISPATCH_PLAN_VERSION = 2
PHYSICAL_DISPATCH_PLAN_HEADER_SIZE = 32
PHYSICAL_DISPATCH_RECORD_SIZE = 48
PHYSICAL_DISPATCH_PLAN_FLAG_AUTHORITATIVE = 1 << 0

PRECISION_FLOAT32 = 1
PRECISION_FLOAT16 = 2
PACKED_LAYOUT_CONV3X3_OC4 = 1
PACKED_LAYOUT_CONV3X3_OC4_FP16 = 2
PACKED_LAYOUT_WINOGRAD_F2X2_FP16 = 3
PACKED_LAYOUT_CONV1X1_OC8_FP16 = 4
PACKED_LAYOUT_WINOGRAD_F2X2_FP32 = 5
PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16 = 6
# Reserved for inspection compatibility with rejected kernel50 probe engines.
# The planner/writer must not emit this layout.
PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP32 = 7
# INT8 dot4 权重镜像：负载 = [oc*ic*kk 个 int8（[oc][ic][kk] 原序）][oc 个 fp32 scale]。
PACKED_LAYOUT_CONV_INT8 = 8
# Winograd F(4x4,3x3) 权重镜像：fp16 变换域 U = G·g·Gᵀ（[oc][ic][36]）。
PACKED_LAYOUT_WINOGRAD_F4X4_FP16 = 9
NO_OFFSET = (1 << 64) - 1
NO_VALUE = (1 << 32) - 1

FUSION_PAIRED_CONV3X3_SILU = 1
FUSION_C2F_TAIL_RESIDUAL = 2
FUSION_LATE_CONCAT_CONV1X1 = 3
FUSION_YOLO_HEAD = 4
FUSION_YOLO_HEAD_FINAL_CONV = 5
FUSION_STRIDE2_CONV1X1 = 6
FUSION_CONCAT_RESIDUAL_CV2 = 7
FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2 = 8
FUSION_LATTICE_CHAIN_C3 = 9

PLAN_FLAG_AUTHORITATIVE = 1 << 0
PLAN_FLAG_DXIL = 1 << 1
PLAN_FLAG_INPUT_FP16 = 1 << 2
PLAN_FLAG_OUTPUT_FP16 = 1 << 3
PLAN_FLAG_INT8_PERGROUP = 1 << 4
PLAN_FLAG_MASK = (
    PLAN_FLAG_AUTHORITATIVE
    | PLAN_FLAG_DXIL
    | PLAN_FLAG_INPUT_FP16
    | PLAN_FLAG_OUTPUT_FP16
    | PLAN_FLAG_INT8_PERGROUP
)

VALUE_INPUT = 1 << 0
VALUE_CONSTANT = 1 << 1
VALUE_ARENA = 1 << 2
VALUE_ALIAS = 1 << 3

COMMAND_KIND = {
    "CONV": 1,
    "CONV_SILU": 2,
    "CONCAT_CONV1X1": 3,
    "VIEW": 4,
    "ALIAS": 5,
    "SLICE": 6,
    "CONCAT": 7,
    "RESIZE": 8,
    "MAXPOOL": 9,
    "UNARY": 10,
    "BINARY": 11,
    "MATMUL": 12,
    "TRANSPOSE": 13,
    "SOFTMAX": 14,
}


def build_aexrt_engine(
    graph: Graph,
    *,
    source_model: str | None = None,
    output_name: str | None = None,
    classes: int | None = None,
    layout: str = "auto",
    objectness: bool | None = None,
    max_candidates: int = 512,
    max_detections: int = 100,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    precision: str = "fp16",
    pipeline_cache: bytes | None = None,
    arena_alignment: int = 256,
    arena_reuse_slots: bool = True,
    arena_reuse_gap: int = 0,
    arena_reuse_exact_size: bool = False,
    parallel_plan: bool = False,
    int8_pergroup: Sequence[int] | None = None,
    fp16_activation_values: Sequence[int] | None = None,
    kernel_plan_seed: Sequence[Sequence[int]] | None = None,
    algo_overrides: Any = None,
) -> bytes:
    precision_id = _precision_id(precision)
    package = build_yolo_native_d3d12_graph_package(
        graph,
        source_model=source_model,
        output_name=output_name,
        classes=classes,
        layout=layout,
        objectness=objectness,
        max_candidates=max_candidates,
        max_detections=max_detections,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        embed_constants=True,
    )
    replay = _parse_replay_text(str(package["cxx_replay_text"]))
    yolo = package["yolo"]
    kernel_plan = _build_kernel_plan(replay, precision=precision_id)
    # 融合计划先行：algo override 必须知道命令是否在融合组内
    # （组内命令的 kernel 由融合组决定，单独 override 会产出物理计划
    # 失配的引擎——与运行时 conv_autotune 的 in_fusion_group 防御一致）。
    fusion_plan = _build_fusion_plan(replay, precision=precision_id, classes=int(yolo["classes"]))
    if algo_overrides:
        # per-device autotune 结果固化（conv_autotune --emit-map 产物）。
        _apply_algo_overrides(replay, kernel_plan, algo_overrides, fusion_plan=fusion_plan)
    import os as _os
    if not _os.environ.get("AEXRTC_NO_MEASURED_FP16"):
        _apply_measured_fp16_physical_kernels(
            replay,
            kernel_plan,
            fusion_plan,
            precision=precision_id,
        )
    physical_dispatch_plan = _build_physical_dispatch_plan(replay, kernel_plan, fusion_plan)
    if _apply_measured_fp16_c3_topology_kernels(
        replay,
        kernel_plan,
        physical_dispatch_plan,
        precision=precision_id,
    ):
        physical_dispatch_plan = _build_physical_dispatch_plan(
            replay, kernel_plan, fusion_plan
        )
    if kernel_plan_seed is not None:
        _apply_kernel_plan_seed(replay, kernel_plan, kernel_plan_seed)
        physical_dispatch_plan = _build_physical_dispatch_plan(
            replay, kernel_plan, fusion_plan
        )
    # Per-group int8 variant selection (round 20): commands listed here use
    # the 4-channel-group activation scale kernels (accuracy escape hatch).
    if int8_pergroup:
        pg_commands = {int(i) for i in int8_pergroup}
        for entry in kernel_plan:
            if entry["command_index"] in pg_commands and entry["planned_kernel"] in {51, 52, 53}:
                entry["flags"] |= 16
                pg_commands.discard(entry["command_index"])
        # Physical dispatch records must carry the flag too (section 10
        # coherence check compares kernel plan vs dispatch flags). Only plain
        # int8 records: a fusion group consuming a pg command runs its own
        # fused kernel and must not inherit the variant bit.
        for record in physical_dispatch_plan:
            if int(record.get("fusion_kind", 0)) != 0:
                continue
            if int(record.get("kernel", 0)) not in {51, 52, 53}:
                continue
            if any(int(li) in {int(i) for i in int8_pergroup} for li in record.get("logical_indices", ())):
                record["flags"] = int(record["flags"]) | 16
    fp16_activation_storage = _plan_fp16_activation_islands(
        replay,
        kernel_plan,
        physical_dispatch_plan,
        fusion_plan,
        precision=precision_id,
        selected_value_ids=fp16_activation_values,
    )
    # Dual-queue parallel window (round 17, opt-in): plain conv ops only; the
    # page allocator separates opposite-side arena values so page barrier
    # plans never interleave across queues. Off by default - the runtime path
    # is experimental and the engine stays byte-identical without it.
    dual_window = _plan_dual_queue_window(replay, physical_dispatch_plan) if parallel_plan else None
    parallel_side_values = None
    if dual_window is not None:
        begin, end, side = dual_window
        parallel_side_values = {}
        for i, record in enumerate(physical_dispatch_plan):
            if begin <= i < end:
                for k in record.get("logical_indices", ()):
                    if not 0 <= k < len(replay["commands"]):
                        continue
                    command = replay["commands"][k]
                    # Outputs carry the producing side. Inputs inherit the
                    # first side that reads them: a window input shared with a
                    # secondary-side output must not reuse its arena slot
                    # (the B write would race the A/B reads of that input).
                    parallel_side_values[int(command["output"])] = (
                        1 if side[i] else 0
                    )
                    for raw_input in command["inputs"]:
                        parallel_side_values.setdefault(int(raw_input), 1 if side[i] else 0)
    memory_plan = build_page_colored_arena(
        replay,
        physical_dispatch_plan=physical_dispatch_plan,
        storage=fp16_activation_storage,
        alignment=arena_alignment,
        reuse_slots=arena_reuse_slots,
        reuse_gap=arena_reuse_gap,
        reuse_exact_size=arena_reuse_exact_size,
        parallel_side_values=parallel_side_values,
    )
    arena_barrier_plan = build_arena_barrier_plan(
        replay,
        physical_dispatch_plan,
        memory_plan,
        parallel_window=dual_window,
    )
    packed_weights = _build_packed_weights(
        replay, kernel_plan, precision=precision_id, fusion_plan=fusion_plan
    )
    source_hash = _source_hash(source_model, replay)

    layout_id = 2 if str(yolo["layout"]) == "channels_last" else 1
    sections = {
        SECTION_VALUES: _encode_values(replay, memory_plan),
        SECTION_COMMANDS: _encode_commands(replay, kernel_plan),
        SECTION_CONSTANTS: _encode_constants(replay, precision=precision_id),
        SECTION_KERNEL_PLAN: _encode_kernel_plan(kernel_plan),
        SECTION_PACKED_WEIGHTS: _encode_packed_weights(packed_weights),
        SECTION_MEMORY_PLAN: encode_memory_plan_v2(memory_plan),
        SECTION_FUSION_PLAN: _encode_fusion_plan(fusion_plan, len(replay["commands"])),
        SECTION_PIPELINE_CACHE: pipeline_cache if pipeline_cache is not None else _encode_pipeline_cache([], []),
        SECTION_PHYSICAL_DISPATCH_PLAN: _encode_physical_dispatch_plan(
            physical_dispatch_plan, len(replay["commands"])
        ),
        SECTION_ARENA_BARRIER_PLAN: encode_arena_barrier_plan(arena_barrier_plan),
    }
    if parallel_plan:
        sections[SECTION_PARALLEL_PLAN] = _encode_parallel_plan(
            dual_window, len(physical_dispatch_plan)
        )
    sections[SECTION_MANIFEST] = _encode_manifest(
        layout=layout_id,
        has_objectness=bool(yolo["objectness"]),
        channels=int(yolo["channels"]),
        anchors=int(yolo["anchors"]),
        classes=int(yolo["classes"]),
        max_candidates=int(yolo["max_candidates"]),
        max_detections=int(yolo["max_detections"]),
        graph_node_count=len(graph.nodes),
        value_count=len(replay["values"]),
        constant_count=sum(bool(v["flags"] & VALUE_CONSTANT) for v in replay["values"]),
        command_count=len(replay["commands"]),
        input_value=int(replay["input_value"]),
        output_value=int(replay["output_value"]),
        input_elements=int(replay["values"][replay["input_value"]]["elements"]),
        arena_nbytes=int(memory_plan["total_nbytes"]),
        conf_threshold=float(yolo["conf_threshold"]),
        iou_threshold=float(yolo["iou_threshold"]),
        source_hash=source_hash,
        precision=precision_id,
    )
    return _build_container(sections)


def save_aexrt_engine(
    graph: Graph,
    path: str | os.PathLike[str],
    **kwargs: Any,
) -> dict[str, Any]:
    output = Path(path)
    if output.suffix.lower() != ".aexrt":
        raise ValueError("AEXRT engine output must use the .aexrt extension")
    data = build_aexrt_engine(graph, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return inspect_aexrt_engine(data)


def save_aexrt_engine_from_onnx(
    model_path: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    default_batch: int | None = 1,
    algo_overrides: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    from .onnx_importer import load_onnx

    model_path_s = os.fspath(model_path)
    graph = load_onnx(model_path_s, default_batch=default_batch) if default_batch != 1 else load_onnx(model_path_s)
    return save_aexrt_engine(graph, path, source_model=model_path_s, algo_overrides=algo_overrides, **kwargs)


def inspect_aexrt_engine(source: bytes | bytearray | memoryview | str | os.PathLike[str]) -> dict[str, Any]:
    data = bytes(source) if not isinstance(source, (str, os.PathLike)) else Path(source).read_bytes()
    if len(data) < ENGINE_HEADER_SIZE:
        raise ValueError("truncated AEXRT engine header")
    magic, version, header_size, section_count, flags, file_size, toc_offset, toc_size, _ = struct.unpack_from(
        "<8sIIIIQQQ16s", data, 0
    )
    if magic != ENGINE_MAGIC or version != ENGINE_VERSION or header_size != ENGINE_HEADER_SIZE:
        raise ValueError("unsupported AEXRT engine")
    if file_size != len(data) or toc_size != section_count * ENGINE_TOC_ENTRY_SIZE:
        raise ValueError("invalid AEXRT engine size")
    if toc_offset + toc_size > len(data):
        raise ValueError("truncated AEXRT section table")
    sections: dict[int, dict[str, int]] = {}
    for index in range(section_count):
        entry = toc_offset + index * ENGINE_TOC_ENTRY_SIZE
        section_type, section_flags, offset, size, checksum, reserved = struct.unpack_from("<IIQQII", data, entry)
        if reserved != 0 or offset + size > len(data):
            raise ValueError("invalid AEXRT section")
        if section_type in sections:
            raise ValueError("duplicate AEXRT section")
        payload = data[offset : offset + size]
        if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
            raise ValueError(f"AEXRT section {section_type} checksum mismatch")
        sections[section_type] = {
            "flags": section_flags,
            "offset": offset,
            "size": size,
            "checksum": checksum,
        }
    required = {
        SECTION_MANIFEST,
        SECTION_VALUES,
        SECTION_COMMANDS,
        SECTION_CONSTANTS,
        SECTION_KERNEL_PLAN,
        SECTION_PACKED_WEIGHTS,
        SECTION_MEMORY_PLAN,
        SECTION_FUSION_PLAN,
        SECTION_PIPELINE_CACHE,
    }
    if not required.issubset(sections):
        raise ValueError("AEXRT engine is missing required sections")
    manifest = _inspect_manifest(data, sections[SECTION_MANIFEST])
    command_kinds = _inspect_command_kinds(
        data, sections[SECTION_COMMANDS], manifest["command_count"]
    )
    manifest["command_kind_ids"] = command_kinds
    manifest["kernel_plan_count"] = struct.unpack_from("<I", data, sections[SECTION_KERNEL_PLAN]["offset"])[0]
    packed_info = _inspect_packed_weights(data, sections[SECTION_PACKED_WEIGHTS])
    fusion_plan = _inspect_fusion_plan(data, sections[SECTION_FUSION_PLAN], manifest["command_count"])
    cache_info = _inspect_pipeline_cache(data, sections[SECTION_PIPELINE_CACHE])
    manifest["packed_weight_count"] = packed_info["count"]
    manifest["packed_layout_counts"] = packed_info["layout_counts"]
    manifest["fusion_group_count"] = len(fusion_plan)
    manifest["fusion_plan"] = fusion_plan
    manifest["shader_cache_count"] = cache_info["shader_count"]
    manifest["dxil_cache_count"] = cache_info["dxil_count"]
    manifest["dxbc_cache_count"] = cache_info["dxbc_count"]
    manifest["pso_cache_count"] = cache_info["pso_count"]
    manifest["pipeline_cache_flags"] = cache_info["flags"]
    manifest["pipeline_cache_records"] = cache_info["records"]
    constant_precision_counts = _inspect_constants(data, sections[SECTION_CONSTANTS])
    if sum(constant_precision_counts.values()) != manifest["constant_count"]:
        raise ValueError("invalid AEXRT constant count")
    manifest["constant_precision_counts"] = constant_precision_counts
    memory_section = sections[SECTION_MEMORY_PLAN]
    memory_offset = memory_section["offset"]
    memory_size = memory_section["size"]
    memory_version_or_alignment = struct.unpack_from("<I", data, memory_offset)[0]
    if memory_version_or_alignment == 2:
        memory_info = inspect_memory_plan_v2(
            data[memory_offset : memory_offset + memory_size]
        )
        arena_slots = {
            (record["page_id"], record["offset"])
            for record in memory_info["records"]
            if not (int(record["flags"]) & VALUE_ALIAS)
        }
        material_records = [
            record
            for record in memory_info["records"]
            if not (int(record["flags"]) & VALUE_ALIAS)
        ]
        alias_records = [
            record
            for record in memory_info["records"]
            if int(record["flags"]) & VALUE_ALIAS
        ]
        manifest["memory_plan_version"] = 2
        manifest["arena_page_count"] = len(memory_info["pages"])
        manifest["arena_page_nbytes"] = [page["nbytes"] for page in memory_info["pages"]]
        manifest["arena_value_count"] = len(memory_info["records"])
        manifest["arena_material_value_count"] = len(material_records)
        manifest["arena_alias_value_count"] = len(alias_records)
        manifest["arena_slot_count"] = len(arena_slots)
        manifest["arena_storage_precision_counts"] = {
            "fp32": sum(record["storage_dtype"] == PRECISION_FLOAT32 for record in memory_info["records"]),
            "fp16": sum(record["storage_dtype"] == PRECISION_FLOAT16 for record in memory_info["records"]),
        }
        manifest["arena_material_storage_precision_counts"] = {
            "fp32": sum(record["storage_dtype"] == PRECISION_FLOAT32 for record in material_records),
            "fp16": sum(record["storage_dtype"] == PRECISION_FLOAT16 for record in material_records),
        }
        manifest["arena_alias_storage_precision_counts"] = {
            "fp32": sum(record["storage_dtype"] == PRECISION_FLOAT32 for record in alias_records),
            "fp16": sum(record["storage_dtype"] == PRECISION_FLOAT16 for record in alias_records),
        }
    else:
        arena_value_count = struct.unpack_from("<I", data, memory_offset + 4)[0]
        arena_offsets = {
            struct.unpack_from("<Q", data, memory_offset + 16 + index * 24 + 8)[0]
            for index in range(arena_value_count)
        }
        manifest["memory_plan_version"] = 1
        manifest["arena_page_count"] = 1
        manifest["arena_page_nbytes"] = [manifest["arena_nbytes"]]
        manifest["arena_value_count"] = arena_value_count
        manifest["arena_material_value_count"] = arena_value_count
        manifest["arena_alias_value_count"] = 0
        manifest["arena_slot_count"] = len(arena_offsets)
        manifest["arena_storage_precision_counts"] = {"fp32": arena_value_count, "fp16": 0}
        manifest["arena_material_storage_precision_counts"] = {
            "fp32": arena_value_count,
            "fp16": 0,
        }
        manifest["arena_alias_storage_precision_counts"] = {"fp32": 0, "fp16": 0}
    kernel_offset = sections[SECTION_KERNEL_PLAN]["offset"] + 8
    manifest["kernel_plan"] = [
        struct.unpack_from("<6I", data, kernel_offset + index * 24)
        for index in range(manifest["kernel_plan_count"])
    ]
    if any(
        (record[5] & PLAN_FLAG_AUTHORITATIVE) == 0
        or (record[5] & ~PLAN_FLAG_MASK) != 0
        for record in manifest["kernel_plan"]
    ):
        raise ValueError("invalid AEXRT kernel-plan flags")
    manifest["dxil_command_count"] = sum((record[5] & PLAN_FLAG_DXIL) != 0 for record in manifest["kernel_plan"])
    physical_section = sections.get(SECTION_PHYSICAL_DISPATCH_PLAN)
    if physical_section is None:
        manifest["physical_dispatch_plan_version"] = 0
        manifest["physical_dispatch_count"] = 0
        manifest["physical_zero_dispatch_count"] = 0
        manifest["physical_dispatch_plan"] = []
    else:
        physical_info = _inspect_physical_dispatch_plan(
            data, physical_section, manifest["command_count"]
        )
        manifest["physical_dispatch_plan_version"] = physical_info["version"]
        manifest["physical_dispatch_count"] = len(physical_info["records"])
        manifest["physical_dispatch_plan"] = physical_info["records"]
        for record in manifest["physical_dispatch_plan"]:
            record["zero_dispatch"] = all(
                command_kinds[int(index)]
                in {COMMAND_KIND["VIEW"], COMMAND_KIND["ALIAS"]}
                for index in record["logical_indices"]
            )
        manifest["physical_zero_dispatch_count"] = sum(
            bool(record["zero_dispatch"])
            for record in manifest["physical_dispatch_plan"]
        )
    barrier_section = sections.get(SECTION_ARENA_BARRIER_PLAN)
    if barrier_section is None:
        manifest["arena_barrier_plan_version"] = 0
        manifest["arena_barrier_count"] = 0
        manifest["arena_transition_count"] = 0
        manifest["arena_uav_barrier_count"] = 0
        manifest["arena_barrier_plan"] = []
    else:
        if physical_section is None or manifest["memory_plan_version"] != 2:
            raise ValueError("AEXRT arena barrier plan requires V2 physical memory")
        barrier_info = inspect_arena_barrier_plan(
            data[
                barrier_section["offset"] :
                barrier_section["offset"] + barrier_section["size"]
            ],
            dispatch_count=manifest["physical_dispatch_count"],
            page_nbytes=manifest["arena_page_nbytes"],
            value_count=manifest["value_count"],
        )
        manifest["arena_barrier_plan_version"] = barrier_info["version"]
        manifest["arena_barrier_count"] = len(barrier_info["records"])
        manifest["arena_transition_count"] = sum(
            record["kind"] in {
                ARENA_BARRIER_KIND_TRANSITION_SRV,
                ARENA_BARRIER_KIND_TRANSITION_UAV,
            }
            for record in barrier_info["records"]
        )
        manifest["arena_uav_barrier_count"] = sum(
            record["kind"] == ARENA_BARRIER_KIND_UAV_REUSE
            for record in barrier_info["records"]
        )
        manifest["arena_barrier_plan"] = barrier_info["records"]
    manifest.update(
        {
            "format": "aexrt.engine",
            "version": version,
            "flags": flags,
            "file_size": file_size,
            "section_count": section_count,
            "sections": sections,
        }
    )
    return manifest


def install_aexrt_pipeline_cache(
    engine_path: str | os.PathLike[str],
    cache_payload: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    path = Path(engine_path)
    data = path.read_bytes()
    info = inspect_aexrt_engine(data)
    cache = bytes(cache_payload)
    if len(cache) < 16:
        raise ValueError("truncated AEXRT pipeline cache")
    cache_info = _inspect_pipeline_cache(cache, {"offset": 0, "size": len(cache)})
    if cache_info["shader_count"] + cache_info["pso_count"] == 0:
        raise ValueError("empty or unsupported AEXRT pipeline cache")
    sections = {
        section_type: data[section["offset"] : section["offset"] + section["size"]]
        for section_type, section in info["sections"].items()
    }
    sections[SECTION_PIPELINE_CACHE] = cache
    rebuilt = _build_container(sections)
    path.write_bytes(rebuilt)
    return inspect_aexrt_engine(rebuilt)


def _inspect_command_kinds(
    data: bytes,
    section: dict[str, int],
    expected_count: int,
) -> list[int]:
    base = int(section["offset"])
    end = base + int(section["size"])
    if end - base < 8:
        raise ValueError("truncated AEXRT commands")
    count, reserved = struct.unpack_from("<II", data, base)
    if reserved != 0 or count != expected_count:
        raise ValueError("invalid AEXRT command header")
    cursor = base + 8
    valid_kinds = set(COMMAND_KIND.values())
    kinds: list[int] = []
    for _ in range(count):
        if cursor > end or 24 > end - cursor:
            raise ValueError("truncated AEXRT command record")
        kind, _, input_count, param_count, _, precision = struct.unpack_from(
            "<6I", data, cursor
        )
        cursor += 24
        tail_nbytes = (int(input_count) + int(param_count)) * 4
        if (
            kind not in valid_kinds
            or precision not in {PRECISION_FLOAT32, PRECISION_FLOAT16}
            or cursor > end
            or tail_nbytes > end - cursor
        ):
            raise ValueError("invalid AEXRT command record")
        cursor += tail_nbytes
        kinds.append(kind)
    if cursor != end:
        raise ValueError("invalid AEXRT command payload size")
    return kinds


def _inspect_constants(data: bytes, section: dict[str, int]) -> dict[str, int]:
    base = section["offset"]
    size = section["size"]
    if size < 8:
        raise ValueError("truncated AEXRT constants")
    count, reserved = struct.unpack_from("<II", data, base)
    if reserved != 0 or 8 + count * 32 > size:
        raise ValueError("invalid AEXRT constants")
    counts = {"fp32": 0, "fp16": 0}
    for index in range(count):
        _, dtype, elements, offset, nbytes = struct.unpack_from("<IIQQQ", data, base + 8 + index * 32)
        if dtype not in {PRECISION_FLOAT32, PRECISION_FLOAT16}:
            raise ValueError("unsupported AEXRT constant precision")
        element_nbytes = 2 if dtype == PRECISION_FLOAT16 else 4
        if nbytes != elements * element_nbytes or offset > size or nbytes > size - offset:
            raise ValueError("invalid AEXRT constant payload")
        counts["fp16" if dtype == PRECISION_FLOAT16 else "fp32"] += 1
    return counts


def _inspect_packed_weights(data: bytes, section: dict[str, int]) -> dict[str, Any]:
    base = section["offset"]
    size = section["size"]
    if size < 8:
        raise ValueError("truncated AEXRT packed weights")
    count, reserved = struct.unpack_from("<II", data, base)
    if reserved != 0 or 8 + count * 40 > size:
        raise ValueError("invalid AEXRT packed weights")
    layout_counts: dict[int, int] = {}
    for index in range(count):
        _, layout, in_channels, out_channels, elements, offset, nbytes = struct.unpack_from(
            "<4IQQQ", data, base + 8 + index * 40
        )
        if layout not in {
            PACKED_LAYOUT_CONV3X3_OC4,
            PACKED_LAYOUT_CONV3X3_OC4_FP16,
            PACKED_LAYOUT_WINOGRAD_F2X2_FP16,
            PACKED_LAYOUT_CONV1X1_OC8_FP16,
            PACKED_LAYOUT_WINOGRAD_F2X2_FP32,
            PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16,
            PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP32,
            PACKED_LAYOUT_CONV_INT8,
            PACKED_LAYOUT_WINOGRAD_F4X4_FP16,
        }:
            raise ValueError("unsupported AEXRT packed-weight layout")
        element_nbytes = 2 if layout in {
            PACKED_LAYOUT_CONV3X3_OC4_FP16,
            PACKED_LAYOUT_WINOGRAD_F2X2_FP16,
            PACKED_LAYOUT_CONV1X1_OC8_FP16,
            PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16,
            PACKED_LAYOUT_WINOGRAD_F4X4_FP16,
        } else 4
        kernel_elements = (
            16
            if layout in {
                PACKED_LAYOUT_WINOGRAD_F2X2_FP16,
                PACKED_LAYOUT_WINOGRAD_F2X2_FP32,
            }
            else 36
            if layout == PACKED_LAYOUT_WINOGRAD_F4X4_FP16
            else 1
            if layout == PACKED_LAYOUT_CONV1X1_OC8_FP16
            else 9
        )
        if layout == PACKED_LAYOUT_CONV_INT8:
            # layout 8：ic/oc 按 4 对齐，elements 整除 4，kk 只能是 1 或 9，
            # 负载 = elements 个 int8 + oc 个 fp32 scale。
            kernel_elements = elements // (in_channels * out_channels) if in_channels and out_channels else 0
            if (
                in_channels == 0
                or out_channels == 0
                or in_channels % 4 != 0
                or out_channels % 4 != 0
                or elements % 4 != 0
                or kernel_elements not in (1, 9)
                or elements != in_channels * out_channels * kernel_elements
                or nbytes != elements + out_channels * 4
                or offset > size
                or nbytes > size - offset
            ):
                raise ValueError("invalid AEXRT packed-weight payload")
            layout_counts[layout] = layout_counts.get(layout, 0) + 1
            continue
        if (
            in_channels == 0
            or out_channels == 0
            or (
                layout == PACKED_LAYOUT_CONV1X1_OC8_FP16
                and (in_channels % 4 != 0 or out_channels % 8 != 0)
            )
            or (
                layout
                in {
                    PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16,
                    PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP32,
                }
                and (in_channels % 4 != 0 or out_channels % 4 != 0)
            )
            or elements != in_channels * out_channels * kernel_elements
            or nbytes != elements * element_nbytes
            or offset > size
            or nbytes > size - offset
        ):
            raise ValueError("invalid AEXRT packed-weight payload")
        layout_counts[layout] = layout_counts.get(layout, 0) + 1
    return {"count": count, "layout_counts": layout_counts}


def _inspect_fusion_plan(data: bytes, section: dict[str, int], command_count: int) -> list[dict[str, int]]:
    base = section["offset"]
    size = section["size"]
    if size < 16:
        raise ValueError("truncated AEXRT fusion plan")
    version, count, planned_command_count, flags = struct.unpack_from("<4I", data, base)
    if (
        version != 1
        or planned_command_count != command_count
        or (flags & 1) == 0
        or count > command_count * 2
        or size != 16 + count * 32
    ):
        raise ValueError("invalid AEXRT fusion plan")
    groups: list[dict[str, int]] = []
    for index in range(count):
        kind, start, end, precision, kernel, group_flags, aux0, aux1 = struct.unpack_from(
            "<8I", data, base + 16 + index * 32
        )
        if (
            kind < FUSION_PAIRED_CONV3X3_SILU
            or kind > FUSION_LATTICE_CHAIN_C3
            or start > end
            or end >= command_count
            or precision not in {PRECISION_FLOAT32, PRECISION_FLOAT16}
            or (group_flags & 1) == 0
        ):
            raise ValueError("invalid AEXRT fusion group")
        groups.append(
            {
                "kind": kind,
                "start": start,
                "end": end,
                "precision": precision,
                "kernel": kernel,
                "flags": group_flags,
                "aux0": aux0,
                "aux1": aux1,
            }
        )
    return groups


def _inspect_physical_dispatch_plan(
    data: bytes, section: dict[str, int], command_count: int
) -> dict[str, Any]:
    base = section["offset"]
    size = section["size"]
    if size < PHYSICAL_DISPATCH_PLAN_HEADER_SIZE:
        raise ValueError("truncated AEXRT physical dispatch plan")
    (
        version,
        record_count,
        planned_command_count,
        plan_flags,
        record_size,
        logical_index_count,
        logical_index_table_offset,
        reserved,
    ) = struct.unpack_from("<8I", data, base)
    expected_table_offset = PHYSICAL_DISPATCH_PLAN_HEADER_SIZE + record_count * PHYSICAL_DISPATCH_RECORD_SIZE
    if (
        version != PHYSICAL_DISPATCH_PLAN_VERSION
        or planned_command_count != command_count
        or plan_flags != PHYSICAL_DISPATCH_PLAN_FLAG_AUTHORITATIVE
        or record_size != PHYSICAL_DISPATCH_RECORD_SIZE
        or record_count == 0
        or record_count > command_count
        or logical_index_count != command_count
        or logical_index_table_offset != expected_table_offset
        or reserved != 0
        or size != logical_index_table_offset + logical_index_count * 4
    ):
        raise ValueError("invalid AEXRT physical dispatch plan")

    logical_indices = struct.unpack_from(
        f"<{logical_index_count}I", data, base + logical_index_table_offset
    )
    records: list[dict[str, Any]] = []
    covered: set[int] = set()
    stable_ids: set[int] = set()
    next_logical_offset = 0
    previous_execution_index = -1
    for record_index in range(record_count):
        (
            stable_id,
            logical_start,
            logical_end,
            execution_index,
            logical_index_offset,
            record_logical_count,
            kernel,
            precision,
            record_flags,
            fusion_kind,
            record_reserved,
        ) = struct.unpack_from(
            "<Q10I",
            data,
            base + PHYSICAL_DISPATCH_PLAN_HEADER_SIZE + record_index * PHYSICAL_DISPATCH_RECORD_SIZE,
        )
        if (
            stable_id == 0
            or stable_id in stable_ids
            or logical_index_offset != next_logical_offset
            or record_logical_count == 0
            or logical_index_offset > logical_index_count
            or record_logical_count > logical_index_count - logical_index_offset
            or precision not in {PRECISION_FLOAT32, PRECISION_FLOAT16}
            or (record_flags & PLAN_FLAG_AUTHORITATIVE) == 0
            or (record_flags & ~PLAN_FLAG_MASK) != 0
            or fusion_kind > FUSION_LATTICE_CHAIN_C3
            or record_reserved != 0
            or execution_index <= previous_execution_index
        ):
            raise ValueError("invalid AEXRT physical dispatch record")
        record_indices = tuple(
            int(value)
            for value in logical_indices[
                logical_index_offset : logical_index_offset + record_logical_count
            ]
        )
        if (
            any(index >= command_count for index in record_indices)
            or any(left >= right for left, right in zip(record_indices, record_indices[1:]))
            or logical_start != record_indices[0]
            or logical_end != record_indices[-1]
            or execution_index not in record_indices
            or any(index in covered for index in record_indices)
            or (fusion_kind == 0 and record_indices != (execution_index,))
            or (fusion_kind != 0 and len(record_indices) < 2)
        ):
            raise ValueError("invalid AEXRT physical dispatch logical indices")
        stable_ids.add(stable_id)
        covered.update(record_indices)
        next_logical_offset += record_logical_count
        previous_execution_index = execution_index
        records.append(
            {
                "stable_id": stable_id,
                "logical_start": logical_start,
                "logical_end": logical_end,
                "execution_index": execution_index,
                "logical_index_offset": logical_index_offset,
                "logical_index_count": record_logical_count,
                "logical_indices": record_indices,
                "kernel": kernel,
                "precision": precision,
                "flags": record_flags,
                "fusion_kind": fusion_kind,
            }
        )
    if next_logical_offset != logical_index_count or covered != set(range(command_count)):
        raise ValueError("invalid AEXRT physical dispatch coverage")
    return {"version": version, "flags": plan_flags, "records": records}


def _inspect_pipeline_cache(data: bytes, section: dict[str, int]) -> dict[str, Any]:
    base = section["offset"]
    size = section["size"]
    if size < 16:
        raise ValueError("truncated AEXRT pipeline cache")
    version, shader_count, pso_count, flags = struct.unpack_from("<4I", data, base)
    record_count = shader_count + pso_count
    table_end = 16 + record_count * 32
    if version != 1 or record_count > 4096 or table_end > size:
        raise ValueError("unsupported AEXRT pipeline cache")
    records: list[dict[str, int | str]] = []
    actual_counts = {1: 0, 2: 0}
    seen: set[tuple[int, int]] = set()
    for index in range(record_count):
        kind, record_flags, key, offset, nbytes = struct.unpack_from("<IIQQQ", data, base + 16 + index * 32)
        if (
            kind not in actual_counts
            or key == 0
            or nbytes == 0
            or (kind == 1 and (record_flags & ~1) != 0)
            or (kind == 2 and record_flags != 0)
            or offset < table_end
            or offset > size
            or nbytes > size - offset
            or (kind, key) in seen
        ):
            raise ValueError("invalid AEXRT pipeline-cache record")
        seen.add((kind, key))
        actual_counts[kind] += 1
        records.append(
            {
                "kind": ("dxil" if (record_flags & 1) else "dxbc") if kind == 1 else "pso",
                "key": key,
                "offset": offset,
                "size": nbytes,
                "flags": record_flags,
            }
        )
    if actual_counts[1] != shader_count or actual_counts[2] != pso_count:
        raise ValueError("invalid AEXRT pipeline-cache counts")
    dxil_count = sum(record["kind"] == "dxil" for record in records)
    return {
        "shader_count": shader_count,
        "dxil_count": dxil_count,
        "dxbc_count": shader_count - dxil_count,
        "pso_count": pso_count,
        "flags": flags,
        "records": records,
    }


def _parse_replay_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0] != "AEXRT_CXX_REPLAY_V1":
        raise ValueError("unsupported replay stream")
    values: dict[int, dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    input_value = -1
    output_value = -1
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split("|")
        tag = parts[0]
        if tag in {"INPUT", "VALUE", "CONST"}:
            if len(parts) < 5:
                raise ValueError("invalid replay value")
            value_id = int(parts[1])
            elements = int(parts[3])
            shape4 = tuple(_csv_u32(parts[4]))
            if len(shape4) != 4:
                raise ValueError("replay values must have rank-4 normalized shapes")
            flags = 0
            raw = b""
            if tag == "INPUT":
                flags |= VALUE_INPUT
                input_value = value_id
            elif tag == "CONST":
                flags |= VALUE_CONSTANT
                if len(parts) < 6:
                    raise ValueError("replay constant has no payload")
                raw = base64.b64decode(parts[5], validate=True)
                if len(raw) != elements * 4:
                    raise ValueError("replay constant payload size mismatch")
            values[value_id] = {
                "id": value_id,
                "flags": flags,
                "elements": elements,
                "shape4": shape4,
                "raw": raw,
            }
        elif tag == "CMD":
            if len(parts) < 5 or parts[1] not in COMMAND_KIND:
                raise ValueError("invalid replay command")
            commands.append(
                {
                    "kind": parts[1],
                    "kind_id": COMMAND_KIND[parts[1]],
                    "output": int(parts[2]),
                    "inputs": _csv_u32(parts[3]),
                    "params": _csv_u32(parts[4]),
                }
            )
        elif tag == "OUTPUT":
            output_value = int(parts[1])
        else:
            raise ValueError(f"unknown replay record {tag}")
    if input_value < 0 or output_value < 0 or not commands:
        raise ValueError("incomplete replay stream")
    dense_values = [values[index] for index in range(max(values) + 1)]
    return {
        "values": dense_values,
        "commands": commands,
        "input_value": input_value,
        "output_value": output_value,
    }


def _build_memory_plan(replay: dict[str, Any]) -> dict[str, Any]:
    values = replay["values"]
    commands = replay["commands"]
    producer = [-1] * len(values)
    last_use = [-1] * len(values)
    alias_source: dict[int, int] = {}
    for index, command in enumerate(commands):
        output = int(command["output"])
        producer[output] = index
        for value_id in command["inputs"]:
            last_use[value_id] = max(last_use[value_id], index)
        if command["kind"] in {"VIEW", "ALIAS"}:
            alias_source[output] = int(command["inputs"][0])
            values[output]["flags"] |= VALUE_ALIAS
    last_use[replay["output_value"]] = len(commands) + 1
    for _ in range(len(alias_source) + 1):
        changed = False
        for alias, source in alias_source.items():
            if last_use[alias] > last_use[source]:
                last_use[source] = last_use[alias]
                changed = True
        if not changed:
            break

    candidates: list[tuple[int, int, int, int]] = []
    for value in values:
        value_id = int(value["id"])
        if value["flags"] & (VALUE_CONSTANT | VALUE_ALIAS):
            continue
        start = 0 if value["flags"] & VALUE_INPUT else producer[value_id] + 1
        end = max(start, last_use[value_id] + 1)
        candidates.append((start, end, int(value["elements"]) * 4, value_id))
    candidates.sort(key=lambda item: item[3])

    records: list[dict[str, int]] = []
    total_nbytes = 0
    for start, end, nbytes, value_id in candidates:
        offset = _align(total_nbytes, ARENA_ALIGNMENT)
        total_nbytes = offset + _align(nbytes, ARENA_ALIGNMENT)
        values[value_id]["flags"] |= VALUE_ARENA
        records.append(
            {
                "value_id": value_id,
                "flags": int(values[value_id]["flags"]),
                "offset": offset,
                "nbytes": nbytes,
                "start": start,
                "end": end,
            }
        )
    by_value = {record["value_id"]: record for record in records}
    return {
        "alignment": ARENA_ALIGNMENT,
        "total_nbytes": _align(total_nbytes, ARENA_ALIGNMENT),
        "records": records,
        "by_value": by_value,
    }


def _build_kernel_plan(replay: dict[str, Any], *, precision: int) -> list[dict[str, int]]:
    plans: list[dict[str, int]] = []
    for index, command in enumerate(replay["commands"]):
        kind = command["kind"]
        planned_kernel = 0
        packed_value = NO_VALUE
        planned_winograd = False
        if kind in {"CONV", "CONV_SILU"}:
            planned_kernel = _stable_conv_algorithm(command["params"], silu=kind == "CONV_SILU")
            planned_winograd = (
                precision == PRECISION_FLOAT16
                and planned_kernel in {40, 46}
                and _uses_native_fp16_winograd(command["params"])
            )
            if (
                precision == PRECISION_FLOAT16
                and _uses_native_fp16_conv1x1(command["params"])
            ):
                planned_kernel = 45
            elif (
                precision == PRECISION_FLOAT16
                and kind == "CONV_SILU"
            ):
                if _uses_native_fp16_position_pair_40x40_256x256(command["params"]):
                    planned_kernel = 47
                elif _uses_native_fp16_pos2_40x40_64x64(command["params"]):
                    planned_kernel = 44
                elif _uses_measured_native_fp16(command["params"]):
                    planned_kernel = 43
            if len(command["inputs"]) >= 2 and (
                planned_kernel in {32, 33, 34, 39, 43, 44, 45, 47}
                or planned_winograd
            ):
                packed_value = int(command["inputs"][1])
        elif kind == "CONCAT_CONV1X1":
            input_count = max(0, len(command["inputs"]) - 2)
            total_in = sum(command["params"][5 : 5 + input_count])
            out_channels = command["params"][3]
            spatial = command["params"][1] * command["params"][2]
            if precision == PRECISION_FLOAT16 and _uses_native_fp16_concat_conv1x1(command):
                planned_kernel = 3
                packed_value = int(command["inputs"][-2])
            else:
                planned_kernel = 1 if total_in >= 64 and out_channels >= 64 and out_channels % 4 == 0 and spatial >= 100 else 0
        heavy_kernel = kind in {"CONV", "CONV_SILU", "CONCAT_CONV1X1", "MATMUL"}
        command_precision = precision if heavy_kernel else PRECISION_FLOAT32
        plan_flags = PLAN_FLAG_AUTHORITATIVE
        native_fp16_kernel = (
            kind in {"CONV", "CONV_SILU"}
            and (planned_kernel in {43, 44, 45, 47} or planned_winograd)
        ) or (kind == "CONCAT_CONV1X1" and planned_kernel == 3)
        if precision == PRECISION_FLOAT16 and (not heavy_kernel or native_fp16_kernel):
            plan_flags |= PLAN_FLAG_DXIL
        plans.append(
            {
                "command_index": index,
                "kind_id": int(command["kind_id"]),
                "planned_kernel": planned_kernel,
                "precision": command_precision,
                "packed_value": packed_value,
                "flags": plan_flags,
            }
        )
    return plans


def _apply_algo_overrides(
    replay: dict[str, Any],
    kernel_plan: list[dict[str, int]],
    overrides: Any,
    *,
    fusion_plan: Sequence[dict[str, int]] = (),
) -> None:
    """把 conv_autotune 的 per-device 实测结果固化进 kernel plan。

    overrides: {command_index: kernel_id}，仅允许 CONV/CONV_SILU 命令；
    调用方必须保证 override 已通过数值门禁（conv_autotune 自带）。
    融合组内的命令会被跳过（组内 kernel 由融合组决定，单独 override
    会产出物理计划失配的引擎）。
    """
    commands = replay["commands"]
    grouped_commands: set[int] = set()
    import os as _os
    _allow_head_interior = bool(_os.environ.get("AEXRTC_OVERRIDE_HEAD_INTERIOR"))
    for group in fusion_plan:
        # Head-fusion (kind 4) spans cover interior commands that still run
        # as standalone dispatches; only the absorbed head kernel is fused.
        # Allowing overrides there converts the leftover fp16 winograd convs.
        if _allow_head_interior and int(group.get("kind", 0)) == 4:
            continue
        grouped_commands.update(
            range(int(group["start"]), int(group["end"]) + 1)
        )
    # 组外生产者：输出被融合组内命令消费的命令。融合组 shader 假定其
    # 输入布局（plain fp16 等）；override 生产者可能切换到不同输出布局的
    # kernel（如 pack4 交错），fused shader 按原布局读 → 越界段错误。
    value_producer: dict[int, int] = {}
    for producer_index, command in enumerate(commands):
        value_producer[int(command["output"])] = producer_index
    group_feeding_commands: set[int] = set()
    for group in fusion_plan:
        start = int(group["start"])
        end = int(group["end"])
        for member_index in range(start, end + 1):
            for raw_input in commands[member_index]["inputs"]:
                producer = value_producer.get(int(raw_input))
                if producer is not None and not (start <= producer <= end):
                    group_feeding_commands.add(producer)
    # 跨族固化安全性（实测结论）：
    # - 目标 plain（0/1/2/3/24/25...）：无打包依赖，直接安全；
    # - 目标 packed（32/33/34/39/43/44/45/47）：_build_packed_weights 按
    #   kernel/形状自动打包（branch1: 47/45；branch2: k3+stride2/39/44+oc%4==0），
    #   packed_value 指向原始权重即可，安全；
    # - 目标 int8/f4x4（51/52/53/54）：layout 8/9 打包完全由 kernel plan 驱动
    #   （_add_packed_conv_int8 / _add_packed_winograd_f4x4_fp16），加载器按
    #   planned_kernel 强一致校验，安全；
    # - 目标 winograd（40/46）：需要权重 Winograd 域变换 + DXIL/fusion 协调，
    #   该变换不由 kernel plan 驱动（_add_packed_winograd 由独立启发式决定），
    #   盲切会产出布局错配引擎（实测 payload 校验失败）。仅此方向拒绝。
    winograd_kernels = {40, 46}
    packed_kernels = {32, 33, 34, 39, 43, 44, 45, 47, 51, 52, 53, 54}
    applied: list[int] = []
    skipped: list[int] = []
    for raw_index, kernel in dict(overrides).items():
        command_index = int(raw_index)
        kernel = int(kernel)
        if command_index < 0 or command_index >= len(kernel_plan):
            raise ValueError(f"algo override command index out of range: {command_index}")
        if commands[command_index]["kind"] not in {"CONV", "CONV_SILU"}:
            raise ValueError(f"algo override target is not a conv command: {command_index}")
        if kernel < 0 or kernel > 54:
            raise ValueError(f"algo override kernel id out of range: {kernel}")
        record = kernel_plan[command_index]
        previous_kernel = record["planned_kernel"]

        if command_index in grouped_commands:
            skipped.append(command_index)
            continue
        # 组外生产者放宽：INT8 dot4（51/52/53）与 Winograd F(4x4)（54）输出
        # plain typed 视图（按 value dtype 自适应 R16/R32），生产者切换不改变
        # 融合组 shader 读到的布局；运行时 override 路径已实测通过数值门禁
        # （autotune 组合复测含 90/91/109→INT8）。其余族（pack 交错布局等）
        # 仍维持 blanket 拒绝。
        if command_index in group_feeding_commands and kernel not in {51, 52, 53, 54}:
            skipped.append(command_index)
            continue

        if kernel in winograd_kernels:
            # winograd 目标一律拒绝（含 46↔40 同族切换）：权重 Winograd 域
            # 变换 + DXIL/fusion 协调不由 kernel plan 单点驱动，运行时单候选
            # 门禁通过不代表编译期打包链路一致（实测 DYv11s 102→40 构建
            # 即崩）。
            skipped.append(command_index)
            continue

        record["planned_kernel"] = kernel
        if kernel in packed_kernels and previous_kernel in packed_kernels:
            pass  # 同为 packed-weight 族：保留原 packed 引用
        elif kernel in packed_kernels and len(commands[command_index]["inputs"]) >= 2:
            record["packed_value"] = int(commands[command_index]["inputs"][1])
        else:
            record["packed_value"] = NO_VALUE
        # flags 只调整 DXIL 位（随算法族变化）；io/dtype 位是布局事实，
        # 由 dtype 校准阶段决定，必须保留——重置会丢失 fp16-io 位，
        # 导致下游融合组（消费该命令输出的 add/winograd）按错误布局读
        # 存储（实测段错误）。
        record["flags"] = (int(record["flags"]) & ~PLAN_FLAG_DXIL) | PLAN_FLAG_AUTHORITATIVE
        if kernel in {43, 44, 45, 47, 51, 52, 53, 54}:
            record["flags"] |= PLAN_FLAG_DXIL
        applied.append(command_index)
    if skipped:
        import sys
        print(
            f"aexrtc: algo-map skipped {len(skipped)} unsafe override(s) "
            f"(winograd-target or inside-fusion-group): {skipped}",
            file=sys.stderr,
        )


def _apply_kernel_plan_seed(
    replay: dict[str, Any],
    kernel_plan: list[dict[str, int]],
    seed: Sequence[Sequence[int]],
) -> None:
    """Freeze a previously promoted kernel plan before dtype calibration.

    This is a compiler input only; the resulting authoritative records remain
    embedded in the binary engine.  It lets a V2 memory/barrier candidate vary
    activation storage without accidentally pulling unrelated planner probes
    into the same promotion decision.
    """
    commands = replay["commands"]
    values = replay["values"]
    if len(kernel_plan) != len(commands) or len(seed) != len(commands):
        raise ValueError("kernel-plan seed does not cover the replay command stream")
    for expected_index, raw_record in enumerate(seed):
        record = tuple(int(value) for value in raw_record)
        if len(record) != 6:
            raise ValueError("kernel-plan seed record must contain six integers")
        command_index, kind_id, kernel, plan_precision, packed_value, flags = record
        command = commands[expected_index]
        if (
            command_index != expected_index
            or kind_id != int(command["kind_id"])
            or kernel < 0
            or plan_precision not in {PRECISION_FLOAT32, PRECISION_FLOAT16}
            or flags & PLAN_FLAG_AUTHORITATIVE == 0
            or flags & ~PLAN_FLAG_MASK
            or (
                packed_value != NO_VALUE
                and (packed_value < 0 or packed_value >= len(values))
            )
        ):
            raise ValueError("kernel-plan seed is incompatible with replay command stream")
        kernel_plan[expected_index] = {
            "command_index": command_index,
            "kind_id": kind_id,
            "planned_kernel": kernel,
            "precision": plan_precision,
            "packed_value": packed_value,
            "flags": flags,
        }


def _plan_fp16_activation_islands(
    replay: dict[str, Any],
    kernel_plan: list[dict[str, int]],
    physical_dispatch_plan: Sequence[dict[str, Any]],
    fusion_plan: Sequence[dict[str, int]] | None = None,
    *,
    precision: int,
    selected_value_ids: Sequence[int] | None = None,
) -> dict[int, tuple[int, int]]:
    """Lower closed homogeneous FP16 stages onto the physical value DAG.

    VIEW/ALIAS records are transparent zero-dispatch bindings.  A material
    value is selected only when its producer can write FP16 and every physical
    consumer can read it.  Residual arithmetic remains homogeneous and atomic.
    Descriptor-backed CONCAT/ConcatConv1x1 records may instead read a deliberate
    R16/R32 branch mix: the memory plan carries each branch dtype and the typed
    SRV descriptors perform the boundary conversion without another dispatch.
    """
    if precision != PRECISION_FLOAT16:
        return {}

    commands = replay["commands"]
    values = replay["values"]
    if len(kernel_plan) != len(commands):
        raise ValueError("kernel plan does not cover the replay command stream")

    dag = build_physical_value_dag(replay, physical_dispatch_plan)
    dispatches = dag["dispatches"]
    canonical_values = dag["canonical_values"]
    # INT8 dot4（51/52/53）与 Winograd F(4x4)（54）同为 fp16-io DXIL kernel：
    # 权重走 layout 8/9 打包，激活/输出仍是 R16 typed 视图。漏列会把
    # override 到这些族的 conv 剔出 fp16 激活图（实测 DYv11s 9→53 后
    # 下游 CONCAT_CONV1X1 融合输入级失质退化 CONCAT_RESIDUAL_CV2，
    # 每处 ~+0.14ms，净收益变净亏损）。
    supported_conv_kernels = {1, 24, 33, 39, 40, 43, 44, 45, 47, 51, 52, 53, 54}
    supported_fusions = {
        FUSION_C2F_TAIL_RESIDUAL,
        FUSION_LATE_CONCAT_CONV1X1,
        FUSION_STRIDE2_CONV1X1,
        FUSION_CONCAT_RESIDUAL_CV2,
        FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2,
    }
    io_mask = PLAN_FLAG_INPUT_FP16 | PLAN_FLAG_OUTPUT_FP16

    def valid_flags(flags: int) -> bool:
        return bool(flags & PLAN_FLAG_AUTHORITATIVE) and not (flags & ~PLAN_FLAG_MASK)

    fusion_groups_by_key: dict[tuple[int, int, int], dict[str, int]] = {}
    for group in fusion_plan or ():
        key = (int(group["kind"]), int(group["start"]), int(group["end"]))
        if key in fusion_groups_by_key:
            raise ValueError("duplicate fusion group in FP16 activation plan")
        fusion_groups_by_key[key] = group

    supported_by_dispatch: dict[int, dict[str, Any]] = {}
    for dispatch in dispatches:
        dispatch_index = int(dispatch["dispatch_index"])
        command_index = int(dispatch["execution_index"])
        record = physical_dispatch_plan[dispatch_index]
        logical_indices = tuple(int(index) for index in dispatch["logical_indices"])
        fusion_kind = int(record["fusion_kind"])
        if (
            command_index < 0
            or command_index >= len(commands)
            or not valid_flags(int(record["flags"]))
        ):
            continue

        execution = commands[command_index]
        execution_plan = kernel_plan[command_index]
        group: dict[str, int] | None = None
        kernel_commands: tuple[int, ...] = ()
        output_command_index = command_index
        atomic_inputs = False

        if fusion_kind == 0:
            if logical_indices != (command_index,):
                continue
            kernel = int(execution_plan["planned_kernel"])
            plan_matches_record = (
                int(execution_plan["command_index"]) == command_index
                and int(record["kernel"]) == kernel
                and int(record["precision"]) == int(execution_plan["precision"])
                and (int(record["flags"]) & ~io_mask)
                == (int(execution_plan["flags"]) & ~io_mask)
                and valid_flags(int(execution_plan["flags"]))
            )
            if not plan_matches_record:
                continue
            if execution["kind"] in {"CONV", "CONV_SILU"}:
                if (
                    len(execution["inputs"]) < 3
                    or kernel not in supported_conv_kernels
                    or int(execution_plan["precision"]) != PRECISION_FLOAT16
                ):
                    continue
            elif execution["kind"] == "CONCAT_CONV1X1":
                if (
                    len(execution["inputs"]) < 3
                    or kernel not in {1, 3}
                    or int(execution_plan["precision"]) != PRECISION_FLOAT16
                ):
                    continue
            elif execution["kind"] == "CONCAT":
                if (
                    len(execution["inputs"]) < 2
                    or len(execution["inputs"]) > 8
                    or len(execution["params"]) < 18
                    or kernel != 0
                ):
                    continue
            elif not _is_binary_add(execution):
                continue
            else:
                atomic_inputs = True
            kernel_commands = (command_index,)
        elif fusion_kind in supported_fusions:
            if len(logical_indices) < 2:
                continue
            if fusion_kind not in {
                FUSION_CONCAT_RESIDUAL_CV2,
                FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2,
            }:
                group = fusion_groups_by_key.get(
                    (
                        fusion_kind,
                        int(record["logical_start"]),
                        int(record["logical_end"]),
                    )
                )
                if (
                    group is None
                    or int(group["kernel"]) != int(record["kernel"])
                    or int(group["precision"]) != int(record["precision"])
                    or (int(group["flags"]) & ~io_mask)
                    != (int(record["flags"]) & ~io_mask)
                    or not valid_flags(int(group["flags"]))
                ):
                    continue
            if fusion_kind == FUSION_C2F_TAIL_RESIDUAL:
                output_command_index = logical_indices[-1]
                atomic_inputs = True
                if (
                    execution["kind"] != "CONV_SILU"
                    or not _is_binary_add(commands[output_command_index])
                ):
                    continue
                # FP16 存储 IO 仅 native fp16 winograd（typed 视图）路径可承载：
                # 运行时 requested_native_winograd 要求 record 与 conv kernel plan
                # 同时为 40 + packed 权重；否则其余算法族（spatial/implicit-gemm/
                # SM5 winograd）录制 float 结构化视图（stride=4），绑到 FP16
                # arena 会 FirstElement 错位并越界 2 倍字节（实测 cs2V8 cmd#17
                # 组 GPU 设备移除）。与运行时 kind-2 fp16-io 白名单语义对齐。
                if (
                    int(record["kernel"]) != 40
                    or int(record["precision"]) != PRECISION_FLOAT16
                    or int(execution_plan["planned_kernel"]) != 40
                    or int(execution_plan["precision"]) != PRECISION_FLOAT16
                    or int(execution_plan["packed_value"])
                    != int(execution["inputs"][1])
                    or int(execution_plan["flags"])
                    & ~(PLAN_FLAG_INPUT_FP16 | PLAN_FLAG_OUTPUT_FP16)
                    != PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
                ):
                    continue
            elif fusion_kind == FUSION_LATE_CONCAT_CONV1X1:
                if (
                    group is None
                    or int(record["kernel"]) not in {1, 3}
                    or int(record["precision"]) != PRECISION_FLOAT16
                    or execution["kind"] not in {"CONV", "CONV_SILU"}
                    or len(execution["inputs"]) < 3
                ):
                    continue
                kernel_commands = (command_index,)
            elif fusion_kind == FUSION_STRIDE2_CONV1X1:
                output_command_index = logical_indices[-1]
                if (
                    execution["kind"] != "CONV_SILU"
                    or commands[output_command_index]["kind"] != "CONV_SILU"
                ):
                    continue
            elif fusion_kind == FUSION_CONCAT_RESIDUAL_CV2:
                atomic_inputs = True
                if (
                    execution["kind"] != "CONCAT_CONV1X1"
                    or int(execution_plan["planned_kernel"]) != 3
                    or int(execution_plan["precision"]) != PRECISION_FLOAT16
                    or int(record["kernel"]) != 3
                    or int(record["precision"]) != PRECISION_FLOAT16
                    or not valid_flags(int(execution_plan["flags"]))
                ):
                    continue
                kernel_commands = (command_index,)
            else:
                output_command_index = logical_indices[-1]
                atomic_inputs = True
                if (
                    int(record["kernel"]) != 40
                    or int(record["precision"]) != PRECISION_FLOAT16
                    or execution["kind"] != "CONV_SILU"
                    or len(logical_indices) not in {3, 4}
                    or commands[logical_indices[1]]["kind"] != "BINARY"
                    or commands[output_command_index]["kind"]
                    != "CONCAT_CONV1X1"
                ):
                    continue
                # Synthetic kind-8 records do not have a serialized fusion
                # group owner.  Persist their physical I/O flags through the
                # execution command so rebuilding section 10 cannot erase the
                # typed stage.
                kernel_commands = (command_index,)
        else:
            continue

        output_value = int(
            canonical_values[int(commands[output_command_index]["output"])]
        )
        if output_value not in dispatch["produced_values"]:
            continue
        activation_inputs = frozenset(
            int(value_id)
            for value_id in dispatch["external_inputs"]
            if not (int(values[int(value_id)]["flags"]) & VALUE_CONSTANT)
        )
        supported_by_dispatch[dispatch_index] = {
            "read_values": activation_inputs,
            "write_values": frozenset((output_value,)),
            "atomic_inputs": atomic_inputs and len(activation_inputs) > 1,
            "kernel_commands": kernel_commands,
            "fusion_group": group,
        }

    eligible_values: set[int] = set()
    consumers_by_value = dag["consumer_dispatches"]
    producer_by_value = dag["producer_by_value"]
    graph_output = int(canonical_values[int(replay["output_value"])])
    for value_id, value in enumerate(values):
        flags = int(value["flags"])
        producer_dispatch = int(producer_by_value[value_id])
        consumer_dispatches = tuple(int(index) for index in consumers_by_value[value_id])
        if (
            int(canonical_values[value_id]) != value_id
            or value_id == graph_output
            or flags & (VALUE_INPUT | VALUE_CONSTANT | VALUE_ALIAS)
            or producer_dispatch not in supported_by_dispatch
            or value_id
            not in supported_by_dispatch[producer_dispatch]["write_values"]
            or not consumer_dispatches
            or any(
                dispatch_index not in supported_by_dispatch
                or value_id
                not in supported_by_dispatch[dispatch_index]["read_values"]
                for dispatch_index in consumer_dispatches
            )
        ):
            continue

        eligible_values.add(value_id)

    if selected_value_ids is not None:
        requested_raw = {int(value_id) for value_id in selected_value_ids}
        if any(value_id < 0 or value_id >= len(values) for value_id in requested_raw):
            raise ValueError("FP16 activation selection names an invalid value")
        requested_values = {
            int(canonical_values[value_id]) for value_id in requested_raw
        }
        invalid_values = requested_values - eligible_values
        if invalid_values:
            raise ValueError(
                "FP16 activation selection contains values outside a closed "
                f"physical-DAG capability: {sorted(invalid_values)}"
            )
        selected_values = requested_values
        for capability in supported_by_dispatch.values():
            if not capability["atomic_inputs"]:
                continue
            inputs = set(capability["read_values"])
            selected_inputs = inputs & selected_values
            if selected_inputs and selected_inputs != inputs:
                raise ValueError(
                    "FP16 activation selection contains a partial homogeneous "
                    "input stage (multi-input): "
                    f"{sorted(selected_inputs)} of {sorted(inputs)}"
                )
    else:
        selected_values = set(eligible_values)
        changed = True
        while changed:
            changed = False
            for capability in supported_by_dispatch.values():
                if not capability["atomic_inputs"]:
                    continue
                inputs = set(capability["read_values"])
                selected_inputs = inputs & selected_values
                if selected_inputs and selected_inputs != inputs:
                    selected_values.difference_update(selected_inputs)
                    changed = True

    io_flags_by_dispatch = [0] * len(dispatches)
    for value_id in selected_values:
        producer_dispatch = int(producer_by_value[value_id])
        io_flags_by_dispatch[producer_dispatch] |= PLAN_FLAG_OUTPUT_FP16
        for consumer_dispatch in consumers_by_value[value_id]:
            io_flags_by_dispatch[int(consumer_dispatch)] |= PLAN_FLAG_INPUT_FP16

    # The kernel plan owns singleton flags; fusion groups own fused-dispatch
    # flags.  Keep the already-built physical records coherent as well, then
    # refresh their stable IDs because flags participate in the identity hash.
    for plan in kernel_plan:
        plan["flags"] = int(plan["flags"]) & ~io_mask
    for group in fusion_plan or ():
        group["flags"] = int(group["flags"]) & ~io_mask
    for record in physical_dispatch_plan:
        record["flags"] = int(record["flags"]) & ~io_mask

    for dispatch_index, capability in supported_by_dispatch.items():
        io_flags = io_flags_by_dispatch[dispatch_index]
        dispatch_flags = io_flags | (PLAN_FLAG_DXIL if io_flags else 0)
        group = capability["fusion_group"]
        for kernel_command in capability["kernel_commands"]:
            kernel_plan[int(kernel_command)]["flags"] |= dispatch_flags
        if group is not None:
            group["flags"] |= dispatch_flags
        physical_dispatch_plan[dispatch_index]["flags"] |= dispatch_flags

    stable_ids: set[int] = set()
    for record in physical_dispatch_plan:
        stable_id = _physical_dispatch_stable_id(replay, record)
        if stable_id in stable_ids:
            raise ValueError("physical dispatch stable-id collision")
        stable_ids.add(stable_id)
        record["stable_id"] = stable_id

    return {
        value_id: (PRECISION_FLOAT16, LAYOUT_LINEAR_NCHW)
        for value_id in sorted(selected_values)
    }


def _uses_native_fp16_1x1_shape(batch: int, height: int, width: int, in_channels: int, out_channels: int) -> bool:
    # Nearby 1x1 shapes can lose badly, so this remains an exact measured list.
    return (batch, height, width, in_channels, out_channels) in {
        (1, 40, 40, 64, 64),
        (1, 40, 40, 128, 64),
        (1, 40, 40, 128, 128),
        (1, 40, 40, 256, 64),
        (1, 20, 20, 128, 128),
        (1, 20, 20, 128, 64),
        (1, 20, 20, 256, 128),
        (1, 20, 20, 256, 256),
        (1, 20, 20, 384, 256),
        (1, 20, 20, 512, 128),
        (1, 10, 10, 512, 512),
        (1, 10, 10, 256, 256),
    }


def _uses_native_fp16_conv1x1(params: Sequence[int]) -> bool:
    if len(params) < 16:
        return False
    batch, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, pt, pl, dh, dw, groups = (
        int(value) for value in params[:16]
    )
    return (
        kh == kw == sh == sw == dh == dw == groups == 1
        and pt == pl == 0
        and ih == oh
        and iw == ow
        and _uses_native_fp16_1x1_shape(batch, oh, ow, ic, oc)
    )


def _uses_native_fp16_concat_conv1x1(command: dict[str, Any]) -> bool:
    input_count = len(command["inputs"]) - 2
    params = command["params"]
    if input_count <= 0 or input_count > 8 or len(params) < 5 + input_count:
        return False
    batch, height, width, out_channels, activation = (
        int(value) for value in params[:5]
    )
    channels = tuple(int(value) for value in params[5 : 5 + input_count])
    signature = (
        batch,
        height,
        width,
        sum(channels),
        out_channels,
        activation,
        channels,
    )
    return signature in {
        (1, 80, 80, 64, 64, 1, (32, 32)),
        (1, 40, 40, 128, 128, 1, (64, 64)),
        (1, 40, 40, 512, 128, 1, (256, 256)),
        (1, 40, 40, 192, 128, 1, (64, 64, 64)),
        (1, 20, 20, 128, 128, 1, (64, 64)),
        (1, 20, 20, 256, 128, 1, (64, 64, 64, 64)),
        (1, 20, 20, 384, 256, 1, (128, 128, 128)),
        (1, 10, 10, 384, 256, 1, (128, 128, 128)),
    }


def _uses_native_fp16_residual_concat_conv1x1(command: dict[str, Any]) -> bool:
    input_count = len(command["inputs"]) - 2
    params = command["params"]
    if input_count <= 0 or input_count > 8 or len(params) < 5 + input_count:
        return False
    batch, height, width, out_channels, activation = (
        int(value) for value in params[:5]
    )
    channels = tuple(int(value) for value in params[5 : 5 + input_count])
    return (
        batch,
        height,
        width,
        sum(channels),
        out_channels,
        activation,
        channels,
    ) == (1, 20, 20, 384, 256, 1, (128, 128, 128))


def _apply_measured_fp16_physical_kernels(
    replay: dict[str, Any],
    kernel_plan: list[dict[str, int]],
    fusion_plan: Sequence[dict[str, int]],
    *,
    precision: int,
) -> None:
    if precision != PRECISION_FLOAT16:
        return
    for command_index, _ in _physical_concat_residual_cv2_groups(
        replay, fusion_plan, kernel_plan
    ):
        command = replay["commands"][command_index]
        if not _uses_native_fp16_residual_concat_conv1x1(command):
            continue
        plan = kernel_plan[command_index]
        plan["planned_kernel"] = 3
        plan["precision"] = PRECISION_FLOAT16
        plan["packed_value"] = int(command["inputs"][-2])
        plan["flags"] = PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL


def _apply_measured_fp16_c3_topology_kernels(
    replay: dict[str, Any],
    kernel_plan: list[dict[str, int]],
    physical_dispatch_plan: Sequence[dict[str, Any]],
    *,
    precision: int,
) -> bool:
    """Apply measured C3 choices only to their serialized physical topology."""
    if precision != PRECISION_FLOAT16:
        return False

    commands = replay["commands"]
    values = replay["values"]
    if len(kernel_plan) != len(commands):
        raise ValueError("kernel plan does not cover the replay command stream")

    producers = [-1] * len(values)
    consumers: list[list[int]] = [[] for _ in values]
    for command_index, command in enumerate(commands):
        output = int(command["output"])
        if 0 <= output < len(producers):
            producers[output] = command_index
        for raw_value in command["inputs"]:
            value_id = int(raw_value)
            if 0 <= value_id < len(consumers):
                consumers[value_id].append(command_index)

    physical_owners: dict[int, dict[str, Any]] = {}
    for record in physical_dispatch_plan:
        for raw_index in record["logical_indices"]:
            physical_owners[int(raw_index)] = record

    def value_shape(value_id: int) -> tuple[int, ...] | None:
        if value_id < 0 or value_id >= len(values):
            return None
        shape = tuple(int(item) for item in values[value_id].get("shape4", ()))
        return shape if len(shape) == 4 else None

    def singleton_owner(command_index: int) -> dict[str, Any] | None:
        owner = physical_owners.get(command_index)
        if (
            owner is None
            or int(owner["fusion_kind"]) != 0
            or int(owner["execution_index"]) != command_index
            or tuple(int(item) for item in owner["logical_indices"])
            != (command_index,)
        ):
            return None
        return owner

    target_desc = (1, 128, 20, 20, 64, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1)
    base_desc = (1, 384, 20, 20, 256, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1)
    successor_desc = (1, 64, 20, 20, 128, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1)
    half_shape = (1, 128, 20, 20)
    base_shape = (1, 256, 20, 20)
    changed = False

    for command_index, command in enumerate(commands):
        if (
            command["kind"] != "CONV_SILU"
            or _conv_desc(command) != target_desc
            or len(command["inputs"]) != 3
        ):
            continue
        target_owner = singleton_owner(command_index)
        plan = kernel_plan[command_index]
        weight_id = int(command["inputs"][1])
        if (
            target_owner is None
            or (
                int(target_owner["kernel"]),
                int(target_owner["precision"]),
                int(target_owner["flags"]),
            )
            != (40, PRECISION_FLOAT16, PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL)
            or (
                int(plan["planned_kernel"]),
                int(plan["precision"]),
                int(plan["packed_value"]),
                int(plan["flags"]),
            )
            != (
                40,
                PRECISION_FLOAT16,
                weight_id,
                PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL,
            )
        ):
            continue

        target_input = int(command["inputs"][0])
        if value_shape(target_input) != half_shape:
            continue
        view_index = producers[target_input]
        if view_index < 0:
            continue
        view = commands[view_index]
        if (
            view["kind"] != "VIEW"
            or len(view["inputs"]) != 1
            or int(view["output"]) != target_input
            or tuple(int(item) for item in view["params"]) != (51200, 51200)
        ):
            continue

        base_value = int(view["inputs"][0])
        if value_shape(base_value) != base_shape or len(consumers[base_value]) != 2:
            continue
        view_indices = tuple(consumers[base_value])
        if any(
            commands[index]["kind"] != "VIEW"
            or len(commands[index]["inputs"]) != 1
            or int(commands[index]["inputs"][0]) != base_value
            or value_shape(int(commands[index]["output"])) != half_shape
            for index in view_indices
        ):
            continue
        views_by_range = {
            tuple(int(item) for item in commands[index]["params"]): index
            for index in view_indices
        }
        if set(views_by_range) != {(0, 51200), (51200, 51200)}:
            continue
        first_view = commands[views_by_range[(0, 51200)]]
        second_view = commands[views_by_range[(51200, 51200)]]
        if int(second_view["output"]) != target_input:
            continue

        base_index = producers[base_value]
        if base_index < 0:
            continue
        base = commands[base_index]
        base_owner = physical_owners.get(base_index)
        if (
            base["kind"] != "CONV_SILU"
            or _conv_desc(base) != base_desc
            or len(base["inputs"]) != 3
            or base_owner is None
            or int(base_owner["fusion_kind"]) != FUSION_LATE_CONCAT_CONV1X1
            or int(base_owner["execution_index"]) != base_index
            or int(base_owner["kernel"]) != 3
            or int(base_owner["precision"]) != PRECISION_FLOAT16
            or int(base_owner["flags"])
            != PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
        ):
            continue
        base_logicals = tuple(int(item) for item in base_owner["logical_indices"])
        if len(base_logicals) != 2 or base_index not in base_logicals:
            continue
        concat_index = next(index for index in base_logicals if index != base_index)
        concat = commands[concat_index]
        concat_input_shapes = tuple(value_shape(int(value_id)) for value_id in concat["inputs"])
        if (
            concat["kind"] != "CONCAT"
            or int(concat["output"]) != int(base["inputs"][0])
            or value_shape(int(concat["output"])) != (1, 384, 20, 20)
            or concat_input_shapes
            != ((1, 128, 20, 20), (1, 256, 20, 20))
        ):
            continue

        target_output = int(command["output"])
        if len(consumers[target_output]) != 1:
            continue
        successor_index = consumers[target_output][0]
        successor = commands[successor_index]
        successor_owner = singleton_owner(successor_index)
        if (
            successor["kind"] != "CONV_SILU"
            or _conv_desc(successor) != successor_desc
            or len(successor["inputs"]) != 3
            or int(successor["inputs"][0]) != target_output
            or successor_owner is None
            or (
                int(successor_owner["kernel"]),
                int(successor_owner["precision"]),
                int(successor_owner["flags"]),
            )
            != (25, PRECISION_FLOAT16, PLAN_FLAG_AUTHORITATIVE)
        ):
            continue

        successor_output = int(successor["output"])
        if len(consumers[successor_output]) != 1:
            continue
        add_index = consumers[successor_output][0]
        add = commands[add_index]
        if (
            not _is_binary_add(add)
            or len(add["inputs"]) != 2
            or {int(value_id) for value_id in add["inputs"]}
            != {target_input, successor_output}
            or len(consumers[int(add["output"])]) != 1
        ):
            continue

        tail_index = consumers[int(add["output"])][0]
        tail = commands[tail_index]
        first_output = int(first_view["output"])
        if (
            tail["kind"] != "CONCAT_CONV1X1"
            or not _uses_native_fp16_residual_concat_conv1x1(tail)
            or tuple(int(value_id) for value_id in tail["inputs"][:3])
            != (first_output, target_input, int(add["output"]))
        ):
            continue
        tail_owner = physical_owners.get(tail_index)
        if (
            tail_owner is None
            or int(tail_owner["fusion_kind"]) != FUSION_CONCAT_RESIDUAL_CV2
            or int(tail_owner["execution_index"]) != tail_index
            or tuple(int(item) for item in tail_owner["logical_indices"])
            != (add_index, tail_index)
            or (
                int(tail_owner["kernel"]),
                int(tail_owner["precision"]),
                int(tail_owner["flags"]),
            )
            != (3, PRECISION_FLOAT16, PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL)
        ):
            continue

        plan["planned_kernel"] = 25
        plan["packed_value"] = NO_VALUE
        plan["flags"] = PLAN_FLAG_AUTHORITATIVE
        changed = True

    return changed


def _uses_native_fp16_winograd(params: Sequence[int]) -> bool:
    if len(params) < 16:
        return False
    _, ic, _, _, _, oh, ow, kh, kw, sh, sw, pt, pl, dh, dw, groups = (
        int(value) for value in params[:16]
    )
    return (
        kh == kw == 3
        and sh == sw == 1
        and pt == pl == 1
        and dh == dw == 1
        and groups == 1
        and ic > 0
        and ic % 32 == 0
        and oh > 0
        and ow > 0
        and oh % 2 == 0
        and ow % 2 == 0
    )


def _stable_conv_algorithm(params: Sequence[int], *, silu: bool) -> int:
    if len(params) < 16:
        return 0
    _, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, pt, pl, dh, dw, groups = (int(v) for v in params[:16])
    if kh == kw == sh == sw == dh == dw == 1 and pt == pl == 0 and groups == 1 and ih == oh and iw == ow:
        return 1
    if (
        silu
        and (ic, ih, iw, oc, oh, ow) == (3, 320, 320, 32, 160, 160)
        and kh == kw == 6
        and sh == sw == 2
        and pt == pl == 2
        and dh == dw == groups == 1
    ):
        return 42
    if not silu or kh != 3 or kw != 3 or pt != 1 or pl != 1 or dh != 1 or dw != 1 or groups != 1:
        return 0
    if sh == sw == 2 and ih >= oh * 2 - 1 and iw >= ow * 2 - 1:
        if (oh, ow, ic, oc) in {(160, 160, 3, 16), (80, 80, 16, 32)}:
            return 24
        if (oh, ow, ic, oc) == (40, 40, 128, 128):
            return 33
        if (oh, ow, ic, oc) == (20, 20, 256, 256):
            return 32
        if ic >= 32 and oc >= 64 and oh * ow >= 100:
            return 24
        return 0
    if sh != 1 or sw != 1 or ih != oh or iw != ow:
        return 0
    if (oh, ow, ic, oc) in {
        (40, 40, 64, 64),
        (40, 40, 128, 64),
    }:
        return 39
    if (oh, ow, ic, oc) in {
        (20, 20, 128, 64),
        (20, 20, 128, 128),
        (20, 20, 256, 64),
        (20, 20, 64, 64),
        (10, 10, 256, 64),
        (10, 10, 128, 128),
        (10, 10, 64, 64),
    }:
        return 40
    if (oh, ow, ic, oc) == (10, 10, 512, 64):
        return 28
    if (oh, ow, ic, oc) == (20, 20, 64, 128):
        return 25
    if oh == ow == 10 and oc == 64:
        return 2
    if ic >= 64 and oc >= 64 and oh * ow <= 400:
        if (oh, ow, ic, oc) == (20, 20, 128, 64):
            return 12
        if (oh, ow, ic, oc) == (20, 20, 64, 64):
            return 10
        if oh == ow == 20 and ic % 32 == 0 and oc % 16 == 0:
            return 7
        if (oh, ow, ic, oc) == (10, 10, 256, 64):
            return 13
        if (oh, ow, ic, oc) == (10, 10, 128, 128):
            return 11
        if (oh, ow, ic, oc) == (10, 10, 256, 256):
            return 46
        if oh == ow == 10 and ic % 32 == 0 and oc % 16 == 0:
            return 8
        return 5
    return 3 if oc >= 4 and oh * ow >= 100 else 2


def _uses_measured_native_fp16(params: Sequence[int]) -> bool:
    if len(params) < 16:
        return False
    _, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, pt, pl, dh, dw, groups = (int(v) for v in params[:16])
    if (
        kh != 3
        or kw != 3
        or sh != 2
        or sw != 2
        or pt != 1
        or pl != 1
        or dh != 1
        or dw != 1
        or groups != 1
        or ic < 32
        or ic % 4 != 0
        or oc % 4 != 0
        or ih < oh * 2 - 1
        or iw < ow * 2 - 1
    ):
        return False
    return (oh == ow == 10) or (oh == ow == 20 and oc <= 128)


def _uses_native_fp16_position_pair_40x40_256x256(params: Sequence[int]) -> bool:
    if len(params) < 16:
        return False
    return tuple(int(value) for value in params[:16]) == (
        1,
        256,
        40,
        40,
        256,
        20,
        20,
        3,
        3,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
    )


def _uses_native_fp16_pos2_40x40_64x64(params: Sequence[int]) -> bool:
    if len(params) < 16:
        return False
    batch, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, pt, pl, dh, dw, groups = (
        int(v) for v in params[:16]
    )
    return (
        batch >= 1
        and (ic, ih, iw, oc, oh, ow) == (64, 40, 40, 64, 40, 40)
        and kh == kw == 3
        and sh == sw == 1
        and pt == pl == 1
        and dh == dw == 1
        and groups == 1
    )


def _build_packed_weights(
    replay: dict[str, Any],
    plans: Sequence[dict[str, int]],
    *,
    precision: int,
    fusion_plan: Sequence[dict[str, int]] = (),
) -> list[dict[str, Any]]:
    values = replay["values"]
    packed: dict[int, dict[str, Any]] = {}
    for command, plan in zip(replay["commands"], plans):
        # INT8 dot4（51/52/53）与 Winograd F(4x4)（54）：layout 8/9 打包由
        # kernel plan 单点驱动（algo-map 固化路径），加载器按 planned_kernel
        # 强一致校验；两族权重域变换均与引擎 precision 无关（fp16 引擎先做
        # fp16 预舍入，与运行时 cpu 镜像位级一致）。
        planned_kernel_new = int(plan["planned_kernel"])
        if command["kind"] in {"CONV", "CONV_SILU"} and planned_kernel_new in {51, 52, 53}:
            params = command["params"]
            _add_packed_conv_int8(
                packed,
                values,
                weight_id=int(command["inputs"][1]),
                in_channels=int(params[1]),
                out_channels=int(params[4]),
                kernel_elems=1 if planned_kernel_new == 51 else 9,
                precision=precision,
            )
        elif command["kind"] in {"CONV", "CONV_SILU"} and planned_kernel_new == 54:
            params = command["params"]
            _add_packed_winograd_f4x4_fp16(
                packed,
                values,
                weight_id=int(command["inputs"][1]),
                in_channels=int(params[1]),
                out_channels=int(params[4]),
                precision=precision,
            )
        if precision == PRECISION_FLOAT16:
            if (
                command["kind"] in {"CONV", "CONV_SILU"}
                and int(plan["planned_kernel"]) == 47
            ):
                params = command["params"]
                _add_packed_conv3x3_k_major_oc4_fp16(
                    packed,
                    values,
                    weight_id=int(command["inputs"][1]),
                    in_channels=int(params[1]),
                    out_channels=int(params[4]),
                )
            elif command["kind"] in {"CONV", "CONV_SILU"} and int(plan["planned_kernel"]) == 45:
                params = command["params"]
                _add_packed_conv1x1_fp16(
                    packed,
                    values,
                    weight_id=int(command["inputs"][1]),
                    in_channels=int(params[1]),
                    out_channels=int(params[4]),
                )
            elif command["kind"] == "CONCAT_CONV1X1" and int(plan["planned_kernel"]) == 3:
                input_count = len(command["inputs"]) - 2
                _add_packed_conv1x1_fp16(
                    packed,
                    values,
                    weight_id=int(command["inputs"][-2]),
                    in_channels=sum(int(value) for value in command["params"][5 : 5 + input_count]),
                    out_channels=int(command["params"][3]),
                )
            elif (
                command["kind"] in {"CONV", "CONV_SILU"}
                and int(plan["planned_kernel"]) == 40
                and int(plan["flags"]) & PLAN_FLAG_DXIL
            ):
                params = command["params"]
                _add_packed_winograd_f2x2_fp16(
                    packed,
                    values,
                    weight_id=int(command["inputs"][1]),
                    in_channels=int(params[1]),
                    out_channels=int(params[4]),
                )
            elif (
                command["kind"] in {"CONV", "CONV_SILU"}
                and int(plan["planned_kernel"]) == 46
                and int(plan["flags"]) & PLAN_FLAG_DXIL
            ):
                params = command["params"]
                _add_packed_winograd_f2x2_fp32(
                    packed,
                    values,
                    weight_id=int(command["inputs"][1]),
                    in_channels=int(params[1]),
                    out_channels=int(params[4]),
                )
        if command["kind"] not in {"CONV", "CONV_SILU"} or len(command["params"]) < 16:
            continue
        params = command["params"]
        ic, oc, kh, kw, sh, sw, groups = params[1], params[4], params[7], params[8], params[9], params[10], params[15]
        stride2_candidate = sh == 2 and sw == 2
        planned_packed_stride1 = int(plan["planned_kernel"]) in {39, 44}
        if kh != 3 or kw != 3 or not (stride2_candidate or planned_packed_stride1) or groups != 1 or oc % 4 != 0:
            continue
        weight_id = int(command["inputs"][1])
        if weight_id in packed:
            continue
        raw = values[weight_id]["raw"]
        weights = np.frombuffer(raw, dtype="<f4")
        if weights.size != oc * ic * 9:
            raise ValueError("conv weight payload does not match its descriptor")
        packed_array = (
            weights.reshape(oc // 4, 4, ic, 9)
            .transpose(0, 2, 3, 1)
            .copy()
        )
        data = packed_array.astype("<f2" if precision == PRECISION_FLOAT16 else "<f4").tobytes(order="C")
        packed[weight_id] = {
            "value_id": weight_id,
            "layout": PACKED_LAYOUT_CONV3X3_OC4_FP16 if precision == PRECISION_FLOAT16 else PACKED_LAYOUT_CONV3X3_OC4,
            "in_channels": int(ic),
            "out_channels": int(oc),
            "elements": int(packed_array.size),
            "data": data,
        }

    if precision == PRECISION_FLOAT16:
        for group in fusion_plan:
            if int(group["kind"]) != FUSION_LATTICE_CHAIN_C3:
                continue
            start = int(group["start"])
            end = int(group["end"])
            if end != start + 5:
                raise ValueError("invalid lattice-chain packed-weight range")
            for command_index in (start + 1, start + 2, start + 4):
                command = replay["commands"][command_index]
                params = command["params"]
                _add_packed_conv1x1_fp16(
                    packed,
                    values,
                    weight_id=int(command["inputs"][1]),
                    in_channels=int(params[1]),
                    out_channels=int(params[4]),
                )
            merge = replay["commands"][end]
            _add_packed_conv1x1_fp16(
                packed,
                values,
                weight_id=int(merge["inputs"][-2]),
                in_channels=sum(int(value) for value in merge["params"][5:]),
                out_channels=int(merge["params"][3]),
            )

        for group in fusion_plan:
            if int(group["kind"]) != FUSION_LATE_CONCAT_CONV1X1 or int(group["kernel"]) != 3:
                continue
            command = replay["commands"][int(group["end"])]
            params = command["params"]
            _add_packed_conv1x1_fp16(
                packed,
                values,
                weight_id=int(command["inputs"][1]),
                in_channels=int(params[1]),
                out_channels=int(params[4]),
            )

        for group in fusion_plan:
            kind = int(group["kind"])
            if int(group["kernel"]) != 40 or not (int(group["flags"]) & PLAN_FLAG_DXIL):
                continue
            if kind == FUSION_PAIRED_CONV3X3_SILU:
                command_indices = (int(group["start"]), int(group["end"]))
            elif kind == FUSION_C2F_TAIL_RESIDUAL:
                command_indices = (int(group["start"]),)
            else:
                continue
            for command_index in command_indices:
                command = replay["commands"][command_index]
                params = command["params"]
                _add_packed_winograd_f2x2_fp16(
                    packed,
                    values,
                    weight_id=int(command["inputs"][1]),
                    in_channels=int(params[1]),
                    out_channels=int(params[4]),
                )

        for _, _, concat_index in _position_owned_winograd_residual_cv2_groups(
            replay, fusion_plan
        ):
            command = replay["commands"][concat_index]
            _add_packed_conv1x1_fp16(
                packed,
                values,
                weight_id=int(command["inputs"][-2]),
                in_channels=sum(int(value) for value in command["params"][5:]),
                out_channels=int(command["params"][3]),
            )

        for _, _, _, concat_index in _position_owned_winograd_residual_branch_cv2_groups(
            replay
        ):
            command = replay["commands"][concat_index]
            _add_packed_conv1x1_fp16(
                packed,
                values,
                weight_id=int(command["inputs"][-2]),
                in_channels=sum(int(value) for value in command["params"][5:]),
                out_channels=int(command["params"][3]),
            )
    return [packed[key] for key in sorted(packed)]


def _add_packed_conv3x3_k_major_oc4_fp16(
    packed: dict[int, dict[str, Any]],
    values: Sequence[dict[str, Any]],
    *,
    weight_id: int,
    in_channels: int,
    out_channels: int,
) -> None:
    existing = packed.get(weight_id)
    if existing is not None:
        if (
            int(existing["layout"]) != PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16
            or int(existing["in_channels"]) != in_channels
            or int(existing["out_channels"]) != out_channels
        ):
            raise ValueError("K-major Conv3x3 weight already has an incompatible packed layout")
        return
    if in_channels % 4 != 0 or out_channels % 4 != 0:
        raise ValueError("K-major Conv3x3 packing requires channels divisible by four")
    source = np.frombuffer(values[weight_id]["raw"], dtype="<f4")
    if source.size != out_channels * in_channels * 9:
        raise ValueError("Conv3x3 weight payload does not match its descriptor")
    packed_array = (
        source.reshape(out_channels // 4, 4, in_channels, 9)
        .transpose(2, 3, 0, 1)
        .copy()
    )
    packed[weight_id] = {
        "value_id": weight_id,
        "layout": PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "elements": int(packed_array.size),
        "data": packed_array.astype("<f2").tobytes(order="C"),
    }


def _add_packed_winograd_f2x2_fp16(
    packed: dict[int, dict[str, Any]],
    values: Sequence[dict[str, Any]],
    *,
    weight_id: int,
    in_channels: int,
    out_channels: int,
) -> None:
    existing = packed.get(weight_id)
    if existing is not None:
        if (
            int(existing["layout"]) != PACKED_LAYOUT_WINOGRAD_F2X2_FP16
            or int(existing["in_channels"]) != in_channels
            or int(existing["out_channels"]) != out_channels
        ):
            raise ValueError("Winograd weight already has an incompatible packed layout")
        return
    source = np.frombuffer(values[weight_id]["raw"], dtype="<f4")
    if source.size != out_channels * in_channels * 9:
        raise ValueError("Winograd weight payload does not match its descriptor")
    source = source.astype("<f2").astype(np.float32).reshape(out_channels, in_channels, 3, 3)
    transform = np.asarray(
        ((1.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    transformed = np.einsum("ak,oikl,bl->oiab", transform, source, transform, optimize=True)
    packed[weight_id] = {
        "value_id": weight_id,
        "layout": PACKED_LAYOUT_WINOGRAD_F2X2_FP16,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "elements": int(transformed.size),
        "data": transformed.astype("<f2").tobytes(order="C"),
    }


def _add_packed_winograd_f2x2_fp32(
    packed: dict[int, dict[str, Any]],
    values: Sequence[dict[str, Any]],
    *,
    weight_id: int,
    in_channels: int,
    out_channels: int,
) -> None:
    existing = packed.get(weight_id)
    if existing is not None:
        if (
            int(existing["layout"]) != PACKED_LAYOUT_WINOGRAD_F2X2_FP32
            or int(existing["in_channels"]) != in_channels
            or int(existing["out_channels"]) != out_channels
        ):
            raise ValueError("Winograd weight already has an incompatible packed layout")
        return
    source = np.frombuffer(values[weight_id]["raw"], dtype="<f4")
    if source.size != out_channels * in_channels * 9:
        raise ValueError("Winograd weight payload does not match its descriptor")
    source = source.astype("<f2").astype(np.float32).reshape(out_channels, in_channels, 3, 3)
    transform = np.asarray(
        ((1.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    transformed = np.einsum("ak,oikl,bl->oiab", transform, source, transform, optimize=True)
    packed[weight_id] = {
        "value_id": weight_id,
        "layout": PACKED_LAYOUT_WINOGRAD_F2X2_FP32,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "elements": int(transformed.size),
        "data": transformed.astype("<f4").tobytes(order="C"),
    }


def _add_packed_conv_int8(
    packed: dict[int, dict[str, Any]],
    values: Sequence[dict[str, Any]],
    *,
    weight_id: int,
    in_channels: int,
    out_channels: int,
    kernel_elems: int,
    precision: int,
) -> None:
    """Per-channel 对称量化（kernel 51/52/53，layout 8）。

    与运行时 ensure_conv_int8_weight 的即时量化位级一致：
    scale_oc = max|w_oc| / 127（amax==0 时 scale=1），q = round_half_away(w/scale)
    截断到 [-127,127]，int8 按 [oc][ic][kk] 原序存储（uint 重排发生在
    GPU 上传阶段）。fp16 引擎中常量权重先经 fp16 舍入（cpu 镜像语义）。
    """
    existing = packed.get(weight_id)
    if existing is not None:
        if (
            int(existing["layout"]) != PACKED_LAYOUT_CONV_INT8
            or int(existing["in_channels"]) != in_channels
            or int(existing["out_channels"]) != out_channels
        ):
            raise ValueError("conv int8 weight already has an incompatible packed layout")
        return
    source = np.frombuffer(values[weight_id]["raw"], dtype="<f4")
    if source.size != out_channels * in_channels * kernel_elems:
        raise ValueError("conv int8 weight payload does not match its descriptor")
    if precision == PRECISION_FLOAT16:
        source = source.astype("<f2").astype(np.float32)
    source = source.reshape(out_channels, in_channels * kernel_elems)
    amax = np.abs(source).max(axis=1) if out_channels else np.zeros(0, dtype=np.float32)
    scale = np.where(amax > 0.0, amax / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    inv = (np.float32(1.0) / scale).astype(np.float32)
    product = (source * inv[:, None]).astype(np.float32)
    # C++ std::round 为 half-away-from-zero；np.rint 是 half-to-even，需手动对齐。
    quantized = np.sign(product) * np.floor(np.abs(product) + np.float32(0.5))
    quantized = np.clip(quantized, np.float32(-127.0), np.float32(127.0)).astype(np.int8)
    data = quantized.tobytes(order="C") + scale.astype("<f4").tobytes(order="C")
    packed[weight_id] = {
        "value_id": weight_id,
        "layout": PACKED_LAYOUT_CONV_INT8,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "elements": int(quantized.size),
        "data": data,
    }


def _add_packed_winograd_f4x4_fp16(
    packed: dict[int, dict[str, Any]],
    values: Sequence[dict[str, Any]],
    *,
    weight_id: int,
    in_channels: int,
    out_channels: int,
    precision: int,
) -> None:
    """Winograd F(4x4,3x3) 权重域变换（kernel 54，layout 9）。

    U = G·g·Gᵀ，G 为 wincnn 规范 F(4,3) 变换矩阵（6x3），与运行时
    ensure_winograd_f4x4_fp16_weight 的逐元素变换一致；fp16 引擎先对源
    权重做 fp16 预舍入（cpu 镜像语义），变换结果存 fp16。
    """
    existing = packed.get(weight_id)
    if existing is not None:
        if (
            int(existing["layout"]) != PACKED_LAYOUT_WINOGRAD_F4X4_FP16
            or int(existing["in_channels"]) != in_channels
            or int(existing["out_channels"]) != out_channels
        ):
            raise ValueError("Winograd F(4x4) weight already has an incompatible packed layout")
        return
    source = np.frombuffer(values[weight_id]["raw"], dtype="<f4")
    if source.size != out_channels * in_channels * 9:
        raise ValueError("Winograd F(4x4) weight payload does not match its descriptor")
    if precision == PRECISION_FLOAT16:
        source = source.astype("<f2").astype(np.float32)
    source = source.reshape(out_channels, in_channels, 3, 3)
    transform = np.asarray(
        (
            (0.25, 0.0, 0.0),
            (-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0),
            (-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0),
            (1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0),
            (1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    transformed = np.einsum("ak,oikl,bl->oiab", transform, source, transform, optimize=True)
    packed[weight_id] = {
        "value_id": weight_id,
        "layout": PACKED_LAYOUT_WINOGRAD_F4X4_FP16,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "elements": int(transformed.size),
        "data": transformed.astype("<f2").tobytes(order="C"),
    }


def _add_packed_conv1x1_fp16(
    packed: dict[int, dict[str, Any]],
    values: Sequence[dict[str, Any]],
    *,
    weight_id: int,
    in_channels: int,
    out_channels: int,
) -> None:
    existing = packed.get(weight_id)
    if existing is not None:
        if (
            int(existing["layout"]) != PACKED_LAYOUT_CONV1X1_OC8_FP16
            or int(existing["in_channels"]) != in_channels
            or int(existing["out_channels"]) != out_channels
        ):
            raise ValueError("conv1x1 weight already has an incompatible packed layout")
        return
    raw = values[weight_id]["raw"]
    weights = np.frombuffer(raw, dtype="<f4")
    if weights.size != out_channels * in_channels:
        raise ValueError("conv1x1 weight payload does not match its descriptor")
    packed_array = (
        weights.reshape(out_channels // 8, 8, in_channels)
        .transpose(0, 2, 1)
        .copy()
    )
    packed[weight_id] = {
        "value_id": weight_id,
        "layout": PACKED_LAYOUT_CONV1X1_OC8_FP16,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "elements": int(packed_array.size),
        "data": packed_array.astype("<f2").tobytes(order="C"),
    }


def _encode_manifest(**fields: Any) -> bytes:
    out = bytearray()
    for value in (
        1,
        1,
        fields["precision"],
        2,
        fields["layout"],
        int(fields["has_objectness"]),
        fields["channels"],
        fields["anchors"],
        fields["classes"],
        fields["max_candidates"],
        fields["max_detections"],
        fields["graph_node_count"],
        fields["value_count"],
        fields["constant_count"],
        fields["command_count"],
        fields["input_value"],
        fields["output_value"],
    ):
        out += struct.pack("<I", int(value))
    out += struct.pack("<QQff", fields["input_elements"], fields["arena_nbytes"], fields["conf_threshold"], fields["iou_threshold"])
    out += bytes(fields["source_hash"])
    return bytes(out)


def _encode_values(replay: dict[str, Any], memory_plan: dict[str, Any]) -> bytes:
    out = bytearray(struct.pack("<II", len(replay["values"]), 0))
    for value in replay["values"]:
        record = memory_plan["by_value"].get(value["id"])
        offset = record["offset"] if record else NO_OFFSET
        nbytes = record["nbytes"] if record else 0
        out += struct.pack(
            "<IIQ4IQQ",
            value["id"],
            record["flags"] if record else value["flags"],
            value["elements"],
            *value["shape4"],
            offset,
            nbytes,
        )
    return bytes(out)


def _encode_commands(replay: dict[str, Any], kernel_plan: list[dict[str, int]]) -> bytes:
    out = bytearray(struct.pack("<II", len(replay["commands"]), 0))
    for command, plan in zip(replay["commands"], kernel_plan):
        out += struct.pack(
            "<6I",
            command["kind_id"],
            command["output"],
            len(command["inputs"]),
            len(command["params"]),
            plan["planned_kernel"],
            plan["precision"],
        )
        if command["inputs"]:
            out += struct.pack(f"<{len(command['inputs'])}I", *command["inputs"])
        if command["params"]:
            out += struct.pack(f"<{len(command['params'])}I", *command["params"])
    return bytes(out)


def _encode_constants(replay: dict[str, Any], *, precision: int) -> bytes:
    constants = [value for value in replay["values"] if value["flags"] & VALUE_CONSTANT]
    fp16_values: set[int] = set()
    if precision == PRECISION_FLOAT16:
        for command in replay["commands"]:
            if command["kind"] in {"CONV", "CONV_SILU"} and len(command["inputs"]) >= 3:
                fp16_values.update(int(value_id) for value_id in command["inputs"][1:3])
            elif command["kind"] == "CONCAT_CONV1X1" and len(command["inputs"]) >= 2:
                fp16_values.update(int(value_id) for value_id in command["inputs"][-2:])
            elif command["kind"] == "MATMUL" and len(command["inputs"]) >= 2:
                fp16_values.add(int(command["inputs"][1]))
    record_size = 32
    data_offset = _align(8 + len(constants) * record_size, ENGINE_ALIGNMENT)
    out = bytearray(data_offset)
    struct.pack_into("<II", out, 0, len(constants), 0)
    cursor = data_offset
    for index, value in enumerate(constants):
        cursor = _align(cursor, ENGINE_ALIGNMENT)
        if cursor > len(out):
            out.extend(b"\0" * (cursor - len(out)))
        raw = value["raw"]
        value_precision = PRECISION_FLOAT16 if value["id"] in fp16_values else PRECISION_FLOAT32
        if value_precision == PRECISION_FLOAT16:
            raw = np.frombuffer(raw, dtype="<f4").astype("<f2").tobytes(order="C")
        struct.pack_into("<IIQQQ", out, 8 + index * record_size, value["id"], value_precision, value["elements"], cursor, len(raw))
        out.extend(raw)
        cursor += len(raw)
    return bytes(out)


def _encode_kernel_plan(plans: list[dict[str, int]]) -> bytes:
    out = bytearray(struct.pack("<II", len(plans), 1))
    for plan in plans:
        out += struct.pack(
            "<6I",
            plan["command_index"],
            plan["kind_id"],
            plan["planned_kernel"],
            plan["precision"],
            plan["packed_value"],
            plan["flags"],
        )
    return bytes(out)


def _encode_packed_weights(packed: list[dict[str, Any]]) -> bytes:
    record_size = 40
    data_offset = _align(8 + len(packed) * record_size, ENGINE_ALIGNMENT)
    out = bytearray(data_offset)
    struct.pack_into("<II", out, 0, len(packed), 0)
    cursor = data_offset
    for index, item in enumerate(packed):
        cursor = _align(cursor, ENGINE_ALIGNMENT)
        if cursor > len(out):
            out.extend(b"\0" * (cursor - len(out)))
        data = item["data"]
        struct.pack_into(
            "<4IQQQ",
            out,
            8 + index * record_size,
            item["value_id"],
            item["layout"],
            item["in_channels"],
            item["out_channels"],
            item["elements"],
            cursor,
            len(data),
        )
        out.extend(data)
        cursor += len(data)
    return bytes(out)


def _encode_memory_plan(plan: dict[str, Any]) -> bytes:
    out = bytearray(struct.pack("<IIQ", plan["alignment"], len(plan["records"]), plan["total_nbytes"]))
    for record in sorted(plan["records"], key=lambda item: item["value_id"]):
        out += struct.pack("<IIQQ", record["value_id"], record["flags"], record["offset"], record["nbytes"])
    return bytes(out)


def _build_fusion_plan(replay: dict[str, Any], *, precision: int, classes: int) -> list[dict[str, int]]:
    commands = replay["commands"]
    values = replay["values"]
    consumer_counts = [0] * len(values)
    producers = [-1] * len(values)
    for index, command in enumerate(commands):
        producers[int(command["output"])] = index
        for value_id in command["inputs"]:
            consumer_counts[int(value_id)] += 1

    groups: list[dict[str, int]] = []
    fusion_precision = precision

    if precision == PRECISION_FLOAT16:
        for start, spatial, entry_channels, hidden_channels, output_channels in (
            _v5_neck_lattice_chain_groups(replay)
        ):
            group = _fusion_record(
                FUSION_LATTICE_CHAIN_C3,
                start,
                start + 5,
                PRECISION_FLOAT16,
            )
            group["flags"] = PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
            # Section 8 owns the selected physical variant.  Packing the four
            # bounded dimensions avoids runtime model-name or command-id
            # switches while keeping the existing 32-byte group record.
            group["aux0"] = spatial | (hidden_channels << 16)
            group["aux1"] = entry_channels | (output_channels << 16)
            groups.append(group)

    import os as _os
    for index in range(len(commands) - 1):
        first = commands[index]
        second = commands[index + 1]
        d0 = _conv_desc(first)
        d1 = _conv_desc(second)
        if (
            precision == PRECISION_FLOAT16
            and first["kind"] == second["kind"] == "CONV_SILU"
            and len(first["inputs"]) >= 3
            and len(second["inputs"]) >= 3
            and first["inputs"][0] == second["inputs"][0]
            and first["output"] != second["output"]
            and d0 is not None
            and d1 is not None
            and _is_conv3x3_tiled(d0)
            and _is_conv3x3_tiled(d1)
            and d0 == d1
            and d0[0:7] == (1, 256, 10, 10, 64, 10, 10)
            and not _os.environ.get("AEXRTC_UNFUSE_PAIRED_CONV")
        ):
            group = _fusion_record(
                FUSION_PAIRED_CONV3X3_SILU, index, index + 1, PRECISION_FLOAT16
            )
            group["kernel"] = 40
            group["flags"] = PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
            groups.append(group)

    for index in range(len(commands) - 1):
        conv1 = commands[index]
        add = commands[index + 1]
        d1 = _conv_desc(conv1)
        if conv1["kind"] != "CONV_SILU" or not conv1["inputs"]:
            continue
        conv0_index = producers[int(conv1["inputs"][0])]
        if conv0_index < 0 or conv0_index >= index:
            continue
        conv0 = commands[conv0_index]
        d0 = _conv_desc(conv0)
        sparse_gap_ok = True
        if conv0_index + 1 != index:
            sparse_gap_ok = False
            if conv0_index + 3 == index:
                detection_conv = commands[conv0_index + 1]
                detection_alias = commands[conv0_index + 2]
                detection_desc = _conv_desc(detection_conv)
                sparse_gap_ok = (
                    d0 is not None
                    and d1 is not None
                    and d0[0:7] == d1[0:7] == (1, 128, 10, 10, 128, 10, 10)
                    and detection_conv["kind"] == "CONV"
                    and detection_desc is not None
                    and detection_desc[0:7] == (1, 128, 20, 20, 9, 20, 20)
                    and detection_desc[7:16] == (1, 1, 1, 1, 0, 0, 1, 1, 1)
                    and detection_alias["kind"] == "ALIAS"
                    and len(detection_alias["inputs"]) == 1
                    and int(detection_alias["inputs"][0])
                    == int(detection_conv["output"])
                )
        if (
            conv0["kind"] == conv1["kind"] == "CONV_SILU"
            and sparse_gap_ok
            and _is_binary_add(add)
            and len(conv0["inputs"]) >= 3
            and len(conv1["inputs"]) >= 3
            and conv0["output"] == conv1["inputs"][0]
            and consumer_counts[int(conv0["output"])] == 1
            and consumer_counts[int(conv1["output"])] == 1
            and int(conv0["inputs"][0]) in add["inputs"]
            and int(conv1["output"]) in add["inputs"]
            and d0 is not None
            and d1 is not None
            and (_is_conv1x1_tiled(d0) or _is_conv3x3_tiled(d0))
            and _is_conv3x3_tiled(d1)
            and d0[0] == d1[0]
            and d0[4] == d1[1]
            and d0[5:7] == d1[2:4]
            and d1[4] == d0[1]
            and d1[5:7] == d0[2:4]
            # These v5 tails are faster as their separately planned kernels.
            and d1[1:7]
            not in {
                (64, 40, 40, 64, 40, 40),
                (256, 10, 10, 256, 10, 10),
            }
        ):
            if precision == PRECISION_FLOAT16:
                # Build switch: leave the tail conv standalone so the int8
                # algo map can claim it (residual add becomes a plain BINARY).
                import os as _os
                if _os.environ.get("AEXRTC_UNFUSE_C2F_TAIL"):
                    continue
                # 运行时物理白名单只接受 typed（winograd kernel=40 + DXIL）的
                # fp16 C2F tail 融合组；其余组合会被 fail-closed 拒绝，
                # 因此无法 typed 化时直接不产融合组（退化为逐命令执行）。
                import os as _os
                if _os.environ.get("AEXRT_DEBUG_C2F"):
                    import sys as _sys
                    stable = _stable_conv_algorithm(d1, silu=True)
                    nat = _uses_native_fp16_winograd(d1)
                    if not (stable == 40 and nat):
                        print(
                            f"[c2f-drop] cmd#{index} d1={tuple(int(v) for v in d1[1:7])} "
                            f"stable={stable} native_fp16_winograd={nat}",
                            file=_sys.stderr,
                        )
                if _stable_conv_algorithm(d1, silu=True) == 40 and _uses_native_fp16_winograd(d1):
                    group = _fusion_record(FUSION_C2F_TAIL_RESIDUAL, index, index + 1, fusion_precision)
                    group["kernel"] = 40
                    group["flags"] = PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
                    groups.append(group)
                elif _uses_native_fp16_winograd(d1) and d1[1] == d1[4]:
                    # 放松分支：planner 单独执行不选 winograd，但形状支持且
                    # 同通道（ic==oc）时融合仍划算（消掉 conv1/silu/conv2/add
                    # 四个 dispatch）。同通道限制：融合 winograd shader 形状
                    # 特化，扩通道变体（32→64 等）不在 shader cache，运行时
                    # PSO 查空段错误（实测 DYv11s ic≠oc 组即崩，cs2V8
                    # (32,40,40)→(32,40,40) 同通道组正常）。
                    group = _fusion_record(FUSION_C2F_TAIL_RESIDUAL, index, index + 1, fusion_precision)
                    group["kernel"] = 40
                    group["flags"] = PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
                    groups.append(group)
                continue
            groups.append(_fusion_record(FUSION_C2F_TAIL_RESIDUAL, index, index + 1, fusion_precision))

    for index, first in enumerate(commands):
        d0 = _conv_desc(first)
        if (
            first["kind"] != "CONV_SILU"
            or d0 is None
            or len(first["inputs"]) < 3
            or consumer_counts[int(first["output"])] != 1
            or d0[7:15] != (3, 3, 2, 2, 1, 1, 1, 1)
            or d0[15] != 1
            # Larger inputs lose to the separate planned kernels on the measured D3D12 path.
            or d0[1] > 64
            or d0[1] % 4 != 0
            or d0[4] % 4 != 0
            or d0[4] > 512
            or d0[5] * d0[6] < 100
        ):
            continue
        for second_index in range(index + 1, min(len(commands), index + 4)):
            second = commands[second_index]
            d1 = _conv_desc(second)
            if (
                second["kind"] != "CONV_SILU"
                or d1 is None
                or len(second["inputs"]) < 3
                or int(second["inputs"][0]) != int(first["output"])
                or d1[7:15] != (1, 1, 1, 1, 0, 0, 1, 1)
                or d1[15] != 1
                or d0[0] != d1[0]
                or d0[4] != d1[1]
                or d0[5:7] != d1[2:4]
                or d1[2:4] != d1[5:7]
                or d1[4] % 4 != 0
                or d1[4] > 512
            ):
                continue
            import os as _os
            if not _os.environ.get("AEXRTC_UNFUSE_STRIDE2_PAIR"):
                groups.append(_fusion_record(FUSION_STRIDE2_CONV1X1, index, second_index, PRECISION_FLOAT32))
            break

    import os as _os
    _unfuse_small = bool(_os.environ.get("AEXRTC_UNFUSE_SMALL_CONCAT"))
    for conv_index, conv in enumerate(commands):
        # Same env switch as the graph-side pattern (shape-aware): small-spatial
        # concat convs are left unfused for the int8 algo map to claim.
        desc = _conv_desc(conv)
        if _unfuse_small and desc is not None and desc[5] <= 20:
            continue
        if desc is None or conv["kind"] not in {"CONV", "CONV_SILU"} or len(conv["inputs"]) < 3:
            continue
        source = int(conv["inputs"][0])
        concat_index = producers[source] if source < len(producers) else -1
        if concat_index < 0 or consumer_counts[source] != 1:
            continue
        concat = commands[concat_index]
        if concat["kind"] != "CONCAT" or not concat["inputs"] or len(concat["inputs"]) > 8 or len(concat["params"]) < 18:
            continue
        if desc[7:11] != (1, 1, 1, 1) or desc[11:15] != (0, 0, 1, 1) or desc[15] != 1:
            continue
        if concat["params"][1] != 4 or concat["params"][2] != 1 or concat["params"][3] != len(concat["inputs"]):
            continue
        group = _fusion_record(
            FUSION_LATE_CONCAT_CONV1X1, concat_index, conv_index, fusion_precision
        )
        if precision == PRECISION_FLOAT16 and _uses_native_fp16_conv1x1(desc):
            group["precision"] = PRECISION_FLOAT16
            group["kernel"] = 3
            group["flags"] = PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
        groups.append(group)

    head = None if os.environ.get("AEXRTC_DISABLE_HEAD_FUSION") else _detect_yolo_head_group(replay, producers, classes=classes)
    if head is not None:
        start, end, _ = head
        groups.append(_fusion_record(FUSION_YOLO_HEAD, start, end, PRECISION_FLOAT32))
        # Type 5 remains reserved until its fused descriptor layout is valid for
        # prepared replay. Advertising it currently invalidates the whole plan.
    return sorted(groups, key=lambda item: (item["start"], item["kind"], item["end"]))


def _position_owned_winograd_residual_cv2_groups(
    replay: dict[str, Any], fusion_plan: Sequence[dict[str, int]]
) -> list[tuple[int, int, int]]:
    commands = replay["commands"]
    values = replay["values"]
    producers = [-1] * len(values)
    consumer_counts = [0] * len(values)
    for command_index, command in enumerate(commands):
        output = int(command["output"])
        if 0 <= output < len(producers):
            producers[output] = command_index
        for raw_value in command["inputs"]:
            value_id = int(raw_value)
            if 0 <= value_id < len(consumer_counts):
                consumer_counts[value_id] += 1

    def shape(value_id: int) -> tuple[int, ...] | None:
        if value_id < 0 or value_id >= len(values):
            return None
        result = tuple(int(item) for item in values[value_id].get("shape4", ()))
        return result if len(result) == 4 else None

    groups: list[tuple[int, int, int]] = []
    target_desc = (1, 128, 10, 10, 128, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1, 1)
    for tail in fusion_plan:
        conv_index = int(tail["start"])
        add_index = int(tail["end"])
        concat_index = add_index + 1
        if (
            int(tail["kind"]) != FUSION_C2F_TAIL_RESIDUAL
            or int(tail["kernel"]) != 40
            or int(tail["precision"]) != PRECISION_FLOAT16
            or int(tail["flags"]) != PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
            or add_index != conv_index + 1
            or concat_index >= len(commands)
        ):
            continue

        conv = commands[conv_index]
        add = commands[add_index]
        concat = commands[concat_index]
        if (
            conv["kind"] != "CONV_SILU"
            or _conv_desc(conv) != target_desc
            or len(conv["inputs"]) != 3
            or not _is_binary_add(add)
            or len(add["inputs"]) != 2
            or not add["params"]
            or int(add["params"][0]) != 12800
            or concat["kind"] != "CONCAT_CONV1X1"
            or len(concat["inputs"]) != 4
            or tuple(int(item) for item in concat["params"])
            != (1, 10, 10, 256, 1, 128, 128)
        ):
            continue

        conv_output = int(conv["output"])
        add_output = int(add["output"])
        if int(add["inputs"][0]) == conv_output:
            residual = int(add["inputs"][1])
        elif int(add["inputs"][1]) == conv_output:
            residual = int(add["inputs"][0])
        else:
            continue
        branch = int(concat["inputs"][1])
        conv_weight, conv_bias = (int(value_id) for value_id in conv["inputs"][1:3])
        cv2_weight, cv2_bias = (int(value_id) for value_id in concat["inputs"][2:4])
        if (
            int(concat["inputs"][0]) != add_output
            or producers[conv_output] != conv_index
            or producers[add_output] != add_index
            or consumer_counts[conv_output] != 1
            or consumer_counts[add_output] != 1
            or shape(int(conv["inputs"][0])) != (1, 128, 10, 10)
            or shape(conv_output) != (1, 128, 10, 10)
            or shape(residual) != (1, 128, 10, 10)
            or shape(add_output) != (1, 128, 10, 10)
            or shape(branch) != (1, 128, 10, 10)
            or shape(conv_weight) != (128, 128, 3, 3)
            or shape(conv_bias) != (1, 1, 1, 128)
            or shape(cv2_weight) != (256, 256, 1, 1)
            or shape(cv2_bias) != (1, 1, 1, 256)
            or shape(int(concat["output"])) != (1, 256, 10, 10)
            or any(
                not (int(values[value_id]["flags"]) & VALUE_CONSTANT)
                for value_id in (conv_weight, conv_bias, cv2_weight, cv2_bias)
            )
        ):
            continue
        groups.append((conv_index, add_index, concat_index))
    return groups


def _position_owned_winograd_residual_branch_cv2_groups(
    replay: dict[str, Any],
) -> list[tuple[int, int, int, int]]:
    commands = replay["commands"]
    values = replay["values"]
    producers = [-1] * len(values)
    consumer_counts = [0] * len(values)
    for command_index, command in enumerate(commands):
        output = int(command["output"])
        if 0 <= output < len(producers):
            producers[output] = command_index
        for raw_value in command["inputs"]:
            value_id = int(raw_value)
            if 0 <= value_id < len(consumer_counts):
                consumer_counts[value_id] += 1

    def shape(value_id: int) -> tuple[int, ...] | None:
        if value_id < 0 or value_id >= len(values):
            return None
        result = tuple(int(item) for item in values[value_id].get("shape4", ()))
        return result if len(result) == 4 else None

    conv_desc = (1, 128, 20, 20, 128, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1)
    branch_desc = (1, 256, 20, 20, 128, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1)
    groups: list[tuple[int, int, int, int]] = []
    for conv_index, conv in enumerate(commands):
        add_index = conv_index + 1
        branch_index = add_index + 1
        concat_index = branch_index + 1
        if concat_index >= len(commands):
            break
        add = commands[add_index]
        branch_command = commands[branch_index]
        concat = commands[concat_index]
        if (
            conv["kind"] != "CONV_SILU"
            or _conv_desc(conv) != conv_desc
            or len(conv["inputs"]) != 3
            or not _is_binary_add(add)
            or len(add["inputs"]) != 2
            or not add["params"]
            or int(add["params"][0]) != 51200
            or branch_command["kind"] != "CONV_SILU"
            or _conv_desc(branch_command) != branch_desc
            or len(branch_command["inputs"]) != 3
            or concat["kind"] != "CONCAT_CONV1X1"
            or len(concat["inputs"]) != 4
            or tuple(int(item) for item in concat["params"])
            != (1, 20, 20, 256, 1, 128, 128)
        ):
            continue

        conv_output = int(conv["output"])
        add_output = int(add["output"])
        branch_output = int(branch_command["output"])
        if int(add["inputs"][0]) == conv_output:
            residual = int(add["inputs"][1])
        elif int(add["inputs"][1]) == conv_output:
            residual = int(add["inputs"][0])
        else:
            continue
        conv_weight, conv_bias = (int(value_id) for value_id in conv["inputs"][1:3])
        branch_weight, branch_bias = (
            int(value_id) for value_id in branch_command["inputs"][1:3]
        )
        cv2_weight, cv2_bias = (int(value_id) for value_id in concat["inputs"][2:4])
        if (
            tuple(int(value_id) for value_id in concat["inputs"][:2])
            != (add_output, branch_output)
            or producers[conv_output] != conv_index
            or producers[add_output] != add_index
            or producers[branch_output] != branch_index
            or consumer_counts[conv_output] != 1
            or consumer_counts[add_output] != 1
            or consumer_counts[branch_output] != 1
            or shape(int(conv["inputs"][0])) != (1, 128, 20, 20)
            or shape(conv_output) != (1, 128, 20, 20)
            or shape(residual) != (1, 128, 20, 20)
            or shape(add_output) != (1, 128, 20, 20)
            or shape(int(branch_command["inputs"][0])) != (1, 256, 20, 20)
            or shape(branch_output) != (1, 128, 20, 20)
            or shape(conv_weight) != (128, 128, 3, 3)
            or shape(conv_bias) != (1, 1, 1, 128)
            or shape(branch_weight) != (128, 256, 1, 1)
            or shape(branch_bias) != (1, 1, 1, 128)
            or shape(cv2_weight) != (256, 256, 1, 1)
            or shape(cv2_bias) != (1, 1, 1, 256)
            or shape(int(concat["output"])) != (1, 256, 20, 20)
            or any(
                not (int(values[value_id]["flags"]) & VALUE_CONSTANT)
                for value_id in (
                    conv_weight,
                    conv_bias,
                    branch_weight,
                    branch_bias,
                    cv2_weight,
                    cv2_bias,
                )
            )
        ):
            continue
        groups.append((conv_index, add_index, branch_index, concat_index))
    return groups


_PHYSICAL_FUSION_PRIORITY = {
    FUSION_LATTICE_CHAIN_C3: 0,
    FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2: 1,
    FUSION_YOLO_HEAD_FINAL_CONV: 2,
    FUSION_YOLO_HEAD: 3,
    FUSION_C2F_TAIL_RESIDUAL: 4,
    FUSION_CONCAT_RESIDUAL_CV2: 5,
    FUSION_STRIDE2_CONV1X1: 6,
    FUSION_PAIRED_CONV3X3_SILU: 7,
    FUSION_LATE_CONCAT_CONV1X1: 8,
}


def _plan_dual_queue_window(replay: dict[str, Any], physical_dispatch_plan):
    """Plan a dual-queue parallel window over the physical dispatch stream.

    Mirror of the C++ planner (round 17): find a contiguous wave window that
    splits into two independent conv-only chains. The secondary chain B runs
    on a second compute queue; fused/non-conv ops stay on the main queue A.
    Returns (begin, end, side) with side[i] in {0, 1} per dispatch, or None.
    """
    commands = replay["commands"]
    dispatches = physical_dispatch_plan
    n = len(dispatches)
    if n < 6:
        return None

    logical_to_op = [-1] * len(commands)
    for i, record in enumerate(dispatches):
        for k in record.get("logical_indices", ()):
            if 0 <= k < len(logical_to_op):
                logical_to_op[k] = i

    producer = [-1] * len(replay["values"])
    for index, command in enumerate(commands):
        out = int(command["output"])
        if 0 <= out < len(producer):
            producer[out] = index

    deps: list[set[int]] = [set() for _ in range(n)]
    for i, record in enumerate(dispatches):
        for k in record.get("logical_indices", ()):
            if not 0 <= k < len(commands):
                continue
            for raw_input in commands[k]["inputs"]:
                value = int(raw_input)
                if not 0 <= value < len(producer):
                    continue
                lp = producer[value]
                if lp < 0 or not 0 <= lp < len(logical_to_op):
                    continue
                op_p = logical_to_op[lp]
                if op_p >= 0 and op_p != i:
                    deps[i].add(op_p)

    wave = [0] * n
    for i in range(n):
        for p in deps[i]:
            wave[i] = max(wave[i], wave[p] + 1)

    def is_conv_candidate(i: int) -> bool:
        record = dispatches[i]
        if int(record.get("fusion_kind", 0)) != 0:
            return False
        exec_index = int(record.get("execution_index", -1))
        if not 0 <= exec_index < len(commands):
            return False
        return commands[exec_index]["kind"] in {"CONV", "CONV_SILU"}

    max_wave = max(wave) if wave else 0
    wave_cands: list[list[int]] = [[] for _ in range(max_wave + 1)]
    for i in range(n):
        if is_conv_candidate(i):
            wave_cands[wave[i]].append(i)

    for w0 in range(max_wave + 1):
        if len(wave_cands[w0]) < 2:
            continue
        w1 = w0
        while w1 + 1 <= max_wave and len(wave_cands[w1 + 1]) >= 2:
            w1 += 1
        window = [i for i in range(n) if w0 <= wave[i] <= w1]
        if not window:
            continue
        s, e = min(window), max(window) + 1
        if e - s < 4 or e >= n or s < 1:
            # s >= 1: the fork dispatch (s-1) must exist to hoist secondary
            # input transitions before the window.
            continue

        side = [0] * n
        queue = list(wave_cands[w0][1:])
        for seed in queue:
            side[seed] = 1
        effective_end = e
        failed = False
        while queue and not failed:
            b = queue.pop()
            for j in range(s, effective_end):
                if side[j] or any(side[p] for p in deps[j]):
                    if not side[j]:
                        if not is_conv_candidate(j):
                            if j <= s:
                                failed = True
                                break
                            effective_end = min(effective_end, j)
                            continue
                        side[j] = 1
                        queue.append(j)
        if failed or effective_end <= s + 1:
            continue
        e = effective_end

        a_count = sum(1 for i in range(s, e) if not side[i])
        b_count = sum(1 for i in range(s, e) if side[i])
        if a_count < 2 or b_count < 2:
            continue

        cross_flow = False
        for i in range(s, e):
            for p in deps[i]:
                if s <= p < e and side[p] != side[i]:
                    cross_flow = True
                    break
            if cross_flow:
                break
        if cross_flow:
            continue
        return s, e, side
    return None


def _encode_parallel_plan(window, dispatch_count: int) -> bytes:
    """Section 12: [version, dispatch_count, begin, end] + side bytes."""
    if window is None:
        return struct.pack("<4I", 1, dispatch_count, 0, 0)
    begin, end, side = window
    out = bytearray(struct.pack("<4I", 1, dispatch_count, begin, end))
    out.extend(bytes(1 if side[i] else 0 for i in range(dispatch_count)))
    return bytes(out)


def _build_physical_dispatch_plan(
    replay: dict[str, Any],
    kernel_plan: Sequence[dict[str, int]],
    fusion_plan: Sequence[dict[str, int]],
) -> list[dict[str, Any]]:
    commands = replay["commands"]
    command_count = len(commands)
    if len(kernel_plan) != command_count:
        raise ValueError("kernel plan does not cover the replay command stream")

    candidates: list[dict[str, Any]] = []
    for group in fusion_plan:
        kind = int(group["kind"])
        if kind not in _PHYSICAL_FUSION_PRIORITY:
            raise ValueError("unsupported physical fusion kind")
        logical_indices = _physical_fusion_logical_indices(replay, group)
        execution_index = int(group["end"] if kind in {
            FUSION_LATE_CONCAT_CONV1X1,
            FUSION_YOLO_HEAD,
            FUSION_YOLO_HEAD_FINAL_CONV,
        } else group["start"])
        if len(logical_indices) < 2 or execution_index not in logical_indices:
            raise ValueError("invalid physical fusion command mapping")
        candidates.append(
            {
                "logical_start": logical_indices[0],
                "logical_end": logical_indices[-1],
                "execution_index": execution_index,
                "logical_indices": logical_indices,
                "kernel": int(group["kernel"]),
                "precision": int(group["precision"]),
                "flags": int(group["flags"]),
                "fusion_kind": kind,
            }
        )

    for conv_index, add_index, concat_index in _position_owned_winograd_residual_cv2_groups(
        replay, fusion_plan
    ):
        conv_plan = kernel_plan[conv_index]
        concat_plan = kernel_plan[concat_index]
        conv = commands[conv_index]
        if (
            int(conv_plan["planned_kernel"]) != 40
            or int(conv_plan["precision"]) != PRECISION_FLOAT16
            or int(conv_plan["packed_value"]) != int(conv["inputs"][1])
            or int(conv_plan["flags"])
            & ~(PLAN_FLAG_INPUT_FP16 | PLAN_FLAG_OUTPUT_FP16)
            != PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
            or int(concat_plan["planned_kernel"]) != 1
            or int(concat_plan["precision"]) != PRECISION_FLOAT16
            or int(concat_plan["packed_value"]) != NO_VALUE
            or int(concat_plan["flags"]) != PLAN_FLAG_AUTHORITATIVE
        ):
            continue
        candidates.append(
            {
                "logical_start": conv_index,
                "logical_end": concat_index,
                "execution_index": conv_index,
                "logical_indices": (conv_index, add_index, concat_index),
                "kernel": 40,
                "precision": PRECISION_FLOAT16,
                "flags": int(conv_plan["flags"]),
                "fusion_kind": FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2,
            }
        )

    for conv_index, add_index, branch_index, concat_index in (
        _position_owned_winograd_residual_branch_cv2_groups(replay)
    ):
        conv_plan = kernel_plan[conv_index]
        branch_plan = kernel_plan[branch_index]
        concat_plan = kernel_plan[concat_index]
        conv = commands[conv_index]
        branch = commands[branch_index]
        if (
            int(conv_plan["planned_kernel"]) != 40
            or int(conv_plan["precision"]) != PRECISION_FLOAT16
            or int(conv_plan["packed_value"]) != int(conv["inputs"][1])
            or int(conv_plan["flags"])
            & ~(PLAN_FLAG_INPUT_FP16 | PLAN_FLAG_OUTPUT_FP16)
            != PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
            or int(branch_plan["planned_kernel"]) != 45
            or int(branch_plan["precision"]) != PRECISION_FLOAT16
            or int(branch_plan["packed_value"]) != int(branch["inputs"][1])
            or int(branch_plan["flags"])
            != PLAN_FLAG_AUTHORITATIVE | PLAN_FLAG_DXIL
            or int(concat_plan["planned_kernel"]) != 1
            or int(concat_plan["precision"]) != PRECISION_FLOAT16
            or int(concat_plan["packed_value"]) != NO_VALUE
            or int(concat_plan["flags"]) != PLAN_FLAG_AUTHORITATIVE
        ):
            continue
        candidates.append(
            {
                "logical_start": conv_index,
                "logical_end": concat_index,
                "execution_index": conv_index,
                "logical_indices": (conv_index, add_index, branch_index, concat_index),
                "kernel": 40,
                "precision": PRECISION_FLOAT16,
                "flags": int(conv_plan["flags"]),
                "fusion_kind": FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2,
            }
        )

    for execution_index, logical_indices in _physical_concat_residual_cv2_groups(
        replay, fusion_plan, kernel_plan
    ):
        plan = kernel_plan[execution_index]
        candidates.append(
            {
                "logical_start": logical_indices[0],
                "logical_end": logical_indices[-1],
                "execution_index": execution_index,
                "logical_indices": logical_indices,
                "kernel": int(plan["planned_kernel"]),
                "precision": int(plan["precision"]),
                "flags": int(plan["flags"]),
                "fusion_kind": FUSION_CONCAT_RESIDUAL_CV2,
            }
        )

    claimed: set[int] = set()
    records: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            _PHYSICAL_FUSION_PRIORITY[item["fusion_kind"]],
            item["execution_index"],
            item["logical_start"],
            item["logical_end"],
        ),
    ):
        logical_indices = candidate["logical_indices"]
        if any(index in claimed for index in logical_indices):
            continue
        claimed.update(logical_indices)
        records.append(candidate)

    for command_index, plan in enumerate(kernel_plan):
        if int(plan["command_index"]) != command_index:
            raise ValueError("kernel plan command indices are not dense")
        if command_index in claimed:
            continue
        records.append(
            {
                "logical_start": command_index,
                "logical_end": command_index,
                "execution_index": command_index,
                "logical_indices": (command_index,),
                "kernel": int(plan["planned_kernel"]),
                "precision": int(plan["precision"]),
                "flags": int(plan["flags"]),
                "fusion_kind": 0,
            }
        )

    records.sort(key=lambda item: item["execution_index"])
    stable_ids: set[int] = set()
    for record in records:
        record["zero_dispatch"] = all(
            commands[int(index)]["kind"] in {"VIEW", "ALIAS"}
            for index in record["logical_indices"]
        )
        stable_id = _physical_dispatch_stable_id(replay, record)
        if stable_id in stable_ids:
            raise ValueError("physical dispatch stable-id collision")
        stable_ids.add(stable_id)
        record["stable_id"] = stable_id
    return records


def _physical_concat_residual_cv2_groups(
    replay: dict[str, Any],
    fusion_plan: Sequence[dict[str, int]],
    kernel_plan: Sequence[dict[str, int]] | None = None,
) -> list[tuple[int, tuple[int, ...]]]:
    commands = replay["commands"]
    values = replay["values"]
    consumer_counts = [0] * len(values)
    producers = [-1] * len(values)
    for command_index, command in enumerate(commands):
        output = int(command["output"])
        if 0 <= output < len(producers):
            producers[output] = command_index
        for value_id in command["inputs"]:
            value = int(value_id)
            if 0 <= value < len(consumer_counts):
                consumer_counts[value] += 1

    # Native gives a planned C2F tail first ownership of its residual Add.
    c2f_owned_adds = {
        int(group["end"])
        for group in fusion_plan
        if int(group["kind"]) == FUSION_C2F_TAIL_RESIDUAL
    }
    groups: list[tuple[int, tuple[int, ...]]] = []
    for command_index, command in enumerate(commands):
        if command["kind"] != "CONCAT_CONV1X1" or len(command["inputs"]) < 3:
            continue
        # 收编 gate（实测回归教训）：C2F tail 收紧后大尺度 concat 的
        # residual Add 不再被认领，这里收编会把它拉进 CONCAT_RESIDUAL_CV2
        # 融合，而 kernel=1（direct pack4）的 concat 独立 dispatch 远快于
        # 融合路径（实测 DYv11s 80x80 0.110→0.252ms、40x40 0.111→0.393ms）。
        # 仅收编 measured native-fp16 fast path（kernel=3）或 ≤20x20 小尺度
        # （小尺度收编消掉 add dispatch 仍是净收益，见 residual 融合测试）。
        if kernel_plan is not None:
            params = command["params"]
            spatial = int(params[1]) * int(params[2]) if len(params) >= 3 else 0
            if (
                int(kernel_plan[command_index]["planned_kernel"]) != 3
                and spatial > 400
            ):
                continue
        branch_count = len(command["inputs"]) - 2
        owned_adds: set[int] = set()
        for raw_value in command["inputs"][:branch_count]:
            value_id = int(raw_value)
            if value_id < 0 or value_id >= len(producers):
                continue
            producer = producers[value_id]
            if (
                producer < 0
                or producer >= command_index
                or producer in c2f_owned_adds
                or consumer_counts[value_id] != 1
                or not _is_binary_add(commands[producer])
            ):
                continue
            owned_adds.add(producer)
        if owned_adds:
            groups.append(
                (command_index, tuple(sorted((*owned_adds, command_index))))
            )
    return groups


def _physical_fusion_logical_indices(
    replay: dict[str, Any], group: dict[str, int]
) -> tuple[int, ...]:
    commands = replay["commands"]
    kind = int(group["kind"])
    start = int(group["start"])
    end = int(group["end"])
    if start < 0 or start >= len(commands) or end < start or end >= len(commands):
        raise ValueError("fusion group is outside the replay command stream")
    if kind == FUSION_LATTICE_CHAIN_C3:
        if end != start + 5:
            raise ValueError("lattice-chain fusion must own six commands")
        return tuple(range(start, end + 1))
    if kind not in {FUSION_YOLO_HEAD, FUSION_YOLO_HEAD_FINAL_CONV}:
        return tuple(sorted({start, end}))

    values = replay["values"]
    candidates: list[tuple[int, int, int, int, int]] = []
    for index, command in enumerate(commands):
        if index < start or index > end:
            continue
        params = command["params"]
        if (
            command["kind"] != "CONCAT"
            or len(command["inputs"]) != 2
            or len(params) < 10
            or params[1:4] != [4, 1, 2]
        ):
            continue
        box, cls = (int(value_id) for value_id in command["inputs"])
        if box >= len(values) or cls >= len(values):
            continue
        box_shape = values[box]["shape4"]
        cls_shape = values[cls]["shape4"]
        if (
            params[8] != 64
            or box_shape[1] != 64
            or params[9] != cls_shape[1]
            or box_shape[2:] != cls_shape[2:]
        ):
            continue
        candidates.append((index, int(command["output"]), box, cls, box_shape[2] * box_shape[3]))
    candidates.sort(key=lambda item: item[4], reverse=True)
    candidates = candidates[:3]
    if len(candidates) != 3 or min(item[0] for item in candidates) != start:
        raise ValueError("YOLO head fusion does not match the replay stream")

    candidate_outputs = {item[1] for item in candidates}
    logical_indices = {item[0] for item in candidates}
    for index, command in enumerate(commands[:end]):
        if (
            command["kind"] in {"ALIAS", "VIEW"}
            and command["inputs"]
            and int(command["inputs"][0]) in candidate_outputs
        ):
            logical_indices.add(index)
    logical_indices.update(range(end, len(commands)))
    if kind == FUSION_YOLO_HEAD_FINAL_CONV:
        producers = [-1] * len(values)
        for index, command in enumerate(commands):
            producers[int(command["output"])] = index
        for _, _, box, cls, _ in candidates:
            for value_id in (box, cls):
                producer = producers[value_id]
                if producer < 0:
                    raise ValueError("YOLO head final-conv input has no producer")
                logical_indices.add(producer)
    return tuple(sorted(logical_indices))


def _physical_dispatch_stable_id(
    replay: dict[str, Any], record: dict[str, Any]
) -> int:
    logical_indices = tuple(int(index) for index in record["logical_indices"])
    digest = hashlib.blake2b(digest_size=8, person=b"AEXRTPHYV2")
    digest.update(
        struct.pack(
            "<8I",
            int(record["logical_start"]),
            int(record["logical_end"]),
            int(record["execution_index"]),
            int(record["kernel"]),
            int(record["precision"]),
            int(record["flags"]),
            int(record["fusion_kind"]),
            len(logical_indices),
        )
    )
    for logical_index in logical_indices:
        command = replay["commands"][logical_index]
        inputs = tuple(int(value) for value in command["inputs"])
        params = tuple(int(value) for value in command["params"])
        digest.update(
            struct.pack(
                "<5I",
                logical_index,
                int(command["kind_id"]),
                int(command["output"]),
                len(inputs),
                len(params),
            )
        )
        if inputs:
            digest.update(struct.pack(f"<{len(inputs)}I", *inputs))
        if params:
            digest.update(struct.pack(f"<{len(params)}I", *params))
    stable_id = int.from_bytes(digest.digest(), "little")
    return stable_id if stable_id != 0 else 1


def _conv_desc(command: dict[str, Any]) -> tuple[int, ...] | None:
    if command["kind"] not in {"CONV", "CONV_SILU"} or len(command["params"]) < 16:
        return None
    return tuple(int(value) for value in command["params"][:16])


def _is_conv3x3_tiled(desc: tuple[int, ...]) -> bool:
    return desc[7:11] == (3, 3, 1, 1) and desc[11:15] == (1, 1, 1, 1) and desc[15] == 1 and desc[2:4] == desc[5:7]


def _is_conv1x1_tiled(desc: tuple[int, ...]) -> bool:
    return desc[7:11] == (1, 1, 1, 1) and desc[11:15] == (0, 0, 1, 1) and desc[15] == 1 and desc[2:4] == desc[5:7]


def _is_binary_add(command: dict[str, Any]) -> bool:
    return command["kind"] == "BINARY" and len(command["inputs"]) >= 2 and len(command["params"]) >= 2 and command["params"][1] == 0


def _v5_neck_lattice_chain_groups(
    replay: dict[str, Any],
) -> list[tuple[int, int, int, int, int]]:
    """Find the measured six-command C3 lattice used by the v5 neck.

    The matcher is deliberately descriptor- and value-flow based.  Command
    ids and model names are not part of the plan ABI, and every internal value
    must have exactly one in-chain consumer (the entry has the two expected
    branches).  This makes it safe for section 10 to suppress all five
    intermediate arena materializations.
    """
    commands = replay["commands"]
    values = replay["values"]
    consumer_counts = [0] * len(values)
    for command in commands:
        for raw_value in command["inputs"]:
            value_id = int(raw_value)
            if 0 <= value_id < len(consumer_counts):
                consumer_counts[value_id] += 1

    def shape(value_id: int) -> tuple[int, ...] | None:
        if value_id < 0 or value_id >= len(values):
            return None
        result = tuple(int(item) for item in values[value_id].get("shape4", ()))
        return result if len(result) == 4 else None

    def constants(command: dict[str, Any]) -> bool:
        return len(command["inputs"]) == 3 and all(
            0 <= int(value_id) < len(values)
            and int(values[int(value_id)]["flags"]) & VALUE_CONSTANT
            for value_id in command["inputs"][1:]
        )

    # These are the three measured v5 neck C3 blocks.  Restricting the first
    # implementation to their exact physical dimensions keeps kind 9 an
    # offline-promoted kernel plan rather than a speculative generic fusion.
    measured_shapes = {
        (20, 512, 128, 256),
        (40, 256, 64, 128),
        (20, 256, 128, 256),
    }
    groups: list[tuple[int, int, int, int, int]] = []
    for start in range(len(commands) - 5):
        entry, reduce, body1, body3, branch, merge = commands[start : start + 6]
        if (
            entry["kind"] != "CONCAT"
            or len(entry["inputs"]) != 2
            or len(entry["params"]) < 18
            or tuple(int(item) for item in entry["params"][1:4]) != (4, 1, 2)
            or reduce["kind"] != "CONV_SILU"
            or body1["kind"] != "CONV_SILU"
            or body3["kind"] != "CONV_SILU"
            or branch["kind"] != "CONV_SILU"
            or merge["kind"] != "CONCAT_CONV1X1"
            or not all(constants(command) for command in (reduce, body1, body3, branch))
            or len(merge["inputs"]) != 4
            or any(
                not (0 <= int(value_id) < len(values))
                or not (int(values[int(value_id)]["flags"]) & VALUE_CONSTANT)
                for value_id in merge["inputs"][-2:]
            )
        ):
            continue

        d_reduce = _conv_desc(reduce)
        d_body1 = _conv_desc(body1)
        d_body3 = _conv_desc(body3)
        d_branch = _conv_desc(branch)
        if (
            d_reduce is None
            or d_body1 is None
            or d_body3 is None
            or d_branch is None
            or not _is_conv1x1_tiled(d_reduce)
            or not _is_conv1x1_tiled(d_body1)
            or not _is_conv3x3_tiled(d_body3)
            or not _is_conv1x1_tiled(d_branch)
        ):
            continue

        entry_output = int(entry["output"])
        reduce_output = int(reduce["output"])
        body1_output = int(body1["output"])
        body3_output = int(body3["output"])
        branch_output = int(branch["output"])
        merge_output = int(merge["output"])
        if any(
            value_id < 0 or value_id >= len(values)
            for value_id in (
                entry_output,
                reduce_output,
                body1_output,
                body3_output,
                branch_output,
                merge_output,
            )
        ):
            continue

        batch, entry_channels, height, width = d_reduce[0:4]
        hidden_channels = d_reduce[4]
        output_channels = int(merge["params"][3]) if len(merge["params"]) >= 7 else 0
        measured = (height, entry_channels, hidden_channels, output_channels)
        entry_input_shapes = [shape(int(value_id)) for value_id in entry["inputs"]]
        entry_input_channels = tuple(
            int(item[1]) if item is not None else -1
            for item in entry_input_shapes
        )
        if (
            batch != 1
            or height != width
            or measured not in measured_shapes
            or any(
                item is None
                or item[0] != 1
                or item[2:] != (height, width)
                for item in entry_input_shapes
            )
            or sum(entry_input_channels) != entry_channels
            or tuple(int(item) for item in entry["params"][4:10])
            != (1, entry_channels, height, width, *entry_input_channels)
            or tuple(int(item) for item in reduce["inputs"][:1]) != (entry_output,)
            or tuple(int(item) for item in body1["inputs"][:1]) != (reduce_output,)
            or tuple(int(item) for item in body3["inputs"][:1]) != (body1_output,)
            or tuple(int(item) for item in branch["inputs"][:1]) != (entry_output,)
            or tuple(int(item) for item in merge["inputs"][:2])
            != (body3_output, branch_output)
            or d_body1[0:7]
            != (1, hidden_channels, height, width, hidden_channels, height, width)
            or d_body3[0:7]
            != (1, hidden_channels, height, width, hidden_channels, height, width)
            or d_branch[0:7]
            != (1, entry_channels, height, width, hidden_channels, height, width)
            or tuple(int(item) for item in merge["params"])
            != (1, height, width, output_channels, 1, hidden_channels, hidden_channels)
            or shape(entry_output) != (1, entry_channels, height, width)
            or shape(reduce_output) != (1, hidden_channels, height, width)
            or shape(body1_output) != (1, hidden_channels, height, width)
            or shape(body3_output) != (1, hidden_channels, height, width)
            or shape(branch_output) != (1, hidden_channels, height, width)
            or shape(merge_output) != (1, output_channels, height, width)
            or shape(int(reduce["inputs"][1]))
            != (hidden_channels, entry_channels, 1, 1)
            or shape(int(reduce["inputs"][2])) != (1, 1, 1, hidden_channels)
            or shape(int(body1["inputs"][1]))
            != (hidden_channels, hidden_channels, 1, 1)
            or shape(int(body1["inputs"][2])) != (1, 1, 1, hidden_channels)
            or shape(int(body3["inputs"][1]))
            != (hidden_channels, hidden_channels, 3, 3)
            or shape(int(body3["inputs"][2])) != (1, 1, 1, hidden_channels)
            or shape(int(branch["inputs"][1]))
            != (hidden_channels, entry_channels, 1, 1)
            or shape(int(branch["inputs"][2])) != (1, 1, 1, hidden_channels)
            or shape(int(merge["inputs"][-2]))
            != (output_channels, hidden_channels * 2, 1, 1)
            or shape(int(merge["inputs"][-1])) != (1, 1, 1, output_channels)
            or consumer_counts[entry_output] != 2
            or any(
                consumer_counts[value_id] != 1
                for value_id in (
                    reduce_output,
                    body1_output,
                    body3_output,
                    branch_output,
                )
            )
        ):
            continue
        groups.append(
            (start, height, entry_channels, hidden_channels, output_channels)
        )
    return groups


def _fusion_record(kind: int, start: int, end: int, precision: int) -> dict[str, int]:
    return {"kind": kind, "start": start, "end": end, "precision": precision, "kernel": 0, "flags": 1, "aux0": 0, "aux1": 0}


def _detect_yolo_head_group(replay: dict[str, Any], producers: Sequence[int], *, classes: int) -> tuple[int, int, bool] | None:
    commands = replay["commands"]
    values = replay["values"]
    candidates: list[tuple[int, int, int, int]] = []
    for index, command in enumerate(commands):
        params = command["params"]
        if command["kind"] != "CONCAT" or len(command["inputs"]) != 2 or len(params) < 10:
            continue
        if params[1:4] != [4, 1, 2] or params[8] != 64 or params[9] != classes:
            continue
        box, cls = (int(value) for value in command["inputs"])
        box_shape = values[box]["shape4"]
        cls_shape = values[cls]["shape4"]
        if box_shape[1] != 64 or cls_shape[1] != classes or box_shape[2:] != cls_shape[2:]:
            continue
        candidates.append((index, int(command["output"]), box, cls))
    if len(candidates) < 3:
        return None
    candidates.sort(key=lambda item: values[item[2]]["shape4"][2] * values[item[2]]["shape4"][3], reverse=True)
    candidates = candidates[:3]
    outputs = {item[1] for item in candidates}
    final_index = -1
    for index, command in enumerate(commands):
        if command["kind"] == "CONCAT" and len(command["inputs"]) == 3:
            resolved = set()
            for value_id in command["inputs"]:
                source = int(value_id)
                for alias in commands:
                    if alias["kind"] in {"VIEW", "ALIAS"} and alias["output"] == source and alias["inputs"]:
                        source = int(alias["inputs"][0])
                        break
                resolved.add(source)
            if outputs.issubset(resolved):
                final_index = index
                break
    if final_index < 0:
        return None
    final_conv_ok = True
    for _, _, box, cls in candidates:
        for value_id, out_channels in ((box, 64), (cls, values[cls]["shape4"][1])):
            producer = producers[value_id]
            desc = _conv_desc(commands[producer]) if producer >= 0 else None
            if desc is None or commands[producer]["kind"] != "CONV" or desc[4] != out_channels or desc[7:11] != (1, 1, 1, 1):
                final_conv_ok = False
    return min(item[0] for item in candidates), final_index, final_conv_ok


def _encode_fusion_plan(groups: list[dict[str, int]], command_count: int) -> bytes:
    out = bytearray(struct.pack("<4I", 1, len(groups), command_count, 1))
    for group in groups:
        out += struct.pack(
            "<8I",
            group["kind"], group["start"], group["end"], group["precision"],
            group["kernel"], group["flags"], group["aux0"], group["aux1"],
        )
    return bytes(out)


def _encode_physical_dispatch_plan(
    records: Sequence[dict[str, Any]], command_count: int
) -> bytes:
    logical_index_count = sum(len(record["logical_indices"]) for record in records)
    logical_index_table_offset = (
        PHYSICAL_DISPATCH_PLAN_HEADER_SIZE + len(records) * PHYSICAL_DISPATCH_RECORD_SIZE
    )
    out = bytearray(logical_index_table_offset)
    struct.pack_into(
        "<8I",
        out,
        0,
        PHYSICAL_DISPATCH_PLAN_VERSION,
        len(records),
        command_count,
        PHYSICAL_DISPATCH_PLAN_FLAG_AUTHORITATIVE,
        PHYSICAL_DISPATCH_RECORD_SIZE,
        logical_index_count,
        logical_index_table_offset,
        0,
    )
    logical_index_offset = 0
    logical_index_table: list[int] = []
    for record_index, record in enumerate(records):
        logical_indices = tuple(int(index) for index in record["logical_indices"])
        struct.pack_into(
            "<Q10I",
            out,
            PHYSICAL_DISPATCH_PLAN_HEADER_SIZE + record_index * PHYSICAL_DISPATCH_RECORD_SIZE,
            int(record["stable_id"]),
            int(record["logical_start"]),
            int(record["logical_end"]),
            int(record["execution_index"]),
            logical_index_offset,
            len(logical_indices),
            int(record["kernel"]),
            int(record["precision"]),
            int(record["flags"]),
            int(record["fusion_kind"]),
            0,
        )
        logical_index_table.extend(logical_indices)
        logical_index_offset += len(logical_indices)
    if logical_index_table:
        out += struct.pack(f"<{len(logical_index_table)}I", *logical_index_table)
    return bytes(out)


def _encode_pipeline_cache(dxil: Sequence[tuple[int, bytes]], pso: Sequence[tuple[int, bytes]]) -> bytes:
    records = [(1, key, payload) for key, payload in dxil] + [(2, key, payload) for key, payload in pso]
    record_size = 32
    data_offset = _align(16 + len(records) * record_size, ENGINE_ALIGNMENT)
    out = bytearray(data_offset)
    struct.pack_into("<4I", out, 0, 1, len(dxil), len(pso), 0)
    cursor = data_offset
    for index, (kind, key, payload) in enumerate(records):
        cursor = _align(cursor, ENGINE_ALIGNMENT)
        if cursor > len(out):
            out.extend(b"\0" * (cursor - len(out)))
        struct.pack_into("<IIQQQ", out, 16 + index * record_size, kind, 0, key, cursor, len(payload))
        out.extend(payload)
        cursor += len(payload)
    return bytes(out)


def _precision_id(precision: str) -> int:
    normalized = str(precision).strip().lower()
    if normalized == "fp16":
        return PRECISION_FLOAT16
    if normalized == "fp32":
        return PRECISION_FLOAT32
    raise ValueError("AEXRT engine precision must be 'fp16' or 'fp32'")


def _build_container(sections: dict[int, bytes]) -> bytes:
    ordered = sorted(sections.items())
    toc_offset = ENGINE_HEADER_SIZE
    toc_size = len(ordered) * ENGINE_TOC_ENTRY_SIZE
    cursor = _align(toc_offset + toc_size, ENGINE_ALIGNMENT)
    entries: list[tuple[int, int, int, int, int, int]] = []
    payloads: list[tuple[int, bytes]] = []
    for section_type, payload in ordered:
        cursor = _align(cursor, ENGINE_ALIGNMENT)
        entries.append((section_type, 0, cursor, len(payload), zlib.crc32(payload) & 0xFFFFFFFF, 0))
        payloads.append((cursor, payload))
        cursor += len(payload)
    file_size = cursor
    out = bytearray(file_size)
    struct.pack_into(
        "<8sIIIIQQQ16s",
        out,
        0,
        ENGINE_MAGIC,
        ENGINE_VERSION,
        ENGINE_HEADER_SIZE,
        len(entries),
        1,
        file_size,
        toc_offset,
        toc_size,
        b"\0" * 16,
    )
    for index, entry in enumerate(entries):
        struct.pack_into("<IIQQII", out, toc_offset + index * ENGINE_TOC_ENTRY_SIZE, *entry)
    for offset, payload in payloads:
        out[offset : offset + len(payload)] = payload
    return bytes(out)


def _inspect_manifest(data: bytes, section: dict[str, int]) -> dict[str, Any]:
    payload = memoryview(data)[section["offset"] : section["offset"] + section["size"]]
    if len(payload) < 124:
        raise ValueError("truncated AEXRT manifest")
    ints = struct.unpack_from("<17I", payload, 0)
    input_elements, arena_nbytes, conf_threshold, iou_threshold = struct.unpack_from("<QQff", payload, 68)
    return {
        "manifest_version": ints[0],
        "runtime_target": ints[1],
        "precision": ints[2],
        "mode": ints[3],
        "layout": ints[4],
        "has_objectness": bool(ints[5]),
        "channels": ints[6],
        "anchors": ints[7],
        "classes": ints[8],
        "max_candidates": ints[9],
        "max_detections": ints[10],
        "graph_node_count": ints[11],
        "value_count": ints[12],
        "constant_count": ints[13],
        "command_count": ints[14],
        "input_value": ints[15],
        "output_value": ints[16],
        "input_elements": input_elements,
        "arena_nbytes": arena_nbytes,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "source_hash": bytes(payload[92:124]).hex(),
    }


def _source_hash(source_model: str | None, replay: dict[str, Any]) -> bytes:
    if source_model and os.path.isfile(source_model):
        digest = hashlib.sha256()
        with open(source_model, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.digest()
    digest = hashlib.sha256()
    for value in replay["values"]:
        digest.update(struct.pack("<IQ", value["id"], value["elements"]))
        digest.update(value["raw"])
    for command in replay["commands"]:
        digest.update(command["kind"].encode("ascii"))
        digest.update(struct.pack("<I", command["output"]))
    return digest.digest()


def _csv_u32(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value]


def _align(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


__all__ = [
    "ENGINE_MAGIC",
    "ENGINE_VERSION",
    "PHYSICAL_DISPATCH_PLAN_VERSION",
    "SECTION_ARENA_BARRIER_PLAN",
    "SECTION_PHYSICAL_DISPATCH_PLAN",
    "build_aexrt_engine",
    "inspect_aexrt_engine",
    "install_aexrt_pipeline_cache",
    "save_aexrt_engine",
    "save_aexrt_engine_from_onnx",
]
