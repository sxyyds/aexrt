import copy
import os
import struct
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import (  # noqa: E402
    Graph,
    NativeCppYoloModel,
    NativeD3D12Device,
    TensorSpec,
    inspect_aexrt_engine,
    save_aexrt_engine,
)
from aexrt import engine as engine_module  # noqa: E402
from aexrt.memory_v2 import (  # noqa: E402
    LAYOUT_BLOCKED_NCHW8,
    PRECISION_FLOAT16,
    encode_memory_plan_v2,
    inspect_memory_plan_v2,
)


IO_MASK = engine_module.PLAN_FLAG_INPUT_FP16 | engine_module.PLAN_FLAG_OUTPUT_FP16


def _continuous_graph():
    graph = Graph("prepared_fp16_physical_validation")
    graph.input("images", TensorSpec((1, 64, 40, 40), "float32"))
    source = "images"
    for index, kernel in enumerate((1, 3)):
        graph.const(
            f"w{index}",
            np.zeros((64, 64, kernel, kernel), dtype="float32"),
        )
        graph.const(f"b{index}", np.zeros((64,), dtype="float32"))
        graph.node(
            "Conv",
            f"c{index}",
            source,
            f"w{index}",
            f"b{index}",
            strides=[1, 1],
            pads=[kernel // 2] * 4,
            dilations=[1, 1],
            group=1,
        )
        graph.sigmoid(f"s{index}", f"c{index}")
        graph.mul(f"a{index}", f"c{index}", f"s{index}")
        source = f"a{index}"
    graph.reshape("output0", source, (1, 64, 1600))
    graph.output("output0")
    return graph


def _sections(path, info):
    data = path.read_bytes()
    return {
        section_type: data[
            section["offset"] : section["offset"] + section["size"]
        ]
        for section_type, section in info["sections"].items()
    }


def _write_engine(path, sections):
    path.write_bytes(engine_module._build_container(sections))
    return inspect_aexrt_engine(path)


def _build_valid_engine(tmp_path):
    path = tmp_path / "continuous_valid.aexrt"
    info = save_aexrt_engine(
        _continuous_graph(), path, classes=60, max_detections=4
    )
    typed_values = info["arena_material_storage_precision_counts"]["fp16"]
    assert typed_values == 1
    assert [record[2] for record in info["kernel_plan"][:2]] == [45, 44]
    assert [record[5] & IO_MASK for record in info["kernel_plan"][:2]] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]
    return path, info


def _require_native_d3d12():
    if not NativeD3D12Device.probe().available:
        pytest.skip("native D3D12 runtime is unavailable")


def test_native_prepared_plan_accepts_matching_linear_fp16_dtype_and_flags(tmp_path):
    _require_native_d3d12()
    path, info = _build_valid_engine(tmp_path)

    model = NativeCppYoloModel(path)
    try:
        assert model.executable
        assert model.prepared_command_count == info["command_count"]
        assert model.unsupported_prepared_command_count == 0
        model.run(np.zeros(model.input_element_count, dtype=np.float32), 1)
    finally:
        model.close()


def test_native_prepared_plan_rejects_fp16_memory_with_cleared_io_flags(tmp_path):
    _require_native_d3d12()
    valid_path, valid_info = _build_valid_engine(tmp_path)
    sections = _sections(valid_path, valid_info)

    kernel_plan = bytearray(sections[engine_module.SECTION_KERNEL_PLAN])
    kernel_count = struct.unpack_from("<I", kernel_plan, 0)[0]
    for index in range(kernel_count):
        flags_offset = 8 + index * 24 + 20
        flags = struct.unpack_from("<I", kernel_plan, flags_offset)[0]
        struct.pack_into("<I", kernel_plan, flags_offset, flags & ~IO_MASK)
    sections[engine_module.SECTION_KERNEL_PLAN] = bytes(kernel_plan)

    physical = bytearray(
        sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN]
    )
    physical_count = struct.unpack_from("<I", physical, 4)[0]
    for index in range(physical_count):
        flags_offset = (
            engine_module.PHYSICAL_DISPATCH_PLAN_HEADER_SIZE
            + index * engine_module.PHYSICAL_DISPATCH_RECORD_SIZE
            + 36
        )
        flags = struct.unpack_from("<I", physical, flags_offset)[0]
        struct.pack_into("<I", physical, flags_offset, flags & ~IO_MASK)
    sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN] = bytes(physical)

    malformed = tmp_path / "fp16_memory_fp32_flags.aexrt"
    malformed_info = _write_engine(malformed, sections)
    assert malformed_info["arena_material_storage_precision_counts"]["fp16"] == 1
    assert all(record[5] & IO_MASK == 0 for record in malformed_info["kernel_plan"])

    model = NativeCppYoloModel(malformed)
    try:
        with pytest.raises(RuntimeError, match="run failed"):
            model.run(np.zeros(model.input_element_count, dtype=np.float32), 1)
    finally:
        model.close()


def test_native_prepared_plan_rejects_schema_valid_blocked_fp16_layout(tmp_path):
    _require_native_d3d12()
    valid_path, valid_info = _build_valid_engine(tmp_path)
    sections = _sections(valid_path, valid_info)
    memory = inspect_memory_plan_v2(
        sections[engine_module.SECTION_MEMORY_PLAN]
    )
    mutated = copy.deepcopy(memory)
    fp16_pages = {
        int(page["page_id"])
        for page in mutated["pages"]
        if int(page["storage_dtype"]) == PRECISION_FLOAT16
    }
    assert fp16_pages
    for page in mutated["pages"]:
        if int(page["page_id"]) in fp16_pages:
            page["storage_layout"] = LAYOUT_BLOCKED_NCHW8
    for record in mutated["records"]:
        if int(record["page_id"]) in fp16_pages:
            record["storage_layout"] = LAYOUT_BLOCKED_NCHW8
    sections[engine_module.SECTION_MEMORY_PLAN] = encode_memory_plan_v2(mutated)

    malformed = tmp_path / "blocked_fp16_layout.aexrt"
    malformed_info = _write_engine(malformed, sections)
    assert malformed_info["arena_material_storage_precision_counts"]["fp16"] == 1

    model = NativeCppYoloModel(malformed)
    try:
        with pytest.raises(RuntimeError, match="run failed"):
            model.run(np.zeros(model.input_element_count, dtype=np.float32), 1)
    finally:
        model.close()
