import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import aexrt.engine as engine_module  # noqa: E402
from aexrt import (  # noqa: E402
    Graph,
    NativeCppYoloModel,
    NativeD3D12Device,
    TensorSpec,
    build_aexrt_engine,
    inspect_aexrt_engine,
    save_aexrt_engine,
)
from aexrt.barrier_v2 import (  # noqa: E402
    ARENA_BARRIER_KIND_TRANSITION_SRV,
    ARENA_BARRIER_KIND_TRANSITION_UAV,
    ARENA_BARRIER_PLAN_HEADER_SIZE,
    ARENA_BARRIER_RECORD_SIZE,
    build_arena_barrier_plan,
    encode_arena_barrier_plan,
    inspect_arena_barrier_plan,
)
from aexrt.memory_v2 import build_page_colored_arena  # noqa: E402


def _value(value_id, elements, flags=0):
    return {"id": value_id, "flags": flags, "elements": elements}


def _command(kind, output, *inputs):
    return {"kind": kind, "output": output, "inputs": list(inputs)}


def _branch_replay():
    replay = {
        "values": [
            _value(0, 64, flags=1),
            _value(1, 64),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            _command("CONV_SILU", 2, 0),
            _command("BINARY", 3, 1, 2),
        ],
        "output_value": 3,
    }
    physical = [
        {
            "logical_indices": [index],
            "execution_index": index,
            "fusion_kind": 0,
        }
        for index in range(3)
    ]
    return replay, physical


def _identity_yolo_graph():
    graph = Graph("barrier_plan_identity")
    graph.input("images", TensorSpec((1, 64, 10, 10), "float32"))
    weight = np.zeros((64, 64, 1, 1), dtype="float32")
    weight[np.arange(64), np.arange(64), 0, 0] = 1.0
    graph.const("w", weight)
    graph.const("b", np.zeros((64,), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1,
    )
    graph.reshape("output0", "c", (1, 64, 100))
    graph.output("output0")
    return graph


def _section_payloads(data, info):
    return {
        section_type: data[section["offset"] : section["offset"] + section["size"]]
        for section_type, section in info["sections"].items()
    }


def test_barrier_compiler_keeps_unchanged_pages_out_of_dispatch_plan():
    replay, physical = _branch_replay()
    memory = build_page_colored_arena(replay, physical_dispatch_plan=physical)
    plan = build_arena_barrier_plan(replay, physical, memory)

    assert plan["dispatch_count"] == 3
    assert {record["kind"] for record in plan["records"]} == {
        ARENA_BARRIER_KIND_TRANSITION_SRV,
        ARENA_BARRIER_KIND_TRANSITION_UAV,
    }
    # Dispatch 1 reads the same input page and appends a disjoint write to the
    # same UAV page as dispatch 0, so it needs no resource-wide barrier.
    assert not any(record["dispatch_index"] == 1 for record in plan["records"])

    payload = encode_arena_barrier_plan(plan)
    decoded = inspect_arena_barrier_plan(
        payload,
        dispatch_count=3,
        page_nbytes=[page["nbytes"] for page in memory["pages"]],
        value_count=len(replay["values"]),
    )
    assert decoded["records"] == plan["records"]


def test_zero_dispatch_alias_has_no_barrier_action():
    replay = {
        "values": [
            _value(0, 64, flags=1),
            _value(1, 64),
            _value(2, 64),
            _value(3, 64),
        ],
        "commands": [
            _command("CONV_SILU", 1, 0),
            {**_command("VIEW", 2, 1), "params": [0, 64]},
            _command("CONV_SILU", 3, 2),
        ],
        "output_value": 3,
    }
    physical = [
        {
            "logical_indices": [index],
            "execution_index": index,
            "fusion_kind": 0,
        }
        for index in range(3)
    ]
    memory = build_page_colored_arena(replay, physical_dispatch_plan=physical)
    plan = build_arena_barrier_plan(replay, physical, memory)

    assert 2 in memory["by_value"]
    assert not any(record["dispatch_index"] == 1 for record in plan["records"])


def test_barrier_binary_section_is_fixed_width_and_strict():
    replay, physical = _branch_replay()
    memory = build_page_colored_arena(replay, physical_dispatch_plan=physical)
    plan = build_arena_barrier_plan(replay, physical, memory)
    payload = encode_arena_barrier_plan(plan)
    header = struct.unpack_from("<8I", payload, 0)
    assert header[:6] == (1, 1, 3, len(memory["pages"]), len(plan["records"]), 40)
    assert len(payload) == ARENA_BARRIER_PLAN_HEADER_SIZE + len(plan["records"]) * ARENA_BARRIER_RECORD_SIZE

    malformed = bytearray(payload)
    struct.pack_into("<Q", malformed, ARENA_BARRIER_PLAN_HEADER_SIZE + 32, 0)
    with pytest.raises(ValueError, match="arena barrier record"):
        inspect_arena_barrier_plan(malformed)


def test_engine_inspector_exposes_barrier_plan_and_rejects_dangling_section():
    data = build_aexrt_engine(_identity_yolo_graph(), classes=60)
    info = inspect_aexrt_engine(data)
    assert info["section_count"] == 11
    assert info["arena_barrier_plan_version"] == 1
    assert info["arena_barrier_count"] > 0
    assert info["arena_transition_count"] == info["arena_barrier_count"]
    assert info["arena_uav_barrier_count"] == 0
    assert engine_module.SECTION_ARENA_BARRIER_PLAN in info["sections"]

    sections = _section_payloads(data, info)
    del sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN]
    with pytest.raises(ValueError, match="requires V2 physical memory"):
        inspect_aexrt_engine(engine_module._build_container(sections))

    sections = _section_payloads(data, info)
    del sections[engine_module.SECTION_ARENA_BARRIER_PLAN]
    legacy = inspect_aexrt_engine(engine_module._build_container(sections))
    assert legacy["arena_barrier_plan_version"] == 0
    assert legacy["arena_transition_count"] == 0
    assert legacy["arena_uav_barrier_count"] == 0
    assert legacy["arena_barrier_plan"] == []


def test_native_loader_rejects_incomplete_authoritative_barrier_plan(tmp_path):
    if not NativeD3D12Device.probe().available:
        return
    valid_path = tmp_path / "valid_barrier.aexrt"
    valid_info = save_aexrt_engine(
        _identity_yolo_graph(), valid_path, classes=60, max_detections=4
    )
    runtime = NativeCppYoloModel(valid_path)
    runtime.close()

    data = valid_path.read_bytes()
    sections = _section_payloads(data, valid_info)
    barrier_type = engine_module.SECTION_ARENA_BARRIER_PLAN
    barrier = bytearray(sections[barrier_type])
    record_count = struct.unpack_from("<I", barrier, 16)[0]
    assert record_count > 1
    struct.pack_into("<I", barrier, 16, record_count - 1)
    del barrier[-ARENA_BARRIER_RECORD_SIZE:]
    sections[barrier_type] = bytes(barrier)
    malformed_path = tmp_path / "missing_barrier_action.aexrt"
    malformed_path.write_bytes(engine_module._build_container(sections))

    # The generic wire inspector accepts a structurally valid forward-compatible
    # subset; the native loader recomputes the exact physical access schedule.
    assert inspect_aexrt_engine(malformed_path)["arena_barrier_count"] == record_count - 1
    with pytest.raises(RuntimeError, match="failed to compile"):
        NativeCppYoloModel(malformed_path)
