import copy
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import aexrt.engine as engine_module
from aexrt import Graph, TensorSpec, build_aexrt_engine, inspect_aexrt_engine


def _tiny_graph() -> Graph:
    graph = Graph("physical_plan_v2")
    graph.input("images", TensorSpec((1, 32, 20, 20), "float32"))
    graph.const("w", np.zeros((64, 32, 3, 3), dtype="float32"))
    graph.const("b", np.zeros((64,), dtype="float32"))
    graph.node(
        "Conv",
        "c",
        "images",
        "w",
        "b",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("s", "c")
    graph.mul("act", "c", "s")
    graph.reshape("output0", "act", (1, 64, 100))
    graph.output("output0")
    return graph


def _synthetic_replay(command_count: int) -> dict:
    return {
        "commands": [
            {
                "kind": "UNARY",
                "kind_id": engine_module.COMMAND_KIND["UNARY"],
                "output": index + 1,
                "inputs": [index],
                "params": [index + 100],
            }
            for index in range(command_count)
        ],
        "values": [],
    }


def _synthetic_kernel_plan(command_count: int) -> list[dict[str, int]]:
    return [
        {
            "command_index": index,
            "kind_id": engine_module.COMMAND_KIND["UNARY"],
            "planned_kernel": index + 10,
            "precision": engine_module.PRECISION_FLOAT32,
            "packed_value": engine_module.NO_VALUE,
            "flags": engine_module.PLAN_FLAG_AUTHORITATIVE,
        }
        for index in range(command_count)
    ]


def _kernel_plan_for(commands: list[dict]) -> list[dict[str, int]]:
    return [
        {
            "command_index": index,
            "kind_id": command["kind_id"],
            "planned_kernel": index + 20,
            "precision": engine_module.PRECISION_FLOAT16,
            "packed_value": engine_module.NO_VALUE,
            "flags": engine_module.PLAN_FLAG_AUTHORITATIVE,
        }
        for index, command in enumerate(commands)
    ]


def _fusion(kind: int, start: int, end: int, *, kernel: int = 0) -> dict[str, int]:
    return {
        "kind": kind,
        "start": start,
        "end": end,
        "precision": engine_module.PRECISION_FLOAT16,
        "kernel": kernel,
        "flags": engine_module.PLAN_FLAG_AUTHORITATIVE,
        "aux0": 0,
        "aux1": 0,
    }


def _position_winograd_residual_concat_replay() -> tuple[dict, list[dict[str, int]]]:
    shapes = (
        (1, 128, 10, 10),
        (1, 128, 10, 10),
        (1, 128, 10, 10),
        (128, 128, 3, 3),
        (1, 1, 1, 128),
        (1, 128, 10, 10),
        (1, 128, 10, 10),
        (256, 256, 1, 1),
        (1, 1, 1, 256),
        (1, 256, 10, 10),
    )
    values = [
        {
            "id": value_id,
            "flags": engine_module.VALUE_CONSTANT if value_id in {3, 4, 7, 8} else 0,
            "elements": int(np.prod(shape)),
            "shape4": shape,
        }
        for value_id, shape in enumerate(shapes)
    ]
    commands = [
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 5,
            "inputs": [0, 3, 4],
            "params": [1, 128, 10, 10, 128, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 6,
            "inputs": [1, 5],
            "params": [12800, 0, 4, 0, 1, 128, 10, 10, 1, 128, 10, 10,
                       1, 128, 10, 10, 12800, 100, 10, 1, 12800, 100, 10, 1],
        },
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": 9,
            "inputs": [6, 2, 7, 8],
            "params": [1, 10, 10, 256, 1, 128, 128],
        },
    ]
    replay = {
        "commands": commands,
        "values": values,
        "input_value": 0,
        "output_value": 9,
    }
    tail = _fusion(engine_module.FUSION_C2F_TAIL_RESIDUAL, 0, 1, kernel=40)
    tail["flags"] = engine_module.PLAN_FLAG_AUTHORITATIVE | engine_module.PLAN_FLAG_DXIL
    return replay, [tail]


def _position_winograd_residual_branch_concat_replay() -> dict:
    shapes = (
        (1, 128, 20, 20),
        (1, 128, 20, 20),
        (1, 256, 20, 20),
        (128, 128, 3, 3),
        (1, 1, 1, 128),
        (1, 128, 20, 20),
        (1, 128, 20, 20),
        (128, 256, 1, 1),
        (1, 1, 1, 128),
        (1, 128, 20, 20),
        (256, 256, 1, 1),
        (1, 1, 1, 256),
        (1, 256, 20, 20),
    )
    constants = {3, 4, 7, 8, 10, 11}
    values = [
        {
            "id": value_id,
            "flags": engine_module.VALUE_CONSTANT if value_id in constants else 0,
            "elements": int(np.prod(shape)),
            "shape4": shape,
        }
        for value_id, shape in enumerate(shapes)
    ]
    commands = [
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 5,
            "inputs": [0, 3, 4],
            "params": [1, 128, 20, 20, 128, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 6,
            "inputs": [1, 5],
            "params": [51200, 0],
        },
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 9,
            "inputs": [2, 7, 8],
            "params": [1, 256, 20, 20, 128, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        },
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": 12,
            "inputs": [6, 9, 10, 11],
            "params": [1, 20, 20, 256, 1, 128, 128],
        },
    ]
    return {
        "commands": commands,
        "values": values,
        "input_value": 0,
        "output_value": 12,
    }


def _measured_late_concat_c3_replay() -> tuple[dict, list[dict[str, int]]]:
    shapes = {
        0: (1, 128, 20, 20),
        1: (1, 256, 20, 20),
        2: (1, 384, 20, 20),
        3: (256, 384, 1, 1),
        4: (1, 1, 1, 256),
        5: (1, 256, 20, 20),
        6: (1, 128, 20, 20),
        7: (1, 128, 20, 20),
        8: (64, 128, 3, 3),
        9: (1, 1, 1, 64),
        10: (1, 64, 20, 20),
        11: (128, 64, 3, 3),
        12: (1, 1, 1, 128),
        13: (1, 128, 20, 20),
        14: (1, 128, 20, 20),
        15: (256, 384, 1, 1),
        16: (1, 1, 1, 256),
        17: (1, 256, 20, 20),
    }
    values = [
        {
            "id": value_id,
            "flags": 0,
            "elements": int(np.prod(shapes.get(value_id, (1, 1, 1, 1)))),
            "shape4": shapes.get(value_id, (1, 1, 1, 1)),
        }
        for value_id in range(23)
    ]
    conv_kind = engine_module.COMMAND_KIND["CONV_SILU"]
    commands = [
        {
            "kind": "CONCAT",
            "kind_id": engine_module.COMMAND_KIND["CONCAT"],
            "output": 2,
            "inputs": [0, 1],
            "params": [153600, 4, 1, 2, 1, 384, 20, 20, 128, 256, 256, 256, 256, 256, 256, 256, 0, 0],
        },
        {
            "kind": "UNARY",
            "kind_id": engine_module.COMMAND_KIND["UNARY"],
            "output": 19,
            "inputs": [18],
            "params": [0],
        },
        {
            "kind": "CONV_SILU",
            "kind_id": conv_kind,
            "output": 5,
            "inputs": [2, 3, 4],
            "params": [1, 384, 20, 20, 256, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        },
        {
            "kind": "VIEW",
            "kind_id": engine_module.COMMAND_KIND["VIEW"],
            "output": 6,
            "inputs": [5],
            "params": [0, 51200],
        },
        {
            "kind": "ALIAS",
            "kind_id": engine_module.COMMAND_KIND["ALIAS"],
            "output": 20,
            "inputs": [19],
            "params": [0, 1],
        },
        {
            "kind": "VIEW",
            "kind_id": engine_module.COMMAND_KIND["VIEW"],
            "output": 7,
            "inputs": [5],
            "params": [51200, 51200],
        },
        {
            "kind": "CONV_SILU",
            "kind_id": conv_kind,
            "output": 10,
            "inputs": [7, 8, 9],
            "params": [1, 128, 20, 20, 64, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "kind": "UNARY",
            "kind_id": engine_module.COMMAND_KIND["UNARY"],
            "output": 21,
            "inputs": [20],
            "params": [0],
        },
        {
            "kind": "CONV_SILU",
            "kind_id": conv_kind,
            "output": 13,
            "inputs": [10, 11, 12],
            "params": [1, 64, 20, 20, 128, 20, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1],
        },
        {
            "kind": "ALIAS",
            "kind_id": engine_module.COMMAND_KIND["ALIAS"],
            "output": 22,
            "inputs": [21],
            "params": [0, 1],
        },
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 14,
            "inputs": [7, 13],
            "params": [51200, 0],
        },
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": 17,
            "inputs": [6, 7, 14, 15, 16],
            "params": [1, 20, 20, 256, 1, 128, 128, 128],
        },
    ]
    replay = {
        "commands": commands,
        "values": values,
        "input_value": 0,
        "output_value": 17,
    }
    late_concat = _fusion(
        engine_module.FUSION_LATE_CONCAT_CONV1X1, 0, 2, kernel=3
    )
    late_concat["flags"] |= engine_module.PLAN_FLAG_DXIL
    return replay, [late_concat]


def _section_payloads(data: bytes, info: dict) -> dict[int, bytes]:
    return {
        section_type: data[section["offset"] : section["offset"] + section["size"]]
        for section_type, section in info["sections"].items()
    }


def test_physical_dispatch_plan_v2_wire_layout_and_stable_ids():
    first = build_aexrt_engine(_tiny_graph(), classes=60)
    second = build_aexrt_engine(_tiny_graph(), classes=60)
    info = inspect_aexrt_engine(first)
    second_info = inspect_aexrt_engine(second)

    assert info["section_count"] == 11
    assert info["physical_dispatch_plan_version"] == 2
    assert 0 < info["physical_dispatch_count"] <= info["command_count"]
    assert [record["stable_id"] for record in info["physical_dispatch_plan"]] == [
        record["stable_id"] for record in second_info["physical_dispatch_plan"]
    ]
    assert sorted(
        logical_index
        for record in info["physical_dispatch_plan"]
        for logical_index in record["logical_indices"]
    ) == list(range(info["command_count"]))

    section = info["sections"][engine_module.SECTION_PHYSICAL_DISPATCH_PLAN]
    header = struct.unpack_from("<8I", first, section["offset"])
    version, record_count, command_count, flags, record_size, index_count, index_offset, reserved = header
    assert (version, command_count, flags, record_size, index_count, reserved) == (
        2,
        info["command_count"],
        1,
        48,
        info["command_count"],
        0,
    )
    assert record_count == info["physical_dispatch_count"]
    assert index_offset == 32 + record_count * 48
    assert section["size"] == index_offset + index_count * 4


def test_engine_forwards_exact_size_arena_reuse_policy(monkeypatch):
    original = engine_module.build_page_colored_arena
    captured = {}

    def capture_policy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "build_page_colored_arena", capture_policy)
    build_aexrt_engine(_tiny_graph(), classes=60, arena_reuse_exact_size=True)

    assert captured["reuse_exact_size"] is True


def test_fp16_activation_dag_propagates_chain_and_stops_at_unsupported_edge():
    def conv_params(ic, oc, kernel):
        pad = 1 if kernel == 3 else 0
        return [1, ic, 40, 40, oc, 40, 40, kernel, kernel, 1, 1, pad, pad, 1, 1, 1]

    values = [
        {
            "id": index,
            "flags": engine_module.VALUE_CONSTANT if index in {1, 2, 4, 5, 7, 8} else 0,
            "elements": 64 * 40 * 40,
            "shape4": (1, 64, 40, 40),
        }
        for index in range(10)
    ]
    values[0]["flags"] = engine_module.VALUE_INPUT
    commands = [
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 3,
            "inputs": [0, 1, 2],
            "params": conv_params(64, 64, 1),
        },
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 6,
            "inputs": [3, 4, 5],
            "params": conv_params(64, 64, 1),
        },
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 9,
            "inputs": [6, 7, 8],
            "params": conv_params(64, 64, 3),
        },
    ]
    replay = {
        "values": values,
        "commands": commands,
        "input_value": 0,
        "output_value": 9,
    }
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    physical = engine_module._build_physical_dispatch_plan(replay, plans, [])

    storage = engine_module._plan_fp16_activation_islands(
        replay, plans, physical, precision=engine_module.PRECISION_FLOAT16
    )

    assert storage == {
        3: (engine_module.PRECISION_FLOAT16, 1),
        6: (engine_module.PRECISION_FLOAT16, 1),
    }
    assert [plan["planned_kernel"] for plan in plans] == [45, 45, 44]
    assert [plan["flags"] & 12 for plan in plans] == [8, 12, 4]

    shared = copy.deepcopy(replay)
    shared["values"].append(
        {"id": 10, "flags": 0, "elements": 64 * 40 * 40, "shape4": (1, 64, 40, 40)}
    )
    shared["commands"].append(
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 10,
            "inputs": [3, 0],
            "params": [100, 0],
        }
    )
    shared_plans = engine_module._build_kernel_plan(
        shared, precision=engine_module.PRECISION_FLOAT16
    )
    shared_physical = engine_module._build_physical_dispatch_plan(
        shared, shared_plans, []
    )
    assert engine_module._plan_fp16_activation_islands(
        shared,
        shared_plans,
        shared_physical,
        precision=engine_module.PRECISION_FLOAT16,
    ) == {6: (engine_module.PRECISION_FLOAT16, 1)}
    assert [plan["flags"] & 12 for plan in shared_plans] == [0, 8, 4, 0]


def test_physical_dispatch_plan_uses_sparse_indices_and_runtime_priority():
    replay = _synthetic_replay(5)
    kernel_plan = _synthetic_kernel_plan(5)
    late_concat = _fusion(engine_module.FUSION_LATE_CONCAT_CONV1X1, 0, 3, kernel=3)
    records = engine_module._build_physical_dispatch_plan(replay, kernel_plan, [late_concat])
    fused = next(record for record in records if record["fusion_kind"] != 0)
    assert fused["fusion_kind"] == engine_module.FUSION_LATE_CONCAT_CONV1X1
    assert fused["logical_indices"] == (0, 3)
    assert fused["execution_index"] == 3
    assert [record["execution_index"] for record in records] == [1, 2, 3, 4]

    c2f_tail = _fusion(engine_module.FUSION_C2F_TAIL_RESIDUAL, 2, 3, kernel=40)
    prioritized = engine_module._build_physical_dispatch_plan(
        replay, kernel_plan, [late_concat, c2f_tail]
    )
    assert [record["fusion_kind"] for record in prioritized if record["fusion_kind"]] == [
        engine_module.FUSION_C2F_TAIL_RESIDUAL
    ]
    assert next(record for record in prioritized if record["fusion_kind"])["logical_indices"] == (2, 3)


def test_position_owned_winograd_residual_concat_owns_three_commands():
    replay, fusion_plan = _position_winograd_residual_concat_replay()
    kernel_plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    records = engine_module._build_physical_dispatch_plan(
        replay, kernel_plan, fusion_plan
    )

    assert len(records) == 1
    record = records[0]
    assert record["fusion_kind"] == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    assert record["logical_indices"] == (0, 1, 2)
    assert record["execution_index"] == 0
    assert (record["kernel"], record["precision"], record["flags"]) == (40, 2, 3)

    payload = engine_module._encode_physical_dispatch_plan(records, 3)
    decoded = engine_module._inspect_physical_dispatch_plan(
        payload, {"offset": 0, "size": len(payload)}, 3
    )
    assert decoded["records"][0]["fusion_kind"] == (
        engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    )
    assert decoded["records"][0]["logical_indices"] == (0, 1, 2)


@pytest.mark.parametrize("mutation", ("shared_conv", "branch_order", "descriptor"))
def test_position_owned_winograd_residual_concat_rejects_near_matches(mutation):
    replay, fusion_plan = _position_winograd_residual_concat_replay()
    if mutation == "shared_conv":
        replay["values"].append(
            {"id": 10, "flags": 0, "elements": 12800, "shape4": (1, 128, 10, 10)}
        )
        replay["commands"].append(
            {
                "kind": "UNARY",
                "kind_id": engine_module.COMMAND_KIND["UNARY"],
                "output": 10,
                "inputs": [5],
                "params": [0],
            }
        )
    elif mutation == "branch_order":
        replay["commands"][2]["inputs"][:2] = [2, 6]
    else:
        replay["commands"][0]["params"][4] = 64

    kernel_plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    records = engine_module._build_physical_dispatch_plan(
        replay, kernel_plan, fusion_plan
    )
    assert not [
        record
        for record in records
        if record["fusion_kind"]
        == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    ]


def test_position_owned_winograd_residual_branch_concat_owns_four_commands():
    replay = _position_winograd_residual_branch_concat_replay()
    kernel_plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    records = engine_module._build_physical_dispatch_plan(replay, kernel_plan, [])

    owner = next(
        record
        for record in records
        if record["fusion_kind"]
        == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    )
    assert owner["logical_indices"] == (0, 1, 2, 3)
    assert owner["execution_index"] == 0
    assert (owner["kernel"], owner["precision"], owner["flags"]) == (40, 2, 3)
    assert [record["execution_index"] for record in records] == [0]


@pytest.mark.parametrize("mutation", ("shared_conv", "branch_order", "descriptor"))
def test_position_owned_winograd_residual_branch_concat_rejects_near_matches(mutation):
    replay = _position_winograd_residual_branch_concat_replay()
    if mutation == "shared_conv":
        replay["values"].append(
            {"id": 13, "flags": 0, "elements": 51200, "shape4": (1, 128, 20, 20)}
        )
        replay["commands"].append(
            {
                "kind": "UNARY",
                "kind_id": engine_module.COMMAND_KIND["UNARY"],
                "output": 13,
                "inputs": [5],
                "params": [0],
            }
        )
    elif mutation == "branch_order":
        replay["commands"][3]["inputs"][:2] = [9, 6]
    else:
        replay["commands"][0]["params"][4] = 64

    kernel_plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    records = engine_module._build_physical_dispatch_plan(replay, kernel_plan, [])
    assert not [
        record
        for record in records
        if record["fusion_kind"]
        == engine_module.FUSION_POSITION_OWNED_WINOGRAD_RESIDUAL_CV2
    ]


def test_c2f_tail_fusion_finds_sparse_first_conv_by_value_producer():
    desc = [1, 128, 10, 10, 128, 10, 10, 3, 3, 1, 1, 1, 1, 1, 1, 1]
    detection_desc = [1, 128, 20, 20, 9, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    commands = [
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 3,
            "inputs": [0, 1, 2],
            "params": desc,
        },
        {
            "kind": "CONV",
            "kind_id": engine_module.COMMAND_KIND["CONV"],
            "output": 7,
            "inputs": [4, 5, 6],
            "params": detection_desc,
        },
        {
            "kind": "ALIAS",
            "kind_id": engine_module.COMMAND_KIND["ALIAS"],
            "output": 8,
            "inputs": [7],
            "params": [0, 3600],
        },
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 11,
            "inputs": [3, 9, 10],
            "params": desc,
        },
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 12,
            "inputs": [0, 11],
            "params": [12800, 0],
        },
    ]
    values = [
        {
            "id": value_id,
            "flags": 0,
            "elements": 12800,
            "shape4": (1, 128, 10, 10),
        }
        for value_id in range(14)
    ]
    replay = {
        "commands": commands,
        "values": values,
        "input_value": 0,
        "output_value": 12,
    }
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    fusion = engine_module._build_fusion_plan(
        replay, precision=engine_module.PRECISION_FLOAT16, classes=9
    )
    tail = next(
        group
        for group in fusion
        if group["kind"] == engine_module.FUSION_C2F_TAIL_RESIDUAL
    )
    assert (tail["start"], tail["end"], tail["kernel"], tail["flags"]) == (
        3,
        4,
        40,
        3,
    )

    physical = engine_module._build_physical_dispatch_plan(replay, plans, fusion)
    fused = next(
        record
        for record in physical
        if record["fusion_kind"] == engine_module.FUSION_C2F_TAIL_RESIDUAL
    )
    assert fused["execution_index"] == 3
    assert fused["logical_indices"] == (3, 4)
    assert [record["execution_index"] for record in physical] == [0, 1, 2, 3]

    unmeasured_gap = copy.deepcopy(replay)
    unmeasured_gap["commands"][1]["params"][4] = 10
    assert not [
        group
        for group in engine_module._build_fusion_plan(
            unmeasured_gap, precision=engine_module.PRECISION_FLOAT16, classes=9
        )
        if group["kind"] == engine_module.FUSION_C2F_TAIL_RESIDUAL
    ]

    shared = copy.deepcopy(replay)
    shared["values"].append(
        {"id": 14, "flags": 0, "elements": 12800, "shape4": (1, 128, 10, 10)}
    )
    shared["commands"].append(
        {
            "kind": "UNARY",
            "kind_id": engine_module.COMMAND_KIND["UNARY"],
            "output": 14,
            "inputs": [3],
            "params": [0],
        }
    )
    assert not [
        group
        for group in engine_module._build_fusion_plan(
            shared, precision=engine_module.PRECISION_FLOAT16, classes=9
        )
        if group["kind"] == engine_module.FUSION_C2F_TAIL_RESIDUAL
    ]


def test_measured_c3_kernel_uses_sparse_physical_topology_not_command_ids():
    replay, fusion_plan = _measured_late_concat_c3_replay()
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    engine_module._apply_measured_fp16_physical_kernels(
        replay,
        plans,
        fusion_plan,
        precision=engine_module.PRECISION_FLOAT16,
    )
    physical = engine_module._build_physical_dispatch_plan(
        replay, plans, fusion_plan
    )

    assert plans[6]["planned_kernel"] == 40
    assert engine_module._apply_measured_fp16_c3_topology_kernels(
        replay,
        plans,
        physical,
        precision=engine_module.PRECISION_FLOAT16,
    )
    assert (
        plans[6]["planned_kernel"],
        plans[6]["precision"],
        plans[6]["packed_value"],
        plans[6]["flags"],
    ) == (25, 2, engine_module.NO_VALUE, engine_module.PLAN_FLAG_AUTHORITATIVE)

    rebuilt = engine_module._build_physical_dispatch_plan(
        replay, plans, fusion_plan
    )
    target = next(record for record in rebuilt if record["execution_index"] == 6)
    assert target["logical_indices"] == (6,)
    assert (target["kernel"], target["precision"], target["flags"]) == (25, 2, 1)


def test_measured_c3_kernel_rejects_descriptor_match_without_late_concat_owner():
    replay, fusion_plan = _measured_late_concat_c3_replay()
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    engine_module._apply_measured_fp16_physical_kernels(
        replay,
        plans,
        fusion_plan,
        precision=engine_module.PRECISION_FLOAT16,
    )
    physical_without_late_concat = engine_module._build_physical_dispatch_plan(
        replay, plans, []
    )

    assert not engine_module._apply_measured_fp16_c3_topology_kernels(
        replay,
        plans,
        physical_without_late_concat,
        precision=engine_module.PRECISION_FLOAT16,
    )
    assert plans[6]["planned_kernel"] == 40


def test_measured_c3_kernel_rejects_shared_target_output():
    replay, fusion_plan = _measured_late_concat_c3_replay()
    replay["values"].append(
        {
            "id": 23,
            "flags": 0,
            "elements": 64 * 20 * 20,
            "shape4": (1, 64, 20, 20),
        }
    )
    replay["commands"].append(
        {
            "kind": "UNARY",
            "kind_id": engine_module.COMMAND_KIND["UNARY"],
            "output": 23,
            "inputs": [10],
            "params": [0],
        }
    )
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    engine_module._apply_measured_fp16_physical_kernels(
        replay,
        plans,
        fusion_plan,
        precision=engine_module.PRECISION_FLOAT16,
    )
    physical = engine_module._build_physical_dispatch_plan(
        replay, plans, fusion_plan
    )

    assert not engine_module._apply_measured_fp16_c3_topology_kernels(
        replay,
        plans,
        physical,
        precision=engine_module.PRECISION_FLOAT16,
    )
    assert plans[6]["planned_kernel"] == 40


@pytest.mark.parametrize(
    "fusion_kind",
    (
        engine_module.FUSION_PAIRED_CONV3X3_SILU,
        engine_module.FUSION_STRIDE2_CONV1X1,
    ),
)
def test_physical_pair_and_stride2_groups_own_both_commands(fusion_kind):
    replay = _synthetic_replay(2)
    kernel_plan = _synthetic_kernel_plan(2)
    records = engine_module._build_physical_dispatch_plan(
        replay, kernel_plan, [_fusion(fusion_kind, 0, 1)]
    )

    assert len(records) == 1
    assert records[0]["fusion_kind"] == fusion_kind
    assert records[0]["logical_indices"] == (0, 1)
    assert records[0]["execution_index"] == 0


def test_physical_dispatch_stable_id_changes_with_command_signature():
    replay = _synthetic_replay(3)
    kernel_plan = _synthetic_kernel_plan(3)
    first = engine_module._build_physical_dispatch_plan(replay, kernel_plan, [])
    changed_replay = copy.deepcopy(replay)
    changed_replay["commands"][1]["params"][0] += 1
    second = engine_module._build_physical_dispatch_plan(changed_replay, kernel_plan, [])

    assert first[0]["stable_id"] == second[0]["stable_id"]
    assert first[1]["stable_id"] != second[1]["stable_id"]
    assert first[2]["stable_id"] == second[2]["stable_id"]


def test_physical_concat_residual_cv2_owns_sparse_adds_and_executes_at_cv2():
    commands = [
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 6,
            "inputs": [0, 1],
            "params": [100, 0],
        },
        {
            "kind": "UNARY",
            "kind_id": engine_module.COMMAND_KIND["UNARY"],
            "output": 7,
            "inputs": [2],
            "params": [0],
        },
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 8,
            "inputs": [3, 4],
            "params": [100, 0],
        },
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": 9,
            "inputs": [6, 5, 8, 10, 11],
            "params": [1, 10, 10, 64, 1, 16, 16, 32],
        },
    ]
    replay = {"commands": commands, "values": [{} for _ in range(12)]}
    kernel_plan = _kernel_plan_for(commands)

    records = engine_module._build_physical_dispatch_plan(replay, kernel_plan, [])
    fused = next(
        record
        for record in records
        if record["fusion_kind"] == engine_module.FUSION_CONCAT_RESIDUAL_CV2
    )

    assert fused["logical_indices"] == (0, 2, 3)
    assert fused["execution_index"] == 3
    assert (fused["kernel"], fused["precision"], fused["flags"]) == (
        kernel_plan[3]["planned_kernel"],
        kernel_plan[3]["precision"],
        kernel_plan[3]["flags"],
    )
    assert [record["execution_index"] for record in records] == [1, 3]

    payload = engine_module._encode_physical_dispatch_plan(records, len(commands))
    decoded = engine_module._inspect_physical_dispatch_plan(
        payload, {"offset": 0, "size": len(payload)}, len(commands)
    )
    decoded_fused = next(
        record
        for record in decoded["records"]
        if record["fusion_kind"] == engine_module.FUSION_CONCAT_RESIDUAL_CV2
    )
    assert decoded_fused["logical_indices"] == (0, 2, 3)


def test_physical_concat_residual_cv2_respects_c2f_tail_ownership():
    commands = [
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": 6,
            "inputs": [0, 10, 11],
            "params": [1] * 16,
        },
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 7,
            "inputs": [1, 6],
            "params": [100, 0],
        },
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": 8,
            "inputs": [2, 3],
            "params": [100, 0],
        },
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": 9,
            "inputs": [7, 8, 10, 11],
            "params": [1, 10, 10, 64, 1, 32, 32],
        },
    ]
    replay = {"commands": commands, "values": [{} for _ in range(12)]}
    kernel_plan = _kernel_plan_for(commands)
    c2f_tail = _fusion(engine_module.FUSION_C2F_TAIL_RESIDUAL, 0, 1, kernel=40)

    records = engine_module._build_physical_dispatch_plan(
        replay, kernel_plan, [c2f_tail]
    )
    fused = {record["fusion_kind"]: record for record in records}

    assert fused[engine_module.FUSION_C2F_TAIL_RESIDUAL]["logical_indices"] == (0, 1)
    assert fused[engine_module.FUSION_CONCAT_RESIDUAL_CV2]["logical_indices"] == (2, 3)


def test_physical_yolo_head_ignores_concat_candidates_before_planned_start():
    values = [{"shape4": (1, 1, 1, 1)} for _ in range(16)]
    commands = []
    for index, (box, cls, output, dim, classes) in enumerate(
        (
            (0, 1, 2, 80, 128),
            (3, 4, 5, 40, 4),
            (6, 7, 8, 20, 4),
            (9, 10, 11, 10, 4),
        )
    ):
        values[box]["shape4"] = (1, 64, dim, dim)
        values[cls]["shape4"] = (1, classes, dim, dim)
        commands.append(
            {
                "kind": "CONCAT",
                "kind_id": engine_module.COMMAND_KIND["CONCAT"],
                "output": output,
                "inputs": [box, cls],
                "params": [0, 4, 1, 2, 0, 0, 0, 0, 64, classes],
            }
        )
    commands.append(
        {
            "kind": "CONCAT",
            "kind_id": engine_module.COMMAND_KIND["CONCAT"],
            "output": 15,
            "inputs": [5, 8, 11],
            "params": [0],
        }
    )
    replay = {"commands": commands, "values": values}
    kernel_plan = [
        {
            "command_index": index,
            "kind_id": command["kind_id"],
            "planned_kernel": 0,
            "precision": engine_module.PRECISION_FLOAT32,
            "packed_value": engine_module.NO_VALUE,
            "flags": engine_module.PLAN_FLAG_AUTHORITATIVE,
        }
        for index, command in enumerate(commands)
    ]
    head = _fusion(engine_module.FUSION_YOLO_HEAD, 1, 4)

    records = engine_module._build_physical_dispatch_plan(replay, kernel_plan, [head])
    fused = next(record for record in records if record["fusion_kind"] == engine_module.FUSION_YOLO_HEAD)
    assert fused["logical_indices"] == (1, 2, 3, 4)
    assert any(record["logical_indices"] == (0,) for record in records)


def test_physical_dispatch_inspector_accepts_v1_engine_without_optional_section():
    data = build_aexrt_engine(_tiny_graph(), classes=60)
    info = inspect_aexrt_engine(data)
    sections = _section_payloads(data, info)
    del sections[engine_module.SECTION_PHYSICAL_DISPATCH_PLAN]
    del sections[engine_module.SECTION_ARENA_BARRIER_PLAN]

    legacy = engine_module._build_container(sections)
    legacy_info = inspect_aexrt_engine(legacy)
    assert legacy_info["section_count"] == 9
    assert legacy_info["physical_dispatch_plan_version"] == 0
    assert legacy_info["physical_dispatch_count"] == 0
    assert legacy_info["physical_dispatch_plan"] == []


def test_physical_dispatch_inspector_rejects_trailing_bytes_and_duplicate_coverage():
    data = build_aexrt_engine(_tiny_graph(), classes=60)
    info = inspect_aexrt_engine(data)
    sections = _section_payloads(data, info)
    physical_type = engine_module.SECTION_PHYSICAL_DISPATCH_PLAN

    sections_with_tail = dict(sections)
    sections_with_tail[physical_type] += b"\0\0\0\0"
    with pytest.raises(ValueError, match="invalid AEXRT physical dispatch plan"):
        inspect_aexrt_engine(engine_module._build_container(sections_with_tail))

    malformed = bytearray(sections[physical_type])
    _, _, command_count, _, _, _, index_offset, _ = struct.unpack_from("<8I", malformed, 0)
    assert command_count >= 2
    struct.pack_into("<I", malformed, index_offset + 4, struct.unpack_from("<I", malformed, index_offset)[0])
    malformed_sections = dict(sections)
    malformed_sections[physical_type] = bytes(malformed)
    with pytest.raises(ValueError, match="physical dispatch"):
        inspect_aexrt_engine(engine_module._build_container(malformed_sections))
