import copy
import os
import struct
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import engine as engine_module  # noqa: E402
from aexrt.barrier_v2 import build_arena_barrier_plan  # noqa: E402
from aexrt.memory_v2 import (  # noqa: E402
    build_page_colored_arena,
    build_physical_value_dag,
)


def _value(value_id, shape, *, flags=0):
    elements = 1
    for dim in shape:
        elements *= int(dim)
    value = {
        "id": value_id,
        "flags": flags,
        "elements": elements,
        "shape4": tuple(int(dim) for dim in shape),
    }
    if flags & engine_module.VALUE_CONSTANT:
        value["raw"] = b"\0" * (elements * 4)
    return value


def _v5_neck_lattice_replay(spatial, entry_channels, hidden_channels, output_channels):
    values = []

    def add_value(shape, *, flags=0):
        value_id = len(values)
        values.append(_value(value_id, shape, flags=flags))
        return value_id

    input_channels = (entry_channels // 2, entry_channels - entry_channels // 2)
    left = add_value(
        (1, input_channels[0], spatial, spatial),
        flags=engine_module.VALUE_INPUT,
    )
    right = add_value(
        (1, input_channels[1], spatial, spatial),
        flags=engine_module.VALUE_INPUT,
    )
    entry = add_value((1, entry_channels, spatial, spatial))
    reduce_weight = add_value(
        (hidden_channels, entry_channels, 1, 1),
        flags=engine_module.VALUE_CONSTANT,
    )
    reduce_bias = add_value(
        (1, 1, 1, hidden_channels), flags=engine_module.VALUE_CONSTANT
    )
    reduce_output = add_value((1, hidden_channels, spatial, spatial))
    body1_weight = add_value(
        (hidden_channels, hidden_channels, 1, 1),
        flags=engine_module.VALUE_CONSTANT,
    )
    body1_bias = add_value(
        (1, 1, 1, hidden_channels), flags=engine_module.VALUE_CONSTANT
    )
    body1_output = add_value((1, hidden_channels, spatial, spatial))
    body3_weight = add_value(
        (hidden_channels, hidden_channels, 3, 3),
        flags=engine_module.VALUE_CONSTANT,
    )
    body3_bias = add_value(
        (1, 1, 1, hidden_channels), flags=engine_module.VALUE_CONSTANT
    )
    body3_output = add_value((1, hidden_channels, spatial, spatial))
    branch_weight = add_value(
        (hidden_channels, entry_channels, 1, 1),
        flags=engine_module.VALUE_CONSTANT,
    )
    branch_bias = add_value(
        (1, 1, 1, hidden_channels), flags=engine_module.VALUE_CONSTANT
    )
    branch_output = add_value((1, hidden_channels, spatial, spatial))
    merge_weight = add_value(
        (output_channels, hidden_channels * 2, 1, 1),
        flags=engine_module.VALUE_CONSTANT,
    )
    merge_bias = add_value(
        (1, 1, 1, output_channels), flags=engine_module.VALUE_CONSTANT
    )
    merge_output = add_value((1, output_channels, spatial, spatial))

    def conv(output, source, weight, bias, in_channels, out_channels, kernel):
        pad = 1 if kernel == 3 else 0
        return {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": output,
            "inputs": [source, weight, bias],
            "params": [
                1,
                in_channels,
                spatial,
                spatial,
                out_channels,
                spatial,
                spatial,
                kernel,
                kernel,
                1,
                1,
                pad,
                pad,
                1,
                1,
                1,
            ],
        }

    entry_elements = entry_channels * spatial * spatial
    commands = [
        {
            "kind": "CONCAT",
            "kind_id": engine_module.COMMAND_KIND["CONCAT"],
            "output": entry,
            "inputs": [left, right],
            "params": [
                entry_elements,
                4,
                1,
                2,
                1,
                entry_channels,
                spatial,
                spatial,
                *input_channels,
                256,
                256,
                256,
                256,
                256,
                256,
                0,
                0,
            ],
        },
        conv(
            reduce_output,
            entry,
            reduce_weight,
            reduce_bias,
            entry_channels,
            hidden_channels,
            1,
        ),
        conv(
            body1_output,
            reduce_output,
            body1_weight,
            body1_bias,
            hidden_channels,
            hidden_channels,
            1,
        ),
        conv(
            body3_output,
            body1_output,
            body3_weight,
            body3_bias,
            hidden_channels,
            hidden_channels,
            3,
        ),
        conv(
            branch_output,
            entry,
            branch_weight,
            branch_bias,
            entry_channels,
            hidden_channels,
            1,
        ),
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": merge_output,
            "inputs": [body3_output, branch_output, merge_weight, merge_bias],
            "params": [
                1,
                spatial,
                spatial,
                output_channels,
                1,
                hidden_channels,
                hidden_channels,
            ],
        },
    ]
    return {
        "values": values,
        "commands": commands,
        "input_value": left,
        "output_value": merge_output,
    }, {
        "external_activations": {left, right},
        "internal": {
            entry,
            reduce_output,
            body1_output,
            body3_output,
            branch_output,
        },
        "constants": {
            reduce_weight,
            reduce_bias,
            body1_weight,
            body1_bias,
            body3_weight,
            body3_bias,
            branch_weight,
            branch_bias,
            merge_weight,
            merge_bias,
        },
        "pointwise_weights": {
            reduce_weight,
            body1_weight,
            branch_weight,
            merge_weight,
        },
        "body3_weight": body3_weight,
        "output": merge_output,
    }


def _compile(replay):
    kernel_plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    fusion_plan = engine_module._build_fusion_plan(
        replay, precision=engine_module.PRECISION_FLOAT16, classes=9
    )
    physical = engine_module._build_physical_dispatch_plan(
        replay, kernel_plan, fusion_plan
    )
    return kernel_plan, fusion_plan, physical


@pytest.mark.parametrize(
    "shape",
    (
        (20, 512, 128, 256),
        (40, 256, 64, 128),
        (20, 256, 128, 256),
    ),
)
def test_v5_neck_lattice_chain_folds_six_commands_into_one_microdag_record(shape):
    replay, _ = _v5_neck_lattice_replay(*shape)
    _, fusion_plan, physical = _compile(replay)

    lattice = [
        group
        for group in fusion_plan
        if group["kind"] == engine_module.FUSION_LATTICE_CHAIN_C3
    ]
    assert len(lattice) == 1
    group = lattice[0]
    spatial, entry_channels, hidden_channels, output_channels = shape
    assert (group["start"], group["end"], group["kernel"], group["precision"]) == (
        0,
        5,
        0,
        engine_module.PRECISION_FLOAT16,
    )
    assert group["flags"] == (
        engine_module.PLAN_FLAG_AUTHORITATIVE | engine_module.PLAN_FLAG_DXIL
    )
    assert group["flags"] & (
        engine_module.PLAN_FLAG_INPUT_FP16 | engine_module.PLAN_FLAG_OUTPUT_FP16
    ) == 0
    assert group["aux0"] == spatial | (hidden_channels << 16)
    assert group["aux1"] == entry_channels | (output_channels << 16)

    assert len(physical) == 1
    record = physical[0]
    assert record["fusion_kind"] == engine_module.FUSION_LATTICE_CHAIN_C3
    assert record["logical_indices"] == (0, 1, 2, 3, 4, 5)
    assert record["execution_index"] == 0
    assert (record["kernel"], record["precision"], record["flags"]) == (0, 2, 3)


def test_lattice_chain_section8_and_section10_round_trip_keep_legacy_versions():
    replay, _ = _v5_neck_lattice_replay(40, 256, 64, 128)
    _, fusion_plan, physical = _compile(replay)

    fusion_payload = engine_module._encode_fusion_plan(fusion_plan, 6)
    assert struct.unpack_from("<I", fusion_payload, 0)[0] == 1
    decoded_fusion = engine_module._inspect_fusion_plan(
        fusion_payload, {"offset": 0, "size": len(fusion_payload)}, 6
    )
    lattice_group = next(
        group
        for group in decoded_fusion
        if group["kind"] == engine_module.FUSION_LATTICE_CHAIN_C3
    )
    assert lattice_group == next(
        group
        for group in fusion_plan
        if group["kind"] == engine_module.FUSION_LATTICE_CHAIN_C3
    )

    physical_payload = engine_module._encode_physical_dispatch_plan(physical, 6)
    assert struct.unpack_from("<I", physical_payload, 0)[0] == 2
    decoded_physical = engine_module._inspect_physical_dispatch_plan(
        physical_payload, {"offset": 0, "size": len(physical_payload)}, 6
    )
    assert decoded_physical["version"] == 2
    assert decoded_physical["records"][0]["fusion_kind"] == 9
    assert decoded_physical["records"][0]["logical_indices"] == tuple(range(6))


def test_lattice_chain_stable_id_covers_all_six_logical_commands():
    replay, _ = _v5_neck_lattice_replay(40, 256, 64, 128)
    _, _, first = _compile(replay)
    _, _, identical = _compile(copy.deepcopy(replay))
    assert first[0]["stable_id"] == identical[0]["stable_id"]

    changed = copy.deepcopy(replay)
    old_bias = int(changed["commands"][4]["inputs"][2])
    replacement_bias = len(changed["values"])
    changed["values"].append(
        _value(
            replacement_bias,
            changed["values"][old_bias]["shape4"],
            flags=engine_module.VALUE_CONSTANT,
        )
    )
    changed["commands"][4]["inputs"][2] = replacement_bias
    _, _, second = _compile(changed)
    assert second[0]["fusion_kind"] == engine_module.FUSION_LATTICE_CHAIN_C3
    assert first[0]["stable_id"] != second[0]["stable_id"]


def test_lattice_chain_plan_packs_every_pointwise_weight():
    replay, ids = _v5_neck_lattice_replay(40, 256, 64, 128)
    kernel_plan, fusion_plan, _ = _compile(replay)

    packed = engine_module._build_packed_weights(
        replay,
        kernel_plan,
        precision=engine_module.PRECISION_FLOAT16,
        fusion_plan=fusion_plan,
    )
    by_value = {record["value_id"]: record for record in packed}
    assert ids["pointwise_weights"] <= set(by_value)
    assert {
        by_value[value_id]["layout"] for value_id in ids["pointwise_weights"]
    } == {engine_module.PACKED_LAYOUT_CONV1X1_OC8_FP16}
    assert ids["body3_weight"] in by_value


def test_lattice_chain_dag_and_barriers_ignore_internal_materialization():
    replay, ids = _v5_neck_lattice_replay(40, 256, 64, 128)
    _, _, physical = _compile(replay)
    dag = build_physical_value_dag(replay, physical)

    dispatch = dag["dispatches"][0]
    assert dispatch["external_inputs"] == frozenset(
        ids["external_activations"] | ids["constants"]
    )
    assert dispatch["internal_outputs"] == frozenset(ids["internal"])
    assert dispatch["materialized_outputs"] == frozenset({ids["output"]})
    assert dag["materialized_values"] == frozenset({ids["output"]})
    assert dag["material_dispatch_count"] == 1

    arena = build_page_colored_arena(replay, physical_dispatch_plan=physical)
    assert ids["internal"].isdisjoint(arena["by_value"])
    assert ids["external_activations"] | {ids["output"]} <= set(arena["by_value"])

    barriers = build_arena_barrier_plan(replay, physical, arena)
    assert barriers["dispatch_count"] == 1
    assert ids["internal"].isdisjoint(
        record["after_value_id"] for record in barriers["records"]
    )
    assert ids["output"] in {
        record["after_value_id"] for record in barriers["records"]
    }


@pytest.mark.parametrize("mutation", ("shared_internal", "body_descriptor", "merge_order"))
def test_lattice_chain_rejects_unsafe_near_matches(mutation):
    replay, ids = _v5_neck_lattice_replay(40, 256, 64, 128)
    if mutation == "shared_internal":
        output = len(replay["values"])
        source = int(replay["commands"][2]["output"])
        replay["values"].append(_value(output, replay["values"][source]["shape4"]))
        replay["commands"].append(
            {
                "kind": "UNARY",
                "kind_id": engine_module.COMMAND_KIND["UNARY"],
                "output": output,
                "inputs": [source],
                "params": [0],
            }
        )
        replay["output_value"] = output
    elif mutation == "body_descriptor":
        replay["commands"][3]["params"][11] = 0
    else:
        replay["commands"][5]["inputs"][:2] = reversed(
            replay["commands"][5]["inputs"][:2]
        )

    _, fusion_plan, physical = _compile(replay)
    assert not [
        group
        for group in fusion_plan
        if group["kind"] == engine_module.FUSION_LATTICE_CHAIN_C3
    ]
    assert not [
        record
        for record in physical
        if record["fusion_kind"] == engine_module.FUSION_LATTICE_CHAIN_C3
    ]
    assert ids["internal"]
