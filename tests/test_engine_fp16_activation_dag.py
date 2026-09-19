import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import engine as engine_module  # noqa: E402
from aexrt.memory_v2 import build_page_colored_arena  # noqa: E402


IO_MASK = engine_module.PLAN_FLAG_INPUT_FP16 | engine_module.PLAN_FLAG_OUTPUT_FP16


def _value(value_id, channels, spatial, *, flags=0):
    return {
        "id": value_id,
        "flags": flags,
        "elements": channels * spatial * spatial,
        "shape4": (1, channels, spatial, spatial),
    }


def _replay(channels=64, spatial=40):
    return {
        "values": [_value(0, channels, spatial, flags=engine_module.VALUE_INPUT)],
        "commands": [],
        "input_value": 0,
        "output_value": 0,
    }


def _add_conv(replay, source, out_channels, kernel):
    values = replay["values"]
    in_shape = values[source]["shape4"]
    _, in_channels, height, width = in_shape
    assert height == width
    weight = len(values)
    values.append(
        _value(
            weight,
            out_channels * in_channels * kernel * kernel,
            1,
            flags=engine_module.VALUE_CONSTANT,
        )
    )
    bias = len(values)
    values.append(
        _value(bias, out_channels, 1, flags=engine_module.VALUE_CONSTANT)
    )
    output = len(values)
    values.append(_value(output, out_channels, height))
    pad = 1 if kernel == 3 else 0
    replay["commands"].append(
        {
            "kind": "CONV_SILU",
            "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
            "output": output,
            "inputs": [source, weight, bias],
            "params": [
                1,
                in_channels,
                height,
                width,
                out_channels,
                height,
                width,
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
    )
    return output


def _add_binary(replay, first, second):
    shape = replay["values"][first]["shape4"]
    output = len(replay["values"])
    replay["values"].append(
        _value(output, int(shape[1]), int(shape[2]))
    )
    replay["commands"].append(
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": output,
            "inputs": [first, second],
            "params": [int(replay["values"][output]["elements"]), 0],
        }
    )
    return output


def _add_concat_conv1x1(replay, inputs, out_channels=64):
    shape = replay["values"][inputs[0]]["shape4"]
    spatial = int(shape[2])
    in_channels = [int(replay["values"][value_id]["shape4"][1]) for value_id in inputs]
    weight = len(replay["values"])
    replay["values"].append(
        _value(
            weight,
            out_channels * sum(in_channels),
            1,
            flags=engine_module.VALUE_CONSTANT,
        )
    )
    bias = len(replay["values"])
    replay["values"].append(
        _value(bias, out_channels, 1, flags=engine_module.VALUE_CONSTANT)
    )
    output = len(replay["values"])
    replay["values"].append(_value(output, out_channels, spatial))
    replay["commands"].append(
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": output,
            "inputs": [*inputs, weight, bias],
            "params": [1, spatial, spatial, out_channels, 1, *in_channels],
        }
    )
    return output


def _add_c2f_tail(replay, residual):
    channels = int(replay["values"][residual]["shape4"][1])
    tail_input = _add_conv(replay, residual, channels, 1)
    tail_conv = _add_conv(replay, tail_input, channels, 3)
    tail_output = _add_binary(replay, residual, tail_conv)
    return tail_input, tail_conv, tail_output


def _plan(replay, physical=None, fusion=None, selected_value_ids=None):
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    fusion = [] if fusion is None else fusion
    if physical is None:
        physical = engine_module._build_physical_dispatch_plan(
            replay, plans, fusion
        )
    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        fusion,
        precision=engine_module.PRECISION_FLOAT16,
        selected_value_ids=selected_value_ids,
    )
    return plans, physical, storage


def _force_kernel(plans, physical, command_index, kernel, *, flags=None):
    flags = (
        engine_module.PLAN_FLAG_AUTHORITATIVE | engine_module.PLAN_FLAG_DXIL
        if flags is None
        else flags
    )
    plans[command_index]["planned_kernel"] = kernel
    plans[command_index]["precision"] = engine_module.PRECISION_FLOAT16
    plans[command_index]["flags"] = flags
    owner = next(
        record
        for record in physical
        if record["fusion_kind"] == 0
        and record["execution_index"] == command_index
    )
    owner["kernel"] = kernel
    owner["precision"] = engine_module.PRECISION_FLOAT16
    owner["flags"] = flags


@pytest.mark.parametrize("kernel", (1, 24, 33, 39, 40, 43, 44, 45, 47))
def test_fp16_activation_dag_supports_complete_singleton_kernel_set(kernel):
    replay = _replay()
    intermediate = _add_conv(replay, 0, 64, 1)
    replay["output_value"] = _add_conv(replay, intermediate, 64, 3)
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    physical = engine_module._build_physical_dispatch_plan(replay, plans, [])
    for command_index in range(2):
        _force_kernel(plans, physical, command_index, kernel)
    before = tuple(record["stable_id"] for record in physical)

    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        precision=engine_module.PRECISION_FLOAT16,
    )

    assert storage == {intermediate: (engine_module.PRECISION_FLOAT16, 1)}
    assert [plan["flags"] & IO_MASK for plan in plans] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]
    assert tuple(record["stable_id"] for record in physical) != before


def test_fp16_activation_dag_supports_plain_conv_and_rejects_unlisted_kernel():
    replay = _replay()
    intermediate = _add_conv(replay, 0, 64, 1)
    replay["output_value"] = _add_conv(replay, intermediate, 64, 1)
    for command in replay["commands"]:
        command["kind"] = "CONV"
        command["kind_id"] = engine_module.COMMAND_KIND["CONV"]
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    physical = engine_module._build_physical_dispatch_plan(replay, plans, [])
    for command_index in range(2):
        _force_kernel(plans, physical, command_index, 1)
    assert engine_module._plan_fp16_activation_islands(
        replay, plans, physical, precision=engine_module.PRECISION_FLOAT16
    ) == {intermediate: (engine_module.PRECISION_FLOAT16, 1)}

    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    physical = engine_module._build_physical_dispatch_plan(replay, plans, [])
    for command_index in range(2):
        _force_kernel(plans, physical, command_index, 25)
    assert engine_module._plan_fp16_activation_islands(
        replay, plans, physical, precision=engine_module.PRECISION_FLOAT16
    ) == {}


def _concat_stage_replay():
    replay = _replay()
    left = _add_conv(replay, 0, 64, 1)
    right = _add_conv(replay, 0, 64, 1)
    joined = _add_concat_conv1x1(replay, (left, right))
    replay["output_value"] = _add_conv(replay, joined, 64, 3)
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    plans[2]["planned_kernel"] = 3
    plans[2]["precision"] = engine_module.PRECISION_FLOAT16
    plans[2]["packed_value"] = int(replay["commands"][2]["inputs"][-2])
    plans[2]["flags"] = (
        engine_module.PLAN_FLAG_AUTHORITATIVE | engine_module.PLAN_FLAG_DXIL
    )
    physical = engine_module._build_physical_dispatch_plan(replay, plans, [])
    return replay, plans, physical, left, right, joined


def test_fp16_concat1x1_accepts_a_typed_boundary_branch_set():
    replay, plans, physical, left, right, joined = _concat_stage_replay()
    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        precision=engine_module.PRECISION_FLOAT16,
    )
    assert storage == {
        left: (engine_module.PRECISION_FLOAT16, 1),
        right: (engine_module.PRECISION_FLOAT16, 1),
        joined: (engine_module.PRECISION_FLOAT16, 1),
    }
    assert plans[2]["flags"] & IO_MASK == IO_MASK

    replay, plans, physical, left, right, _ = _concat_stage_replay()
    selected = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        precision=engine_module.PRECISION_FLOAT16,
        selected_value_ids=(left,),
    )
    assert set(selected) == {left}
    assert plans[2]["flags"] & IO_MASK == engine_module.PLAN_FLAG_INPUT_FP16

    replay, plans, physical, left, right, _ = _concat_stage_replay()
    selected = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        precision=engine_module.PRECISION_FLOAT16,
        selected_value_ids=(left, right),
    )
    assert set(selected) == {left, right}
    assert plans[2]["flags"] & IO_MASK == engine_module.PLAN_FLAG_INPUT_FP16


def test_fp16_late_concat_fusion_syncs_execution_kernel_and_group_flags():
    replay = _replay()
    left = _add_conv(replay, 0, 64, 1)
    right = _add_conv(replay, 0, 64, 1)
    concat = len(replay["values"])
    replay["values"].append(_value(concat, 128, 40))
    replay["commands"].append(
        {
            "kind": "CONCAT",
            "kind_id": engine_module.COMMAND_KIND["CONCAT"],
            "output": concat,
            "inputs": [left, right],
            "params": [128 * 40 * 40, 4, 1, 2],
        }
    )
    fused_output = _add_conv(replay, concat, 64, 1)
    replay["output_value"] = _add_conv(replay, fused_output, 64, 3)
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    group = engine_module._fusion_record(
        engine_module.FUSION_LATE_CONCAT_CONV1X1,
        2,
        3,
        engine_module.PRECISION_FLOAT16,
    )
    group["kernel"] = 3
    group["flags"] = (
        engine_module.PLAN_FLAG_AUTHORITATIVE | engine_module.PLAN_FLAG_DXIL
    )
    fusion = [group]
    physical = engine_module._build_physical_dispatch_plan(replay, plans, fusion)
    owner = next(
        record
        for record in physical
        if record["fusion_kind"]
        == engine_module.FUSION_LATE_CONCAT_CONV1X1
    )
    before_id = owner["stable_id"]

    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        fusion,
        precision=engine_module.PRECISION_FLOAT16,
    )

    assert set(storage) == {left, right, fused_output}
    assert plans[3]["flags"] & IO_MASK == IO_MASK
    assert group["flags"] & IO_MASK == IO_MASK
    assert owner["flags"] & IO_MASK == IO_MASK
    assert owner["stable_id"] != before_id
    rebuilt = engine_module._build_physical_dispatch_plan(replay, plans, fusion)
    rebuilt_owner = next(
        record
        for record in rebuilt
        if record["fusion_kind"]
        == engine_module.FUSION_LATE_CONCAT_CONV1X1
    )
    assert rebuilt_owner["flags"] == owner["flags"]
    assert rebuilt_owner["stable_id"] == owner["stable_id"]


def test_fp16_stride2_superblock_propagates_external_stage_io():
    replay = _replay()
    stage_input = _add_conv(replay, 0, 64, 1)
    internal = _add_conv(replay, stage_input, 64, 3)
    stage_output = _add_conv(replay, internal, 64, 1)
    replay["output_value"] = _add_conv(replay, stage_output, 64, 3)
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    group = engine_module._fusion_record(
        engine_module.FUSION_STRIDE2_CONV1X1,
        1,
        2,
        engine_module.PRECISION_FLOAT32,
    )
    fusion = [group]
    physical = engine_module._build_physical_dispatch_plan(replay, plans, fusion)

    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        fusion,
        precision=engine_module.PRECISION_FLOAT16,
    )

    assert set(storage) == {stage_input, stage_output}
    assert group["flags"] & IO_MASK == IO_MASK
    owner = next(
        record
        for record in physical
        if record["fusion_kind"] == engine_module.FUSION_STRIDE2_CONV1X1
    )
    assert owner["flags"] & IO_MASK == IO_MASK


def test_fp16_concat_residual_fusion_syncs_execution_kernel_flags():
    replay = _replay()
    left = _add_conv(replay, 0, 64, 1)
    right = _add_conv(replay, 0, 64, 1)
    residual = _add_binary(replay, left, right)
    fused_output = _add_concat_conv1x1(replay, (residual,))
    replay["output_value"] = _add_conv(replay, fused_output, 64, 3)
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    execution_index = 3
    plans[execution_index]["planned_kernel"] = 3
    plans[execution_index]["precision"] = engine_module.PRECISION_FLOAT16
    plans[execution_index]["packed_value"] = int(
        replay["commands"][execution_index]["inputs"][-2]
    )
    plans[execution_index]["flags"] = (
        engine_module.PLAN_FLAG_AUTHORITATIVE | engine_module.PLAN_FLAG_DXIL
    )
    physical = engine_module._build_physical_dispatch_plan(replay, plans, [])
    owner = next(
        record
        for record in physical
        if record["fusion_kind"]
        == engine_module.FUSION_CONCAT_RESIDUAL_CV2
    )
    before_id = owner["stable_id"]

    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        precision=engine_module.PRECISION_FLOAT16,
    )

    assert set(storage) == {left, right, fused_output}
    assert plans[execution_index]["flags"] & IO_MASK == IO_MASK
    assert owner["flags"] & IO_MASK == IO_MASK
    assert owner["stable_id"] != before_id
    rebuilt = engine_module._build_physical_dispatch_plan(replay, plans, [])
    rebuilt_owner = next(
        record
        for record in rebuilt
        if record["fusion_kind"]
        == engine_module.FUSION_CONCAT_RESIDUAL_CV2
    )
    assert rebuilt_owner["flags"] == owner["flags"]
    assert rebuilt_owner["stable_id"] == owner["stable_id"]


def test_fp16_activation_dag_serializes_an_explicit_calibrated_subset():
    replay = _replay()
    first = _add_conv(replay, 0, 64, 1)
    second = _add_conv(replay, first, 64, 3)
    third = _add_conv(replay, second, 64, 1)
    replay["output_value"] = _add_conv(replay, third, 64, 3)

    plans, physical, storage = _plan(
        replay, selected_value_ids=(second,)
    )

    assert storage == {second: (engine_module.PRECISION_FLOAT16, 1)}
    assert [plan["flags"] & IO_MASK for plan in plans] == [
        0,
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        engine_module.PLAN_FLAG_INPUT_FP16,
        0,
    ]
    assert [record["flags"] & IO_MASK for record in physical] == [
        0,
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        engine_module.PLAN_FLAG_INPUT_FP16,
        0,
    ]


def test_fp16_activation_dag_rejects_uncapable_explicit_values():
    replay = _replay()
    intermediate = _add_conv(replay, 0, 64, 1)
    replay["output_value"] = _add_binary(replay, intermediate, 0)

    with pytest.raises(ValueError, match="partial homogeneous input stage"):
        _plan(replay, selected_value_ids=(intermediate,))


def _c2f_plan(replay):
    fusion = engine_module._build_fusion_plan(
        replay, precision=engine_module.PRECISION_FLOAT16, classes=9
    )
    plans, physical, storage = _plan(replay, fusion=fusion)
    tail_group = next(
        group
        for group in fusion
        if group["kind"] == engine_module.FUSION_C2F_TAIL_RESIDUAL
    )
    tail_record = next(
        record
        for record in physical
        if record["fusion_kind"] == engine_module.FUSION_C2F_TAIL_RESIDUAL
    )
    return plans, fusion, physical, storage, tail_group, tail_record


@pytest.mark.parametrize(
    ("channels", "spatial", "kernels", "expected_kernels"),
    [
        (64, 40, (1, 3, 1, 3), (45, 44, 45, 44)),
        (128, 20, (3, 1, 3), (40, 45, 40)),
    ],
)
def test_fp16_activation_dag_propagates_complete_supported_chains(
    channels, spatial, kernels, expected_kernels
):
    replay = _replay(channels, spatial)
    source = 0
    intermediates = []
    for kernel in kernels:
        source = _add_conv(replay, source, channels, kernel)
        intermediates.append(source)
    replay["output_value"] = source

    plans, physical, storage = _plan(replay)

    assert tuple(plan["planned_kernel"] for plan in plans) == expected_kernels
    assert storage == {
        value_id: (engine_module.PRECISION_FLOAT16, 1)
        for value_id in intermediates[:-1]
    }
    assert [plan["flags"] & IO_MASK for plan in plans] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        *(
            engine_module.PLAN_FLAG_INPUT_FP16
            | engine_module.PLAN_FLAG_OUTPUT_FP16
            for _ in kernels[1:-1]
        ),
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]

    arena = build_page_colored_arena(
        replay, physical_dispatch_plan=physical, storage=storage
    )
    assert all(
        arena["by_value"][value_id]["storage_dtype"]
        == engine_module.PRECISION_FLOAT16
        for value_id in intermediates[:-1]
    )


def test_fp16_activation_dag_accepts_fanout_only_when_every_branch_is_supported():
    replay = _replay()
    shared = _add_conv(replay, 0, 64, 1)
    left = _add_conv(replay, shared, 64, 3)
    right = _add_conv(replay, shared, 64, 1)
    replay["output_value"] = _add_binary(replay, left, right)

    plans, _, storage = _plan(replay)

    assert storage == {
        shared: (engine_module.PRECISION_FLOAT16, 1),
        left: (engine_module.PRECISION_FLOAT16, 1),
        right: (engine_module.PRECISION_FLOAT16, 1),
    }
    assert [plan["flags"] & IO_MASK for plan in plans] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        IO_MASK,
        IO_MASK,
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]

    blocked = _replay()
    shared = _add_conv(blocked, 0, 64, 1)
    _add_conv(blocked, shared, 64, 3)
    blocked["output_value"] = _add_binary(blocked, shared, 0)

    blocked_plans, _, blocked_storage = _plan(blocked)

    assert blocked_storage == {}
    assert all(plan["flags"] & IO_MASK == 0 for plan in blocked_plans)


def test_fp16_activation_dag_propagates_through_zero_dispatch_alias():
    replay = _replay()
    source = _add_conv(replay, 0, 64, 1)
    alias = len(replay["values"])
    replay["values"].append(_value(alias, 64, 40))
    replay["commands"].append(
        {
            "kind": "VIEW",
            "kind_id": engine_module.COMMAND_KIND["VIEW"],
            "output": alias,
            "inputs": [source],
            "params": [0, 64 * 40 * 40],
        }
    )
    replay["output_value"] = _add_conv(replay, alias, 64, 3)

    plans, physical, storage = _plan(replay)

    assert storage == {source: (engine_module.PRECISION_FLOAT16, 1)}
    assert replay["values"][alias]["flags"] & engine_module.VALUE_ALIAS
    assert [plan["flags"] & IO_MASK for plan in plans] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        0,
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]
    assert [record["zero_dispatch"] for record in physical] == [False, True, False]
    assert [record["logical_indices"] for record in physical] == [(0,), (1,), (2,)]
    arena = build_page_colored_arena(
        replay, physical_dispatch_plan=physical, storage=storage
    )
    assert arena["by_value"][source]["storage_dtype"] == engine_module.PRECISION_FLOAT16
    assert arena["by_value"][alias]["storage_dtype"] == engine_module.PRECISION_FLOAT16
    assert (
        arena["by_value"][alias]["page_id"],
        arena["by_value"][alias]["offset"],
    ) == (
        arena["by_value"][source]["page_id"],
        arena["by_value"][source]["offset"],
    )


def test_fp16_activation_dag_never_crosses_a_fused_physical_dispatch():
    replay = _replay()
    shared = _add_conv(replay, 0, 64, 1)
    branch = _add_conv(replay, shared, 64, 3)
    replay["output_value"] = _add_binary(replay, branch, 0)
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    singleton = engine_module._build_physical_dispatch_plan(replay, plans, [])
    fused = [
        singleton[0],
        {
            **singleton[1],
            "logical_end": 2,
            "logical_indices": (1, 2),
            "fusion_kind": engine_module.FUSION_PAIRED_CONV3X3_SILU,
        },
    ]

    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        fused,
        precision=engine_module.PRECISION_FLOAT16,
    )

    assert storage == {}
    assert all(plan["flags"] & IO_MASK == 0 for plan in plans)


def test_fp16_activation_dag_propagates_through_c2f_dual_input_long_chain():
    replay = _replay(channels=128, spatial=20)
    residual = _add_conv(replay, 0, 128, 1)
    tail_input, tail_conv, tail_output = _add_c2f_tail(replay, residual)
    post_1x1 = _add_conv(replay, tail_output, 128, 1)
    graph_output = _add_conv(replay, post_1x1, 128, 3)
    replay["output_value"] = graph_output

    plans, fusion, physical, storage, tail_group, tail_record = _c2f_plan(
        replay
    )

    assert storage == {
        residual: (engine_module.PRECISION_FLOAT16, 1),
        tail_input: (engine_module.PRECISION_FLOAT16, 1),
        tail_output: (engine_module.PRECISION_FLOAT16, 1),
        post_1x1: (engine_module.PRECISION_FLOAT16, 1),
    }
    assert tail_conv not in storage
    assert graph_output not in storage
    assert tail_group["flags"] & IO_MASK == IO_MASK
    assert tail_record["flags"] & IO_MASK == IO_MASK
    assert [plan["flags"] & IO_MASK for plan in plans] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        IO_MASK,
        0,
        0,
        IO_MASK,
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]
    assert [record["flags"] & IO_MASK for record in physical] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        IO_MASK,
        IO_MASK,
        IO_MASK,
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]

    # Rebuilding is the exact serialization path: the typed flags must come
    # from the kernel/fusion plans, not only from a transient DAG annotation.
    rebuilt = engine_module._build_physical_dispatch_plan(replay, plans, fusion)
    assert [record["flags"] for record in rebuilt] == [
        record["flags"] for record in physical
    ]
    assert [record["stable_id"] for record in rebuilt] == [
        record["stable_id"] for record in physical
    ]

    arena = build_page_colored_arena(
        replay, physical_dispatch_plan=rebuilt, storage=storage
    )
    for value_id in storage:
        assert arena["by_value"][value_id]["storage_dtype"] == (
            engine_module.PRECISION_FLOAT16
        )
        assert arena["by_value"][value_id]["storage_layout"] == 1
    assert arena["by_value"][graph_output]["storage_dtype"] == (
        engine_module.PRECISION_FLOAT32
    )


def test_fp16_c2f_fusion_rejects_partial_residual_inputs():
    replay = _replay(channels=128, spatial=20)
    residual = _add_conv(replay, 0, 128, 1)
    tail_input, _, tail_output = _add_c2f_tail(replay, residual)
    replay["output_value"] = _add_conv(replay, tail_output, 128, 1)
    fusion = engine_module._build_fusion_plan(
        replay, precision=engine_module.PRECISION_FLOAT16, classes=9
    )
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    physical = engine_module._build_physical_dispatch_plan(replay, plans, fusion)

    with pytest.raises(ValueError, match="partial homogeneous input stage"):
        engine_module._plan_fp16_activation_islands(
            replay,
            plans,
            physical,
            fusion,
            precision=engine_module.PRECISION_FLOAT16,
            selected_value_ids=(residual,),
        )

    selected = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        fusion,
        precision=engine_module.PRECISION_FLOAT16,
        selected_value_ids=(residual, tail_input),
    )
    assert set(selected) == {residual, tail_input}
    group = next(
        group
        for group in fusion
        if group["kind"] == engine_module.FUSION_C2F_TAIL_RESIDUAL
    )
    assert group["flags"] & IO_MASK == engine_module.PLAN_FLAG_INPUT_FP16


def test_fp16_activation_dag_c2f_fanout_requires_every_consumer_capability():
    replay = _replay(channels=128, spatial=20)
    residual = _add_conv(replay, 0, 128, 1)
    tail_input, _, tail_output = _add_c2f_tail(replay, residual)
    left = _add_conv(replay, tail_output, 128, 1)
    right = _add_conv(replay, tail_output, 128, 3)
    replay["output_value"] = _add_binary(replay, left, right)

    plans, _, _, storage, tail_group, tail_record = _c2f_plan(replay)

    assert storage == {
        residual: (engine_module.PRECISION_FLOAT16, 1),
        tail_input: (engine_module.PRECISION_FLOAT16, 1),
        tail_output: (engine_module.PRECISION_FLOAT16, 1),
        left: (engine_module.PRECISION_FLOAT16, 1),
        right: (engine_module.PRECISION_FLOAT16, 1),
    }
    assert tail_group["flags"] & IO_MASK == IO_MASK
    assert tail_record["flags"] & IO_MASK == IO_MASK
    assert plans[4]["flags"] & IO_MASK == IO_MASK
    assert plans[5]["flags"] & IO_MASK == IO_MASK

    blocked = _replay(channels=128, spatial=20)
    blocked_residual = _add_conv(blocked, 0, 128, 1)
    blocked_input, _, blocked_output = _add_c2f_tail(
        blocked, blocked_residual
    )
    supported_branch = _add_conv(blocked, blocked_output, 128, 1)
    blocked["output_value"] = _add_binary(
        blocked, blocked_output, supported_branch
    )

    (
        blocked_plans,
        _,
        _,
        blocked_storage,
        blocked_group,
        blocked_record,
    ) = _c2f_plan(blocked)

    assert blocked_storage == {
        blocked_residual: (engine_module.PRECISION_FLOAT16, 1),
        blocked_input: (engine_module.PRECISION_FLOAT16, 1),
        blocked_output: (engine_module.PRECISION_FLOAT16, 1),
        supported_branch: (engine_module.PRECISION_FLOAT16, 1),
    }
    assert blocked_group["flags"] & IO_MASK == IO_MASK
    assert blocked_record["flags"] & IO_MASK == IO_MASK
    assert blocked_plans[4]["flags"] & IO_MASK == IO_MASK
