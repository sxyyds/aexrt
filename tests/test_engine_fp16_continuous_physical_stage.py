import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aexrt import engine as engine_module  # noqa: E402
from aexrt.barrier_v2 import build_arena_barrier_plan  # noqa: E402
from aexrt.memory_v2 import (  # noqa: E402
    build_page_colored_arena,
    build_physical_value_dag,
)


IO_MASK = engine_module.PLAN_FLAG_INPUT_FP16 | engine_module.PLAN_FLAG_OUTPUT_FP16


def _value(value_id, shape, *, flags=0):
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return {
        "id": value_id,
        "flags": flags,
        "elements": elements,
        "shape4": tuple(int(dim) for dim in shape),
    }


def _replay(channels, spatial):
    return {
        "values": [
            _value(
                0,
                (1, channels, spatial, spatial),
                flags=engine_module.VALUE_INPUT,
            )
        ],
        "commands": [],
        "input_value": 0,
        "output_value": 0,
    }


def _constant(replay, shape):
    value_id = len(replay["values"])
    replay["values"].append(
        _value(value_id, shape, flags=engine_module.VALUE_CONSTANT)
    )
    return value_id


def _add_conv(
    replay,
    source,
    out_channels,
    *,
    kernel=1,
    stride=1,
    out_spatial=None,
):
    _, in_channels, in_height, in_width = replay["values"][source]["shape4"]
    assert in_height == in_width
    if out_spatial is None:
        out_spatial = in_height if stride == 1 else in_height // stride
    weight = _constant(
        replay, (out_channels, in_channels, kernel, kernel)
    )
    bias = _constant(replay, (1, out_channels, 1, 1))
    output = len(replay["values"])
    replay["values"].append(
        _value(output, (1, out_channels, out_spatial, out_spatial))
    )
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
                in_height,
                in_width,
                out_channels,
                out_spatial,
                out_spatial,
                kernel,
                kernel,
                stride,
                stride,
                pad,
                pad,
                1,
                1,
                1,
            ],
        }
    )
    return output


def _add_concat_conv1x1(replay, sources, out_channels):
    shapes = [replay["values"][source]["shape4"] for source in sources]
    assert len({(shape[0], shape[2], shape[3]) for shape in shapes}) == 1
    batch, _, height, width = shapes[0]
    channels = [int(shape[1]) for shape in shapes]
    weight = _constant(replay, (out_channels, sum(channels), 1, 1))
    bias = _constant(replay, (1, out_channels, 1, 1))
    output = len(replay["values"])
    replay["values"].append(
        _value(output, (batch, out_channels, height, width))
    )
    replay["commands"].append(
        {
            "kind": "CONCAT_CONV1X1",
            "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
            "output": output,
            "inputs": [*sources, weight, bias],
            "params": [batch, height, width, out_channels, 1, *channels],
        }
    )
    return output


def _add_binary_add(replay, first, second):
    shape = replay["values"][first]["shape4"]
    assert shape == replay["values"][second]["shape4"]
    output = len(replay["values"])
    replay["values"].append(_value(output, shape))
    elements = replay["values"][output]["elements"]
    replay["commands"].append(
        {
            "kind": "BINARY",
            "kind_id": engine_module.COMMAND_KIND["BINARY"],
            "output": output,
            "inputs": [first, second],
            # The first two words are the compiler-side ABI used by
            # _is_binary_add; the remaining words mirror the prepared runtime
            # descriptor width.
            "params": [elements, 0, *([0] * 22)],
        }
    )
    return output


def _add_alias(replay, source, kind):
    output = len(replay["values"])
    source_value = replay["values"][source]
    replay["values"].append(_value(output, source_value["shape4"]))
    replay["commands"].append(
        {
            "kind": kind,
            "kind_id": engine_module.COMMAND_KIND[kind],
            "output": output,
            "inputs": [source],
            "params": [0, source_value["elements"]],
        }
    )
    return output


def _compile(replay, *, selected=None):
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    fusion = engine_module._build_fusion_plan(
        replay, precision=engine_module.PRECISION_FLOAT16, classes=9
    )
    engine_module._apply_measured_fp16_physical_kernels(
        replay,
        plans,
        fusion,
        precision=engine_module.PRECISION_FLOAT16,
    )
    physical = engine_module._build_physical_dispatch_plan(
        replay, plans, fusion
    )
    storage = engine_module._plan_fp16_activation_islands(
        replay,
        plans,
        physical,
        fusion,
        precision=engine_module.PRECISION_FLOAT16,
        selected_value_ids=selected,
    )
    if storage:
        physical = engine_module._build_physical_dispatch_plan(
            replay, plans, fusion
        )
    arena = build_page_colored_arena(
        replay, physical_dispatch_plan=physical, storage=storage
    )
    barriers = build_arena_barrier_plan(replay, physical, arena)
    return plans, fusion, physical, storage, arena, barriers


def _continuous_mixed_stage():
    replay = _replay(128, 80)
    stride33 = _add_conv(
        replay, 0, 128, kernel=3, stride=2, out_spatial=40
    )
    shared = _add_conv(replay, stride33, 128)
    left = _add_conv(replay, shared, 64)
    right = _add_conv(replay, shared, 64)
    concat = _add_concat_conv1x1(replay, (left, right), 128)
    residual = _add_binary_add(replay, concat, shared)
    view = _add_alias(replay, residual, "VIEW")
    alias = _add_alias(replay, view, "ALIAS")
    stride43 = _add_conv(
        replay, alias, 128, kernel=3, stride=2, out_spatial=20
    )
    winograd = _add_conv(replay, stride43, 128, kernel=3)
    pointwise = _add_conv(replay, winograd, 128)
    replay["output_value"] = _add_conv(
        replay, pointwise, 128, kernel=3
    )
    material_fp16 = {
        stride33,
        shared,
        left,
        right,
        concat,
        residual,
        stride43,
        winograd,
        pointwise,
    }
    return replay, material_fp16, (view, alias)


def test_continuous_physical_stage_crosses_concat_binary_and_transparent_aliases():
    replay, expected_values, aliases = _continuous_mixed_stage()

    plans, _, physical, storage, arena, barriers = _compile(replay)

    assert [plan["planned_kernel"] for plan in plans] == [
        33,
        45,
        45,
        45,
        3,
        0,
        0,
        0,
        43,
        40,
        45,
        40,
    ]
    assert set(storage) == expected_values
    assert all(
        storage[value_id]
        == (engine_module.PRECISION_FLOAT16, engine_module.LAYOUT_LINEAR_NCHW)
        for value_id in expected_values
    )
    assert [plan["flags"] & IO_MASK for plan in plans] == [
        engine_module.PLAN_FLAG_OUTPUT_FP16,
        IO_MASK,
        IO_MASK,
        IO_MASK,
        IO_MASK,
        IO_MASK,
        0,
        0,
        IO_MASK,
        IO_MASK,
        IO_MASK,
        engine_module.PLAN_FLAG_INPUT_FP16,
    ]

    dag = build_physical_value_dag(replay, physical)
    assert dag["material_dispatch_count"] == len(replay["commands"]) - 2
    assert tuple(
        physical[index]["execution_index"]
        for index in dag["zero_dispatch_indices"]
    ) == (6, 7)
    assert barriers["dispatch_count"] == len(physical)

    canonical = arena["by_value"][int(replay["commands"][5]["output"])]
    for alias in aliases:
        binding = arena["by_value"][alias]
        assert binding["flags"] & engine_module.VALUE_ALIAS
        assert (
            binding["page_id"],
            binding["offset"],
            binding["nbytes"],
            binding["storage_dtype"],
            binding["storage_layout"],
        ) == (
            canonical["page_id"],
            canonical["offset"],
            canonical["nbytes"],
            engine_module.PRECISION_FLOAT16,
            engine_module.LAYOUT_LINEAR_NCHW,
        )


@pytest.mark.parametrize(
    (
        "input_channels",
        "input_spatial",
        "stride_channels",
        "output_spatial",
        "expected_stride_kernel",
        "pointwise_channels",
        "tail_kernel",
    ),
    (
        (64, 80, 64, 40, 24, 64, 3),
        (128, 80, 128, 40, 33, 128, 1),
        (128, 40, 128, 20, 43, 128, 3),
        (256, 40, 256, 20, 47, 128, 3),
    ),
)
def test_stride2_kernel_is_a_typed_internal_stage_not_an_fp32_boundary(
    input_channels,
    input_spatial,
    stride_channels,
    output_spatial,
    expected_stride_kernel,
    pointwise_channels,
    tail_kernel,
):
    replay = _replay(input_channels, input_spatial)
    stride = _add_conv(
        replay,
        0,
        stride_channels,
        kernel=3,
        stride=2,
        out_spatial=output_spatial,
    )
    pointwise = _add_conv(replay, stride, pointwise_channels)
    replay["output_value"] = _add_conv(
        replay, pointwise, pointwise_channels, kernel=tail_kernel
    )

    plans, _, physical, storage, arena, _ = _compile(replay)

    assert [plan["planned_kernel"] for plan in plans] == [
        expected_stride_kernel,
        45,
        44 if (output_spatial, pointwise_channels, tail_kernel) == (40, 64, 3)
        else 45 if tail_kernel == 1
        else 40,
    ]
    if physical[0]["fusion_kind"] == engine_module.FUSION_STRIDE2_CONV1X1:
        # The stride2 result stays group-local and is never materialized.  The
        # combined physical dispatch writes the pointwise result directly in
        # FP16, which is stronger than retaining a typed intermediate edge.
        assert storage == {
            pointwise: (
                engine_module.PRECISION_FLOAT16,
                engine_module.LAYOUT_LINEAR_NCHW,
            )
        }
        assert [plan["flags"] & IO_MASK for plan in plans] == [
            0,
            0,
            engine_module.PLAN_FLAG_INPUT_FP16,
        ]
        assert [record["flags"] & IO_MASK for record in physical] == [
            engine_module.PLAN_FLAG_OUTPUT_FP16,
            engine_module.PLAN_FLAG_INPUT_FP16,
        ]
        # The logical value may retain an ABI binding, but it is not part of
        # the selected FP16 storage set and the fused shader never writes it.
        assert arena["by_value"][stride]["storage_dtype"] == (
            engine_module.PRECISION_FLOAT32
        )
    else:
        assert storage == {
            stride: (
                engine_module.PRECISION_FLOAT16,
                engine_module.LAYOUT_LINEAR_NCHW,
            ),
            pointwise: (
                engine_module.PRECISION_FLOAT16,
                engine_module.LAYOUT_LINEAR_NCHW,
            ),
        }
        assert [plan["flags"] & IO_MASK for plan in plans] == [
            engine_module.PLAN_FLAG_OUTPUT_FP16,
            IO_MASK,
            engine_module.PLAN_FLAG_INPUT_FP16,
        ]
        assert [record["flags"] & IO_MASK for record in physical] == [
            engine_module.PLAN_FLAG_OUTPUT_FP16,
            IO_MASK,
            engine_module.PLAN_FLAG_INPUT_FP16,
        ]
        assert arena["by_value"][stride]["nbytes"] == (
            replay["values"][stride]["elements"] * 2
        )


def test_concat_typed_stage_accepts_a_partial_explicit_boundary_selection():
    replay, expected_values, _ = _continuous_mixed_stage()
    concat_index = next(
        index
        for index, command in enumerate(replay["commands"])
        if command["kind"] == "CONCAT_CONV1X1"
    )
    left, right = replay["commands"][concat_index]["inputs"][:2]
    assert {left, right}.issubset(expected_values)

    plans, _, physical, storage, _, _ = _compile(replay, selected=(left,))
    assert set(storage) == {left}
    assert plans[concat_index]["flags"] & IO_MASK == (
        engine_module.PLAN_FLAG_INPUT_FP16
    )
    concat_record = next(
        record for record in physical if concat_index in record["logical_indices"]
    )
    assert concat_record["flags"] & IO_MASK == (
        engine_module.PLAN_FLAG_INPUT_FP16
    )


def test_winograd_residual_fusion_reduces_physical_dispatches_inside_fp16_stage():
    replay = _replay(128, 20)
    residual = _add_conv(replay, 0, 128)
    tail_input = _add_conv(replay, residual, 128)
    tail_conv = _add_conv(replay, tail_input, 128, kernel=3)
    tail_output = _add_binary_add(replay, residual, tail_conv)
    post = _add_conv(replay, tail_output, 128)
    replay["output_value"] = _add_conv(replay, post, 128, kernel=3)

    plans, fusion, physical, storage, arena, barriers = _compile(replay)

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
    assert tail_record["logical_indices"] == (2, 3)
    assert tail_record["execution_index"] == 2
    assert tail_group["flags"] & IO_MASK == IO_MASK
    assert tail_record["flags"] & IO_MASK == IO_MASK
    assert len(physical) == len(replay["commands"]) - 1
    assert build_physical_value_dag(replay, physical)["material_dispatch_count"] == (
        len(replay["commands"]) - 1
    )
    assert barriers["dispatch_count"] == len(physical)
    assert set(storage) == {residual, tail_input, tail_output, post}
    assert tail_conv not in storage
    assert all(
        arena["by_value"][value_id]["storage_dtype"]
        == engine_module.PRECISION_FLOAT16
        for value_id in storage
    )
    assert [plan["planned_kernel"] for plan in plans] == [45, 45, 40, 0, 45, 40]
