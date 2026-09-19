import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import aexrt.engine as engine_module
from aexrt import Graph, TensorSpec, save_aexrt_engine


NO_VALUE = (1 << 32) - 1


def _replay_value(value_id, shape, *, raw=b""):
    return {
        "id": value_id,
        "flags": engine_module.VALUE_CONSTANT if raw else 0,
        "elements": int(np.prod(shape)),
        "shape4": tuple(shape),
        "raw": raw,
    }


def test_fp16_conv1x1_plan_embeds_oc8_packed_weight(tmp_path):
    channels = 64
    dim = 40
    weight = (np.arange(channels * channels, dtype=np.float32) / 1024.0).reshape(
        channels, channels, 1, 1
    )
    graph = Graph("fp16_conv1x1_oc8")
    graph.input("images", TensorSpec((1, channels, dim, dim), "float32"))
    graph.const("w", weight)
    graph.const("b", np.zeros((channels,), dtype="float32"))
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
    graph.reshape("output0", "c", (1, channels, dim * dim))
    graph.output("output0")

    path = tmp_path / "conv1x1.aexrt"
    info = save_aexrt_engine(graph, path, classes=channels - 4, precision="fp16")
    plan = info["kernel_plan"][0]
    assert plan[2:6] == (45, 2, 1, 3)
    assert info["packed_layout_counts"] == {4: 1}

    data = path.read_bytes()
    section_base = info["sections"][engine_module.SECTION_PACKED_WEIGHTS]["offset"]
    _, layout, in_channels, out_channels, elements, offset, nbytes = struct.unpack_from(
        "<4IQQQ", data, section_base + 8
    )
    assert (layout, in_channels, out_channels, elements, nbytes) == (
        4,
        channels,
        channels,
        channels * channels,
        channels * channels * 2,
    )
    actual = np.frombuffer(
        data, dtype="<f2", count=elements, offset=section_base + offset
    ).reshape(channels // 8, channels, 8)
    expected = weight.reshape(channels // 8, 8, channels).transpose(0, 2, 1)
    np.testing.assert_array_equal(actual, expected.astype("<f2"))


@pytest.mark.parametrize(
    ("in_channels", "dim", "out_channels"),
    (
        (64, 40, 64),
        (128, 40, 64),
        (128, 40, 128),
        (256, 40, 64),
        (128, 20, 128),
        (256, 20, 128),
        (384, 20, 256),
        (512, 20, 128),
        (512, 10, 512),
        (256, 10, 256),
    ),
)
def test_fp16_conv1x1_audit_gate_accepts_measured_winner(
    in_channels, dim, out_channels
):
    params = [
        1,
        in_channels,
        dim,
        dim,
        out_channels,
        dim,
        dim,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
    ]
    replay = {
        "commands": [
            {
                "kind": "CONV_SILU",
                "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
                "inputs": [0, 1, 2],
                "output": 3,
                "params": params,
            }
        ]
    }
    plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )[0]
    assert (plan["planned_kernel"], plan["precision"], plan["packed_value"], plan["flags"]) == (
        45,
        2,
        1,
        3,
    )


@pytest.mark.parametrize(
    "params",
    (
        [2, 64, 20, 20, 64, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 64, 20, 20, 64, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 64, 20, 10, 64, 20, 10, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 60, 20, 20, 64, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 68, 20, 20, 68, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 512, 10, 10, 256, 10, 10, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    ),
)
def test_fp16_conv1x1_audit_gate_rejects_unprofiled_shapes(params):
    replay = {
        "commands": [
            {
                "kind": "CONV",
                "kind_id": engine_module.COMMAND_KIND["CONV"],
                "inputs": [0, 1, 2],
                "output": 3,
                "params": params,
            }
        ]
    }
    plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )[0]
    assert plan["planned_kernel"] == 1
    assert plan["packed_value"] == NO_VALUE
    assert plan["flags"] == 1


def test_fp16_direct_concat_conv1x1_plan_stays_on_measured_sm5_winner():
    in_channels = out_channels = 64
    weight_id = 2
    weight = np.arange(out_channels * in_channels, dtype=np.float32)
    replay = {
        "values": [
            _replay_value(0, (1, 32, 20, 20)),
            _replay_value(1, (1, 32, 20, 20)),
            _replay_value(weight_id, (out_channels, in_channels, 1, 1), raw=weight.tobytes()),
            _replay_value(3, (out_channels,), raw=np.zeros(out_channels, dtype=np.float32).tobytes()),
            _replay_value(4, (1, out_channels, 20, 20)),
        ],
        "commands": [
            {
                "kind": "CONCAT_CONV1X1",
                "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
                "inputs": [0, 1, weight_id, 3],
                "output": 4,
                "params": [1, 20, 20, out_channels, 1, 32, 32],
            }
        ],
    }
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    assert (
        plans[0]["planned_kernel"],
        plans[0]["precision"],
        plans[0]["packed_value"],
        plans[0]["flags"],
    ) == (1, 2, NO_VALUE, 1)
    packed = engine_module._build_packed_weights(
        replay, plans, precision=engine_module.PRECISION_FLOAT16
    )
    assert packed == []

    replay["commands"][0]["params"][3] = 68
    fallback = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )[0]
    assert (
        fallback["planned_kernel"],
        fallback["precision"],
        fallback["packed_value"],
        fallback["flags"],
    ) == (1, 2, NO_VALUE, 1)


def test_fp16_late_concat_plan_keeps_sm5_without_packed_weight():
    in_channels = out_channels = 64
    weight_id = 3
    weight = np.arange(out_channels * in_channels, dtype=np.float32)
    concat_params = [
        1 * in_channels * 20 * 20,
        4,
        1,
        2,
        1,
        in_channels,
        20,
        20,
        32,
        32,
        32,
        32,
        32,
        32,
        32,
        32,
        0,
        0,
    ]
    conv_params = [1, in_channels, 20, 20, out_channels, 20, 20, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    replay = {
        "values": [
            _replay_value(0, (1, 32, 20, 20)),
            _replay_value(1, (1, 32, 20, 20)),
            _replay_value(2, (1, in_channels, 20, 20)),
            _replay_value(weight_id, (out_channels, in_channels, 1, 1), raw=weight.tobytes()),
            _replay_value(4, (out_channels,), raw=np.zeros(out_channels, dtype=np.float32).tobytes()),
            _replay_value(5, (1, out_channels, 20, 20)),
        ],
        "commands": [
            {
                "kind": "CONCAT",
                "kind_id": engine_module.COMMAND_KIND["CONCAT"],
                "inputs": [0, 1],
                "output": 2,
                "params": concat_params,
            },
            {
                "kind": "CONV_SILU",
                "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
                "inputs": [2, weight_id, 4],
                "output": 5,
                "params": conv_params,
            },
        ],
    }
    fusion_plan = engine_module._build_fusion_plan(
        replay, precision=engine_module.PRECISION_FLOAT16, classes=60
    )
    late_groups = [group for group in fusion_plan if group["kind"] == 3]
    assert len(late_groups) == 1
    assert (late_groups[0]["precision"], late_groups[0]["kernel"], late_groups[0]["flags"]) == (2, 0, 1)

    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    packed = engine_module._build_packed_weights(
        replay,
        plans,
        precision=engine_module.PRECISION_FLOAT16,
        fusion_plan=fusion_plan,
    )
    assert packed == []

    plans[1] = {
        **plans[1],
        "planned_kernel": 1,
        "packed_value": NO_VALUE,
        "flags": 1,
    }
    fusion_only_packed = engine_module._build_packed_weights(
        replay,
        plans,
        precision=engine_module.PRECISION_FLOAT16,
        fusion_plan=fusion_plan,
    )
    assert fusion_only_packed == []

    unprofiled_replay = {
        **replay,
        "commands": [
            replay["commands"][0],
            {
                **replay["commands"][1],
                "params": [
                    *conv_params[:4],
                    68,
                    *conv_params[5:],
                ],
            },
        ],
    }
    fallback_groups = engine_module._build_fusion_plan(
        unprofiled_replay, precision=engine_module.PRECISION_FLOAT16, classes=60
    )
    fallback = next(group for group in fallback_groups if group["kind"] == 3)
    assert (fallback["precision"], fallback["kernel"], fallback["flags"]) == (2, 0, 1)


def test_fp16_residual_concat_matches_direct_measured_shape_and_physical_ownership():
    channels = 128
    out_channels = 256
    weight_id = 6
    command = {
        "kind": "CONCAT_CONV1X1",
        "kind_id": engine_module.COMMAND_KIND["CONCAT_CONV1X1"],
        "inputs": [2, 3, 4, weight_id, 7],
        "output": 8,
        "params": [1, 20, 20, out_channels, 1, channels, channels, channels],
    }
    replay = {
        "values": [
            _replay_value(0, (1, channels, 20, 20)),
            _replay_value(1, (1, channels, 20, 20)),
            _replay_value(2, (1, channels, 20, 20)),
            _replay_value(3, (1, channels, 20, 20)),
            _replay_value(4, (1, channels, 20, 20)),
            _replay_value(5, (1, channels, 20, 20)),
            _replay_value(
                weight_id,
                (out_channels, channels * 3, 1, 1),
                raw=np.zeros(out_channels * channels * 3, dtype=np.float32).tobytes(),
            ),
            _replay_value(
                7,
                (out_channels,),
                raw=np.zeros(out_channels, dtype=np.float32).tobytes(),
            ),
            _replay_value(8, (1, out_channels, 20, 20)),
        ],
        "commands": [
            {
                "kind": "BINARY",
                "kind_id": engine_module.COMMAND_KIND["BINARY"],
                "inputs": [0, 1],
                "output": 2,
                "params": [0, 0],
            },
            command,
        ],
    }
    assert engine_module._uses_native_fp16_concat_conv1x1(command)

    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    assert plans[1]["planned_kernel"] == 3
    engine_module._apply_measured_fp16_physical_kernels(
        replay,
        plans,
        [],
        precision=engine_module.PRECISION_FLOAT16,
    )
    assert (
        plans[1]["planned_kernel"],
        plans[1]["precision"],
        plans[1]["packed_value"],
        plans[1]["flags"],
    ) == (3, 2, weight_id, 3)

    no_residual = {**replay, "commands": [command]}
    no_residual_plans = engine_module._build_kernel_plan(
        no_residual, precision=engine_module.PRECISION_FLOAT16
    )
    engine_module._apply_measured_fp16_physical_kernels(
        no_residual,
        no_residual_plans,
        [],
        precision=engine_module.PRECISION_FLOAT16,
    )
    assert no_residual_plans[0]["planned_kernel"] == 3


def test_fp16_late_concat_serializes_measured_conv_and_fusion_kernels():
    in_channels = 384
    out_channels = 256
    weight_id = 3
    replay = {
        "values": [
            _replay_value(0, (1, 128, 20, 20)),
            _replay_value(1, (1, 256, 20, 20)),
            _replay_value(2, (1, in_channels, 20, 20)),
            _replay_value(
                weight_id,
                (out_channels, in_channels, 1, 1),
                raw=np.zeros(out_channels * in_channels, dtype=np.float32).tobytes(),
            ),
            _replay_value(
                4,
                (out_channels,),
                raw=np.zeros(out_channels, dtype=np.float32).tobytes(),
            ),
            _replay_value(5, (1, out_channels, 20, 20)),
        ],
        "commands": [
            {
                "kind": "CONCAT",
                "kind_id": engine_module.COMMAND_KIND["CONCAT"],
                "inputs": [0, 1],
                "output": 2,
                "params": [
                    153600,
                    4,
                    1,
                    2,
                    1,
                    in_channels,
                    20,
                    20,
                    128,
                    256,
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
            {
                "kind": "CONV_SILU",
                "kind_id": engine_module.COMMAND_KIND["CONV_SILU"],
                "inputs": [2, weight_id, 4],
                "output": 5,
                "params": [
                    1,
                    in_channels,
                    20,
                    20,
                    out_channels,
                    20,
                    20,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                ],
            },
        ],
    }
    plans = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )
    assert (
        plans[1]["planned_kernel"],
        plans[1]["precision"],
        plans[1]["packed_value"],
        plans[1]["flags"],
    ) == (45, 2, weight_id, 3)

    fusion_plan = engine_module._build_fusion_plan(
        replay, precision=engine_module.PRECISION_FLOAT16, classes=60
    )
    late = next(
        group
        for group in fusion_plan
        if group["kind"] == engine_module.FUSION_LATE_CONCAT_CONV1X1
    )
    assert (late["precision"], late["kernel"], late["flags"]) == (2, 3, 3)
    packed = engine_module._build_packed_weights(
        replay,
        plans,
        precision=engine_module.PRECISION_FLOAT16,
        fusion_plan=fusion_plan,
    )
    assert len(packed) == 1
    assert packed[0]["layout"] == engine_module.PACKED_LAYOUT_CONV1X1_OC8_FP16


def test_packed_weight_inspector_rejects_invalid_oc8_layout_dimensions():
    in_channels = 66
    out_channels = 64
    elements = in_channels * out_channels
    offset = 64
    payload = bytearray(offset + elements * 2)
    struct.pack_into("<II", payload, 0, 1, 0)
    struct.pack_into(
        "<4IQQQ",
        payload,
        8,
        0,
        engine_module.PACKED_LAYOUT_CONV1X1_OC8_FP16,
        in_channels,
        out_channels,
        elements,
        offset,
        elements * 2,
    )
    with pytest.raises(ValueError, match="invalid AEXRT packed-weight payload"):
        engine_module._inspect_packed_weights(
            bytes(payload), {"offset": 0, "size": len(payload)}
        )
