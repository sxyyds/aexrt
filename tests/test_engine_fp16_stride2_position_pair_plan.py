import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import aexrt.engine as engine_module
from aexrt import Graph, TensorSpec, inspect_aexrt_engine, save_aexrt_engine


EXACT_DESC = (
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


def _build_exact_graph() -> Graph:
    graph = Graph("native_fp16_stride2_position_pair_40x40_256x256")
    graph.input("images", TensorSpec((1, 256, 40, 40), "float32"))
    graph.const("weight", np.zeros((256, 256, 3, 3), dtype="float32"))
    graph.const("bias", np.zeros((256,), dtype="float32"))
    graph.node(
        "Conv",
        "conv",
        "images",
        "weight",
        "bias",
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph.sigmoid("sigmoid", "conv")
    graph.mul("activated", "conv", "sigmoid")
    graph.reshape("output0", "activated", (1, 256, 400))
    graph.output("output0")
    return graph


def test_k_major_oc4_pack_orders_each_k_slice_by_output_group():
    in_channels = 4
    out_channels = 8
    weight_id = 1
    weights = np.arange(
        out_channels * in_channels * 9, dtype=np.float32
    ).reshape(out_channels, in_channels, 3, 3)
    packed = {}
    engine_module._add_packed_conv3x3_k_major_oc4_fp16(
        packed,
        [{"raw": b""}, {"raw": weights.tobytes(order="C")}],
        weight_id=weight_id,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    item = packed[weight_id]
    assert item["layout"] == engine_module.PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16
    expected = (
        weights.reshape(out_channels // 4, 4, in_channels, 9)
        .transpose(2, 3, 0, 1)
        .astype("<f2")
    )
    actual = np.frombuffer(item["data"], dtype="<f2").reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


def test_fp16_stride2_position_pair_plan_is_exact_and_embeds_k_major_layout(tmp_path):
    graph = _build_exact_graph()
    optimized = save_aexrt_engine(
        graph,
        tmp_path / "position_pair.aexrt",
        classes=252,
        max_detections=4,
    )
    fallback = save_aexrt_engine(
        graph,
        tmp_path / "kernel32.aexrt",
        classes=252,
        max_detections=4,
        precision="fp32",
    )

    assert optimized["kernel_plan"][0][2:] == (47, 2, 1, 3)
    assert optimized["packed_layout_counts"] == {
        engine_module.PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP16: 1
    }
    owner = next(
        record
        for record in optimized["physical_dispatch_plan"]
        if record["logical_indices"] == (0,)
    )
    assert (owner["kernel"], owner["precision"], owner["flags"]) == (47, 2, 3)
    assert fallback["kernel_plan"][0][2] == 32


@pytest.mark.parametrize(
    ("in_channels", "out_channels"),
    ((1, 65536), (65536, 1)),
)
def test_layout6_inspector_rejects_channels_not_divisible_by_four(
    tmp_path, in_channels, out_channels
):
    path = tmp_path / "position_pair.aexrt"
    info = save_aexrt_engine(
        _build_exact_graph(),
        path,
        classes=252,
        max_detections=4,
    )
    data = path.read_bytes()
    sections = {
        section_type: data[section["offset"] : section["offset"] + section["size"]]
        for section_type, section in info["sections"].items()
    }
    packed = bytearray(sections[engine_module.SECTION_PACKED_WEIGHTS])
    struct.pack_into("<II", packed, 16, in_channels, out_channels)
    sections[engine_module.SECTION_PACKED_WEIGHTS] = bytes(packed)

    with pytest.raises(ValueError, match="invalid AEXRT packed-weight payload"):
        inspect_aexrt_engine(engine_module._build_container(sections))


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (0, 2),
        (1, 128),
        (2, 39),
        (4, 128),
        (5, 10),
        (7, 1),
        (9, 1),
        (11, 0),
        (13, 2),
        (15, 2),
    ),
)
def test_fp16_stride2_position_pair_plan_rejects_nearby_shapes(field, replacement):
    desc = list(EXACT_DESC)
    desc[field] = replacement
    assert engine_module._uses_native_fp16_position_pair_40x40_256x256(EXACT_DESC)
    assert not engine_module._uses_native_fp16_position_pair_40x40_256x256(desc)
