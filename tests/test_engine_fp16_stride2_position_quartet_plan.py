import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import aexrt.engine as engine_module
from aexrt import Graph, TensorSpec, save_aexrt_engine


EXACT_DESC = (
    1,
    128,
    80,
    80,
    128,
    40,
    40,
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
    graph = Graph("native_fp16_stride2_position_quartet_80x80_128x128")
    graph.input("images", TensorSpec((1, 128, 80, 80), "float32"))
    graph.const("weight", np.zeros((128, 128, 3, 3), dtype="float32"))
    graph.const("bias", np.zeros((128,), dtype="float32"))
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
    graph.reshape("output0", "activated", (1, 128, 1600))
    graph.output("output0")
    return graph


def test_fp16_v11_cmd9_keeps_stable_kernel33_and_layout2(tmp_path):
    info = save_aexrt_engine(
        _build_exact_graph(),
        tmp_path / "stable_kernel33.aexrt",
        classes=124,
        max_detections=4,
    )

    assert info["kernel_plan"][0][2:] == (33, 2, 1, 1)
    assert info["packed_layout_counts"] == {
        engine_module.PACKED_LAYOUT_CONV3X3_OC4_FP16: 1
    }
    assert engine_module.PACKED_LAYOUT_CONV3X3_K_MAJOR_OC4_FP32 not in info[
        "packed_layout_counts"
    ]
    owner = next(
        record
        for record in info["physical_dispatch_plan"]
        if record["logical_indices"] == (0,)
    )
    assert (owner["kernel"], owner["precision"], owner["flags"]) == (33, 2, 1)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (0, 2),
        (1, 64),
        (2, 79),
        (4, 64),
        (5, 20),
        (7, 1),
        (9, 1),
        (11, 0),
        (13, 2),
        (15, 2),
    ),
)
def test_rejected_stride2_candidates_are_not_generated(field, replacement):
    desc = list(EXACT_DESC)
    desc[field] = replacement
    replay = {
        "commands": [
            {
                "kind": "CONV_SILU",
                "kind_id": 1,
                "inputs": [0, 1, 2],
                "params": desc,
            }
        ]
    }

    plan = engine_module._build_kernel_plan(
        replay, precision=engine_module.PRECISION_FLOAT16
    )[0]

    assert plan["planned_kernel"] not in {48, 49, 50}
