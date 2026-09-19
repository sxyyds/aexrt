import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper

from aexrt.graph import Graph
from aexrt.onnx_importer import load_onnx


def _save_model(tmp_path, name, nodes, inputs, outputs, initializers=(), opset=13):
    graph = helper.make_graph(
        nodes,
        name,
        inputs,
        outputs,
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    return path


def _constant_node(name, value):
    tensor = numpy_helper.from_array(np.asarray(value), name=f"{name}_value")
    return helper.make_node("Constant", [], [name], value=tensor)


def test_load_onnx_imports_constant_tensor_value(tmp_path):
    value = np.asarray([[1.5, -2.0], [3.25, 4.5]], dtype=np.float16)
    path = _save_model(
        tmp_path,
        "constant_tensor",
        [_constant_node("output", value)],
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [2, 2])],
    )

    graph = load_onnx(str(path))

    assert graph.nodes == []
    assert graph.constants["output"].dtype == np.float16
    np.testing.assert_array_equal(graph.constants["output"], value)


@pytest.mark.parametrize(
    ("op", "left", "right", "expected"),
    [
        ("Add", [1, 2], [3, 4], [4, 6]),
        ("Sub", [5, 3], [2, 4], [3, -1]),
        ("Mul", [2, -3], [4, 5], [8, -15]),
        ("Div", [-5, 5], [2, 2], [-2, 2]),
        ("Pow", [2, 3], [3, 2], [8, 9]),
    ],
)
def test_load_onnx_folds_binary_constant_nodes(tmp_path, op, left, right, expected):
    initializers = [
        numpy_helper.from_array(np.asarray(left, dtype=np.int64), name="left"),
        numpy_helper.from_array(np.asarray(right, dtype=np.int64), name="right"),
    ]
    path = _save_model(
        tmp_path,
        f"fold_{op.lower()}",
        [helper.make_node(op, ["left", "right"], ["output"])],
        [],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [2])],
        initializers,
    )

    graph = load_onnx(str(path))

    assert graph.nodes == []
    np.testing.assert_array_equal(
        graph.constants["output"], np.asarray(expected, dtype=np.int64)
    )


def test_load_onnx_resolves_folded_slice_bounds(tmp_path):
    nodes = [
        _constant_node("start_base", np.asarray([0, 0], dtype=np.int64)),
        _constant_node("start_delta", np.asarray([1, 2], dtype=np.int64)),
        helper.make_node("Add", ["start_base", "start_delta"], ["starts"]),
        _constant_node("end_base", np.asarray([10, 12], dtype=np.int64)),
        helper.make_node("Div", ["end_base", "end_divisor"], ["ends"]),
        _constant_node("axes", np.asarray([2, 3], dtype=np.int64)),
        _constant_node("step_base", np.asarray([1, 1], dtype=np.int64)),
        helper.make_node("Mul", ["step_base", "step_scale"], ["steps"]),
        helper.make_node("Slice", ["images", "starts", "ends", "axes", "steps"], ["output"]),
    ]
    initializers = [
        numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64), name="end_divisor"),
        numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64), name="step_scale"),
    ]
    path = _save_model(
        tmp_path,
        "folded_slice_bounds",
        nodes,
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 1, 6, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
        initializers,
    )

    graph = load_onnx(str(path))

    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.op == "Slice"
    assert node.inputs == ["images"]
    assert node.attrs["starts"] == [1, 2]
    assert node.attrs["ends"] == [5, 6]
    assert node.attrs["axes"] == [2, 3]
    assert node.attrs["steps"] == [2, 2]


def test_load_onnx_preserves_data_dependent_pow(tmp_path):
    exponent = numpy_helper.from_array(np.asarray([2.0], dtype=np.float32), name="exponent")
    path = _save_model(
        tmp_path,
        "dynamic_pow",
        [helper.make_node("Pow", ["images", "exponent"], ["output"])],
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [exponent],
    )

    graph = load_onnx(str(path))

    assert len(graph.nodes) == 1
    assert graph.nodes[0].op == "Pow"
    assert graph.nodes[0].inputs == ["images", "exponent"]
    assert graph.nodes[0].outputs == ["output"]


def test_load_onnx_preserves_metadata_through_graph_json(tmp_path):
    path = _save_model(
        tmp_path,
        "metadata",
        [helper.make_node("Identity", ["images"], ["output"])],
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
    )
    model = onnx.load(path)
    helper.set_model_props(model, {"names": "{0: 'a', 1: 'b'}", "stride": "32"})
    onnx.save(model, path)

    graph = load_onnx(str(path))

    assert graph.metadata == {"names": "{0: 'a', 1: 'b'}", "stride": "32"}
    assert Graph.from_json(graph.to_json()).metadata == graph.metadata
