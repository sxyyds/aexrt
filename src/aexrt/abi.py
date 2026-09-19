from __future__ import annotations

from typing import Any, Dict, List, Tuple
import json

import numpy as np

from .graph import Graph, TensorSpec
from .execution import infer_value_specs


ABI_FORMAT = "aexrt.graph"
ABI_VERSION = 1

OP_TO_ABI = {
    "Relu": "Relu",
    "Gelu": "Gelu",
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
}

AEXRT_ABI_MAX_INPUTS = 16
UNARY_OPS = {"Relu", "Gelu", "Sigmoid", "Tanh"}
BINARY_OPS = {"Add", "Sub", "Mul", "Div"}


def export_graph_abi(graph: Graph) -> Dict[str, Any]:
    """Export the portable AEXRT Graph ABI v1.

    v1 is intentionally small and maps directly to the native C++ node-list
    compiler: float32 elementwise graphs with one or two runtime inputs.
    """

    if not graph.outputs or len(graph.outputs) != 1:
        raise NotImplementedError("AEXRT ABI v1 requires exactly one graph output")
    if not (1 <= len(graph.inputs) <= AEXRT_ABI_MAX_INPUTS):
        raise NotImplementedError(f"AEXRT ABI v1 supports 1..{AEXRT_ABI_MAX_INPUTS} runtime inputs")

    specs = infer_value_specs(graph)
    input_names = list(graph.inputs)
    first_spec = _require_spec(specs, input_names[0])
    if first_spec.dtype != "float32":
        raise NotImplementedError("AEXRT ABI v1 supports float32 only")
    element_count = _numel(first_spec)

    value_ids: Dict[str, int] = {}
    values: List[Dict[str, Any]] = []
    constants: List[Dict[str, Any]] = []

    def intern(name: str) -> int:
        if name in value_ids:
            return value_ids[name]
        spec = _require_spec(specs, name)
        if spec.dtype != "float32":
            raise NotImplementedError(f"AEXRT ABI v1 supports float32 only: {name}={spec.dtype}")
        is_scalar_constant = name in graph.constants and _numel(spec) == 1
        if _numel(spec) != element_count and not is_scalar_constant:
            raise NotImplementedError("AEXRT ABI v1 requires all values to have the same element count")
        value_id = len(values)
        value_ids[name] = value_id
        values.append({"id": value_id, "name": name, "dtype": spec.dtype, "shape": list(spec.shape)})
        if is_scalar_constant:
            arr = np.asarray(graph.constants[name], dtype=np.float32).reshape(-1)
            constants.append({"name": name, "value": value_id, "scalar": float(arr[0])})
        return value_id

    for name in input_names:
        intern(name)

    nodes = []
    for node in graph.nodes:
        if node.op not in OP_TO_ABI:
            raise NotImplementedError(f"AEXRT ABI v1 does not support op {node.op}")
        if len(node.outputs) != 1:
            raise NotImplementedError("AEXRT ABI v1 requires single-output nodes")
        if node.op in UNARY_OPS and len(node.inputs) != 1:
            raise NotImplementedError(f"{node.op} requires one input")
        if node.op in BINARY_OPS and len(node.inputs) != 2:
            raise NotImplementedError(f"{node.op} requires two inputs")
        input0 = intern(node.inputs[0])
        input1 = intern(node.inputs[1]) if len(node.inputs) > 1 else 0
        output = intern(node.outputs[0])
        nodes.append({
            "op": OP_TO_ABI[node.op],
            "input0": input0,
            "input1": input1,
            "output": output,
        })

    output_name = graph.outputs[0]
    output_id = intern(output_name)
    return {
        "format": ABI_FORMAT,
        "version": ABI_VERSION,
        "name": graph.name,
        "element_count": element_count,
        "inputs": [{"name": name, "value": value_ids[name]} for name in input_names],
        "output": {"name": output_name, "value": output_id},
        "values": values,
        "constants": constants,
        "nodes": nodes,
    }


def save_graph_abi(graph: Graph, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_graph_abi(graph), f, ensure_ascii=False, indent=2)


def load_graph_abi(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != ABI_FORMAT or int(data.get("version", 0)) != ABI_VERSION:
        raise ValueError("not an AEXRT Graph ABI v1 file")
    return data


def _require_spec(specs: Dict[str, TensorSpec], name: str) -> TensorSpec:
    spec = specs.get(name)
    if spec is None:
        raise NotImplementedError(f"missing shape/dtype for value {name}")
    return spec


def _numel(spec: TensorSpec) -> int:
    total = 1
    for dim in spec.shape:
        if dim < 0:
            raise NotImplementedError("AEXRT ABI v1 requires static shapes")
        total *= int(dim)
    return total
