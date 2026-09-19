from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import json
import numpy as np

Shape = Tuple[int, ...]


@dataclass(frozen=True)
class TensorSpec:
    shape: Shape
    dtype: str = "float32"
    layout: str = "contiguous"

    def to_json(self) -> Dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "layout": self.layout}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "TensorSpec":
        return TensorSpec(tuple(int(x) for x in data["shape"]), data.get("dtype", "float32"), data.get("layout", "contiguous"))


@dataclass
class Node:
    op: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"op": self.op, "name": self.name, "inputs": self.inputs, "outputs": self.outputs, "attrs": self.attrs}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Node":
        return Node(data["op"], data["name"], list(data["inputs"]), list(data["outputs"]), dict(data.get("attrs", {})))


class Graph:
    """Small backend-neutral graph IR for inference.

    Values are named tensors. Initializers/constants live in `constants` and are
    uploaded once by a backend during session creation.
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.metadata: Dict[str, str] = {}
        self.inputs: Dict[str, TensorSpec] = {}
        self.outputs: List[str] = []
        self.constants: Dict[str, np.ndarray] = {}
        self.nodes: List[Node] = []
        self.value_specs: Dict[str, TensorSpec] = {}

    def clone(self) -> "Graph":
        return copy.deepcopy(self)

    def input(self, name: str, spec: TensorSpec) -> "Graph":
        self.inputs[name] = spec
        self.value_specs[name] = spec
        return self

    def const(self, name: str, value: Any, dtype: Optional[str] = None) -> "Graph":
        arr = np.asarray(value, dtype=dtype)
        self.constants[name] = arr
        self.value_specs[name] = TensorSpec(tuple(arr.shape), str(arr.dtype))
        return self

    def output(self, name: str) -> "Graph":
        if name not in self.outputs:
            self.outputs.append(name)
        return self

    def node(self, op: str, out: str | Sequence[str], *inputs: str, name: Optional[str] = None, **attrs: Any) -> "Graph":
        outputs = [out] if isinstance(out, str) else list(out)
        n = Node(op=op, name=name or f"{op}_{len(self.nodes)}", inputs=list(inputs), outputs=outputs, attrs=dict(attrs))
        self.nodes.append(n)
        return self

    # Builder helpers
    def identity(self, out: str, x: str) -> "Graph": return self.node("Identity", out, x)
    def add(self, out: str, a: str, b: str) -> "Graph": return self.node("Add", out, a, b)
    def sub(self, out: str, a: str, b: str) -> "Graph": return self.node("Sub", out, a, b)
    def mul(self, out: str, a: str, b: str) -> "Graph": return self.node("Mul", out, a, b)
    def div(self, out: str, a: str, b: str) -> "Graph": return self.node("Div", out, a, b)
    def matmul(self, out: str, a: str, b: str) -> "Graph": return self.node("MatMul", out, a, b)
    def conv2d(
        self,
        out: str,
        x: str,
        weight: str,
        bias: Optional[str] = None,
        strides: Sequence[int] = (1, 1),
        pads: Sequence[int] = (0, 0, 0, 0),
        dilations: Sequence[int] = (1, 1),
        group: int = 1,
    ) -> "Graph":
        inputs = [x, weight] + ([bias] if bias is not None else [])
        return self.node("Conv", out, *inputs, strides=list(strides), pads=list(pads), dilations=list(dilations), group=int(group))
    def batchnorm(
        self,
        out: str,
        x: str,
        scale: str,
        bias: str,
        mean: str,
        var: str,
        epsilon: float = 1e-5,
    ) -> "Graph":
        return self.node("BatchNormalization", out, x, scale, bias, mean, var, epsilon=float(epsilon))
    def relu(self, out: str, x: str) -> "Graph": return self.node("Relu", out, x)
    def gelu(self, out: str, x: str, approximate: str = "tanh") -> "Graph": return self.node("Gelu", out, x, approximate=approximate)
    def sigmoid(self, out: str, x: str) -> "Graph": return self.node("Sigmoid", out, x)
    def tanh(self, out: str, x: str) -> "Graph": return self.node("Tanh", out, x)
    def softmax(self, out: str, x: str, axis: int = -1) -> "Graph": return self.node("Softmax", out, x, axis=axis)
    def reshape(self, out: str, x: str, shape: Sequence[int]) -> "Graph": return self.node("Reshape", out, x, shape=list(shape))
    def transpose(self, out: str, x: str, axes: Sequence[int]) -> "Graph": return self.node("Transpose", out, x, axes=list(axes))
    def concat(self, out: str, *xs: str, axis: int = -1) -> "Graph": return self.node("Concat", out, *xs, axis=axis)
    def layernorm(self, out: str, x: str, weight: str, bias: Optional[str] = None, eps: float = 1e-5) -> "Graph":
        inputs = [x, weight] + ([bias] if bias is not None else [])
        return self.node("LayerNorm", out, *inputs, eps=eps)
    def embedding(self, out: str, ids: str, table: str) -> "Graph": return self.node("Embedding", out, ids, table)
    def sdpa(self, out: str, q: str, k: str, v: str, mask: Optional[str] = None, scale: Optional[float] = None, causal: bool = False) -> "Graph":
        inputs = [q, k, v] + ([mask] if mask is not None else [])
        return self.node("SDPA", out, *inputs, scale=scale, causal=causal)
    def rope(self, out: str, x: str, cos: str, sin: str, interleaved: bool = False) -> "Graph":
        return self.node("RoPE", out, x, cos, sin, interleaved=interleaved)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "inputs": {k: v.to_json() for k, v in self.inputs.items()},
            "outputs": self.outputs,
            "constants": {k: v.tolist() for k, v in self.constants.items()},
            "constant_dtypes": {k: str(v.dtype) for k, v in self.constants.items()},
            "nodes": [n.to_json() for n in self.nodes],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    def to_aexrt_abi(self) -> Dict[str, Any]:
        from .abi import export_graph_abi
        return export_graph_abi(self)

    def save_aexrt(self, path: str) -> None:
        from .abi import save_graph_abi
        save_graph_abi(self, path)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Graph":
        g = Graph(data.get("name", "graph"))
        g.metadata = {str(k): str(v) for k, v in data.get("metadata", {}).items()}
        for name, spec in data.get("inputs", {}).items():
            g.input(name, TensorSpec.from_json(spec))
        dtypes = data.get("constant_dtypes", {})
        for name, value in data.get("constants", {}).items():
            g.const(name, value, dtype=dtypes.get(name))
        for n in data.get("nodes", []):
            g.nodes.append(Node.from_json(n))
        for out in data.get("outputs", []):
            g.output(out)
        return g

    @staticmethod
    def load_json(path: str) -> "Graph":
        with open(path, "r", encoding="utf-8") as f:
            return Graph.from_json(json.load(f))
