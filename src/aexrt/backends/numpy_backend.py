from __future__ import annotations

from typing import Any, Dict
import math
import numpy as np

from .base import Backend, BackendInfo
from ..graph import Graph, Node
from .. import _numpy_ops


def _gelu(x: np.ndarray, approximate: str = "tanh") -> np.ndarray:
    if approximate == "none":
        return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, interleaved: bool = False) -> np.ndarray:
    # x: [..., dim], cos/sin broadcastable to [..., dim/2] or [..., dim]
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = cos[..., :x1.shape[-1]]
        s = sin[..., :x1.shape[-1]]
        y0 = x1 * c - x2 * s
        y1 = x1 * s + x2 * c
        y = np.empty_like(x)
        y[..., 0::2] = y0
        y[..., 1::2] = y1
        return y
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[..., :half]
    s = sin[..., :half]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


class NumpyBackend(Backend):
    name = "numpy"
    # 基础算子 + 共享 numpy 求值表覆盖的扩展算子（LeakyRelu/池化/布局/Reduce 等）。
    supported_ops = frozenset({
        "Add", "Sub", "Mul", "Div", "MatMul", "FusedLinear",
        "BatchNormalization",
        "Relu", "Gelu", "Sigmoid", "Tanh", "Softmax", "LayerNorm",
        "Reshape", "Transpose", "Concat", "Embedding", "SDPA", "RoPE", "Identity",
    }) | _numpy_ops.NUMPY_EVAL_OPS
    supported_dtypes = frozenset({"float32", "float64", "int32", "int64", "bool"})
    supported_features = frozenset({"persistent_constants", "static_execution_plan"})

    def __init__(self) -> None:
        self.graph: Graph | None = None
        self.constants: Dict[str, np.ndarray] = {}
        self.execution_plan = None

    def info(self) -> BackendInfo:
        return BackendInfo("numpy", "cpu", {
            "fp32": True,
            "gpu": False,
            "device_type": "cpu",
            "device_name": "CPU",
            "ops": sorted(self.supported_ops),
            "dtypes": sorted(self.supported_dtypes),
            "features": sorted(self.supported_features),
        })

    def prepare(self, graph: Graph) -> None:
        self.graph = graph
        self.execution_plan = self.compile_plan(graph)
        self.constants = {k: np.asarray(v) for k, v in graph.constants.items()}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        assert self.graph is not None, "backend not prepared"
        vals: Dict[str, np.ndarray] = dict(self.constants)
        vals.update({k: np.asarray(v) for k, v in inputs.items()})
        for n in self.graph.nodes:
            vals[n.outputs[0]] = self._eval(n, vals)
        return {k: vals[k] for k in self.graph.outputs}

    def _eval(self, n: Node, vals: Dict[str, np.ndarray]) -> np.ndarray:
        xs = [vals[i] for i in n.inputs]
        op = n.op
        if op == "Identity": return xs[0]
        if op == "Add": return xs[0] + xs[1]
        if op == "Sub": return xs[0] - xs[1]
        if op == "Mul": return xs[0] * xs[1]
        if op == "Div": return xs[0] / xs[1]
        if op == "MatMul": return xs[0] @ xs[1]
        if op == "FusedLinear":
            y = xs[0] @ xs[1]
            if n.attrs.get("has_bias", len(xs) > 2): y = y + xs[2]
            act = n.attrs.get("activation")
            if act == "Relu": y = np.maximum(y, 0)
            elif act == "Gelu": y = _gelu(y, n.attrs.get("approximate", "tanh"))
            elif act == "Sigmoid": y = 1.0 / (1.0 + np.exp(-y))
            elif act == "Tanh": y = np.tanh(y)
            return y
        if op == "BatchNormalization":
            x, scale, bias, mean, var = xs[:5]
            eps = float(n.attrs.get("epsilon", 1e-5))
            shape = (1, -1) + (1,) * (x.ndim - 2)
            return (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps) * scale.reshape(shape) + bias.reshape(shape)
        if op == "Relu": return np.maximum(xs[0], 0)
        if op == "Gelu": return _gelu(xs[0], n.attrs.get("approximate", "tanh"))
        if op == "Sigmoid": return 1.0 / (1.0 + np.exp(-xs[0]))
        if op == "Tanh": return np.tanh(xs[0])
        if op == "Softmax":
            axis = int(n.attrs.get("axis", -1))
            z = xs[0] - np.max(xs[0], axis=axis, keepdims=True)
            e = np.exp(z)
            return e / np.sum(e, axis=axis, keepdims=True)
        if op == "LayerNorm":
            x, w = xs[0], xs[1]
            b = xs[2] if len(xs) > 2 else 0
            eps = float(n.attrs.get("eps", 1e-5))
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + eps) * w + b
        if op == "Reshape": return np.reshape(xs[0], tuple(n.attrs["shape"]))
        if op == "Transpose": return np.transpose(xs[0], tuple(n.attrs["axes"]))
        if op == "Concat": return np.concatenate(xs, axis=int(n.attrs.get("axis", -1)))
        if op == "Embedding": return xs[1][xs[0].astype(np.int64)]
        if op == "SDPA":
            q, k, v = xs[:3]
            scale = n.attrs.get("scale") or (1.0 / math.sqrt(q.shape[-1]))
            scores = q @ np.swapaxes(k, -1, -2) * scale
            if len(xs) > 3: scores = scores + xs[3]
            if n.attrs.get("causal", False):
                L, S = q.shape[-2], k.shape[-2]
                mask = np.triu(np.ones((L, S), dtype=bool), k=1)
                scores = np.where(mask, -np.inf, scores)
            z = scores - np.max(scores, axis=-1, keepdims=True)
            p = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
            return p @ v
        if op == "RoPE": return _rope(xs[0], xs[1], xs[2], bool(n.attrs.get("interleaved", False)))
        # 其余算子（LeakyRelu/池化/布局/Cast/Reduce/一元等）路由到共享 numpy 求值表。
        try:
            return _numpy_ops.eval_node(op, dict(n.attrs), xs)
        except NotImplementedError:
            raise NotImplementedError(f"NumPy backend does not support op {op}") from None
