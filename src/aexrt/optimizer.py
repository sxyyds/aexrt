from __future__ import annotations

from typing import Dict, List, Set
import numpy as np

from .graph import Graph, Node

PURE_NUMPY = {"Add", "Sub", "Mul", "Div", "MatMul", "BatchNormalization", "Relu", "Gelu", "Sigmoid", "Tanh", "Softmax", "Reshape", "Transpose", "Concat", "LayerNorm", "Identity"}


def _np_gelu(x: np.ndarray, approximate: str = "tanh") -> np.ndarray:
    if approximate == "none":
        # NumPy has no erf on minimal installs; vectorize math.erf to stay dependency-light.
        import math
        return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def _eval_const(node: Node, vals: Dict[str, np.ndarray]) -> np.ndarray:
    xs = [vals[i] for i in node.inputs]
    op = node.op
    if op == "Identity": return xs[0]
    if op == "Add": return xs[0] + xs[1]
    if op == "Sub": return xs[0] - xs[1]
    if op == "Mul": return xs[0] * xs[1]
    if op == "Div": return xs[0] / xs[1]
    if op == "MatMul": return xs[0] @ xs[1]
    if op == "BatchNormalization":
        x, scale, bias, mean, var = xs[:5]
        eps = float(node.attrs.get("epsilon", 1e-5))
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps) * scale.reshape(shape) + bias.reshape(shape)
    if op == "Relu": return np.maximum(xs[0], 0)
    if op == "Gelu": return _np_gelu(xs[0], node.attrs.get("approximate", "tanh"))
    if op == "Sigmoid": return 1.0 / (1.0 + np.exp(-xs[0]))
    if op == "Tanh": return np.tanh(xs[0])
    if op == "Softmax":
        axis = int(node.attrs.get("axis", -1))
        z = xs[0] - np.max(xs[0], axis=axis, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=axis, keepdims=True)
    if op == "Reshape": return np.reshape(xs[0], tuple(node.attrs["shape"]))
    if op == "Transpose": return np.transpose(xs[0], tuple(node.attrs["axes"]))
    if op == "Concat": return np.concatenate(xs, axis=int(node.attrs.get("axis", -1)))
    if op == "LayerNorm":
        x, weight = xs[0], xs[1]
        bias = xs[2] if len(xs) > 2 else 0
        eps = float(node.attrs.get("eps", 1e-5))
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias
    raise NotImplementedError(op)


def _consumer_count(nodes: List[Node]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for n in nodes:
        for i in n.inputs:
            counts[i] = counts.get(i, 0) + 1
    return counts


def _replace_input(nodes: List[Node], old: str, new: str) -> None:
    for n in nodes:
        n.inputs = [new if x == old else x for x in n.inputs]


def optimize_graph(graph: Graph, *, level: int = 2) -> Graph:
    """Return optimized graph with conservative semantics-preserving passes."""
    g = graph.clone()

    # Pass 1: constant folding. Keep graph outputs visible even if constant.
    new_nodes: List[Node] = []
    for n in g.nodes:
        if n.op in PURE_NUMPY and all(i in g.constants for i in n.inputs):
            try:
                g.constants[n.outputs[0]] = _eval_const(n, g.constants)
                continue
            except Exception:
                pass
        new_nodes.append(n)
    g.nodes = new_nodes

    # Pass 2: identity elimination.
    new_nodes = []
    for n in g.nodes:
        if n.op == "Identity" and n.outputs[0] not in g.outputs:
            _replace_input(g.nodes, n.outputs[0], n.inputs[0])
            continue
        new_nodes.append(n)
    g.nodes = new_nodes

    if level < 2:
        return g

    # Pass 3: fuse MatMul + optional Add + optional activation.
    counts = _consumer_count(g.nodes)
    out_to_node = {n.outputs[0]: n for n in g.nodes if len(n.outputs) == 1}
    skip: Set[int] = set()
    fused: List[Node] = []
    act_ops = {"Relu", "Gelu", "Sigmoid", "Tanh"}

    i = 0
    while i < len(g.nodes):
        if i in skip:
            i += 1
            continue
        n = g.nodes[i]
        if n.op != "MatMul" or len(n.outputs) != 1 or n.outputs[0] in g.outputs:
            fused.append(n)
            i += 1
            continue

        current_out = n.outputs[0]
        bias = None
        final_out = current_out
        activation = None
        attrs = {}
        consumed = []

        # MatMul -> Add bias
        if counts.get(current_out, 0) == 1:
            add = next((m for m in g.nodes if m.op == "Add" and current_out in m.inputs and len(m.outputs) == 1), None)
            if add is not None:
                bias = add.inputs[1] if add.inputs[0] == current_out else add.inputs[0]
                final_out = add.outputs[0]
                consumed.append(id(add))

        # optional activation after current final_out
        if counts.get(final_out, 0) == 1:
            act = next((m for m in g.nodes if m.op in act_ops and m.inputs == [final_out] and len(m.outputs) == 1), None)
            if act is not None:
                activation = act.op
                attrs.update(act.attrs)
                final_out = act.outputs[0]
                consumed.append(id(act))

        if bias is not None or activation is not None:
            inputs = list(n.inputs) + ([bias] if bias is not None else [])
            attrs["activation"] = activation
            attrs["has_bias"] = bias is not None
            fused.append(Node("FusedLinear", f"FusedLinear_{len(fused)}", inputs, [final_out], attrs))
            # Mark consumed by object identity.
            for j, m in enumerate(g.nodes):
                if id(m) in consumed:
                    skip.add(j)
        else:
            fused.append(n)
        i += 1

    g.nodes = fused
    return g
