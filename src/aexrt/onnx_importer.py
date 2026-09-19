from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np

from .graph import Graph, TensorSpec

_ONNX_TO_NP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    16: "bfloat16",
}

# 视为 fp32 路径可剥离的量化辅助算子（ai.onnx / com.microsoft 域）。
_QUANT_PASSTHROUGH_OPS = {
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
}


def _attr_value(attr: Any) -> Any:
    import onnx
    if attr.type == onnx.AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == onnx.AttributeProto.INT:
        return int(attr.i)
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if attr.type == onnx.AttributeProto.FLOATS:
        return [float(x) for x in attr.floats]
    if attr.type == onnx.AttributeProto.INTS:
        return [int(x) for x in attr.ints]
    if attr.type == onnx.AttributeProto.STRINGS:
        return [x.decode("utf-8") for x in attr.strings]
    if attr.type == onnx.AttributeProto.TENSOR:
        from onnx import numpy_helper

        return numpy_helper.to_array(attr.t)
    return None


def _attrs(node: Any) -> Dict[str, Any]:
    return {a.name: _attr_value(a) for a in node.attribute}


def _shape_from_vi(vi: Any) -> tuple[int, ...]:
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append(-1)
    return tuple(dims)


def _as_float(arr: np.ndarray) -> np.ndarray:
    """int8/uint8 张量按数值语义还原为 float。"""
    return arr.astype(np.float32)


def _dequant_const(arr: np.ndarray, scale: np.ndarray, zero_point: np.ndarray, axis: int = -1) -> np.ndarray:
    """常量反量化：支持 per-tensor 与 per-channel axis。

    语义: fp32 = (q - zero_point) * scale（scale/zp 为常量输入）。
    """
    q = _as_float(arr)
    s = np.asarray(scale, dtype=np.float32).reshape(-1)
    z = np.asarray(zero_point).reshape(-1)
    if s.size == 1:
        return (q - float(z.reshape(-1)[0])) * float(s.reshape(-1)[0])
    if axis < 0:
        axis = arr.ndim + axis
    shape = [1] * arr.ndim
    shape[axis] = s.size
    s = s.reshape(shape).astype(np.float32)
    z = _as_float(z).reshape(shape)
    return (q - z) * s


def load_onnx(path: str, default_batch: Optional[int] = 1) -> Graph:
    """Load a practical subset of ONNX into AEXRT IR.

    Supported common inference ops: Constant, Conv, BatchNormalization, MaxPool, Resize,
    Split, Slice, Add/Sub/Mul/Div/Pow, MatMul, Relu, Gelu, Sigmoid, Tanh, Softmax,
    Reshape, Transpose, Concat, LayerNormalization, Identity, plus LeakyRelu/PRelu/Elu/
    Mish/HardSigmoid, AveragePool/GlobalAveragePool/GlobalMaxPool, Flatten, Squeeze/
    Unsqueeze, Cast, Gather, Pad, Shape, Clip, 常见一元算子、Min/Max/Mod、Reduce*。

    量化模型处理（QDQ 剥离）：``QuantizeLinear``/``DequantizeLinear`` 会被透传或常量折叠，
    ``QLinearConv``/``ConvInteger`` 的常量权重被反量化后按 fp32 Conv 导入，
    ``MatMulInteger`` 常量对折叠为 fp32 常量。
    动态 batch：输入第 0 维为动态(-1)时替换为 ``default_batch``（None 表示保留 -1）。

    未支持的算子会在扫描全图后一次性汇总报告，而不是在第一个节点处中断。
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    og = model.graph
    g = Graph(og.name or "onnx_model")
    g.metadata = {str(prop.key): str(prop.value) for prop in model.metadata_props}

    initializers = {t.name: numpy_helper.to_array(t) for t in og.initializer}
    for name, arr in initializers.items():
        g.const(name, arr)

    init_names = set(initializers)
    int_input_names = {
        inp.name
        for inp in og.input
        if inp.name not in init_names
        and _ONNX_TO_NP.get(inp.type.tensor_type.elem_type, "float32") in {"int8", "uint8"}
    }
    for inp in og.input:
        if inp.name in init_names:
            continue
        elem = inp.type.tensor_type.elem_type
        dims = _shape_from_vi(inp)
        if default_batch is not None and len(dims) >= 1 and dims[0] == -1:
            dims = (int(default_batch),) + dims[1:]
        g.input(inp.name, TensorSpec(dims, _ONNX_TO_NP.get(elem, "float32")))

    quant_stripped: set[str] = set()
    unsupported: list[str] = []

    def _fold_const_binary(op: str, out: str, ins: Sequence[str]) -> None:
        left = np.asarray(g.constants[ins[0]])
        right = np.asarray(g.constants[ins[1]])
        if op == "Add":
            value = left + right
        elif op == "Sub":
            value = left - right
        elif op == "Mul":
            value = left * right
        elif op == "Div" and np.issubdtype(left.dtype, np.integer):
            value = np.trunc(left / right).astype(left.dtype)
        elif op == "Div":
            value = left / right
        elif op == "MatMul":
            value = np.matmul(_as_float(left), _as_float(right))
        else:
            value = np.power(left, right)
        g.const(out, np.asarray(value))

    for node in og.node:
        op = node.op_type
        domain = node.domain or ""
        ins = list(node.input)
        outs = list(node.output)
        a = _attrs(node)
        out = outs[0]
        name = node.name or f"{op}_{len(g.nodes)}"
        fq = f"{op}({domain})" if domain else op

        # ---- 量化辅助算子：QDQ 剥离为 fp32 路径 ----
        if op in _QUANT_PASSTHROUGH_OPS and domain in ("", "ai.onnx", "com.microsoft"):
            src = ins[0]
            if op == "DequantizeLinear" and src in g.constants and len(ins) >= 3:
                axis = int(a.get("axis", -1))
                try:
                    value = _dequant_const(
                        g.constants[src], g.constants.get(ins[1], np.asarray(1.0, np.float32)),
                        g.constants.get(ins[2], np.asarray(0)), axis,
                    )
                    g.const(out, value)
                    quant_stripped.add(fq)
                    continue
                except Exception:
                    pass
            # 激活 Q/DQ：透传（Identity 由 optimizer 消除）。
            g.identity(out, src)
            quant_stripped.add(fq)
            continue
        if op in {"QLinearConv", "ConvInteger"} and domain in ("", "ai.onnx", "com.microsoft"):
            if op == "QLinearConv":
                x_name, w_name = ins[0], ins[1]
                x_scale, x_zp, w_scale, w_zp = ins[2], ins[3], ins[4], ins[5]
                bias = ins[8] if len(ins) > 8 and ins[8] else None
            else:  # ConvInteger: x, w, x_zp, w_zp
                x_name, w_name = ins[0], ins[1]
                x_zp, w_zp = ins[2], ins[3]
                x_scale = w_scale = None
                bias = None
            if w_name not in g.constants:
                unsupported.append(f"{fq}: non-constant quantized weights")
                continue
            w_dq = _dequant_const(
                g.constants[w_name],
                g.constants.get(w_scale, np.asarray(1.0, np.float32)) if w_scale else np.asarray(1.0, np.float32),
                g.constants.get(w_zp, np.asarray(0)) if w_zp else np.asarray(0),
                axis=0,
            )
            w_dq_name = f"{w_name}__dequant"
            g.const(w_dq_name, w_dq.astype(np.float32))
            conv_ins = [x_name, w_dq_name]
            if bias:
                conv_ins.append(bias)
            if x_name in g.constants and x_scale is not None:
                x_dq = _dequant_const(
                    g.constants[x_name], g.constants.get(x_scale, np.asarray(1.0, np.float32)),
                    g.constants.get(x_zp, np.asarray(0)), axis=0,
                )
                x_dq_name = f"{x_name}__dequant"
                g.const(x_dq_name, x_dq.astype(np.float32))
                conv_ins[0] = x_dq_name
            elif x_name in int_input_names:
                unsupported.append(f"{fq}: runtime int8 activations are not supported; re-export as QDQ or fp32")
                continue
            g.node(
                "Conv",
                out,
                *conv_ins,
                name=name,
                strides=list(a.get("strides", [1, 1])),
                pads=list(a.get("pads", [0, 0, 0, 0])),
                dilations=list(a.get("dilations", [1, 1])),
                group=int(a.get("group", 1)),
            )
            quant_stripped.add(fq)
            continue
        if op == "MatMulInteger" and domain in ("", "ai.onnx"):
            a_name, b_name = ins[0], ins[1]
            a_zp = g.constants.get(ins[2], np.asarray(0)) if len(ins) > 2 else np.asarray(0)
            b_zp = g.constants.get(ins[3], np.asarray(0)) if len(ins) > 3 else np.asarray(0)
            if a_name in g.constants and b_name in g.constants:
                a_dq = _dequant_const(g.constants[a_name], np.asarray(1.0, np.float32), a_zp)
                b_dq = _dequant_const(g.constants[b_name], np.asarray(1.0, np.float32), b_zp)
                g.const(out, np.matmul(a_dq, b_dq).astype(np.float32))
                quant_stripped.add(fq)
                continue
            if a_name in int_input_names or b_name in int_input_names:
                unsupported.append(f"{fq}: runtime int8 activations are not supported")
                continue
            g.node("MatMul", out, a_name, b_name, name=name)
            quant_stripped.add(fq)
            continue

        # ---- 常量二元折叠（MatMul 折叠曾改变 YOLO 图结构导致引擎物理计划失配，暂不折叠） ----
        if op in {"Add", "Sub", "Mul", "Div", "Pow"} and all(inp in g.constants for inp in ins):
            _fold_const_binary(op, out, ins)
        elif op in {"Add", "Sub", "Mul", "Div", "Pow", "MatMul", "Relu", "Sigmoid", "Tanh", "Identity"}:
            g.node(op, out, *ins, name=name)
        elif op == "LeakyRelu":
            g.node("LeakyRelu", out, ins[0], name=name, alpha=float(a.get("alpha", 0.01)))
        elif op == "PRelu":
            g.node("PRelu", out, *ins[:2], name=name)
        elif op == "Elu":
            g.node("Elu", out, ins[0], name=name, alpha=float(a.get("alpha", 1.0)))
        elif op == "Mish":
            g.node("Mish", out, ins[0], name=name)
        elif op == "HardSigmoid":
            g.node("HardSigmoid", out, ins[0], name=name,
                   alpha=float(a.get("alpha", 0.2)), beta=float(a.get("beta", 0.5)))
        elif op == "Clip":
            lo = g.constants.get(ins[1]) if len(ins) > 1 and ins[1] else None
            hi = g.constants.get(ins[2]) if len(ins) > 2 and ins[2] else None
            attrs_clip: Dict[str, Any] = {}
            if lo is not None:
                attrs_clip["min"] = float(np.asarray(lo).reshape(-1)[0])
            if hi is not None:
                attrs_clip["max"] = float(np.asarray(hi).reshape(-1)[0])
            if "min" in a:
                attrs_clip["min"] = float(a["min"])
            if "max" in a:
                attrs_clip["max"] = float(a["max"])
            g.node("Clip", out, ins[0], name=name, **attrs_clip)
        elif op in {"Exp", "Log", "Sqrt", "Neg", "Abs", "Floor", "Ceil", "Round", "Sin", "Cos", "Reciprocal", "Sign", "Erf"}:
            g.node(op, out, ins[0], name=name)
        elif op in {"Min", "Max"}:
            g.node(op, out, *ins, name=name)
        elif op == "Mod":
            g.node("Mod", out, *ins, name=name, fmod=int(a.get("fmod", 0)))
        elif op == "Conv":
            g.node(
                "Conv",
                out,
                *ins,
                name=name,
                strides=list(a.get("strides", [1, 1])),
                pads=list(a.get("pads", [0, 0, 0, 0])),
                dilations=list(a.get("dilations", [1, 1])),
                group=int(a.get("group", 1)),
            )
        elif op == "BatchNormalization":
            g.batchnorm(out, ins[0], ins[1], ins[2], ins[3], ins[4], epsilon=float(a.get("epsilon", 1e-5)))
        elif op == "MaxPool":
            g.node(
                "MaxPool",
                out,
                ins[0],
                name=name,
                kernel_shape=list(a["kernel_shape"]),
                strides=list(a.get("strides", a["kernel_shape"])),
                pads=list(a.get("pads", [0, 0, 0, 0])),
                dilations=list(a.get("dilations", [1, 1])),
                ceil_mode=int(a.get("ceil_mode", 0)),
            )
        elif op == "AveragePool":
            g.node(
                "AveragePool",
                out,
                ins[0],
                name=name,
                kernel_shape=list(a["kernel_shape"]),
                strides=list(a.get("strides", a["kernel_shape"])),
                pads=list(a.get("pads", [0, 0, 0, 0])),
                count_include_pad=int(a.get("count_include_pad", 1)),
            )
        elif op == "GlobalAveragePool":
            g.node("GlobalAveragePool", out, ins[0], name=name)
        elif op == "GlobalMaxPool":
            g.node("GlobalMaxPool", out, ins[0], name=name)
        elif op in {"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"}:
            axes = None
            if len(ins) > 1 and ins[1] in g.constants:
                axes = [int(v) for v in np.asarray(g.constants[ins[1]]).reshape(-1).tolist()]
            elif "axes" in a:
                axes = [int(v) for v in a["axes"]]
            attrs_reduce: Dict[str, Any] = {"keepdims": int(a.get("keepdims", 1))}
            if axes is not None:
                attrs_reduce["axes"] = axes
            g.node(op, out, ins[0], name=name, **attrs_reduce)
        elif op == "Flatten":
            g.node("Flatten", out, ins[0], name=name, axis=int(a.get("axis", 1)))
        elif op == "Squeeze":
            axes = None
            if len(ins) > 1 and ins[1] in g.constants:
                axes = [int(v) for v in np.asarray(g.constants[ins[1]]).reshape(-1).tolist()]
            elif "axes" in a:
                axes = [int(v) for v in a["axes"]]
            if axes is None:
                g.node("Squeeze", out, ins[0], name=name)
            else:
                g.node("Squeeze", out, ins[0], name=name, axes=axes)
        elif op == "Unsqueeze":
            axes = None
            if len(ins) > 1 and ins[1] in g.constants:
                axes = [int(v) for v in np.asarray(g.constants[ins[1]]).reshape(-1).tolist()]
            elif "axes" in a:
                axes = [int(v) for v in a["axes"]]
            if axes is None:
                unsupported.append(f"{fq}: Unsqueeze requires constant axes")
                continue
            g.node("Unsqueeze", out, ins[0], name=name, axes=axes)
        elif op == "Cast":
            g.node("Cast", out, ins[0], name=name, to=int(a.get("to", 1)))
        elif op == "Shape":
            g.node("Shape", out, ins[0], name=name)
        elif op == "Gather":
            if len(ins) < 2 or ins[1] not in g.constants:
                unsupported.append(f"{fq}: Gather requires constant indices")
                continue
            g.const(f"{out}__indices", np.asarray(g.constants[ins[1]]).astype(np.int64))
            g.node("Gather", out, ins[0], f"{out}__indices", name=name, axis=int(a.get("axis", 0)))
        elif op == "Pad":
            pads = None
            value = 0.0
            if len(ins) > 1 and ins[1] in g.constants:
                pads = [int(v) for v in np.asarray(g.constants[ins[1]]).reshape(-1).tolist()]
            elif "pads" in a:
                pads = [int(v) for v in a["pads"]]
            if len(ins) > 3 and ins[3] in g.constants:
                value = float(np.asarray(g.constants[ins[3]]).reshape(-1)[0])
            if pads is None:
                unsupported.append(f"{fq}: Pad requires constant pads")
                continue
            g.node(
                "Pad",
                out,
                ins[0],
                name=name,
                pads=pads,
                value=value,
                mode=str(a.get("mode", "constant")),
            )
        elif op == "Resize":
            mode = a.get("mode", "nearest")
            if isinstance(mode, bytes):
                mode = mode.decode("utf-8")
            nearest_mode = a.get("nearest_mode", "round_prefer_floor")
            if isinstance(nearest_mode, bytes):
                nearest_mode = nearest_mode.decode("utf-8")
            coord = a.get("coordinate_transformation_mode", "half_pixel")
            if isinstance(coord, bytes):
                coord = coord.decode("utf-8")
            attrs = {"mode": mode, "nearest_mode": nearest_mode, "coordinate_transformation_mode": coord}
            if len(ins) > 3 and ins[3] in g.constants:
                attrs["sizes"] = [int(x) for x in g.constants[ins[3]].tolist()]
            elif len(ins) > 2 and ins[2] in g.constants:
                attrs["scales"] = [float(x) for x in g.constants[ins[2]].tolist()]
            else:
                unsupported.append(f"{fq}: Resize requires constant sizes or scales")
                continue
            g.node("Resize", out, ins[0], name=name, **attrs)
        elif op == "Split":
            split = None
            if len(ins) > 1 and ins[1] in g.constants:
                split = [int(x) for x in g.constants[ins[1]].tolist()]
            elif "split" in a:
                split = [int(x) for x in a["split"]]
            g.node("Split", outs, ins[0], name=name, axis=int(a.get("axis", 0)), split=split)
            continue
        elif op == "Slice":
            if len(ins) < 3 or ins[1] not in g.constants or ins[2] not in g.constants:
                unsupported.append(f"{fq}: Slice requires constant starts and ends")
                continue
            starts = [int(x) for x in g.constants[ins[1]].reshape(-1).tolist()]
            ends = [int(x) for x in g.constants[ins[2]].reshape(-1).tolist()]
            axes = [int(x) for x in g.constants[ins[3]].reshape(-1).tolist()] if len(ins) > 3 and ins[3] in g.constants else None
            steps = [int(x) for x in g.constants[ins[4]].reshape(-1).tolist()] if len(ins) > 4 and ins[4] in g.constants else None
            g.node("Slice", out, ins[0], name=name, starts=starts, ends=ends, axes=axes, steps=steps)
        elif op == "Gelu":
            g.gelu(out, ins[0], approximate=a.get("approximate", "tanh"))
        elif op == "Softmax":
            g.softmax(out, ins[0], axis=int(a.get("axis", -1)))
        elif op == "Reshape":
            if len(ins) > 1 and ins[1] in g.constants:
                shape = [int(x) for x in g.constants[ins[1]].tolist()]
                g.reshape(out, ins[0], shape)
            else:
                unsupported.append(f"{fq}: Reshape with dynamic shape input is not supported yet")
                continue
        elif op == "Transpose":
            perm = a.get("perm")
            if perm is None:
                unsupported.append(f"{fq}: Transpose without perm is not supported yet")
                continue
            g.transpose(out, ins[0], perm)
        elif op == "Concat":
            g.concat(out, *ins, axis=int(a.get("axis", 0)))
        elif op in {"LayerNormalization", "LayerNorm"}:
            eps = float(a.get("epsilon", a.get("eps", 1e-5)))
            g.layernorm(out, ins[0], ins[1], ins[2] if len(ins) > 2 else None, eps=eps)
        elif op == "Constant":
            value = a.get("value")
            if value is None:
                for key in ("value_float", "value_int", "value_floats", "value_ints"):
                    if key in a:
                        value = a[key]
                        break
            if value is None:
                unsupported.append(f"{fq}: Constant attribute variant is not supported")
                continue
            g.const(out, np.asarray(value))
        else:
            unsupported.append(fq)
            continue

    for out_vi in og.output:
        g.output(out_vi.name)

    if unsupported:
        unique = sorted(set(unsupported))
        raise NotImplementedError(
            "ONNX ops not supported yet: "
            + "; ".join(unique)
            + " (full-node census; nothing was partially imported. "
              "Supported quant ops are stripped to fp32; other missing ops need backend support)"
        )
    if quant_stripped:
        g.metadata["aexrt_quant_stripped"] = ",".join(sorted(quant_stripped))
    return g
