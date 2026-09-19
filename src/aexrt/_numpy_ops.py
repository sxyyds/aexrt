"""共享的 NumPy 算子求值表。

两个用途：
1. ``NumpyBackend`` 遇到自身分发链不认识的算子时，路由到这里执行。
2. ``native_d3d12_backend`` 的 CPU 逐算子回退路径（``AEXRT_NATIVE_D3D12_CPU_FALLBACK``）
   在 GPU 无对应 kernel 时下载到主机执行，保证“任给 ONNX 不崩”。

这里的实现以正确性优先，不追求性能；只覆盖 NCHW float32 常见推理形态。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ONNX_ELEM_TO_NP: Dict[int, Any] = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    16: np.float32,  # bfloat16 数值回退按 float32 处理
}

_UNARY = {
    "Exp": np.exp,
    "Log": np.log,
    "Sqrt": np.sqrt,
    "Neg": np.negative,
    "Abs": np.abs,
    "Floor": np.floor,
    "Ceil": np.ceil,
    "Round": np.round,
    "Sin": np.sin,
    "Cos": np.cos,
    "Reciprocal": np.reciprocal,
    "Sign": np.sign,
}

_erf_vec = np.vectorize(math.erf, otypes=[np.float64])

# 本求值表覆盖的算子集合：backend 用它扩展 supported_ops 声明。
NUMPY_EVAL_OPS = frozenset({
    "Identity", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Cast", "Shape",
    "Gather", "Slice", "Pad", "Expand",
    "LeakyRelu", "PRelu", "Clip", "HardSigmoid", "Mish", "Erf", "Elu",
    "Exp", "Log", "Sqrt", "Neg", "Abs", "Floor", "Ceil", "Round",
    "Sin", "Cos", "Reciprocal", "Sign",
    "Min", "Max", "Mod",
    "AveragePool", "MaxPool", "GlobalAveragePool", "GlobalMaxPool",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
})


def _axes_list(attrs: Dict[str, Any], rank: int) -> List[int]:
    axes = attrs.get("axes")
    if axes is None:
        return list(range(rank))
    return [_axis_positive(int(a), rank) for a in axes]


def _axis_positive(axis: int, rank: int) -> int:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= max(rank, 1):
        raise ValueError(f"axis {axis} out of range for rank {rank}")
    return axis


def _pad_spec(pads: Sequence[int]) -> List[int]:
    values = [int(p) for p in pads]
    if len(values) % 2 != 0:
        raise ValueError("pad spec must have even length")
    # ONNX 顺序 [begin0..beginN, end0..endN] -> numpy 顺序 [(b0,e0),...]
    half = len(values) // 2
    return list(zip(values[:half], values[half:]))


def _pool2d(
    x: np.ndarray,
    kernel: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    mode: str,
    dilations: Sequence[int] = (1, 1),
    count_include_pad: bool = True,
) -> np.ndarray:
    if x.ndim != 4:
        raise NotImplementedError("numpy pool fallback supports NCHW rank-4 only")
    kh, kw = int(kernel[0]), int(kernel[1])
    sh, sw = int(strides[0]), int(strides[1])
    dh, dw = int(dilations[0]), int(dilations[1])
    pt, pl, pb, pr = (int(p) for p in (pads if len(pads) == 4 else (pads + pads)[:4]))
    fill = -np.inf if mode == "max" else 0.0
    xp = np.pad(
        x,
        ((0, 0), (0, 0), (pt, pb), (pl, pr)),
        mode="constant",
        constant_values=fill,
    )
    n, c, h, w = xp.shape
    eh = (kh - 1) * dh + 1
    ew = (kw - 1) * dw + 1
    ho = (h - eh) // sh + 1
    wo = (w - ew) // sw + 1
    if ho <= 0 or wo <= 0:
        raise ValueError("pool window larger than padded input")
    out_shape = (n, c, ho, wo)
    windows = np.empty((kh * kw, n, c, ho, wo), dtype=xp.dtype)
    idx = 0
    for di in range(kh):
        for dj in range(kw):
            i0 = di * dh
            j0 = dj * dw
            windows[idx] = xp[:, :, i0 : i0 + ho * sh : sh, j0 : j0 + wo * sw : sw]
            idx += 1
    if mode == "max":
        return windows.max(axis=0)
    total = windows.sum(axis=0)
    if count_include_pad:
        return total / float(kh * kw)
    # count_include_pad=0：按每个窗口内有效（非 pad）元素计数
    valid = np.empty((kh * kw, n, c, ho, wo), dtype=np.float32)
    mask_source = np.zeros((1, 1, h, w), dtype=np.float32)
    mask_source[:, :, pt : h - pb or None, pl : w - pr or None] = 1.0
    idx = 0
    for di in range(kh):
        for dj in range(kw):
            i0 = di * dh
            j0 = dj * dw
            valid[idx] = mask_source[:, :, i0 : i0 + ho * sh : sh, j0 : j0 + wo * sw : sw]
            idx += 1
    counts = valid.sum(axis=0)
    return np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)


def eval_node(op: str, attrs: Dict[str, Any], xs: List[np.ndarray]) -> np.ndarray:
    """按 AEXRT IR 语义在 numpy 上求值一个算子。

    ``xs`` 为按 ``node.inputs`` 顺序排列的数组；``attrs`` 为算子属性。
    未覆盖的算子抛 ``NotImplementedError``。
    """
    if not xs and op != "Shape":
        raise NotImplementedError(f"numpy op evaluator got no inputs for {op}")
    x = xs[0] if xs else None

    # ---- 布局 / 形状 / 类型 ----
    if op == "Identity":
        return x
    if op == "Reshape":
        return np.reshape(x, tuple(int(v) for v in attrs["shape"]))
    if op == "Flatten":
        axis = _axis_positive(int(attrs.get("axis", 1)), x.ndim)
        lead = int(np.prod(x.shape[:axis], dtype=np.int64)) if axis > 0 else 1
        rest = int(np.prod(x.shape[axis:], dtype=np.int64))
        return np.reshape(x, (lead, rest))
    if op == "Squeeze":
        axes = attrs.get("axes")
        if axes is None:
            return np.squeeze(x)
        return np.squeeze(x, tuple(_axis_positive(int(a), x.ndim) for a in axes))
    if op == "Unsqueeze":
        axes = sorted(_axis_positive(int(a), x.ndim + len(attrs["axes"])) for a in attrs["axes"])
        out = x
        for a in axes:
            out = np.expand_dims(out, a)
        return out
    if op == "Cast":
        target = _ONNX_ELEM_TO_NP.get(int(attrs.get("to", 1)), np.float32)
        return np.asarray(x).astype(target)
    if op == "Shape":
        return np.asarray(x.shape, dtype=np.int64)
    if op == "Gather":
        indices = xs[1]
        axis = _axis_positive(int(attrs.get("axis", 0)), x.ndim)
        return np.take(x, indices.astype(np.int64), axis=axis)
    if op == "Slice":
        starts = [int(v) for v in attrs.get("starts") or [0]]
        ends = [int(v) for v in attrs.get("ends") or []]
        axes = [_axis_positive(int(a), x.ndim) for a in (attrs.get("axes") or range(len(starts)))]
        steps = [int(v) for v in attrs.get("steps") or [1] * len(starts)]
        index = [slice(None)] * x.ndim
        for a, s, e, t in zip(axes, starts, ends, steps):
            index[a] = slice(s, e, t)
        return x[tuple(index)]
    if op == "Pad":
        mode = str(attrs.get("mode", "constant"))
        pad_pairs = _pad_spec(attrs.get("pads") or [])
        value = float(attrs.get("value", 0.0))
        if mode != "constant":
            raise NotImplementedError("numpy Pad fallback supports constant mode only")
        return np.pad(x, pad_pairs, mode="constant", constant_values=value)
    if op == "Expand":
        return np.broadcast_to(x, tuple(int(v) for v in attrs["shape"]))

    # ---- 激活 / 逐元素 ----
    if op == "LeakyRelu":
        alpha = float(attrs.get("alpha", 0.01))
        return np.where(x >= 0, x, x * alpha)
    if op == "PRelu":
        slope = xs[1]
        return np.where(x >= 0, x, x * slope)
    if op == "Clip":
        lo = attrs.get("min")
        hi = attrs.get("max")
        return np.clip(x, None if lo is None else float(lo), None if hi is None else float(hi))
    if op == "HardSigmoid":
        alpha = float(attrs.get("alpha", 0.2))
        beta = float(attrs.get("beta", 0.5))
        return np.clip(x * alpha + beta, 0.0, 1.0)
    if op == "Mish":
        return x * np.tanh(np.log1p(np.exp(x)))
    if op == "Erf":
        return _erf_vec(x)
    if op == "Elu":
        alpha = float(attrs.get("alpha", 1.0))
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))
    if op in _UNARY:
        return _UNARY[op](x)
    if op in {"Min", "Max"}:
        result = xs[0]
        for other in xs[1:]:
            result = np.minimum(result, other) if op == "Min" else np.maximum(result, other)
        return result
    if op == "Mod":
        fmod = int(attrs.get("fmod", 0)) != 0
        if fmod:
            return np.fmod(xs[0], xs[1])
        return np.mod(xs[0], xs[1])

    # ---- 池化 ----
    if op in {"AveragePool", "MaxPool"}:
        kernel = attrs["kernel_shape"]
        strides = attrs.get("strides") or list(kernel)
        pads = attrs.get("pads") or [0, 0, 0, 0]
        dilations = attrs.get("dilations") or [1, 1]
        mode = "max" if op == "MaxPool" else "avg"
        count_include_pad = bool(attrs.get("count_include_pad", 1 if op == "AveragePool" else 0))
        if int(attrs.get("ceil_mode", 0)) != 0:
            raise NotImplementedError("numpy pool fallback supports ceil_mode=0 only")
        return _pool2d(x, kernel, strides, pads, mode, dilations, count_include_pad)
    if op == "GlobalAveragePool":
        return np.mean(x, axis=(2, 3), keepdims=True)
    if op == "GlobalMaxPool":
        return np.max(x, axis=(2, 3), keepdims=True)

    # ---- 归约 ----
    if op in {"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"}:
        axes = _axes_list(attrs, x.ndim)
        keepdims = bool(attrs.get("keepdims", 1))
        fn = {
            "ReduceMean": np.mean,
            "ReduceSum": np.sum,
            "ReduceMax": np.max,
            "ReduceMin": np.min,
            "ReduceProd": np.prod,
        }[op]
        return fn(x, axis=tuple(axes), keepdims=keepdims)

    raise NotImplementedError(f"numpy op evaluator does not support {op}")
