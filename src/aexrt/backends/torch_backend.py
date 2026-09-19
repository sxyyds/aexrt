from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import numpy as np

from .base import Backend, BackendInfo
from ..graph import Graph, Node

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _resolve_device(device: str) -> "torch.device":
    if torch is None:
        raise RuntimeError(f"PyTorch is not available: {_IMPORT_ERROR}")
    if device in ("dml", "directml"):
        try:
            import torch_directml
        except Exception as e:
            raise RuntimeError(f"torch-directml is not available: {e}") from e
        return torch_directml.device()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _torch_dtype(np_dtype: str):
    s = str(np_dtype)
    if s in ("float16", "fp16"): return torch.float16
    if s in ("bfloat16", "bf16"): return torch.bfloat16
    if s in ("float32", "fp32"): return torch.float32
    if s in ("float64", "fp64"): return torch.float64
    if s in ("int32",): return torch.int32
    if s in ("int64",): return torch.int64
    if s in ("bool",): return torch.bool
    return None


def _as_torch(x: Any, device: "torch.device") -> "torch.Tensor":
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    arr = np.asarray(x)
    return torch.as_tensor(arr, device=device)


def _rope(x: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor", interleaved: bool = False) -> "torch.Tensor":
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = cos[..., : x1.shape[-1]]
        s = sin[..., : x1.shape[-1]]
        y0 = x1 * c - x2 * s
        y1 = x1 * s + x2 * c
        return torch.stack((y0, y1), dim=-1).flatten(-2)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[..., :half]
    s = sin[..., :half]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


class TorchBackend(Backend):
    name = "torch"
    supported_ops = frozenset({
        "Add", "Sub", "Mul", "Div", "MatMul", "FusedLinear",
        "Conv", "BatchNormalization", "MaxPool", "Resize", "Split", "Slice",
        "Relu", "Gelu", "Sigmoid", "Tanh", "Softmax", "LayerNorm",
        "Reshape", "Transpose", "Concat", "Embedding", "SDPA", "RoPE", "Identity",
    })
    supported_dtypes = frozenset({"float16", "bfloat16", "float32", "float64", "int32", "int64", "bool"})
    supported_features = frozenset({
        "persistent_constants",
        "static_execution_plan",
        "async_device_copy",
        "cuda_graph_replay",
    })

    def __init__(
        self,
        device: str = "auto",
        output_numpy: bool = True,
        dtype: str | None = None,
        cuda_graph: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError(f"PyTorch is not available: {_IMPORT_ERROR}")
        self.device = _resolve_device(device)
        self.output_numpy = output_numpy
        self.dtype = _torch_dtype(dtype) if dtype else None
        self.enable_cuda_graph = bool(cuda_graph and self.device.type == "cuda")
        self.graph: Graph | None = None
        self.constants: Dict[str, torch.Tensor] = {}
        self._cuda_graph = None
        self._static_inputs: Dict[str, torch.Tensor] = {}
        self._static_vals: Dict[str, torch.Tensor] = {}
        self._static_outputs: Dict[str, torch.Tensor] = {}
        self._captured_signature: Dict[str, Tuple[Tuple[int, ...], torch.dtype]] = {}
        self.execution_plan = None

    def info(self) -> BackendInfo:
        caps = {
            "gpu": self.device.type == "cuda",
            "fp16": self.device.type == "cuda",
            "bf16": self.device.type == "cuda" and torch.cuda.is_bf16_supported(),
            "sdpa": True,
            "cuda_graph": self.enable_cuda_graph,
            "device_type": self.device.type,
            "device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "ops": sorted(self.supported_ops),
            "dtypes": sorted(self.supported_dtypes),
            "features": sorted(self.supported_features if self.device.type == "cuda" else self.supported_features - {"cuda_graph_replay", "async_device_copy"}),
        }
        return BackendInfo("torch", str(self.device), caps)

    def prepare(self, graph: Graph) -> None:
        self.graph = graph
        self.execution_plan = self.compile_plan(graph)
        self.constants = {}
        for k, v in graph.constants.items():
            t = torch.as_tensor(v, device=self.device)
            if self.dtype is not None and t.is_floating_point():
                t = t.to(self.dtype)
            self.constants[k] = t.contiguous()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        assert self.graph is not None, "backend not prepared"
        if self.enable_cuda_graph:
            return self._run_cuda_graph(inputs)

        vals = self._make_vals(inputs)
        with torch.inference_mode():
            outs = self._execute(vals)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return self._format_outputs(outs)

    def _make_vals(self, inputs: Dict[str, Any]) -> Dict[str, "torch.Tensor"]:
        vals: Dict[str, torch.Tensor] = dict(self.constants)
        for k, v in inputs.items():
            vals[k] = self._prepare_input_tensor(v)
        return vals

    def _prepare_input_tensor(self, value: Any) -> "torch.Tensor":
        t = _as_torch(value, self.device)
        if self.dtype is not None and t.is_floating_point():
            t = t.to(self.dtype)
        return t.contiguous()

    def _execute(self, vals: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        assert self.graph is not None
        for n in self.graph.nodes:
            result = self._eval(n, vals)
            if len(n.outputs) == 1:
                vals[n.outputs[0]] = result
            else:
                for name, value in zip(n.outputs, result):
                    vals[name] = value
        outs = {k: vals[k] for k in self.graph.outputs}
        return outs

    def _format_outputs(self, outs: Dict[str, "torch.Tensor"]) -> Dict[str, Any]:
        if self.output_numpy:
            return {k: v.detach().cpu().numpy() for k, v in outs.items()}
        return outs

    def _signature(self, inputs: Dict[str, Any]) -> Dict[str, Tuple[Tuple[int, ...], "torch.dtype"]]:
        sig = {}
        for k, v in inputs.items():
            t = self._prepare_input_tensor(v)
            sig[k] = (tuple(int(x) for x in t.shape), t.dtype)
        return sig

    def _copy_inputs_to_static(self, inputs: Dict[str, Any]) -> None:
        for k, v in inputs.items():
            t = self._prepare_input_tensor(v)
            self._static_inputs[k].copy_(t, non_blocking=True)

    def _capture_cuda_graph(self, inputs: Dict[str, Any]) -> None:
        assert self.graph is not None
        self._captured_signature = self._signature(inputs)
        self._static_inputs = {k: self._prepare_input_tensor(v).clone() for k, v in inputs.items()}
        self._static_vals = dict(self.constants)
        self._static_vals.update(self._static_inputs)

        # Warmup on a side stream lets PyTorch's caching allocator settle before capture.
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(3):
                vals = dict(self._static_vals)
                self._execute(vals)
        torch.cuda.current_stream(self.device).wait_stream(side)

        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph), torch.inference_mode():
            vals = dict(self._static_vals)
            self._static_outputs = self._execute(vals)

    def _run_cuda_graph(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sig = self._signature(inputs)
        if self._cuda_graph is None or sig != self._captured_signature:
            self._capture_cuda_graph(inputs)
        else:
            self._copy_inputs_to_static(inputs)
        assert self._cuda_graph is not None
        self._cuda_graph.replay()
        torch.cuda.synchronize(self.device)
        return self._format_outputs(self._static_outputs)

    def _eval(self, n: Node, vals: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        xs = [vals[i] for i in n.inputs]
        op = n.op
        if op == "Identity": return xs[0]
        if op == "Add": return xs[0] + xs[1]
        if op == "Sub": return xs[0] - xs[1]
        if op == "Mul": return xs[0] * xs[1]
        if op == "Div": return xs[0] / xs[1]
        if op == "MatMul": return torch.matmul(xs[0], xs[1])
        if op == "Conv":
            pads = tuple(int(x) for x in n.attrs.get("pads", [0, 0, 0, 0]))
            x = xs[0]
            if any(pads):
                x = torch.nn.functional.pad(x, (pads[1], pads[3], pads[0], pads[2]))
            return torch.nn.functional.conv2d(
                x,
                xs[1],
                xs[2] if len(xs) > 2 else None,
                stride=tuple(int(x) for x in n.attrs.get("strides", [1, 1])),
                padding=0,
                dilation=tuple(int(x) for x in n.attrs.get("dilations", [1, 1])),
                groups=int(n.attrs.get("group", 1)),
            )
        if op == "BatchNormalization":
            x, scale, bias, mean, var = xs[:5]
            eps = float(n.attrs.get("epsilon", 1e-5))
            shape = (1, -1) + (1,) * (x.dim() - 2)
            return (x - mean.reshape(shape)) / torch.sqrt(var.reshape(shape) + eps) * scale.reshape(shape) + bias.reshape(shape)
        if op == "MaxPool":
            pads = tuple(int(x) for x in n.attrs.get("pads", [0, 0, 0, 0]))
            x = xs[0]
            if any(pads):
                x = torch.nn.functional.pad(x, (pads[1], pads[3], pads[0], pads[2]), value=float("-inf"))
            return torch.nn.functional.max_pool2d(
                x,
                kernel_size=tuple(int(x) for x in n.attrs["kernel_shape"]),
                stride=tuple(int(x) for x in n.attrs.get("strides", n.attrs["kernel_shape"])),
                padding=0,
                dilation=tuple(int(x) for x in n.attrs.get("dilations", [1, 1])),
                ceil_mode=bool(n.attrs.get("ceil_mode", 0)),
            )
        if op == "Resize":
            mode = str(n.attrs.get("mode", "nearest"))
            kwargs = {"mode": mode}
            if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
                kwargs["align_corners"] = False
            if "sizes" in n.attrs:
                size = tuple(int(x) for x in n.attrs["sizes"][-(xs[0].dim() - 2):])
                return torch.nn.functional.interpolate(xs[0], size=size, **kwargs)
            scales = tuple(float(x) for x in n.attrs["scales"][-(xs[0].dim() - 2):])
            return torch.nn.functional.interpolate(xs[0], scale_factor=scales, **kwargs)
        if op == "Split":
            axis = int(n.attrs.get("axis", 0))
            split = n.attrs.get("split")
            return torch.split(xs[0], tuple(int(x) for x in split), dim=axis) if split is not None else torch.chunk(xs[0], len(n.outputs), dim=axis)
        if op == "Slice":
            starts = [int(x) for x in n.attrs["starts"]]
            ends = [int(x) for x in n.attrs["ends"]]
            axes = n.attrs.get("axes")
            if axes is None:
                axes = list(range(len(starts)))
            steps = n.attrs.get("steps") or [1] * len(starts)
            slices = [slice(None)] * xs[0].dim()
            for start, end, axis, step in zip(starts, ends, axes, steps):
                axis = int(axis)
                dim = xs[0].shape[axis]
                end = int(end)
                if end > dim:
                    end = dim
                slices[axis] = slice(int(start), end, int(step))
            return xs[0][tuple(slices)]
        if op == "FusedLinear":
            # `addmm` maps the ubiquitous 2-D Linear+bias pattern to a single
            # vendor BLAS epilogue where available, reducing launch overhead.
            if n.attrs.get("has_bias", len(xs) > 2) and xs[0].dim() == 2 and xs[1].dim() == 2 and xs[2].dim() == 1:
                y = torch.addmm(xs[2], xs[0], xs[1])
            else:
                y = torch.matmul(xs[0], xs[1])
                if n.attrs.get("has_bias", len(xs) > 2): y = y + xs[2]
            act = n.attrs.get("activation")
            if act == "Relu": y = torch.relu(y)
            elif act == "Gelu": y = torch.nn.functional.gelu(y, approximate=n.attrs.get("approximate", "tanh"))
            elif act == "Sigmoid": y = torch.sigmoid(y)
            elif act == "Tanh": y = torch.tanh(y)
            return y
        if op == "Relu": return torch.relu(xs[0])
        if op == "Gelu": return torch.nn.functional.gelu(xs[0], approximate=n.attrs.get("approximate", "tanh"))
        if op == "Sigmoid": return torch.sigmoid(xs[0])
        if op == "Tanh": return torch.tanh(xs[0])
        if op == "Softmax": return torch.softmax(xs[0], dim=int(n.attrs.get("axis", -1)))
        if op == "LayerNorm":
            x, w = xs[0], xs[1]
            b = xs[2] if len(xs) > 2 else None
            return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps=float(n.attrs.get("eps", 1e-5)))
        if op == "Reshape": return torch.reshape(xs[0], tuple(int(i) for i in n.attrs["shape"]))
        if op == "Transpose": return torch.permute(xs[0], tuple(int(i) for i in n.attrs["axes"]))
        if op == "Concat": return torch.cat(xs, dim=int(n.attrs.get("axis", -1)))
        if op == "Embedding": return torch.nn.functional.embedding(xs[0].long(), xs[1])
        if op == "SDPA":
            q, k, v = xs[:3]
            scale = n.attrs.get("scale") or (1.0 / math.sqrt(q.shape[-1]))
            scores = torch.matmul(q, k.transpose(-1, -2)) * float(scale)
            if len(xs) > 3: scores = scores + xs[3]
            if n.attrs.get("causal", False):
                L, S = q.shape[-2], k.shape[-2]
                mask = torch.ones((L, S), dtype=torch.bool, device=q.device).triu(1)
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            p = torch.softmax(scores, dim=-1)
            return torch.matmul(p, v)
        if op == "RoPE": return _rope(xs[0], xs[1], xs[2], bool(n.attrs.get("interleaved", False)))
        raise NotImplementedError(f"Torch backend does not support op {op}")
