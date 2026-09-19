from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple

from .graph import Graph, TensorSpec


@dataclass(frozen=True)
class DeviceInfo:
    """Backend-neutral device description used by execution planning."""

    backend: str
    device_type: str
    name: str
    vendor: str | None = None
    index: int = 0


@dataclass(frozen=True)
class BackendCapabilities:
    """Structured capability contract for portable backend selection."""

    ops: FrozenSet[str]
    dtypes: FrozenSet[str]
    features: FrozenSet[str]
    max_rank: int | None = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BackendCapabilities":
        return BackendCapabilities(
            ops=frozenset(str(x) for x in data.get("ops", ())),
            dtypes=frozenset(str(x) for x in data.get("dtypes", ())),
            features=frozenset(str(x) for x in data.get("features", ())),
            max_rank=data.get("max_rank"),
        )

    def supports_op(self, op: str) -> bool:
        return not self.ops or op in self.ops

    def supports_dtype(self, dtype: str) -> bool:
        return not self.dtypes or str(dtype) in self.dtypes


@dataclass(frozen=True)
class ValuePlan:
    name: str
    kind: str
    spec: TensorSpec | None
    nbytes: int | None
    producer: int | None
    first_use: int | None
    last_use: int | None


@dataclass(frozen=True)
class BufferPlan:
    name: str
    kind: str
    shape: Tuple[int, ...] | None
    dtype: str | None
    nbytes: int | None
    allocation_id: int | None
    offset: int = 0


@dataclass(frozen=True)
class AllocationPlan:
    allocation_id: int
    kind: str
    nbytes: int | None
    values: Tuple[str, ...]


@dataclass(frozen=True)
class MemoryPlan:
    constants: Tuple[str, ...]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    temporaries: Tuple[str, ...]
    values: Tuple[ValuePlan, ...]
    buffers: Tuple[BufferPlan, ...]
    allocations: Tuple[AllocationPlan, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    graph_name: str
    device: DeviceInfo
    capabilities: BackendCapabilities
    node_ops: Tuple[str, ...]
    memory: MemoryPlan


def _device_type(device: str) -> str:
    lowered = str(device).lower()
    if "cuda" in lowered:
        return "cuda"
    if "cpu" in lowered:
        return "cpu"
    if "dml" in lowered or "directml" in lowered:
        return "directml"
    if "vulkan" in lowered:
        return "vulkan"
    if "metal" in lowered or "mps" in lowered:
        return "metal"
    return lowered or "unknown"


def device_from_backend_info(info: Any) -> DeviceInfo:
    caps = dict(getattr(info, "capabilities", {}) or {})
    return DeviceInfo(
        backend=str(getattr(info, "name", "unknown")),
        device_type=str(caps.get("device_type") or _device_type(getattr(info, "device", "unknown"))),
        name=str(caps.get("device_name") or getattr(info, "device", "unknown")),
        vendor=caps.get("vendor"),
        index=int(caps.get("device_index", 0)),
    )


def validate_graph_supported(graph: Graph, capabilities: BackendCapabilities, value_specs: Dict[str, TensorSpec]) -> None:
    unsupported_ops = sorted({n.op for n in graph.nodes if not capabilities.supports_op(n.op)})
    if unsupported_ops:
        raise NotImplementedError(f"backend does not support ops: {unsupported_ops}")

    unsupported_dtypes = sorted(
        {
            spec.dtype
            for spec in value_specs.values()
            if spec is not None and not capabilities.supports_dtype(spec.dtype)
        }
    )
    if unsupported_dtypes:
        raise NotImplementedError(f"backend does not support dtypes: {unsupported_dtypes}")

    if capabilities.max_rank is not None:
        too_large = sorted(
            name
            for name, spec in value_specs.items()
            if spec.shape and len(spec.shape) > capabilities.max_rank
        )
        if too_large:
            raise NotImplementedError(f"backend max_rank={capabilities.max_rank} rejected values: {too_large}")


_DTYPE_NBYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def _estimate_nbytes(spec: TensorSpec | None) -> int | None:
    if spec is None:
        return None
    itemsize = _DTYPE_NBYTES.get(str(spec.dtype))
    if itemsize is None:
        return None
    numel = 1
    for dim in spec.shape:
        if dim < 0:
            return None
        numel *= int(dim)
    return numel * itemsize


def _broadcast_shape(shapes: Sequence[Tuple[int, ...]]) -> Tuple[int, ...] | None:
    if not shapes:
        return None
    out: List[int] = []
    max_rank = max(len(s) for s in shapes)
    for i in range(max_rank):
        dims = []
        for shape in shapes:
            offset = max_rank - len(shape)
            dims.append(1 if i < offset else shape[i - offset])
        known = [d for d in dims if d not in (1, -1)]
        if len(set(known)) > 1:
            return None
        if known:
            out.append(known[0])
        elif -1 in dims:
            out.append(-1)
        else:
            out.append(1)
    return tuple(out)


def _matmul_shape(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...] | None:
    if len(a) == 0 or len(b) == 0:
        return None
    if len(a) == 1 and len(b) == 1:
        return ()
    if len(a) == 1:
        batch = b[:-2]
        return tuple(batch) + (b[-1],)
    if len(b) == 1:
        batch = a[:-2]
        return tuple(batch) + (a[-2],)
    batch = _broadcast_shape([a[:-2], b[:-2]])
    if batch is None:
        return None
    return tuple(batch) + (a[-2], b[-1])


def _reshape_shape(input_shape: Tuple[int, ...], requested: Sequence[int]) -> Tuple[int, ...]:
    shape = [int(x) for x in requested]
    if shape.count(-1) != 1 or any(d < 0 for d in input_shape):
        return tuple(shape)
    known = 1
    for dim in shape:
        if dim != -1:
            known *= dim
    total = 1
    for dim in input_shape:
        total *= dim
    if known != 0 and total % known == 0:
        shape[shape.index(-1)] = total // known
    return tuple(shape)


def _conv_pool_shape(input_shape: Tuple[int, ...], kernel: Sequence[int], pads: Sequence[int], strides: Sequence[int], dilations: Sequence[int], out_channels: int | None = None, ceil_mode: bool = False) -> Tuple[int, ...] | None:
    if len(input_shape) != 4:
        return None
    n, c, h, w = input_shape
    kh, kw = int(kernel[0]), int(kernel[1])
    pt, pl, pb, pr = [int(x) for x in pads]
    sh, sw = int(strides[0]), int(strides[1])
    dh, dw = int(dilations[0]), int(dilations[1])
    if h < 0 or w < 0:
        oh = ow = -1
    else:
        num_h = h + pt + pb - dh * (kh - 1) - 1
        num_w = w + pl + pr - dw * (kw - 1) - 1
        if ceil_mode:
            import math
            oh = math.floor((num_h + sh - 1) / sh + 1)
            ow = math.floor((num_w + sw - 1) / sw + 1)
        else:
            oh = num_h // sh + 1
            ow = num_w // sw + 1
    return (n, int(out_channels) if out_channels is not None else c, oh, ow)


def infer_value_specs(graph: Graph) -> Dict[str, TensorSpec]:
    specs: Dict[str, TensorSpec] = dict(graph.value_specs)

    for name, spec in graph.inputs.items():
        specs[name] = spec
    for name, value in graph.constants.items():
        specs.setdefault(name, TensorSpec(tuple(value.shape), str(value.dtype)))

    for node in graph.nodes:
        input_specs = [specs.get(name) for name in node.inputs]
        if not node.outputs:
            continue

        out_spec: TensorSpec | None = None
        op = node.op

        if op == "Identity" and input_specs[0] is not None:
            out_spec = input_specs[0]
        elif op == "Conv" and input_specs[0] is not None and input_specs[1] is not None:
            w_shape = input_specs[1].shape
            if len(w_shape) >= 4:
                out_shape = _conv_pool_shape(
                    input_specs[0].shape,
                    w_shape[2:4],
                    node.attrs.get("pads", [0, 0, 0, 0]),
                    node.attrs.get("strides", [1, 1]),
                    node.attrs.get("dilations", [1, 1]),
                    out_channels=w_shape[0],
                )
                if out_shape is not None:
                    out_spec = TensorSpec(out_shape, input_specs[0].dtype)
        elif op == "MaxPool" and input_specs[0] is not None:
            out_shape = _conv_pool_shape(
                input_specs[0].shape,
                node.attrs["kernel_shape"],
                node.attrs.get("pads", [0, 0, 0, 0]),
                node.attrs.get("strides", node.attrs["kernel_shape"]),
                node.attrs.get("dilations", [1, 1]),
                ceil_mode=bool(node.attrs.get("ceil_mode", 0)),
            )
            if out_shape is not None:
                out_spec = TensorSpec(out_shape, input_specs[0].dtype)
        elif op == "Resize" and input_specs[0] is not None:
            if "sizes" in node.attrs:
                out_spec = TensorSpec(tuple(int(x) for x in node.attrs["sizes"]), input_specs[0].dtype)
            elif "scales" in node.attrs and all(d >= 0 for d in input_specs[0].shape):
                out_spec = TensorSpec(tuple(int(d * s) for d, s in zip(input_specs[0].shape, node.attrs["scales"])), input_specs[0].dtype)
        elif op == "Slice" and input_specs[0] is not None:
            shape = list(input_specs[0].shape)
            axes = node.attrs.get("axes") or list(range(len(node.attrs["starts"])))
            steps = node.attrs.get("steps") or [1] * len(node.attrs["starts"])
            for start, end, axis, step in zip(node.attrs["starts"], node.attrs["ends"], axes, steps):
                axis = int(axis)
                dim = shape[axis]
                if dim >= 0:
                    start = max(0, int(start))
                    end = min(dim, int(end))
                    step = int(step)
                    shape[axis] = max(0, (end - start + step - 1) // step)
                else:
                    shape[axis] = -1
            out_spec = TensorSpec(tuple(shape), input_specs[0].dtype)
        elif op == "Split" and input_specs[0] is not None:
            axis = int(node.attrs.get("axis", 0))
            if axis < 0:
                axis += len(input_specs[0].shape)
            split = node.attrs.get("split")
            if split is None:
                dim = input_specs[0].shape[axis]
                if dim >= 0 and node.outputs:
                    split = [dim // len(node.outputs)] * len(node.outputs)
            if split is not None:
                for out_name, size in zip(node.outputs, split):
                    shape = list(input_specs[0].shape)
                    shape[axis] = int(size)
                    specs[out_name] = TensorSpec(tuple(shape), input_specs[0].dtype)
            continue
        elif op in {"Add", "Sub", "Mul", "Div", "Pow"} and all(s is not None for s in input_specs):
            shape = _broadcast_shape([s.shape for s in input_specs if s is not None])
            if shape is not None:
                out_spec = TensorSpec(shape, input_specs[0].dtype)  # type: ignore[union-attr]
        elif op in {"MatMul", "FusedLinear"} and input_specs[0] is not None and input_specs[1] is not None:
            shape = _matmul_shape(input_specs[0].shape, input_specs[1].shape)
            if shape is not None:
                out_spec = TensorSpec(shape, input_specs[0].dtype)
        elif op in {"BatchNormalization", "Relu", "Gelu", "Sigmoid", "Tanh", "Softmax", "LayerNorm", "RoPE"} and input_specs[0] is not None:
            out_spec = input_specs[0]
        elif op == "Reshape" and input_specs[0] is not None:
            out_spec = TensorSpec(_reshape_shape(input_specs[0].shape, node.attrs["shape"]), input_specs[0].dtype)
        elif op == "Transpose" and input_specs[0] is not None:
            axes = tuple(int(i) for i in node.attrs["axes"])
            out_spec = TensorSpec(tuple(input_specs[0].shape[i] for i in axes), input_specs[0].dtype)
        elif op == "Concat" and all(s is not None for s in input_specs):
            axis = int(node.attrs.get("axis", -1))
            first = input_specs[0]
            if first is not None:
                rank = len(first.shape)
                if axis < 0:
                    axis += rank
                shape = list(first.shape)
                total = 0
                unknown = False
                for spec in input_specs:
                    assert spec is not None
                    dim = spec.shape[axis]
                    if dim < 0:
                        unknown = True
                    else:
                        total += dim
                shape[axis] = -1 if unknown else total
                out_spec = TensorSpec(tuple(shape), first.dtype)
        elif op == "Embedding" and input_specs[0] is not None and input_specs[1] is not None:
            ids, table = input_specs[0], input_specs[1]
            out_spec = TensorSpec(tuple(ids.shape) + (table.shape[-1],), table.dtype)
        elif op == "SDPA" and input_specs[0] is not None and input_specs[2] is not None:
            q, v = input_specs[0], input_specs[2]
            out_spec = TensorSpec(tuple(q.shape[:-1]) + (v.shape[-1],), q.dtype)

        if out_spec is not None:
            for out in node.outputs:
                specs[out] = out_spec

    return specs


def _ranges_overlap(a_first: int | None, a_last: int | None, b_first: int | None, b_last: int | None) -> bool:
    if a_first is None or a_last is None or b_first is None or b_last is None:
        return True
    return not (a_last < b_first or b_last < a_first)


def _value_lifetime(value: ValuePlan) -> Tuple[int | None, int | None]:
    start = value.producer if value.producer is not None else value.first_use
    end = value.last_use if value.last_use is not None else value.producer
    return start, end


def _allocate_buffers(value_plans: Sequence[ValuePlan]) -> Tuple[Dict[str, int], Tuple[AllocationPlan, ...]]:
    allocation_ids: Dict[str, int] = {}
    allocation_values: Dict[int, List[str]] = {}
    allocation_sizes: Dict[int, int | None] = {}
    allocation_lifetimes: Dict[int, Tuple[int | None, int | None]] = {}
    allocation_kinds: Dict[int, str] = {}
    next_id = 0

    def new_allocation(kind: str, value: ValuePlan) -> int:
        nonlocal next_id
        allocation_id = next_id
        next_id += 1
        allocation_ids[value.name] = allocation_id
        allocation_values[allocation_id] = [value.name]
        allocation_sizes[allocation_id] = value.nbytes
        allocation_lifetimes[allocation_id] = _value_lifetime(value)
        allocation_kinds[allocation_id] = kind
        return allocation_id

    for value in value_plans:
        if value.kind != "temporary" or value.nbytes is None:
            new_allocation(value.kind, value)
            continue

        chosen = None
        for allocation_id, size in allocation_sizes.items():
            if allocation_kinds[allocation_id] not in {"temporary", "shared"}:
                continue
            if size is None or size < value.nbytes:
                continue
            first, last = allocation_lifetimes[allocation_id]
            if not _ranges_overlap(first, last, value.first_use, value.last_use):
                chosen = allocation_id
                break

        if chosen is None:
            new_allocation(value.kind, value)
        else:
            allocation_ids[value.name] = chosen
            allocation_values[chosen].append(value.name)
            allocation_kinds[chosen] = "shared"
            first, last = allocation_lifetimes[chosen]
            value_first, value_last = _value_lifetime(value)
            starts = [x for x in (first, value_first) if x is not None]
            ends = [x for x in (last, value_last) if x is not None]
            allocation_lifetimes[chosen] = (min(starts) if starts else None, max(ends) if ends else None)

    allocations = tuple(
        AllocationPlan(
            allocation_id=allocation_id,
            kind=allocation_kinds[allocation_id],
            nbytes=allocation_sizes[allocation_id],
            values=tuple(values),
        )
        for allocation_id, values in sorted(allocation_values.items())
    )
    return allocation_ids, allocations


def build_memory_plan(graph: Graph, value_specs: Dict[str, TensorSpec] | None = None) -> MemoryPlan:
    specs = value_specs or infer_value_specs(graph)
    producer: Dict[str, int] = {name: None for name in graph.inputs}
    producer.update({name: None for name in graph.constants})
    first_use: Dict[str, int] = {}
    last_use: Dict[str, int] = {}

    for idx, node in enumerate(graph.nodes):
        for inp in node.inputs:
            first_use.setdefault(inp, idx)
            last_use[inp] = idx
        for out in node.outputs:
            producer[out] = idx

    output_set = set(graph.outputs)
    constant_set = set(graph.constants)
    input_set = set(graph.inputs)
    temps = []
    value_names = set(producer) | set(first_use) | output_set
    values: List[ValuePlan] = []
    buffers: List[BufferPlan] = []

    for name in sorted(value_names):
        if name in constant_set:
            kind = "constant"
        elif name in input_set:
            kind = "input"
        elif name in output_set:
            kind = "output"
        else:
            kind = "temporary"
            temps.append(name)
        spec = specs.get(name)
        nbytes = _estimate_nbytes(spec)
        values.append(
            ValuePlan(
                name=name,
                kind=kind,
                spec=spec,
                nbytes=nbytes,
                producer=producer.get(name),
                first_use=first_use.get(name),
                last_use=last_use.get(name),
            )
        )

    allocation_ids, allocations = _allocate_buffers(values)
    for value in values:
        buffers.append(
            BufferPlan(
                name=value.name,
                kind=value.kind,
                shape=value.spec.shape if value.spec is not None else None,
                dtype=value.spec.dtype if value.spec is not None else None,
                nbytes=value.nbytes,
                allocation_id=allocation_ids.get(value.name),
            )
        )

    return MemoryPlan(
        constants=tuple(graph.constants),
        inputs=tuple(graph.inputs),
        outputs=tuple(graph.outputs),
        temporaries=tuple(sorted(temps)),
        values=tuple(values),
        buffers=tuple(buffers),
        allocations=allocations,
    )


def compile_execution_plan(graph: Graph, backend_info: Any) -> ExecutionPlan:
    capabilities = BackendCapabilities.from_dict(dict(getattr(backend_info, "capabilities", {}) or {}))
    value_specs = infer_value_specs(graph)
    validate_graph_supported(graph, capabilities, value_specs)
    return ExecutionPlan(
        graph_name=graph.name,
        device=device_from_backend_info(backend_info),
        capabilities=capabilities,
        node_ops=tuple(n.op for n in graph.nodes),
        memory=build_memory_plan(graph, value_specs),
    )
