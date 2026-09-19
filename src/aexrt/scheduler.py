from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .execution import AllocationPlan, ExecutionPlan, infer_value_specs
from .graph import Graph, Node, TensorSpec


ELEMENTWISE_OPS = {"Add", "Sub", "Mul", "Div", "Relu", "Gelu", "Sigmoid", "Tanh", "Identity"}
REDUCTION_OPS = {"Softmax", "LayerNorm"}
MATMUL_OPS = {"MatMul", "FusedLinear"}
LAYOUT_OPS = {"Reshape", "Transpose", "Concat"}


@dataclass(frozen=True)
class ArenaSlot:
    allocation_id: int
    offset: int
    nbytes: int
    values: Tuple[str, ...]


@dataclass(frozen=True)
class ArenaPlan:
    alignment: int
    total_nbytes: int
    slots: Tuple[ArenaSlot, ...]


@dataclass(frozen=True)
class TilePlan:
    tile_m: int
    tile_n: int
    tile_k: int
    waves_per_group: int
    vector_width: int
    strategy: str


@dataclass(frozen=True)
class KernelPlan:
    kernel_id: int
    op: str
    name: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    category: str
    tile: TilePlan | None
    fusion_group: int | None
    estimated_work_items: int


@dataclass(frozen=True)
class FusionGroup:
    group_id: int
    kind: str
    nodes: Tuple[str, ...]
    outputs: Tuple[str, ...]


@dataclass(frozen=True)
class GraphSchedule:
    signature: str
    arena: ArenaPlan
    kernels: Tuple[KernelPlan, ...]
    fusion_groups: Tuple[FusionGroup, ...]
    algorithm_tags: Tuple[str, ...]


def compile_graph_schedule(graph: Graph, plan: ExecutionPlan) -> GraphSchedule:
    value_specs = infer_value_specs(graph)
    arena = build_lifetime_colored_arena(plan.memory.allocations)
    fusion_groups = build_fusion_groups(graph.nodes)
    fusion_by_node = {
        node_name: group.group_id
        for group in fusion_groups
        for node_name in group.nodes
    }
    kernels = tuple(
        build_kernel_plan(i, node, value_specs, plan, fusion_by_node.get(node.name))
        for i, node in enumerate(graph.nodes)
    )
    return GraphSchedule(
        signature=shape_specialized_signature(graph, plan),
        arena=arena,
        kernels=kernels,
        fusion_groups=fusion_groups,
        algorithm_tags=(
            "lifetime_colored_arena_v1",
            "shape_specialized_kernel_abi_v1",
            "tile_wave_scheduler_v1",
        ),
    )


def build_lifetime_colored_arena(
    allocations: Sequence[AllocationPlan],
    *,
    alignment: int = 256,
) -> ArenaPlan:
    """Pack compatible allocation plans into a single arena address space.

    The execution planner has already colored values that can share an
    allocation. This pass gives each allocation a stable aligned offset so a
    native backend can bind one large GPU arena plus offsets instead of many
    scattered resources.
    """

    offset = 0
    slots: List[ArenaSlot] = []
    for allocation in sorted(allocations, key=_arena_sort_key):
        nbytes = int(allocation.nbytes or 0)
        aligned = _align(offset, alignment)
        slots.append(
            ArenaSlot(
                allocation_id=allocation.allocation_id,
                offset=aligned,
                nbytes=nbytes,
                values=allocation.values,
            )
        )
        offset = aligned + nbytes
    return ArenaPlan(alignment=alignment, total_nbytes=_align(offset, alignment), slots=tuple(slots))


def build_fusion_groups(nodes: Sequence[Node]) -> Tuple[FusionGroup, ...]:
    groups: List[FusionGroup] = []
    current: List[Node] = []
    group_id = 0

    def flush() -> None:
        nonlocal group_id, current
        if len(current) < 2:
            current = []
            return
        groups.append(
            FusionGroup(
                group_id=group_id,
                kind="elementwise_chain",
                nodes=tuple(n.name for n in current),
                outputs=tuple(current[-1].outputs),
            )
        )
        group_id += 1
        current = []

    for node in nodes:
        if node.op in ELEMENTWISE_OPS:
            current.append(node)
        else:
            flush()
    flush()
    return tuple(groups)


def build_kernel_plan(
    kernel_id: int,
    node: Node,
    value_specs: Dict[str, TensorSpec],
    plan: ExecutionPlan,
    fusion_group: int | None,
) -> KernelPlan:
    category = op_category(node.op)
    output_spec = value_specs.get(node.outputs[0]) if node.outputs else None
    tile = choose_tile_plan(node, value_specs, plan) if category == "matmul" else None
    return KernelPlan(
        kernel_id=kernel_id,
        op=node.op,
        name=node.name,
        inputs=tuple(node.inputs),
        outputs=tuple(node.outputs),
        category=category,
        tile=tile,
        fusion_group=fusion_group,
        estimated_work_items=estimate_work_items(output_spec),
    )


def choose_tile_plan(node: Node, value_specs: Dict[str, TensorSpec], plan: ExecutionPlan) -> TilePlan:
    a = value_specs.get(node.inputs[0])
    b = value_specs.get(node.inputs[1]) if len(node.inputs) > 1 else None
    out = value_specs.get(node.outputs[0]) if node.outputs else None
    dtype = (out or a or b).dtype if (out or a or b) is not None else "float32"
    vendor = (plan.device.vendor or plan.device.name or "").lower()

    m, n, k = _matmul_mnk(a, b, out)
    if dtype in {"float16", "bfloat16"}:
        vector_width = 8
    else:
        vector_width = 4

    if "nvidia" in vendor:
        if max(m, n) >= 2048:
            return TilePlan(128, 128, 64, 4, vector_width, "nvidia_large_square")
        return TilePlan(64, 128, 64, 4, vector_width, "nvidia_latency_balanced")
    if "amd" in vendor:
        if max(m, n) >= 2048:
            return TilePlan(128, 64, 64, 4, vector_width, "amd_wave64_large")
        return TilePlan(64, 64, 64, 2, vector_width, "amd_wave64_compact")
    if "intel" in vendor:
        return TilePlan(64, 64, 32, 2, vector_width, "intel_cache_sensitive")
    return TilePlan(64, 64, 32, 2, vector_width, "portable_baseline")


def shape_specialized_signature(graph: Graph, plan: ExecutionPlan) -> str:
    parts = [graph.name, plan.device.device_type, plan.device.name]
    for name, spec in sorted(infer_value_specs(graph).items()):
        parts.append(f"{name}:{spec.dtype}:{'x'.join(str(d) for d in spec.shape)}")
    parts.extend(plan.node_ops)
    return "|".join(parts)


def op_category(op: str) -> str:
    if op in MATMUL_OPS:
        return "matmul"
    if op in ELEMENTWISE_OPS:
        return "elementwise"
    if op in REDUCTION_OPS:
        return "reduction"
    if op in LAYOUT_OPS:
        return "layout"
    return "special"


def estimate_work_items(spec: TensorSpec | None) -> int:
    if spec is None:
        return 0
    total = 1
    for dim in spec.shape:
        if dim < 0:
            return 0
        total *= int(dim)
    return total


def _matmul_mnk(
    a: TensorSpec | None,
    b: TensorSpec | None,
    out: TensorSpec | None,
) -> Tuple[int, int, int]:
    if a is None or b is None:
        return 1, 1, 1
    if len(a.shape) == 1:
        m = 1
        k = a.shape[0]
    else:
        m = a.shape[-2]
        k = a.shape[-1]
    if len(b.shape) == 1:
        n = 1
    else:
        n = b.shape[-1]
    if out is not None and len(out.shape) >= 2:
        m = out.shape[-2]
        n = out.shape[-1]
    return _positive(m), _positive(n), _positive(k)


def _positive(value: int) -> int:
    return int(value) if int(value) > 0 else 1


def _arena_sort_key(allocation: AllocationPlan) -> Tuple[int, int]:
    kind_order = {
        "constant": 0,
        "input": 1,
        "output": 2,
        "shared": 3,
        "temporary": 4,
    }
    return (kind_order.get(allocation.kind, 9), allocation.allocation_id)


def _align(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment
